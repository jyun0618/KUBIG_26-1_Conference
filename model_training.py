"""
model_training.py
=================
반도체 업황 YoY% 6개월 선행 예측 모델 벤치마크 모듈.

모델 라인업:
    1. Ridge 회귀         -- L2 정규화, 다중공선성 강건
    2. Lasso 회귀         -- L1 정규화, 자동 피쳐 선택
    3. XGBoost            -- 트리 부스팅, 비선형 사이클 패턴
    4. LightGBM           -- 경량 그래디언트 부스팅, 빠른 학습
    5. N-HiTS (선택적)    -- 딥러닝 시계열 모델 (neuralforecast)

평가 전략:
    - TimeSeriesSplit (시계열 교차검증): 미래 데이터 누설 방지
    - 평가지표: RMSE, MAE, MAPE, 사이클 방향 정확도(Bull/Bear 분류)

손실 함수 제안:
    반도체 Bull/Bear Market 특성 반영을 위해 Asymmetric Loss 사용:
    - Bear 국면(YoY < 0) 예측 실패에 더 높은 페널티 부여
    - 산업 실무상 하락 예측 실패가 더 큰 리스크를 초래

입력:
    conference/outputs/data/features_dataset.csv

출력:
    conference/outputs/models/benchmark_results.csv   -- 모델별 성능 비교
    conference/outputs/models/predictions.csv         -- 각 모델 예측값
    conference/outputs/models/feature_importance.png  -- 피쳐 중요도 (XGBoost/LGBM)
    conference/outputs/models/benchmark_plot.png      -- 성능 비교 차트
    conference/outputs/models/{model_name}.pkl        -- 저장된 모델
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH   = os.path.join(BASE_DIR, "outputs", "data", "features_dataset.csv")
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 예측 타겟 (Worldwide YoY% T+6을 주 타겟으로 사용)
PRIMARY_TARGET   = "TARGET_Worldwide_YoY_T6"
SECONDARY_TARGET = "TARGET_Asia_Pacific_YoY_T6"

# 시계열 교차검증 설정
N_SPLITS    = 5    # 폴드 수
MIN_TRAIN   = 60   # 최소 학습 기간 (개월)
TEST_SIZE   = 12   # 테스트 폴드 크기 (12개월)


# ──────────────────────────────────────────────
# 평가 지표 함수
# ──────────────────────────────────────────────
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    """MAPE (절대 퍼센트 오차 평균). 0값 방지를 위해 small epsilon 추가."""
    eps = 1e-6
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100


def direction_accuracy(y_true, y_pred):
    """
    Bull/Bear 방향 분류 정확도.
    YoY% > 0 = Bull, YoY% <= 0 = Bear로 이진 분류 후 정확도 계산.
    반도체 투자 관점에서 방향성 예측이 핵심.
    """
    true_dir = (y_true > 0).astype(int)
    pred_dir = (y_pred > 0).astype(int)
    return (true_dir == pred_dir).mean()


def asymmetric_loss(y_true, y_pred, bear_penalty: float = 1.5):
    """
    비대칭 손실 함수 (평가용).
    Bear 국면(실제 YoY < 0)에서의 예측 오차에 bear_penalty 배 가중치 부여.
    반도체 산업에서 하락 사이클 미예측이 더 큰 손실을 초래함을 반영.

    bear_penalty = 1.5: Bear 오차를 Bull 오차보다 50% 더 중요하게 평가.
    """
    errors = y_true - y_pred
    weights = np.where(y_true < 0, bear_penalty, 1.0)
    return np.mean(weights * errors ** 2)


def evaluate_metrics(y_true, y_pred, name="") -> dict:
    """종합 평가 지표 계산."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "model":          name,
        "RMSE":           round(rmse(y_true, y_pred), 4),
        "MAE":            round(mean_absolute_error(y_true, y_pred), 4),
        "MAPE(%)":        round(mape(y_true, y_pred), 4),
        "Direction_Acc":  round(direction_accuracy(y_true, y_pred), 4),
        "Asym_Loss(1.5x)": round(asymmetric_loss(y_true, y_pred, bear_penalty=1.5), 4),
    }


# ──────────────────────────────────────────────
# 데이터 준비
# ──────────────────────────────────────────────
def prepare_data(df: pd.DataFrame, target_col: str):
    """
    피쳐/타겟 분리 및 결측치 처리.
    타겟 컬럼이 없는 행 및 NaN이 과도한 피쳐 제거.
    """
    # 타겟 컬럼 분리
    target_cols_all = [c for c in df.columns if c.startswith("TARGET_")]
    feature_cols = [c for c in df.columns if not c.startswith("TARGET_")]

    if target_col not in df.columns:
        raise ValueError(f"타겟 컬럼 '{target_col}'이 데이터에 없습니다.")

    # 타겟 NaN 제거
    df_clean = df.dropna(subset=[target_col])

    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()

    # 피쳐 NaN: 전방 채움 후 0으로 대체
    X = X.ffill().fillna(0)

    return X, y, df_clean.index


# ──────────────────────────────────────────────
# 시계열 교차검증 (Time Series CV)
# ──────────────────────────────────────────────
def timeseries_cv(model, X: pd.DataFrame, y: pd.Series,
                  n_splits: int = N_SPLITS, test_size: int = TEST_SIZE) -> dict:
    """
    TimeSeriesSplit 기반 교차검증.

    전략:
        - 훈련셋은 항상 과거 데이터만 포함 (슬라이딩 윈도우)
        - 테스트셋은 훈련셋 이후 12개월
        - 미래 데이터 누설 완전 차단

    Returns:
        fold별 예측 및 평균 성능 지표 딕셔너리
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    all_preds = pd.Series(index=y.index, dtype=float)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # 최소 학습 기간 확인
        if len(train_idx) < MIN_TRAIN:
            continue

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        all_preds.iloc[test_idx] = preds
        metrics = evaluate_metrics(y_test.values, preds, name=f"fold_{fold_idx+1}")
        fold_metrics.append(metrics)

    # 전체 예측이 있는 구간 평가
    valid_mask = all_preds.notna()
    if valid_mask.sum() == 0:
        return {"error": "예측 실패"}, all_preds

    overall = evaluate_metrics(y[valid_mask].values, all_preds[valid_mask].values)
    return {
        "overall": overall,
        "folds": fold_metrics,
        "avg_RMSE": round(np.mean([m["RMSE"] for m in fold_metrics]), 4),
        "avg_MAE":  round(np.mean([m["MAE"] for m in fold_metrics]), 4),
        "avg_DirAcc": round(np.mean([m["Direction_Acc"] for m in fold_metrics]), 4),
    }, all_preds


# ──────────────────────────────────────────────
# 모델 정의
# ──────────────────────────────────────────────
def get_models() -> dict:
    """
    벤치마크 모델 딕셔너리 반환.
    각 모델은 sklearn Pipeline으로 래핑하여 스케일링 자동 처리.
    """
    models = {}

    # 1. Ridge (L2 정규화)
    models["Ridge"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0))
    ])

    # 2. Lasso (L1 정규화 + 피쳐 선택)
    models["Lasso"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Lasso(alpha=0.1, max_iter=5000))
    ])

    # 3. XGBoost
    models["XGBoost"] = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )

    # 4. LightGBM
    models["LightGBM"] = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )

    return models


# ──────────────────────────────────────────────
# 피쳐 중요도 시각화
# ──────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list, model_name: str, save_path: str,
                             top_n: int = 25):
    """
    트리 기반 모델(XGBoost, LightGBM)의 피쳐 중요도 상위 N개 시각화.
    어떤 선행 지표가 반도체 업황 예측에 중요한지 해석.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        inner = model.named_steps.get("model")
        if inner and hasattr(inner, "feature_importances_"):
            importances = inner.feature_importances_
        else:
            return
    else:
        return

    # 상위 top_n 피쳐 선택
    idx = np.argsort(importances)[::-1][:top_n]
    top_feats  = [feature_names[i] for i in idx]
    top_scores = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    bars = ax.barh(range(top_n), top_scores[::-1], color="steelblue", alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feats[::-1], fontsize=9)
    ax.set_title(f"{model_name} - 피쳐 중요도 (상위 {top_n}개)", fontsize=12)
    ax.set_xlabel("Importance Score")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {save_path}")


# ──────────────────────────────────────────────
# 예측 결과 시각화
# ──────────────────────────────────────────────
def plot_predictions(y_true: pd.Series, all_preds: dict, save_path: str):
    """
    실제값과 각 모델 예측값을 동시에 플롯.
    Bull/Bear 전환 시점 예측 품질을 시각적으로 평가.
    """
    fig, axes = plt.subplots(len(all_preds), 1,
                              figsize=(14, 4 * len(all_preds)), sharex=True)
    if len(all_preds) == 1:
        axes = [axes]

    colors = ["darkorange", "green", "crimson", "purple", "steelblue"]
    for ax, (model_name, y_pred), color in zip(axes, all_preds.items(), colors):
        valid_mask = y_pred.notna()
        ax.plot(y_true.index, y_true.values,
                color="steelblue", linewidth=1.5, label="실제값", alpha=0.7)
        ax.plot(y_pred[valid_mask].index, y_pred[valid_mask].values,
                color=color, linewidth=1.5, linestyle="--", label=f"{model_name} 예측")
        ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax.fill_between(y_true.index, y_true.values, 0,
                        where=(y_true.values > 0), alpha=0.08, color="green")
        ax.fill_between(y_true.index, y_true.values, 0,
                        where=(y_true.values < 0), alpha=0.08, color="red")
        ax.set_ylabel("YoY (%)")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_title(f"{model_name}", fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("모델별 Worldwide YoY% T+6 예측 vs 실제", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  저장: {save_path}")


# ──────────────────────────────────────────────
# 벤치마크 성능 비교 차트
# ──────────────────────────────────────────────
def plot_benchmark_comparison(results_df: pd.DataFrame, save_path: str):
    """모델별 RMSE, MAE, Direction Accuracy 비교 막대 그래프."""
    metrics = ["RMSE", "MAE", "Direction_Acc"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric in zip(axes, metrics):
        col = metric
        if col not in results_df.columns:
            col = "avg_" + metric
        if col not in results_df.columns:
            continue
        vals = results_df.set_index("model")[col]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(vals)))
        bars = ax.bar(vals.index, vals.values, color=colors, alpha=0.85)
        ax.set_title(metric, fontsize=12)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
        # 값 레이블
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * abs(bar.get_height()),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("모델 성능 벤치마크 비교 (Worldwide YoY% T+6)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  저장: {save_path}")


# ──────────────────────────────────────────────
# N-HiTS (neuralforecast) - 선택적 실행
# ──────────────────────────────────────────────
def train_nhits(df_feat: pd.DataFrame, target_col: str) -> dict:
    """
    N-HiTS 딥러닝 시계열 모델 학습.

    neuralforecast 포맷 요구사항:
        - 'ds' (날짜), 'y' (타겟), 'unique_id' (시계열 ID) 컬럼 필요
        - exogenous 피쳐는 futr_exog_list 또는 hist_exog_list로 지정

    학습 전략:
        - 마지막 24개월을 validation으로 사용
        - 학습 기간: 전체에서 24개월 제외

    Returns:
        성능 지표 딕셔너리 또는 오류 메시지
    """
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS
    except ImportError:
        print("  [N-HiTS] neuralforecast 미설치 - 건너뜀 (pip install neuralforecast)")
        return {"model": "N-HiTS", "error": "미설치"}

    print("  [N-HiTS] 학습 시작...")

    # neuralforecast 포맷으로 변환
    df_nf = df_feat[[target_col]].dropna().copy()
    df_nf = df_nf.reset_index()
    df_nf.columns = ["ds", "y"]
    df_nf["unique_id"] = "semiconductor"
    df_nf["ds"] = pd.to_datetime(df_nf["ds"])

    if len(df_nf) < 48:
        print("  [N-HiTS] 데이터 부족 (최소 48개월 필요) - 건너뜀")
        return {"model": "N-HiTS", "error": "데이터 부족"}

    # validation 구간: 마지막 24개월
    val_size = 24
    df_train = df_nf.iloc[:-val_size]
    df_val   = df_nf.iloc[-val_size:]

    # N-HiTS 모델 구성
    # horizon=6: 6개월 선행 예측, input_size=24: 과거 24개월 입력
    model = NHITS(
        h=6,
        input_size=24,
        max_steps=500,
        n_freq_downsample=[2, 1, 1],
        learning_rate=1e-3,
        scaler_type="standard",
    )
    nf = NeuralForecast(models=[model], freq="ME")

    try:
        nf.fit(df=df_train)
        forecast = nf.predict(futr_df=None)
        y_pred = forecast["NHITS"].values
        y_true = df_val["y"].values[-len(y_pred):]

        metrics = evaluate_metrics(y_true, y_pred, name="N-HiTS")
        print(f"  [N-HiTS] RMSE={metrics['RMSE']}, DirAcc={metrics['Direction_Acc']}")

        # 모델 저장
        nf_path = os.path.join(OUTPUT_DIR, "nhits_model")
        nf.save(nf_path, overwrite=True)
        return metrics
    except Exception as e:
        print(f"  [N-HiTS] 학습 오류: {e}")
        return {"model": "N-HiTS", "error": str(e)}


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  반도체 업황 예측 파이프라인 - Step 4: 모델 학습 & 벤치마크")
    print("=" * 60)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"피쳐 데이터 없음: {INPUT_PATH}\n"
            "먼저 feature_engineering.py를 실행하세요."
        )

    df_feat = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    print(f"[로드] {df_feat.shape[0]}행 × {df_feat.shape[1]}열\n")

    # 데이터 준비 (주 타겟: Worldwide YoY T+6)
    target_col = PRIMARY_TARGET
    if target_col not in df_feat.columns:
        # fallback
        target_candidates = [c for c in df_feat.columns if c.startswith("TARGET_")]
        if not target_candidates:
            raise ValueError("타겟 컬럼이 없습니다. feature_engineering.py를 다시 실행하세요.")
        target_col = target_candidates[0]
        print(f"[주의] PRIMARY_TARGET 없음 → {target_col} 사용")

    X, y, dates = prepare_data(df_feat, target_col)
    print(f"[준비] X: {X.shape}, y: {y.shape}")
    print(f"       타겟 통계: mean={y.mean():.2f}%, std={y.std():.2f}%")
    print(f"       Bull 구간: {(y > 0).sum()}개월 / Bear 구간: {(y <= 0).sum()}개월\n")

    # 모델 딕셔너리
    models = get_models()
    all_results = []
    all_preds   = {}

    # 시계열 CV 벤치마크
    for model_name, model in models.items():
        print(f"[{model_name}] 시계열 교차검증 시작...")
        cv_result, preds = timeseries_cv(model, X, y)

        if "error" in cv_result:
            print(f"  오류: {cv_result['error']}")
            continue

        metrics = cv_result["overall"]
        metrics["model"] = model_name
        all_results.append({
            "model":       model_name,
            "avg_RMSE":    cv_result["avg_RMSE"],
            "avg_MAE":     cv_result["avg_MAE"],
            "avg_DirAcc":  cv_result["avg_DirAcc"],
            "RMSE":        metrics["RMSE"],
            "MAE":         metrics["MAE"],
            "MAPE(%)":     metrics["MAPE(%)"],
            "Direction_Acc": metrics["Direction_Acc"],
            "Asym_Loss":   metrics["Asym_Loss(1.5x)"],
        })
        all_preds[model_name] = preds
        print(f"  RMSE={metrics['RMSE']:.3f}, MAE={metrics['MAE']:.3f}, "
              f"DirAcc={metrics['Direction_Acc']:.3f}\n")

        # 최종 전체 데이터로 모델 재학습 후 저장
        model.fit(X, y)
        pkl_path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_model.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"model": model, "feature_names": list(X.columns)}, f)
        print(f"  모델 저장: {pkl_path}")

    # N-HiTS 선택적 실행
    nhits_result = train_nhits(df_feat, target_col)
    if "error" not in nhits_result:
        all_results.append({
            "model": "N-HiTS",
            "avg_RMSE": nhits_result.get("RMSE", np.nan),
            "avg_MAE":  nhits_result.get("MAE", np.nan),
            "avg_DirAcc": nhits_result.get("Direction_Acc", np.nan),
            "RMSE":     nhits_result.get("RMSE", np.nan),
            "MAE":      nhits_result.get("MAE", np.nan),
            "MAPE(%)":  nhits_result.get("MAPE(%)", np.nan),
            "Direction_Acc": nhits_result.get("Direction_Acc", np.nan),
            "Asym_Loss": nhits_result.get("Asym_Loss(1.5x)", np.nan),
        })

    # 결과 저장
    if all_results:
        results_df = pd.DataFrame(all_results).sort_values("avg_RMSE")
        results_path = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\n  → 성능 비교 저장: {results_path}")
        print("\n" + "=" * 55)
        print("  모델 벤치마크 결과 (avg_RMSE 기준 정렬)")
        print("=" * 55)
        print(results_df[["model", "avg_RMSE", "avg_MAE", "avg_DirAcc"]].to_string(index=False))

        # 최고 성능 모델 표시
        best = results_df.iloc[0]["model"]
        print(f"\n  ▶ 최고 성능 모델: {best}")

    # 예측값 저장
    if all_preds:
        pred_df = pd.DataFrame(all_preds)
        pred_df["y_true"] = y
        pred_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"))

        # 예측 시각화
        plot_predictions(y, all_preds, os.path.join(OUTPUT_DIR, "predictions_plot.png"))

    # 성능 비교 차트
    if all_results:
        plot_benchmark_comparison(results_df, os.path.join(OUTPUT_DIR, "benchmark_plot.png"))

    # 피쳐 중요도 (XGBoost, LightGBM)
    for mname in ["XGBoost", "LightGBM"]:
        if mname in models:
            plot_feature_importance(
                models[mname], list(X.columns), mname,
                os.path.join(OUTPUT_DIR, f"feature_importance_{mname.lower()}.png")
            )

    print("\n[완료] 모든 모델 학습 및 저장이 완료되었습니다.")
    print("       outputs/models/ 폴더를 확인하세요.")

    return results_df if all_results else pd.DataFrame()


if __name__ == "__main__":
    main()
