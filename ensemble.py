"""
ensemble.py
===========
앙상블 모델 학습 및 평가 (Step 6).

전략:
    1. Simple Average    -- XGBoost + LightGBM 단순 평균
    2. Weighted Average  -- CV RMSE 역수 기반 가중 평균
    3. Stacking (Ridge)  -- OOF 예측값을 메타 피쳐로, Ridge 메타 학습기
    4. Stacking (Lasso)  -- OOF 예측값을 메타 피쳐로, Lasso 메타 학습기

기반 모델:
    Optuna 최적화된 XGBoost / LightGBM / Ridge / Lasso (best_params_summary.csv)

입력:
    outputs/data/features_dataset.csv
    outputs/models/best_params_summary.csv

출력:
    outputs/models/ensemble_results.csv
    outputs/models/ensemble_predictions.csv
    outputs/models/ensemble_comparison_plot.png
    outputs/models/best_ensemble.pkl
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH  = os.path.join(BASE_DIR, "outputs", "data", "features_dataset.csv")
PARAMS_PATH = os.path.join(BASE_DIR, "outputs", "models", "best_params_summary.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRIMARY_TARGET = "TARGET_Worldwide_YoY_T6"
N_SPLITS   = 5
TEST_SIZE  = 12
MIN_TRAIN  = 60


# ──────────────────────────────────────────────
# 평가 지표
# ──────────────────────────────────────────────
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def direction_accuracy(y_true, y_pred):
    return ((y_true > 0) == (y_pred > 0)).mean()

def asymmetric_loss(y_true, y_pred, penalty=1.5):
    w = np.where(y_true < 0, penalty, 1.0)
    return np.mean(w * (y_true - y_pred) ** 2)

def evaluate(y_true, y_pred, name=""):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps = 1e-6
    return {
        "model":        name,
        "RMSE":         round(rmse(y_true, y_pred), 4),
        "MAE":          round(mean_absolute_error(y_true, y_pred), 4),
        "DirAcc":       round(direction_accuracy(y_true, y_pred), 4),
        "Asym_Loss":    round(asymmetric_loss(y_true, y_pred), 4),
        "MAPE(%)":      round(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100, 4),
    }


# ──────────────────────────────────────────────
# Optuna 최적 파라미터로 모델 생성
# ──────────────────────────────────────────────
def build_base_models(params_path: str) -> dict:
    """best_params_summary.csv를 읽어 최적 파라미터로 기반 모델 딕셔너리 반환."""
    p = pd.read_csv(params_path).set_index("model")

    def get(model, col, default):
        try:
            v = p.loc[model, col]
            return default if pd.isna(v) else v
        except KeyError:
            return default

    models = {}

    models["Ridge"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=float(get("Ridge", "alpha", 1.0))))
    ])

    models["Lasso"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Lasso(alpha=float(get("Lasso", "alpha", 0.1)), max_iter=10000))
    ])

    models["XGBoost"] = xgb.XGBRegressor(
        n_estimators=      int(get("XGBoost", "n_estimators", 300)),
        learning_rate=     float(get("XGBoost", "learning_rate", 0.05)),
        max_depth=         int(get("XGBoost", "max_depth", 6)),
        subsample=         float(get("XGBoost", "subsample", 0.8)),
        colsample_bytree=  float(get("XGBoost", "colsample_bytree", 0.8)),
        reg_alpha=         float(get("XGBoost", "reg_alpha", 0.1)),
        reg_lambda=        float(get("XGBoost", "reg_lambda", 1.0)),
        min_child_weight=  int(get("XGBoost", "min_child_weight", 3)),
        random_state=42, verbosity=0, n_jobs=-1,
    )

    models["LightGBM"] = lgb.LGBMRegressor(
        n_estimators=     int(get("LightGBM", "n_estimators", 300)),
        learning_rate=    float(get("LightGBM", "learning_rate", 0.05)),
        num_leaves=       int(get("LightGBM", "num_leaves", 31)),
        max_depth=        int(get("LightGBM", "max_depth", 5)),
        min_child_samples=int(get("LightGBM", "min_child_samples", 20)),
        subsample=        float(get("LightGBM", "subsample", 0.8)),
        colsample_bytree= float(get("LightGBM", "colsample_bytree", 0.8)),
        reg_alpha=        float(get("LightGBM", "reg_alpha", 0.1)),
        reg_lambda=       float(get("LightGBM", "reg_lambda", 1.0)),
        random_state=42, verbose=-1, n_jobs=-1,
    )

    return models


# ──────────────────────────────────────────────
# TimeSeriesSplit OOF 예측 수집
# ──────────────────────────────────────────────
def collect_oof(models: dict, X: pd.DataFrame, y: pd.Series):
    """
    각 기반 모델의 Out-Of-Fold 예측을 TimeSeriesSplit으로 수집.
    앙상블 평가 및 스태킹 메타 피쳐로 사용.
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    oof = pd.DataFrame(index=y.index, columns=list(models.keys()), dtype=float)
    fold_rmses = {name: [] for name in models}

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if len(train_idx) < MIN_TRAIN:
            continue
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        for name, model in models.items():
            import copy
            m = copy.deepcopy(model)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)
            oof.loc[oof.index[test_idx], name] = preds
            fold_rmses[name].append(rmse(y_te.values, preds))

    # 유효 구간 마스크 (모든 기반 모델 예측이 있는 행)
    valid = oof.notna().all(axis=1)

    avg_rmses = {name: np.mean(v) for name, v in fold_rmses.items()}
    print("\n  [기반 모델 CV RMSE]")
    for name, v in avg_rmses.items():
        print(f"    {name:12s}: {v:.4f}")

    return oof, valid, avg_rmses


# ──────────────────────────────────────────────
# 앙상블 전략들
# ──────────────────────────────────────────────
def simple_average(oof: pd.DataFrame, models: list, valid):
    """지정된 모델들의 OOF 예측 단순 평균."""
    return oof[models][valid].mean(axis=1)


def weighted_average(oof: pd.DataFrame, avg_rmses: dict, models: list, valid):
    """RMSE 역수 비례 가중 평균. 성능이 좋은 모델에 더 큰 가중치."""
    weights = {m: 1.0 / avg_rmses[m] for m in models}
    total = sum(weights.values())
    weights = {m: w / total for m, w in weights.items()}

    print("\n  [가중 평균 가중치]")
    for m, w in weights.items():
        print(f"    {m:12s}: {w:.4f}  (1/RMSE={1/avg_rmses[m]:.4f})")

    result = sum(oof[m][valid] * w for m, w in weights.items())
    return result, weights


def stacking_cv(models: dict, X: pd.DataFrame, y: pd.Series, meta_name: str = "Ridge"):
    """
    스태킹 앙상블: OOF 예측을 메타 피쳐로 메타 학습기 훈련.

    구조:
        - 기반 모델: XGBoost, LightGBM, Ridge, Lasso
        - 메타 피쳐: 각 기반 모델의 OOF 예측값 4개
        - 메타 학습기: Ridge 또는 Lasso

    누설 방지:
        - 외부 fold마다 내부 CV로 OOF 수집 → 메타 학습기 학습 → 외부 테스트 예측
        - 메타 학습기가 학습할 때 테스트 데이터 정보를 보지 않음
    """
    tscv_outer = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    tscv_inner = TimeSeriesSplit(n_splits=3, test_size=TEST_SIZE)

    if meta_name == "Ridge":
        meta_model = Pipeline([("scaler", StandardScaler()), ("m", Ridge(alpha=1.0))])
    else:
        meta_model = Pipeline([("scaler", StandardScaler()), ("m", Lasso(alpha=0.1, max_iter=10000))])

    all_preds = pd.Series(index=y.index, dtype=float)
    base_names = list(models.keys())

    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(tscv_outer.split(X)):
        if len(outer_train_idx) < MIN_TRAIN:
            continue

        X_out_tr = X.iloc[outer_train_idx]
        y_out_tr  = y.iloc[outer_train_idx]
        X_out_te  = X.iloc[outer_test_idx]
        y_out_te  = y.iloc[outer_test_idx]

        # 내부 CV로 메타 피쳐 생성 (OOF)
        meta_tr = pd.DataFrame(0.0, index=X_out_tr.index, columns=base_names)
        count_tr = pd.Series(0, index=X_out_tr.index)

        for _, (in_tr_idx, in_te_idx) in enumerate(tscv_inner.split(X_out_tr)):
            if len(in_tr_idx) < MIN_TRAIN:
                continue
            X_in_tr = X_out_tr.iloc[in_tr_idx]
            y_in_tr = y_out_tr.iloc[in_tr_idx]
            X_in_te = X_out_tr.iloc[in_te_idx]

            for name, model in models.items():
                import copy
                m = copy.deepcopy(model)
                m.fit(X_in_tr, y_in_tr)
                meta_tr.loc[meta_tr.index[in_te_idx], name] += m.predict(X_in_te)
                count_tr.iloc[in_te_idx] += 1

        # 평균으로 OOF 메타 피쳐 완성
        count_tr = count_tr.replace(0, 1)
        for name in base_names:
            meta_tr[name] /= count_tr

        # 외부 테스트용 메타 피쳐: 전체 outer_train으로 기반 모델 재학습
        meta_te = pd.DataFrame(index=X_out_te.index, columns=base_names, dtype=float)
        for name, model in models.items():
            import copy
            m = copy.deepcopy(model)
            m.fit(X_out_tr, y_out_tr)
            meta_te[name] = m.predict(X_out_te)

        # 메타 학습기 학습 및 예측
        valid_rows = (count_tr > 0)
        import copy
        meta = copy.deepcopy(meta_model)
        meta.fit(meta_tr[valid_rows], y_out_tr[valid_rows])
        all_preds.iloc[outer_test_idx] = meta.predict(meta_te)

    return all_preds


# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
def plot_ensemble_comparison(results_df: pd.DataFrame, y: pd.Series,
                              all_preds: dict, save_dir: str):
    """앙상블 전략 성능 비교 및 예측 시각화."""
    # 1. 성능 비교 막대 그래프
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["RMSE", "DirAcc", "Asym_Loss"]
    labels  = ["RMSE (낮을수록↓)", "방향 정확도 (높을수록↑)", "비대칭 손실 (낮을수록↓)"]

    for ax, metric, label in zip(axes, metrics, labels):
        df_sorted = results_df.sort_values(metric, ascending=(metric != "DirAcc"))
        colors = ["#2ecc71" if "Ensemble" in m or "Stack" in m else "#95a5a6"
                  for m in df_sorted["model"]]
        bars = ax.bar(df_sorted["model"], df_sorted[metric], color=colors, alpha=0.85)
        ax.set_title(label, fontsize=11)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, df_sorted[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)

    fig.suptitle("앙상블 vs 개별 모델 성능 비교 (Worldwide YoY% T+6)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(save_dir, "ensemble_comparison_plot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")

    # 2. 최고 앙상블 예측 시각화
    best_ens = results_df[results_df["model"].str.contains("Ensemble|Stack")]\
               .sort_values("RMSE").iloc[0]["model"]
    valid = all_preds[best_ens].notna()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y.index, y.values, color="steelblue", linewidth=1.5,
            label="실제값", alpha=0.8)
    ax.plot(all_preds[best_ens][valid].index, all_preds[best_ens][valid].values,
            color="#e74c3c", linewidth=1.5, linestyle="--",
            label=f"{best_ens} 예측")
    # 개별 최고 모델 (XGBoost) 비교
    if "XGBoost" in all_preds:
        vx = all_preds["XGBoost"].notna()
        ax.plot(all_preds["XGBoost"][vx].index, all_preds["XGBoost"][vx].values,
                color="#95a5a6", linewidth=1.0, linestyle=":",
                label="XGBoost (단일)", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.fill_between(y.index, y.values, 0,
                    where=(y.values > 0), alpha=0.07, color="green")
    ax.fill_between(y.index, y.values, 0,
                    where=(y.values < 0), alpha=0.07, color="red")
    ax.set_title(f"최고 앙상블({best_ens}) vs XGBoost 단일 모델 예측 비교", fontsize=12)
    ax.set_ylabel("YoY (%)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path2 = os.path.join(save_dir, "ensemble_best_prediction.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  저장: {path2}")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  반도체 업황 예측 파이프라인 - Step 6: 앙상블")
    print("=" * 60)

    # 데이터 로드
    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    target_cols_all = [c for c in df.columns if c.startswith("TARGET_")]
    feat_cols = [c for c in df.columns if not c.startswith("TARGET_")]
    df_clean = df.dropna(subset=[PRIMARY_TARGET])
    X = df_clean[feat_cols].ffill().fillna(0)
    y = df_clean[PRIMARY_TARGET]
    print(f"[로드] X: {X.shape}, y: {y.shape}")

    # 최적 파라미터로 기반 모델 생성
    models = build_base_models(PARAMS_PATH)
    print(f"[기반 모델] {list(models.keys())}")

    # OOF 수집
    print("\n[Step 1] OOF 예측 수집 중...")
    oof, valid, avg_rmses = collect_oof(models, X, y)
    y_valid = y[valid]

    all_results = []
    all_preds   = {}

    # 기반 모델 개별 성능 (참고용)
    print("\n[Step 2] 기반 모델 개별 성능")
    for name in models:
        m = evaluate(y_valid.values, oof[name][valid].values, name=name)
        all_results.append(m)
        all_preds[name] = oof[name]
        print(f"  {name:12s} RMSE={m['RMSE']:.4f}  DirAcc={m['DirAcc']:.4f}")

    # 앙상블 1: XGB + LGBM 단순 평균
    print("\n[Step 3] 앙상블 전략 평가")
    tree_models = ["XGBoost", "LightGBM"]
    pred_avg = simple_average(oof, tree_models, valid)
    m = evaluate(y_valid.values, pred_avg.values, name="Ensemble_Avg(XGB+LGBM)")
    all_results.append(m)
    all_preds["Ensemble_Avg(XGB+LGBM)"] = pred_avg.reindex(y.index)
    print(f"  Avg(XGB+LGBM)    RMSE={m['RMSE']:.4f}  DirAcc={m['DirAcc']:.4f}")

    # 앙상블 2: 전체 4모델 단순 평균
    pred_avg4 = simple_average(oof, list(models.keys()), valid)
    m = evaluate(y_valid.values, pred_avg4.values, name="Ensemble_Avg(All4)")
    all_results.append(m)
    all_preds["Ensemble_Avg(All4)"] = pred_avg4.reindex(y.index)
    print(f"  Avg(All4)        RMSE={m['RMSE']:.4f}  DirAcc={m['DirAcc']:.4f}")

    # 앙상블 3: 가중 평균 (XGB + LGBM)
    pred_wt, weights = weighted_average(oof, avg_rmses, tree_models, valid)
    m = evaluate(y_valid.values, pred_wt.values, name="Ensemble_Weighted(XGB+LGBM)")
    all_results.append(m)
    all_preds["Ensemble_Weighted(XGB+LGBM)"] = pred_wt.reindex(y.index)
    print(f"  Weighted(XGB+LGBM) RMSE={m['RMSE']:.4f}  DirAcc={m['DirAcc']:.4f}")

    # 앙상블 4: 스태킹 (Ridge 메타)
    print("\n[Step 4] 스태킹 학습 중 (Ridge 메타)...")
    pred_stack_ridge = stacking_cv(models, X, y, meta_name="Ridge")
    sv = pred_stack_ridge.notna()
    m = evaluate(y[sv].values, pred_stack_ridge[sv].values, name="Stacking_Ridge")
    all_results.append(m)
    all_preds["Stacking_Ridge"] = pred_stack_ridge
    print(f"  Stacking(Ridge)  RMSE={m['RMSE']:.4f}  DirAcc={m['DirAcc']:.4f}")

    # 앙상블 5: 스태킹 (Lasso 메타)
    print("\n[Step 5] 스태킹 학습 중 (Lasso 메타)...")
    pred_stack_lasso = stacking_cv(models, X, y, meta_name="Lasso")
    sv = pred_stack_lasso.notna()
    m = evaluate(y[sv].values, pred_stack_lasso[sv].values, name="Stacking_Lasso")
    all_results.append(m)
    all_preds["Stacking_Lasso"] = pred_stack_lasso
    print(f"  Stacking(Lasso)  RMSE={m['RMSE']:.4f}  DirAcc={m['DirAcc']:.4f}")

    # 결과 정리
    results_df = pd.DataFrame(all_results).sort_values("RMSE")
    print("\n" + "=" * 60)
    print("  앙상블 결과 요약 (RMSE 기준 정렬)")
    print("=" * 60)
    print(results_df[["model", "RMSE", "MAE", "DirAcc", "Asym_Loss"]].to_string(index=False))

    best = results_df.iloc[0]
    print(f"\n  ▶ 최고 성능: {best['model']}  (RMSE={best['RMSE']:.4f}, DirAcc={best['DirAcc']:.4f})")

    # 저장
    results_df.to_csv(os.path.join(OUTPUT_DIR, "ensemble_results.csv"), index=False)

    pred_df = pd.DataFrame({k: v for k, v in all_preds.items()})
    pred_df["y_true"] = y
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "ensemble_predictions.csv"))

    # 최고 앙상블 모델 전체 데이터로 재학습 후 저장
    best_name = best["model"]
    if "Stacking" in best_name:
        meta_type = "Ridge" if "Ridge" in best_name else "Lasso"
        print(f"\n[저장] 최고 앙상블({best_name}) 전체 데이터 재학습...")
        meta_feats = pd.DataFrame(index=X.index, columns=list(models.keys()), dtype=float)
        for name, model in models.items():
            import copy
            m = copy.deepcopy(model)
            m.fit(X, y)
            meta_feats[name] = m.predict(X)
        meta_model = Pipeline([("scaler", StandardScaler()),
                                ("m", Ridge(alpha=1.0) if meta_type == "Ridge"
                                      else Lasso(alpha=0.1, max_iter=10000))])
        meta_model.fit(meta_feats, y)
        pkl_obj = {"type": "stacking", "base_models": models,
                   "meta_model": meta_model, "feature_names": list(X.columns)}
    else:
        pkl_obj = {"type": "averaging", "weights": weights if "Weighted" in best_name else None,
                   "models": tree_models if "XGB+LGBM" in best_name else list(models.keys()),
                   "feature_names": list(X.columns)}

    with open(os.path.join(OUTPUT_DIR, "best_ensemble.pkl"), "wb") as f:
        pickle.dump(pkl_obj, f)
    print(f"  저장: outputs/models/best_ensemble.pkl")

    # 시각화
    plot_ensemble_comparison(results_df, y, all_preds, OUTPUT_DIR)
    print("\n[완료] 앙상블 학습 및 저장 완료.")

    return results_df


if __name__ == "__main__":
    main()
