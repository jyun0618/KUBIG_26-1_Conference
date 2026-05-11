"""
model_training.py
=================
반도체 업황 YoY% 6개월 선행 예측 모델 벤치마크 모듈.

입력:
    conference/outputs/core/data/features_dataset.csv

출력:
    conference/outputs/core/models/benchmark_results.csv
    conference/outputs/core/models/predictions.csv
    conference/outputs/core/models/benchmark_plot.png
    conference/outputs/core/models/{model_name}.pkl
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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "..", "outputs", "core", "data", "features_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs", "core", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRIMARY_TARGET   = "TARGET_Worldwide_YoY_T6"
SECONDARY_TARGET = "TARGET_Asia_Pacific_YoY_T6"
N_SPLITS  = 5
MIN_TRAIN = 60
TEST_SIZE = 12


# ──────────────────────────────────────────────
# 평가 지표
# ──────────────────────────────────────────────
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def rmse_bull(y_true, y_pred):
    mask = np.array(y_true) >= 0
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(mean_squared_error(np.array(y_true)[mask], np.array(y_pred)[mask]))

def rmse_bear(y_true, y_pred):
    mask = np.array(y_true) < 0
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(mean_squared_error(np.array(y_true)[mask], np.array(y_pred)[mask]))

def mape(y_true, y_pred):
    eps = 1e-6
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100

def direction_accuracy(y_true, y_pred):
    return ((y_true > 0).astype(int) == (y_pred > 0).astype(int)).mean()

def asymmetric_loss(y_true, y_pred, bear_penalty: float = 1.5):
    errors  = y_true - y_pred
    weights = np.where(y_true < 0, bear_penalty, 1.0)
    return np.mean(weights * errors ** 2)

def weighted_rmse(y_true, y_pred):
    """방향 정확도 기반 가중 RMSE.
    Bull 맞춤=1.0, Bull 틀림=2.0, Bear 맞춤=1.5, Bear 틀림=3.0
    """
    y_true    = np.array(y_true)
    y_pred    = np.array(y_pred)
    is_bear   = y_true < 0
    dir_wrong = (y_true > 0) != (y_pred > 0)
    bear_w    = np.where(is_bear, 1.5, 1.0)
    dir_w     = np.where(dir_wrong, 2.0, 1.0)
    weights   = bear_w * dir_w
    return np.sqrt(np.mean(weights * (y_true - y_pred) ** 2))

def evaluate_metrics(y_true, y_pred, name="") -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "model":         name,
        "RMSE":          round(rmse(y_true, y_pred), 4),
        "RMSE_Bull":     round(rmse_bull(y_true, y_pred), 4),
        "RMSE_Bear":     round(rmse_bear(y_true, y_pred), 4),
        "Direction_Acc": round(direction_accuracy(y_true, y_pred), 4),
        "Asym_Loss":     round(asymmetric_loss(y_true, y_pred, 1.5), 4),
        "Weighted_RMSE": round(weighted_rmse(y_true, y_pred), 4),
    }


# ──────────────────────────────────────────────
# 데이터 준비
# ──────────────────────────────────────────────
def prepare_data(df: pd.DataFrame, target_col: str):
    target_cols_all = [c for c in df.columns if c.startswith("TARGET_")]
    feature_cols    = [c for c in df.columns if not c.startswith("TARGET_")]

    if target_col not in df.columns:
        raise ValueError(f"타겟 컬럼 '{target_col}'이 데이터에 없습니다.")

    df_clean = df.dropna(subset=[target_col])
    X = df_clean[feature_cols].copy().ffill().fillna(0)
    y = df_clean[target_col].copy()
    return X, y, df_clean.index


# ──────────────────────────────────────────────
# 시계열 교차검증
# ──────────────────────────────────────────────
def timeseries_cv(model, X: pd.DataFrame, y: pd.Series,
                  n_splits: int = N_SPLITS, test_size: int = TEST_SIZE) -> tuple:
    tscv      = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    all_preds = pd.Series(index=y.index, dtype=float)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if len(train_idx) < MIN_TRAIN:
            continue
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        all_preds.iloc[test_idx] = preds
        fold_metrics.append(evaluate_metrics(y.iloc[test_idx].values, preds, f"fold_{fold_idx+1}"))

    valid = all_preds.notna()
    if valid.sum() == 0:
        return {"error": "예측 실패"}, all_preds

    def _fold_avg(key):
        vals = [m[key] for m in fold_metrics if not np.isnan(m.get(key, np.nan))]
        return round(np.mean(vals), 4) if vals else np.nan

    overall = evaluate_metrics(y[valid].values, all_preds[valid].values)
    return {
        "overall":          overall,
        "folds":            fold_metrics,
        "avg_RMSE":         _fold_avg("RMSE"),
        "avg_RMSE_Bull":    _fold_avg("RMSE_Bull"),
        "avg_RMSE_Bear":    _fold_avg("RMSE_Bear"),
        "avg_DirAcc":       _fold_avg("Direction_Acc"),
        "avg_Asym_Loss":    _fold_avg("Asym_Loss"),
        "avg_Weighted_RMSE":_fold_avg("Weighted_RMSE"),
    }, all_preds


# ──────────────────────────────────────────────
# 모델 정의
# ──────────────────────────────────────────────
def get_models() -> dict:
    return {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Lasso": Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.1, max_iter=5000))]),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0, n_jobs=-1),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1, n_jobs=-1),
    }


# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────
def plot_feature_importance(model, feature_names, model_name, save_path, top_n=25):
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

    idx        = np.argsort(importances)[::-1][:top_n]
    top_feats  = [feature_names[i] for i in idx]
    top_scores = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(range(top_n), top_scores[::-1], color="steelblue", alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feats[::-1], fontsize=9)
    ax.set_title(f"{model_name} - 피쳐 중요도 (상위 {top_n}개)", fontsize=12)
    ax.set_xlabel("Importance Score")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {save_path}")


def plot_predictions(y_true: pd.Series, all_preds: dict, save_path: str):
    fig, axes = plt.subplots(len(all_preds), 1, figsize=(14, 4 * len(all_preds)), sharex=True)
    if len(all_preds) == 1:
        axes = [axes]
    colors = ["darkorange","green","crimson","purple","steelblue"]
    for ax, (model_name, y_pred), color in zip(axes, all_preds.items(), colors):
        valid = y_pred.notna()
        ax.plot(y_true.index, y_true.values, color="steelblue", linewidth=1.5, label="실제값", alpha=0.7)
        ax.plot(y_pred[valid].index, y_pred[valid].values, color=color, linewidth=1.5, linestyle="--", label=f"{model_name} 예측")
        ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax.fill_between(y_true.index, y_true.values, 0, where=(y_true.values > 0), alpha=0.08, color="green")
        ax.fill_between(y_true.index, y_true.values, 0, where=(y_true.values < 0), alpha=0.08, color="red")
        ax.set_ylabel("YoY (%)")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_title(f"{model_name}", fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_title("모델별 Worldwide YoY% T+6 예측 vs 실제", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  저장: {save_path}")


def plot_benchmark_comparison(results_df: pd.DataFrame, save_path: str):
    metrics = [
        ("avg_RMSE",          "RMSE (전체)"),
        ("avg_RMSE_Bull",     "RMSE (Bull)"),
        ("avg_RMSE_Bear",     "RMSE (Bear)"),
        ("avg_DirAcc",        "Direction Accuracy"),
        ("avg_Asym_Loss",     "Asymmetric Loss"),
        ("avg_Weighted_RMSE", "Weighted RMSE"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for ax, (col, title) in zip(axes, metrics):
        if col not in results_df.columns:
            ax.set_visible(False)
            continue
        vals   = results_df.set_index("model")[col]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(vals)))
        bars   = ax.bar(vals.index, vals.values, color=colors, alpha=0.85)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * abs(bar.get_height()),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("모델 성능 벤치마크 비교 (Worldwide YoY% T+6, CV 평균)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  저장: {save_path}")


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  반도체 업황 예측 파이프라인 - Step 4: 모델 학습 & 벤치마크")
    print("=" * 60)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"피쳐 데이터 없음: {INPUT_PATH}\n먼저 feature_engineering.py를 실행하세요.")

    df_feat    = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    target_col = PRIMARY_TARGET
    if target_col not in df_feat.columns:
        cands = [c for c in df_feat.columns if c.startswith("TARGET_")]
        if not cands:
            raise ValueError("타겟 컬럼이 없습니다.")
        target_col = cands[0]

    X, y, dates = prepare_data(df_feat, target_col)
    print(f"[준비] X: {X.shape}, y: {y.shape}\n")

    models      = get_models()
    all_results = []
    all_preds   = {}

    for model_name, model in models.items():
        print(f"[{model_name}] 시계열 교차검증...")
        cv_result, preds = timeseries_cv(model, X, y)
        if "error" in cv_result:
            continue
        m = cv_result["overall"]
        all_results.append({
            "model":            model_name,
            "avg_RMSE":         cv_result["avg_RMSE"],
            "avg_RMSE_Bull":    cv_result["avg_RMSE_Bull"],
            "avg_RMSE_Bear":    cv_result["avg_RMSE_Bear"],
            "avg_DirAcc":       cv_result["avg_DirAcc"],
            "avg_Asym_Loss":    cv_result["avg_Asym_Loss"],
            "avg_Weighted_RMSE":cv_result["avg_Weighted_RMSE"],
            "RMSE":             m["RMSE"],
            "RMSE_Bull":        m["RMSE_Bull"],
            "RMSE_Bear":        m["RMSE_Bear"],
            "Direction_Acc":    m["Direction_Acc"],
            "Asym_Loss":        m["Asym_Loss"],
            "Weighted_RMSE":    m["Weighted_RMSE"],
        })
        all_preds[model_name] = preds
        print(f"  avg_RMSE={cv_result['avg_RMSE']:.3f}  "
              f"avg_RMSE_Bull={cv_result['avg_RMSE_Bull']:.3f}  "
              f"avg_RMSE_Bear={cv_result['avg_RMSE_Bear']:.3f}  "
              f"avg_DirAcc={cv_result['avg_DirAcc']:.3f}  "
              f"avg_Weighted_RMSE={cv_result['avg_Weighted_RMSE']:.3f}\n")

        model.fit(X, y)
        pkl_path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_model.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"model": model, "feature_names": list(X.columns)}, f)

    if all_results:
        results_df = pd.DataFrame(all_results).sort_values("avg_Weighted_RMSE")
        results_df.to_csv(os.path.join(OUTPUT_DIR, "benchmark_results.csv"), index=False)
        print(results_df[["model","avg_RMSE","avg_RMSE_Bull","avg_RMSE_Bear",
                           "avg_DirAcc","avg_Asym_Loss","avg_Weighted_RMSE"]].to_string(index=False))

    if all_preds:
        pred_df = pd.DataFrame(all_preds)
        pred_df["y_true"] = y
        pred_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"))
        plot_predictions(y, all_preds, os.path.join(OUTPUT_DIR, "predictions_plot.png"))
        plot_benchmark_comparison(results_df, os.path.join(OUTPUT_DIR, "benchmark_plot.png"))

    for mname in ["XGBoost","LightGBM"]:
        if mname in models:
            plot_feature_importance(models[mname], list(X.columns), mname,
                                    os.path.join(OUTPUT_DIR, f"feature_importance_{mname.lower()}.png"))

    print("\n[완료] 모든 모델 학습 완료.")
    return results_df if all_results else pd.DataFrame()


if __name__ == "__main__":
    main()
