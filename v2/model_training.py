"""
v2/model_training.py
====================
core/model_training.py 확장판.

변경사항:
    1. SHAP 기반 피쳐 선택 (171개 → 상위 50개)
    2. 비대칭 손실 함수 (Bear 국면 ×1.5) XGBoost/LightGBM 학습 연결
    3. v1 vs v2 성능 비교 리포트

입력:
    conference/outputs/v2/data/features_dataset.csv

출력:
    conference/outputs/v2/models/benchmark_results.csv
    conference/outputs/v2/models/shap_selected_features.txt
    conference/outputs/v2/models/shap_importance.png
    conference/outputs/v2/models/predictions.png
    conference/outputs/v2/models/*_asym_shap.pkl
"""

import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb

from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "core"))

from model_training import (
    rmse, mape, direction_accuracy, asymmetric_loss, evaluate_metrics,
    prepare_data, timeseries_cv, plot_predictions, plot_benchmark_comparison,
    PRIMARY_TARGET, N_SPLITS, MIN_TRAIN, TEST_SIZE,
)

# v2 전용 경로
BASE_DIR   = _THIS_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs", "v2", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEAR_PENALTY = 1.5
SHAP_TOP_N   = 50


# ──────────────────────────────────────────────
# 비대칭 손실 Custom Objective
# ──────────────────────────────────────────────
def asymmetric_mse_xgb(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """XGBoost sklearn API custom objective (y_true, y_pred) 시그니처."""
    w    = np.where(y_true < 0, BEAR_PENALTY, 1.0)
    grad = -2.0 * w * (y_true - y_pred)
    hess = 2.0 * w * np.ones_like(y_pred)
    return grad, hess


def asymmetric_mse_lgb(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """LightGBM sklearn API custom objective (y_true, y_pred) 시그니처."""
    w    = np.where(y_true < 0, BEAR_PENALTY, 1.0)
    grad = -2.0 * w * (y_true - y_pred)
    hess = 2.0 * w * np.ones_like(y_pred)
    return grad, hess


# ──────────────────────────────────────────────
# SHAP 기반 피쳐 선택
# ──────────────────────────────────────────────
def select_features_by_shap(X: pd.DataFrame, y: pd.Series, n_top: int = SHAP_TOP_N) -> tuple:
    """전체 피쳐 XGBoost → SHAP mean|value| 기준 상위 n_top 선택."""
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP 미설치: pip install shap")

    print(f"[SHAP] 전체 {X.shape[1]}개 피쳐로 XGBoost 학습 중...")
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                              random_state=42, verbosity=0, n_jobs=-1)
    model.fit(X, y)

    print("[SHAP] SHAP 값 계산 중...")
    explainer     = shap.TreeExplainer(model)
    shap_values   = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx       = np.argsort(mean_abs_shap)[::-1][:n_top]
    selected      = [X.columns[i] for i in top_idx]

    print(f"[SHAP] 상위 {n_top}개 피쳐 선택 완료 ({X.shape[1]}개 → {n_top}개)")
    return selected, mean_abs_shap


def plot_shap_importance(feature_names, mean_abs_shap, save_path, top_n=30):
    idx    = np.argsort(mean_abs_shap)[::-1][:top_n]
    feats  = [feature_names[i] for i in idx]
    scores = mean_abs_shap[idx]
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(range(top_n), scores[::-1], color="steelblue", alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feats[::-1], fontsize=8)
    ax.set_title(f"SHAP mean|value| 피쳐 중요도 (상위 {top_n}개)", fontsize=12)
    ax.set_xlabel("mean |SHAP value|")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {save_path}")


# ──────────────────────────────────────────────
# 비대칭 손실 + SHAP 피쳐 모델 정의
# ──────────────────────────────────────────────
def get_models_v2() -> dict:
    return {
        "Ridge_SHAP": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Lasso_SHAP": Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.1, max_iter=5000))]),
        "XGBoost_Asym_SHAP": xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            objective=asymmetric_mse_xgb, random_state=42, verbosity=0, n_jobs=-1),
        "LightGBM_Asym_SHAP": lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            objective=asymmetric_mse_lgb, random_state=42, verbose=-1, n_jobs=-1),
    }


def timeseries_cv_v2(model, X, y):
    tscv      = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
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
        "avg_AsymLoss":     _fold_avg("Asym_Loss"),
        "avg_Weighted_RMSE":_fold_avg("Weighted_RMSE"),
    }, all_preds


def print_comparison(results_v1, results_v2):
    print("\n" + "=" * 85)
    print("  v1 (기존 모델) vs v2 (비대칭 손실 + SHAP 피쳐) 비교")
    print("=" * 85)
    def fmt(rows, label):
        df = pd.DataFrame(rows)
        sort_col = "avg_Weighted_RMSE" if "avg_Weighted_RMSE" in df.columns else "avg_RMSE"
        df = df.sort_values(sort_col, na_position="last")
        print(f"\n  [{label}]")
        print(f"  {'모델':<28} {'avg_RMSE':>10} {'avg_RMSE_Bull':>14} {'avg_RMSE_Bear':>14} "
              f"{'avg_DirAcc':>11} {'avg_WtRMSE':>11}")
        print("  " + "-" * 90)
        for _, r in df.iterrows():
            print(f"  {r['model']:<28} {r.get('avg_RMSE', float('nan')):>10.4f} "
                  f"{r.get('avg_RMSE_Bull', float('nan')):>14.4f} "
                  f"{r.get('avg_RMSE_Bear', float('nan')):>14.4f} "
                  f"{r.get('avg_DirAcc', float('nan')):>11.4f} "
                  f"{r.get('avg_Weighted_RMSE', float('nan')):>11.4f}")
    fmt(results_v1, "v1 기존")
    fmt(results_v2, "v2 신규 (비대칭 손실 + SHAP)")


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  반도체 업황 예측 파이프라인 - Step 4 v2: 모델 학습")
    print("  [비대칭 손실 학습] + [SHAP 기반 피쳐 선택]")
    print("=" * 60)

    v2_path = os.path.join(BASE_DIR, "..", "outputs", "v2", "data", "features_dataset.csv")
    v1_path = os.path.join(BASE_DIR, "..", "outputs", "core", "data", "features_dataset.csv")
    if os.path.exists(v2_path):
        input_path = v2_path
        print(f"[로드] features_dataset.csv (v2) 사용\n")
    elif os.path.exists(v1_path):
        print(f"[주의] v2 피쳐 없음 → core 피쳐 사용\n")
        input_path = v1_path
    else:
        raise FileNotFoundError("피쳐 데이터 없음. v2/feature_engineering.py를 실행하세요.")

    df_feat = pd.read_csv(input_path, index_col=0, parse_dates=True)
    print(f"[로드] {df_feat.shape[0]}행 × {df_feat.shape[1]}열")

    target_col = PRIMARY_TARGET
    if target_col not in df_feat.columns:
        cands = [c for c in df_feat.columns if c.startswith("TARGET_")]
        if not cands:
            raise ValueError("타겟 컬럼 없음.")
        target_col = cands[0]

    X_full, y, _ = prepare_data(df_feat, target_col)
    print(f"[준비] X: {X_full.shape}, y: {y.shape}")
    print(f"       Bear 구간: {(y <= 0).sum()}개월 / Bull 구간: {(y > 0).sum()}개월\n")

    # STEP 1: SHAP 피쳐 선택
    print("─" * 50)
    print("STEP 1: SHAP 기반 피쳐 선택")
    print("─" * 50)
    selected_features, mean_abs_shap = select_features_by_shap(X_full, y, SHAP_TOP_N)

    feat_list_path = os.path.join(OUTPUT_DIR, "shap_selected_features.txt")
    with open(feat_list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(selected_features))
    print(f"  선택 피쳐 목록 저장: {feat_list_path}")

    plot_shap_importance(list(X_full.columns), mean_abs_shap,
                         os.path.join(OUTPUT_DIR, "shap_importance.png"))
    X_shap = X_full[selected_features]

    # STEP 2: 비대칭 손실 + SHAP 피쳐 학습
    print("\n" + "─" * 50)
    print("STEP 2: 비대칭 손실 + SHAP 피쳐 모델 학습")
    print("─" * 50)

    models_v2  = get_models_v2()
    results_v2 = []
    preds_v2   = {}

    for name, model in models_v2.items():
        print(f"\n[{name}] 시계열 교차검증...")
        cv_result, preds = timeseries_cv_v2(model, X_shap, y)
        if "error" in cv_result:
            continue
        m = cv_result["overall"]
        results_v2.append({
            "model":            name,
            "avg_RMSE":         cv_result["avg_RMSE"],
            "avg_RMSE_Bull":    cv_result["avg_RMSE_Bull"],
            "avg_RMSE_Bear":    cv_result["avg_RMSE_Bear"],
            "avg_DirAcc":       cv_result["avg_DirAcc"],
            "avg_AsymLoss":     cv_result["avg_AsymLoss"],
            "avg_Weighted_RMSE":cv_result["avg_Weighted_RMSE"],
            "RMSE":             m["RMSE"],
            "RMSE_Bull":        m["RMSE_Bull"],
            "RMSE_Bear":        m["RMSE_Bear"],
            "Direction_Acc":    m["Direction_Acc"],
            "Asym_Loss":        m["Asym_Loss"],
            "Weighted_RMSE":    m["Weighted_RMSE"],
        })
        preds_v2[name] = preds
        print(f"  avg_RMSE={cv_result['avg_RMSE']:.3f}  "
              f"avg_RMSE_Bull={cv_result['avg_RMSE_Bull']:.3f}  "
              f"avg_RMSE_Bear={cv_result['avg_RMSE_Bear']:.3f}  "
              f"avg_DirAcc={cv_result['avg_DirAcc']:.3f}  "
              f"avg_Weighted_RMSE={cv_result['avg_Weighted_RMSE']:.3f}")

        model.fit(X_shap, y)
        pkl_name = name.lower().replace(" ", "_") + ".pkl"
        with open(os.path.join(OUTPUT_DIR, pkl_name), "wb") as f:
            pickle.dump({"model": model, "feature_names": selected_features}, f)

    # STEP 3: 저장 및 비교
    if results_v2:
        df_res = pd.DataFrame(results_v2).sort_values("avg_Weighted_RMSE")
        df_res.to_csv(os.path.join(OUTPUT_DIR, "benchmark_results.csv"), index=False)

    if preds_v2:
        pred_df = pd.DataFrame(preds_v2)
        pred_df["y_true"] = y
        pred_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"))
        plot_predictions(y, preds_v2, os.path.join(OUTPUT_DIR, "predictions.png"))
        plot_benchmark_comparison(pd.DataFrame(results_v2),
                                  os.path.join(OUTPUT_DIR, "benchmark_plot.png"))

    v1_bench = os.path.join(BASE_DIR, "..", "outputs", "core", "models", "benchmark_results.csv")
    if os.path.exists(v1_bench) and results_v2:
        v1_df   = pd.read_csv(v1_bench)
        v1_rows = v1_df.to_dict("records")
        for r in v1_rows:
            if "avg_AsymLoss" not in r:
                r["avg_AsymLoss"] = float("nan")
        print_comparison(v1_rows, results_v2)

    if results_v2:
        best = min(results_v2, key=lambda x: x.get("avg_Weighted_RMSE") or float("inf"))
        print(f"\n  ▶ v2 최고 성능: {best['model']} "
              f"(avg_RMSE={best['avg_RMSE']:.4f}, avg_DirAcc={best['avg_DirAcc']:.4f}, "
              f"avg_Weighted_RMSE={best['avg_Weighted_RMSE']:.4f})")
        print(f"\n  다음 단계: v2/hyperparameter_tuning.py 실행")

    print("\n[완료] model_training v2 종료.")
    return results_v2


if __name__ == "__main__":
    main()
