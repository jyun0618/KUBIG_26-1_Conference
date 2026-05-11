"""
ensemble.py
===========
앙상블 모델 학습 및 평가 (Step 6).

입력:
    outputs/core/data/features_dataset.csv
    outputs/core/models/best_params_summary.csv

출력:
    outputs/core/models/ensemble_results.csv
    outputs/core/models/ensemble_predictions.csv
    outputs/core/models/ensemble_comparison_plot.png
    outputs/core/models/best_ensemble.pkl
"""

import os
import pickle
import copy
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
INPUT_PATH  = os.path.join(BASE_DIR, "..", "outputs", "core", "data", "features_dataset.csv")
PARAMS_PATH = os.path.join(BASE_DIR, "..", "outputs", "core", "models", "best_params_summary.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "..", "outputs", "core", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRIMARY_TARGET = "TARGET_Worldwide_YoY_T6"
N_SPLITS  = 5
TEST_SIZE = 12
MIN_TRAIN = 60


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
        "model":     name,
        "RMSE":      round(rmse(y_true, y_pred), 4),
        "MAE":       round(mean_absolute_error(y_true, y_pred), 4),
        "DirAcc":    round(direction_accuracy(y_true, y_pred), 4),
        "Asym_Loss": round(asymmetric_loss(y_true, y_pred), 4),
        "MAPE(%)":   round(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100, 4),
    }


def build_base_models(params_path: str) -> dict:
    p = pd.read_csv(params_path).set_index("model")
    def get(model, col, default):
        try:
            v = p.loc[model, col]
            return default if pd.isna(v) else v
        except KeyError:
            return default

    return {
        "Ridge": Pipeline([("scaler", StandardScaler()),
                           ("model", Ridge(alpha=float(get("Ridge","alpha",1.0))))]),
        "Lasso": Pipeline([("scaler", StandardScaler()),
                           ("model", Lasso(alpha=float(get("Lasso","alpha",0.1)), max_iter=10000))]),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=int(get("XGBoost","n_estimators",300)),
            learning_rate=float(get("XGBoost","learning_rate",0.05)),
            max_depth=int(get("XGBoost","max_depth",6)),
            subsample=float(get("XGBoost","subsample",0.8)),
            colsample_bytree=float(get("XGBoost","colsample_bytree",0.8)),
            reg_alpha=float(get("XGBoost","reg_alpha",0.1)),
            reg_lambda=float(get("XGBoost","reg_lambda",1.0)),
            min_child_weight=int(get("XGBoost","min_child_weight",3)),
            random_state=42, verbosity=0, n_jobs=-1),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=int(get("LightGBM","n_estimators",300)),
            learning_rate=float(get("LightGBM","learning_rate",0.05)),
            num_leaves=int(get("LightGBM","num_leaves",31)),
            max_depth=int(get("LightGBM","max_depth",5)),
            min_child_samples=int(get("LightGBM","min_child_samples",20)),
            subsample=float(get("LightGBM","subsample",0.8)),
            colsample_bytree=float(get("LightGBM","colsample_bytree",0.8)),
            reg_alpha=float(get("LightGBM","reg_alpha",0.1)),
            reg_lambda=float(get("LightGBM","reg_lambda",1.0)),
            random_state=42, verbose=-1, n_jobs=-1),
    }


def collect_oof(models, X, y):
    tscv      = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    oof       = pd.DataFrame(index=y.index, columns=list(models.keys()), dtype=float)
    fold_rmses= {name: [] for name in models}
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if len(train_idx) < MIN_TRAIN:
            continue
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        for name, model in models.items():
            m = copy.deepcopy(model)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)
            oof.loc[oof.index[test_idx], name] = preds
            fold_rmses[name].append(rmse(y_te.values, preds))
    valid    = oof.notna().all(axis=1)
    avg_rmses= {name: np.mean(v) for name, v in fold_rmses.items()}
    print("\n  [기반 모델 CV RMSE]")
    for name, v in avg_rmses.items():
        print(f"    {name:12s}: {v:.4f}")
    return oof, valid, avg_rmses


def simple_average(oof, models, valid):
    return oof[models][valid].mean(axis=1)

def weighted_average(oof, avg_rmses, models, valid):
    weights = {m: 1.0 / avg_rmses[m] for m in models}
    total   = sum(weights.values())
    weights = {m: w / total for m, w in weights.items()}
    result  = sum(oof[m][valid] * w for m, w in weights.items())
    return result, weights

def stacking_cv(models, X, y, meta_name="Ridge"):
    tscv_outer = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    tscv_inner = TimeSeriesSplit(n_splits=3, test_size=TEST_SIZE)
    meta_model = (Pipeline([("scaler", StandardScaler()), ("m", Ridge(alpha=1.0))]) if meta_name == "Ridge"
                  else Pipeline([("scaler", StandardScaler()), ("m", Lasso(alpha=0.1, max_iter=10000))]))
    all_preds  = pd.Series(index=y.index, dtype=float)
    base_names = list(models.keys())
    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(tscv_outer.split(X)):
        if len(outer_train_idx) < MIN_TRAIN:
            continue
        X_out_tr = X.iloc[outer_train_idx]
        y_out_tr = y.iloc[outer_train_idx]
        X_out_te = X.iloc[outer_test_idx]
        meta_tr  = pd.DataFrame(0.0, index=X_out_tr.index, columns=base_names)
        count_tr = pd.Series(0, index=X_out_tr.index)
        for _, (in_tr_idx, in_te_idx) in enumerate(tscv_inner.split(X_out_tr)):
            if len(in_tr_idx) < MIN_TRAIN:
                continue
            for name, model in models.items():
                m = copy.deepcopy(model)
                m.fit(X_out_tr.iloc[in_tr_idx], y_out_tr.iloc[in_tr_idx])
                meta_tr.loc[meta_tr.index[in_te_idx], name] += m.predict(X_out_tr.iloc[in_te_idx])
                count_tr.iloc[in_te_idx] += 1
        count_tr = count_tr.replace(0, 1)
        for name in base_names:
            meta_tr[name] /= count_tr
        meta_te = pd.DataFrame(index=X_out_te.index, columns=base_names, dtype=float)
        for name, model in models.items():
            m = copy.deepcopy(model)
            m.fit(X_out_tr, y_out_tr)
            meta_te[name] = m.predict(X_out_te)
        valid_rows = (count_tr > 0)
        meta = copy.deepcopy(meta_model)
        meta.fit(meta_tr[valid_rows], y_out_tr[valid_rows])
        all_preds.iloc[outer_test_idx] = meta.predict(meta_te)
    return all_preds


def plot_ensemble_comparison(results_df, y, all_preds, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["RMSE","DirAcc","Asym_Loss"]
    labels  = ["RMSE (낮을수록↓)","방향 정확도 (높을수록↑)","비대칭 손실 (낮을수록↓)"]
    for ax, metric, label in zip(axes, metrics, labels):
        df_sorted = results_df.sort_values(metric, ascending=(metric != "DirAcc"))
        colors = ["#2ecc71" if "Ensemble" in m or "Stack" in m else "#95a5a6" for m in df_sorted["model"]]
        bars   = ax.bar(df_sorted["model"], df_sorted[metric], color=colors, alpha=0.85)
        ax.set_title(label, fontsize=11)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, df_sorted[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("앙상블 vs 개별 모델 성능 비교 (Worldwide YoY% T+6)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "ensemble_comparison_plot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    best_ens = results_df[results_df["model"].str.contains("Ensemble|Stack")].sort_values("RMSE").iloc[0]["model"]
    valid    = all_preds[best_ens].notna()
    fig, ax  = plt.subplots(figsize=(14, 5))
    ax.plot(y.index, y.values, color="steelblue", linewidth=1.5, label="실제값", alpha=0.8)
    ax.plot(all_preds[best_ens][valid].index, all_preds[best_ens][valid].values,
            color="#e74c3c", linewidth=1.5, linestyle="--", label=f"{best_ens} 예측")
    ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
    ax.set_title(f"최고 앙상블({best_ens}) 예측", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "ensemble_best_prediction.png"), dpi=150)
    plt.close(fig)


def main():
    print("=" * 60)
    print("  반도체 업황 예측 파이프라인 - Step 6: 앙상블")
    print("=" * 60)

    df       = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    feat_cols= [c for c in df.columns if not c.startswith("TARGET_")]
    df_clean = df.dropna(subset=[PRIMARY_TARGET])
    X = df_clean[feat_cols].ffill().fillna(0)
    y = df_clean[PRIMARY_TARGET]

    models = build_base_models(PARAMS_PATH)
    print("\n[Step 1] OOF 예측 수집 중...")
    oof, valid, avg_rmses = collect_oof(models, X, y)
    y_valid = y[valid]

    all_results, all_preds = [], {}
    for name in models:
        m = evaluate(y_valid.values, oof[name][valid].values, name=name)
        all_results.append(m)
        all_preds[name] = oof[name]

    tree_models  = ["XGBoost","LightGBM"]
    pred_avg     = simple_average(oof, tree_models, valid)
    all_results.append(evaluate(y_valid.values, pred_avg.values, name="Ensemble_Avg(XGB+LGBM)"))
    all_preds["Ensemble_Avg(XGB+LGBM)"] = pred_avg.reindex(y.index)

    pred_avg4 = simple_average(oof, list(models.keys()), valid)
    all_results.append(evaluate(y_valid.values, pred_avg4.values, name="Ensemble_Avg(All4)"))
    all_preds["Ensemble_Avg(All4)"] = pred_avg4.reindex(y.index)

    pred_wt, weights = weighted_average(oof, avg_rmses, tree_models, valid)
    all_results.append(evaluate(y_valid.values, pred_wt.values, name="Ensemble_Weighted(XGB+LGBM)"))
    all_preds["Ensemble_Weighted(XGB+LGBM)"] = pred_wt.reindex(y.index)

    pred_stack_ridge = stacking_cv(models, X, y, "Ridge")
    sv = pred_stack_ridge.notna()
    all_results.append(evaluate(y[sv].values, pred_stack_ridge[sv].values, name="Stacking_Ridge"))
    all_preds["Stacking_Ridge"] = pred_stack_ridge

    pred_stack_lasso = stacking_cv(models, X, y, "Lasso")
    sv = pred_stack_lasso.notna()
    all_results.append(evaluate(y[sv].values, pred_stack_lasso[sv].values, name="Stacking_Lasso"))
    all_preds["Stacking_Lasso"] = pred_stack_lasso

    results_df = pd.DataFrame(all_results).sort_values("RMSE")
    print(results_df[["model","RMSE","MAE","DirAcc","Asym_Loss"]].to_string(index=False))

    results_df.to_csv(os.path.join(OUTPUT_DIR, "ensemble_results.csv"), index=False)
    pred_df = pd.DataFrame({k: v for k, v in all_preds.items()})
    pred_df["y_true"] = y
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "ensemble_predictions.csv"))

    best = results_df.iloc[0]
    with open(os.path.join(OUTPUT_DIR, "best_ensemble.pkl"), "wb") as f:
        pickle.dump({"type": "averaging", "weights": weights,
                     "models": tree_models, "feature_names": list(X.columns)}, f)

    plot_ensemble_comparison(results_df, y, all_preds, OUTPUT_DIR)
    print("\n[완료] 앙상블 학습 및 저장 완료.")
    return results_df


if __name__ == "__main__":
    main()
