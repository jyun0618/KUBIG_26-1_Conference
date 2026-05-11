"""
hyperparameter_tuning.py
========================
Optuna를 활용한 하이퍼파라미터 최적화 및 최적 모델 저장 모듈.

입력:
    conference/outputs/core/data/features_dataset.csv

출력:
    conference/outputs/core/models/best_*.pkl
    conference/outputs/core/models/optuna_study_results.csv
    conference/outputs/core/models/best_params_summary.csv
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import lightgbm as lgb

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "..", "outputs", "core", "data", "features_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs", "core", "models")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "optuna_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,   exist_ok=True)

PRIMARY_TARGET = "TARGET_Worldwide_YoY_T6"
N_TRIALS  = 50
N_SPLITS  = 5
TEST_SIZE = 12
MIN_TRAIN = 60


def prepare_data(path: str, target_col: str):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if target_col not in df.columns:
        cands      = [c for c in df.columns if c.startswith("TARGET_")]
        target_col = cands[0]
    feature_cols = [c for c in df.columns if not c.startswith("TARGET_")]
    df_clean     = df.dropna(subset=[target_col])
    X = df_clean[feature_cols].ffill().fillna(0)
    y = df_clean[target_col]
    return X, y


def cv_rmse(model, X, y) -> float:
    tscv      = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    rmse_list = []
    for train_idx, test_idx in tscv.split(X):
        if len(train_idx) < MIN_TRAIN:
            continue
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        rmse_list.append(np.sqrt(mean_squared_error(y.iloc[test_idx], preds)))
    return float(np.mean(rmse_list)) if rmse_list else float("inf")


def objective_ridge(trial, X, y):
    alpha = trial.suggest_float("alpha", 1e-3, 1e3, log=True)
    return cv_rmse(Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=alpha))]), X, y)


def objective_lasso(trial, X, y):
    alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
    return cv_rmse(Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=alpha, max_iter=10000))]), X, y)


def objective_xgboost(trial, X, y):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 7),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42, "verbosity": 0, "n_jobs": -1,
    }
    return cv_rmse(xgb.XGBRegressor(**params), X, y)


def objective_lightgbm(trial, X, y):
    params = {
        "num_leaves":        trial.suggest_int("num_leaves", 15, 63),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42, "verbose": -1, "n_jobs": -1,
    }
    return cv_rmse(lgb.LGBMRegressor(**params), X, y)


def run_optuna_study(model_name, objective_fn, X, y, n_trials=N_TRIALS):
    print(f"\n[Optuna] {model_name} 최적화 시작 (n_trials={n_trials})...")
    study = optuna.create_study(
        study_name=f"{model_name}_optimization", direction="minimize",
        sampler=TPESampler(seed=42), pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )
    def callback(study, trial):
        if trial.number % 10 == 0 or trial.number == n_trials - 1:
            print(f"  Trial {trial.number+1:3d}/{n_trials} | Best RMSE: {study.best_value:.4f}")
    study.optimize(lambda trial: objective_fn(trial, X, y), n_trials=n_trials,
                   callbacks=[callback], show_progress_bar=False)
    print(f"  ✓ {model_name} 최적 RMSE: {study.best_value:.4f}")
    print(f"    최적 파라미터: {study.best_params}")
    return study


def build_best_model(model_name, best_params):
    if model_name == "Ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(**best_params))])
    elif model_name == "Lasso":
        return Pipeline([("scaler", StandardScaler()), ("model", Lasso(**best_params, max_iter=10000))])
    elif model_name == "XGBoost":
        return xgb.XGBRegressor(**best_params, random_state=42, verbosity=0, n_jobs=-1)
    elif model_name == "LightGBM":
        return lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1, n_jobs=-1)
    raise ValueError(f"알 수 없는 모델명: {model_name}")


def main():
    print("=" * 60)
    print("  반도체 업황 예측 파이프라인 - Step 5: 하이퍼파라미터 최적화")
    print("=" * 60)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"피쳐 데이터 없음: {INPUT_PATH}")

    X, y = prepare_data(INPUT_PATH, PRIMARY_TARGET)
    print(f"[준비] X: {X.shape}, y: {y.shape}\n")

    objectives   = {"Ridge": objective_ridge, "Lasso": objective_lasso,
                    "XGBoost": objective_xgboost, "LightGBM": objective_lightgbm}
    studies      = {}
    best_results = {}

    for model_name, obj_fn in objectives.items():
        study      = run_optuna_study(model_name, obj_fn, X, y)
        studies[model_name] = study
        best_model = build_best_model(model_name, study.best_params)
        best_model.fit(X, y)
        pkl_path = os.path.join(OUTPUT_DIR, f"best_{model_name.lower()}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"model": best_model, "feature_names": list(X.columns),
                         "best_cv_rmse": study.best_value}, f)
        print(f"  → 최적 모델 저장: {pkl_path}")
        best_results[model_name] = {"best_rmse": study.best_value, "best_params": study.best_params}

    all_rows = []
    for mname, study in studies.items():
        for trial in study.trials:
            row = {"model": mname, "trial": trial.number, "rmse": trial.value, "state": trial.state.name}
            row.update(trial.params)
            all_rows.append(row)
    if all_rows:
        pd.DataFrame(all_rows).to_csv(os.path.join(OUTPUT_DIR, "optuna_study_results.csv"), index=False)

    summary_rows = []
    for mname, result in best_results.items():
        row = {"model": mname, "best_cv_rmse": result["best_rmse"]}
        row.update(result["best_params"])
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, "best_params_summary.csv"), index=False)

    print("\n" + "=" * 55)
    print(f"  {'모델':<15} {'최적 CV RMSE':>15}")
    for mname, result in sorted(best_results.items(), key=lambda x: x[1]["best_rmse"]):
        print(f"  {mname:<15} {result['best_rmse']:>15.4f}")
    print("\n[완료] 하이퍼파라미터 최적화 완료.")
    return best_results


if __name__ == "__main__":
    main()
