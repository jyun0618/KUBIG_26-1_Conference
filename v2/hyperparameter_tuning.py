"""
v2/hyperparameter_tuning.py
============================
core/hyperparameter_tuning.py 확장판.

변경사항:
    1. SHAP 선택 피쳐 사용 (shap_selected_features.txt 로드)
    2. XGBoost/LightGBM을 비대칭 손실로 학습 + 비대칭 손실로 Optuna 최소화

입력:
    conference/outputs/v2/data/features_dataset.csv
    conference/outputs/v2/models/shap_selected_features.txt

출력:
    conference/outputs/v2/models/best_xgboost.pkl
    conference/outputs/v2/models/best_lightgbm.pkl
    conference/outputs/v2/models/best_ridge.pkl
    conference/outputs/v2/models/best_lasso.pkl
    conference/outputs/v2/models/optuna_study_results.csv
    conference/outputs/v2/models/best_params_summary.csv
"""

import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "core"))

# v2 전용 경로
BASE_DIR   = _THIS_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs", "v2", "models")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "optuna_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,   exist_ok=True)

SHAP_FEATURES_PATH = os.path.join(OUTPUT_DIR, "shap_selected_features.txt")
PRIMARY_TARGET     = "TARGET_Worldwide_YoY_T6"
BEAR_PENALTY       = 1.5
N_TRIALS           = 50
N_SPLITS           = 5
TEST_SIZE          = 12
MIN_TRAIN          = 60


# ──────────────────────────────────────────────
# 비대칭 손실 함수 (v2/model_training.py와 동일, 독립 정의)
# ──────────────────────────────────────────────
def asymmetric_mse_xgb(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    w    = np.where(y_true < 0, BEAR_PENALTY, 1.0)
    grad = -2.0 * w * (y_true - y_pred)
    hess = 2.0 * w * np.ones_like(y_pred)
    return grad, hess


def asymmetric_mse_lgb(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    w    = np.where(y_true < 0, BEAR_PENALTY, 1.0)
    grad = -2.0 * w * (y_true - y_pred)
    hess = 2.0 * w * np.ones_like(y_pred)
    return grad, hess


# ──────────────────────────────────────────────
# 데이터 준비
# ──────────────────────────────────────────────
def load_shap_features(path: str) -> list:
    if not os.path.exists(path):
        print(f"[주의] SHAP 피쳐 목록 없음 ({path})\n       v2/model_training.py를 먼저 실행하세요.")
        return []
    with open(path, encoding="utf-8") as f:
        features = [line.strip() for line in f if line.strip()]
    print(f"[SHAP 피쳐] {len(features)}개 로드")
    return features


def prepare_data(path: str, target_col: str, feature_subset: list = None):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if target_col not in df.columns:
        cands      = [c for c in df.columns if c.startswith("TARGET_")]
        target_col = cands[0]
    feature_cols = [c for c in df.columns if not c.startswith("TARGET_")]
    if feature_subset:
        feature_cols = [c for c in feature_subset if c in feature_cols] or feature_cols
    df_clean = df.dropna(subset=[target_col])
    X = df_clean[feature_cols].ffill().fillna(0)
    y = df_clean[target_col]
    return X, y


# ──────────────────────────────────────────────
# 비대칭 손실 기반 CV
# ──────────────────────────────────────────────
def cv_asym_loss(model, X: pd.DataFrame, y: pd.Series) -> float:
    tscv   = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    losses = []
    for tr_idx, te_idx in tscv.split(X):
        if len(tr_idx) < MIN_TRAIN:
            continue
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = model.predict(X.iloc[te_idx])
        y_te  = y.iloc[te_idx].values
        w     = np.where(y_te < 0, BEAR_PENALTY, 1.0)
        losses.append(float(np.mean(w * (y_te - preds) ** 2)))
    return float(np.mean(losses)) if losses else float("inf")


def cv_rmse_standard(model, X: pd.DataFrame, y: pd.Series) -> float:
    tscv      = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    rmse_list = []
    for tr_idx, te_idx in tscv.split(X):
        if len(tr_idx) < MIN_TRAIN:
            continue
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = model.predict(X.iloc[te_idx])
        rmse_list.append(np.sqrt(mean_squared_error(y.iloc[te_idx], preds)))
    return float(np.mean(rmse_list)) if rmse_list else float("inf")


# ──────────────────────────────────────────────
# Optuna 목적함수
# ──────────────────────────────────────────────
def objective_ridge_v2(trial, X, y):
    alpha = trial.suggest_float("alpha", 1e-3, 1e3, log=True)
    return cv_rmse_standard(
        Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=alpha))]), X, y)


def objective_lasso_v2(trial, X, y):
    alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
    return cv_rmse_standard(
        Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=alpha, max_iter=10000))]), X, y)


def objective_xgboost_v2(trial, X, y):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 7),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "objective": asymmetric_mse_xgb,
        "random_state": 42, "verbosity": 0, "n_jobs": -1,
    }
    return cv_asym_loss(xgb.XGBRegressor(**params), X, y)


def objective_lightgbm_v2(trial, X, y):
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
        "objective": asymmetric_mse_lgb,
        "random_state": 42, "verbose": -1, "n_jobs": -1,
    }
    return cv_asym_loss(lgb.LGBMRegressor(**params), X, y)


# ──────────────────────────────────────────────
# Optuna Study 실행 및 저장
# ──────────────────────────────────────────────
def run_optuna_study(model_name, objective_fn, X, y, n_trials=N_TRIALS):
    print(f"\n[Optuna v2] {model_name} 최적화 시작 (n_trials={n_trials})...")
    study = optuna.create_study(
        study_name=f"{model_name}_v2_optimization", direction="minimize",
        sampler=TPESampler(seed=42), pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )
    def callback(study, trial):
        if trial.number % 10 == 0 or trial.number == n_trials - 1:
            print(f"  Trial {trial.number+1:3d}/{n_trials} | Best Loss: {study.best_value:.4f}")
    study.optimize(lambda trial: objective_fn(trial, X, y), n_trials=n_trials,
                   callbacks=[callback], show_progress_bar=False)
    print(f"  ✓ {model_name} 최적 손실: {study.best_value:.4f}")
    print(f"    최적 파라미터: {study.best_params}")
    return study


def build_best_model_v2(model_name, best_params):
    if model_name == "Ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(**best_params))])
    elif model_name == "Lasso":
        return Pipeline([("scaler", StandardScaler()), ("model", Lasso(**best_params, max_iter=10000))])
    elif model_name == "XGBoost":
        return xgb.XGBRegressor(**best_params, objective=asymmetric_mse_xgb,
                                 random_state=42, verbosity=0, n_jobs=-1)
    elif model_name == "LightGBM":
        return lgb.LGBMRegressor(**best_params, objective=asymmetric_mse_lgb,
                                  random_state=42, verbose=-1, n_jobs=-1)
    raise ValueError(f"알 수 없는 모델명: {model_name}")


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  반도체 업황 예측 파이프라인 - Step 5 v2: 하이퍼파라미터 최적화")
    print("  [비대칭 손실 Optuna] + [SHAP 선택 피쳐]")
    print("=" * 60)

    v2_path = os.path.join(BASE_DIR, "..", "outputs", "v2", "data", "features_dataset.csv")
    v1_path = os.path.join(BASE_DIR, "..", "outputs", "core", "data", "features_dataset.csv")
    input_path = v2_path if os.path.exists(v2_path) else v1_path
    if not os.path.exists(input_path):
        raise FileNotFoundError("피쳐 데이터 없음. v2/feature_engineering.py를 실행하세요.")

    shap_features = load_shap_features(SHAP_FEATURES_PATH)
    X, y          = prepare_data(input_path, PRIMARY_TARGET, shap_features or None)
    print(f"[준비] X: {X.shape}, y: {y.shape}")
    print(f"       Bear 구간: {(y <= 0).sum()}개월 / Bull 구간: {(y > 0).sum()}개월\n")

    objectives = {
        "Ridge": objective_ridge_v2, "Lasso": objective_lasso_v2,
        "XGBoost": objective_xgboost_v2, "LightGBM": objective_lightgbm_v2,
    }
    studies      = {}
    best_results = {}

    for model_name, obj_fn in objectives.items():
        study      = run_optuna_study(model_name, obj_fn, X, y)
        studies[model_name] = study
        best_model = build_best_model_v2(model_name, study.best_params)
        best_model.fit(X, y)
        pkl_path = os.path.join(OUTPUT_DIR, f"best_{model_name.lower()}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"model": best_model, "feature_names": list(X.columns),
                         "best_cv_loss": study.best_value,
                         "loss_type": "asymmetric_mse (Bear x1.5)" if model_name in ("XGBoost","LightGBM") else "RMSE"}, f)
        print(f"  → 최적 모델 저장: {pkl_path}")
        best_results[model_name] = {"best_loss": study.best_value, "best_params": study.best_params}

    all_rows = []
    for mname, study in studies.items():
        for trial in study.trials:
            row = {"model": mname, "trial": trial.number, "loss": trial.value, "state": trial.state.name}
            row.update(trial.params)
            all_rows.append(row)
    if all_rows:
        pd.DataFrame(all_rows).to_csv(os.path.join(OUTPUT_DIR, "optuna_study_results.csv"), index=False)

    summary_rows = []
    for mname, result in best_results.items():
        row = {"model": mname, "best_cv_loss": result["best_loss"]}
        row.update(result["best_params"])
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR, "best_params_summary.csv"), index=False)

    print("\n" + "=" * 55)
    print("  Optuna v2 결과 (비대칭 손실 기준)")
    print("=" * 55)
    print(f"  {'모델':<15} {'최적 CV Loss':>20}")
    for mname, result in sorted(best_results.items(), key=lambda x: x[1]["best_loss"]):
        print(f"  {mname:<15} {result['best_loss']:>20.4f}")

    best_model = min(best_results.items(), key=lambda x: x[1]["best_loss"])
    print(f"\n  ▶ 최고 성능: {best_model[0]} (손실={best_model[1]['best_loss']:.4f})")
    print("\n[완료] hyperparameter_tuning v2 종료.")
    return best_results


if __name__ == "__main__":
    main()
