"""
hyperparameter_tuning.py
========================
Optuna를 활용한 하이퍼파라미터 최적화 및 최적 모델 저장 모듈.

최적화 전략:
    - Optuna TPE Sampler (Tree-structured Parzen Estimator)
    - 목적함수: 시계열 CV의 평균 RMSE 최소화
    - 조기 종료: MedianPruner (중간 성능 미달 Trial 가지치기)
    - 탐색 공간: 각 모델의 핵심 하이퍼파라미터

최적화 대상 모델:
    1. Ridge / Lasso (alpha)
    2. XGBoost (learning_rate, max_depth, n_estimators, subsample 등)
    3. LightGBM (learning_rate, num_leaves, max_depth 등)

입력:
    conference/outputs/data/features_dataset.csv

출력:
    conference/outputs/models/best_ridge.pkl
    conference/outputs/models/best_lasso.pkl
    conference/outputs/models/best_xgboost.pkl
    conference/outputs/models/best_lightgbm.pkl
    conference/outputs/models/optuna_study_results.csv  -- 전체 Trial 기록
    conference/outputs/models/optuna_plots/             -- Optuna 시각화 결과
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
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH  = os.path.join(BASE_DIR, "outputs", "data", "features_dataset.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs", "models")
PLOT_DIR    = os.path.join(OUTPUT_DIR, "optuna_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# 최적화 설정
PRIMARY_TARGET = "TARGET_Worldwide_YoY_T6"
N_TRIALS    = 50    # 각 모델당 탐색 Trial 수 (빠른 실행 원하면 20으로 낮춤)
N_SPLITS    = 5     # 시계열 CV 폴드 수
TEST_SIZE   = 12    # 테스트 폴드 크기
MIN_TRAIN   = 60    # 최소 학습 기간


# ──────────────────────────────────────────────
# 데이터 준비
# ──────────────────────────────────────────────
def prepare_data(path: str, target_col: str):
    """피쳐 데이터 로드 및 X, y 분리."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    if target_col not in df.columns:
        target_candidates = [c for c in df.columns if c.startswith("TARGET_")]
        target_col = target_candidates[0]
        print(f"[주의] PRIMARY_TARGET 없음 → {target_col} 사용")

    target_cols_all = [c for c in df.columns if c.startswith("TARGET_")]
    feature_cols = [c for c in df.columns if not c.startswith("TARGET_")]

    df_clean = df.dropna(subset=[target_col])
    X = df_clean[feature_cols].ffill().fillna(0)
    y = df_clean[target_col]

    return X, y


# ──────────────────────────────────────────────
# 시계열 CV RMSE 계산
# ──────────────────────────────────────────────
def cv_rmse(model, X: pd.DataFrame, y: pd.Series) -> float:
    """
    TimeSeriesSplit 교차검증으로 평균 RMSE 계산.
    Optuna 목적함수에서 호출.
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    rmse_list = []

    for train_idx, test_idx in tscv.split(X):
        if len(train_idx) < MIN_TRAIN:
            continue
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        rmse_val = np.sqrt(mean_squared_error(y_te, preds))
        rmse_list.append(rmse_val)

    return float(np.mean(rmse_list)) if rmse_list else float("inf")


# ──────────────────────────────────────────────
# Ridge 목적함수
# ──────────────────────────────────────────────
def objective_ridge(trial, X, y):
    """
    Ridge 하이퍼파라미터 탐색 공간:
        alpha: 정규화 강도 (클수록 강한 수축)
               반도체 피쳐가 많고 다중공선성이 높아 alpha 범위를 넓게 설정.
    """
    alpha = trial.suggest_float("alpha", 1e-3, 1e3, log=True)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=alpha))
    ])
    return cv_rmse(model, X, y)


# ──────────────────────────────────────────────
# Lasso 목적함수
# ──────────────────────────────────────────────
def objective_lasso(trial, X, y):
    """
    Lasso 하이퍼파라미터 탐색 공간:
        alpha: 정규화 강도 (클수록 더 많은 피쳐가 0으로 수축)
               Lasso는 자동 피쳐 선택 역할도 수행.
    """
    alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Lasso(alpha=alpha, max_iter=10000))
    ])
    return cv_rmse(model, X, y)


# ──────────────────────────────────────────────
# XGBoost 목적함수
# ──────────────────────────────────────────────
def objective_xgboost(trial, X, y):
    """
    XGBoost 하이퍼파라미터 탐색 공간:
        n_estimators   : 트리 수 (과적합 방지를 위해 learning_rate와 반비례 설정)
        learning_rate  : 학습률 (작을수록 안정적이나 많은 트리 필요)
        max_depth      : 트리 깊이 (반도체 피쳐 수 고려해 3~6)
        subsample      : 행 샘플링 비율 (과적합 방지)
        colsample_bytree: 열 샘플링 비율
        reg_alpha      : L1 정규화
        reg_lambda     : L2 정규화
        min_child_weight: 리프 노드 최소 샘플 수
    """
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":         trial.suggest_int("max_depth", 3, 7),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1,
    }
    model = xgb.XGBRegressor(**params)
    return cv_rmse(model, X, y)


# ──────────────────────────────────────────────
# LightGBM 목적함수
# ──────────────────────────────────────────────
def objective_lightgbm(trial, X, y):
    """
    LightGBM 하이퍼파라미터 탐색 공간:
        num_leaves     : 리프 노드 수 (max_depth보다 직접적인 모델 복잡도 제어)
        max_depth      : 최대 트리 깊이 (-1이면 무제한, num_leaves로 제어)
        learning_rate  : 학습률
        n_estimators   : 부스팅 라운드 수
        min_child_samples: 리프 노드 최소 샘플 (과적합 방지)
        subsample      : 행 샘플링
        colsample_bytree: 열 샘플링
        reg_alpha      : L1
        reg_lambda     : L2
    """
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
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }
    model = lgb.LGBMRegressor(**params)
    return cv_rmse(model, X, y)


# ──────────────────────────────────────────────
# Optuna Study 실행 공통 함수
# ──────────────────────────────────────────────
def run_optuna_study(model_name: str, objective_fn, X, y,
                     n_trials: int = N_TRIALS) -> optuna.Study:
    """
    Optuna Study 실행.
        - TPE Sampler: 베이즈 최적화 계열 (랜덤 서치보다 효율적)
        - MedianPruner: 중간 평가에서 성능 미달 Trial 조기 종료
        - direction="minimize": RMSE 최소화 방향
    """
    print(f"\n[Optuna] {model_name} 최적화 시작 (n_trials={n_trials})...")

    sampler = TPESampler(seed=42)
    pruner  = MedianPruner(n_startup_trials=10, n_warmup_steps=5)

    study = optuna.create_study(
        study_name=f"{model_name}_optimization",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    # tqdm 없이 진행 출력
    def callback(study, trial):
        if trial.number % 10 == 0 or trial.number == n_trials - 1:
            print(f"  Trial {trial.number+1:3d}/{n_trials} | "
                  f"Best RMSE: {study.best_value:.4f}")

    study.optimize(
        lambda trial: objective_fn(trial, X, y),
        n_trials=n_trials,
        callbacks=[callback],
        show_progress_bar=False,
    )

    print(f"  ✓ {model_name} 최적 RMSE: {study.best_value:.4f}")
    print(f"    최적 파라미터: {study.best_params}")

    return study


# ──────────────────────────────────────────────
# 최적 모델 구성 및 저장
# ──────────────────────────────────────────────
def build_best_model(model_name: str, best_params: dict):
    """최적 파라미터로 모델 인스턴스 생성."""
    if model_name == "Ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(**best_params))
        ])
    elif model_name == "Lasso":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Lasso(**best_params, max_iter=10000))
        ])
    elif model_name == "XGBoost":
        return xgb.XGBRegressor(**best_params, random_state=42, verbosity=0, n_jobs=-1)
    elif model_name == "LightGBM":
        return lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1, n_jobs=-1)
    else:
        raise ValueError(f"알 수 없는 모델명: {model_name}")


def save_best_model(model_name: str, model, feature_names: list, best_rmse: float):
    """최적 모델을 pickle로 저장."""
    pkl_path = os.path.join(OUTPUT_DIR, f"best_{model_name.lower()}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "model":         model,
            "feature_names": feature_names,
            "best_cv_rmse":  best_rmse,
        }, f)
    print(f"  → 최적 모델 저장: {pkl_path}")


# ──────────────────────────────────────────────
# Optuna 시각화 (중요도, 최적화 이력)
# ──────────────────────────────────────────────
def save_optuna_plots(study: optuna.Study, model_name: str):
    """
    Optuna 시각화 저장:
        - optimization_history: Trial별 RMSE 변화
        - param_importances: 하이퍼파라미터 중요도
    """
    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
        )

        # 최적화 이력
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_optimization_history(study, ax=ax)
        ax.set_title(f"{model_name} - Optuna 최적화 이력", fontsize=12)
        path = os.path.join(PLOT_DIR, f"{model_name.lower()}_opt_history.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 파라미터 중요도 (Trial 수가 충분할 때만)
        if len(study.trials) >= 10:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_param_importances(study, ax=ax)
            ax.set_title(f"{model_name} - 하이퍼파라미터 중요도", fontsize=12)
            path = os.path.join(PLOT_DIR, f"{model_name.lower()}_param_importance.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"  Optuna 시각화 저장: outputs/models/optuna_plots/")
    except Exception as e:
        print(f"  [Optuna 시각화] 오류 (건너뜀): {e}")


# ──────────────────────────────────────────────
# 모든 Trial 결과 저장
# ──────────────────────────────────────────────
def save_all_trials(studies: dict):
    """모든 모델의 Optuna Trial 결과를 하나의 CSV로 저장."""
    all_rows = []
    for model_name, study in studies.items():
        for trial in study.trials:
            row = {"model": model_name, "trial": trial.number,
                   "rmse": trial.value, "state": trial.state.name}
            row.update(trial.params)
            all_rows.append(row)

    if all_rows:
        df_trials = pd.DataFrame(all_rows)
        path = os.path.join(OUTPUT_DIR, "optuna_study_results.csv")
        df_trials.to_csv(path, index=False)
        print(f"\n  → 전체 Trial 기록 저장: {path}")


# ──────────────────────────────────────────────
# 최종 성능 비교 출력
# ──────────────────────────────────────────────
def print_final_comparison(best_results: dict):
    """베이스라인(default)과 최적화 후 성능 비교 출력."""
    print("\n" + "=" * 55)
    print("  Optuna 최적화 결과 요약")
    print("=" * 55)
    print(f"  {'모델':<15} {'최적 CV RMSE':>15}")
    print("  " + "-" * 32)
    for model_name, result in sorted(best_results.items(), key=lambda x: x[1]["best_rmse"]):
        print(f"  {model_name:<15} {result['best_rmse']:>15.4f}")
    best_model = min(best_results.items(), key=lambda x: x[1]["best_rmse"])
    print(f"\n  ▶ 최고 성능 모델: {best_model[0]} (RMSE={best_model[1]['best_rmse']:.4f})")


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  반도체 업황 예측 파이프라인 - Step 5: 하이퍼파라미터 최적화")
    print("=" * 60)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"피쳐 데이터 없음: {INPUT_PATH}\n"
            "먼저 feature_engineering.py를 실행하세요."
        )

    X, y = prepare_data(INPUT_PATH, PRIMARY_TARGET)
    print(f"[준비] X: {X.shape}, y: {y.shape}\n")

    # 최적화 대상 모델 및 목적함수
    objectives = {
        "Ridge":    objective_ridge,
        "Lasso":    objective_lasso,
        "XGBoost":  objective_xgboost,
        "LightGBM": objective_lightgbm,
    }

    studies     = {}
    best_results = {}

    for model_name, obj_fn in objectives.items():
        # Optuna Study 실행
        study = run_optuna_study(model_name, obj_fn, X, y, n_trials=N_TRIALS)
        studies[model_name] = study

        # 최적 모델 구성 및 전체 데이터로 학습
        best_model = build_best_model(model_name, study.best_params)
        best_model.fit(X, y)

        # 저장
        save_best_model(model_name, best_model, list(X.columns), study.best_value)
        best_results[model_name] = {
            "best_rmse":   study.best_value,
            "best_params": study.best_params,
        }

        # 시각화
        save_optuna_plots(study, model_name)

    # 전체 Trial 기록 저장
    save_all_trials(studies)

    # 최종 비교 출력
    print_final_comparison(best_results)

    # best_params 요약 저장
    summary_rows = []
    for model_name, result in best_results.items():
        row = {"model": model_name, "best_cv_rmse": result["best_rmse"]}
        row.update(result["best_params"])
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "best_params_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  → 최적 파라미터 요약 저장: {summary_path}")
    print("\n[완료] 하이퍼파라미터 최적화가 완료되었습니다.")
    print("       최적 모델: outputs/models/best_*.pkl")

    return best_results


if __name__ == "__main__":
    main()
