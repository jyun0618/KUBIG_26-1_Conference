"""
v2/walk_forward_predictions.py
===============================
Walk-forward 방식으로 v2 반도체 매출 예측 모델의 예측값을 전 구간에 걸쳐 생성.

개념:
  - 매 시점 t에서, t 이전 데이터로만 모델을 학습한 뒤 t 시점의 값을 예측
  - 미래 데이터 누수(lookahead) 없는 진정한 pseudo-OOS 예측값 생성
  - 결과를 Stock 예측 파이프라인의 [F] 피처 그룹으로 사용

입력:
    conference/outputs/v2/data/features_dataset.csv
    conference/outputs/v2/models/shap_selected_features.txt

출력:
    conference/outputs/v2/models/walk_forward_predictions.csv
      - date 인덱스, v2_xgb 컬럼 (월별 walk-forward 예측값)

설정:
    MIN_TRAIN  = 60   (최소 학습 샘플 수, 약 5년치 월별 데이터)
    TARGET_COL = "TARGET_Worldwide_YoY_T6"
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FEAT_PATH  = os.path.join(_THIS_DIR, "..", "outputs", "v2", "data", "features_dataset.csv")
SHAP_PATH  = os.path.join(_THIS_DIR, "..", "outputs", "v2", "models", "shap_selected_features.txt")
OUT_PATH   = os.path.join(_THIS_DIR, "..", "outputs", "v2", "models", "walk_forward_predictions.csv")

TARGET_COL = "TARGET_Worldwide_YoY_T6"
MIN_TRAIN  = 60
BEAR_PENALTY = 1.5


def asymmetric_mse_xgb(y_true: np.ndarray, y_pred: np.ndarray):
    w    = np.where(y_true < 0, BEAR_PENALTY, 1.0)
    grad = -2.0 * w * (y_true - y_pred)
    hess = 2.0 * w * np.ones_like(y_pred)
    return grad, hess


def main():
    print("=" * 60)
    print("  v2 Walk-Forward Prediction Generator")
    print("=" * 60)

    # ── 데이터 로드 ───────────────────────────────────────────
    df = pd.read_csv(FEAT_PATH, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    print(f"[로드] features_dataset: {df.shape}")

    with open(SHAP_PATH, "r", encoding="utf-8") as f:
        shap_features = [line.strip() for line in f if line.strip()]
    shap_features = [c for c in shap_features if c in df.columns]
    print(f"[로드] SHAP 선택 피처: {len(shap_features)}개")

    if TARGET_COL not in df.columns:
        raise ValueError(f"타겟 컬럼 없음: {TARGET_COL}")

    # target이 있는 행만 사용
    df_valid = df.dropna(subset=[TARGET_COL]).copy()
    print(f"[준비] 타겟 유효 행: {len(df_valid)}  "
          f"({df_valid.index[0].date()} ~ {df_valid.index[-1].date()})")

    X_all = df_valid[shap_features].copy()
    y_all = df_valid[TARGET_COL].copy()

    # 행 단위 결측값: 컬럼 중앙값으로 대체 (전체 기준이 아닌 학습구간 기준으로 매 iter마다 계산)
    dates = df_valid.index.tolist()
    n = len(dates)
    print(f"\n[Walk-Forward] {MIN_TRAIN}개 학습 → {n - MIN_TRAIN}개 예측 시작\n")

    preds = {}

    for i in range(MIN_TRAIN, n):
        X_train = X_all.iloc[:i].copy()
        y_train = y_all.iloc[:i].copy()
        X_pred  = X_all.iloc[[i]].copy()

        # 학습 구간 중앙값으로 결측 대체
        train_median = X_train.median()
        X_train = X_train.fillna(train_median)
        X_pred  = X_pred.fillna(train_median)

        # 타겟 결측 제거
        mask = y_train.notna()
        X_train, y_train = X_train[mask], y_train[mask]
        if len(y_train) < 20:
            continue

        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=1.0,
            objective=asymmetric_mse_xgb,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds[dates[i]] = float(model.predict(X_pred)[0])

        if (i - MIN_TRAIN + 1) % 50 == 0 or i == n - 1:
            print(f"  [{i - MIN_TRAIN + 1:3d}/{n - MIN_TRAIN}] {dates[i].date()}  pred={preds[dates[i]]:.2f}%")

    pred_series = pd.Series(preds, name="v2_xgb")
    pred_series.index.name = "date"
    pred_series.to_csv(OUT_PATH)
    print(f"\n[완료] {len(pred_series)}개 walk-forward 예측값 저장")
    print(f"  기간: {pred_series.index[0].date()} ~ {pred_series.index[-1].date()}")
    print(f"  저장: {OUT_PATH}")
    print(f"\n  예측값 요약:")
    print(pred_series.describe().round(2).to_string())


if __name__ == "__main__":
    main()
