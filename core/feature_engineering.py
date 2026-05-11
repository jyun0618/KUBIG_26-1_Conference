"""
feature_engineering.py
=======================
6개월 선행 타겟 생성, 이동평균/변동성/Lag 피쳐 엔지니어링 모듈.

입력:
    conference/outputs/core/data/merged_dataset.csv

출력:
    conference/outputs/core/data/features_dataset.csv
    conference/outputs/core/eda/08_feature_overview.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH  = os.path.join(BASE_DIR, "..", "outputs", "core", "data", "merged_dataset.csv")
OUTPUT_DATA = os.path.join(BASE_DIR, "..", "outputs", "core", "data")
OUTPUT_EDA  = os.path.join(BASE_DIR, "..", "outputs", "core", "eda")
os.makedirs(OUTPUT_DATA, exist_ok=True)
os.makedirs(OUTPUT_EDA,  exist_ok=True)

TARGET_HORIZON = 6
TARGETS        = ["Worldwide", "Asia_Pacific"]
MA_WINDOWS     = [3, 6, 12]
VOL_WINDOWS    = [3, 6]
LAG_MONTHS     = [6, 12]

EQUITY_COLS    = ["Ret_SOX","Ret_NVDA","Ret_TSM","Ret_ASML","Ret_Samsung","Ret_SKHynix"]
FRED_LEVEL_COLS= ["FRED_SemiProd","FRED_ISM_Mfg","FRED_T10Y2Y","FRED_IndProd",
                  "FRED_PCE_Core","FRED_MfgEmp","FRED_ConsSenti","FRED_NewOrder"]


# ──────────────────────────────────────────────
# 헬퍼 함수
# ──────────────────────────────────────────────
def yoy_pct(s: pd.Series) -> pd.Series:
    return s.pct_change(periods=12) * 100

def mom_pct(s: pd.Series) -> pd.Series:
    return s.pct_change() * 100

def add_lag_features(df, col, lags):
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

def add_moving_average(df, col, windows):
    for w in windows:
        df[f"{col}_ma{w}"] = df[col].rolling(window=w, min_periods=w).mean()
    return df

def add_volatility(df, col, windows):
    for w in windows:
        df[f"{col}_vol{w}"] = df[col].rolling(window=w, min_periods=w).std()
    return df

def add_momentum(df, col):
    df[f"{col}_momentum_3_12"] = df[col].diff(3) - df[col].diff(12)
    return df

def add_acceleration(df, col):
    df[f"{col}_accel"] = df[col].diff(1)
    return df

def add_cycle_position(df, col, window=24):
    def percentile_rank(x):
        if len(x) < 3:
            return np.nan
        return (x[-1] - x.min()) / (x.max() - x.min() + 1e-9)
    df[f"{col}_cycle_pct{window}"] = (
        df[col].rolling(window=window, min_periods=window // 2)
               .apply(percentile_rank, raw=True)
    )
    return df

def create_shifted_target(df, col, horizon):
    return yoy_pct(df[col]).shift(-horizon)


# ──────────────────────────────────────────────
# 메인 피쳐 엔지니어링
# ──────────────────────────────────────────────
def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    print("[피쳐 엔지니어링] 시작...")

    region_cols = [c for c in ["Americas","Europe","Japan","Asia_Pacific","Worldwide"] if c in df.columns]
    for col in region_cols:
        yoy_col = f"{col}_YoY"
        feat[yoy_col] = yoy_pct(df[col])
        feat = add_lag_features(feat, yoy_col, LAG_MONTHS)
        feat = add_moving_average(feat, yoy_col, MA_WINDOWS)
        feat = add_volatility(feat, yoy_col, VOL_WINDOWS)
        feat = add_momentum(feat, yoy_col)
        feat = add_acceleration(feat, yoy_col)
        feat = add_cycle_position(feat, yoy_col, window=24)
    print(f"  A. WSTS 지역 YoY% 피쳐: {len([c for c in feat.columns])}개")

    existing_eq = [c for c in EQUITY_COLS if c in df.columns]
    for col in existing_eq:
        feat[col] = df[col]
        feat = add_lag_features(feat, col, LAG_MONTHS)
        feat = add_moving_average(feat, col, [3, 6])
        feat = add_volatility(feat, col, [3, 6])
    if existing_eq:
        feat["Eq_AvgRet"] = feat[[f for f in existing_eq if f in feat.columns]].mean(axis=1)
        feat = add_lag_features(feat, "Eq_AvgRet", LAG_MONTHS)
        print(f"  B. 주가 수익률 피쳐: {len(existing_eq)}개 종목")

    existing_fred = [c for c in FRED_LEVEL_COLS if c in df.columns]
    for col in existing_fred:
        if col == "FRED_T10Y2Y":
            feat[col] = df[col]
            feat = add_lag_features(feat, col, LAG_MONTHS)
            feat[f"{col}_chg3"] = df[col].diff(3)
        else:
            yoy_col = f"{col}_YoY"
            feat[yoy_col] = yoy_pct(df[col])
            feat = add_lag_features(feat, yoy_col, LAG_MONTHS)
            feat = add_moving_average(feat, yoy_col, [3, 6])
    if existing_fred:
        print(f"  C. FRED 매크로 피쳐: {len(existing_fred)}개 시리즈")

    if "FRED_ISM_Mfg" in df.columns:
        ism = df["FRED_ISM_Mfg"]
        feat["ISM_above50"] = (ism > 50).astype(int)
        feat["ISM_mom3"]    = ism.diff(3)
        print("  D. ISM PMI 파생 피쳐 추가")

    feat["month"]     = feat.index.month
    feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12)
    feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12)
    feat = feat.drop(columns=["month"])

    for target_col in TARGETS:
        if target_col in df.columns:
            col_name = f"TARGET_{target_col}_YoY_T{TARGET_HORIZON}"
            feat[col_name] = create_shifted_target(df, target_col, TARGET_HORIZON)
            print(f"  타겟 '{col_name}' 생성 완료")

    target_cols_in_feat = [c for c in feat.columns if c.startswith("TARGET_")]
    if target_cols_in_feat:
        feat_clean = feat.dropna(subset=[target_cols_in_feat[0]])
    else:
        feat_clean = feat

    feat_clean = feat_clean.ffill().dropna(axis=1, thresh=int(len(feat_clean) * 0.5))

    total_feats = len([c for c in feat_clean.columns if not c.startswith("TARGET_")])
    print(f"\n  ▶ 최종 피쳐셋: {feat_clean.shape[0]}개 월 × {total_feats}개 피쳐 + {len(target_cols_in_feat)}개 타겟")
    print(f"     기간: {feat_clean.index.min().date()} ~ {feat_clean.index.max().date()}")
    return feat_clean


def plot_feature_overview(feat: pd.DataFrame, save_path: str):
    plot_cols = [c for c in ["Worldwide_YoY","TARGET_Worldwide_YoY_T6",
                              "Ret_SOX","FRED_ISM_Mfg_YoY","FRED_SemiProd_YoY"]
                 if c in feat.columns]
    if not plot_cols:
        return
    n    = len(plot_cols)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    colors = ["steelblue","darkorange","green","crimson","purple"]
    for ax, col, color in zip(axes, plot_cols, colors):
        s = feat[col].dropna()
        ax.plot(s.index, s.values, color=color, linewidth=1.2, label=col)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.set_ylabel(col, fontsize=8)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[0].set_title("핵심 피쳐 및 타겟(T+6) 시계열 비교", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  저장: {save_path}")


def main():
    print("=" * 60)
    print("  반도체 업황 예측 파이프라인 - Step 3: 피쳐 엔지니어링")
    print("=" * 60)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"병합 데이터 없음: {INPUT_PATH}\n먼저 data_acquisition.py를 실행하세요.")

    df   = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    print(f"[로드] {df.shape[0]}행 × {df.shape[1]}열\n")

    feat = build_feature_dataset(df)

    out_path = os.path.join(OUTPUT_DATA, "features_dataset.csv")
    feat.to_csv(out_path)
    print(f"\n  → 피쳐셋 저장: outputs/core/data/features_dataset.csv")

    plot_feature_overview(feat, os.path.join(OUTPUT_EDA, "08_feature_overview.png"))
    return feat


if __name__ == "__main__":
    main()
