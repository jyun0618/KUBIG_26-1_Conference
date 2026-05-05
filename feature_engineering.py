"""
feature_engineering.py
=======================
6개월 선행 타겟 생성, 이동평균/변동성/Lag 피쳐 엔지니어링 모듈.

입력:
    conference/outputs/data/merged_dataset.csv

출력:
    conference/outputs/data/features_dataset.csv   -- 학습용 피쳐 + 타겟 데이터셋
    conference/outputs/eda/feature_importance_preview.png  -- 피쳐 개요 플롯

피쳐 설계 철학:
    - 타겟: T+6 YoY% (현재 시점에서 6개월 뒤의 전년 동월 대비 변화율)
    - 모든 피쳐는 현재(T) 시점까지의 정보만 사용 (데이터 누설 방지)
    - Lag 피쳐로 과거 사이클 패턴 반영
    - Bull/Bear 구간 감지를 위한 모멘텀 피쳐 포함
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
INPUT_PATH  = os.path.join(BASE_DIR, "outputs", "data", "merged_dataset.csv")
OUTPUT_DATA = os.path.join(BASE_DIR, "outputs", "data")
OUTPUT_EDA  = os.path.join(BASE_DIR, "outputs", "eda")
os.makedirs(OUTPUT_DATA, exist_ok=True)
os.makedirs(OUTPUT_EDA, exist_ok=True)

# 타겟 설정
TARGET_HORIZON = 6   # 6개월 선행 예측
TARGETS = ["Worldwide", "Asia_Pacific"]   # YoY% 타겟 원천 컬럼

# 이동평균/변동성 윈도우 설정
MA_WINDOWS  = [3, 6, 12]        # 이동평균 윈도우 (개월)
VOL_WINDOWS = [3, 6]             # 변동성(Rolling Std) 윈도우
LAG_MONTHS  = [6, 12]            # Lag 피쳐 생성 시차 (6개월 이내 제외)

# 주가 수익률 피쳐 (선행 지표 후보)
EQUITY_COLS = ["Ret_SOX", "Ret_NVDA", "Ret_TSM", "Ret_ASML", "Ret_Samsung", "Ret_SKHynix"]
# FRED 레벨 피쳐 (YoY 변환 후 사용)
FRED_LEVEL_COLS = ["FRED_SemiProd", "FRED_ISM_Mfg", "FRED_T10Y2Y", "FRED_IndProd", "FRED_PCE_Core",
                   "FRED_MfgEmp", "FRED_ConsSenti", "FRED_NewOrder"]


# ──────────────────────────────────────────────
# 헬퍼 함수들
# ──────────────────────────────────────────────
def yoy_pct(s: pd.Series) -> pd.Series:
    """전년 동월 대비 변화율(YoY%) 계산."""
    return s.pct_change(periods=12) * 100


def mom_pct(s: pd.Series) -> pd.Series:
    """전월 대비 변화율(MoM%) 계산."""
    return s.pct_change() * 100


def add_lag_features(df: pd.DataFrame, col: str, lags: list) -> pd.DataFrame:
    """
    지정 컬럼에 대해 여러 Lag 피쳐 생성.
    예: lag=3이면 3개월 전 값을 피쳐로 사용 → 과거 사이클 패턴 반영.
    """
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_moving_average(df: pd.DataFrame, col: str, windows: list) -> pd.DataFrame:
    """
    Rolling Mean 피쳐 생성.
    단기/중기/장기 이동평균으로 추세 방향 포착.
    """
    for w in windows:
        df[f"{col}_ma{w}"] = df[col].rolling(window=w, min_periods=w).mean()
    return df


def add_volatility(df: pd.DataFrame, col: str, windows: list) -> pd.DataFrame:
    """
    Rolling Std(변동성) 피쳐 생성.
    반도체 사이클의 변동성이 높아지면 전환점 근처임을 시사.
    """
    for w in windows:
        df[f"{col}_vol{w}"] = df[col].rolling(window=w, min_periods=w).std()
    return df


def add_momentum(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    모멘텀 피쳐: 3개월 변화 - 12개월 변화 (단기-장기 모멘텀 격차).
    음수 → Bull→Bear 전환 신호, 양수 → Bear→Bull 전환 신호로 활용.
    """
    df[f"{col}_momentum_3_12"] = df[col].diff(3) - df[col].diff(12)
    return df


def add_acceleration(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    가속도 피쳐: YoY%의 1차 차분 (기울기 변화).
    가속도가 양수 → 회복 / 음수 → 둔화 국면 시사.
    """
    df[f"{col}_accel"] = df[col].diff(1)
    return df


def add_cycle_position(df: pd.DataFrame, col: str, window: int = 24) -> pd.DataFrame:
    """
    사이클 위치 피쳐: 현재 값이 과거 N개월 범위 중 어느 위치(Percentile)에 있는지.
    0 근처 → 사이클 바닥, 1 근처 → 사이클 정점.
    """
    def percentile_rank(x):
        if len(x) < 3:
            return np.nan
        return (x[-1] - x.min()) / (x.max() - x.min() + 1e-9)

    df[f"{col}_cycle_pct{window}"] = (
        df[col]
        .rolling(window=window, min_periods=window // 2)
        .apply(percentile_rank, raw=True)
    )
    return df


def create_shifted_target(df: pd.DataFrame, col: str, horizon: int) -> pd.Series:
    """
    T+horizon 시점의 YoY% 타겟 생성.

    예: horizon=6이면 현재(T) 피쳐로 6개월 후 YoY%를 예측.
    shift(-6)으로 타겟을 앞당겨 정렬.

    주의: 데이터셋 끝 horizon개 행은 타겟이 NaN이 됨 → 학습 시 제거.
    """
    yoy = yoy_pct(df[col])
    return yoy.shift(-horizon)


# ──────────────────────────────────────────────
# 메인 피쳐 엔지니어링
# ──────────────────────────────────────────────
def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    전체 피쳐 엔지니어링 파이프라인 실행.

    생성 피쳐 분류:
        A. WSTS 지역별 YoY% + Lag + MA + Volatility + Momentum + Cycle Position
        B. 주가 수익률 (Ret_*) 의 Lag 피쳐
        C. FRED 매크로 지표의 YoY% + Lag 피쳐
        D. 장단기 금리차(T10Y2Y) 레벨 및 변화량
        E. 타겟: Worldwide_YoY_T+6, Asia_Pacific_YoY_T+6
    """
    feat = pd.DataFrame(index=df.index)
    print("[피쳐 엔지니어링] 시작...")

    # ────── A. WSTS 지역 YoY% 피쳐 ──────
    region_cols = [c for c in ["Americas", "Europe", "Japan", "Asia_Pacific", "Worldwide"]
                   if c in df.columns]
    for col in region_cols:
        yoy_col = f"{col}_YoY"
        feat[yoy_col] = yoy_pct(df[col])
        # Lag 피쳐
        feat = add_lag_features(feat, yoy_col, LAG_MONTHS)
        # 이동평균
        feat = add_moving_average(feat, yoy_col, MA_WINDOWS)
        # 변동성
        feat = add_volatility(feat, yoy_col, VOL_WINDOWS)
        # 모멘텀 (단기-장기 격차)
        feat = add_momentum(feat, yoy_col)
        # 가속도
        feat = add_acceleration(feat, yoy_col)
        # 사이클 위치 (24개월 Percentile Rank)
        feat = add_cycle_position(feat, yoy_col, window=24)

    print(f"  A. WSTS 지역 YoY% 피쳐: {len([c for c in feat.columns])}개")

    # ────── B. 주가 수익률 피쳐 ──────
    existing_eq = [c for c in EQUITY_COLS if c in df.columns]
    for col in existing_eq:
        feat[col] = df[col]
        feat = add_lag_features(feat, col, LAG_MONTHS)
        feat = add_moving_average(feat, col, [3, 6])
        feat = add_volatility(feat, col, [3, 6])
    if existing_eq:
        # SOX와 반도체 기업 수익률 평균 (동행 지수)
        feat["Eq_AvgRet"] = feat[[f for f in existing_eq if f in feat.columns]].mean(axis=1)
        feat = add_lag_features(feat, "Eq_AvgRet", LAG_MONTHS)
        print(f"  B. 주가 수익률 피쳐: {len(existing_eq)}개 종목")

    # ────── C. FRED 매크로 YoY% 피쳐 ──────
    existing_fred = [c for c in FRED_LEVEL_COLS if c in df.columns]
    for col in existing_fred:
        if col == "FRED_T10Y2Y":
            # 금리차는 레벨 그대로 사용 (이미 차이값)
            feat[col] = df[col]
            feat = add_lag_features(feat, col, LAG_MONTHS)
            feat[f"{col}_chg3"] = df[col].diff(3)   # 3개월 변화
        else:
            yoy_col = f"{col}_YoY"
            feat[yoy_col] = yoy_pct(df[col])
            feat = add_lag_features(feat, yoy_col, LAG_MONTHS)
            feat = add_moving_average(feat, yoy_col, [3, 6])
    if existing_fred:
        print(f"  C. FRED 매크로 피쳐: {len(existing_fred)}개 시리즈")

    # ────── D. ISM PMI 파생 피쳐 ──────
    if "FRED_ISM_Mfg" in df.columns:
        ism = df["FRED_ISM_Mfg"]
        # 50 기준 확장/수축 바이너리 피쳐
        feat["ISM_above50"] = (ism > 50).astype(int)
        # ISM 레벨 변화 모멘텀
        feat["ISM_mom3"] = ism.diff(3)
        print("  D. ISM PMI 파생 피쳐 추가")

    # ────── E. 계절성 더미 ──────
    feat["month"] = feat.index.month
    # 월별 사인/코사인 인코딩 (순환 특성 보존)
    feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12)
    feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12)
    feat = feat.drop(columns=["month"])

    # ────── F. 타겟 생성 (T+6 YoY%) ──────
    for target_col in TARGETS:
        if target_col in df.columns:
            col_name = f"TARGET_{target_col}_YoY_T{TARGET_HORIZON}"
            feat[col_name] = create_shifted_target(df, target_col, TARGET_HORIZON)
            print(f"  타겟 '{col_name}' 생성 완료")

    # ────── G. NaN 처리 ──────
    # 타겟 NaN 행 (미래 데이터 없는 끝부분) 제거
    target_cols_in_feat = [c for c in feat.columns if c.startswith("TARGET_")]
    if target_cols_in_feat:
        feat_clean = feat.dropna(subset=[target_cols_in_feat[0]])
    else:
        feat_clean = feat

    # 피쳐 NaN은 후방 채움(forward fill) 후 남은 것 제거
    feat_clean = feat_clean.ffill().dropna(axis=1, thresh=int(len(feat_clean) * 0.5))

    total_feats = len([c for c in feat_clean.columns if not c.startswith("TARGET_")])
    print(f"\n  ▶ 최종 피쳐셋: {feat_clean.shape[0]}개 월 × {total_feats}개 피쳐 + {len(target_cols_in_feat)}개 타겟")
    print(f"     기간: {feat_clean.index.min().date()} ~ {feat_clean.index.max().date()}")

    return feat_clean


# ──────────────────────────────────────────────
# 피쳐 개요 시각화
# ──────────────────────────────────────────────
def plot_feature_overview(feat: pd.DataFrame, save_path: str):
    """
    핵심 피쳐(Worldwide YoY%, 타겟 T+6, SOX 수익률, ISM) 동시 비교 플롯.
    피쳐와 타겟의 선행 관계를 시각적으로 확인.
    """
    plot_cols = []
    for cand in ["Worldwide_YoY", "TARGET_Worldwide_YoY_T6",
                 "Ret_SOX", "FRED_ISM_Mfg_YoY", "FRED_SemiProd_YoY"]:
        if cand in feat.columns:
            plot_cols.append(cand)

    if not plot_cols:
        print("  [피쳐 개요] 시각화할 피쳐 없음 - 건너뜀")
        return

    n = len(plot_cols)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = ["steelblue", "darkorange", "green", "crimson", "purple"]
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


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  반도체 업황 예측 파이프라인 - Step 3: 피쳐 엔지니어링")
    print("=" * 60)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"병합 데이터 없음: {INPUT_PATH}\n"
            "먼저 data_acquisition.py를 실행하세요."
        )

    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    print(f"[로드] {df.shape[0]}행 × {df.shape[1]}열\n")

    # 피쳐 생성
    feat = build_feature_dataset(df)

    # 저장
    out_path = os.path.join(OUTPUT_DATA, "features_dataset.csv")
    feat.to_csv(out_path)
    print(f"\n  → 피쳐셋 저장: outputs/data/features_dataset.csv")

    # 피쳐 개요 시각화
    plot_feature_overview(feat, os.path.join(OUTPUT_EDA, "08_feature_overview.png"))

    # 피쳐 목록 출력
    target_cols = [c for c in feat.columns if c.startswith("TARGET_")]
    feature_cols = [c for c in feat.columns if not c.startswith("TARGET_")]
    print(f"\n  타겟 컬럼 ({len(target_cols)}개): {target_cols}")
    print(f"  피쳐 컬럼 ({len(feature_cols)}개):")
    for c in feature_cols[:30]:
        print(f"    {c}")
    if len(feature_cols) > 30:
        print(f"    ... (총 {len(feature_cols)}개)")

    return feat


if __name__ == "__main__":
    main()
