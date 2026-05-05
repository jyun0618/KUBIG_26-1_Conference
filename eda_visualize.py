"""
eda_visualize.py
================
YoY% 변환 후 정상성 검정, 시차 교차상관, 시계열 분해 수행 모듈.

입력:
    conference/outputs/data/merged_dataset.csv   (data_acquisition.py 출력)

출력 (conference/outputs/eda/):
    01_yoy_worldwide.png        -- Worldwide YoY% 시계열 + Bull/Bear 구간 음영
    02_yoy_asia_pacific.png     -- Asia Pacific YoY% 시계열
    03_stationarity_report.txt  -- ADF / KPSS 정상성 검정 결과
    04_cross_correlation.png    -- 주요 피쳐와 타겟 YoY%의 시차 교차상관
    05_decomposition.png        -- 원본 매출 시계열 분해 (Trend/Seasonal/Residual)
    06_correlation_heatmap.png  -- YoY% 변환 피쳐 상관 히트맵
    07_yoy_all_regions.png      -- 전 지역 YoY% 비교
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # GUI 없는 환경 대응
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss, ccf
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "outputs", "data", "merged_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 분석 대상 타겟 컬럼
TARGET_COLS   = ["Worldwide", "Asia_Pacific"]
# YoY% 분석에 사용할 외부 피쳐 (존재 여부 확인 후 사용)
CANDIDATE_FEATS = [
    "FRED_SemiProd", "FRED_ISM_Mfg", "FRED_T10Y2Y", "FRED_IndProd",
    "Price_SOX", "Ret_SOX", "Ret_NVDA", "Ret_TSM",
    "Price_Samsung", "Price_SKHynix",
]


# ──────────────────────────────────────────────
# 헬퍼: YoY% 변환
# ──────────────────────────────────────────────
def compute_yoy(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """지정된 컬럼에 대해 전년 동월 대비 변화율(YoY%) 계산."""
    df_yoy = df[cols].pct_change(periods=12) * 100
    df_yoy.columns = [f"{c}_YoY" for c in cols]
    return df_yoy


# ──────────────────────────────────────────────
# 1. YoY% 시계열 플롯 (Bull/Bear 구간 음영)
# ──────────────────────────────────────────────
def plot_yoy_series(df_yoy: pd.DataFrame, col: str, title: str, save_path: str):
    """
    YoY% 시계열에 Bull(양수) / Bear(음수) 구간을 색상으로 구분하여 시각화.
    반도체 Cycle 패턴 식별에 유용.
    """
    s = df_yoy[col].dropna()
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(s.index, s.values, color="steelblue", linewidth=1.5, label=col)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    # Bull(양수) 구간: 녹색 음영
    ax.fill_between(s.index, s.values, 0,
                    where=(s.values > 0), alpha=0.25, color="green", label="Bull")
    # Bear(음수) 구간: 붉은 음영
    ax.fill_between(s.index, s.values, 0,
                    where=(s.values < 0), alpha=0.25, color="red", label="Bear")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("YoY (%)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  저장: {save_path}")


# ──────────────────────────────────────────────
# 2. 전 지역 YoY% 비교
# ──────────────────────────────────────────────
def plot_all_regions_yoy(df: pd.DataFrame, save_path: str):
    """5개 지역 YoY% 동시 비교 플롯."""
    region_cols = ["Americas", "Europe", "Japan", "Asia_Pacific", "Worldwide"]
    existing = [c for c in region_cols if c in df.columns]
    df_yoy = compute_yoy(df, existing)

    fig, axes = plt.subplots(len(existing), 1, figsize=(14, 3 * len(existing)), sharex=True)
    if len(existing) == 1:
        axes = [axes]

    colors = ["steelblue", "darkorange", "green", "crimson", "purple"]
    for ax, col, color in zip(axes, existing, colors):
        s = df_yoy[f"{col}_YoY"].dropna()
        ax.plot(s.index, s.values, color=color, linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.fill_between(s.index, s.values, 0,
                        where=(s.values > 0), alpha=0.2, color="green")
        ax.fill_between(s.index, s.values, 0,
                        where=(s.values < 0), alpha=0.2, color="red")
        ax.set_ylabel(f"{col}\nYoY%", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("WSTS 지역별 반도체 매출 YoY% 비교", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  저장: {save_path}")


# ──────────────────────────────────────────────
# 3. 정상성 검정 (ADF + KPSS)
# ──────────────────────────────────────────────
def run_stationarity_tests(series: pd.Series, name: str) -> dict:
    """
    ADF(Augmented Dickey-Fuller) + KPSS 정상성 검정.

    ADF:  귀무가설 = 단위근 있음(비정상) → p < 0.05이면 정상
    KPSS: 귀무가설 = 정상 → p < 0.05이면 비정상

    두 검정 모두 정상을 지지해야 신뢰할 수 있는 정상 시계열로 판단.
    """
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    result = {"series": name}

    # ADF 검정
    adf_out = adfuller(s, autolag="AIC")
    result["ADF_stat"]  = round(adf_out[0], 4)
    result["ADF_pval"]  = round(adf_out[1], 4)
    result["ADF_lags"]  = adf_out[2]
    result["ADF_judge"] = "정상" if adf_out[1] < 0.05 else "비정상"

    # KPSS 검정
    try:
        kpss_out = kpss(s, regression="c", nlags="auto")
        result["KPSS_stat"]  = round(kpss_out[0], 4)
        result["KPSS_pval"]  = round(kpss_out[1], 4)
        result["KPSS_judge"] = "정상" if kpss_out[1] > 0.05 else "비정상"
    except Exception:
        result["KPSS_stat"]  = np.nan
        result["KPSS_pval"]  = np.nan
        result["KPSS_judge"] = "검정 실패"

    # 종합 판단
    if result["ADF_judge"] == "정상" and result["KPSS_judge"] == "정상":
        result["Final"] = "정상 (ADF + KPSS 모두 지지)"
    elif result["ADF_judge"] == "정상":
        result["Final"] = "약한 정상 (ADF만 지지)"
    elif result["KPSS_judge"] == "정상":
        result["Final"] = "약한 비정상 (KPSS만 지지)"
    else:
        result["Final"] = "비정상 (차분 필요)"

    return result


def save_stationarity_report(df: pd.DataFrame, df_yoy: pd.DataFrame, save_path: str):
    """원본 및 YoY% 시계열의 정상성 검정 결과를 텍스트 파일로 저장."""
    lines = ["=" * 70, "  반도체 업황 예측 파이프라인 - 정상성 검정 리포트", "=" * 70, ""]

    # 원본 Worldwide 검정
    for col in ["Worldwide", "Asia_Pacific"]:
        if col in df.columns:
            r = run_stationarity_tests(df[col], f"{col}(원본)")
            lines.append(f"[{r['series']}]")
            lines.append(f"  ADF: 통계량={r['ADF_stat']}, p={r['ADF_pval']} → {r['ADF_judge']}")
            lines.append(f"  KPSS: 통계량={r['KPSS_stat']}, p={r['KPSS_pval']} → {r['KPSS_judge']}")
            lines.append(f"  ▶ 종합: {r['Final']}")
            lines.append("")

    lines.append("-" * 70)
    lines.append("  YoY% 변환 후 정상성 검정")
    lines.append("-" * 70)
    lines.append("")

    for col in df_yoy.columns:
        r = run_stationarity_tests(df_yoy[col], col)
        lines.append(f"[{r['series']}]")
        lines.append(f"  ADF: 통계량={r['ADF_stat']}, p={r['ADF_pval']} → {r['ADF_judge']}")
        lines.append(f"  KPSS: 통계량={r['KPSS_stat']}, p={r['KPSS_pval']} → {r['KPSS_judge']}")
        lines.append(f"  ▶ 종합: {r['Final']}")
        lines.append("")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  저장: {save_path}")


# ──────────────────────────────────────────────
# 4. 교차상관 분석 (Lead-Lag)
# ──────────────────────────────────────────────
def plot_cross_correlation(df_yoy: pd.DataFrame, target_col: str,
                            feat_cols: list, save_path: str, max_lag: int = 18):
    """
    타겟(Worldwide_YoY%)과 각 피쳐의 교차상관(CCF) 분석.
    lag=+k이면 피쳐가 k개월 선행함을 의미.
    반도체 사이클 선행 지표 발굴에 활용.
    """
    existing_feats = [c for c in feat_cols if c in df_yoy.columns]
    if not existing_feats or target_col not in df_yoy.columns:
        print(f"  [교차상관] 분석 가능한 피쳐 없음 - 건너뜀")
        return

    n_feats = len(existing_feats)
    fig, axes = plt.subplots(n_feats, 1,
                             figsize=(12, 3 * n_feats), sharex=True)
    if n_feats == 1:
        axes = [axes]

    target = df_yoy[target_col].dropna()
    lags = np.arange(-max_lag, max_lag + 1)

    for ax, feat in zip(axes, existing_feats):
        feat_s = df_yoy[feat].dropna()
        # 공통 인덱스 사용
        common = target.index.intersection(feat_s.index)
        if len(common) < 24:
            ax.set_title(f"{feat} → 데이터 부족")
            continue
        t_vals = target[common].values
        f_vals = feat_s[common].values

        # 교차상관 계산
        corrs = []
        for lag in lags:
            if lag >= 0:
                # 피쳐가 lag개월 선행
                corr = pd.Series(t_vals[lag:]).corr(pd.Series(f_vals[:len(t_vals)-lag]))
            else:
                # 타겟이 |lag|개월 선행
                l = abs(lag)
                corr = pd.Series(t_vals[:len(t_vals)-l]).corr(pd.Series(f_vals[l:]))
            corrs.append(corr)

        colors = ["steelblue" if c >= 0 else "firebrick" for c in corrs]
        ax.bar(lags, corrs, color=colors, alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.8)
        # 95% 신뢰구간
        ci = 1.96 / np.sqrt(len(common))
        ax.axhline(ci, color="gray", linewidth=0.8, linestyle="--")
        ax.axhline(-ci, color="gray", linewidth=0.8, linestyle="--")
        # 최대 상관 위치 표시
        best_lag = lags[np.argmax(np.abs(corrs))]
        ax.axvline(best_lag, color="orange", linewidth=1.5, linestyle="--",
                   label=f"최대상관 lag={best_lag}")
        ax.set_title(f"{feat} vs {target_col}  (최대상관 lag={best_lag}개월)", fontsize=9)
        ax.set_ylabel("Correlation")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Lag (개월, 양수 = 피쳐 선행)")
    fig.suptitle(f"피쳐 vs {target_col} 교차상관 분석 (CCF)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {save_path}")


# ──────────────────────────────────────────────
# 5. 시계열 분해 (STL-like Seasonal Decompose)
# ──────────────────────────────────────────────
def plot_decomposition(df: pd.DataFrame, col: str, save_path: str, period: int = 12):
    """
    원본 매출 시계열을 Trend / Seasonal / Residual로 분해.
    반도체 사이클의 트렌드 성분과 잔차 특성 파악에 활용.
    """
    s = df[col].dropna()
    if len(s) < period * 2:
        print(f"  [분해] {col}: 데이터 부족 - 건너뜀")
        return

    result = seasonal_decompose(s, model="multiplicative", period=period, extrapolate_trend="freq")

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    titles = ["원본 (Observed)", "트렌드 (Trend)", "계절성 (Seasonal)", "잔차 (Residual)"]
    data   = [result.observed, result.trend, result.seasonal, result.resid]

    for ax, title, d in zip(axes, titles, data):
        ax.plot(d.index, d.values, linewidth=1.2)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{col} 시계열 분해 (Multiplicative, period={period})", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  저장: {save_path}")


# ──────────────────────────────────────────────
# 6. 상관 히트맵
# ──────────────────────────────────────────────
def plot_correlation_heatmap(df_yoy: pd.DataFrame, save_path: str):
    """YoY% 피쳐 간 상관계수 히트맵."""
    corr = df_yoy.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(10, len(corr) * 0.8), max(8, len(corr) * 0.7)))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, linewidths=0.5, ax=ax,
        annot_kws={"size": 7}
    )
    ax.set_title("YoY% 피쳐 상관 히트맵", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  저장: {save_path}")


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  반도체 업황 예측 파이프라인 - Step 2: EDA & 시각화")
    print("=" * 60)

    # 데이터 로드
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"병합 데이터 없음: {INPUT_PATH}\n"
            "먼저 data_acquisition.py를 실행하세요."
        )
    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    print(f"[로드] {df.shape[0]}행 × {df.shape[1]}열\n")

    # ── YoY% 변환 ──
    region_cols = [c for c in ["Americas", "Europe", "Japan", "Asia_Pacific", "Worldwide"]
                   if c in df.columns]
    df_yoy_regions = compute_yoy(df, region_cols)

    # 외부 피쳐 YoY% (주가 수익률은 이미 MoM이므로 제외, 레벨 지표만 YoY 변환)
    level_feats = [c for c in CANDIDATE_FEATS
                   if c in df.columns and not c.startswith("Ret_")]
    df_yoy_feats = compute_yoy(df, level_feats) if level_feats else pd.DataFrame(index=df.index)

    # 수익률 피쳐(이미 변화율)는 그대로 추가
    ret_feats = [c for c in CANDIDATE_FEATS if c in df.columns and c.startswith("Ret_")]
    df_yoy_all = pd.concat([df_yoy_regions, df_yoy_feats, df[ret_feats]], axis=1)

    # ── 1. Worldwide YoY% 시계열 ──
    plot_yoy_series(
        df_yoy_all, "Worldwide_YoY",
        "WSTS Worldwide 반도체 매출 YoY%  (Bull/Bear 구간)",
        os.path.join(OUTPUT_DIR, "01_yoy_worldwide.png")
    )

    # ── 2. Asia Pacific YoY% ──
    plot_yoy_series(
        df_yoy_all, "Asia_Pacific_YoY",
        "WSTS Asia Pacific 반도체 매출 YoY%  (Bull/Bear 구간)",
        os.path.join(OUTPUT_DIR, "02_yoy_asia_pacific.png")
    )

    # ── 7. 전 지역 YoY% 비교 ──
    plot_all_regions_yoy(df, os.path.join(OUTPUT_DIR, "07_yoy_all_regions.png"))

    # ── 3. 정상성 검정 ──
    save_stationarity_report(
        df, df_yoy_all,
        os.path.join(OUTPUT_DIR, "03_stationarity_report.txt")
    )

    # ── 4. 교차상관 분석 ──
    feat_yoy_cols = [c for c in df_yoy_all.columns if c not in ["Worldwide_YoY", "Asia_Pacific_YoY"]]
    plot_cross_correlation(
        df_yoy_all, "Worldwide_YoY",
        feat_yoy_cols,
        os.path.join(OUTPUT_DIR, "04_cross_correlation.png")
    )

    # ── 5. 시계열 분해 ──
    plot_decomposition(
        df, "Worldwide",
        os.path.join(OUTPUT_DIR, "05_decomposition.png")
    )

    # ── 6. 상관 히트맵 ──
    plot_correlation_heatmap(
        df_yoy_all,
        os.path.join(OUTPUT_DIR, "06_correlation_heatmap.png")
    )

    print("\n[완료] EDA 결과가 outputs/eda/ 폴더에 저장되었습니다.")
    return df_yoy_all


if __name__ == "__main__":
    main()
