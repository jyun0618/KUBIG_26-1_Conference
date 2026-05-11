"""
v2/feature_engineering.py
==========================
core/feature_engineering.py 확장판.

추가 피쳐:
    1. 반도체 장비주 (AMAT, LRCX, KLAC) 수익률 + Lag/MA/Vol
    2. 장비주 합성 B2B Proxy 인덱스 (3종 평균 + 모멘텀)
    3. 반도체 PPI YoY% + Lag (DRAM 현물가 Proxy)
    4. SEMI B2B Ratio 레벨 + 변화량 (데이터 있을 경우)

입력:
    conference/outputs/v2/data/merged_dataset.csv

출력:
    conference/outputs/v2/data/features_dataset.csv
    conference/outputs/v2/eda/09_new_features.png
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "core"))

from feature_engineering import (
    yoy_pct,
    add_lag_features,
    add_moving_average,
    add_volatility,
    add_momentum,
    add_acceleration,
    add_cycle_position,
    create_shifted_target,
    build_feature_dataset,
    TARGET_HORIZON,
    TARGETS,
    MA_WINDOWS,
    VOL_WINDOWS,
    LAG_MONTHS,
)

# v2 전용 경로
BASE_DIR    = _THIS_DIR
INPUT_PATH  = os.path.join(BASE_DIR, "..", "outputs", "v2", "data", "merged_dataset.csv")
OUTPUT_DATA = os.path.join(BASE_DIR, "..", "outputs", "v2", "data")
OUTPUT_EDA  = os.path.join(BASE_DIR, "..", "outputs", "v2", "eda")
os.makedirs(OUTPUT_DATA, exist_ok=True)
os.makedirs(OUTPUT_EDA,  exist_ok=True)

EQUIPMENT_COLS = ["Ret_AMAT", "Ret_LRCX", "Ret_KLAC"]
SEMI_PPI_COLS  = ["FRED_SemiPPI", "FRED_ElecCompPPI"]


# ──────────────────────────────────────────────
# 신규 피쳐 블록
# ──────────────────────────────────────────────
def add_equipment_features(feat: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    반도체 장비주 3종 피쳐 블록.
    장비주 수익률은 SEMI B2B ratio와 높은 상관, 매출 3~9개월 선행.
    """
    existing = [c for c in EQUIPMENT_COLS if c in df.columns]
    if not existing:
        print("  [장비주] 데이터 없음 (v2/data_acquisition.py 먼저 실행)")
        return feat

    for col in existing:
        feat[col] = df[col]
        feat = add_lag_features(feat, col, LAG_MONTHS)
        feat = add_moving_average(feat, col, [3, 6])
        feat = add_volatility(feat, col, [3, 6])

    feat["Equip_B2B_Proxy"] = feat[existing].mean(axis=1)
    feat = add_lag_features(feat, "Equip_B2B_Proxy", LAG_MONTHS)
    feat = add_momentum(feat, "Equip_B2B_Proxy")
    feat = add_moving_average(feat, "Equip_B2B_Proxy", [3, 6])

    print(f"  [장비주] {len(existing)}종 피쳐 + B2B Proxy 인덱스 추가")
    return feat


def add_semi_ppi_features(feat: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """반도체/전자부품 PPI YoY% 피쳐 블록. DRAM 현물가 Proxy."""
    existing = [c for c in SEMI_PPI_COLS if c in df.columns]
    if not existing:
        print("  [반도체 PPI] 데이터 없음 (FRED API 키 확인)")
        return feat

    for col in existing:
        yoy_col = f"{col}_YoY"
        feat[yoy_col] = yoy_pct(df[col])
        feat = add_lag_features(feat, yoy_col, LAG_MONTHS)
        feat = add_moving_average(feat, yoy_col, [3, 6])
        feat = add_acceleration(feat, yoy_col)

    print(f"  [반도체 PPI] {len(existing)}개 시리즈 YoY% 피쳐 추가")
    return feat


def add_semi_b2b_features(feat: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """SEMI Book-to-Bill Ratio 피쳐 블록 (CSV 배치 시 활성화)."""
    b2b_cols = [c for c in df.columns if c.startswith("SEMI_") and "b2b" in c.lower()]
    if not b2b_cols:
        return feat

    for col in b2b_cols:
        feat[col]              = df[col]
        feat[f"{col}_chg3"]    = df[col].diff(3)
        feat[f"{col}_above1"]  = (df[col] > 1.0).astype(float)
        feat = add_lag_features(feat, col, LAG_MONTHS)

    print(f"  [SEMI B2B] {len(b2b_cols)}개 컬럼 피쳐 추가")
    return feat


# ──────────────────────────────────────────────
# 메인 피쳐 엔지니어링 (v2)
# ──────────────────────────────────────────────
def build_feature_dataset_v2(df: pd.DataFrame) -> pd.DataFrame:
    """기존 build_feature_dataset() 이후 신규 피쳐 블록을 추가."""
    print("[피쳐 엔지니어링 v2] 기존 피쳐 생성 중...")
    feat = build_feature_dataset(df)

    target_cols = [c for c in feat.columns if c.startswith("TARGET_")]
    targets_df  = feat[target_cols].copy()
    feat_base   = feat.drop(columns=target_cols)

    print("\n[피쳐 엔지니어링 v2] 신규 피쳐 추가 중...")
    new_feat = pd.DataFrame(index=df.index)
    new_feat = add_equipment_features(new_feat, df)
    new_feat = add_semi_ppi_features(new_feat, df)
    new_feat = add_semi_b2b_features(new_feat, df)

    feat_all = pd.concat([feat_base, new_feat, targets_df], axis=1)

    if target_cols:
        feat_all = feat_all.dropna(subset=[target_cols[0]])

    feat_all = feat_all.ffill().dropna(axis=1, thresh=int(len(feat_all) * 0.5))

    n_feat   = len([c for c in feat_all.columns if not c.startswith("TARGET_")])
    n_target = len([c for c in feat_all.columns if c.startswith("TARGET_")])
    print(f"\n  ▶ 최종 피쳐셋 v2: {feat_all.shape[0]}개 월 × {n_feat}개 피쳐 + {n_target}개 타겟")
    print(f"     기간: {feat_all.index.min().date()} ~ {feat_all.index.max().date()}")
    print(f"     기존 대비 신규 피쳐: +{n_feat - (feat.shape[1] - len(target_cols))}개")
    return feat_all


def plot_new_feature_overview(feat: pd.DataFrame, save_path: str):
    cands     = ["Equip_B2B_Proxy","Ret_AMAT","FRED_SemiPPI_YoY","SEMI_semi_b2b","TARGET_Worldwide_YoY_T6"]
    plot_cols = [c for c in cands if c in feat.columns]
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
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_ylabel(col, fontsize=8)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[0].set_title("v2 신규 피쳐 및 타겟(T+6) 비교", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  저장: {save_path}")


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  반도체 업황 예측 파이프라인 - Step 3 v2: 피쳐 엔지니어링")
    print("  [추가] 장비주 B2B Proxy + 반도체 PPI + SEMI B2B Ratio")
    print("=" * 60)

    if not os.path.exists(INPUT_PATH):
        fallback = os.path.join(BASE_DIR, "..", "outputs", "core", "data", "merged_dataset.csv")
        if os.path.exists(fallback):
            print(f"[주의] v2 병합 데이터 없음 → core 데이터 사용\n"
                  "       v2/data_acquisition.py 실행 시 신규 피쳐 활성화\n")
            path = fallback
        else:
            raise FileNotFoundError("병합 데이터 없음. v2/data_acquisition.py를 먼저 실행하세요.")
    else:
        path = INPUT_PATH

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"[로드] {df.shape[0]}행 × {df.shape[1]}열\n")

    feat = build_feature_dataset_v2(df)

    out_path = os.path.join(OUTPUT_DATA, "features_dataset.csv")
    feat.to_csv(out_path)
    print(f"\n  → v2 피쳐셋 저장: outputs/v2/data/features_dataset.csv")

    plot_new_feature_overview(feat, os.path.join(OUTPUT_EDA, "09_new_features.png"))
    return feat


if __name__ == "__main__":
    main()
