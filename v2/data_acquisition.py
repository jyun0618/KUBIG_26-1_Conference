"""
v2/data_acquisition.py
=======================
core/data_acquisition.py 확장판.

추가 데이터 소스:
    1. 반도체 장비주 (AMAT, LRCX, KLAC) -- SEMI Book-to-Bill Proxy
    2. 반도체 PPI (FRED: PCU334413334413, WPU1174) -- DRAM/NAND 현물가 Proxy
    3. SEMI Book-to-Bill Ratio CSV (conference/data/semi_b2b.csv 배치 시 자동 로드)

출력:
    conference/outputs/v2/data/merged_dataset.csv
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

warnings.filterwarnings("ignore")

# core/ 디렉토리를 sys.path에 추가 (core 모듈 import용)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "core"))

from data_acquisition import (
    load_wsts_data,
    fetch_fred_data,
    fetch_equity_data,
    merge_all_data,
    FRED_API_KEY,
    START_DATE,
    END_DATE,
)

# v2 전용 경로 (core의 경로 상수를 덮어씀)
BASE_DIR      = _THIS_DIR
WSTS_PATH     = os.path.join(BASE_DIR, "..", "data", "wsts_historical.xlsx")
OUTPUT_DIR    = os.path.join(BASE_DIR, "..", "outputs", "v2", "data")
SEMI_B2B_PATH = os.path.join(BASE_DIR, "..", "data", "semi_b2b.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# 신규 Step A: 반도체 장비주 (SEMI B2B Proxy)
# ──────────────────────────────────────────────
def fetch_equipment_stocks() -> pd.DataFrame:
    """
    반도체 장비주 3종 (AMAT, LRCX, KLAC) 월별 주가/수익률 수집.
    수주-출하 사이클이 반도체 매출 3~9개월 선행.
    """
    print("[yfinance v2] 반도체 장비주 수집 중 (AMAT, LRCX, KLAC)...")

    tickers = {"AMAT": "AMAT", "LRCX": "LRCX", "KLAC": "KLAC"}
    price_dfs = {}
    for name, ticker in tickers.items():
        try:
            raw = yf.download(ticker, start=START_DATE, end=END_DATE,
                              interval="1mo", auto_adjust=True, progress=False)
            if raw.empty:
                print(f"  ✗ {name} ({ticker}): 데이터 없음")
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"].iloc[:, 0]
            else:
                close = raw["Close"]
            close.index = close.index.to_period("M").to_timestamp("M")
            price_dfs[f"Price_{name}"] = close.rename(f"Price_{name}")
            ret = close.pct_change() * 100
            price_dfs[f"Ret_{name}"] = ret.rename(f"Ret_{name}")
            print(f"  ✓ {name}: {len(close)}개 월")
        except Exception as e:
            print(f"  ✗ {name} 수집 실패: {e}")

    if not price_dfs:
        return pd.DataFrame()
    df = pd.DataFrame(price_dfs)
    df.index.name = "date"
    return df


# ──────────────────────────────────────────────
# 신규 Step B: 반도체 PPI (DRAM/NAND 현물가 Proxy)
# ──────────────────────────────────────────────
def fetch_semi_ppi(api_key: str) -> pd.DataFrame:
    """
    FRED 반도체/전자부품 생산자물가지수(PPI).
    DRAM/NAND 현물가 직접 수집이 어려우므로 PPI를 Proxy로 활용.
    """
    if api_key == "YOUR_FRED_API_KEY_HERE":
        print("[FRED PPI] API 키 미설정 - 건너뜀")
        return pd.DataFrame()

    print("[FRED v2] 반도체/전자부품 PPI 수집 중...")
    fred = Fred(api_key=api_key)
    series_map = {
        "FRED_SemiPPI":     "PCU334413334413",
        "FRED_ElecCompPPI": "WPU1174",
    }
    dfs = {}
    for col_name, series_id in series_map.items():
        try:
            s = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
            s.index = pd.to_datetime(s.index)
            s = s.resample("ME").last()
            dfs[col_name] = s
            print(f"  ✓ {col_name} ({series_id}): {len(s)}개 관측치")
        except Exception as e:
            print(f"  ✗ {col_name} ({series_id}) 수집 실패 (선택 지표): {e}")

    if not dfs:
        return pd.DataFrame()
    df = pd.DataFrame(dfs)
    df.index.name = "date"
    return df


# ──────────────────────────────────────────────
# 신규 Step C: SEMI Book-to-Bill Ratio CSV 로드
# ──────────────────────────────────────────────
def load_semi_b2b(path: str) -> pd.DataFrame:
    """
    SEMI B2B Ratio CSV 로드. 없으면 장비주 Proxy로 대체.
    CSV 경로: conference/data/semi_b2b.csv
    형식: date, semi_b2b
    """
    if not os.path.exists(path):
        print(f"[SEMI B2B] {os.path.basename(path)} 없음 → 장비주 Proxy 사용")
        return pd.DataFrame()
    print(f"[SEMI B2B] CSV 로드 중: {path}")
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index) + pd.offsets.MonthEnd(0)
        df.index.name = "date"
        df.columns = [c if c.startswith("SEMI") else f"SEMI_{c}" for c in df.columns]
        print(f"  ✓ SEMI B2B: {len(df)}개 관측치")
        return df
    except Exception as e:
        print(f"  ✗ SEMI B2B 로드 실패: {e}")
        return pd.DataFrame()


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  반도체 업황 예측 파이프라인 - Step 1 v2: 데이터 수집")
    print("  [추가] 장비주(AMAT/LRCX/KLAC) + 반도체 PPI + SEMI B2B")
    print("=" * 60)

    # 기존 데이터 수집 (core 함수 재사용)
    df_wsts = load_wsts_data(WSTS_PATH)

    df_fred  = fetch_fred_data(FRED_API_KEY)
    df_eq    = fetch_equity_data()
    print()

    # 신규 데이터 수집
    df_equip = fetch_equipment_stocks()
    df_ppi   = fetch_semi_ppi(FRED_API_KEY)
    df_b2b   = load_semi_b2b(SEMI_B2B_PATH)
    print()

    # 병합
    merged = merge_all_data(df_wsts, df_fred, df_eq, pd.DataFrame())

    def to_month_end(df):
        if df.empty:
            return df
        df.index = pd.to_datetime(df.index) + pd.offsets.MonthEnd(0)
        return df

    for df_new in [df_equip, df_ppi, df_b2b]:
        if df_new.empty:
            continue
        df_new  = to_month_end(df_new)
        overlap = set(merged.columns) & set(df_new.columns)
        if overlap:
            df_new = df_new.drop(columns=list(overlap))
        merged = merged.join(df_new, how="left")

    out_path = os.path.join(OUTPUT_DIR, "merged_dataset.csv")
    merged.to_csv(out_path)
    print(f"\n  → v2 병합 데이터 저장: outputs/v2/data/merged_dataset.csv")
    print(f"  → 총 {merged.shape[1]}개 컬럼 × {merged.shape[0]}개 월")

    new_cols = [c for c in merged.columns
                if any(kw in c for kw in ["AMAT","LRCX","KLAC","SemiPPI","ElecComp","SEMI_"])]
    if new_cols:
        print(f"\n  [신규 피쳐 ({len(new_cols)}개)]")
        for c in new_cols:
            print(f"     {c:42s}  {merged[c].notna().sum()}개 유효 관측치")

    return merged


if __name__ == "__main__":
    main()
