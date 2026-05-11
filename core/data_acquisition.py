"""
data_acquisition.py
====================
WSTS 엑셀, FRED API, yfinance를 통한 데이터 수집 및 병합 모듈.

실행 전 준비사항:
    1. FRED API 키 발급: https://fred.stlouisfed.org/docs/api/api_key.html
    2. 환경변수 설정: set FRED_API_KEY=your_key_here
       또는 아래 FRED_API_KEY 변수에 직접 입력

출력:
    conference/outputs/core/data/merged_dataset.csv
    conference/outputs/core/data/wsts_monthly.csv
"""

import os
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
WSTS_PATH  = os.path.join(BASE_DIR, "..", "data", "wsts_historical.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs", "core", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRED_API_KEY = os.environ.get("FRED_API_KEY", "611878a66228a152fc523aeefc78bd67")

START_DATE = "1993-01-01"
END_DATE   = "2026-03-31"


# ──────────────────────────────────────────────
# Step 1: WSTS 엑셀 파싱
# ──────────────────────────────────────────────
def load_wsts_data(path: str) -> pd.DataFrame:
    print("[WSTS] 엑셀 파일 로딩 중...")
    df_raw = pd.read_excel(path, sheet_name="Monthly Data", header=None)

    month_cols  = list(range(1, 13))
    month_names = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]

    records = []
    current_year = None
    regions_order = ["Americas", "Europe", "Japan", "Asia Pacific", "Worldwide"]

    for _, row in df_raw.iterrows():
        cell0 = row.iloc[0]
        if isinstance(cell0, (int, float)) and not pd.isna(cell0) and 1980 <= int(cell0) <= 2030:
            current_year = int(cell0)
            continue
        if isinstance(cell0, str) and cell0.strip() in regions_order:
            region = cell0.strip()
            for m_idx, m_name in zip(month_cols, month_names):
                val = row.iloc[m_idx]
                if pd.notna(val) and current_year is not None:
                    date = pd.Timestamp(year=current_year, month=m_idx, day=1) + pd.offsets.MonthEnd(0)
                    records.append({"date": date, "region": region, "revenue_1000usd": float(val)})

    df_long = pd.DataFrame(records)
    if df_long.empty:
        raise ValueError("WSTS 데이터 파싱 실패 - 엑셀 구조를 확인하세요.")

    df_long["region"] = df_long["region"].str.replace(" ", "_")
    df_wide = df_long.pivot_table(index="date", columns="region", values="revenue_1000usd")
    df_wide.columns.name = None
    df_wide = df_wide.sort_index()

    print(f"[WSTS] 파싱 완료: {df_wide.shape[0]}개 월 × {df_wide.shape[1]}개 지역")
    print(f"       기간: {df_wide.index.min().date()} ~ {df_wide.index.max().date()}")
    return df_wide


# ──────────────────────────────────────────────
# Step 2: FRED 데이터 수집
# ──────────────────────────────────────────────
def fetch_fred_data(api_key: str) -> pd.DataFrame:
    if api_key == "YOUR_FRED_API_KEY_HERE":
        print("[FRED] ⚠️  API 키 미설정 - FRED 데이터 수집 건너뜀.")
        return pd.DataFrame()

    print("[FRED] 거시경제 지표 수집 중...")
    fred = Fred(api_key=api_key)

    series_map = {
        "FRED_SemiProd":  "IPG3344S",
        "FRED_T10Y2Y":    "T10Y2Y",
        "FRED_IndProd":   "INDPRO",
        "FRED_PCE_Core":  "PCEPILFE",
        "FRED_MfgEmp":    "MANEMP",
        "FRED_ConsSenti": "UMCSENT",
        "FRED_NewOrder":  "NEWORDER",
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
            print(f"  ✗ {col_name} ({series_id}) 수집 실패: {e}")

    if not dfs:
        return pd.DataFrame()

    df_fred = pd.DataFrame(dfs)
    df_fred.index.name = "date"
    return df_fred


# ──────────────────────────────────────────────
# Step 3: yfinance 주가 데이터 수집
# ──────────────────────────────────────────────
def fetch_equity_data() -> pd.DataFrame:
    print("[yfinance] 주가 데이터 수집 중...")

    tickers = {
        "SOX":     "^SOX",
        "NVDA":    "NVDA",
        "TSM":     "TSM",
        "ASML":    "ASML",
        "Samsung": "005930.KS",
        "SKHynix": "000660.KS",
    }

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
            close.name = f"Price_{name}"
            price_dfs[f"Price_{name}"] = close
            ret = close.pct_change() * 100
            ret.name = f"Ret_{name}"
            price_dfs[f"Ret_{name}"] = ret
            print(f"  ✓ {name} ({ticker}): {len(close)}개 월")
        except Exception as e:
            print(f"  ✗ {name} ({ticker}) 수집 실패: {e}")

    if not price_dfs:
        return pd.DataFrame()

    df_eq = pd.DataFrame(price_dfs)
    df_eq.index.name = "date"
    return df_eq


# ──────────────────────────────────────────────
# Step 4: 대만 전자수출 (FRED 보조 지표)
# ──────────────────────────────────────────────
def fetch_taiwan_export(api_key: str) -> pd.DataFrame:
    if api_key == "YOUR_FRED_API_KEY_HERE":
        return pd.DataFrame()

    print("[FRED] 대만/아시아 수출 대리 지표 수집 중...")
    fred = Fred(api_key=api_key)

    series_map = {
        "FRED_US_SemiImport": "IR9440",
        "FRED_US_CompEquip":  "A00000USQ363NNBR",
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
# Step 5: 전체 병합
# ──────────────────────────────────────────────
def merge_all_data(df_wsts, df_fred, df_eq, df_tw) -> pd.DataFrame:
    print("\n[병합] 데이터 통합 중...")

    def to_month_end(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df.index = pd.to_datetime(df.index)
        df.index = df.index + pd.offsets.MonthEnd(0)
        return df

    df_wsts = to_month_end(df_wsts)
    df_fred = to_month_end(df_fred)
    df_eq   = to_month_end(df_eq)
    df_tw   = to_month_end(df_tw)

    merged = df_wsts.copy()
    for df_ext in [df_fred, df_eq, df_tw]:
        if not df_ext.empty:
            overlap = set(merged.columns) & set(df_ext.columns)
            if overlap:
                df_ext = df_ext.drop(columns=list(overlap))
            merged = merged.join(df_ext, how="left")

    merged = merged[merged.index >= pd.Timestamp(START_DATE)]
    merged = merged.sort_index()

    print(f"[병합] 완료: {merged.shape[0]}개 월 × {merged.shape[1]}개 피쳐")
    print(f"       기간: {merged.index.min().date()} ~ {merged.index.max().date()}")
    return merged


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  반도체 업황 예측 파이프라인 - Step 1: 데이터 수집")
    print("=" * 60)

    df_wsts = load_wsts_data(WSTS_PATH)
    df_wsts.to_csv(os.path.join(OUTPUT_DIR, "wsts_monthly.csv"))

    df_fred = fetch_fred_data(FRED_API_KEY)
    df_eq   = fetch_equity_data()
    df_tw   = fetch_taiwan_export(FRED_API_KEY)

    df_merged = merge_all_data(df_wsts, df_fred, df_eq, df_tw)
    out_path  = os.path.join(OUTPUT_DIR, "merged_dataset.csv")
    df_merged.to_csv(out_path)
    print(f"\n  → 병합 데이터 저장: outputs/core/data/merged_dataset.csv")

    return df_merged


if __name__ == "__main__":
    main()
