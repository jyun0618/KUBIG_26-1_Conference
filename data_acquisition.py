"""
data_acquisition.py
====================
WSTS 엑셀, FRED API, yfinance를 통한 데이터 수집 및 병합 모듈.

실행 전 준비사항:
    1. FRED API 키 발급: https://fred.stlouisfed.org/docs/api/api_key.html
    2. 환경변수 설정: set FRED_API_KEY=your_key_here
       또는 아래 FRED_API_KEY 변수에 직접 입력

출력:
    conference/outputs/data/merged_dataset.csv  -- 병합된 월별 피쳐 데이터셋
    conference/outputs/data/wsts_monthly.csv     -- WSTS 월별 매출 원본 (파싱 결과)
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WSTS_PATH = os.path.join(BASE_DIR, "wsts_historical.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FRED API 키: 환경변수 우선, 없으면 직접 입력란
FRED_API_KEY = os.environ.get("FRED_API_KEY", "611878a66228a152fc523aeefc78bd67")

# 데이터 수집 기간
START_DATE = "1993-01-01"
END_DATE   = "2026-03-31"


# ──────────────────────────────────────────────
# Step 1: WSTS 엑셀 파싱
# ──────────────────────────────────────────────
def load_wsts_data(path: str) -> pd.DataFrame:
    """
    WSTS 월별 매출 데이터를 파싱하여 tidy 형태(날짜 × 지역)로 반환.

    원본 구조:
        - 헤더 3행 (메타데이터 + 컬럼명)
        - 연도행(NaN) + 5개 지역 행(Americas, Europe, Japan, Asia Pacific, Worldwide) 반복
        - 컬럼: 지역명, Jan~Dec, Total Year, Q1~Q4

    Returns:
        DataFrame with columns: [date, Americas, Europe, Japan, Asia_Pacific, Worldwide]
        단위: 천 달러(1000 USD)
    """
    print("[WSTS] 엑셀 파일 로딩 중...")
    df_raw = pd.read_excel(path, sheet_name="Monthly Data", header=None)

    # 월 컬럼 인덱스: 1~12 (January ~ December)
    month_cols = list(range(1, 13))
    month_names = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]

    records = []
    current_year = None
    regions_order = ["Americas", "Europe", "Japan", "Asia Pacific", "Worldwide"]

    for _, row in df_raw.iterrows():
        cell0 = row.iloc[0]

        # 연도 행 감지 (숫자 + 나머지 NaN)
        if isinstance(cell0, (int, float)) and not pd.isna(cell0) and 1980 <= int(cell0) <= 2030:
            current_year = int(cell0)
            continue

        # 지역 행 감지
        if isinstance(cell0, str) and cell0.strip() in regions_order:
            region = cell0.strip()
            for m_idx, m_name in zip(month_cols, month_names):
                val = row.iloc[m_idx]
                if pd.notna(val) and current_year is not None:
                    # 월 번호로 날짜 생성 (월말 기준)
                    date = pd.Timestamp(year=current_year, month=m_idx, day=1) + pd.offsets.MonthEnd(0)
                    records.append({
                        "date": date,
                        "region": region,
                        "revenue_1000usd": float(val)
                    })

    df_long = pd.DataFrame(records)
    if df_long.empty:
        raise ValueError("WSTS 데이터 파싱 실패 - 엑셀 구조를 확인하세요.")

    # 지역명 컬럼명 정리 (공백 → 언더스코어)
    df_long["region"] = df_long["region"].str.replace(" ", "_")

    # Long → Wide 피벗
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
    """
    FRED API를 통해 반도체 업황 관련 거시경제 지표 수집.

    수집 시리즈:
        IPG3344S  : 반도체 산업 생산지수 (Industrial Production: Semiconductors)
        ISM_MFG   : ISM 제조업 PMI (NAPM - 제조업 구매관리자지수)
        T10Y2Y    : 미국 장단기 금리차 (10년 - 2년)
        OECD_CLI  : OECD 경기선행지수 (미국)
        PCE_CORE  : 근원 PCE 물가지수 (경기 체감 지표)
        INDPRO    : 미국 전체 산업생산지수

    Returns:
        DataFrame with monthly frequency, index = date
    """
    if api_key == "YOUR_FRED_API_KEY_HERE":
        print("[FRED] ⚠️  API 키 미설정 - FRED 데이터 수집 건너뜀.")
        print("       FRED_API_KEY 환경변수를 설정하거나 코드에 직접 입력하세요.")
        return pd.DataFrame()

    print("[FRED] 거시경제 지표 수집 중...")
    fred = Fred(api_key=api_key)

    # 수집할 시리즈 딕셔너리: {컬럼명: FRED 시리즈 ID}
    series_map = {
        "FRED_SemiProd":   "IPG3344S",   # 반도체 생산지수 (월별)
        "FRED_T10Y2Y":     "T10Y2Y",      # 장단기 금리차
        "FRED_IndProd":    "INDPRO",      # 전체 산업생산지수
        "FRED_PCE_Core":   "PCEPILFE",    # 근원 PCE
        "FRED_MfgEmp":     "MANEMP",      # 제조업 고용자 수
        "FRED_ConsSenti":  "UMCSENT",     # 미시간대 소비자심리지수
        "FRED_NewOrder":   "NEWORDER",    # 신규 제조업 수주
    }

    dfs = {}
    for col_name, series_id in series_map.items():
        try:
            s = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
            s.index = pd.to_datetime(s.index)
            # 월말 기준으로 리샘플
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
    """
    yfinance를 통해 반도체 관련 주가 지수 및 종목 수집.

    수집 종목:
        ^SOX      : 필라델피아 반도체 지수 (SOX)
        NVDA      : NVIDIA
        TSM       : TSMC (대만 반도체 제조)
        ASML      : ASML (반도체 장비 선행 지표)
        005930.KS : 삼성전자
        000660.KS : SK하이닉스

    월별 종가(Adj Close)를 취하여 전월 대비 수익률(MoM%) 추가.

    Returns:
        DataFrame with monthly frequency, index = date
    """
    print("[yfinance] 주가 데이터 수집 중...")

    tickers = {
        "SOX":        "^SOX",
        "NVDA":       "NVDA",
        "TSM":        "TSM",
        "ASML":       "ASML",
        "Samsung":    "005930.KS",
        "SKHynix":    "000660.KS",
    }

    price_dfs = {}
    for name, ticker in tickers.items():
        try:
            raw = yf.download(ticker, start=START_DATE, end=END_DATE,
                              interval="1mo", auto_adjust=True, progress=False)
            if raw.empty:
                print(f"  ✗ {name} ({ticker}): 데이터 없음")
                continue
            # Close 컬럼 추출 (MultiIndex 처리)
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"].iloc[:, 0]
            else:
                close = raw["Close"]
            close.index = close.index.to_period("M").to_timestamp("M")  # 월말 기준
            close.name = f"Price_{name}"
            price_dfs[f"Price_{name}"] = close
            # MoM% 수익률 추가
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
    """
    대만 전자제품 수출 지표 수집.
    FRED에 대만 수출 직접 시리즈가 없을 경우,
    미국 반도체 수입(FRED: IMP9000) 또는 아시아 무역 대리 변수 활용.

    Returns:
        DataFrame or empty DataFrame
    """
    if api_key == "YOUR_FRED_API_KEY_HERE":
        return pd.DataFrame()

    print("[FRED] 대만/아시아 수출 대리 지표 수집 중...")
    fred = Fred(api_key=api_key)

    series_map = {
        "FRED_US_SemiImport": "IR9440",   # 미국 반도체 수입 (대만 주요 수출 대상)
        "FRED_US_CompEquip":  "A00000USQ363NNBR",  # 컴퓨터 장비 생산
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
def merge_all_data(
    df_wsts: pd.DataFrame,
    df_fred: pd.DataFrame,
    df_eq:   pd.DataFrame,
    df_tw:   pd.DataFrame,
) -> pd.DataFrame:
    """
    WSTS, FRED, 주가, 대만수출 데이터를 월별 날짜 기준으로 outer join 후 병합.

    인덱스 정규화:
        - 모든 데이터를 월말(Month-End) 기준으로 통일
        - 1993-01-01 이후 데이터만 사용 (WSTS 아시아태평양 데이터 안정화 시점)
    """
    print("\n[병합] 데이터 통합 중...")

    # 월말 기준 인덱스 통일 함수
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

    # WSTS를 기준으로 left join
    merged = df_wsts.copy()
    for df_ext in [df_fred, df_eq, df_tw]:
        if not df_ext.empty:
            # 중복 컬럼 방지
            overlap = set(merged.columns) & set(df_ext.columns)
            if overlap:
                df_ext = df_ext.drop(columns=list(overlap))
            merged = merged.join(df_ext, how="left")

    # 기간 필터 (1993-01 이후)
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

    # 1. WSTS 파싱
    df_wsts = load_wsts_data(WSTS_PATH)
    df_wsts.to_csv(os.path.join(OUTPUT_DIR, "wsts_monthly.csv"))
    print(f"  → WSTS 월별 데이터 저장: outputs/data/wsts_monthly.csv\n")

    # 2. FRED 수집
    df_fred = fetch_fred_data(FRED_API_KEY)

    # 3. 주가 수집
    df_eq = fetch_equity_data()

    # 4. 대만 수출 (선택)
    df_tw = fetch_taiwan_export(FRED_API_KEY)

    # 5. 전체 병합
    df_merged = merge_all_data(df_wsts, df_fred, df_eq, df_tw)
    out_path = os.path.join(OUTPUT_DIR, "merged_dataset.csv")
    df_merged.to_csv(out_path)
    print(f"\n  → 병합 데이터 저장: outputs/data/merged_dataset.csv")
    print(f"  → 최종 피쳐 목록:")
    for c in df_merged.columns:
        non_null = df_merged[c].notna().sum()
        print(f"     {c:35s}  {non_null}개 유효 관측치")

    return df_merged


if __name__ == "__main__":
    main()
