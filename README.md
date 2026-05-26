# 반도체 업황 YoY% 6개월 선행 예측 모델

---

## 결과 요약

| 지표 | 값 |
|------|----|
| CV RMSE (5-fold) | **6.05** |
| CV Dir Acc | **93.3%** (Bull 96.0% / Bear 87.1%) |
| CV AsymLoss | **6.14** |
| Hold-out RMSE (최근 24개월) | 16.78 |
| 최종 선택 피처 수 | 30개 / 165개 |

> **AsymLoss**: Bear 국면(YoY% ≤ 0) 오예측에 ×3.0 페널티를 부여한 커스텀 손실함수.

---

## 디렉토리 구조

```
KUBIG26-1_Conference/
├── wsts_historical.xlsx        WSTS 원본 데이터
│
└── model/
    ├── config.py               공통 경로·상수·하이퍼파라미터
    ├── pipeline.py             전체 파이프라인 마스터 실행기
    ├── s1_data.py              Step 1: 데이터 수집 + 피처 엔지니어링
    ├── s2_tune.py              Step 2: Optuna RMSE 하이퍼파라미터 튜닝
    ├── s3_select.py            Step 3: 다중공선성 제거 + RFE 피처 선택
    ├── s4_optimize.py          Step 4: AsymLoss Bear 최적화
    ├── s5_evaluate.py          Step 5: 최종 평가 + 시각화
    ├── requirements.txt        
    └── outputs/                실행 후 자동 생성
        ├── data/               CSV 데이터 파일
        ├── models/             학습된 모델 pkl 파일
        ├── figures/            시각화 png 파일
        └── metrics/            평가 지표 CSV 파일
```

---

## 환경 설정

### 1. Python 버전

Python **3.10 이상**을 권장합니다.

```bash
python --version  # 3.10+
```

### 2. 패키지 설치

```bash
pip install -r model/requirements.txt
```

### 3. FRED API 키 설정

```bash
# Mac / Linux
export FRED_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:FRED_API_KEY="your_api_key_here"

# Windows (cmd)
set FRED_API_KEY=your_api_key_here
```

또는 `model/config.py`의 `FRED_API_KEY` 값을 직접 수정합니다.

```python
FRED_API_KEY = "your_api_key_here"
```

---

## 실행 방법

### 전체 파이프라인 한 번에 실행 (권장)

프로젝트 루트에서 실행합니다.

```bash
python model/pipeline.py
```

예상 소요시간: **약 30~40분** (Optuna 최적화 2회 각 50 trial 포함)

### 단계별 개별 실행

각 스텝을 순서대로 실행할 수 있습니다.

```bash
python model/s1_data.py      # Step 1: 데이터 수집 + 피처 생성     (~2분)
python model/s2_tune.py      # Step 2: RMSE 기준 Optuna 튜닝       (~10분)
python model/s3_select.py    # Step 3: 피처 선택 (RFE)             (~5분)
python model/s4_optimize.py  # Step 4: AsymLoss Bear 최적화        (~10분)
python model/s5_evaluate.py  # Step 5: 최종 평가 + 그래프 저장      (~1분)
```

> **주의**: 각 스텝은 이전 스텝의 출력 파일에 의존합니다. 반드시 순서대로 실행하세요.

---

## 파이프라인 단계별 설명

### Step 1 — 데이터 수집 + 피처 엔지니어링 (`s1_data.py`)

| 데이터 소스 | 내용 |
|------------|------|
| `wsts_historical.xlsx` | WSTS 월별 반도체 출하량 (Americas / Europe / Japan / Asia Pacific / Worldwide) |
| FRED API | 산업생산지수, 장단기 금리차, 소비자심리지수, 신규수주, 제조업 고용 등 |
| yfinance | 필라델피아 반도체지수(SOX), NVDA, TSM, ASML, 삼성전자, SK하이닉스 주가 |

생성되는 피처 카테고리:

- **YoY% 기본값** / **Lag 피처** (lag6, lag12) / **이동평균** (ma3, ma6, ma12)
- **변동성** (vol3, vol6) / **모멘텀** / **가속도** / **사이클 위치 Percentile**

타겟 변수:

```
TARGET_Worldwide_YoY_T6   = Worldwide 매출 YoY%  (shift(-6), 6개월 후 예측)
TARGET_Asia_Pacific_YoY_T6 = Asia Pacific 매출 YoY% (보조 타겟)
```

### Step 2 — Optuna 하이퍼파라미터 튜닝 (`s2_tune.py`)

- 목적함수: TimeSeriesSplit 5-fold CV **RMSE 최소화**
- Sampler: TPE / Pruner: MedianPruner
- 탐색 공간: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `min_child_weight`

### Step 3 — 피처 선택 (`s3_select.py`)

1. **다중공선성 제거**: |Pearson r| ≥ 0.9인 피처 쌍에서 하나 제거
2. **SHAP 중요도 계산**: 전체 피처 중요도 순위 산출
3. **RFE 커브**: n = 10, 15, 18, 20, 22, 25, 30, 35, 40, 50, 70개 피처로 CV AsymLoss 측정
4. **최적 피처 수 선택**: 가장 낮은 AsymLoss를 달성하는 피처 수 선택 (이번 실행: **30개**)

### Step 4 — Bear 최적화 (`s4_optimize.py`)

- 목적함수: **AsymLoss** (Bear 오예측 ×3.0 페널티) 최소화
- 선택된 30개 피처 + Bear 월 `sample_weight=2.0` 적용
- 최종 모델: `outputs/models/best_xgboost_final.pkl`

### Step 5 — 최종 평가 + 시각화 (`s5_evaluate.py`)

- Hold-out 평가 (최근 24개월: 2023-08 ~ 2025-07)
- 그래프 저장: 예측 타임라인, CV fold별 지표, Bear/Bull 개선 비교
- 지표 저장: `outputs/metrics/final_cv_metrics.csv`

---

## 주요 설정값

`model/config.py`에서 수정할 수 있습니다.

| 상수 | 기본값 | 설명 |
|------|--------|------|
| `START_DATE` | `"1993-01-01"` | 데이터 수집 시작일 |
| `END_DATE` | `"2026-03-31"` | 데이터 수집 종료일 |
| `TEST_EVAL_SIZE` | `24` | Hold-out 개월 수 |
| `N_SPLITS` | `5` | TimeSeriesSplit fold 수 |
| `TEST_SIZE` | `12` | fold당 test 개월 수 |
| `N_TRIALS` | `50` | Optuna trial 수 |
| `BEAR_SAMPLE_W` | `2.0` | Bear 월 sample_weight |
| `W_BEAR_WRONG` | `3.0` | AsymLoss Bear 오예측 페널티 |

---

## 출력 결과물

파이프라인 실행 후 `model/outputs/` 아래에 다음 파일들이 생성됩니다.

```
model/outputs/
├── data/
│   ├── merged_dataset.csv          병합 원본 데이터 (397개월 × 27피처)
│   └── features_dataset.csv        엔지니어링된 피처셋 (385개월 × 165피처 + 타겟)
│
├── models/
│   ├── best_xgboost.pkl            Step 2 RMSE 최적 파라미터
│   ├── best_xgboost_selected.pkl   Step 3 선택 피처 30개로 재학습
│   └── best_xgboost_final.pkl      Step 4 AsymLoss 최적화 최종 모델 ★
│
├── metrics/
│   ├── selected_features.csv       피처별 중요도·선택 여부
│   └── final_cv_metrics.csv        fold별 RMSE / DirAcc / AsymLoss
│
└── figures/
    ├── 01_prediction_timeline.png  Hold-out 구간 예측 vs 실제
    ├── 02_cv_metrics.png           CV fold별 성능 비교
    ├── 03_bear_improvement.png     Bear 국면 성능 개선 비교
    ├── rfe_curve.png               피처 수 vs CV AsymLoss 커브
    └── shap_summary.png            SHAP 피처 중요도 요약
```
