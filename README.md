# [KUBIG 26-1_Conference] 
# WSTS & FRED 데이터 기반 반도체 업황 YoY% 6개월 선행 예측 파이프라인

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **목표** | 반도체 글로벌 매출(YoY%)의 **T+6 시점(6개월 뒤)** 예측 |
| **타겟** | WSTS Worldwide YoY%, Asia Pacific YoY% |
| **데이터 범위** | 1993-01 ~ 2026-01 (월별) |
| **핵심 데이터** | WSTS 월별 매출, FRED 거시지표, 필라델피아 반도체지수(SOX), 주요 반도체 기업 주가 |

---

## 1. 파이프라인 구조

```
conference/
├── data_acquisition.py       # Step 1: 데이터 수집 & 병합
├── eda_visualize.py          # Step 2: EDA & 시각화
├── feature_engineering.py   # Step 3: 피쳐 엔지니어링
├── model_training.py         # Step 4: 모델 학습 & 벤치마크
├── hyperparameter_tuning.py  # Step 5: Optuna 하이퍼파라미터 최적화
├── PIPELINE_SUMMARY.md       # 본 문서
├── wsts_historical.xlsx      # 원본 WSTS 데이터
└── outputs/
    ├── data/
    │   ├── wsts_monthly.csv        # WSTS 파싱 결과 (월별 × 지역)
    │   ├── merged_dataset.csv      # 전체 피쳐 병합 데이터
    │   └── features_dataset.csv   # 엔지니어링된 피쳐 + 타겟
    ├── eda/
    │   ├── 01_yoy_worldwide.png
    │   ├── 02_yoy_asia_pacific.png
    │   ├── 03_stationarity_report.txt
    │   ├── 04_cross_correlation.png
    │   ├── 05_decomposition.png
    │   ├── 06_correlation_heatmap.png
    │   ├── 07_yoy_all_regions.png
    │   └── 08_feature_overview.png
    └── models/
        ├── benchmark_results.csv
        ├── predictions.csv
        ├── predictions_plot.png
        ├── benchmark_plot.png
        ├── feature_importance_xgboost.png
        ├── feature_importance_lightgbm.png
        ├── best_params_summary.csv
        ├── optuna_study_results.csv
        ├── best_ridge.pkl / best_lasso.pkl
        ├── best_xgboost.pkl / best_lightgbm.pkl
        └── optuna_plots/
```

---

## 2. 실행 순서

```bash
# FRED API 키 설정 (필수)
set FRED_API_KEY=your_api_key_here   # Windows
export FRED_API_KEY=your_api_key_here  # Mac/Linux

# Step 1: 데이터 수집
py -3 data_acquisition.py

# Step 2: EDA
py -3 eda_visualize.py

# Step 3: 피쳐 엔지니어링
py -3 feature_engineering.py

# Step 4: 모델 학습
py -3 model_training.py

# Step 5: 하이퍼파라미터 최적화
py -3 hyperparameter_tuning.py
```

> **FRED API 키 없이 실행 가능**: WSTS + yfinance 데이터만으로도 파이프라인 동작.  
> FRED 데이터 없이도 주가 기반 피쳐로 학습이 가능하며, FRED 추가 시 성능 향상 기대.

---

## 3. 데이터셋 상세

### 3-1. WSTS 월별 반도체 매출 (`wsts_historical.xlsx`)

| 항목 | 내용 |
|------|------|
| 출처 | WSTS (World Semiconductor Trade Statistics) |
| 파일 | `wsts_historical.xlsx` → 시트: `Monthly Data` |
| 단위 | 천 달러(1,000 USD) |
| 기간 | 1986 ~ 2026년 1월 |
| 지역 구분 | Americas, Europe, Japan, **Asia Pacific**, **Worldwide** |

**파싱 방식**: 원본이 연도-지역 블록 구조이므로, 연도 행과 지역 행을 감지하여 `(날짜, 지역, 매출)` long format으로 변환 후 wide pivot 처리.

### 3-2. FRED 거시경제 지표

| 컬럼명 | FRED ID | 설명 |
|--------|---------|------|
| `FRED_SemiProd` | `IPG3344S` | 반도체 산업 생산지수 (월별) |
| `FRED_T10Y2Y` | `T10Y2Y` | 미국 장단기 금리차 (10년-2년) |
| `FRED_IndProd` | `INDPRO` | 미국 전체 산업생산지수 |
| `FRED_PCE_Core` | `PCEPILFE` | 근원 PCE 물가지수 |
| `FRED_MfgEmp` | `MANEMP` | 제조업 고용자 수 (제조업 경기 동행) |
| `FRED_ConsSenti` | `UMCSENT` | 미시간대 소비자심리지수 (소비 수요 선행) |
| `FRED_NewOrder` | `NEWORDER` | 신규 제조업 수주 (수요 선행) |

> `NAPM`(ISM PMI) 시리즈는 FRED에서 폐기되어 수집 불가. 대신 `MANEMP`(제조업 고용), `UMCSENT`(소비자심리), `NEWORDER`(신규수주)로 유사 정보를 커버.

### 3-3. 주가 지수 및 종목 (yfinance)

| 컬럼 | Ticker | 설명 |
|------|--------|------|
| `Price_SOX` / `Ret_SOX` | `^SOX` | 필라델피아 반도체 지수 |
| `Price_NVDA` / `Ret_NVDA` | `NVDA` | NVIDIA |
| `Price_TSM` / `Ret_TSM` | `TSM` | TSMC |
| `Price_ASML` / `Ret_ASML` | `ASML` | ASML (장비 선행 지표) |
| `Price_Samsung` / `Ret_Samsung` | `005930.KS` | 삼성전자 |
| `Price_SKHynix` / `Ret_SKHynix` | `000660.KS` | SK하이닉스 |

---

## 4. 피쳐 엔지니어링 상세

### 4-1. 타겟 변수

```
TARGET_Worldwide_YoY_T6   = Worldwide 매출 YoY% (T+6 shift)
TARGET_Asia_Pacific_YoY_T6 = Asia Pacific 매출 YoY% (T+6 shift)
```

- `shift(-6)` 적용: 현재(T) 시점 피쳐로 6개월 후 YoY%를 예측
- 타겟이 NaN인 마지막 6행은 학습에서 제외

### 4-2. 피쳐 카테고리

| 카테고리 | 피쳐 예시 | 의미 |
|----------|-----------|------|
| **YoY% 기본** | `Worldwide_YoY`, `Asia_Pacific_YoY` | 현재 시점 업황 |
| **Lag 피쳐** | `Worldwide_YoY_lag6`, `Ret_SOX_lag12` | 과거 사이클 패턴 (lag6, lag12만 사용) |
| **이동평균** | `Worldwide_YoY_ma6`, `ma12` | 중기/장기 추세 방향 |
| **변동성** | `Worldwide_YoY_vol3` | 사이클 전환 전 변동성 급등 포착 |
| **모멘텀** | `Worldwide_YoY_momentum_3_12` | 단기-장기 추세 괴리 (전환점 신호) |
| **가속도** | `Worldwide_YoY_accel` | YoY% 기울기 변화 |
| **사이클 위치** | `Worldwide_YoY_cycle_pct24` | 24개월 내 Percentile Rank (0=바닥, 1=정점) |
| **계절성** | `month_sin`, `month_cos` | 월별 계절성 순환 인코딩 |

**총 피쳐 수**: 136개 (385개 월 × 136 피쳐)

> **lag1~lag5 제외 이유**: 예측 지평(T+6)과 너무 가까운 단기 lag는 타겟과 거의 동일한 정보를 포함해 과적합을 유발하고, lag1~lag3는 이동평균(ma3)과 상관 0.99 수준의 중복 피쳐를 형성함. lag6, lag12만 사용해 실질적 선행 정보만 보존.

### 4-3. 데이터 누설 방지

- 모든 Lag/MA/Volatility 피쳐는 `T` 시점까지의 과거 데이터만 사용
- TimeSeriesSplit으로 학습셋이 항상 테스트셋보다 과거 데이터로 구성

---

## 5. 모델 설명

### 5-1. 모델 라인업 및 선택 이유

| 모델 | 특징 | 반도체 사이클 적합성 |
|------|------|---------------------|
| **Ridge** | L2 정규화 선형 회귀 | 다중공선성 높은 피쳐 간 안정적 계수 추정 |
| **Lasso** | L1 정규화, 자동 피쳐 선택 | 수백 개 피쳐 중 핵심 선행 지표 자동 선별 |
| **XGBoost** | 그래디언트 부스팅 트리 | 비선형 사이클 전환 패턴, 임계 효과 포착 |
| **LightGBM** | 경량 그래디언트 부스팅 | 빠른 학습, leaf-wise 분기로 세밀한 패턴 |
| **N-HiTS** | 딥러닝 시계열 (선택적) | 다중 해상도 분해, 장기 의존성 학습 |

### 5-2. 평가 지표

| 지표 | 설명 | 특이사항 |
|------|------|----------|
| **RMSE** | 예측 오차의 제곱 평균 제곱근 | 이상치(급락/급등)에 민감 |
| **MAE** | 절대 오차 평균 | 이상치에 강건 |
| **MAPE(%)** | 절대 퍼센트 오차 평균 | YoY% 0 근처에서 불안정 (ε 보정 적용) |
| **Direction Accuracy** | Bull/Bear 방향 분류 정확도 | **투자 관점 핵심 지표** |
| **Asymmetric Loss (1.5×)** | Bear 국면 오차에 1.5배 가중치 | 하락 사이클 미예측 페널티 강화 |

### 5-3. 현재 파이프라인 성능 (Optuna 최적화 후)

데이터: 1993-07 ~ 2025-07 (385개월), 피쳐 136개, TimeSeriesSplit 5-fold CV

| 모델 | CV RMSE | 방향 정확도 |
|------|---------|------------|
| **XGBoost** | **9.51** | 95.0% |
| **LightGBM** | **9.96** | 91.7% |
| Ridge | 12.74 | — |
| Lasso | 12.81 | — |

> 선형 모델(Ridge/Lasso)은 피쳐 간 다중공선성(이동평균-lag 중복, 지역간 상관) 영향으로 트리 모델 대비 성능이 낮음.

### 5-4. 비대칭 손실 함수 제안

반도체 산업의 Bull/Bear 특성상, **Bear 국면(YoY < 0) 예측 실패가 더 큰 손해**를 초래합니다.

```python
# Asymmetric Weighted MSE (Bear 페널티 1.5배)
weight = 1.5 if y_true < 0 else 1.0
loss = weight × (y_true - y_pred)²
```

XGBoost/LightGBM에서는 커스텀 Objective로 구현 가능:
```python
def asymmetric_mse_obj(y_pred, dtrain):
    y_true = dtrain.get_label()
    weights = np.where(y_true < 0, 1.5, 1.0)
    grad = -2 * weights * (y_true - y_pred)
    hess = 2 * weights
    return grad, hess
```

---

## 6. 하이퍼파라미터 최적화 (Optuna)

| 항목 | 설정 |
|------|------|
| Sampler | TPE (Tree-structured Parzen Estimator) |
| Pruner | MedianPruner (n_startup=10, warmup=5) |
| Trials | 50회 (모델당) |
| 목적함수 | 시계열 CV 평균 RMSE 최소화 |
| CV 설정 | TimeSeriesSplit(n_splits=5, test_size=12) |

---

## 7. 시계열 교차검증 설계

```
전체 시계열 (약 390개월)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fold 1: [━━━━Train━━━━][━Test━]
Fold 2: [━━━━━━Train━━━━━━][━Test━]
Fold 3: [━━━━━━━━Train━━━━━━━━][━Test━]
Fold 4: [━━━━━━━━━━Train━━━━━━━━━━][━Test━]
Fold 5: [━━━━━━━━━━━━Train━━━━━━━━━━━━][━Test━]
                                           ↑ 12개월
```

- 훈련셋은 항상 테스트셋보다 **과거** 데이터만 포함
- 최소 학습 기간 60개월 보장 (반도체 1~2사이클 포함)
- 미래 데이터 누설 완전 차단

---

## 8. 패키지 의존성

```
pandas >= 2.0
numpy
matplotlib
seaborn
statsmodels       # ADF/KPSS, Seasonal Decompose
scikit-learn      # Ridge, Lasso, TimeSeriesSplit, Pipeline
xgboost >= 1.7
lightgbm >= 4.0
optuna >= 3.0
yfinance >= 0.2
fredapi >= 0.5
openpyxl          # WSTS 엑셀 파싱
neuralforecast    # N-HiTS (선택)
```

설치:
```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn xgboost lightgbm optuna yfinance fredapi openpyxl neuralforecast
```

---

## 9. 주요 가설 및 기대 선행 지표

| 선행 지표 | 예상 Lead (개월) | 근거 |
|-----------|-----------------|------|
| 필라델피아 반도체지수 (SOX) | 3~9개월 | 주식시장의 경기 선행성 |
| 신규 제조업 수주 (NEWORDER) | 3~6개월 | 전방 산업 수요 선행 (ISM PMI 대체) |
| 소비자심리지수 (UMCSENT) | 3~6개월 | 소비 전자 수요 선행 |
| 장단기 금리차 | 6~12개월 | 경기 사이클 선행 지표 |
| ASML 주가 | 6~12개월 | 반도체 설비 투자 선행 |
| 아시아 반도체 기업 주가 | 3~6개월 | 재고 사이클 선행 |

---

## 10. 확장 방향

1. **재고 사이클 피쳐 추가**: DRAM/NAND 현물가격, 반도체 재고/출하 비율
2. **Transformer 기반 모델**: PatchTST, iTransformer 등 최신 시계열 모델
3. **앙상블**: 선형 + 트리 + 딥러닝 모델 가중 앙상블
4. **실시간 업데이트**: 월별 WSTS 발표 후 자동 재학습 파이프라인
5. **설명 가능 AI**: SHAP으로 예측의 피쳐 기여도 시각화
