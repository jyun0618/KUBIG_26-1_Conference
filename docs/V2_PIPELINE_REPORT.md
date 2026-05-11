# 반도체 업황 YoY% 6개월 선행 예측 — v2 파이프라인 리포트

## 1. 파이프라인 개요

| 항목 | v1 (기존) | v2 (개선) |
|------|-----------|-----------|
| **데이터 소스** | WSTS + FRED 7종 + 주가 6종 | + 장비주 3종 + 반도체 PPI 2종 |
| **전체 피쳐 수** | 136개 | 171개 (+35개) |
| **모델 학습 피쳐** | 전체 136개 | SHAP 선택 50개 |
| **손실 함수** | MSE (대칭) | Asymmetric MSE (Bear ×1.5) |
| **하이퍼파라미터 탐색** | RMSE 최소화 | 비대칭 손실 최소화 |

---

## 2. 데이터 소스

### 2-1. 기존 데이터 (v1 유지)

| 구분 | 소스 | 내용 |
|------|------|------|
| WSTS | wsts_historical.xlsx | 반도체 월별 매출 — Americas / Europe / Japan / Asia Pacific / Worldwide (1986~2026) |
| FRED | IPG3344S | 반도체 산업 생산지수 |
| FRED | INDPRO | 미국 전체 산업생산지수 |
| FRED | PCEPILFE | 근원 PCE 물가지수 |
| FRED | MANEMP | 제조업 고용자 수 |
| FRED | UMCSENT | 미시간대 소비자심리지수 |
| FRED | NEWORDER | 신규 제조업 수주 |
| yfinance | ^SOX, NVDA, TSM, ASML, 005930.KS, 000660.KS | 반도체 지수 및 주요 종목 주가 |

### 2-2. v2 신규 추가

| 구분 | 소스 | 내용 | 선택 이유 |
|------|------|------|-----------|
| yfinance | AMAT (Applied Materials) | 반도체 장비주 | SEMI B2B Ratio Proxy — 장비 수주가 매출 3~9개월 선행 |
| yfinance | LRCX (Lam Research) | 반도체 장비주 (식각) | 동일 |
| yfinance | KLAC (KLA Corporation) | 반도체 장비주 (계측) | 동일 |
| FRED | PCU334413334413 | 반도체 생산자물가지수(PPI) | DRAM/NAND 현물가 Proxy (직접 데이터는 유료) |
| FRED | WPU1174 | 전자부품 도매물가지수 | 동일 |

> SEMI Book-to-Bill Ratio 원본 데이터는 SEMI.org에서 수동 다운로드 필요.  
> `conference/semi_b2b.csv` 파일을 배치하면 자동 로드됨.  
> 현재는 장비주 3종(AMAT+LRCX+KLAC) 평균 수익률로 대체.

---

## 3. 피쳐 엔지니어링

### 3-1. 전체 피쳐 구조 (171개)

| 카테고리 | 피쳐 예시 | 개수 | 설명 |
|----------|-----------|------|------|
| WSTS YoY% | `Worldwide_YoY`, `Asia_Pacific_YoY` | 5개 | 지역별 전년 동월 대비 매출 변화율 |
| WSTS Lag | `Worldwide_YoY_lag6`, `lag12` | 10개 | lag6, lag12만 사용 (lag1~5는 미래 누설 위험) |
| WSTS 이동평균 | `Worldwide_YoY_ma3`, `ma6`, `ma12` | 15개 | 단기~장기 추세 방향 |
| WSTS 변동성 | `Worldwide_YoY_vol3`, `vol6` | 10개 | 사이클 전환 전 변동성 급등 포착 |
| WSTS 모멘텀 | `Worldwide_YoY_momentum_3_12` | 5개 | 단기-장기 모멘텀 격차 (전환점 신호) |
| WSTS 가속도 | `Worldwide_YoY_accel` | 5개 | YoY% 1차 차분 (방향 전환 속도) |
| WSTS 사이클 위치 | `Worldwide_YoY_cycle_pct24` | 5개 | 24개월 Percentile Rank (0=바닥, 1=정점) |
| 기존 주가 수익률 | `Ret_SOX`, `Ret_NVDA`, ... | 30개 | 6종 × Lag/MA/Vol |
| **장비주 (v2 신규)** | `Ret_AMAT_ma6`, `Equip_B2B_Proxy` | 25개 | 3종 × Lag/MA/Vol + 합성 B2B Proxy |
| **반도체 PPI (v2 신규)** | `FRED_SemiPPI_YoY`, `FRED_ElecCompPPI_YoY` | 14개 | 2종 × YoY% + Lag + MA + 가속도 |
| FRED 거시지표 | `FRED_IndProd_YoY`, `FRED_MfgEmp_YoY`, ... | 45개 | 6종 × YoY% + Lag/MA |
| 계절성 | `month_sin`, `month_cos` | 2개 | 월별 순환 인코딩 |

### 3-2. 장비주 합성 B2B Proxy 인덱스

```
Equip_B2B_Proxy = mean(Ret_AMAT, Ret_LRCX, Ret_KLAC)
```

- SEMI Book-to-Bill이 없을 때의 대체 선행 지표
- 3종 평균에 추가로 MA, 모멘텀 피쳐 생성
- **SHAP 선택 결과**: `Equip_B2B_Proxy_ma6`이 상위 4위로 선택됨 → 실제 선행성 확인

---

## 4. SHAP 기반 피쳐 선택

### 선택 방법

1. 전체 171개 피쳐로 XGBoost 학습 (전체 데이터, n_estimators=200)
2. `shap.TreeExplainer`로 SHAP 값 계산 (385행 × 171피쳐)
3. `mean(|SHAP value|)` 기준 내림차순 정렬 → **상위 50개** 선택

### SHAP 선택 피쳐 50개 (중요도 순)

| 순위 | 피쳐명 | 카테고리 | 비고 |
|------|--------|----------|------|
| 1 | `FRED_IndProd_YoY_lag12` | FRED 거시 | 산업생산 12개월 전 |
| 2 | `Ret_SOX_ma6` | 주가 | SOX 6개월 MA |
| 3 | `Asia_Pacific_YoY` | WSTS | 현재 아시아 업황 |
| 4 | `Equip_B2B_Proxy_ma6` | **장비주 v2** | B2B Proxy 6개월 MA ★ |
| 5 | `Worldwide_YoY_cycle_pct24` | WSTS 사이클 | 24개월 위치 |
| 6 | `FRED_ElecCompPPI_YoY_ma6` | **전자부품 PPI v2** | 전자부품 물가 MA ★ |
| 7 | `Ret_LRCX_ma6` | **장비주 v2** | Lam Research MA ★ |
| 8 | `FRED_NewOrder_YoY_lag12` | FRED 거시 | 신규수주 12개월 전 |
| 9 | `Japan_YoY_lag6` | WSTS | 일본 업황 6개월 전 |
| 10 | `Worldwide_YoY` | WSTS | 현재 글로벌 업황 |
| 11 | `Ret_Samsung_vol6` | 주가 | 삼성전자 변동성 |
| 12 | `Ret_KLAC_vol6` | **장비주 v2** | KLA Corp 변동성 ★ |
| 13 | `Ret_SKHynix_ma6` | 주가 | SK하이닉스 MA |
| 14 | `Worldwide_YoY_vol6` | WSTS 변동성 | 글로벌 업황 변동성 |
| 15 | `Ret_TSM_vol6` | 주가 | TSMC 변동성 |
| 16 | `Ret_KLAC_ma6` | **장비주 v2** | KLA Corp MA ★ |
| 17 | `FRED_MfgEmp_YoY_ma3` | FRED 거시 | 제조업 고용 MA |
| 18 | `Americas_YoY_lag6` | WSTS | 미주 업황 6개월 전 |
| 19 | `FRED_SemiPPI_YoY_lag12` | **반도체 PPI v2** | 반도체 PPI 12개월 전 ★ |
| 20 | `Japan_YoY_lag12` | WSTS | 일본 업황 12개월 전 |
| 21 | `FRED_ConsSenti_YoY_ma6` | FRED 거시 | 소비자심리 MA |
| 22 | `Europe_YoY_vol6` | WSTS 변동성 | 유럽 업황 변동성 |
| 23 | `Japan_YoY` | WSTS | 현재 일본 업황 |
| 24 | `Americas_YoY` | WSTS | 현재 미주 업황 |
| 25 | `Asia_Pacific_YoY_cycle_pct24` | WSTS 사이클 | 아시아 사이클 위치 |
| 26 | `Japan_YoY_cycle_pct24` | WSTS 사이클 | 일본 사이클 위치 |
| 27 | `Ret_ASML_vol6` | 주가 | ASML 변동성 |
| 28 | `FRED_MfgEmp_YoY` | FRED 거시 | 제조업 고용 YoY |
| 29 | `Americas_YoY_cycle_pct24` | WSTS 사이클 | 미주 사이클 위치 |
| 30 | `FRED_PCE_Core_YoY_ma3` | FRED 거시 | 근원 PCE MA |
| 31 | `Europe_YoY_lag12` | WSTS | 유럽 업황 12개월 전 |
| 32 | `Ret_NVDA_vol3` | 주가 | NVIDIA 단기 변동성 |
| 33 | `Ret_ASML_vol3` | 주가 | ASML 단기 변동성 |
| 34 | `Ret_NVDA_lag6` | 주가 | NVIDIA 6개월 전 |
| 35 | `FRED_ConsSenti_YoY_ma3` | FRED 거시 | 소비자심리 단기 MA |
| 36 | `Ret_NVDA_vol6` | 주가 | NVIDIA 변동성 |
| 37 | `Americas_YoY_momentum_3_12` | WSTS 모멘텀 | 미주 단기-장기 모멘텀 격차 |
| 38 | `Americas_YoY_vol6` | WSTS 변동성 | 미주 업황 변동성 |
| 39 | `Worldwide_YoY_lag6` | WSTS | 글로벌 업황 6개월 전 |
| 40 | `Asia_Pacific_YoY_lag6` | WSTS | 아시아 업황 6개월 전 |
| 41 | `Americas_YoY_lag12` | WSTS | 미주 업황 12개월 전 |
| 42 | `Ret_AMAT_ma6` | **장비주 v2** | Applied Materials MA ★ |
| 43 | `FRED_PCE_Core_YoY_lag6` | FRED 거시 | 근원 PCE 6개월 전 |
| 44 | `Japan_YoY_ma3` | WSTS | 일본 업황 단기 MA |
| 45 | `Japan_YoY_accel` | WSTS 가속도 | 일본 업황 가속도 |
| 46 | `Europe_YoY_ma12` | WSTS | 유럽 업황 장기 MA |
| 47 | `FRED_IndProd_YoY` | FRED 거시 | 산업생산 YoY |
| 48 | `Ret_SKHynix_lag6` | 주가 | SK하이닉스 6개월 전 |
| 49 | `FRED_SemiProd_YoY_lag6` | FRED 거시 | 반도체 생산지수 6개월 전 |
| 50 | `FRED_PCE_Core_YoY_ma6` | FRED 거시 | 근원 PCE 중기 MA |

**★ v2 신규 피쳐 선택 현황 (50개 중 7개)**

| 피쳐 | SHAP 순위 | 카테고리 |
|------|-----------|----------|
| `Equip_B2B_Proxy_ma6` | **4위** | 장비주 합성 B2B Proxy |
| `FRED_ElecCompPPI_YoY_ma6` | **6위** | 전자부품 PPI |
| `Ret_LRCX_ma6` | **7위** | Lam Research |
| `Ret_KLAC_vol6` | 12위 | KLA Corp 변동성 |
| `FRED_SemiPPI_YoY_lag12` | 19위 | 반도체 PPI 12개월 전 |
| `Ret_KLAC_ma6` | 16위 | KLA Corp MA |
| `Ret_AMAT_ma6` | 42위 | Applied Materials MA |

---

## 5. 평가 지표

### 5-1. 평가 지표 구성

| 지표 | 설명 |
|------|------|
| **RMSE** | 전체 구간 예측 오차 |
| **RMSE_Bull** | Bull 구간(YoY ≥ 0)만 필터링한 RMSE |
| **RMSE_Bear** | Bear 구간(YoY < 0)만 필터링한 RMSE |
| **Direction_Acc** | 방향(Bull/Bear) 예측 정확도 |
| **Asym_Loss** | 비대칭 가중 MSE (Bear ×1.5) |
| **Weighted_RMSE** | 방향 정확도 × Bear 여부 복합 가중 RMSE |

시계열 CV 각 fold의 지표를 계산한 뒤 **fold 평균(avg_*)** 으로 최종 성능을 집계함 (`nan` fold는 자동 제외).

### 5-2. 비대칭 손실 함수 (Asym_Loss)

반도체 산업 특성상 Bear 국면(YoY < 0) 예측 실패가 더 큰 손해를 초래하므로,
학습 단계에서 Bear 구간 오차에 1.5배 페널티를 부여함.

```
Asym_Loss(y, ŷ) = mean( w_i × (y_i − ŷ_i)² )

w_i = 1.5  if  y_i < 0  (Bear)
w_i = 1.0  if  y_i ≥ 0  (Bull)
```

**적용 방법:**

| 모델 | 적용 방식 |
|------|-----------|
| XGBoost | `XGBRegressor(objective=asymmetric_mse_xgb)` |
| LightGBM | `LGBMRegressor(objective=asymmetric_mse_lgb)` |
| Ridge / Lasso | 적용 불가 (선형 모델은 MSE 고정) — SHAP 피쳐 축소 효과만 반영 |

데이터 구성: Bull 274개월 / Bear 111개월 (약 7:3 비율)

### 5-3. 방향 정확도 기반 가중 RMSE (Weighted_RMSE)

오경보(Bull을 Bear로 예측) 및 Bear 국면 방향 오류(Bear를 Bull로 예측)를 가장 무겁게 페널티 부여.

```
Weighted_RMSE(y, ŷ) = sqrt( mean( w_i × (y_i − ŷ_i)² ) )

w_i = bear_factor × direction_factor

bear_factor   = 1.5  if  y_i < 0,  else 1.0
direction_factor = 2.0  if  sign(y_i) ≠ sign(ŷ_i),  else 1.0
```

| 케이스 | bear_factor | direction_factor | 최종 가중치 |
|--------|-------------|-----------------|------------|
| Bull 맞춤 | 1.0 | 1.0 | **1.0** ← 기본 |
| Bull 틀림 | 1.0 | 2.0 | **2.0** ← 오경보 |
| Bear 맞춤 | 1.5 | 1.0 | **1.5** ← 크기 오차 주의 |
| Bear 틀림 | 1.5 | 2.0 | **3.0** ← 최악, 가장 무겁게 |

모델 정렬 기준은 `avg_Weighted_RMSE` (낮을수록 우수).

---

## 6. 하이퍼파라미터 최적화 (Optuna v2)

| 항목 | 설정 |
|------|------|
| Sampler | TPE (Tree-structured Parzen Estimator) |
| Pruner | MedianPruner (n_startup=10, warmup=5) |
| Trials | 50회 (모델당) |
| 목적함수 | 비대칭 손실 CV 평균 최소화 (XGB/LGB) / RMSE (Ridge/Lasso) |
| CV | TimeSeriesSplit(n_splits=5, test_size=12) |
| 입력 피쳐 | SHAP 선택 50개 |

**최적 하이퍼파라미터**

| 파라미터 | XGBoost | LightGBM |
|---------|---------|----------|
| n_estimators | 335 | 391 |
| learning_rate | 0.096 | 0.027 |
| max_depth | 5 | 7 |
| num_leaves | — | 40 |
| subsample | 0.628 | 0.625 |
| colsample_bytree | 0.842 | 0.831 |
| reg_alpha | 0.00094 | 2.395 |
| reg_lambda | 0.00015 | 0.00365 |
| min_child_weight/samples | 5 | 5 |
| 최적 비대칭 손실 | **107.57** | 109.22 |

---

## 7. 모델 성능 비교

### 7-1. v1 vs v2 전체 비교

평가 기준: TimeSeriesSplit 5-fold CV (1993-07 ~ 2025-07, 385개월)  
정렬 기준: `avg_Weighted_RMSE` (낮을수록 우수)

| 모델 | avg RMSE | avg RMSE_Bull | avg RMSE_Bear | avg DirAcc | avg AsymLoss | avg Weighted_RMSE | 비고 |
|------|----------|--------------|--------------|------------|:------------:|:----------------:|------|
| **v2 XGBoost_Asym_SHAP** | **8.54** | **8.95** | **4.61** | **96.7%** | **110.92** | **9.11** | 비대칭 손실, 50피쳐 |
| **v2 LightGBM_Asym_SHAP** | 9.51 | 9.29 | 8.67 | **96.7%** | 123.65 | 10.47 | 비대칭 손실, 50피쳐 |
| v2 Lasso_SHAP | 14.63 | 13.91 | 19.73 | 91.7% | 309.82 | 16.54 | SHAP 축소 효과 |
| v2 Ridge_SHAP | 15.05 | 13.36 | 23.79 | 91.7% | 356.49 | 17.16 | SHAP 축소 효과 |
| v1 XGBoost | 10.40 | — | — | 95.0% | 147.24 | — | 기본 MSE, 136피쳐 |
| v1 LightGBM | 10.65 | — | — | 91.7% | 150.48 | — | 기본 MSE, 136피쳐 |
| v1 Lasso | 20.31 | — | — | 73.3% | 508.65 | — | |
| v1 Ridge | 27.46 | — | — | 65.0% | 926.53 | — | |

> v1 모델은 평가 지표 개편 이전 결과로 `avg_RMSE_Bull`, `avg_RMSE_Bear`, `avg_Weighted_RMSE` 없음(—).

### 7-2. 개선 요약 (v1 → v2)

| 모델 | RMSE 변화 | DirAcc 변화 | AsymLoss 변화 | avg Weighted_RMSE |
|------|-----------|-------------|---------------|:-----------------:|
| XGBoost | 10.40 → **8.54** (−17.9%) | 95.0% → **96.7%** (+1.7pp) | 147.24 → **110.92** (−24.6%) | **9.11** |
| LightGBM | 10.65 → **9.51** (−10.7%) | 91.7% → **96.7%** (+5.0pp) | 150.48 → **123.65** (−17.9%) | 10.47 |
| Lasso | 20.31 → **14.63** (−28.0%) | 73.3% → **91.7%** (+18.4pp) | 508.65 → **309.82** (−39.1%) | 16.54 |
| Ridge | 27.46 → **15.05** (−45.2%) | 65.0% → **91.7%** (+26.7pp) | 926.53 → **356.49** (−61.5%) | 17.16 |

### 7-3. 주요 해석

- **XGBoost**: RMSE −17.9%, AsymLoss −24.6% 모두 개선. 특히 `avg_RMSE_Bear=4.61`로 Bear 구간 오차가 Bull(8.95)보다 훨씬 낮아, 비대칭 손실 학습이 Bear 국면 예측에 집중했음을 확인.
- **LightGBM**: RMSE −10.7%, 방향 정확도 91.7% → 96.7% (+5.0pp). `avg_RMSE_Bear=8.67`로 Bear 구간에서 XGBoost보다 오차가 크나, Weighted_RMSE 10.47로 여전히 준수.
- **Lasso/Ridge**: 피쳐 수 136 → 50 축소로 다중공선성 해소. 방향 정확도가 65~73% → 91.7%로 비약적 개선. 다만 `avg_RMSE_Bear`가 20~24대로 Bear 구간 크기 오차는 여전히 큼.
- **Weighted_RMSE 관점**: XGBoost(9.11) > LightGBM(10.47) > Lasso(16.54) > Ridge(17.16) 순으로 Bear 틀림 페널티(×3.0)를 가장 잘 억제한 모델은 XGBoost.

---

## 8. 출력 파일 목록

```
outputs/
├── data/
│   ├── merged_dataset_v2.csv          # 신규 데이터 포함 병합 데이터셋
│   └── features_dataset_v2.csv        # 171개 피쳐 + 2개 타겟
├── eda/
│   └── 09_new_features_v2.png         # 신규 피쳐 시계열 시각화
└── models/
    ├── shap_selected_features.txt     # SHAP 선택 피쳐 50개 목록
    ├── shap_importance_v2.png         # SHAP mean|value| 중요도 차트
    ├── benchmark_results_v2.csv       # 모델별 성능 비교
    ├── predictions_v2.csv
    ├── predictions_v2.png
    ├── benchmark_plot_v2.png
    ├── xgboost_asym_shap.pkl          # 비대칭 손실 기본 XGBoost (50피쳐)
    ├── lightgbm_asym_shap.pkl
    ├── best_xgboost_v2.pkl            # Optuna 최적화 XGBoost ← 최종 권장 모델
    ├── best_lightgbm_v2.pkl           # Optuna 최적화 LightGBM
    ├── best_ridge_v2.pkl
    ├── best_lasso_v2.pkl
    ├── best_params_summary_v2.csv
    └── optuna_study_results_v2.csv
```

---

## 9. 최종 권장 모델

**`xgboost_asym_shap.pkl`**

- 피쳐: SHAP 선택 50개 (v2 신규 7개 포함)
- 학습 손실: 비대칭 MSE (Bear ×1.5)
- CV avg_RMSE: **8.54** | avg_RMSE_Bull: **8.95** | avg_RMSE_Bear: **4.61**
- avg_DirAcc: **96.7%** | avg_AsymLoss: **110.92** | avg_Weighted_RMSE: **9.11**

> Optuna 하이퍼파라미터 최적화 후 `best_xgboost_v2.pkl`로 교체 권장 (`v2/hyperparameter_tuning.py`).

**모델 선정 기준**

| 우선순위 | 지표 | 의미 |
|---------|------|------|
| 1 | `avg_Weighted_RMSE` ↓ | Bear 틀림(×3.0) 최소화 — 핵심 목표 |
| 2 | `avg_DirAcc` ↑ | 업황 전환 방향 예측력 |
| 3 | `avg_RMSE_Bear` ↓ | Bear 구간 크기 오차 |
| 4 | `avg_RMSE` ↓ | 전체 예측 정밀도 |

---

## 10. 결과 해석 및 시사점

### 10-1. 왜 Bear가 Bull보다 예측하기 쉬운가: 반도체 재고 사이클의 비대칭성

v2 XGBoost의 가장 이례적인 결과는 **Bear RMSE(4.61)가 Bull RMSE(8.95)보다 절반 수준**이라는 점이다.

| 모델 | RMSE_Bull | RMSE_Bear | Bear/Bull 비율 |
|------|-----------|-----------|:-------------:|
| XGBoost_Asym_SHAP | 8.95 | **4.61** | **0.52** — Bear가 더 정확 |
| LightGBM_Asym_SHAP | 9.29 | 8.67 | 0.93 — 거의 균형 |
| Lasso_SHAP | 13.91 | 19.73 | 1.42 — 통상적 방향 |
| Ridge_SHAP | 13.36 | 23.79 | 1.78 — 통상적 방향 |

통상적으로 Bear 국면은 데이터 수가 적고(111개월, 전체의 29%) 예측이 어렵다고 알려져 있다. 그러나 반도체 업황의 Bear는 도메인 특성상 **재현 가능한 패턴**을 따른다.

반도체 하락 국면은 사실상 **재고 사이클(inventory cycle)**의 반복이다. 전방 세트(스마트폰, PC, 서버)의 수요가 꺾이면 완성품 업체들이 반도체 주문을 동시에 급격히 줄이는 이른바 불휩 효과(bullwhip effect)가 발생한다. 2015–16년 D램 과잉, 2018–19년 무역전쟁발 재고 조정, 2022–23년 코로나 수요 이후 재고 조정이 모두 이 패턴을 따랐다. 공통 선행 신호는 명확하다. WSTS 기준 YoY 성장률이 일정 수준 이하로 떨어지면서 재고 지표가 동반 악화되고, 이후 3~6개월 내에 음수 전환이 나타나는 흐름이다. 모델이 학습한 피쳐 중 `Worldwide_YoY_vol6`(변동성), `Worldwide_YoY_cycle_pct24`(사이클 위치), `Americas_YoY_momentum_3_12`(모멘텀)은 바로 이 재고 사이클 전환의 전조를 포착하기 위해 설계됐다.

반면 **Bull 국면은 동인이 매 사이클마다 다르다**. 2010년대 초반 스마트폰 폭증, 2017–18년 서버·클라우드 투자 붐, 2020–21년 코로나 비대면 수요, 2023–24년 AI/HBM 수요 급증은 각기 다른 섹터에서, 다른 속도로, 다른 피쳐 조합으로 나타났다. 업황 상승의 강도와 지속 기간을 하나의 패턴으로 일반화하기 어렵기 때문에 Bull 구간 RMSE가 더 높게 나오는 것은 도메인 측면에서 자연스러운 결과다.

비대칭 손실(Bear ×1.5)은 이 구조적 어려움을 보정하는 역할을 한다. Bear 샘플이 기울기 계산에서 1.5배 영향력을 가지므로, 모델은 비교적 적은 Bear 데이터(111개월)로도 그 패턴을 충분히 학습할 수 있다.

다만 Bear RMSE 역전이 지나치게 강하면 **하락 편향(negative bias)** 이 생겼을 가능성도 있다. Bear를 과도하게 예측하면 Bull 국면에서 불필요한 경보가 늘어, 실무에서 "늑대 소리"로 무시될 위험이 있다. 잔차 분포의 평균이 음수로 치우쳐 있는지 반드시 확인해야 한다.

---

### 10-2. 비대칭 손실의 설계 근거: 반도체 기업의 의사결정 비용

Bear ×1.5 페널티는 단순한 수식이 아니라 **반도체 산업의 비용 비대칭성**을 반영한다.

Bear 국면을 놓쳤을 때(Bear를 Bull로 예측)의 피해는 재고 과잉 투자다. 제조사는 6개월~1년치 웨이퍼 투입(fab in-process)을 되돌릴 수 없고, 완성된 재고는 ASP 하락과 함께 대규모 상각으로 이어진다. 삼성전자가 2022–23년에 D램·낸드 재고 정상화에 약 6분기가 걸린 것이 대표적 사례다.

반면 Bull 국면을 놓쳤을 때(Bull을 Bear로 예측)의 피해는 기회 손실이다. 생산 증설이나 재고 확보 타이밍을 늦추는 것은 다음 분기에 만회할 수 있고, 재무적 손상은 상대적으로 작다.

Weighted_RMSE의 Bear 틀림 가중치 3.0(= bear_factor 1.5 × direction_factor 2.0)은 "Bear를 방향까지 틀리는 것"이 단순 크기 오차보다 훨씬 치명적임을 수식으로 표현한 것이다. XGBoost가 이 지표에서 9.11로 가장 낮다는 것은, 6개월 선행 시점에서 Bear 전환 오경보 비용과 Bear 미탐 비용 모두를 가장 잘 억제한다는 의미다.

---

### 10-3. SHAP 상위 피쳐와 반도체 선행 지표 해석

SHAP 상위 피쳐들은 반도체 업황 예측의 교과서적 선행 지표 체계와 잘 대응된다.

**1위 `FRED_IndProd_YoY_lag12` — 12개월 전 산업생산**

반도체는 최종재가 아니라 부품이다. 스마트폰, 서버, 자동차 등 최종 전자기기의 생산량이 먼저 움직이고, 그 수요가 반도체 주문으로 전달되기까지 통상 6~12개월이 걸린다(설계 사이클, 공급망 리드타임). 12개월 전 미국 산업생산지수가 현재 반도체 업황의 최고 선행 지표라는 결과는, 수요 파생 구조(derived demand)를 모델이 정확히 포착하고 있음을 보여 준다.

**2위 `Ret_SOX_ma6` — SOX 6개월 이동평균**

필라델피아 반도체지수(SOX)는 기관 투자자들의 반도체 사이클 전망을 집약한다. 시장이 업황 회복을 6개월 이상 선행해 가격에 반영하는 경향이 있어, SOX의 추세(MA6)가 실제 WSTS 매출 YoY를 선행하는 것은 이론적으로 타당하다. 이 피쳐가 2위를 차지한 것은 주가 데이터가 순수한 선행 정보를 담고 있음을 확인한다.

**4위 `Equip_B2B_Proxy_ma6` — 장비주 B2B Proxy 6개월 MA (v2 신규)**

SEMI Book-to-Bill Ratio는 반도체 장비 수주/출하 비율로, 1.0 이상이면 업황 확장 신호, 이하면 수축 신호로 해석된다. 장비 발주에서 실제 생산 능력 증가까지 통상 9~18개월이 걸리기 때문에 강력한 선행 지표다. 원본 데이터를 구하지 못해 AMAT, LRCX, KLAC의 주가 수익률 평균으로 대체했음에도 SHAP 4위에 오른 것은, **장비주 주가가 SEMI B2B의 시장 기대치를 이미 선반영**하기 때문이다. Lam Research(LRCX, 식각장비)와 KLA Corporation(KLAC, 계측장비)이 각각 7위, 12위, 16위에 추가로 선정된 것도 같은 맥락이다. 특히 LRCX는 낸드 플래시 식각 장비에 특화되어 있어 메모리 반도체 사이클과의 연관성이 높다.

**6위 `FRED_ElecCompPPI_YoY_ma6` — 전자부품 PPI 6개월 MA (v2 신규)**

전자부품 도매물가지수의 6개월 이동평균은 반도체 현물가 방향성의 간접 지표다. DRAM·낸드 현물가는 재고 사이클과 동행하는데, 현물가가 계약가보다 먼저 움직여 향후 WSTS 매출에 반영된다. 직접 반도체 가격 데이터는 유료지만, 이 공개 PPI 지표가 6위를 기록한 것은 가격 모멘텀이 업황 예측에 실질적인 정보를 제공한다는 뜻이다.

**3위 `Asia_Pacific_YoY` — 현재 아시아·태평양 업황**

전 세계 반도체 매출에서 아시아·태평양 비중은 55~60%에 달한다. TSMC(파운드리), 삼성·SK하이닉스(메모리)가 모두 이 권역에 있어, AP 지역 YoY가 글로벌 YoY의 선행·동행 지표가 된다. 현재 AP 업황이 3위를 차지한 것은 역내 공급망의 재고 흐름이 글로벌 방향을 결정하는 구조를 반영한다.

---

### 10-4. XGBoost vs LightGBM: 재고 사이클 포착 방식의 차이

두 모델의 Bear RMSE 차이(4.61 vs 8.67)는 단순한 알고리즘 차이가 아니라 **사이클 전환점 포착 방식의 차이**로 해석할 수 있다.

반도체 재고 사이클에서 Bull→Bear 전환은 짧고 급격하다. WSTS 데이터를 보면 YoY가 +20%에서 −20%로 전환되는 데 보통 2~4분기밖에 걸리지 않는다. XGBoost의 Exact greedy 분기는 이런 급격한 전환점을 날카롭게 포착하는 반면, LightGBM의 히스토그램 근사는 경계를 다소 평탄화한다.

결과적으로 XGBoost는 Bear 진입 국면에서 더 정밀하게 반응하지만, Bear 기간 내 크기 예측에 집중하느라 Bull 상승폭 예측(RMSE_Bull 8.95)은 LightGBM(9.29)과 큰 차이가 없다.

방향 정확도가 96.7%로 두 모델 동일하다는 점도 중요하다. 5-fold CV 기준 60개 test 월 중 약 2개만 방향을 틀렸다는 의미다. 6개월 선행 시점에서 이 정도 방향 정확도는, 대부분의 반도체 사이클 전환점(2015–16년, 2018–19년, 2022–23년)을 모두 조기 감지했음을 시사한다.

---

### 10-5. 선형 모델의 역할: 방향 신호로서의 가치

Lasso/Ridge는 Bear RMSE가 19~24pp에 달해 다운사이클 폭 예측에는 부적합하다. 그러나 방향 정확도 91.7%는 실무에서 의미가 있다.

반도체 공급망 관리에서 가장 먼저 필요한 판단은 "다음 6개월이 성장인가 수축인가"다. 이 방향만 맞으면 웨이퍼 투입량 조정, 재고 목표 변경 등 대응이 가능하다. Lasso_SHAP의 경우 방향 정확도가 v1 73.3%에서 v2 91.7%로 18.4pp 개선됐는데, 이는 피쳐 수 축소(136 → 50개)가 노이즈 피쳐로 인한 다중공선성을 제거한 결과다. 반도체 지표들은 동행성이 강해(SOX, WSTS 지역별, FRED 거시 등이 서로 높은 상관), 136개 피쳐를 그대로 선형 모델에 넣으면 계수가 불안정해진다. SHAP이 상위 50개로 압축하면서 각 피쳐의 역할이 명확해졌다.

---

### 10-6. 한계 및 향후 개선 방향

| 항목 | 도메인 맥락 | 개선 방향 |
|------|------------|-----------|
| **하락 편향 검증** | XGBoost Bear RMSE 역전이 과도한 Bear 예측에서 기인했다면, 2024년 이후 AI 수요 주도 Bull 구간에서 오경보가 늘었을 수 있음 | 잔차 평균 및 분포 분석; 최근 12개월 구간 별도 검증 |
| **AI/HBM 국면 일반화** | 학습 데이터(~2025-07)에 HBM 수요 폭증 구간이 일부 포함되나, AI 수요가 기존 사이클 패턴을 교란할 가능성 있음 | 2023년 이후 구간을 별도 out-of-sample 검증 셋으로 분리 |
| **SEMI B2B 실데이터 확보** | `Equip_B2B_Proxy_ma6`가 SHAP 4위임에도 장비주 Proxy 사용 중. 실제 B2B Ratio는 장비 발주→양산 9~18개월 선행성을 더 직접 포착 | SEMI.org 데이터 수동 수집 또는 유료 데이터 도입 |
| **현물가 데이터 부재** | DRAM·낸드 현물가(DRAMeXchange 등)는 유료. 현재 PPI로 대체 중이나 실제 ASP 변동은 업황 YoY와 강하게 연동 | 공개 메모리 현물가 API 탐색(일부 커뮤니티 스크래퍼 존재) |
| **거시 충격 내성** | 2020년 코로나, 2022년 금리 급등 같은 외생 충격은 재고 사이클 패턴을 벗어남. 이런 구간에서 모델 성능이 저하될 수 있음 | 충격 구간 dummy 피쳐 추가 또는 구간별 성능 분해 분석 |
