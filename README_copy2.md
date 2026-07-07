# 📈 반도체 업황 기반 SK하이닉스 주가 예측

#### 🚦 Team 신호등바뀌었어redred : 김석우, 윤채영, 이지윤

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [대시보드](#대시보드)
3. [Stage별 모델 성능 Report](#Stage별-모델-성능-Report)
4. [환경 설정](#환경-설정)
5. [실행 방법](#실행-방법)
6. [파이프라인 단계별 설명](#파이프라인-단계별-설명)
7. [디렉토리 구조](#디렉토리-구조)

---

## 프로젝트 개요

주식 시장은 미래를 선반영한다. 투자자들은 지금 업황이 아니라 **"앞으로의 업황이 어떻게 될 것인가"** 를 보고 현재에 미리 사고판다. 특히 반도체 산업의 사이클에서 이러한 패턴을 확인할 수 있다. 반도체 공급망의 흐름은 `글로벌 업황 변동 → 반도체 기업 실적 변동 → 주가 변동` 순서를 따른다. 이 인과 구조를 모델에 직접 반영하면 단순 패턴 매칭보다 해석 가능한 예측이 가능하다.

즉, **업황이 실제로 좋아지는 시점보다 주가가 6개월 정도 먼저 움직이는 패턴** 을 활용한다면, **"앞으로 6개월 후 업황 방향을 미리 알 수 있다면, 지금부터 6개월간 주가가 오를지 내릴지 예측할 수 있다"** 는 가설이 성립한다. 따라서 본 프로젝트는 이를 2단계로 모델링한다.

```
[반도체 업황 선행 예측] ──────────────────→ [SK하이닉스 주가 방향 예측]
     Stage 1            Bridge 피처            Stage 2
  (Worldwide YoY%,   (v2_pred_ww_yoy)    (6개월 후 종가 수익률
   6개월 선행, 월간)                            방향, 분기)
```

---

## 대시보드

Streamlit Cloud에 배포된 공개 대시보드에서 예측 결과를 확인할 수 있습니다.

> 🔗 [대시보드 바로가기](https://kubig26-1conference-jtydr4ccfejqcsoz3wwsms.streamlit.app/)

### 대시보드 동작 방식

사용자가 대시보드에 접속하면, 사전에 파이프라인에 따라 학습·저장된 모델 파라미터(pkl)와 피처 데이터(csv)를 로드해 Stage 2 XGBoost 모델을 그 자리에서 재학습한 뒤, 가장 최근 Hold-out 구간을 예측해 "오를까 내릴까" 결과를 표시한다. 사이드바에서는 KOSPI·SOX 최근 3개월 등락률을 yfinance로 실시간 조회해 참고 시장 신호로 함께 보여준다.

### 자동 재학습 (GitHub Actions)

SK하이닉스 실적 발표일(1·4·7·10월 넷째 주 금요일)마다 자동으로 실행됩니다.

1. cron이 매월 22~28일 실행을 트리거하면, 당일이 KST 기준 금요일인지 확인 후 금요일이 아니면 즉시 종료 (넷째 주 금요일에만 재학습이 trigger되는 구조)
2. WSTS 사이트에서 최신 데이터(`wsts_historical.xlsx`) 자동 파싱·다운로드
3. Stage 1 → Stage 2 파이프라인 전체 순서대로 재학습
4. 모델(pkl), 피처 데이터(csv), 시각화(png), 지표(csv) 파일만 선택적으로 커밋·푸시
5. Streamlit Cloud가 새 커밋을 감지해 대시보드 자동 재배포

> GitHub Secrets에 `FRED_API_KEY`를 등록해야 합니다.
> Settings → Secrets and variables → Actions → New repository secret

---

## Stage별 모델 성능 Report

### Stage 1️⃣ — 반도체 업황 YoY% 6개월 선행 예측 (월간)

**Stage 1의 Task**: WSTS 반도체 Worldwide 매출의 전년 대비 성장률(YoY%)이 6개월 후에 어떻게 될지를 예측한다. 양수(+)이면 업황 확장(Bull), 음수(-)이면 업황 수축(Bear)을 의미한다.

| 지표                        | 값                                         |
| --------------------------- | ------------------------------------------ |
| CV RMSE (5-fold)            | **6.06**                             |
| CV DirAcc                   | **95.0%** (Bull 100.0% / Bear 87.1%) |
| CV AsymLoss                 | **6.01**                             |
| Hold-out RMSE (최근 24개월) | 17.41                                      |
| Hold-out DirAcc             | 95.8%                                      |
| 최종 선택 피처 수           | 30개 / 165개                               |

> **DirAcc**: 하락을 놓치는 것에 가장 큰 패널티를 주기 위해 가중치를 부여한 Accuracy 지표.
> 
> *Bull 정답: ×1.0  /  Bull 오답: ×2.0 / Bear 정답: ×1.5  /  Bear 오답: ×3.0*

### Stage 2️⃣ — SK하이닉스 6개월 후 종가 수익률 방향 예측 (분기)

**Stage 2의 Task**: 앞서 Stage 1에서 생성한 **6개월 후의 반도체 업황 예측값** 을 활용해서, 관찰일로부터 6개월 후 실적발표일까지의 SK하이닉스 종가 수익률의 변동 방향을 예측한다. 업황 개선 기대가 이 6개월 구간 안에서 주가에 선반영되는 패턴을 포착하는 것이 핵심이다. 양수(+)이면 상승(Bull), 음수(-)이면 하락(Bear) 신호다.
`TARGET = (실적발표일 종가 / 관찰일 종가 − 1) × 100 (%)`

평가 환경: Tune 86분기 / Hold-out 20분기 (2021-01-28 ~ 2025-10-23)

| 지표                          | 값                                         |
| ----------------------------- | ------------------------------------------ |
| CV RMSE (5-fold)              | **18.974%**                          |
| CV DirAcc                     | **90.0%** (Bull 88.3% / Bear 100.0%) |
| CV AsymLoss                   | **19.320**                           |
| CV IC (Spearman)              | **+0.600**                           |
| Hold-out RMSE                 | **62.389%**                          |
| Hold-out DirAcc (최근 20분기) | **60.0%** (Bull 91.7% / Bear 12.5%)  |
| Hold-out IC                   | **+0.033**                           |
| 최적 recency_scale            | 0.4087                                     |
| 최종 선택 피처 수             | 25개                                       |

---

## 환경 설정

### 1. Python 버전

Python **3.9 이상**을 권장

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. FRED API 키 설정

```bash
# Mac / Linux
export FRED_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:FRED_API_KEY="your_api_key_here"
```

---

## 실행 방법

**Stage 1을 먼저 실행해야 합니다.** Stage 2는 Stage 1의 피처셋과 학습된 모델 파라미터를 입력으로 사용합니다.

### Stage 1️⃣ — 반도체 업황 YoY% 예측

```bash
python stage1/pipeline.py
```

### Stage 2️⃣ — SK하이닉스 주가 수익률 예측

```bash
python stage2/pipeline.py
```

---

## 파이프라인 단계별 설명

> 스텝별 상세 로직(피처 정의, 손실 함수, Dynamic Sample Weight 수식 등)은 각 스크립트 상단 docstring에 정리되어 있습니다.

### Stage 1️⃣

#### Step 1 — 데이터 수집 + 피처 엔지니어링 (`s1_data.py`)

| 데이터 소스              | 내용                                                                                                |
| ------------------------ | --------------------------------------------------------------------------------------------------- |
| `wsts_historical.xlsx` | WSTS 월별 반도체 출하량 (Americas / Europe / Japan / Asia Pacific / Worldwide)                      |
| FRED API                 | 산업생산지수, 장단기 금리차(T10Y2Y·T10Y3M), 소비자심리지수, 신규수주, 제조업 고용, 연방기금금리 등 |
| yfinance                 | 필라델피아 반도체지수(SOX), NVDA, TSM, ASML, 삼성전자, SK하이닉스 주가                              |

#### Step 2 — Optuna 하이퍼파라미터 튜닝 (`s2_tune.py`)

- 목적함수: TimeSeriesSplit 5-fold CV **RMSE 최소화**

#### Step 3 — 피처 선택 (`s3_select.py`)

1. **다중공선성 제거**: |Pearson r| ≥ 0.9인 피처 쌍에서 하나 제거
2. **SHAP 중요도 계산**: 전체 피처 중요도 순위 산출
3. **RFE 커브**: 피처 수 변화에 따른 CV AsymLoss 측정 후 최적 수 선택
4. 최종 선택: **30개** / 165개

#### Step 4 — Bear 최적화 (`s4_optimize.py`)

- 목적함수: **AsymLoss** (Bear 오예측 ×3.0 페널티) 최소화
- 출력: `stage1/outputs/models/best_xgboost_final.pkl` ← Stage 2 입력으로 활용됨

#### Step 5 — 최종 평가 + 시각화 (`s5_evaluate.py`)

---

### Stage 2️⃣

#### Step 1 — 분기 날짜 생성 + 타겟 수익률 산출 (`s1_dates.py`)

- SK하이닉스 **분기 실적발표일**: 매년 1·4·7·10월 넷째 주 목요일
- **관찰일**: 실적발표일 정확히 6개월 전 같은 요일

#### Step 2 — 피처 데이터 수집 (`s2_data.py`)

관찰일 기준 5개 피처 그룹 수집 (lookahead 없이, 관찰일 이전 데이터만 사용):

| 그룹                      | 내용                                                                          |
| ------------------------- | ----------------------------------------------------------------------------- |
| A. SK하이닉스 기술적 지표 | 가격, 1·3·6·12개월 수익률, 변동성(60d), RSI(14), MA 괴리율, 52주 고저 위치 |
| B. 시장 센티먼트          | VIX, SOX, NVDA, TSM, ASML, Samsung, S&P500의 1·3·6개월 수익률               |
| C. WSTS 역사 데이터       | Worldwide·Asia Pacific YoY%, 이동평균, 모멘텀, 사이클 위치                   |
| D. FRED 거시지표          | 장단기 금리차(T10Y2Y·T10Y3M), 기준금리, 산업생산, PCE, 소비자심리            |
| E. 환율·원자재           | USD/KRW, WTI 유가의 3·6개월 수익률                                           |

#### Step 3 — Stage 1 Expanding Window Pseudo 예측 (`s3_stage1_feat.py`)

- 각 관찰일 기준으로 Stage 1 모델을 **expanding window** 방식으로 재훈련.

#### Step 4 — 피처 엔지니어링 + 피처 선택 (`s4_features.py`)

- 달력·사이클 피처 추가: 실적 발표 분기, 반도체 슈퍼사이클 위치, 장기 추세 proxy
- **3단계 피처 선택**: NaN 비율 필터 → VIF 다중공선성 제거(임계 10) → XGBoost importance 상위 60% → RFE 최종 **25개**

#### Step 5 — XGBoost Optuna 튜닝 + Dynamic Sample Weight (`s5_tune.py`)

학습은 2단계로 나뉜다.

- **Phase A** — RMSE 최소화: XGBoost 하이퍼파라미터 초기 탐색 → `skh_xgb_tuned.pkl`
- **Phase B** — AsymLoss 최소화: 방향성 최적화 + Dynamic Sample Weight 적용 → `skh_xgb_final.pkl`
- CV 구조: TimeSeriesSplit 5-fold (test_size=4분기, min_train=20분기)

#### Step 6 — 최종 평가 + 시각화 (`s6_evaluate.py`)

---

## 디렉토리 구조

```
├── app.py                          Streamlit 대시보드
├── requirements.txt                패키지 설치
├── packages.txt                    시스템 패키지 (Streamlit Cloud용, libgomp1)
├── Dockerfile                      배포용 Docker 이미지 정의
├── docker-compose.yml              로컬 Docker 실행 구성
├── wsts_historical.xlsx            WSTS 원본 데이터
│
├── .devcontainer/                  VS Code Dev Container 설정
├── .github/workflows/
│   └── monthly-retrain.yml         분기별 자동 재학습 워크플로우
│
├── stage1/                         Stage 1 — 반도체 업황 YoY% 예측
│   ├── config.py
│   ├── pipeline.py
│   ├── requirements.txt
│   ├── s1_data.py                  Step 1: 데이터 수집 + 피처 엔지니어링
│   ├── s2_tune.py                  Step 2: Optuna RMSE 튜닝
│   ├── s3_select.py                Step 3: 다중공선성 제거 + RFE 피처 선택
│   ├── s4_optimize.py              Step 4: AsymLoss Bear 최적화
│   ├── s5_evaluate.py              Step 5: 최종 평가 + 시각화
│   └── outputs/
│       ├── data/                   CSV 데이터 파일
│       ├── models/                 학습된 모델 pkl ★ Stage 2 및 대시보드 입력
│       ├── figures/                시각화 png
│       └── metrics/                평가 지표 CSV
│
├── stage2/                         Stage 2 — SK하이닉스 주가 수익률 예측
│   ├── config.py
│   ├── pipeline.py
│   ├── s1_dates.py                 Step 1: 분기 날짜 생성 + 타겟 수익률 산출
│   ├── s2_data.py                  Step 2: A~E 피처 수집
│   ├── s3_stage1_feat.py           Step 3: Stage 1 Expanding Window 예측
│   ├── s4_features.py              Step 4: 피처 엔지니어링 + 피처 선택
│   ├── s5_tune.py                  Step 5: XGBoost Optuna 튜닝
│   ├── s6_evaluate.py              Step 6: 최종 평가 + 시각화
│   └── outputs/
│       ├── data/                   CSV 데이터 파일 ★ 대시보드 입력
│       ├── models/                 학습된 모델 pkl ★ 대시보드 입력
│       ├── figures/                시각화 png
│       └── metrics/                평가 지표 CSV
│
└── ablation_study/                 Stage 1 공급 신호의 주가 선행성 검증 (Ablation Study)
    ├── README.md                   실행 방법 · 분석 구조 상세 설명
    ├── skhynix/                    SK하이닉스 ablation 파이프라인 (sk0~sk4)
    ├── asml/                       ASML ablation 파이프라인 (asml1~asml3)
    ├── model/                      Stage 1 스냅샷 산출물 보관 (outputs/만 존재)
    └── docs/                       분석 결과 보고서 (stage2_summary.md)
```
