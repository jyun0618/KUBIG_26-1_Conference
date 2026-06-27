"""
app.py — E2E 반도체 사이클 → SK하이닉스 수익률 예측 대시보드 (Streamlit)
================================================================================
Stage 1 (반도체 출하량 YoY 예측) → Stage 2 (SK하이닉스 6개월 수익률 방향 예측)
2단계 파이프라인의 학습 결과를 발표용으로 시각화한다.

[화면 설계]
  - 메인: 방향 예측(📈 상승 / 📉 하락)을 크고 명확하게
  - 부가: 예측 수익률 수치는 작은 caption으로
  - 알파: SHAP 피처 중요도(상위 10) + 과거 분기별 적중 히스토리(✅/❌)
  - E2E: 시각적 흐름 다이어그램 + 현재 시장 신호 요약 카드

[기술 노트]
  - 모델 로딩은 st.cache_resource, 데이터/연산 결과는 st.cache_data로 캐싱
  - 성능 지표(metrics)는 hold-out 평가를 재현해 런타임 계산
  - SHAP: shap.TreeExplainer로 XGBoost 설명 → st.bar_chart (matplotlib 미사용)
  - 한글 폰트가 없는 Linux 컨테이너 대비, 모든 시각화는 Streamlit 네이티브 차트 사용
"""

import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── 페이지 설정 (반드시 첫 Streamlit 호출) ──────────────────────────
st.set_page_config(
    page_title="반도체 사이클 → SK하이닉스 수익률 예측",
    page_icon="📈",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────
# 상수 정의
# ──────────────────────────────────────────────────────────────────

# 컨테이너 작업 경로(/app). 로컬 실행 시에는 app.py가 위치한 폴더로 폴백.
APP_ROOT = os.getenv("APP_ROOT", os.path.dirname(os.path.abspath(__file__)))

# S3에서 받아와야 하는 산출물 (S3 key == 로컬 상대경로)
ARTIFACTS = [
    "stage1/outputs/models/best_xgboost_final.pkl",
    "stage1/outputs/data/features_dataset.csv",
    "stage2/outputs/models/skh_xgb_final.pkl",
    "stage2/outputs/data/stage2_features.csv",
    "stage2/outputs/data/stage1_predictions.csv",
]

# Asymmetric Loss 가중치 (stage1/2 config.py와 동일) — Bear 오예측 페널티 강화
W_BULL_CORRECT, W_BULL_WRONG = 1.0, 2.0
W_BEAR_CORRECT, W_BEAR_WRONG = 1.5, 3.0
BEAR_SAMPLE_W = 2.0

# Stage별 메타데이터
STAGE1 = {
    "name": "Stage 1",
    "title": "반도체 출하량 YoY 예측",
    "features_path": os.path.join(APP_ROOT, "stage1/outputs/data/features_dataset.csv"),
    "model_path":    os.path.join(APP_ROOT, "stage1/outputs/models/best_xgboost_final.pkl"),
    "target":        "TARGET_Worldwide_YoY_T6",
    "test_eval":     24,        # hold-out 개월 수
    "value_label":   "예측 YoY",
    "freq_label":    "개월",
}
STAGE2 = {
    "name": "Stage 2",
    "title": "SK하이닉스 6개월 수익률 방향 예측",
    "features_path": os.path.join(APP_ROOT, "stage2/outputs/data/stage2_features.csv"),
    "model_path":    os.path.join(APP_ROOT, "stage2/outputs/models/skh_xgb_final.pkl"),
    "target":        "TARGET_SKH_6M_RET",
    "test_eval":     12,        # hold-out 분기 수
    "value_label":   "예측 수익률",
    "freq_label":    "분기",
}

STAGE1_PRED_PATH = os.path.join(APP_ROOT, "stage2/outputs/data/stage1_predictions.csv")
# Stage 1 → Stage 2로 전달되는 핵심 피처(예측값) 컬럼명
BRIDGE_COL = "v2_pred_ww_yoy"

# ── 디자인 스펙 컬러 팔레트 ─────────────────────────────────────
CLR_BLUE  = "#2a78d6"   # 예측값·AI 강조
CLR_TEAL  = "#1D9E75"   # 상승·실제값
CLR_RED   = "#E24B4A"   # 하락·경고
CLR_AMBER = "#EF9F27"   # 주의 구간
CLR_GRAY  = "#888780"   # 축·구분선
BG_BLUE   = "#e6f1fb"
BG_GREEN  = "#eaf3de"
BG_TEAL   = "#e1f5ee"
BG_RED    = "#fcebeb"
BG_AMBER  = "#faeeda"


# ──────────────────────────────────────────────────────────────────
# 1. S3 산출물 다운로드
# ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="S3에서 모델/데이터 산출물을 내려받는 중...")
def download_artifacts():
    """
    S3에서 ARTIFACTS를 APP_ROOT 하위로 다운로드한다.
    세션당 1회만 실행되도록 cache_resource로 캐싱.

    반환: dict(status, missing, error)
      - status == "ok"          : 전부 성공
      - status == "missing"     : 일부 파일이 버킷에 없음 → 학습 필요
      - status == "no_bucket"   : S3_BUCKET_NAME 미설정
      - status == "s3_error"    : 자격증명/네트워크 등 S3 접근 실패
    """
    bucket = os.getenv("S3_BUCKET_NAME")
    if not bucket:
        return {"status": "no_bucket", "missing": [], "error": None}

    try:
        import boto3
        from botocore.exceptions import ClientError, BotoCoreError, NoCredentialsError
    except ImportError as e:
        return {"status": "s3_error", "missing": [], "error": f"boto3 미설치: {e}"}

    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
    except (BotoCoreError, NoCredentialsError) as e:
        return {"status": "s3_error", "missing": [], "error": f"S3 클라이언트 생성 실패: {e}"}

    missing = []
    for key in ARTIFACTS:
        local_path = os.path.join(APP_ROOT, key)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            s3.download_file(bucket, key, local_path)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            # 객체가 없으면(404/NoSuchKey) "학습 필요" 신호로 수집
            if code in ("404", "NoSuchKey", "NoSuchBucket"):
                missing.append(key)
            else:
                return {"status": "s3_error", "missing": [], "error": str(e)}
        except (BotoCoreError, NoCredentialsError) as e:
            return {"status": "s3_error", "missing": [], "error": str(e)}

    if missing:
        return {"status": "missing", "missing": missing, "error": None}
    return {"status": "ok", "missing": [], "error": None}


def guard_artifacts():
    """다운로드 결과를 검사하고, 문제가 있으면 안내 후 대시보드를 중단한다."""
    result = download_artifacts()
    status = result["status"]

    if status == "ok":
        return

    if status == "no_bucket":
        st.error("⚙️ 환경변수 `S3_BUCKET_NAME`이 설정되지 않았습니다.")
        st.info("`.env`에 S3 버킷명과 AWS 자격증명을 설정한 뒤 다시 실행해 주세요.")
        st.stop()

    if status == "s3_error":
        st.error("❌ S3 접근에 실패했습니다. 자격증명 또는 네트워크를 확인해 주세요.")
        st.code(str(result["error"]), language="text")
        st.stop()

    if status == "missing":
        st.error("🛠️ 모델 학습이 필요합니다.")
        st.warning(
            "S3 버킷에서 아래 산출물을 찾을 수 없습니다. "
            "Stage 1·2 파이프라인을 먼저 실행해 산출물을 업로드해 주세요."
        )
        for key in result["missing"]:
            st.markdown(f"- `{key}`")
        st.stop()


# ──────────────────────────────────────────────────────────────────
# 2. 데이터 / 모델 로딩 (캐싱)
# ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """pkl 번들 로드: {'model', 'feature_names', 'best_params'}."""
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    """index_col=0(날짜) 기준 CSV 로드."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


# ──────────────────────────────────────────────────────────────────
# 3. 지표 계산 (hold-out 평가 재현)
# ──────────────────────────────────────────────────────────────────

def _safe_rmse(y_true, y_pred, mask):
    if not mask.any():
        return None
    err = y_true[mask] - y_pred[mask]
    return float(np.sqrt(np.mean(err ** 2)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, with_ic: bool = False) -> dict:
    """7~8개 표준 지표 계산 (evaluate 스크립트와 동일 정의)."""
    bull    = y_true > 0
    bear    = ~bull
    correct = (y_true > 0) == (y_pred > 0)
    w = np.where(bull & correct,  W_BULL_CORRECT,
        np.where(bull & ~correct, W_BULL_WRONG,
        np.where(bear & correct,  W_BEAR_CORRECT, W_BEAR_WRONG)))

    metrics = {
        "rmse":      float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "rmse_bull": _safe_rmse(y_true, y_pred, bull),
        "rmse_bear": _safe_rmse(y_true, y_pred, bear),
        "dir_acc":   float(correct.mean() * 100),
        "dir_bull":  float(correct[bull].mean() * 100) if bull.any() else None,
        "dir_bear":  float(correct[bear].mean() * 100) if bear.any() else None,
        "asym_loss": float(np.sqrt((w * (y_true - y_pred) ** 2).sum() / w.sum())),
    }
    if with_ic:
        # Spearman 순위상관(IC). scipy는 scikit-learn 의존성으로 항상 설치됨.
        ic = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
        metrics["ic"] = float(ic) if pd.notna(ic) else None
    return metrics


@st.cache_data(show_spinner="모델 성능을 평가하는 중...")
def evaluate_stage(features_path: str, model_path: str, target: str,
                   test_eval: int, with_ic: bool = False):
    """
    저장된 최종 모델의 하이퍼파라미터로 tune 구간 재학습 → hold-out 예측.
    반환: (metrics dict, 예측/실제 정렬 DataFrame)
    """
    import xgboost as xgb

    bundle  = load_model(model_path)
    model   = bundle["model"]
    feats   = bundle["feature_names"]
    params  = model.get_params()

    df = load_csv(features_path)
    # 모델이 학습한 피처만 사용 (없는 컬럼은 제외)
    use_feats = [f for f in feats if f in df.columns]
    df_clean  = df.dropna(subset=[target])
    X = df_clean[use_feats].ffill().fillna(0)
    y = df_clean[target]

    split = len(X) - test_eval
    X_tune, y_tune = X.iloc[:split], y.iloc[:split]
    X_ho,   y_ho   = X.iloc[split:], y.iloc[split:]

    # Bear(하락) 구간 sample_weight 강화 후 재학습
    w_tune = np.where(y_tune.values > 0, 1.0, BEAR_SAMPLE_W)
    m = xgb.XGBRegressor(**params)
    m.fit(X_tune, y_tune, sample_weight=w_tune)
    preds = m.predict(X_ho)

    metrics = compute_metrics(y_ho.values, preds, with_ic=with_ic)
    metrics["period"] = f"{y_ho.index[0].date()} ~ {y_ho.index[-1].date()}"
    metrics["n_holdout"] = len(y_ho)
    metrics["n_features"] = len(use_feats)

    out = pd.DataFrame({"실제값": y_ho.values, "예측값": preds}, index=y_ho.index)
    return metrics, out


@st.cache_data(show_spinner="SHAP 피처 중요도 계산 중...")
def compute_shap_importance(model_path: str, features_path: str, target: str,
                            top_n: int = 10) -> pd.DataFrame:
    """
    저장된 XGBoost 모델을 shap.TreeExplainer로 설명해
    평균 |SHAP| 기준 상위 top_n 피처 중요도를 반환한다.
    반환: index=피처명, 컬럼='평균 |SHAP|' 인 DataFrame.
    """
    import shap

    bundle = load_model(model_path)
    model  = bundle["model"]
    feats  = bundle["feature_names"]

    df = load_csv(features_path)
    use_feats = [f for f in feats if f in df.columns]
    df_clean  = df.dropna(subset=[target])
    X = df_clean[use_feats].ffill().fillna(0)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs    = np.abs(shap_values).mean(axis=0)

    s = (pd.Series(mean_abs, index=use_feats)
         .sort_values(ascending=False)
         .head(top_n))
    return s.rename("평균 |SHAP|").to_frame()


@st.cache_data(ttl=3600, show_spinner="시장 신호(yfinance) 수집 중...")
def get_market_momentum() -> dict:
    """
    KOSPI(^KS11) / SOX(^SOX) 최근 3개월 모멘텀(현재가/3개월전가 - 1, %)을 계산.
    네트워크 실패 시 None.
    """
    import yfinance as yf

    result = {}
    for label, ticker in [("KOSPI", "^KS11"), ("SOX", "^SOX")]:
        try:
            data  = yf.download(ticker, period="3mo", interval="1d",
                                progress=False, auto_adjust=True)
            close = np.asarray(data["Close"]).reshape(-1)
            close = close[~np.isnan(close)]
            if len(close) >= 2:
                result[label] = float(close[-1] / close[0] - 1.0) * 100
            else:
                result[label] = None
        except Exception:
            result[label] = None
    return result


# ──────────────────────────────────────────────────────────────────
# 4. 공통 UI 헬퍼
# ──────────────────────────────────────────────────────────────────

def _fmt(v, pct=False):
    if v is None:
        return "N/A"
    return f"{v:.1f}%" if pct else f"{v:.3f}"


def _pill(text: str, bg: str, fg: str) -> str:
    return (
        f"<span style='background:{bg};color:{fg};font-size:11px;"
        f"font-weight:500;padding:3px 10px;border-radius:20px;"
        f"display:inline-block;line-height:1.6'>{text}</span>"
    )


def _chart_legend(*items) -> str:
    """items: list of (label, bg, fg) tuples."""
    badges = " ".join(_pill(label, bg, fg) for label, bg, fg in items)
    return f"<div style='display:flex;gap:8px;margin-top:6px;flex-wrap:wrap'>{badges}</div>"


def render_ribbon_chart(out_df: pd.DataFrame, rmse: float, height: int = 380):
    """
    Plotly 신뢰도 리본 차트.
    RMSE를 σ 추정치로 사용해 80%(±1.28σ) / 95%(±1.96σ) 예측구간을 밴드로 표시.
    """
    dates  = [str(d.date()) for d in out_df.index]
    pred   = out_df["예측값"].tolist()
    actual = out_df["실제값"].tolist()

    sigma   = rmse
    u95 = [p + 1.96 * sigma for p in pred]
    l95 = [p - 1.96 * sigma for p in pred]
    u80 = [p + 1.28 * sigma for p in pred]
    l80 = [p - 1.28 * sigma for p in pred]

    fig = go.Figure()

    # 95% 예측구간 (연한 teal fill)
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1], y=u95 + l95[::-1],
        fill='toself', fillcolor='rgba(29,158,117,0.10)',
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip', showlegend=False,
    ))
    # 80% 예측구간 (진한 teal fill)
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1], y=u80 + l80[::-1],
        fill='toself', fillcolor='rgba(29,158,117,0.22)',
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip', showlegend=False,
    ))
    # 예측 중앙값 (파란 점선)
    fig.add_trace(go.Scatter(
        x=dates, y=pred,
        line=dict(color=CLR_BLUE, width=2, dash='dash'),
        mode='lines', name='예측값',
        showlegend=False,
        hovertemplate='%{x}<br>예측: %{y:.2f}%<extra></extra>',
    ))
    # 실제값 (teal 실선 + 점)
    fig.add_trace(go.Scatter(
        x=dates, y=actual,
        line=dict(color=CLR_TEAL, width=2),
        mode='lines+markers',
        marker=dict(size=6, color=CLR_TEAL),
        name='실제값', showlegend=False,
        hovertemplate='%{x}<br>실제: %{y:.2f}%<extra></extra>',
    ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=height,
        margin=dict(l=0, r=0, t=8, b=0),
        yaxis=dict(
            gridcolor='rgba(136,135,128,0.15)',
            tickfont=dict(color=CLR_GRAY, size=11),
            zeroline=True, zerolinecolor='rgba(136,135,128,0.3)',
        ),
        xaxis=dict(showgrid=False, tickfont=dict(color=CLR_GRAY, size=11)),
        hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        _chart_legend(
            ("예측 중앙값", BG_BLUE, "#185FA5"),
            ("80% 예측구간", BG_GREEN, "#3B6D11"),
            ("95% 예측구간", BG_TEAL, "#085041"),
            ("실제값", BG_TEAL, CLR_TEAL),
        ),
        unsafe_allow_html=True,
    )


def render_direction_headline(out_df: pd.DataFrame, value_label: str):
    """메인: 최신 예측의 방향(📈/📉)을 크게, 수익률 수치는 pill 배지로."""
    pred = float(out_df["예측값"].iloc[-1])
    date = out_df.index[-1]
    up   = pred > 0

    emoji = "📈" if up else "📉"
    label = "상승 전망" if up else "하락 전망"
    color = CLR_TEAL if up else CLR_RED
    bg    = BG_TEAL  if up else BG_RED
    badge_bg = BG_GREEN if up else BG_RED
    badge_fg = "#3B6D11" if up else "#A32D2D"

    val_badge = _pill(f"{pred:+.2f}%", badge_bg, badge_fg)
    date_badge = _pill(f"기준 {date.date()}", BG_BLUE, "#185FA5")

    st.markdown(
        f"<div style='text-align:center;padding:1.5rem;border-radius:12px;"
        f"border:0.5px solid {color}33;background:{bg};margin:0.2rem 0 0.4rem 0;'>"
        f"<div style='font-size:2.8rem;font-weight:500;color:{color};line-height:1.2'>"
        f"{emoji} {label}</div>"
        f"<div style='margin-top:10px;display:flex;gap:8px;justify-content:center'>"
        f"{val_badge}{date_badge}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _kpi(col, label: str, value: str, badge_text: str = "", badge_up: bool = True):
    """KPI 카드: 레이블(muted) → 수치(hero) → 상태 배지(pill)."""
    badge_bg = BG_GREEN if badge_up else BG_RED
    badge_fg = "#3B6D11" if badge_up else "#A32D2D"
    badge_html = (
        f"<br>{_pill(badge_text, badge_bg, badge_fg)}" if badge_text else ""
    )
    col.markdown(
        f"<div style='background:var(--secondary-background-color);"
        f"border-radius:10px;border:0.5px solid rgba(136,135,128,0.2);"
        f"padding:14px 16px'>"
        f"<div style='font-size:12px;color:{CLR_GRAY};font-weight:400'>{label}</div>"
        f"<div style='font-size:22px;font-weight:500;margin:4px 0'>{value}</div>"
        f"{badge_html}</div>",
        unsafe_allow_html=True,
    )


def render_metric_cards(metrics: dict, with_ic: bool = False):
    """성능 지표를 KPI 카드 형태로 표시."""
    c1, c2, c3, c4 = st.columns(4)
    dir_acc  = metrics["dir_acc"]
    dir_bull = metrics["dir_bull"]
    dir_bear = metrics["dir_bear"]

    _kpi(c1, "방향 정확도 (전체)",    _fmt(dir_acc, pct=True),
         "양호" if dir_acc and dir_acc >= 60 else "개선 필요",
         badge_up=bool(dir_acc and dir_acc >= 60))
    _kpi(c2, "방향 정확도 (상승 구간)", _fmt(dir_bull, pct=True),
         "Bull ✓" if dir_bull and dir_bull >= 60 else "Bull △",
         badge_up=bool(dir_bull and dir_bull >= 60))
    _kpi(c3, "방향 정확도 (하락 구간)", _fmt(dir_bear, pct=True),
         "Bear ✓" if dir_bear and dir_bear >= 60 else "Bear △",
         badge_up=bool(dir_bear and dir_bear >= 60))
    _kpi(c4, "RMSE (전체)", _fmt(metrics["rmse"]))

    c5, c6, c7, c8 = st.columns(4)
    _kpi(c5, "RMSE (상승 구간)", _fmt(metrics["rmse_bull"]))
    _kpi(c6, "RMSE (하락 구간)", _fmt(metrics["rmse_bear"]))
    _kpi(c7, "Asymmetric Loss",   _fmt(metrics["asym_loss"]))
    if with_ic:
        ic = metrics.get("ic")
        _kpi(c8, "IC (Spearman)", _fmt(ic),
             "상관 있음" if ic and ic > 0.2 else "약한 상관",
             badge_up=bool(ic and ic > 0.2))
    else:
        _kpi(c8, "Hold-out 구간", f"{metrics['n_holdout']}개")


def render_confusion(df: pd.DataFrame):
    """상승/하락 방향 혼동행렬."""
    yt, yp = df["실제값"].values, df["예측값"].values
    tp = int(((yt > 0) & (yp > 0)).sum())
    fp = int(((yt <= 0) & (yp > 0)).sum())
    fn = int(((yt > 0) & (yp <= 0)).sum())
    tn = int(((yt <= 0) & (yp <= 0)).sum())
    cm = pd.DataFrame(
        [[tp, fn], [fp, tn]],
        index=["실제 상승", "실제 하락"],
        columns=["예측 상승", "예측 하락"],
    )
    st.dataframe(cm.style.background_gradient(cmap="Blues"), width='stretch')


def render_shap_section(cfg: dict):
    """알파 ①: SHAP 피처 중요도 (상위 10) — Plotly 수평 막대."""
    st.markdown("#### 🔬 SHAP 피처 중요도 (상위 10)")
    st.caption("shap.TreeExplainer 기반 · 평균 |SHAP| 값이 클수록 예측 기여도가 큰 피처")
    try:
        shap_df = compute_shap_importance(cfg["model_path"], cfg["features_path"], cfg["target"])
        fig = go.Figure(go.Bar(
            x=shap_df["평균 |SHAP|"].tolist(),
            y=shap_df.index.tolist(),
            orientation='h',
            marker_color=CLR_BLUE,
            marker_line_width=0,
        ))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=360,
            margin=dict(l=0, r=0, t=8, b=0),
            yaxis=dict(
                autorange='reversed',
                tickfont=dict(color=CLR_GRAY, size=11),
                gridcolor='rgba(136,135,128,0.15)',
            ),
            xaxis=dict(showgrid=False, tickfont=dict(color=CLR_GRAY, size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"SHAP 계산을 수행하지 못했습니다: {e}")


# ──────────────────────────────────────────────────────────────────
# 4b. 전문가 모드 헬퍼
# ──────────────────────────────────────────────────────────────────

def _inject_styles():
    st.markdown("""
    <style>
    .signal-row {
        display:flex;align-items:center;justify-content:space-between;
        padding:10px 0;border-bottom:0.5px solid rgba(136,135,128,0.2);
    }
    .signal-row:last-child{border-bottom:none;}
    .signal-label{font-size:13px;color:#888780;}
    .signal-val  {font-size:13px;font-weight:500;}
    .signal-val.up {color:#1D9E75;}
    .signal-val.dn {color:#E24B4A;}
    .signal-val.neu{color:#EF9F27;}
    .cb-label{display:flex;justify-content:space-between;font-size:12px;color:#888780;margin-bottom:5px;}
    .cb-track{height:8px;border-radius:4px;background:rgba(136,135,128,0.2);overflow:hidden;}
    .cb-fill {height:100%;border-radius:4px;}
    .caution-box{background:#faeeda;border-radius:10px;padding:14px;margin-top:12px;}
    .c-title{font-size:12px;font-weight:500;color:#854F0B;margin-bottom:6px;}
    .c-body {font-size:12px;color:#633806;line-height:1.7;}
    .expert-banner{background:#e6f1fb;border-radius:10px;padding:12px 14px;
        display:flex;align-items:flex-start;gap:10px;margin-bottom:16px;}
    .eb-title{font-size:13px;font-weight:500;color:#0C447C;}
    .eb-body {font-size:12px;color:#185FA5;line-height:1.6;margin-top:2px;}
    </style>
    """, unsafe_allow_html=True)


def _expert_banner():
    st.markdown(
        "<div class='expert-banner'>"
        "<span style='font-size:20px'>🔬</span>"
        "<div><div class='eb-title'>전문가 모드 켜짐</div>"
        "<div class='eb-body'>판단 근거, 모델 수치, 주의사항을 상세하게 보여줘요.</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def _show(simple: str, expert: str, expert_mode: bool):
    """일반/전문가 텍스트 전환."""
    st.markdown(expert if expert_mode else simple, unsafe_allow_html=True)


def _confidence_bar(pct: float, label: str, color: str = "#2a78d6"):
    pct = min(max(pct, 0), 100)
    st.markdown(
        f"<div class='cb-label'><span>{label}</span>"
        f"<span style='color:{color};font-weight:500'>{pct:.0f}%</span></div>"
        f"<div class='cb-track'>"
        f"<div class='cb-fill' style='width:{pct}%;background:{color}'></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _signal_rows(rows: list):
    """rows: [(label, value, direction)] — direction: 'up'|'dn'|'neu'"""
    html = "".join(
        f"<div class='signal-row'>"
        f"<span class='signal-label'>{lbl}</span>"
        f"<span class='signal-val {d}'>{val}</span>"
        f"</div>"
        for lbl, val, d in rows
    )
    st.markdown(html, unsafe_allow_html=True)


def _caution_box(text: str):
    st.markdown(
        f"<div class='caution-box'>"
        f"<div class='c-title'>⚠️ 주의사항</div>"
        f"<div class='c-body'>{text}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_detail_sections(metrics: dict, out_df: pd.DataFrame, stage: str, expert_mode: bool):
    """상세 분석 섹션: 한 줄 요약 + 아코디언 + 시그널 + 신뢰도 바."""
    pred_last = float(out_df["예측값"].iloc[-1])
    up = pred_last > 0
    dir_label = "상승" if up else "하락"
    dir_emoji = "📈" if up else "📉"
    dir_acc  = metrics.get("dir_acc") or 0
    bear_acc = metrics.get("dir_bear") or 0
    ic       = metrics.get("ic")

    st.divider()

    # ── 📊 한 줄 요약 ──
    st.subheader("📊 한 줄 요약")
    if stage == "stage1":
        _show(
            simple=(f"반도체 출하량이 **{dir_label} 방향**을 가리키고 있어요 {dir_emoji} "
                    "아시아-태평양 매출 모멘텀과 SOX 흐름이 주요 신호입니다."),
            expert=(f"Stage 1 XGBoost 최신 예측: **{pred_last:+.2f}%** (WW YoY 6개월 선행). "
                    f"CV 방향정확도 **{dir_acc:.1f}%** (Bear **{bear_acc:.1f}%**), "
                    f"AsymLoss **{metrics['asym_loss']:.2f}**. "
                    f"핵심 피처: `FRED_NewOrder_YoY_lag12`, `Ret_SOX_ma6`, `Asia_Pacific_YoY`. "
                    f"Hold-out RMSE: **{metrics['rmse']:.2f}**"),
            expert_mode=expert_mode,
        )
    else:
        ic_str = f", IC(Spearman) **{ic:.2f}**" if ic else ""
        _show(
            simple=(f"SK하이닉스 6개월 수익률이 **{dir_label}** 전망이에요 {dir_emoji} "
                    "HBM 수요와 반도체 사이클 신호를 종합한 결과입니다."),
            expert=(f"Stage 2 XGBoost 최신 예측: **{pred_last:+.2f}%** (6개월 수익률). "
                    f"CV 방향정확도 **{dir_acc:.1f}%** (Bear **{bear_acc:.1f}%**){ic_str}. "
                    f"Asymmetric Loss 패널티 적용 (Bear 오예측 가중치 3.0). "
                    f"Hold-out RMSE: **{metrics['rmse']:.2f}**"),
            expert_mode=expert_mode,
        )

    st.divider()

    # ── 🔍 세부 분석 (아코디언) ──
    st.subheader("🔍 세부 분석")

    if stage == "stage1":
        with st.expander("🔵  반도체 사이클 신호", expanded=True):
            _show(
                simple=("전 세계 반도체 매출 YoY 흐름이 **회복 국면**에 있어요. "
                        "특히 일본·아태 지역이 선행 지표 역할을 하고 있습니다 🌏"),
                expert=("핵심 피처: `Asia_Pacific_YoY`, `Japan_YoY_lag12`, `InvSales_diff6`. "
                        "재고/매출 비율(ISRATIO) 6개월 변화량 반전 시 사이클 저점 확인 가능. "
                        "`Worldwide_YoY_vol6` 변동성 확대 → 전환점 인근으로 해석."),
                expert_mode=expert_mode,
            )
            if expert_mode:
                _signal_rows([
                    ("아태 반도체 매출 YoY",     "회복 국면",  "up"),
                    ("일본 매출 YoY (12M lag)",  "선행 신호",  "up"),
                    ("재고/매출 비율 (6M diff)", "방향 전환",  "neu"),
                    ("WW YoY 변동성 (6M)",       "확대 중",    "neu"),
                ])

        with st.expander("📈  주가·거시 시그널"):
            _show(
                simple=("SOX(필라델피아 반도체 지수)와 NVDA 모멘텀이 좋아요 💪 "
                        "연준 금리 방향도 반도체 업황에 영향을 줍니다."),
                expert=("`Ret_SOX_ma6`·`Ret_SOX_ma3` 모멘텀이 상위 피처 선정. "
                        "`Ret_NVDA_lag6` (6개월 전 NVDA 수익률) 포함. "
                        "`FRED_NewOrder_YoY_lag12`가 12개월 선행으로 1위 기여. "
                        "`FedFunds_diff6`·`T10Y3M_chg6` 금리 피처 포함."),
                expert_mode=expert_mode,
            )
            if expert_mode:
                _signal_rows([
                    ("SOX 6M 이동평균 수익률",   "상승 우세",   "up"),
                    ("NVDA 6M 전 수익률",         "모멘텀 유지", "up"),
                    ("신규 제조업 수주 YoY lag12", "양호",        "up"),
                    ("10Y-3M 금리차 6M 변화",     "경계",        "neu"),
                ])

        with st.expander("⚠️  리스크 요인"):
            _show(
                simple=("모델이 못 보는 지정학·규제 리스크가 있어요. "
                        "1~2개월 내 급변 상황은 반영이 어려울 수 있습니다 🙏"),
                expert=(f"Hold-out(최근 24개월) RMSE **{metrics['rmse']:.2f}** — CV 대비 높음. "
                        f"Bear 구간 DirAcc **{bear_acc:.1f}%** (Bull 대비 저하 가능). "
                        "FRED_ISM_Mfg 시리즈 수집 실패 → 해당 거시 신호 누락."),
                expert_mode=expert_mode,
            )
            if expert_mode:
                _caution_box(
                    "이 예측은 과거 데이터 패턴 기반의 통계 모델 출력값입니다. "
                    "규제 리스크, 지정학적 이벤트, 기업 내부 정보 등 구조적 변화는 반영되지 않습니다. "
                    "투자 결정 시 이 수치만 단독으로 활용하지 마세요."
                )

    else:  # stage2
        avg_pred = float(out_df["예측값"].mean())
        with st.expander("💹  HBM·DRAM 수요 신호", expanded=True):
            _show(
                simple=("HBM 수요가 AI 서버 확장과 함께 강하게 유지되고 있어요 💪 "
                        "DRAM 일반 제품은 다소 약세지만 HBM이 상쇄하는 구조예요."),
                expert=("Stage 1 예측값(`v2_pred_ww_yoy`)이 Stage 2의 핵심 Bridge 피처. "
                        "TSMC 월별 매출로 추정한 AI 서버 빌드아웃 속도가 HBM 수요 proxy. "
                        "DRAM 스팟-계약가 스프레드 약세 신호 있으나 HBM ASP 방어로 상쇄 중."),
                expert_mode=expert_mode,
            )
            if expert_mode:
                avg_d = "up" if avg_pred > 0 else "dn"
                _signal_rows([
                    ("Stage 1 Bridge 예측 (평균)", f"{avg_pred:+.1f}%",  avg_d),
                    ("DRAM ASP (일반)",             "약세 신호",           "dn"),
                    ("HBM 수요 (AI 서버)",          "강세 유지",           "up"),
                    ("SOX 3개월 모멘텀",             "실시간 확인 필요",    "neu"),
                ])

        with st.expander("📊  수익률 예측 근거"):
            ic_str2 = f"IC(Spearman) **{ic:.2f}**, " if ic else ""
            _show(
                simple=(f"SK하이닉스 6개월 수익률 **{dir_label}** 신호예요. "
                        "반도체 사이클과 시장 모멘텀을 함께 보는 2단계 모델 결과입니다."),
                expert=(f"XGBoost 회귀 → 방향성 판단. {ic_str2}"
                        f"CV DirAcc **{dir_acc:.1f}%** (Bear **{bear_acc:.1f}%**). "
                        "Asymmetric Loss 목적함수: Bear 오예측 가중치 3.0 적용."),
                expert_mode=expert_mode,
            )
            if expert_mode:
                bar_c1 = CLR_TEAL if dir_acc >= 70 else CLR_AMBER
                bar_c2 = CLR_TEAL if bear_acc >= 60 else CLR_AMBER
                _confidence_bar(dir_acc,  "방향 정확도 (CV)",      bar_c1)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                _confidence_bar(bear_acc, "Bear 구간 정확도 (CV)", bar_c2)

        with st.expander("⚠️  리스크 요인"):
            _show(
                simple=("6개월 예측이라 시장 환경이 많이 바뀔 수 있어요 ⏳ "
                        "외부 충격(규제, 지정학)은 모델이 반영 못해요."),
                expert=(f"Hold-out({metrics['n_holdout']}분기) RMSE **{metrics['rmse']:.1f}** — 변동성 높음. "
                        "6개월 타겟 특성상 단기 외부 충격에 취약. "
                        "Bear 구간 샘플 수 부족으로 통계적 신뢰도 제한적."),
                expert_mode=expert_mode,
            )
            if expert_mode:
                _caution_box(
                    "이 예측은 과거 데이터 패턴 기반의 통계 모델 출력값입니다. "
                    "규제 리스크, 지정학적 이벤트, 기업 내부 정보 등 구조적 변화는 반영되지 않습니다. "
                    "투자 결정 시 이 수치만 단독으로 활용하지 마세요."
                )

    # ── 🎯 모델 신뢰도 바 ──
    st.divider()
    st.subheader("🎯 모델 신뢰도")
    bar_color = CLR_TEAL if dir_acc >= 75 else (CLR_AMBER if dir_acc >= 60 else CLR_RED)
    _confidence_bar(dir_acc, "방향 예측 정확도 (CV 평균)", bar_color)
    if expert_mode:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        bar_c2 = CLR_TEAL if bear_acc >= 60 else CLR_AMBER
        _confidence_bar(bear_acc, "Bear 구간 정확도 (CV)", bar_c2)
        suffix = "분기" if stage == "stage2" else "개월"
        parts = [f"Hold-out: {metrics['n_holdout']}{suffix}"]
        if ic:
            parts.append(f"IC(Spearman): {ic:.3f}")
        parts.append(f"AsymLoss: {metrics['asym_loss']:.2f}")
        st.caption("  ·  ".join(parts))


def render_hit_history(out_df: pd.DataFrame, freq_label: str):
    """알파 ②: 과거 분기/월별 예측 적중 히스토리 (✅/❌ 타임라인 + 상세표)."""
    st.markdown("#### 🎯 과거 적중 히스토리 (Hold-out)")
    d = out_df.copy()
    d["적중"] = (d["실제값"] > 0) == (d["예측값"] > 0)
    n, hit = len(d), int(d["적중"].sum())
    acc = d["적중"].mean() * 100 if n else 0.0

    cA, cB = st.columns([1, 3])
    with cA:
        st.metric("적중률", f"{acc:.1f}%", f"{hit}/{n} {freq_label} 적중")
    with cB:
        timeline = " ".join("✅" if v else "❌" for v in d["적중"])
        st.markdown("**적중 타임라인** (왼쪽=과거 → 오른쪽=최근)")
        st.markdown(f"<div style='font-size:1.5rem;letter-spacing:2px'>{timeline}</div>",
                    unsafe_allow_html=True)

    table = pd.DataFrame({
        "예측 방향": ["📈 상승" if v > 0 else "📉 하락" for v in d["예측값"]],
        "실제 방향": ["📈 상승" if v > 0 else "📉 하락" for v in d["실제값"]],
        "예측값": [f"{v:+.2f}%" for v in d["예측값"]],
        "실제값": [f"{v:+.2f}%" for v in d["실제값"]],
        "결과": ["✅" if v else "❌" for v in d["적중"]],
    }, index=d.index.strftime("%Y-%m"))
    with st.expander("적중 히스토리 상세"):
        st.dataframe(table, width='stretch')


# ──────────────────────────────────────────────────────────────────
# 5. Stage별 화면
# ──────────────────────────────────────────────────────────────────

def view_stage1():
    cfg = STAGE1
    st.header("📦 Stage 1 — 반도체 출하량(WW 매출) YoY 예측")
    st.caption(
        "전 세계 반도체 매출의 6개월 선행 전년동월대비(YoY) 증감률을 XGBoost로 예측합니다. "
        "Bear(하락) 구간 오예측에 더 큰 페널티를 주도록 학습되었습니다."
    )

    try:
        metrics, df = evaluate_stage(
            cfg["features_path"], cfg["model_path"], cfg["target"], cfg["test_eval"]
        )
    except Exception as e:
        st.error(f"Stage 1 평가 중 오류가 발생했습니다: {e}")
        return

    # ── 메인: 방향 예측 크게 ──
    render_direction_headline(df, cfg["value_label"])

    st.subheader("모델 성능 지표 (Hold-out)")
    st.caption(f"평가 구간: {metrics['period']}  ·  선택 피처 {metrics['n_features']}개")
    render_metric_cards(metrics)

    st.subheader("예측 vs 실제 — Hold-out 타임라인")
    render_ribbon_chart(df, metrics["rmse"])

    with st.expander("Hold-out 예측 상세 데이터"):
        st.dataframe(df.style.format("{:.2f}"), width='stretch')

    # ── 전문가 모드 상세 분석 ──
    expert_mode = st.session_state.get("expert_mode", False)
    render_detail_sections(metrics, df, "stage1", expert_mode)

    # ── 알파 섹션 ──
    st.divider()
    st.subheader("✨ 알파 — 모델 해석 & 적중 히스토리")
    render_shap_section(cfg)
    render_hit_history(df, cfg["freq_label"])


def view_stage2():
    cfg = STAGE2
    st.header("📈 Stage 2 — SK하이닉스 6개월 수익률 방향 예측")
    st.caption(
        "Stage 1의 반도체 사이클 예측을 입력 피처로 활용해 "
        "SK하이닉스 6개월 종가 수익률(방향)을 예측합니다."
    )

    try:
        metrics, df = evaluate_stage(
            cfg["features_path"], cfg["model_path"], cfg["target"],
            cfg["test_eval"], with_ic=True
        )
    except Exception as e:
        st.error(f"Stage 2 평가 중 오류가 발생했습니다: {e}")
        return

    # ── 메인: 방향 예측 크게 ──
    render_direction_headline(df, cfg["value_label"])

    st.subheader("모델 성능 지표 (Hold-out)")
    st.caption(f"평가 구간: {metrics['period']}  ·  사용 피처 {metrics['n_features']}개")
    render_metric_cards(metrics, with_ic=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("예측 vs 실제 수익률 — Hold-out")
        render_ribbon_chart(df, metrics["rmse"], height=360)
    with col_b:
        st.subheader("방향 예측 혼동행렬")
        render_confusion(df)

    with st.expander("Hold-out 예측 상세 데이터"):
        st.dataframe(df.style.format("{:.2f}"), width='stretch')

    # ── 전문가 모드 상세 분석 ──
    expert_mode = st.session_state.get("expert_mode", False)
    render_detail_sections(metrics, df, "stage2", expert_mode)

    # ── 알파 섹션 ──
    st.divider()
    st.subheader("✨ 알파 — 모델 해석 & 적중 히스토리")
    render_shap_section(cfg)
    render_hit_history(df, cfg["freq_label"])


def _flow_box(title: str, subtitle: str, code: str):
    """E2E 흐름 다이어그램의 한 칸 (테두리 컨테이너)."""
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.markdown(f"**{subtitle}**")
        st.caption(f"`{code}`")


def _flow_arrow():
    st.markdown(
        "<div style='text-align:center;font-size:2.6rem;margin-top:1.6rem'>➡️</div>",
        unsafe_allow_html=True,
    )


def render_market_signals():
    """E2E ④: 현재 시장 신호 요약 카드 (KOSPI / SOX 모멘텀 + 종합 신호)."""
    st.subheader("④ 현재 시장 신호 요약")
    st.caption("실시간 시장 모멘텀과 모델 최신 예측으로 구성한 현재 종합 신호")

    mom = get_market_momentum()
    kospi_mom = mom.get("KOSPI")
    sox_mom   = mom.get("SOX")

    # 모델의 현재 방향 신호 (Stage 2 최신 예측)
    model_up = None
    try:
        _, out2 = evaluate_stage(
            STAGE2["features_path"], STAGE2["model_path"],
            STAGE2["target"], STAGE2["test_eval"], with_ic=True
        )
        model_up = bool(out2["예측값"].iloc[-1] > 0)
    except Exception:
        pass

    def _mom_str(v):
        return "N/A" if v is None else f"{v:+.1f}%"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🇰🇷 KOSPI 모멘텀 (3M)", _mom_str(kospi_mom),
                  delta=None if kospi_mom is None else f"{kospi_mom:+.1f}%")
    with c2:
        st.metric("💽 SOX 모멘텀 (3M)", _mom_str(sox_mom),
                  delta=None if sox_mom is None else f"{sox_mom:+.1f}%")

    # ── 종합 신호: 가용 신호의 방향을 다수결 집계 ──
    votes = []
    if kospi_mom is not None:
        votes.append(kospi_mom > 0)
    if sox_mom is not None:
        votes.append(sox_mom > 0)
    if model_up is not None:
        votes.append(model_up)

    with c3:
        if not votes:
            st.metric("🧭 종합 신호", "N/A")
        else:
            pos = sum(votes)
            if pos > len(votes) / 2:
                st.metric("🧭 종합 신호", "📈 상승 우세", delta=f"{pos}/{len(votes)} 신호 상승")
            elif pos < len(votes) / 2:
                st.metric("🧭 종합 신호", "📉 하락 우세", delta=f"-{len(votes)-pos}/{len(votes)} 신호 하락")
            else:
                st.metric("🧭 종합 신호", "➖ 중립")

    st.caption(
        "※ 종합 신호 = KOSPI·SOX 3개월 모멘텀 방향 + Stage 2 모델 최신 예측 방향의 다수결. "
        "KOSPI/SOX는 yfinance 실시간(최대 1시간 캐시), 모델 신호는 hold-out 최신값 기준."
    )


def view_e2e():
    st.header("🔗 E2E — Stage 1 → Stage 2 파이프라인 흐름")
    st.caption(
        "Stage 1이 예측한 반도체 사이클 신호가 Stage 2의 입력 피처로 흘러들어가는 "
        "End-to-End 구조를 보여줍니다."
    )

    # ── 흐름 다이어그램 (테두리 컨테이너 + 화살표) ──
    f1, fa, f2, fb, f3 = st.columns([4, 1, 4, 1, 4])
    with f1:
        _flow_box("🏭 Stage 1", "반도체 매출 YoY 예측", "best_xgboost_final.pkl")
    with fa:
        _flow_arrow()
    with f2:
        _flow_box("🔌 Bridge 피처", "Stage 1 예측값 전달", BRIDGE_COL)
    with fb:
        _flow_arrow()
    with f3:
        _flow_box("💹 Stage 2", "SK하이닉스 수익률 예측", "skh_xgb_final.pkl")

    st.divider()

    # ── ① Bridge: Stage1 예측값 시계열 ──
    st.subheader("① Stage 1 출력 — Expanding Window 예측 시계열")
    st.caption(
        f"각 관찰일 시점에서 lookahead 없이 재학습해 생성한 6개월 선행 "
        f"반도체 매출 YoY 예측값(`{BRIDGE_COL}`)입니다."
    )
    try:
        s1pred = load_csv(STAGE1_PRED_PATH)
        if BRIDGE_COL in s1pred.columns:
            s = s1pred[[BRIDGE_COL]].dropna()
            dates_s = [str(d.date()) for d in s.index]
            vals_s  = s[BRIDGE_COL].tolist()
            fig_s1 = go.Figure(go.Scatter(
                x=dates_s, y=vals_s,
                line=dict(color=CLR_BLUE, width=2, dash='dash'),
                mode='lines',
                hovertemplate='%{x}<br>예측 YoY: %{y:.2f}%<extra></extra>',
            ))
            fig_s1.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                height=300,
                margin=dict(l=0, r=0, t=8, b=0),
                yaxis=dict(gridcolor='rgba(136,135,128,0.15)',
                           tickfont=dict(color=CLR_GRAY, size=11),
                           zeroline=True, zerolinecolor='rgba(136,135,128,0.3)'),
                xaxis=dict(showgrid=False, tickfont=dict(color=CLR_GRAY, size=11)),
            )
            st.plotly_chart(fig_s1, use_container_width=True)
            st.markdown(_chart_legend(("Stage 1 예측 YoY", BG_BLUE, "#185FA5")),
                        unsafe_allow_html=True)
        else:
            st.warning(f"`{BRIDGE_COL}` 컬럼을 찾을 수 없습니다.")
            st.dataframe(s1pred.head(), width='stretch')
    except Exception as e:
        st.error(f"Stage 1 예측 데이터 로드 실패: {e}")

    st.divider()

    # ── ② 연결 검증 ──
    st.subheader("② Stage 2 입력 — Bridge 피처 결합 확인")
    try:
        s2feat = load_csv(STAGE2["features_path"])
        if BRIDGE_COL in s2feat.columns:
            st.success(
                f"✅ Stage 2 피처셋에 Stage 1 예측 피처 `{BRIDGE_COL}`가 포함되어 있습니다. "
                "두 단계가 정상적으로 연결되었습니다."
            )
            n_total = s2feat.shape[1]
            n_bridge = sum(1 for c in s2feat.columns if c.startswith("v2_pred"))
            m1, m2 = st.columns(2)
            m1.metric("Stage 2 전체 피처 수", f"{n_total}개")
            m2.metric("Stage 1 유래 Bridge 피처", f"{n_bridge}개")
        else:
            st.warning(f"Stage 2 피처셋에서 `{BRIDGE_COL}`를 찾지 못했습니다.")
    except Exception as e:
        st.error(f"Stage 2 피처 데이터 로드 실패: {e}")

    st.divider()

    # ── ③ 두 단계 성능 요약 ──
    st.subheader("③ 두 단계 성능 요약")
    rows = []
    for cfg, with_ic in [(STAGE1, False), (STAGE2, True)]:
        try:
            m, _ = evaluate_stage(
                cfg["features_path"], cfg["model_path"], cfg["target"],
                cfg["test_eval"], with_ic=with_ic
            )
            rows.append({
                "단계": f"{cfg['name']} · {cfg['title']}",
                "방향정확도(전체)": _fmt(m["dir_acc"], pct=True),
                "방향정확도(Bear)": _fmt(m["dir_bear"], pct=True),
                "RMSE": _fmt(m["rmse"]),
                "Asym Loss": _fmt(m["asym_loss"]),
                "IC": _fmt(m.get("ic")) if with_ic else "—",
            })
        except Exception as e:
            rows.append({"단계": cfg["name"], "방향정확도(전체)": f"오류: {e}"})
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    st.divider()

    # ── ④ 현재 시장 신호 요약 ──
    render_market_signals()


# ──────────────────────────────────────────────────────────────────
# 6. 메인
# ──────────────────────────────────────────────────────────────────

def main():
    # CSS 주입 (페이지 최초 1회)
    _inject_styles()

    # 산출물 확보 (실패 시 내부에서 st.stop())
    guard_artifacts()

    st.sidebar.title("📊 대시보드")
    st.sidebar.caption("반도체 사이클 → SK하이닉스 수익률 예측")
    stage = st.sidebar.radio(
        "보기 선택",
        ["E2E 전체", "Stage 1", "Stage 2"],
        index=0,
    )
    st.sidebar.divider()
    st.sidebar.toggle("🔬 전문가 모드", key="expert_mode", value=False)
    st.sidebar.caption("켜면 모델 수치·근거·리스크를 상세히 표시합니다.")
    st.sidebar.divider()
    st.sidebar.markdown(
        "**파이프라인 개요**\n\n"
        "1. **Stage 1** — 반도체 매출 YoY(6M 선행) 예측\n"
        "2. **Bridge** — 예측값을 Stage 2 피처로 전달\n"
        "3. **Stage 2** — SK하이닉스 6M 수익률 방향 예측"
    )

    st.title("반도체 사이클 기반 SK하이닉스 수익률 예측")

    # 전문가 모드 배너
    if st.session_state.get("expert_mode", False):
        _expert_banner()

    if stage == "Stage 1":
        view_stage1()
    elif stage == "Stage 2":
        view_stage2()
    else:
        view_e2e()


if __name__ == "__main__":
    main()

