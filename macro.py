# ============================================================
# 유가·환율 / 미국 국채금리 / CNN 피어앤그리드 화면
# app.py가 이 파일의 show_... 함수를 불러서 화면을 그림
# ============================================================

from io import StringIO

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

PERIODS = {"1개월": "1mo", "3개월": "3mo", "6개월": "6mo", "1년": "1y", "2년": "2y"}
PERIOD_MONTHS = {"1개월": 1, "3개월": 3, "6개월": 6, "1년": 12, "2년": 24}


# ------------------------------------------------------------
# 공통: 야후 파이낸스 종가 받아 오기 (10분 저장)
# ------------------------------------------------------------
@st.cache_data(ttl=600)
def load_close(ticker, period):
    data = yf.Ticker(ticker).history(period=period)
    return data["Close"].dropna()


# ============================================================
# 1) 유가·환율
# ============================================================

# "화면 이름": ("야후 종목 코드", "단위", 소수점 자리수)
OIL_FX = {
    "WTI유 선물": ("CL=F", "달러/배럴", 2),
    "달러/원": ("KRW=X", "원", 2),
    "달러/엔": ("JPY=X", "엔", 2),
}


def show_oil_fx():
    st.title("유가·환율")
    st.caption("WTI유 선물 · 달러/원 · 달러/엔 (야후 파이낸스 기준)")
    period_name = st.sidebar.selectbox("기간", list(PERIODS), index=1, key="oilfx_period")

    columns = st.columns(len(OIL_FX))
    for column, (name, (ticker, unit, digits)) in zip(columns, OIL_FX.items()):
        with column:
            close = load_close(ticker, PERIODS[period_name])
            if len(close) < 2:
                st.warning(f"{name} 데이터를 불러오지 못했습니다.")
                continue
            last = close.iloc[-1]
            prev = close.iloc[-2]
            change = (last / prev - 1) * 100
            st.metric(
                f"{name} ({unit})",
                f"{last:,.{digits}f}",
                f"{change:+.2f}%",
                delta_color="inverse",
            )
            st.line_chart(close)


# ============================================================
# 2) 미국 국채금리
#    1순위: 미국 연준 데이터베이스 FRED (DGS10, DGS2)
#    FRED가 안 되면: 야후 파이낸스 (^TNX, 2YY=F)
# ============================================================

@st.cache_data(ttl=3600)
def load_fred(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    table = pd.read_csv(StringIO(response.text))
    table.columns = ["date", "value"]
    table["date"] = pd.to_datetime(table["date"])
    table["value"] = pd.to_numeric(table["value"], errors="coerce")  # 휴일의 "." 은 빈값 처리
    return table.set_index("date")["value"].dropna()


def load_yields():
    """10년물과 2년물 금리를 표로 돌려줌. (표, 출처 이름)"""
    try:
        yields = pd.DataFrame({"10년물": load_fred("DGS10"), "2년물": load_fred("DGS2")})
        source = "FRED (미국 세인트루이스 연준)"
    except Exception:
        yields = pd.DataFrame(
            {"10년물": load_close("^TNX", "2y"), "2년물": load_close("2YY=F", "2y")}
        )
        yields.index = yields.index.tz_localize(None).normalize()
        source = "야후 파이낸스 (2년물은 금리 선물 기준)"
    return yields.dropna(), source


def show_yields():
    st.title("미국 국채금리")
    period_name = st.sidebar.selectbox("기간", list(PERIODS), index=3, key="yield_period")

    try:
        yields, source = load_yields()
    except Exception as error:
        st.error(f"금리 데이터를 불러오지 못했습니다: {error}")
        return
    if len(yields) < 2:
        st.warning("금리 데이터를 불러오지 못했습니다.")
        return

    yields["10년-2년"] = yields["10년물"] - yields["2년물"]
    latest = yields.iloc[-1]
    prev = yields.iloc[-2]
    st.caption(f"출처: {source} · 기준일 {yields.index[-1].strftime('%Y-%m-%d')}")

    # 금리 변화는 bp(0.01%p) 단위로 표시
    c1, c2, c3 = st.columns(3)
    c1.metric("10년물", f"{latest['10년물']:.2f}%",
              f"{(latest['10년물'] - prev['10년물']) * 100:+.1f}bp", delta_color="inverse")
    c2.metric("2년물", f"{latest['2년물']:.2f}%",
              f"{(latest['2년물'] - prev['2년물']) * 100:+.1f}bp", delta_color="inverse")
    c3.metric("10년-2년 금리차", f"{latest['10년-2년']:+.2f}%p",
              f"{(latest['10년-2년'] - prev['10년-2년']) * 100:+.1f}bp", delta_color="off")

    cutoff = yields.index.max() - pd.DateOffset(months=PERIOD_MONTHS[period_name])
    recent = yields[yields.index >= cutoff]

    st.subheader("10년물 · 2년물 금리 (%)")
    st.line_chart(recent[["10년물", "2년물"]])
    st.subheader("10년-2년 금리차 (%p)")
    st.caption("0 아래로 내려가면 장단기 금리 역전입니다.")
    st.line_chart(recent["10년-2년"])


# ============================================================
# 3) CNN 피어앤그리드
#    CNN은 공식 API를 제공하지 않음. CNN 웹페이지가 쓰는 데이터 주소를
#    읽어 오는 방식이라, CNN 쪽 사정에 따라 안 열릴 수 있음.
# ============================================================

CNN_PAGE = "https://edition.cnn.com/markets/fear-and-greed"
CNN_DATA = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

RATING_KR = {
    "extreme fear": "극단적 공포",
    "fear": "공포",
    "neutral": "중립",
    "greed": "탐욕",
    "extreme greed": "극단적 탐욕",
}


@st.cache_data(ttl=1800)
def load_cnn_fear_greed():
    response = requests.get(CNN_DATA, timeout=15, headers={"Accept": "application/json"})
    response.raise_for_status()
    return response.json()


def show_cnn_fear_greed():
    st.title("CNN 피어앤그리드")
    st.caption("CNN Business Fear & Greed Index · 0은 극단적 공포, 100은 극단적 탐욕")

    try:
        raw = load_cnn_fear_greed()
        now = raw["fear_and_greed"]
        history = pd.DataFrame(raw["fear_and_greed_historical"]["data"])
        history["date"] = pd.to_datetime(history["x"], unit="ms")
        series = history.set_index("date")["y"].rename("Fear & Greed")
    except Exception as error:
        st.warning(
            "CNN 데이터를 불러오지 못했습니다. CNN이 자동 접속을 막았거나 "
            "데이터 주소·형식이 바뀌었을 수 있습니다."
        )
        st.caption(f"오류 내용: {error}")
        st.link_button("CNN 사이트에서 직접 보기", CNN_PAGE)
        return

    rating = RATING_KR.get(str(now.get("rating", "")).lower(), now.get("rating", ""))
    score = float(now["score"])

    # 현재 값과, 과거 시점 대비 변화
    labels = {
        "previous_close": "전일",
        "previous_1_week": "1주 전",
        "previous_1_month": "1개월 전",
        "previous_1_year": "1년 전",
    }
    columns = st.columns(len(labels) + 1)
    columns[0].metric("현재", f"{score:.0f}", rating, delta_color="off")
    for column, (key, label) in zip(columns[1:], labels.items()):
        if key in now:
            past = float(now[key])
            column.metric(label, f"{past:.0f}", f"현재와 차이 {score - past:+.0f}", delta_color="off")

    months = st.sidebar.selectbox(
        "기간", [3, 6, 12], index=2, format_func=lambda m: f"최근 {m}개월", key="cnn_period"
    )
    cutoff = series.index.max() - pd.DateOffset(months=months)
    st.line_chart(series[series.index >= cutoff])
    st.caption("25 이하 극단적 공포 · 45 이하 공포 · 55 이하 중립 · 75 이하 탐욕 · 그 위 극단적 탐욕")
    st.link_button("CNN 원본 페이지 열기", CNN_PAGE)
