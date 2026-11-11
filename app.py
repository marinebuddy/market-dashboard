import streamlit as st
import yfinance as yf

# 1) 페이지 기본 설정: 브라우저 탭 제목과 화면 너비
st.set_page_config(page_title="마켓 대시보드", layout="wide")
st.title("마켓 대시보드")
st.caption("코스피 · 코스닥 · 나스닥 종가 흐름")

# 2) 보여줄 지수 목록: "화면에 쓸 이름": "야후 파이낸스 종목 코드"
INDICES = {
    "코스피": "^KS11",
    "코스닥": "^KQ11",
    "나스닥": "^IXIC",
}

# 3) 왼쪽 사이드바에 기간 선택 상자 만들기
PERIODS = {"1개월": "1mo", "3개월": "3mo", "6개월": "6mo", "1년": "1y"}
period_name = st.sidebar.selectbox("기간", list(PERIODS), index=1)


# 4) 데이터 받아 오기. 같은 요청은 10분(600초) 동안 저장해 두고 다시 씀
@st.cache_data(ttl=600)
def load_close(ticker, period):
    data = yf.Ticker(ticker).history(period=period)
    return data["Close"].dropna()


# 5) 지수 개수만큼 칸을 나누고, 칸마다 숫자 카드와 차트를 그림
columns = st.columns(len(INDICES))
for column, (name, ticker) in zip(columns, INDICES.items()):
    with column:
        close = load_close(ticker, PERIODS[period_name])
        if len(close) < 2:
            st.warning(f"{name} 데이터를 불러오지 못했습니다.")
            continue
        last = close.iloc[-1]   # 가장 최근 종가
        prev = close.iloc[-2]   # 그 전날 종가
        change = (last / prev - 1) * 100
        # delta_color="inverse": 한국식으로 상승은 빨강, 하락은 파랑 계열
        st.metric(name, f"{last:,.2f}", f"{change:+.2f}%", delta_color="inverse")
        st.line_chart(close)
