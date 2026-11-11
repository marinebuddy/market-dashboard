import streamlit as st
import yfinance as yf

import fear_greed  # 같은 폴더의 fear_greed.py 파일을 불러옴

# 1) 페이지 기본 설정: 브라우저 탭 제목과 화면 너비
st.set_page_config(page_title="마켓 대시보드", layout="wide")

# 2) 보여줄 지수 목록: "화면에 쓸 이름": "야후 파이낸스 종목 코드"
INDICES = {
    "코스피": "^KS11",
    "코스닥": "^KQ11",
    "나스닥": "^IXIC",
    "S&P 500": "^GSPC",
}

PERIODS = {"1개월": "1mo", "3개월": "3mo", "6개월": "6mo", "1년": "1y"}


# 3) 데이터 받아 오기. 같은 요청은 10분(600초) 동안 저장해 두고 다시 씀
@st.cache_data(ttl=600)
def load_close(ticker, period):
    data = yf.Ticker(ticker).history(period=period)
    return data["Close"].dropna()


# 4) 첫 번째 화면: 지수 흐름
def show_indices():
    st.title("마켓 대시보드")
    st.caption("주요 지수 종가 흐름")
    period_name = st.sidebar.selectbox("기간", list(PERIODS), index=1)

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
            # delta_color="inverse": 상승은 빨강, 하락은 초록으로 표시
            st.metric(name, f"{last:,.2f}", f"{change:+.2f}%", delta_color="inverse")
            st.line_chart(close)


# 5) 메뉴: "메뉴에 보일 이름": 그 화면을 그리는 함수
#    새 화면을 만들면 여기에 한 줄만 추가하면 됨
MENU = {
    "지수 흐름": show_indices,
    "미국 피어앤그리드": fear_greed.show,
}

choice = st.sidebar.radio("메뉴", list(MENU))
st.sidebar.divider()
MENU[choice]()  # 선택한 화면의 함수를 실행
