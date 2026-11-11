import streamlit as st
import yfinance as yf

import fear_greed  # 같은 폴더의 fear_greed.py 파일을 불러옴
import macro       # 같은 폴더의 macro.py 파일을 불러옴

# 1) 페이지 기본 설정: 브라우저 탭 제목과 화면 너비
SITE_TITLE = "동하아빠의 주식투자"
st.set_page_config(page_title=SITE_TITLE, layout="wide")

# 2) 사이드바 꾸미기 (CSS). 색을 바꾸고 싶으면 # 뒤의 색상 코드만 고치면 됨
st.markdown(
    """
    <style>
    /* 사이드바 배경: 짙은 남색 */
    [data-testid="stSidebar"] { background-color: #0d1526; }

    /* 사이드바 안의 글자색 */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #e8edf7;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #93a1bd; }

    /* 메뉴 버튼: 가로로 꽉 찬 둥근 상자 */
    [data-testid="stSidebar"] .stButton button {
        background-color: #152038;
        border: 1px solid #2a3a5e;
        border-radius: 8px;
    }
    [data-testid="stSidebar"] .stButton button p { font-weight: 600; }
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #1d2c4d;
        border-color: #4a64a0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 3) 보여줄 지수 목록: "화면에 쓸 이름": "야후 파이낸스 종목 코드"
INDICES = {
    "코스피": "^KS11",
    "코스닥": "^KQ11",
    "나스닥": "^IXIC",
    "S&P 500": "^GSPC",
}

PERIODS = {"1개월": "1mo", "3개월": "3mo", "6개월": "6mo", "1년": "1y"}


# 4) 데이터 받아 오기. 같은 요청은 10분(600초) 동안 저장해 두고 다시 씀
@st.cache_data(ttl=600)
def load_close(ticker, period):
    data = yf.Ticker(ticker).history(period=period)
    return data["Close"].dropna()


# 5) 첫 번째 화면: 지수 흐름
def show_indices():
    st.title("지수 흐름")
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


# 6) 메뉴 구성: "묶음 제목": {"메뉴 이름": 그 화면을 그리는 함수}
#    새 화면은 원하는 묶음에 한 줄 추가, 새 묶음은 통째로 하나 추가
MENU = {
    "시장 지표": {
        "지수 흐름": show_indices,
        "유가·환율": macro.show_oil_fx,
        "미국 국채금리": macro.show_yields,
    },
    "심리 지표": {
        "미국 피어앤그리드": fear_greed.show,
        "CNN 피어앤그리드": macro.show_cnn_fear_greed,
    },
}

# 모든 화면을 "메뉴 이름": 함수 형태로 한데 모음
PAGES = {name: func for group in MENU.values() for name, func in group.items()}

# 지금 보고 있는 화면을 기억해 둠. 처음에는 첫 번째 메뉴
if "page" not in st.session_state:
    st.session_state["page"] = next(iter(PAGES))

# 7) 사이드바 그리기: 제목 -> 묶음 제목 -> 메뉴 버튼
st.sidebar.markdown(f"### {SITE_TITLE}")
st.sidebar.caption("Market Dashboard")
st.sidebar.divider()

for group_name, items in MENU.items():
    st.sidebar.markdown(f"**{group_name}**")
    for name in items:
        is_current = name == st.session_state["page"]
        label = f"● {name}" if is_current else name   # 현재 화면 앞에 ● 표시
        if st.sidebar.button(label, key=f"menu_{name}", width="stretch"):
            st.session_state["page"] = name
            st.rerun()

st.sidebar.divider()

# 8) 선택된 화면의 함수를 실행
PAGES[st.session_state["page"]]()
