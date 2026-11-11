# ============================================================
# 미국 피어앤그리드 화면
# Colab 코드의 계산·그래프 함수는 그대로 두고,
# 화면에 보여주는 부분만 Streamlit용으로 바꾼 파일
#   plt.show()  ->  st.pyplot(fig)
#   print(...)  ->  st.metric / st.dataframe / st.info
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

plt.rcParams['axes.unicode_minus'] = False

# 색상
COL_FG_INDEX = '#1565C0'
COL_FG_EMA20 = '#F57C00'
COL_OSC = '#6A5ACD'
COL_PRICE = '#555555'
GRID_ALPHA = 0.30

TICKERS = ['^GSPC', '^IXIC', '^VIX', '^TNX', '^FVX', 'HYG', 'IEF', 'QQQ', 'SPY']

REQUIRED_COLS = ['S&P500', 'NASDAQ', 'VIX', '10Y', '5Y', 'HYG', 'IEF', 'QQQ', 'SPY']


# ============================================================
# 1) 데이터 다운로드 (1시간 동안 저장해 두고 다시 씀)
# ============================================================

@st.cache_data(ttl=3600)
def load_data():
    raw = yf.download(
        TICKERS,
        start='2024-01-01',
        auto_adjust=False,
        progress=False,
        group_by='column'
    )

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)
        if 'Close' in level0:
            data = raw['Close'].copy()
        elif 'Close' in level1:
            data = raw.xs('Close', axis=1, level=1).copy()
        else:
            raise KeyError('yfinance 결과에서 Close 컬럼을 찾을 수 없습니다.')
    else:
        data = raw.copy()

    data = data.rename(
        columns={
            '^GSPC': 'S&P500',
            '^IXIC': 'NASDAQ',
            '^VIX': 'VIX',
            '^TNX': '10Y',
            '^FVX': '5Y'
        }
    )

    missing_cols = [col for col in REQUIRED_COLS if col not in data.columns]
    if missing_cols:
        raise KeyError(f'다운로드되지 않은 컬럼이 있습니다: {missing_cols}')

    data = data[REQUIRED_COLS].copy()
    for col in REQUIRED_COLS:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.replace([np.inf, -np.inf], np.nan)
    return data


# ============================================================
# 2) 계산 함수 (Colab 코드 그대로)
# ============================================================

def calculate_rsi(
    series,
    window=10
):
    delta = series.diff()

    gain = (
        delta.where(delta > 0, 0)
        .rolling(window=window)
        .mean()
    )

    loss = (
        -delta.where(delta < 0, 0)
        .rolling(window=window)
        .mean()
    )

    rs = (
        gain
        / loss.replace(0, np.nan)
    )

    rsi = (
        100
        - (100 / (1 + rs))
    )

    rsi.loc[
        (loss == 0)
        & (gain > 0)
    ] = 100

    return rsi


# ============================================================
# 3) MACD 함수
# ============================================================

def calculate_macd_components(
    series
):
    ema12 = (
        series
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema26 = (
        series
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    macd = ema12 - ema26

    signal = (
        macd
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    oscillator = macd - signal

    return macd, signal, oscillator


def calculate_macd(
    df,
    column
):
    _, _, oscillator = (
        calculate_macd_components(
            df[column]
        )
    )

    return oscillator


def calculate_fear_greed(
    df,
    index_col,
    label
):
    """
    Yahoo에서 충분히 제공되는 데이터만 사용:
    - 125일 Momentum
    - RSI(10)
    - VIX
    - Risk Appetite (HYG / IEF)

    10Y / 5Y 금리 스프레드는 현재 Yahoo 장기 데이터가
    부족하므로 Fear & Greed 계산에서 제외한다.

    4개 지표를 동일가중 25%로 계산한다.
    결측값은 채우지 않고 실제 값이 존재하는 날짜만 사용한다.
    """

    result = df.copy()

    output_cols = [
        f'{label}_125MA',
        f'{label}_Momentum',
        f'{label}_RSI',
        f'{label}_VIX',
        f'{label}_RiskAppetite',
        f'{label}_FGI',
        f'{label}_FG_Score',
        f'{label}_FG_EMA20',
        f'{label}_FG_EMA20_Score',
        f'{label}_MACD',
        f'{label}_Signal',
        f'{label}_Oscillator'
    ]

    for col in output_cols:
        result[col] = np.nan

    # 지수 가격이 실제로 있는 날짜만 사용
    price_work = (
        result[[index_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[index_col])
        .copy()
    )

    if price_work.empty:
        return result

    # 125일 Momentum
    price_work[f'{label}_125MA'] = (
        price_work[index_col]
        .rolling(window=125)
        .mean()
    )

    price_work[f'{label}_Momentum'] = (
        (
            price_work[index_col]
            - price_work[f'{label}_125MA']
        )
        / price_work[f'{label}_125MA']
        * 100
    )

    # RSI 10
    price_work[f'{label}_RSI'] = (
        calculate_rsi(
            price_work[index_col],
            window=10
        )
    )

    result.loc[
        price_work.index,
        f'{label}_125MA'
    ] = price_work[f'{label}_125MA']

    result.loc[
        price_work.index,
        f'{label}_Momentum'
    ] = price_work[f'{label}_Momentum']

    result.loc[
        price_work.index,
        f'{label}_RSI'
    ] = price_work[f'{label}_RSI']

    # VIX
    result[f'{label}_VIX'] = result['VIX']

    # Risk Appetite
    result[f'{label}_RiskAppetite'] = (
        result['HYG']
        / result['IEF'].replace(0, np.nan)
    )

    # 금리 스프레드 제외: 4개 지표만 사용
    feature_cols = [
        f'{label}_Momentum',
        f'{label}_RSI',
        f'{label}_VIX',
        f'{label}_RiskAppetite'
    ]

    valid = (
        result[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=feature_cols)
        .copy()
    )

    if valid.empty:
        return result

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(
        valid[feature_cols]
    )

    scaled_cols = [
        f'{label}_Momentum_Scaled',
        f'{label}_RSI_Scaled',
        f'{label}_VIX_Scaled',
        f'{label}_RiskAppetite_Scaled'
    ]

    scaled_df = pd.DataFrame(
        scaled,
        index=valid.index,
        columns=scaled_cols
    )

    # 4개 지표 동일가중 25%
    fgi = (
        scaled_df[f'{label}_Momentum_Scaled'] * 0.25
        + scaled_df[f'{label}_RiskAppetite_Scaled'] * 0.25
        + (1 - scaled_df[f'{label}_VIX_Scaled']) * 0.25
        + scaled_df[f'{label}_RSI_Scaled'] * 0.25
    )

    fg_series = pd.Series(
        fgi,
        index=valid.index,
        name=f'{label}_FGI'
    )

    # EMA20
    fg_ema20 = (
        fg_series
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    # 기존 정상 코드와 같은 MACD / Signal / Oscillator
    macd, signal, oscillator = (
        calculate_macd_components(
            fg_series
        )
    )

    result.loc[
        valid.index,
        f'{label}_FGI'
    ] = fg_series

    result.loc[
        valid.index,
        f'{label}_FG_Score'
    ] = fg_series * 100

    result.loc[
        valid.index,
        f'{label}_FG_EMA20'
    ] = fg_ema20

    result.loc[
        valid.index,
        f'{label}_FG_EMA20_Score'
    ] = fg_ema20 * 100

    result.loc[
        valid.index,
        f'{label}_MACD'
    ] = macd

    result.loc[
        valid.index,
        f'{label}_Signal'
    ] = signal

    result.loc[
        valid.index,
        f'{label}_Oscillator'
    ] = oscillator

    return result


def add_super_ma_gap(
    df,
    price_col,
    label,
    windows=(20, 60, 120, 200)
):
    """
    해당 가격 데이터가 실제로 존재하는 날짜만 사용해
    SuperMA와 Gap%를 계산한다.
    """

    result = df.copy()

    price = (
        result[[price_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[price_col])
        .copy()
    )

    for window in windows:
        price[f'{label}_MA{window}'] = (
            price[price_col]
            .rolling(window=window, min_periods=window)
            .mean()
        )

    ma_cols = [
        f'{label}_MA{window}'
        for window in windows
    ]

    # 200일선까지 실제로 존재하는 구간에서만 SuperMA 계산
    price[f'{label}_SuperMA'] = (
        price[ma_cols]
        .mean(axis=1, skipna=False)
    )

    price[f'{label}_GapPct'] = (
        (
            price[price_col]
            - price[f'{label}_SuperMA']
        )
        / price[f'{label}_SuperMA']
        * 100
    )

    for col in ma_cols + [
        f'{label}_SuperMA',
        f'{label}_GapPct'
    ]:
        result[col] = np.nan
        result.loc[price.index, col] = price[col]

    return result


def add_impulse_components(
    df,
    price_col,
    label
):
    """
    해당 가격이 존재하는 날짜만 사용해 Elder Impulse 구성값 계산.
    """

    result = df.copy()

    work = (
        result[[price_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[price_col])
        .copy()
    )

    work[f'{label}_EMA13'] = (
        work[price_col]
        .ewm(
            span=13,
            adjust=False
        )
        .mean()
    )

    work[f'{label}_MACD_Hist'] = (
        calculate_macd(
            work,
            price_col
        )
    )

    result[f'{label}_EMA13'] = np.nan
    result[f'{label}_MACD_Hist'] = np.nan

    result.loc[
        work.index,
        f'{label}_EMA13'
    ] = work[f'{label}_EMA13']

    result.loc[
        work.index,
        f'{label}_MACD_Hist'
    ] = work[f'{label}_MACD_Hist']

    return result


def get_impulse_colors(
    df,
    ema_col,
    macd_col
):
    colors = []

    ema = df[ema_col]
    macd = df[macd_col]

    for i in range(len(df)):
        if i == 0:
            colors.append('gray')
            continue

        ema_up = (
            ema.iloc[i]
            > ema.iloc[i - 1]
        )

        ema_down = (
            ema.iloc[i]
            < ema.iloc[i - 1]
        )

        macd_up = (
            macd.iloc[i]
            > macd.iloc[i - 1]
        )

        macd_down = (
            macd.iloc[i]
            < macd.iloc[i - 1]
        )

        if ema_up and macd_up:
            colors.append('green')

        elif ema_down and macd_down:
            colors.append('red')

        else:
            colors.append('blue')

    return colors


def add_td_setup_counts(
    df,
    price_col,
    label
):
    """
    가격이 존재하는 거래일만 이어 붙여 TD Setup 계산.
    """

    result = df.copy()

    work = (
        result[[price_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[price_col])
        .copy()
    )

    n = len(work)

    sell = np.zeros(n)
    buy = np.zeros(n)

    prices = work[price_col].to_numpy()

    for i in range(n):

        # Sell Setup
        if (
            i >= 4
            and np.isfinite(prices[i])
            and np.isfinite(prices[i - 4])
            and prices[i] > prices[i - 4]
        ):
            sell[i] = sell[i - 1] + 1
        else:
            sell[i] = 0

        # Buy Setup
        if (
            i >= 2
            and np.isfinite(prices[i])
            and np.isfinite(prices[i - 2])
            and prices[i] < prices[i - 2]
        ):
            buy[i] = buy[i - 1] + 1
        else:
            buy[i] = 0

    work[f'{label}_TD_SellSetup'] = sell
    work[f'{label}_TD_BuySetup'] = buy

    result[f'{label}_TD_SellSetup'] = np.nan
    result[f'{label}_TD_BuySetup'] = np.nan

    result.loc[
        work.index,
        f'{label}_TD_SellSetup'
    ] = work[f'{label}_TD_SellSetup']

    result.loc[
        work.index,
        f'{label}_TD_BuySetup'
    ] = work[f'{label}_TD_BuySetup']

    return result


# ============================================================
# 3) 그래프 함수 (Colab 코드에서 plt.show()만 st.pyplot으로 변경)
# ============================================================

def plot_fear_greed_combined(
    df,
    label,
    price_col,
    market_name
):
    required = [
        f'{label}_FG_Score',
        f'{label}_FG_EMA20_Score',
        f'{label}_Oscillator',
        price_col
    ]

    chart_df = (
        df
        .dropna(
            subset=required
        )
        .copy()
    )

    if chart_df.empty:
        st.info(
            f'{market_name}: '
            f'그래프로 표시할 데이터가 없습니다.'
        )
        return

    fig, ax_fg = plt.subplots(
        figsize=(16, 8)
    )

    fig.subplots_adjust(
        right=0.82
    )

    # ----------------------------------
    # 왼쪽 축: F&G Index + EMA20
    # ----------------------------------

    line_fg, = ax_fg.plot(
        chart_df.index,
        chart_df[
            f'{label}_FG_Score'
        ],
        color=COL_FG_INDEX,
        linewidth=2.5,
        label='Fear & Greed Index'
    )

    line_ema, = ax_fg.plot(
        chart_df.index,
        chart_df[
            f'{label}_FG_EMA20_Score'
        ],
        color=COL_FG_EMA20,
        linewidth=2.1,
        label='F&G EMA20'
    )

    ax_fg.set_ylim(
        0,
        100
    )

    ax_fg.set_ylabel(
        'Fear & Greed Index',
        color=COL_FG_INDEX
    )

    ax_fg.tick_params(
        axis='y',
        labelcolor=COL_FG_INDEX
    )

    for level in [20, 50, 80]:
        ax_fg.axhline(
            level,
            color='gray',
            linestyle='--',
            linewidth=0.8,
            alpha=0.5,
            label='_nolegend_'
        )

    ax_fg.grid(
        True,
        alpha=GRID_ALPHA
    )

    ax_fg.set_xlabel('Date')


    # ----------------------------------
    # 오른쪽 축 1: 시장 지수
    # ----------------------------------

    ax_price = ax_fg.twinx()

    line_price, = ax_price.plot(
        chart_df.index,
        chart_df[price_col],
        color=COL_PRICE,
        linewidth=1.5,
        alpha=0.50,
        label=market_name
    )

    ax_price.set_ylabel(
        market_name,
        color=COL_PRICE
    )

    ax_price.tick_params(
        axis='y',
        labelcolor=COL_PRICE
    )


    # ----------------------------------
    # 오른쪽 축 2: Oscillator
    # ----------------------------------

    ax_osc = ax_fg.twinx()

    ax_osc.spines[
        'right'
    ].set_position(
        ('axes', 1.11)
    )

    line_osc, = ax_osc.plot(
        chart_df.index,
        chart_df[
            f'{label}_Oscillator'
        ],
        color=COL_OSC,
        linewidth=2.0,
        label='Fear & Greed Oscillator'
    )

    ax_osc.axhline(
        0,
        color=COL_OSC,
        linewidth=0.9,
        alpha=0.7,
        label='_nolegend_'
    )

    ax_osc.set_ylabel(
        'Oscillator',
        color=COL_OSC
    )

    ax_osc.tick_params(
        axis='y',
        labelcolor=COL_OSC
    )


    # ----------------------------------
    # 최신값
    # ----------------------------------

    latest = chart_df.iloc[-1]

    latest_fg = (
        latest[
            f'{label}_FG_Score'
        ]
    )

    latest_ema = (
        latest[
            f'{label}_FG_EMA20_Score'
        ]
    )

    latest_osc = (
        latest[
            f'{label}_Oscillator'
        ]
    )


    # ----------------------------------
    # 범례
    # ----------------------------------

    lines = [
        line_fg,
        line_ema,
        line_osc,
        line_price
    ]

    labels = [
        line.get_label()
        for line in lines
    ]

    ax_fg.legend(
        lines,
        labels,
        loc='upper left',
        ncol=2,
        frameon=True
    )


    # ----------------------------------
    # 제목
    # ----------------------------------

    ax_fg.set_title(
        f'{market_name} – '
        f'Fear & Greed Index + EMA20 + Oscillator\n'
        f'Fear & Greed {latest_fg:.1f} | '
        f'EMA20 {latest_ema:.1f} | '
        f'Oscillator {latest_osc:.4f}',
        fontsize=14
    )

    st.pyplot(fig)
    plt.close(fig)


def plot_super_ma_gap(
    df,
    price_col,
    label,
    title
):
    chart_df = df.dropna(
        subset=[
            price_col,
            f'{label}_SuperMA',
            f'{label}_GapPct'
        ]
    ).copy()

    if chart_df.empty:
        st.info(
            f'{title}: 표시할 데이터가 없습니다.'
        )
        return

    fig, ax1 = plt.subplots(
        figsize=(14, 7)
    )

    ax1.plot(
        chart_df.index,
        chart_df[price_col],
        label=f'{label} Price',
        color='black'
    )

    ax1.plot(
        chart_df.index,
        chart_df[
            f'{label}_SuperMA'
        ],
        label=(
            f'{label} SuperMA '
            f'(20/60/120/200 avg)'
        ),
        linestyle='--',
        color='orange'
    )

    ax1.set_ylabel(
        f'{label} Price'
    )

    ax1.grid(True)

    ax2 = ax1.twinx()

    ax2.plot(
        chart_df.index,
        chart_df[
            f'{label}_GapPct'
        ],
        label=(
            f'{label} Gap% '
            f'vs SuperMA'
        ),
        color='purple',
        alpha=0.7
    )

    ax2.set_ylabel(
        'Gap %',
        color='purple'
    )

    lines1, labels1 = (
        ax1.get_legend_handles_labels()
    )

    lines2, labels2 = (
        ax2.get_legend_handles_labels()
    )

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='upper left'
    )

    plt.title(title)

    st.pyplot(fig)
    plt.close(fig)


def plot_impulse(
    df,
    price_col,
    ema_col,
    macd_col,
    title
):
    chart_df = df.dropna(
        subset=[
            price_col,
            ema_col,
            macd_col
        ]
    ).copy()

    if chart_df.empty:
        st.info(
            f'{title}: 표시할 데이터가 없습니다.'
        )
        return

    impulse_colors = (
        get_impulse_colors(
            chart_df,
            ema_col,
            macd_col
        )
    )

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    ax.plot(
        chart_df.index,
        chart_df[price_col],
        color='black',
        label=f'{price_col} Price'
    )

    ax.scatter(
        chart_df.index,
        chart_df[price_col],
        c=impulse_colors,
        s=20,
        label='Impulse'
    )

    ax.set_ylabel(
        f'{price_col} Price'
    )

    ax.grid(True)

    ax.legend(
        loc='upper left'
    )

    plt.title(title)

    st.pyplot(fig)
    plt.close(fig)


def plot_td_setup(
    df,
    price_col,
    label,
    title
):
    sell_col = (
        f'{label}_TD_SellSetup'
    )

    buy_col = (
        f'{label}_TD_BuySetup'
    )

    chart_df = df.dropna(
        subset=[
            price_col,
            sell_col,
            buy_col
        ]
    ).copy()

    if chart_df.empty:
        st.info(
            f'{title}: 표시할 데이터가 없습니다.'
        )
        return

    fig, ax1 = plt.subplots(
        figsize=(14, 7)
    )

    line_price, = ax1.plot(
        chart_df.index,
        chart_df[price_col],
        color='black',
        label=f'{label} Price'
    )

    ax1.set_ylabel(
        f'{label} Price',
        color='black'
    )

    ax1.tick_params(
        axis='y',
        labelcolor='black'
    )

    ax1.grid(True)


    ax2 = ax1.twinx()

    line_sell, = ax2.plot(
        chart_df.index,
        chart_df[sell_col],
        color='red',
        label='TD Sell Setup'
    )

    line_buy, = ax2.plot(
        chart_df.index,
        chart_df[buy_col],
        color='blue',
        label='TD Buy Setup'
    )

    ax2.set_ylabel(
        'TD Setup Count',
        color='gray'
    )

    ax2.tick_params(
        axis='y',
        labelcolor='gray'
    )

    max_td = (
        chart_df[
            [
                sell_col,
                buy_col
            ]
        ]
        .max()
        .max()
    )

    ax2.set_ylim(
        0,
        max_td + 2
    )

    lines = [
        line_price,
        line_sell,
        line_buy
    ]

    ax1.legend(
        lines,
        [
            line.get_label()
            for line in lines
        ],
        loc='upper left'
    )

    plt.title(title)

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# 4) 전체 계산을 한 번에 실행
# ============================================================

def build_all(data):
    data = data.copy()
    data['Risk_Appetite'] = data['HYG'] / data['IEF'].replace(0, np.nan)
    data = calculate_fear_greed(data, index_col='S&P500', label='SPX')
    data = calculate_fear_greed(data, index_col='NASDAQ', label='NDX')
    for label in ['QQQ', 'SPY']:
        data = add_super_ma_gap(data, price_col=label, label=label)
        data = add_impulse_components(data, price_col=label, label=label)
        data = add_td_setup_counts(data, price_col=label, label=label)
    return data


# ============================================================
# 5) 최신 수치를 숫자 카드로 표시 (Colab의 print 부분)
# ============================================================

def show_latest_fear_greed(df, label, market_name, price_col):
    required = [
        f'{label}_FG_Score',
        f'{label}_FG_EMA20_Score',
        f'{label}_Oscillator',
        price_col
    ]
    valid = df.dropna(subset=required)
    if valid.empty:
        st.info(f'{market_name}: 계산 가능한 데이터가 없습니다.')
        return
    latest = valid.iloc[-1]
    st.caption(f'{market_name} · 기준일 {latest.name.strftime("%Y-%m-%d")}')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(market_name, f'{latest[price_col]:,.2f}')
    c2.metric('Fear & Greed', f'{latest[f"{label}_FG_Score"]:.1f}')
    c3.metric('F&G EMA20', f'{latest[f"{label}_FG_EMA20_Score"]:.1f}')
    c4.metric('Oscillator', f'{latest[f"{label}_Oscillator"]:.4f}')


# ============================================================
# 6) 화면 구성: app.py가 이 함수를 불러서 화면을 그림
# ============================================================

def show():
    st.title('미국 피어앤그리드')
    st.caption(
        '125일 모멘텀 · RSI 10일 · VIX · Risk Appetite(HYG/IEF) 동일가중 '
        '+ EMA20 + Oscillator'
    )

    # Colab의 DISPLAY_MONTHS를 사이드바 선택 상자로 바꿈
    display_months = st.sidebar.selectbox(
        '그래프 표시 기간',
        [6, 12, 18],
        format_func=lambda m: f'최근 {m}개월'
    )

    try:
        data = build_all(load_data())
    except Exception as error:
        st.error(f'데이터를 불러오지 못했습니다: {error}')
        return

    cutoff_date = data.index.max() - pd.DateOffset(months=display_months)
    recent = data[data.index >= cutoff_date].copy()

    show_latest_fear_greed(data, 'SPX', 'S&P500', 'S&P500')
    show_latest_fear_greed(data, 'NDX', 'NASDAQ', 'NASDAQ')

    tab_fg, tab_ma, tab_impulse, tab_td = st.tabs(
        ['Fear & Greed', 'SuperMA · 괴리율', 'Elder Impulse', 'DeMark TD Setup']
    )

    with tab_fg:
        plot_fear_greed_combined(recent, label='SPX', price_col='S&P500', market_name='S&P500')
        plot_fear_greed_combined(recent, label='NDX', price_col='NASDAQ', market_name='NASDAQ')

    with tab_ma:
        for label in ['QQQ', 'SPY']:
            plot_super_ma_gap(
                recent, price_col=label, label=label,
                title=f'{label} vs SuperMA and Gap% (Last {display_months} Months)'
            )

    with tab_impulse:
        for label in ['QQQ', 'SPY']:
            plot_impulse(
                recent, price_col=label,
                ema_col=f'{label}_EMA13', macd_col=f'{label}_MACD_Hist',
                title=f'{label} Price with Elder Impulse System (Last {display_months} Months)'
            )

    with tab_td:
        for label in ['SPY', 'QQQ']:
            plot_td_setup(
                recent, price_col=label, label=label,
                title=f'{label} Price with DeMark TD Setup Counts (Last {display_months} Months)'
            )

    # Colab에서 print로 찍던 데이터 상태 확인
    with st.expander('데이터 상태 확인'):
        st.write(f'전체 기간: {data.index.min().date()} ~ {data.index.max().date()} · 전체 행 수: {len(data)}')
        st.write('각 데이터 유효값 개수')
        st.dataframe(data[REQUIRED_COLS].notna().sum().rename('유효값 개수'))
        st.write('최근 데이터')
        st.dataframe(data[REQUIRED_COLS].tail())
