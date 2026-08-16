import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from backtest import run_backtest, summarize

st.set_page_config(page_title="TQQQ SMA Strategy", layout="wide")
st.title("TQQQ SMA Strategy")
st.caption("Research and signal prototype. It does not place live orders.")
today = pd.Timestamp.today().normalize()

with st.sidebar:
    st.header("Market")
    ticker = st.text_input("Ticker", "TQQQ")
    st.header("SMA rules")
    buy_days = st.number_input("Buy SMA (days)", 2, 500, 50)
    sell_days = st.number_input("Sell SMA (days)", 2, 500, 50)
    use_sma_buffer = st.toggle("Use SMA buffer", value=True)
    sma_buffer = st.number_input("SMA buffer (%)", 0.0, 20.0, 1.0, 0.1, disabled=not use_sma_buffer)
    use_confirmation = st.toggle("Use confirmation period", value=True)
    confirmation_days = st.number_input("Confirmation period (trading days)", 1, 20, 2, disabled=not use_confirmation)

    st.header("52-week high rule")
    use_52_week_high = st.toggle("Use 52-week high buy/sell rule", value=False)
    high_buy_discount = st.number_input(
        "Buy when price is this far below 52-week high (%)",
        0.0, 99.0, 20.0, 1.0, disabled=not use_52_week_high,
    )
    high_sell_above = st.number_input(
        "Sell when price is this far above 52-week high (%)",
        0.0, 99.0, 0.0, 1.0, disabled=not use_52_week_high,
    )

    st.header("RSI buy filter")
    use_rsi = st.toggle("Use RSI as a buy condition", value=True)
    rsi_period = st.number_input("RSI period (days)", 2, 100, 14, disabled=not use_rsi)
    rsi_threshold = st.number_input("Buy only when RSI is above", 0.0, 100.0, 50.0, 1.0, disabled=not use_rsi)
    show_rsi = st.toggle("Show RSI graph", value=True)
    rsi_lower = st.number_input("RSI lower graph level", 0.0, 100.0, 30.0, 1.0, disabled=not show_rsi)
    rsi_upper = st.number_input("RSI upper graph level", 0.0, 100.0, 70.0, 1.0, disabled=not show_rsi)

    st.header("Backtest")
    start = st.date_input("Backtest start", (today - pd.DateOffset(years=10)).date())
    end = st.date_input("Backtest end", today.date())
    initial_cash = st.number_input("Initial capital", 100.0, 1_000_000.0, 10000.0, 100.0)
    slippage = st.number_input("Slippage (%)", 0.0, 2.0, 0.10, 0.01) / 100

buffer_text = f" with a {sma_buffer:.1f}% buffer" if use_sma_buffer else ""
confirm_text = f" for {confirmation_days} consecutive days" if use_confirmation else ""
rsi_text = f" and RSI above {rsi_threshold:.0f}" if use_rsi else ""
high_text = (
    f" Independent 52-week rule: buy {high_buy_discount:.1f}% below the prior high; "
    f"sell {high_sell_above:.1f}% above the prior high."
    if use_52_week_high else ""
)
st.info(
    f"SMA BUY: above {buy_days}-day SMA{buffer_text}{confirm_text}{rsi_text}. "
    f"SMA SELL: below {sell_days}-day SMA{buffer_text}{confirm_text}.{high_text} "
    "Trades execute at the next open. No commissions or trailing stop."
)

@st.cache_data(ttl=3600)
def load_data(ticker, start, end, warmup):
    fetch_start = pd.Timestamp(start) - pd.Timedelta(days=max(800, int(warmup) * 3))
    df = yf.download(ticker, start=fetch_start.strftime("%Y-%m-%d"),
                     end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                     auto_adjust=False, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Open", "Close"])

if st.button("Run backtest", type="primary"):
    if pd.Timestamp(start) >= pd.Timestamp(end):
        st.error("Start must be before end.")
        st.stop()
    if show_rsi and rsi_lower >= rsi_upper:
        st.error("RSI lower level must be below upper level.")
        st.stop()
    df = load_data(ticker, start, end, max(buy_days, sell_days, rsi_period, 252) + confirmation_days)
    if df.empty:
        st.error("No market data was returned.")
        st.stop()
    result = run_backtest(
        df=df, start=start, end=end, buy_days=buy_days, sell_days=sell_days,
        use_rsi=use_rsi, rsi_period=rsi_period, rsi_buy_threshold=rsi_threshold,
        use_sma_buffer=use_sma_buffer, sma_buffer_pct=sma_buffer / 100,
        use_confirmation=use_confirmation, confirmation_days=confirmation_days,
        use_52_week_high=use_52_week_high,
        high_buy_discount_pct=high_buy_discount / 100,
        high_sell_pct=high_sell_above / 100,
        initial_cash=initial_cash, slippage=slippage,
    )
    stats = summarize(result)
    cols = st.columns(4)
    cols[0].metric("Strategy return", f"{stats['strategy_return']:.2%}")
    cols[1].metric("Buy & hold", f"{stats['buy_hold_return']:.2%}")
    cols[2].metric("Max drawdown", f"{stats['max_drawdown']:.2%}")
    cols[3].metric("Trades", stats["trades"])

    e = go.Figure()
    e.add_trace(go.Scatter(x=result.index, y=result["equity"], name="Strategy"))
    e.add_trace(go.Scatter(x=result.index, y=result["buy_hold_equity"], name="Buy & hold"))
    e.update_layout(title="Equity curve", yaxis_title="Portfolio value")
    st.plotly_chart(e, use_container_width=True)

    p = go.Figure()
    p.add_trace(go.Scatter(x=result.index, y=result["Close"], name="Price"))
    p.add_trace(go.Scatter(x=result.index, y=result["BuyThreshold"], name="SMA buy threshold"))
    p.add_trace(go.Scatter(x=result.index, y=result["SellThreshold"], name="SMA sell threshold"))
    if use_52_week_high:
        p.add_trace(go.Scatter(x=result.index, y=result["Prior52WeekHigh"], name="Prior 52-week high", line=dict(dash="dash")))
        p.add_trace(go.Scatter(x=result.index, y=result["HighBuyLevel"], name=f"{high_buy_discount:.1f}% below high", line=dict(dash="dot")))
        p.add_trace(go.Scatter(x=result.index, y=result["HighSellLevel"], name=f"{high_sell_above:.1f}% above high", line=dict(dash="dot")))
    buys = result[result["signal"] == 1]
    sells = result[result["signal"] == -1]
    p.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers", name="Buy signal", marker=dict(symbol="triangle-up", color="green", size=10)))
    p.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers", name="Sell signal", marker=dict(symbol="triangle-down", color="red", size=10)))
    p.update_layout(title="Price, thresholds, and signals", yaxis_title="Price")
    st.plotly_chart(p, use_container_width=True)

    if show_rsi:
        r = go.Figure()
        r.add_trace(go.Scatter(x=result.index, y=result["RSI"], name="RSI"))
        if use_rsi:
            r.add_hline(y=rsi_threshold, line_color="blue", annotation_text=f"Buy threshold: {rsi_threshold:.0f}")
        r.add_hline(y=rsi_lower, line_dash="dash", line_color="green", annotation_text=f"Lower: {rsi_lower:.0f}")
        r.add_hline(y=rsi_upper, line_dash="dash", line_color="red", annotation_text=f"Upper: {rsi_upper:.0f}")
        r.update_layout(title=f"RSI ({rsi_period} days)", yaxis_range=[0, 100])
        st.plotly_chart(r, use_container_width=True)

    st.subheader("Trades")
    trades = result[result["trade"].notna()][["trade", "trade_reason", "execution_price", "shares_held"]].rename(columns={"shares_held": "shares"})
    st.dataframe(trades, use_container_width=True)
    st.subheader("Daily data")
    st.dataframe(result.tail(100), use_container_width=True)
