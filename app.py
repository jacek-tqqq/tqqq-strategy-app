import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from backtest import run_backtest, summarize

st.set_page_config(page_title="TQQQ Strategy", layout="wide")
st.title("TQQQ Strategy Backtest")
st.caption("Research prototype only. It does not place live orders.")
today = pd.Timestamp.today().normalize()

with st.sidebar:
    st.header("Market")
    ticker = st.text_input("Ticker", "TQQQ")

    st.header("SMA strategy")
    use_sma = st.toggle("Use SMA strategy", value=True)
    buy_days = st.number_input(
        "Buy SMA (days)", 2, 500, 50, disabled=not use_sma
    )
    sell_days = st.number_input(
        "Sell SMA (days)", 2, 500, 50, disabled=not use_sma
    )
    use_sma_buffer = st.toggle(
        "Use SMA buffer", value=True, disabled=not use_sma
    )
    sma_buffer = st.number_input(
        "SMA buffer (%)",
        0.0,
        20.0,
        1.0,
        0.1,
        disabled=not (use_sma and use_sma_buffer),
    )
    use_confirmation = st.toggle(
        "Use confirmation period", value=True, disabled=not use_sma
    )
    confirmation_days = st.number_input(
        "Confirmation period (trading days)",
        1,
        20,
        2,
        disabled=not (use_sma and use_confirmation),
    )

    st.header("52-week low/high strategy")
    use_52 = st.toggle("Use 52-week low/high strategy", value=False)
    low_buy = st.number_input(
        "Buy below 52-week low (%)",
        0.0,
        99.0,
        20.0,
        1.0,
        disabled=not use_52,
    )
    high_sell = st.number_input(
        "Sell above 52-week high (%)",
        0.0,
        99.0,
        0.0,
        1.0,
        disabled=not use_52,
    )

    st.header("RSI buy filter")
    use_rsi = st.toggle(
        "Use RSI with SMA buys", value=True, disabled=not use_sma
    )
    rsi_period = st.number_input(
        "RSI period", 2, 100, 14, disabled=not (use_sma and use_rsi)
    )
    rsi_threshold = st.number_input(
        "Buy only when RSI is above",
        0.0,
        100.0,
        50.0,
        1.0,
        disabled=not (use_sma and use_rsi),
    )
    show_rsi = st.toggle("Show RSI graph", value=True)

    st.header("Backtest")
    start = st.date_input(
        "Start", (today - pd.DateOffset(years=10)).date()
    )
    end = st.date_input("End", today.date())
    initial_cash = st.number_input(
        "Initial capital", 100.0, 1_000_000.0, 10000.0, 100.0
    )
    slippage = st.number_input(
        "Slippage (%)", 0.0, 2.0, 0.10, 0.01
    ) / 100

if use_sma and use_52:
    rule_text = "SMA and 52-week low/high strategies are enabled as independent triggers."
elif use_sma:
    rule_text = "Only the SMA strategy is enabled."
elif use_52:
    rule_text = "Only the 52-week low/high strategy is enabled."
else:
    rule_text = "No trading strategy is enabled, so the backtest will remain in cash."

st.info(
    rule_text
    + " The last-trade-price block has been removed. Trades execute at the next "
      "market open. No trailing stop or Z-score."
)


@st.cache_data(ttl=3600)
def load_data(ticker, start, end, warmup):
    fetch_start = pd.Timestamp(start) - pd.Timedelta(
        days=max(800, int(warmup) * 3)
    )
    df = yf.download(
        ticker,
        start=fetch_start.strftime("%Y-%m-%d"),
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Open", "Close"])


if st.button("Run backtest", type="primary"):
    if pd.Timestamp(start) >= pd.Timestamp(end):
        st.error("Start must be before end.")
        st.stop()

    warmup = max(buy_days, sell_days, rsi_period, 252) + confirmation_days
    df = load_data(ticker, start, end, warmup)
    if df.empty:
        st.error("No market data was returned.")
        st.stop()

    result = run_backtest(
        df=df,
        start=start,
        end=end,
        use_sma=use_sma,
        buy_days=buy_days,
        sell_days=sell_days,
        use_rsi=use_rsi if use_sma else False,
        rsi_period=rsi_period,
        rsi_buy_threshold=rsi_threshold,
        use_sma_buffer=use_sma_buffer,
        sma_buffer_pct=sma_buffer / 100,
        use_confirmation=use_confirmation,
        confirmation_days=confirmation_days,
        use_52_week_high=use_52,
        low_buy_discount_pct=low_buy / 100,
        high_sell_pct=high_sell / 100,
        initial_cash=initial_cash,
        slippage=slippage,
    )
    stats = summarize(result)

    cols = st.columns(4)
    cols[0].metric("Strategy return", f"{stats['strategy_return']:.2%}")
    cols[1].metric("Buy & hold", f"{stats['buy_hold_return']:.2%}")
    cols[2].metric("Max drawdown", f"{stats['max_drawdown']:.2%}")
    cols[3].metric("Trades", stats["trades"])

    equity = go.Figure()
    equity.add_trace(
        go.Scatter(x=result.index, y=result["equity"], name="Strategy")
    )
    equity.add_trace(
        go.Scatter(
            x=result.index,
            y=result["buy_hold_equity"],
            name="Buy & hold",
        )
    )
    equity.update_layout(title="Equity curve")
    st.plotly_chart(equity, use_container_width=True)

    price = go.Figure()
    price.add_trace(
        go.Scatter(x=result.index, y=result["Close"], name="Price")
    )
    if use_sma:
        price.add_trace(
            go.Scatter(
                x=result.index,
                y=result["BuyThreshold"],
                name="SMA buy threshold",
            )
        )
        price.add_trace(
            go.Scatter(
                x=result.index,
                y=result["SellThreshold"],
                name="SMA sell threshold",
            )
        )
    if use_52:
        price.add_trace(
            go.Scatter(
                x=result.index,
                y=result["Prior52WeekLow"],
                name="Prior 52-week low",
                line=dict(color="teal", dash="solid", width=2),
            )
        )
        price.add_trace(
            go.Scatter(
                x=result.index,
                y=result["Prior52WeekHigh"],
                name="Prior 52-week high",
                line=dict(color="purple", dash="solid", width=2),
            )
        )
        price.add_trace(
            go.Scatter(
                x=result.index,
                y=result["LowBuyLevel"],
                name="52-week-low buy level",
                line=dict(dash="dot"),
            )
        )
        price.add_trace(
            go.Scatter(
                x=result.index,
                y=result["HighSellLevel"],
                name="52-week sell level",
                line=dict(dash="dash"),
            )
        )

    buys = result[result["signal"] == 1]
    sells = result[result["signal"] == -1]
    price.add_trace(
        go.Scatter(
            x=buys.index,
            y=buys["Close"],
            mode="markers",
            name="Buy",
            marker=dict(symbol="triangle-up", color="green", size=10),
        )
    )
    price.add_trace(
        go.Scatter(
            x=sells.index,
            y=sells["Close"],
            mode="markers",
            name="Sell",
            marker=dict(symbol="triangle-down", color="red", size=10),
        )
    )
    price.update_layout(title="Price, thresholds, and signals")
    st.plotly_chart(price, use_container_width=True)

    if show_rsi:
        rsi_chart = go.Figure()
        rsi_chart.add_trace(
            go.Scatter(x=result.index, y=result["RSI"], name="RSI")
        )
        if use_sma and use_rsi:
            rsi_chart.add_hline(
                y=rsi_threshold,
                line_dash="dash",
                annotation_text=f"SMA buy filter: {rsi_threshold:.0f}",
            )
        rsi_chart.update_layout(
            title=f"RSI ({rsi_period} days)", yaxis_range=[0, 100]
        )
        st.plotly_chart(rsi_chart, use_container_width=True)

    st.subheader("Trades")
    trades = result[result["trade"].notna()][
        [
            "trade",
            "trade_reason",
            "execution_price",
            "shares_held",
            "transaction_gain",
            "cumulative_gain",
        ]
    ].copy()
    trades = trades.rename(
        columns={
            "execution_price": "price",
            "shares_held": "shares_after_trade",
            "transaction_gain": "gain_on_transaction",
            "cumulative_gain": "total_accumulative_gain",
        }
    )
    st.dataframe(
        trades.style.format(
            {
                "price": "${:,.2f}",
                "gain_on_transaction": "${:,.2f}",
                "total_accumulative_gain": "${:,.2f}",
            },
            na_rep="-",
        ),
        use_container_width=True,
    )
    st.caption(
        "Gain is realized when a position is sold. Buy rows show no transaction "
        "gain; cumulative gain carries forward from completed buy/sell cycles."
    )
