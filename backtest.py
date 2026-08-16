import numpy as np
import pandas as pd


def calculate_rsi(close, period=14):
    period = int(period)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(avg_loss != 0, 100).where(avg_gain != 0, 0)
    return rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50).clip(0, 100)


def confirmed_signal(condition, days):
    days = max(1, int(days))
    confirmed = condition.fillna(False).rolling(days).sum().eq(days)
    return confirmed & ~confirmed.shift(1, fill_value=False)


def run_backtest(
    df,
    start,
    end,
    use_sma=True,
    buy_days=50,
    sell_days=50,
    use_rsi=True,
    rsi_period=14,
    rsi_buy_threshold=50.0,
    use_sma_buffer=True,
    sma_buffer_pct=0.01,
    use_confirmation=True,
    confirmation_days=2,
    use_52_week_high=False,
    low_buy_discount_pct=0.20,
    high_sell_pct=0.0,
    initial_cash=10000.0,
    slippage=0.001,
):
    x = df.copy()
    x.index = pd.to_datetime(x.index).tz_localize(None)
    x = x.sort_index()

    x["BuySMA"] = x["Close"].rolling(int(buy_days)).mean()
    x["SellSMA"] = x["Close"].rolling(int(sell_days)).mean()
    x["RSI"] = calculate_rsi(x["Close"], rsi_period)

    buffer = float(sma_buffer_pct) if use_sma_buffer else 0.0
    periods = int(confirmation_days) if use_confirmation else 1
    x["BuyThreshold"] = x["BuySMA"] * (1 + buffer)
    x["SellThreshold"] = x["SellSMA"] * (1 - buffer)
    x["sma_buy_signal"] = confirmed_signal(
        x["Close"] > x["BuyThreshold"], periods
    )
    x["sma_sell_signal"] = confirmed_signal(
        x["Close"] < x["SellThreshold"], periods
    )

    # Use the prior 252 trading days and exclude today's close.
    x["Prior52WeekLow"] = x["Close"].rolling(252, min_periods=252).min().shift(1)
    x["Prior52WeekHigh"] = x["Close"].rolling(252, min_periods=252).max().shift(1)
    x["LowBuyLevel"] = x["Prior52WeekLow"] * (1 - float(low_buy_discount_pct))
    x["HighSellLevel"] = x["Prior52WeekHigh"] * (1 + float(high_sell_pct))
    x["high_buy_signal"] = (
        bool(use_52_week_high)
        & x["Prior52WeekLow"].notna()
        & (x["Close"] <= x["LowBuyLevel"])
    )
    x["high_sell_signal"] = (
        bool(use_52_week_high)
        & x["Prior52WeekHigh"].notna()
        & (x["Close"] >= x["HighSellLevel"])
    )

    x = x.loc[(x.index >= pd.Timestamp(start)) & (x.index <= pd.Timestamp(end))].copy()
    if x.empty:
        raise ValueError("No data in requested period.")

    for col in ["cash", "equity", "execution_price", "transaction_gain", "cumulative_gain"]:
        x[col] = np.nan
    x["shares_held"] = 0
    x["trade"] = None
    x["trade_reason"] = None
    x["signal"] = 0

    cash = float(initial_cash)
    shares = 0
    pending_action = None
    pending_reason = None
    current_buy_cost = None
    cumulative_realized_gain = 0.0
    bh_shares = initial_cash / float(x.iloc[0]["Close"])

    for i in range(len(x)):
        row = x.iloc[i]
        open_price = float(row["Open"])
        close_price = float(row["Close"])

        # Execute the prior close's accepted signal at today's opening price.
        if pending_action == "BUY" and shares == 0:
            price = open_price * (1 + slippage)
            qty = int(max(0, cash // price))
            if qty:
                cash -= qty * price
                shares = qty
                current_buy_cost = qty * price
                x.iat[i, x.columns.get_loc("trade")] = "BUY"
                x.iat[i, x.columns.get_loc("trade_reason")] = pending_reason
                x.iat[i, x.columns.get_loc("execution_price")] = price
        elif pending_action == "SELL" and shares > 0:
            price = open_price * (1 - slippage)
            sale_value = shares * price
            transaction_gain = sale_value - current_buy_cost
            cumulative_realized_gain += transaction_gain
            cash += sale_value
            shares = 0
            x.iat[i, x.columns.get_loc("transaction_gain")] = transaction_gain
            x.iat[i, x.columns.get_loc("trade")] = "SELL"
            x.iat[i, x.columns.get_loc("trade_reason")] = pending_reason
            x.iat[i, x.columns.get_loc("execution_price")] = price
            current_buy_cost = None
        pending_action = None
        pending_reason = None

        rsi_passes = (not use_rsi) or (
            pd.notna(row["RSI"])
            and float(row["RSI"]) > float(rsi_buy_threshold)
        )
        sma_buy = bool(use_sma) and bool(row["sma_buy_signal"]) and rsi_passes
        sma_sell = bool(use_sma) and bool(row["sma_sell_signal"])
        high_buy = bool(row["high_buy_signal"])
        high_sell = bool(row["high_sell_signal"])

        if shares == 0 and (high_buy or sma_buy):
            pending_action = "BUY"
            if high_buy:
                pending_reason = (
                    f"Price {low_buy_discount_pct:.1%} below prior 52-week low"
                )
            else:
                pending_reason = "SMA buy" + (" + RSI" if use_rsi else "")
            x.iat[i, x.columns.get_loc("signal")] = 1
        elif shares > 0 and (high_sell or sma_sell):
            pending_action = "SELL"
            if high_sell:
                pending_reason = (
                    f"Price {high_sell_pct:.1%} above prior 52-week high"
                )
            else:
                pending_reason = "SMA sell"
            x.iat[i, x.columns.get_loc("signal")] = -1

        x.iat[i, x.columns.get_loc("cash")] = cash
        x.iat[i, x.columns.get_loc("shares_held")] = shares
        x.iat[i, x.columns.get_loc("equity")] = cash + shares * close_price
        x.iat[i, x.columns.get_loc("cumulative_gain")] = cumulative_realized_gain

    x["buy_hold_equity"] = bh_shares * x["Close"]
    return x


def summarize(result):
    drawdown = result["equity"] / result["equity"].cummax() - 1
    return {
        "strategy_return": float(
            result["equity"].iloc[-1] / result["equity"].iloc[0] - 1
        ),
        "buy_hold_return": float(
            result["buy_hold_equity"].iloc[-1]
            / result["buy_hold_equity"].iloc[0]
            - 1
        ),
        "max_drawdown": float(drawdown.min()),
        "trades": int(result["trade"].notna().sum()),
    }
