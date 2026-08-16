import numpy as np
import pandas as pd


def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / int(period), min_periods=int(period), adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / int(period), min_periods=int(period), adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(avg_loss != 0, 100).where(avg_gain != 0, 0)
    return rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50).clip(0, 100)


def confirmed_signal(condition, periods):
    periods = max(1, int(periods))
    confirmed = condition.fillna(False).rolling(periods).sum().eq(periods)
    return confirmed & ~confirmed.shift(1, fill_value=False)


def run_backtest(
    df, start, end, buy_days=50, sell_days=50, use_rsi=True,
    rsi_period=14, rsi_buy_threshold=50.0, use_sma_buffer=True,
    sma_buffer_pct=0.01, use_confirmation=True, confirmation_days=2,
    use_52_week_high=False, high_buy_discount_pct=0.20,
    high_sell_pct=0.0, initial_cash=10000.0, slippage=0.001,
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
    x["sma_buy_signal"] = confirmed_signal(x["Close"] > x["BuyThreshold"], periods)
    x["sma_sell_signal"] = confirmed_signal(x["Close"] < x["SellThreshold"], periods)

    # The current close is excluded from the 52-week high to avoid look-ahead bias.
    x["Prior52WeekHigh"] = x["Close"].rolling(252, min_periods=252).max().shift(1)
    x["HighBuyLevel"] = x["Prior52WeekHigh"] * (1 - float(high_buy_discount_pct))
    x["HighSellLevel"] = x["Prior52WeekHigh"] * (1 + float(high_sell_pct))
    x["high_buy_signal"] = (
        bool(use_52_week_high) & x["Prior52WeekHigh"].notna()
        & (x["Close"] <= x["HighBuyLevel"])
    )
    x["high_sell_signal"] = (
        bool(use_52_week_high) & x["Prior52WeekHigh"].notna()
        & (x["Close"] >= x["HighSellLevel"])
    )

    x = x.loc[(x.index >= pd.Timestamp(start)) & (x.index <= pd.Timestamp(end))].copy()
    if x.empty:
        raise ValueError("No data in requested period.")

    for col in ["cash", "equity", "execution_price"]:
        x[col] = np.nan
    x["shares_held"] = 0
    x["trade"] = None
    x["trade_reason"] = None
    x["signal"] = 0
    cash, shares = float(initial_cash), 0
    pending_action = pending_reason = None
    bh_shares = initial_cash / float(x.iloc[0]["Close"])

    for i in range(len(x)):
        open_price = float(x.iloc[i]["Open"])
        close_price = float(x.iloc[i]["Close"])
        if pending_action == "BUY" and shares == 0:
            price = open_price * (1 + slippage)
            qty = int(max(0, cash // price))
            if qty:
                cash -= qty * price
                shares = qty
                x.iat[i, x.columns.get_loc("trade")] = "BUY"
                x.iat[i, x.columns.get_loc("trade_reason")] = pending_reason
                x.iat[i, x.columns.get_loc("execution_price")] = price
        elif pending_action == "SELL" and shares > 0:
            price = open_price * (1 - slippage)
            cash += shares * price
            shares = 0
            x.iat[i, x.columns.get_loc("trade")] = "SELL"
            x.iat[i, x.columns.get_loc("trade_reason")] = pending_reason
            x.iat[i, x.columns.get_loc("execution_price")] = price
        pending_action = pending_reason = None

        sma_buy = bool(x.iloc[i]["sma_buy_signal"])
        sma_sell = bool(x.iloc[i]["sma_sell_signal"])
        high_buy = bool(x.iloc[i]["high_buy_signal"])
        high_sell = bool(x.iloc[i]["high_sell_signal"])
        rsi_passes = (not use_rsi) or (
            pd.notna(x.iloc[i]["RSI"]) and float(x.iloc[i]["RSI"]) > float(rsi_buy_threshold)
        )
        if shares == 0 and ((sma_buy and rsi_passes) or high_buy):
            pending_action = "BUY"
            pending_reason = (
                f"Price at least {high_buy_discount_pct:.1%} below prior 52-week high"
                if high_buy else "Confirmed SMA buy signal" + (" and RSI above threshold" if use_rsi else "")
            )
            x.iat[i, x.columns.get_loc("signal")] = 1
        elif shares > 0 and (sma_sell or high_sell):
            pending_action = "SELL"
            pending_reason = (
                f"Price at least {high_sell_pct:.1%} above prior 52-week high"
                if high_sell else "Confirmed SMA sell signal"
            )
            x.iat[i, x.columns.get_loc("signal")] = -1

        x.iat[i, x.columns.get_loc("cash")] = cash
        x.iat[i, x.columns.get_loc("shares_held")] = shares
        x.iat[i, x.columns.get_loc("equity")] = cash + shares * close_price

    x["buy_hold_equity"] = bh_shares * x["Close"]
    return x


def summarize(result):
    drawdown = result["equity"] / result["equity"].cummax() - 1
    return {
        "strategy_return": float(result["equity"].iloc[-1] / result["equity"].iloc[0] - 1),
        "buy_hold_return": float(result["buy_hold_equity"].iloc[-1] / result["buy_hold_equity"].iloc[0] - 1),
        "max_drawdown": float(drawdown.min()),
        "trades": int(result["trade"].notna().sum()),
    }
