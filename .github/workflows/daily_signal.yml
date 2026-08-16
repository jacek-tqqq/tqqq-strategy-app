"""Daily TQQQ strategy checker for GitHub Actions.

The checker supports the same signal inputs as the Streamlit backtest:
SMA on/off, separate buy/sell SMA periods, SMA buffer, confirmation,
RSI buy filter, and prior 52-week low/high thresholds.

It sends an email when a new BUY or SELL condition is triggered. If
SEND_DAILY_STATUS=true, it also sends a status email when there is no signal.
It reports signals only and does not place orders.
"""

import os
import smtplib
from email.message import EmailMessage

import numpy as np
import pandas as pd
import yfinance as yf


def env_bool(name, default=False):
    value = os.getenv(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


TICKER = os.getenv("TICKER", "TQQQ")
USE_SMA = env_bool("USE_SMA", True)
BUY_DAYS = int(os.getenv("BUY_DAYS", "50"))
SELL_DAYS = int(os.getenv("SELL_DAYS", "50"))
USE_SMA_BUFFER = env_bool("USE_SMA_BUFFER", True)
SMA_BUFFER_PCT = float(os.getenv("SMA_BUFFER_PCT", "1.0")) / 100.0
USE_CONFIRMATION = env_bool("USE_CONFIRMATION", True)
CONFIRMATION_DAYS = int(os.getenv("CONFIRMATION_DAYS", "2"))
USE_RSI = env_bool("USE_RSI", True)
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_BUY_THRESHOLD = float(os.getenv("RSI_BUY_THRESHOLD", "50"))
USE_52_WEEK = env_bool("USE_52_WEEK", False)
LOW_BUY_DISCOUNT_PCT = float(os.getenv("LOW_BUY_DISCOUNT_PCT", "20")) / 100.0
HIGH_SELL_PCT = float(os.getenv("HIGH_SELL_PCT", "0")) / 100.0
SEND_DAILY_STATUS = env_bool("SEND_DAILY_STATUS", True)


def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(avg_loss != 0, 100).where(avg_gain != 0, 0)
    return rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50).clip(0, 100)


def confirmed_state(condition, days):
    days = max(1, int(days))
    return condition.fillna(False).rolling(days).sum().eq(days)


def get_data():
    df = yf.download(
        TICKER,
        period="3y",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"]).sort_index()
    if df.empty:
        raise RuntimeError(f"No market data was returned for {TICKER}.")
    return df


def send_email(subject, body):
    required = ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "ALERT_TO"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError("Missing email settings: " + ", ".join(missing))

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ["SMTP_USER"]
    msg["To"] = os.environ["ALERT_TO"]
    msg.set_content(body)

    with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ["SMTP_PORT"])) as server:
        server.starttls()
        server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        server.send_message(msg)


def on_off(value):
    return "ON" if value else "OFF"


def main():
    df = get_data()
    minimum_rows = max(BUY_DAYS, SELL_DAYS, RSI_PERIOD, 252) + max(CONFIRMATION_DAYS, 2) + 2
    if len(df) < minimum_rows:
        raise RuntimeError(f"Not enough data: received {len(df)} rows; need at least {minimum_rows}.")

    df["BuySMA"] = df["Close"].rolling(BUY_DAYS).mean()
    df["SellSMA"] = df["Close"].rolling(SELL_DAYS).mean()
    df["RSI"] = calculate_rsi(df["Close"], RSI_PERIOD)

    buffer = SMA_BUFFER_PCT if USE_SMA_BUFFER else 0.0
    periods = CONFIRMATION_DAYS if USE_CONFIRMATION else 1
    df["BuyThreshold"] = df["BuySMA"] * (1 + buffer)
    df["SellThreshold"] = df["SellSMA"] * (1 - buffer)

    buy_state = confirmed_state(df["Close"] > df["BuyThreshold"], periods)
    sell_state = confirmed_state(df["Close"] < df["SellThreshold"], periods)
    df["SMABuyEvent"] = buy_state & ~buy_state.shift(1, fill_value=False)
    df["SMASellEvent"] = sell_state & ~sell_state.shift(1, fill_value=False)

    df["Prior52WeekLow"] = df["Close"].rolling(252, min_periods=252).min().shift(1)
    df["Prior52WeekHigh"] = df["Close"].rolling(252, min_periods=252).max().shift(1)
    df["LowBuyLevel"] = df["Prior52WeekLow"] * (1 - LOW_BUY_DISCOUNT_PCT)
    df["HighSellLevel"] = df["Prior52WeekHigh"] * (1 + HIGH_SELL_PCT)

    low_condition = df["Prior52WeekLow"].notna() & (df["Close"] <= df["LowBuyLevel"])
    high_condition = df["Prior52WeekHigh"].notna() & (df["Close"] >= df["HighSellLevel"])
    df["LowBuyEvent"] = low_condition & ~low_condition.shift(1, fill_value=False)
    df["HighSellEvent"] = high_condition & ~high_condition.shift(1, fill_value=False)

    cur = df.iloc[-1]
    market_date = df.index[-1].date()
    rsi_passes = (not USE_RSI) or (
        pd.notna(cur["RSI"]) and float(cur["RSI"]) > RSI_BUY_THRESHOLD
    )

    sma_buy = USE_SMA and bool(cur["SMABuyEvent"]) and rsi_passes
    sma_sell = USE_SMA and bool(cur["SMASellEvent"])
    low_buy = USE_52_WEEK and bool(cur["LowBuyEvent"])
    high_sell = USE_52_WEEK and bool(cur["HighSellEvent"])

    buy_triggered = sma_buy or low_buy
    sell_triggered = sma_sell or high_sell

    if buy_triggered and sell_triggered:
        result = "CONFLICTING BUY AND SELL SIGNALS"
    elif buy_triggered:
        result = "BUY"
    elif sell_triggered:
        result = "SELL"
    else:
        result = "NO NEW SIGNAL"

    reasons = []
    if sma_buy:
        reasons.append("SMA buy condition" + (" with RSI filter" if USE_RSI else ""))
    if sma_sell:
        reasons.append("SMA sell condition")
    if low_buy:
        reasons.append("52-week-low buy condition")
    if high_sell:
        reasons.append("52-week-high sell condition")
    reason_text = ", ".join(reasons) if reasons else "None"

    report = f"""Daily {TICKER} Strategy Check

Result: {result}
Market date: {market_date}
Closing price: ${float(cur['Close']):,.2f}
Trigger reason: {reason_text}

Current indicators:
- Buy SMA: ${float(cur['BuySMA']):,.2f}
- Buy threshold: ${float(cur['BuyThreshold']):,.2f}
- Sell SMA: ${float(cur['SellSMA']):,.2f}
- Sell threshold: ${float(cur['SellThreshold']):,.2f}
- RSI ({RSI_PERIOD} days): {float(cur['RSI']):.2f}
- Prior 52-week low: ${float(cur['Prior52WeekLow']):,.2f}
- 52-week-low buy level: ${float(cur['LowBuyLevel']):,.2f}
- Prior 52-week high: ${float(cur['Prior52WeekHigh']):,.2f}
- 52-week-high sell level: ${float(cur['HighSellLevel']):,.2f}

Active settings:
- SMA strategy: {on_off(USE_SMA)}
- Buy SMA: {BUY_DAYS} trading days
- Sell SMA: {SELL_DAYS} trading days
- SMA buffer: {on_off(USE_SMA_BUFFER)} ({SMA_BUFFER_PCT:.2%})
- Confirmation: {on_off(USE_CONFIRMATION)} ({periods} trading day(s))
- RSI buy filter: {on_off(USE_RSI)}
- RSI period: {RSI_PERIOD} trading days
- Buy only when RSI is above: {RSI_BUY_THRESHOLD:g}
- 52-week strategy: {on_off(USE_52_WEEK)}
- Buy at/below prior 52-week low minus: {LOW_BUY_DISCOUNT_PCT:.2%}
- Sell at/above prior 52-week high plus: {HIGH_SELL_PCT:.2%}
- Daily no-signal status email: {on_off(SEND_DAILY_STATUS)}

This application reports research signals only. No order was placed.
"""

    print(report)
    if result != "NO NEW SIGNAL" or SEND_DAILY_STATUS:
        send_email(f"{TICKER}: {result} - {market_date}", report)
        print("Email sent successfully.")
    else:
        print("No email sent because there was no new signal and daily status email is OFF.")


if __name__ == "__main__":
    main()
