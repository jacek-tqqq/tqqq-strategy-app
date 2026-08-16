"""
Daily signal checker.

Run this once after the market closes, e.g. with:
    python daily_signal.py

Set environment variables before running:
    TICKER=TQQQ.TO
    BUY_DAYS=50
    SELL_DAYS=50
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=your_email@gmail.com
    SMTP_PASSWORD=your_app_password
    ALERT_TO=your_email@gmail.com

The checker sends an email ONLY when a new buy/sell crossover is detected.
"""

import os
import smtplib
from email.message import EmailMessage

import pandas as pd
import yfinance as yf

TICKER = os.getenv("TICKER", "TQQQ.TO")
BUY_DAYS = int(os.getenv("BUY_DAYS", "50"))
SELL_DAYS = int(os.getenv("SELL_DAYS", "50"))

def get_data():
    df = yf.download(
        TICKER,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])

def send_email(subject, body):
    required = ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "ALERT_TO"]
    missing = [x for x in required if not os.getenv(x)]
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

def main():
    df = get_data()
    needed = max(BUY_DAYS, SELL_DAYS) + 2
    if len(df) < needed:
        raise RuntimeError("Not enough data to calculate the moving averages.")

    df["BuySMA"] = df["Close"].rolling(BUY_DAYS).mean()
    df["SellSMA"] = df["Close"].rolling(SELL_DAYS).mean()

    prev = df.iloc[-2]
    cur = df.iloc[-1]
    date = df.index[-1].date()

    buy = prev["Close"] <= prev["BuySMA"] and cur["Close"] > cur["BuySMA"]
    sell = prev["Close"] >= prev["SellSMA"] and cur["Close"] < cur["SellSMA"]

    if buy:
        subject = f"TQQQ BUY signal — {TICKER}"
        body = (
            f"{TICKER} BUY signal on {date}\n\n"
            f"Close: {cur['Close']:.4f}\n"
            f"{BUY_DAYS}-day SMA: {cur['BuySMA']:.4f}\n"
            f"Condition: price crossed above the buy moving average.\n\n"
            "No order was placed."
        )
        send_email(subject, body)
        print(subject)
    elif sell:
        subject = f"TQQQ SELL signal — {TICKER}"
        body = (
            f"{TICKER} SELL signal on {date}\n\n"
            f"Close: {cur['Close']:.4f}\n"
            f"{SELL_DAYS}-day SMA: {cur['SellSMA']:.4f}\n"
            f"Condition: price crossed below the sell moving average.\n\n"
            "No order was placed."
        )
        send_email(subject, body)
        print(subject)
    else:
        print(f"{date}: no new signal.")

if __name__ == "__main__":
    main()
