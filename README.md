# TQQQ Moving Average Strategy — v2

This version adds:

1. Configurable buy moving average.
2. Configurable sell moving average.
3. Backtesting with separate buy/sell averages.
4. A manual latest-signal checker.
5. `daily_signal.py` for unattended daily checks and email alerts.

## Run the web app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints.

## Daily email checker

The daily checker is separate from the web dashboard. This is intentional: a web page should not be responsible for staying alive 24/7.

Set these environment variables:

```text
TICKER=TQQQ.TO
BUY_DAYS=50
SELL_DAYS=50
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_gmail_app_password
ALERT_TO=your_email@gmail.com
```

Then run:

```bash
python daily_signal.py
```

For Gmail, use a Google App Password rather than your normal Gmail password. App passwords require appropriate Google account security settings.

## Automating the daily check

### Easiest beginner approach: GitHub Actions

You can put this project in a private GitHub repository and create:

`.github/workflows/daily_signal.yml`

with:

```yaml
name: Daily TQQQ signal

on:
  schedule:
    - cron: "30 23 * * 1-5"
  workflow_dispatch:

jobs:
  signal:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: python daily_signal.py
        env:
          TICKER: TQQQ.TO
          BUY_DAYS: 50
          SELL_DAYS: 50
          SMTP_HOST: smtp.gmail.com
          SMTP_PORT: 587
          SMTP_USER: ${{ secrets.SMTP_USER }}
          SMTP_PASSWORD: ${{ secrets.SMTP_PASSWORD }}
          ALERT_TO: ${{ secrets.ALERT_TO }}
```

The schedule is UTC and should be adjusted for the desired market-close time and daylight-saving changes. `workflow_dispatch` lets you test it manually.

Store the email credentials as GitHub Actions Secrets. Never put a password directly into the Python file or YAML.

## Important

- This system sends signals; it does NOT place live trades.
- The daily checker should run only after the relevant exchange has closed and the final daily candle is available.
- Yahoo Finance is being used as a convenient data source for this prototype. Before relying on alerts for real money, use a broker/data feed with reliable market data and verify the signal.
- If you use TQQQ.TO, remember the ETF's short trading history. For long backtests, TQQQ (U.S.) can be used as a historical research proxy, but it is not identical to the Canadian ETF.
- Before live trading, add persistent state, duplicate-alert protection, data validation, market-holiday handling, logging, and a kill switch.
