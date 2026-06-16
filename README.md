# Weather Signal Engine (ForecastEx / IBKR ForecastTrader MVP)

This repository is a research-only signal engine that estimates the probability that daily high-temperature binary contracts ("Above X°F") will resolve YES/NO, compares model probabilities to market prices, and emits positive expected value signals.

IMPORTANT: This is NOT an auto-trading bot. Do not connect to brokers or place real trades with this code.

## Features

- Read manual market prices from `data/market_prices_sample.csv`
- Fetch weather forecasts from Open-Meteo (no API key required)
- Compute a baseline probabilistic forecast using a normal error model
- Compare model probability to market prices to compute edge
- Emit signals (BUY_YES / BUY_NO / NO_TRADE) with risk metrics
- Streamlit dashboard for interactive exploration
- Simple backtest framework and tests

## Project structure

weather_signal_engine/
- README.md
- requirements.txt
- .env.example
- config.yaml
- data/
  - market_prices_sample.csv
  - signals.csv
  - settlements.csv
- src/
  - __init__.py
  - config.py
  - models.py
  - market_data.py
  - weather_data.py
  - probability_model.py
  - signals.py
  - backtest.py
  - dashboard.py
  - utils.py
- tests/
  - test_probability_model.py
  - test_signals.py

## Quick start (Windows)

Open PowerShell or CMD and run:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```powershell
streamlit run src/dashboard.py
```

Run tests:

```powershell
pytest -q
```

## How to use

- Edit `data/market_prices_sample.csv` to add market snapshots. Fields explained in the CSV header.
- Start the dashboard to fetch fresh forecasts and compute signals.
- Generated signals are written to `data/signals.csv` for later backtesting.

## Probability model

We use a normal-distribution error model around the forecasted daily high:

final_high ~ Normal(predicted_high + station_bias, error_std)

For integer settlement observations we model rounding by evaluating probabilities at `threshold + 0.5` (configurable).

## Risk & Disclaimer

This software is for research and educational purposes only. Binary options and event contracts can lose 100% of stake. This is not financial advice.

## Extending to brokers

The codebase is designed so a broker integration (eg. IBKR) can be added later. Version 1 purposely reads market data from CSV and does not place orders.


## Contact

If you want additional features (ensemble models, historical calibration, live market connectors), open an issue or ask.
