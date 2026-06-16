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

final_high ~ Normal(predicted_high + station_bias, sigma)

For integer settlement observations we model rounding by evaluating probabilities at `threshold + 0.5` (configurable).

### v2: ensemble + lead-time aware uncertainty (default)

`sigma` is no longer a single fixed number. It is computed as:

- the per-station base error std (`error_std_f`), scaled up with **forecast lead time**
  (a 7-day-out bet is far less certain than a same-day bet), then
- widened by **ensemble spread** — the disagreement among several weather models
  (GFS, ECMWF, ICON, GEM) fetched from Open-Meteo in one request, then
- floored at `min_sigma_f` to avoid overconfident 0/1 probabilities.

Tunables live under `defaults:` in `config.yaml` (`lead_time_slope`, `lead_time_ref_days`,
`lead_time_cap_mult`, `ensemble_spread_inflation`, `ensemble_blend`, `min_sigma_f`).
Set `method="normal_error_v1"` to fall back to the legacy fixed-sigma model.

## Measuring accuracy over time

The goal is to see, as data accumulates, how accurate the engine is — without
placing any trades.

1. Generate signals over time (writes `data/signals.csv`, archives snapshots):

   ```bash
   python -m scripts.fetch_and_signal          # or scripts.run_signal_engine
   ```

2. Auto-fill actual outcomes for past dates from the Open-Meteo archive
   (no manual entry; writes `data/settlements.csv`):

   ```bash
   python -m scripts.backfill_settlements        # add --dry-run to preview
   ```

3. Run the accuracy report:

   ```bash
   python -m scripts.run_accuracy_report
   ```

   This produces:
   - **Forecast accuracy** (MAE / bias / RMSE) by station, by model, and by lead time
     → `data/forecast_accuracy_by_*.csv`
   - **Signal win-rate over time** (cumulative + rolling) → `data/signal_winrate_timeseries.csv`
   - **Suggested calibration** (per-station bias + error std) → `data/calibration_suggested.csv`;
     copy values into `data/calibration.csv` to self-tune the model.

Note: archive observations are a good proxy for settlement but may not exactly
match the official ForecastEx settlement source — verify before relying on them.

## Risk & Disclaimer

This software is for research and educational purposes only. Binary options and event contracts can lose 100% of stake. This is not financial advice.

## Extending to brokers

The codebase is designed so a broker integration (eg. IBKR) can be added later. Version 1 purposely reads market data from CSV and does not place orders.


## Contact

If you want additional features (ensemble models, historical calibration, live market connectors), open an issue or ask.
