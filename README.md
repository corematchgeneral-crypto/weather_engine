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

## Contract types & settlement (ForecastEx)

ForecastEx "Daily Temperature" contracts ask: *"Will the [high/low/avg] temperature
in [region] [exceed/be below] [N] F on [date]?"* and settle on the whole-degree
value in the **Weather Underground** daily summary table for the contract's station.

The engine models this directly:

- **Direction** — `ABOVE` ("exceed N": YES if value > N) or `BELOW` ("be below N":
  YES if value < N). Integer settlement is handled by shifting the threshold by
  +0.5 (ABOVE) or -0.5 (BELOW), so "exceed 72" wins only at 73+.
- **Underlying** — `HIGH`, `LOW`, or `AVG` (average of high and low).

Add optional `underlying` and `direction` columns to `data/market_prices_sample.csv`
(both default to `HIGH` / `ABOVE` if omitted), e.g.:

```
timestamp_utc,market,city,station,target_date,threshold_f,underlying,direction,yes_ask,no_ask,...
2026-06-16T10:30:00Z,Chicago High,Chicago,KMDW,2026-06-17,72,HIGH,ABOVE,0.62,0.41,...
```

### Settlement source — IMPORTANT

Official settlement is the **Weather Underground** daily "Actual" High/Low value for
the contract's station. For real/traded contracts, record the official value with:

```bash
python -m scripts.add_settlement --station KMDW --target-date 2026-06-17 \
    --final-high-f 84 --source "Weather Underground (KMDW)"
# AVG/LOW contracts: also pass --final-low-f
```

`scripts/backfill_settlements.py` (Open-Meteo archive) is a **proxy for testing
forecast accuracy only** — its values can differ from Weather Underground by a
degree or two, which flips outcomes near the threshold. Don't use it to judge real
trade win-rates; use the official values via `add_settlement`.

The per-station calibration loop corrects systematic offset between the Open-Meteo
*forecast* and the Weather Underground *settlement* — provided your settlements are
the official WU values.

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

### Calibrate cities first (recommended before trusting signals)

```bash
python -m scripts.calibrate_stations --days 21          # preview per-city bias + error spread
python -m scripts.calibrate_stations --days 21 --apply  # write data/calibration.csv
```

This compares each city's archived past forecasts to observed actuals and sets a
per-city `station_bias_f` (bias correction) and `error_std_f` (realistic
uncertainty). It corrects model bias and right-sizes the bucket probabilities.
A residual location mismatch (our coordinate vs the market's official station)
is only fully corrected by the settlement loop below.

### Then the settlement loop
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
