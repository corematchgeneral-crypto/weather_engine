#!/usr/bin/env python3
"""#1 -- Immediate per-city calibration from historical data.

For each city, compares the model's archived past forecasts to the observed
actual highs over the last N days (both from Open-Meteo) and computes:
  - station_bias_f  = -mean(forecast - actual)   (corrects systematic bias)
  - error_std_f     = std(forecast - actual)      (realistic per-city spread)

With --apply it writes these into data/calibration.csv, which config.yaml reads,
so the next scan uses calibrated forecasts and realistic uncertainty.

NOTE: this measures error at our configured coordinate vs reanalysis. It fixes
model bias and sets a real error spread, but a pure location mismatch (our
coordinate vs the market's official station) is only fully corrected by the
settlement loop (scripts/fit_calibration.py) using real outcomes. Cities whose
forecast still disagrees wildly with the market after this are likely a station
mismatch -- don't trade them until validated.

Requires internet (run on your own machine).
"""
from pathlib import Path
from datetime import date, timedelta
import argparse
import pandas as pd

from src.config import load_config
from src.weather_data import fetch_archive_highs, fetch_historical_forecast_highs
from src.calibration_fit import summarize_forecast_errors


def main():
    base = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=21, help="history window length in days")
    parser.add_argument("--min-samples", type=int, default=7)
    parser.add_argument("--cities", default=None, help="comma-separated station keys (default: all in config)")
    parser.add_argument("--apply", action="store_true", help="write results into data/calibration.csv")
    args = parser.parse_args()

    cfg = load_config(base / "config.yaml")
    stations = list(cfg.stations.keys())
    if args.cities:
        wanted = {c.strip() for c in args.cities.split(",")}
        stations = [s for s in stations if s in wanted]

    end = date.today() - timedelta(days=1)        # last fully-observed day
    start = end - timedelta(days=args.days - 1)
    print(f"Calibrating {len(stations)} cities over {start} .. {end} (forecast vs actual)\n")

    rows = []
    print(f"{'station':12s} {'city':14s} {'n':>3s} {'mean_err_F':>10s} {'MAE_F':>7s} {'bias_F':>7s} {'std_F':>7s}")
    print("-" * 64)
    for st in stations:
        try:
            fc = fetch_historical_forecast_highs(st, start, end)
            ac = fetch_archive_highs(st, start, end)
        except Exception as e:
            print(f"{st:12s} fetch failed: {e}")
            continue
        errors = [fc[d] - ac[d] for d in fc if d in ac and fc[d] is not None and ac[d] is not None]
        s = summarize_forecast_errors(errors)
        if not s:
            print(f"{st:12s} no overlapping data")
            continue
        city = cfg.stations[st].city
        print(f"{st:12s} {city[:14]:14s} {s['n']:>3} {s['mean_error_f']:>10} {s['mae_f']:>7} {s['station_bias_f']:>7} {str(s['error_std_f']):>7}")
        if s["n"] >= args.min_samples:
            rows.append({"station": st, "station_bias_f": s["station_bias_f"], "error_std_f": s["error_std_f"]})

    if not rows:
        print("\nNo cities had enough samples to calibrate.")
        return

    print(f"\n{len(rows)} cities have >= {args.min_samples} samples.")
    if args.apply:
        calib_path = base / "data" / "calibration.csv"
        pd.DataFrame(rows).to_csv(calib_path, index=False)
        print(f"Wrote calibration for {len(rows)} cities to {calib_path}")
        print("Re-run the scan to get bias-corrected signals with realistic per-city uncertainty.")
    else:
        print("Re-run with --apply to write data/calibration.csv.")


if __name__ == "__main__":
    main()
