#!/usr/bin/env python3
"""Run the full accuracy report: trade evaluation + forecast accuracy + win-rate.

This is the "how accurate is the tool" command. It:
  1. evaluates settled signals (PnL, CLV, calibration) -> evaluated_trades.csv
  2. computes forecast accuracy (MAE/bias/RMSE by station, model, lead time)
  3. builds the cumulative/rolling signal win-rate time series

Run after you have collected forecasts, market prices/signals, and settlements
(use scripts/backfill_settlements.py to auto-fill past outcomes).
"""
from pathlib import Path
import json

from src.historical_evaluation import evaluate_signals
from src.forecast_accuracy import compute_forecast_accuracy, compute_winrate_timeseries


def _fmt(x):
    return "n/a" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))


def main():
    base = Path(__file__).resolve().parent.parent / "data"

    print("=" * 60)
    print("1) TRADE EVALUATION (settled signals)")
    print("=" * 60)
    ev = evaluate_signals(base / "signals.csv", base / "market_snapshots.csv", base / "settlements.csv")
    if "error" in ev:
        print("  ", ev["error"])
    else:
        for k, v in ev["metrics"].items():
            print(f"  {k:28s}: {_fmt(v)}")

    print()
    print("=" * 60)
    print("2) FORECAST ACCURACY (predicted high vs actual)")
    print("=" * 60)
    acc = compute_forecast_accuracy(base / "forecast_snapshots.csv", base / "settlements.csv", data_dir=base)
    if "error" in acc:
        print("  ", acc["error"])
    else:
        ov = acc["overall"]
        print(f"  overall: n={ov.get('n')}  MAE={_fmt(ov.get('mae_f'))}F  bias={_fmt(ov.get('bias_f'))}F  RMSE={_fmt(ov.get('rmse_f'))}F")
        print("  by station:")
        for row in acc["by_station"]:
            print(f"    {row['station']:6s} n={row['n']:>3}  MAE={_fmt(row['mae_f'])}F  bias={_fmt(row['bias_f'])}F  RMSE={_fmt(row['rmse_f'])}F")
        if acc["by_source"]:
            print("  by model (which forecast model is most accurate):")
            for row in acc["by_source"]:
                print(f"    {row['station']:6s} {str(row['source_name']):16s} n={row['n']:>3}  MAE={_fmt(row['mae_f'])}F  bias={_fmt(row['bias_f'])}F")
        if acc["suggested_calibration"]:
            print("  suggested calibration (copy into data/calibration.csv to self-tune):")
            for row in acc["suggested_calibration"]:
                print(f"    {row['station']:6s} bias={row['station_bias_f']}  error_std={row['error_std_f']}  (n={row['n']})")

    print()
    print("=" * 60)
    print("3) SIGNAL WIN-RATE OVER TIME")
    print("=" * 60)
    wr = compute_winrate_timeseries(base / "evaluated_trades.csv", data_dir=base)
    if "error" in wr:
        print("  ", wr["error"])
    else:
        print(f"  settled trades        : {wr['n_settled_trades']}")
        print(f"  overall win rate      : {_fmt(wr['overall_win_rate'])}")
        print(f"  latest rolling ({wr['rolling_window']}) : {_fmt(wr['latest_rolling_win_rate'])}")
        print(f"  total pnl (per 1 unit): {_fmt(wr['total_pnl'])}")
        print(f"  timeseries written to : {wr['timeseries_path']}")


if __name__ == "__main__":
    main()
