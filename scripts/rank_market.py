#!/usr/bin/env python3
"""Fetch forecasts for the cities in a market CSV and rank every contract by edge.

Works for any market file (ForecastEx or Polymarket-style buckets). It only
fetches forecasts for stations that exist in config.yaml; unknown stations are
skipped with a warning.

Usage:
  python -m scripts.rank_market                              # uses Polymarket file
  python -m scripts.rank_market --market-csv data/foo.csv --top 15
"""
from pathlib import Path
import argparse
import pandas as pd

from src.config import load_config
from src.weather_data import fetch_forecasts_for_stations
from src.signals import generate_signals


def _fmt(x, p="{:.3f}"):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return p.format(x) if isinstance(x, (int, float)) else str(x)


def rank_and_print(market_csv, top=20, min_edge=None):
    """Fetch forecasts for the market file's cities and print contracts by edge."""
    base = Path(__file__).resolve().parent.parent
    cfg = load_config(base / "config.yaml")
    market_csv = Path(market_csv)
    mdf = pd.read_csv(market_csv)
    mdf["target_date"] = pd.to_datetime(mdf["target_date"]).dt.date

    stations = sorted(set(str(s) for s in mdf["station"].dropna() if str(s).strip()))
    known = [s for s in stations if s in cfg.stations]
    unknown = [s for s in stations if s not in cfg.stations]
    if unknown:
        print(f"WARNING: stations not in config.yaml (skipped): {unknown}")
    if not known:
        print("No known stations to forecast. Add their coordinates to config.yaml.")
        return

    today = pd.Timestamp.today().date()
    start = min([today] + list(mdf["target_date"]))
    end = max(list(mdf["target_date"]))
    print(f"Fetching ensemble forecasts for {len(known)} cities, {start} -> {end} ...")
    forecasts = fetch_forecasts_for_stations(known, start, end)
    print(f"Forecast rows: {len(forecasts)}")

    data_dir = base / "data" / "market_runs"
    out = generate_signals(market_csv, forecasts, data_dir=data_dir)
    if out.empty:
        print("No signals produced (no forecast/station overlap?).")
        return

    out = out.copy()
    out["best_edge"] = out[["yes_edge", "no_edge"]].max(axis=1)
    out["_is_trade"] = out["signal"].isin(["BUY_YES", "BUY_NO"]).astype(int)
    # Real trades first, then by edge (suppressed/no-trade rows sink to the bottom)
    ranked = out.sort_values(["_is_trade", "best_edge"], ascending=[False, False])
    if min_edge is not None:
        ranked = ranked[ranked["best_edge"] >= min_edge]

    print()
    header = f"{'city':12s} {'date':5s} {'bucket':14s} {'cmp':8s} {'fcst_F':>7s} {'P(yes)':>7s} {'yes':>5s} {'no':>5s} {'signal':9s} {'edge':>7s} conf"
    print(header)
    print("-" * len(header))
    for _, r in ranked.head(top).iterrows():
        td = str(r.get("target_date"))[5:] if r.get("target_date") is not None else ""
        print(f"{str(r.get('city'))[:12]:12s} "
              f"{td:5s} "
              f"{str(r.get('notes'))[:14]:14s} "
              f"{str(r.get('comparison'))[:8]:8s} "
              f"{_fmt(r.get('predicted_value_f'),'{:.1f}'):>7s} "
              f"{_fmt(r.get('model_prob_yes')):>7s} "
              f"{_fmt(r.get('yes_ask'),'{:.2f}'):>5s} "
              f"{_fmt(r.get('no_ask'),'{:.2f}'):>5s} "
              f"{str(r.get('signal')):9s} "
              f"{_fmt(r.get('best_edge')):>7s} "
              f"{r.get('confidence_label')}")

    n_buy = int((ranked["signal"].isin(["BUY_YES", "BUY_NO"])).sum())
    print(f"\n{n_buy} contract(s) flagged as a trade (edge >= minimum). "
          f"Signals are research-only; verify each market's settlement source before trading.")
    return ranked


def main():
    base = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-csv", default=str(base / "data" / "polymarket_2026-06-17.csv"))
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--min-edge", type=float, default=None)
    args = parser.parse_args()
    rank_and_print(args.market_csv, top=args.top, min_edge=args.min_edge)


if __name__ == "__main__":
    main()
