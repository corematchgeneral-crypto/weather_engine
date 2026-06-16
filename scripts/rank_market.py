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
    return "n/a" if (x is None or (isinstance(x, float) and pd.isna(x))) else (p.format(x) if isinstance(x, (int, float)) else str(x))


def main():
    base = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-csv", default=str(base / "data" / "polymarket_2026-06-17.csv"))
    parser.add_argument("--top", type=int, default=20, help="how many ranked rows to show")
    parser.add_argument("--min-edge", type=float, default=None, help="only show rows whose best edge >= this")
    args = parser.parse_args()

    cfg = load_config(base / "config.yaml")
    market_csv = Path(args.market_csv)
    mdf = pd.read_csv(market_csv)
    mdf["target_date"] = pd.to_datetime(mdf["target_date"]).dt.date

    stations = sorted(set(mdf["station"].dropna()))
    known = [s for s in stations if s in cfg.stations]
    unknown = [s for s in stations if s not in cfg.stations]
    if unknown:
        print(f"WARNING: stations not in config.yaml (skipped): {unknown}")

    today = pd.Timestamp.today().date()
    start = min([today] + list(mdf["target_date"]))
    end = max(list(mdf["target_date"]))
    print(f"Fetching ensemble forecasts for {len(known)} cities, {start} -> {end} ...")
    forecasts = fetch_forecasts_for_stations(known, start, end)
    print(f"Forecast rows: {len(forecasts)}")

    # Isolate archives for this market so they don't mix with other runs.
    data_dir = base / "data" / "market_runs"
    out = generate_signals(market_csv, forecasts, data_dir=data_dir)
    if out.empty:
        print("No signals produced (no forecast/station overlap?).")
        return

    out = out.copy()
    out["best_edge"] = out[["yes_edge", "no_edge"]].max(axis=1)
    ranked = out.sort_values("best_edge", ascending=False)
    if args.min_edge is not None:
        ranked = ranked[ranked["best_edge"] >= args.min_edge]

    print()
    header = f"{'market':18s} {'bucket':14s} {'cmp':8s} {'fcst_F':>7s} {'P(yes)':>7s} {'yes':>5s} {'no':>5s} {'signal':9s} {'edge':>7s} conf"
    print(header)
    print("-" * len(header))
    for _, r in ranked.head(args.top).iterrows():
        print(f"{str(r.get('market'))[:18]:18s} "
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


if __name__ == "__main__":
    main()
