#!/usr/bin/env python3
"""Scan ALL Polymarket daily-temperature markets across cities and dates,
forecast each, and rank every contract by edge.

For each known city and each date in the window (today .. today+N), it builds
the Polymarket event slug, pulls the live bucket ladder + prices, forecasts the
city, and ranks everything. Same-day and bucket-agreement guards apply, so only
genuine forecast disagreements survive as trade signals.

Usage:
  python -m scripts.scan_polymarket                 # today..today+2, highs
  python -m scripts.scan_polymarket --days 3 --include-lows --top 40
  python -m scripts.scan_polymarket --cities paris,london,seoul

Requires internet (run on your own machine).
"""
from pathlib import Path
from datetime import date, timedelta
import argparse
import pandas as pd

from src.polymarket_data import CITY_SLUGS, build_event_slug, fetch_event_safe, event_to_rows
from scripts.rank_market import rank_and_print


def main():
    base = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=3, help="number of days from today to scan (today..today+days-1)")
    parser.add_argument("--include-lows", action="store_true", help="also scan 'lowest temperature' markets")
    parser.add_argument("--cities", default=None, help="comma-separated city slugs to limit the scan")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--out", default=str(base / "data" / "polymarket_scan.csv"))
    args = parser.parse_args()

    today = date.today()
    dates = [today + timedelta(days=i) for i in range(max(1, args.days))]
    kinds = ["highest", "lowest"] if args.include_lows else ["highest"]

    cities = CITY_SLUGS
    if args.cities:
        wanted = {c.strip().lower() for c in args.cities.split(",")}
        cities = {k: v for k, v in CITY_SLUGS.items() if k in wanted}

    print(f"Scanning {len(cities)} cities x {len(dates)} day(s) x {len(kinds)} kind(s) "
          f"= up to {len(cities)*len(dates)*len(kinds)} events ...")

    rows = []
    found_events = 0
    for city_slug, station in cities.items():
        for kind in kinds:
            for d in dates:
                slug = build_event_slug(kind, city_slug, d)
                ev = fetch_event_safe(slug)
                if not ev:
                    continue
                ev_rows = event_to_rows(ev, station=station)
                if ev_rows:
                    found_events += 1
                    rows.extend(ev_rows)
        print(f"  {city_slug:14s} done")

    if not rows:
        print("No Polymarket temperature events found. The slug format may have "
              "changed, or none are open for these dates. Try --cities paris to test one.")
        return

    out_path = Path(args.out)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nFound {found_events} events, {len(rows)} bucket contracts. Wrote {out_path}")
    print("Forecasting and ranking (same-day & bucket-agreement guards applied)...\n")
    rank_and_print(out_path, top=args.top)


if __name__ == "__main__":
    main()
