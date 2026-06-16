#!/usr/bin/env python3
"""Paste a Polymarket event link -> pull the full bucket ladder + live prices,
then forecast and rank the buckets by edge.

Usage:
  python -m scripts.fetch_polymarket --url https://polymarket.com/event/highest-temperature-in-paris-on-june-16-2026
  python -m scripts.fetch_polymarket --slug highest-temperature-in-paris-on-june-17-2026 --no-rank

Requires internet (run on your own machine).
"""
from pathlib import Path
import argparse
import pandas as pd

from src.config import load_config
from src.polymarket_data import slug_from_url, fetch_event, event_to_rows, city_to_station_map
from scripts.rank_market import rank_and_print


def main():
    base = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--url", help="Polymarket event URL")
    g.add_argument("--slug", help="Polymarket event slug")
    parser.add_argument("--out", default=str(base / "data" / "polymarket_live.csv"))
    parser.add_argument("--no-rank", action="store_true", help="only write the CSV, do not forecast/rank")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    slug = slug_from_url(args.url or args.slug)
    print(f"Fetching Polymarket event: {slug}")
    event = fetch_event(slug)

    cfg = load_config(base / "config.yaml")
    city_map = city_to_station_map(cfg)

    rows = event_to_rows(event)
    if not rows:
        print("No bucket contracts parsed from this event.")
        return

    # Resolve the station for each row by matching the city name to config.
    city = rows[0].get("city")
    station = city_map.get(str(city).strip().lower()) if city else None
    if station is None:
        print(f"WARNING: city {city!r} is not in config.yaml. Add its coordinates "
              f"(lat/lon/timezone) under stations: to forecast it. Writing CSV anyway.")
    for r in rows:
        r["station"] = station or (str(city).upper().replace(" ", "") if city else "")

    out_path = Path(args.out)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} bucket rows for '{city}' to {out_path}")
    print(df[["notes", "comparison", "threshold_f", "threshold_high", "unit", "yes_ask", "no_ask", "volume"]].to_string(index=False))

    if not args.no_rank:
        print()
        rank_and_print(out_path, top=args.top)


if __name__ == "__main__":
    main()
