from pathlib import Path
from src.weather_data import fetch_forecasts_for_stations
from src.signals import generate_signals
from src.config import load_config
import pandas as pd
import argparse


def main():
    base = Path(__file__).parent.parent
    cfg = load_config(base / "config.yaml")
    market_path = base / "data" / "market_prices_sample.csv"
    stations = list(cfg.stations.keys())
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", help="start date yyyy-mm-dd", required=False)
    parser.add_argument("--end", help="end date yyyy-mm-dd", required=False)
    parser.add_argument("--sources", help="comma separated sources", required=False)
    args = parser.parse_args()

    if args.start:
        start = pd.to_datetime(args.start).date()
    else:
        start = pd.to_datetime("today").date()
    if args.end:
        end = pd.to_datetime(args.end).date()
    else:
        end = start

    sources = args.sources.split(",") if args.sources else None
    forecasts = fetch_forecasts_for_stations(stations, start, end, sources=sources)
    out = generate_signals(market_path, forecasts)
    print(f"Signals generated: {len(out)}")


if __name__ == "__main__":
    main()
