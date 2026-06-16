#!/usr/bin/env python3
from pathlib import Path
from datetime import date, timedelta
import traceback
import sys

from src.config import load_config
from src.weather_data import fetch_forecasts_for_stations
from src.signals import generate_signals


def main():
    try:
        root = Path(__file__).resolve().parent.parent
        cfg = load_config(root / 'config.yaml')
        stations = list(cfg.stations.keys())
        start = date.today()
        end = start + timedelta(days=2)
        print(f'Fetching forecasts for stations: {stations} from {start} to {end}')
        forecasts = fetch_forecasts_for_stations(stations, start, end)
        print('Forecasts fetched rows:', len(forecasts))
        market_csv = root / 'data' / 'market_prices_sample.csv'
        print('Loading market CSV at', market_csv)
        out = generate_signals(market_csv, forecasts, fee_buffer=cfg.defaults.get('fee_buffer', None), minimum_edge=cfg.defaults.get('minimum_edge', None), integer_settlement_mode=cfg.defaults.get('integer_settlement_mode', True))
        print('Signals generated:', len(out))
        if not out.empty:
            cols = ['timestamp_utc','market','station','target_date','threshold_f','signal','yes_edge','no_edge','confidence_label']
            print(out[cols].to_string(index=False))
        print('Signals saved to data/signals.csv')
    except Exception as e:
        print('Error during fetch/signaling:', e)
        traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    main()
