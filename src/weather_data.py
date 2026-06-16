from typing import Dict, Any
from pathlib import Path
import requests
import pandas as pd
from datetime import datetime, date, timezone
from src.config import load_config


def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def fetch_open_meteo_daily(lat: float, lon: float, timezone: str, start_date: str, end_date: str) -> Dict[str, Any]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
            "cloudcover_mean",
            "windspeed_10m_max",
        ]),
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def build_forecast_df(station: str, start_date: date, end_date: date, source: str = "open-meteo") -> pd.DataFrame:
    cfg = load_config(Path(__file__).parent.parent / "config.yaml")
    stations = cfg.stations
    if station not in stations:
        raise KeyError(f"Station {station} not in config")
    s = stations[station]
    start = start_date.isoformat()
    end = end_date.isoformat()
    raw = None
    if source == "open-meteo":
        raw = fetch_open_meteo_daily(s.lat, s.lon, s.timezone, start, end)
    else:
        # Unknown source: fall back to open-meteo
        raw = fetch_open_meteo_daily(s.lat, s.lon, s.timezone, start, end)
    # Parse daily arrays into rows
    daily = raw.get("daily", {})
    dates = daily.get("time", [])
    highs_c = daily.get("temperature_2m_max", [])
    lows_c = daily.get("temperature_2m_min", [])
    precip_p = daily.get("precipitation_probability_max", [])
    cloud_mean = daily.get("cloudcover_mean", [])
    wind_max = daily.get("windspeed_10m_max", [])

    rows = []
    run_time_utc = datetime.now(timezone.utc)
    for i, d in enumerate(dates):
        try:
            high_c = highs_c[i]
        except Exception:
            high_c = None
        try:
            low_c = lows_c[i]
        except Exception:
            low_c = None
        rows.append({
            "station": station,
            "target_date": pd.to_datetime(d).date(),
            "forecast_source": source,
            "run_time_utc": run_time_utc,
            "predicted_high_f": c_to_f(high_c) if high_c is not None else None,
            "predicted_low_f": c_to_f(low_c) if low_c is not None else None,
            "precip_probability": float(precip_p[i]) if i < len(precip_p) else None,
            "cloud_cover": float(cloud_mean[i]) if i < len(cloud_mean) else None,
            "wind_speed": float(wind_max[i]) if i < len(wind_max) else None,
            "raw_json_path_or_summary": "open-meteo",
        })
    df = pd.DataFrame(rows)
    return df


def fetch_forecasts_for_stations(station_list, start_date, end_date, sources=None):
    """
    Fetch forecasts for multiple stations and optionally multiple sources.
    `sources` is a list of source names (e.g. ["open-meteo"]).
    """
    if sources is None:
        sources = ["open-meteo"]
    all_df = []
    for src in sources:
        for st in station_list:
            try:
                df = build_forecast_df(st, start_date, end_date, source=src)
                all_df.append(df)
            except Exception as e:
                print(f"Failed to fetch for {st} source {src}: {e}")
    if all_df:
        return pd.concat(all_df, ignore_index=True)
    return pd.DataFrame()
