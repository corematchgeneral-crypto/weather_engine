"""Weather forecast ingestion from Open-Meteo (free, no API key).

By default this now fetches several deterministic weather models in a single
request (an ensemble). Each model becomes its own forecast row, which lets the
signal layer compute both the ensemble mean (point forecast) and the spread
across models (a live measure of forecast uncertainty).
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import requests
import pandas as pd
from datetime import datetime, date, timezone

from src.config import load_config


# Free deterministic models available from Open-Meteo's forecast endpoint.
# Using several gives us a real ensemble spread instead of one public forecast.
DEFAULT_MODELS: List[str] = [
    "gfs_seamless",        # NOAA GFS (USA)
    "ecmwf_ifs025",        # ECMWF IFS 0.25 deg (Europe)
    "icon_seamless",       # DWD ICON (Germany)
    "gem_seamless",        # Environment Canada GEM
]

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_probability_max",
    "cloudcover_mean",
    "windspeed_10m_max",
]

# Map our output column -> Open-Meteo daily variable base name.
_VARIABLE_MAP = {
    "predicted_high_f": ("temperature_2m_max", True),   # True => convert C->F
    "predicted_low_f": ("temperature_2m_min", True),
    "precip_probability": ("precipitation_probability_max", False),
    "cloud_cover": ("cloudcover_mean", False),
    "wind_speed": ("windspeed_10m_max", False),
}

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def fetch_open_meteo_daily(
    lat: float,
    lon: float,
    timezone: str,
    start_date: str,
    end_date: str,
    models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARIABLES),
    }
    if models:
        params["models"] = ",".join(models)
    r = requests.get(FORECAST_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def _model_suffixes(daily: Dict[str, Any], base: str = "temperature_2m_max") -> List[str]:
    """Discover which per-model suffixes are present in the response.

    Open-Meteo returns plain keys (``temperature_2m_max``) for a single model
    and suffixed keys (``temperature_2m_max_gfs_seamless``) when multiple models
    are requested. Returns a list of suffixes ("" means the plain key).
    """
    suffixes = []
    for k in daily.keys():
        if k == base:
            suffixes.append("")
        elif k.startswith(base + "_"):
            suffixes.append(k[len(base) + 1:])
    return suffixes or [""]


def _parse_daily(
    daily: Dict[str, Any],
    station: str,
    requested_models: Optional[List[str]],
    run_time_utc: datetime,
) -> List[dict]:
    times = daily.get("time", [])
    suffixes = _model_suffixes(daily)
    rows: List[dict] = []

    for suffix in suffixes:
        if suffix:
            model_name = suffix
        elif requested_models and len(requested_models) == 1:
            model_name = requested_models[0]
        else:
            model_name = "open-meteo"

        for i, d in enumerate(times):
            row = {
                "station": station,
                "target_date": pd.to_datetime(d).date(),
                "forecast_source": model_name,
                "run_time_utc": run_time_utc,
                "raw_json_path_or_summary": f"open-meteo:{model_name}",
            }
            for out_col, (base, convert) in _VARIABLE_MAP.items():
                key = base if not suffix else f"{base}_{suffix}"
                arr = daily.get(key, [])
                val = arr[i] if i < len(arr) else None
                if val is None:
                    row[out_col] = None
                elif convert:
                    row[out_col] = c_to_f(float(val))
                else:
                    row[out_col] = float(val)
            rows.append(row)
    return rows


def build_forecast_df(
    station: str,
    start_date: date,
    end_date: date,
    source: str = "open-meteo",
    models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build a forecast DataFrame for one station.

    If ``models`` is provided (or ``source`` requests an ensemble), one row per
    model is returned so the caller can compute mean and spread.
    """
    cfg = load_config(Path(__file__).parent.parent / "config.yaml")
    stations = cfg.stations
    if station not in stations:
        raise KeyError(f"Station {station} not in config")
    s = stations[station]

    if models is None and source in ("open-meteo-multi", "ensemble"):
        models = DEFAULT_MODELS

    raw = fetch_open_meteo_daily(
        s.lat, s.lon, s.timezone, start_date.isoformat(), end_date.isoformat(), models=models
    )
    daily = raw.get("daily", {})
    run_time_utc = datetime.now(timezone.utc)
    rows = _parse_daily(daily, station, models, run_time_utc)
    return pd.DataFrame(rows)


def fetch_forecasts_for_stations(
    station_list,
    start_date,
    end_date,
    sources=None,
    models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fetch forecasts for multiple stations.

    By default a multi-model ensemble (``DEFAULT_MODELS``) is fetched so the
    signal layer gets a real spread. Pass ``models=["open-meteo"]`` or a single
    model name to fall back to a single deterministic forecast.
    """
    if models is None:
        if sources is None or sources == ["open-meteo"]:
            models = DEFAULT_MODELS
        else:
            # treat explicit source names as Open-Meteo model ids
            models = sources

    all_df = []
    for st in station_list:
        try:
            df = build_forecast_df(st, start_date, end_date, models=models)
            all_df.append(df)
        except Exception as e:  # network/model errors are per-station, non-fatal
            print(f"Failed to fetch for {st}: {e}")
    if all_df:
        return pd.concat(all_df, ignore_index=True)
    return pd.DataFrame()


def fetch_actual_high_low_f(station: str, target_date: date) -> Dict[str, Optional[float]]:
    """Fetch the observed daily high AND low (deg F) for a past date.

    Uses Open-Meteo's historical archive endpoint. Returns
    ``{"high": float|None, "low": float|None}``.

    IMPORTANT: This is reanalysis/observation data and is only an *approximation*
    of the official ForecastEx settlement value. ForecastEx resolves on the
    whole-degree value shown in the Weather Underground daily summary table for
    the specified station. For real/traded contracts, enter the official Weather
    Underground value via scripts/add_settlement.py. Use this archive fetch only
    for rough forecast-accuracy tracking.
    """
    cfg = load_config(Path(__file__).parent.parent / "config.yaml")
    s = cfg.stations.get(station)
    if not s:
        raise KeyError(f"Station {station} not in config")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": s.lat,
        "longitude": s.lon,
        "timezone": s.timezone,
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    daily = r.json().get("daily", {})
    hi = daily.get("temperature_2m_max", [])
    lo = daily.get("temperature_2m_min", [])
    return {
        "high": c_to_f(float(hi[0])) if (hi and hi[0] is not None) else None,
        "low": c_to_f(float(lo[0])) if (lo and lo[0] is not None) else None,
    }


def fetch_actual_high_f(station: str, target_date: date) -> Optional[float]:
    """Backward-compatible helper: observed daily high (deg F), archive proxy."""
    return fetch_actual_high_low_f(station, target_date)["high"]



ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def _fetch_daily_max_f(station: str, start_date, end_date, base_url: str, timeout: int = 30) -> Dict[str, Optional[float]]:
    """Fetch daily max temperature (deg F) keyed by ISO date from an Open-Meteo endpoint."""
    cfg = load_config(Path(__file__).parent.parent / "config.yaml")
    s = cfg.stations.get(station)
    if not s:
        raise KeyError(f"Station {station} not in config")
    params = {
        "latitude": s.lat,
        "longitude": s.lon,
        "timezone": s.timezone,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "daily": "temperature_2m_max",
    }
    r = requests.get(base_url, params=params, timeout=timeout)
    r.raise_for_status()
    daily = r.json().get("daily", {})
    out: Dict[str, Optional[float]] = {}
    for t, v in zip(daily.get("time", []), daily.get("temperature_2m_max", [])):
        out[t] = c_to_f(float(v)) if v is not None else None
    return out


def fetch_archive_highs(station: str, start_date, end_date) -> Dict[str, Optional[float]]:
    """Observed daily highs (deg F) from the Open-Meteo archive (reanalysis)."""
    return _fetch_daily_max_f(station, start_date, end_date, ARCHIVE_URL)


def fetch_historical_forecast_highs(station: str, start_date, end_date) -> Dict[str, Optional[float]]:
    """The model's archived past forecasts of daily highs (deg F)."""
    return _fetch_daily_max_f(station, start_date, end_date, HISTORICAL_FORECAST_URL)
