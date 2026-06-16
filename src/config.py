from pathlib import Path
from typing import Dict, Any
import csv
import yaml
from pydantic import BaseModel


class StationConfig(BaseModel):
    city: str
    lat: float
    lon: float
    timezone: str
    station_bias_f: float = 0.0
    error_std_f: float = 2.5


class Config(BaseModel):
    stations: Dict[str, StationConfig]
    defaults: Dict[str, Any] = {}


def load_config(path: str | Path = None) -> Config:
    if path is None:
        p = Path(__file__).parent.parent / "config.yaml"
    else:
        p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    raw = yaml.safe_load(p.read_text())
    stations = {k: StationConfig(**v) for k, v in raw.get("stations", {}).items()}
    defaults = raw.get("defaults", {})
    # Optional calibration CSV to override station bias/error
    calib_path = defaults.get("calibration_csv")
    if calib_path:
        calib_file = Path(calib_path)
        if not calib_file.is_absolute():
            calib_file = p.parent / calib_file
        if calib_file.exists():
            try:
                with calib_file.open() as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        st = row.get("station")
                        if not st:
                            continue
                        if st in stations:
                            b = row.get("station_bias_f")
                            e = row.get("error_std_f")
                            try:
                                if b is not None and b != "":
                                    stations[st].station_bias_f = float(b)
                                if e is not None and e != "":
                                    stations[st].error_std_f = float(e)
                            except Exception:
                                # ignore parse errors
                                pass
            except Exception:
                pass

    return Config(stations=stations, defaults=defaults)
