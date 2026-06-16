from pathlib import Path
import pandas as pd
from typing import List
from datetime import datetime


def load_market_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Market CSV not found: {p}")
    df = pd.read_csv(p, parse_dates=["timestamp_utc", "target_date"], keep_default_na=True)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df
