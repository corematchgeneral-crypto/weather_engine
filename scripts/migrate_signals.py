#!/usr/bin/env python3
"""Migration helper: normalize legacy rows in data/signals.csv
- Fills missing `signal_id` by hashing row contents
- Ensures `generated_at_utc` and `timestamp_utc` are parsed
- Adds placeholder `market_snapshot_id` and `forecast_bundle_id` when missing
- Writes a backup to data/signals.csv.bak and overwrites data/signals.csv
"""
from pathlib import Path
import pandas as pd
import hashlib

BASE = Path(__file__).parent.parent
SIG_PATH = BASE / "data" / "signals.csv"
BAK_PATH = BASE / "data" / "signals.csv.bak"


def _hash_row(row: pd.Series) -> str:
    s = "|".join([str(row.get(c, "")) for c in ["market", "city", "station", "target_date", "generated_at_utc", "model_prob_yes"]])
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def migrate():
    if not SIG_PATH.exists():
        print("No signals.csv found; nothing to do.")
        return
    df = pd.read_csv(SIG_PATH, dtype=str)
    df_backup = df.copy()
    df.to_csv(BAK_PATH, index=False)
    print(f"Backed up signals.csv to {BAK_PATH}")

    # Parse dates
    for col in ["generated_at_utc", "timestamp_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Fill missing signal_id
    if "signal_id" not in df.columns:
        df["signal_id"] = ""
    for idx, row in df.iterrows():
        if not row.get("signal_id"):
            df.at[idx, "signal_id"] = _hash_row(row)

    # Ensure market_snapshot_id and forecast_bundle_id exist
    if "market_snapshot_id" not in df.columns:
        df["market_snapshot_id"] = ""
    if "forecast_bundle_id" not in df.columns:
        df["forecast_bundle_id"] = ""

    df.to_csv(SIG_PATH, index=False)
    print(f"Migrated signals written to {SIG_PATH}")


if __name__ == "__main__":
    migrate()
