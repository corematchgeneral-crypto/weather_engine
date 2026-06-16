#!/usr/bin/env python3
"""Auto-fill settlements for past target dates from the Open-Meteo archive.

Looks at the (station, target_date) pairs that appear in the signals and
forecast-snapshot archives, finds those whose target date is in the past and
that do not yet have a settlement, fetches the observed daily high from the
Open-Meteo historical archive, and appends them to ``data/settlements.csv``.

This closes the loop automatically so win-rate / accuracy can be tracked over
time without hand-entering every outcome.

NOTE: archive observations are a good proxy but are not guaranteed to match the
official ForecastEx settlement source. Rows written here use
source=OPEN_METEO_ARCHIVE so you can distinguish them.

Requires internet access (run on your own machine, not the sandbox).
"""
from pathlib import Path
from datetime import date, datetime, timezone
import argparse
import pandas as pd

from src.weather_data import fetch_actual_high_f


def _collect_pairs(base: Path) -> pd.DataFrame:
    pairs = []
    for name in ("signals.csv", "forecast_snapshots.csv"):
        p = base / "data" / name
        if not p.exists():
            continue
        df = pd.read_csv(p, comment="#")
        if {"station", "target_date"}.issubset(df.columns):
            sub = df[["station", "target_date"]].dropna()
            pairs.append(sub)
    if not pairs:
        return pd.DataFrame(columns=["station", "target_date"])
    allp = pd.concat(pairs, ignore_index=True)
    allp["target_date"] = pd.to_datetime(allp["target_date"], errors="coerce").dt.date
    allp = allp.dropna(subset=["target_date"]).drop_duplicates()
    return allp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-date", help="only settle target dates strictly before this yyyy-mm-dd (default: today UTC)")
    parser.add_argument("--dry-run", action="store_true", help="show what would be fetched without writing")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    cutoff = pd.to_datetime(args.before_date).date() if args.before_date else datetime.now(timezone.utc).date()

    pairs = _collect_pairs(base)
    if pairs.empty:
        print("No (station, target_date) pairs found in signals/forecast archives.")
        return
    pairs = pairs[pairs["target_date"] < cutoff]

    settle_path = base / "data" / "settlements.csv"
    if settle_path.exists():
        settle = pd.read_csv(settle_path)
        settle["target_date"] = pd.to_datetime(settle["target_date"], errors="coerce").dt.date
        existing = set(zip(settle["station"], settle["target_date"]))
    else:
        settle = pd.DataFrame(columns=["station", "target_date", "final_high_f", "source", "resolved_at_utc", "notes"])
        existing = set()

    todo = [(r.station, r.target_date) for r in pairs.itertuples() if (r.station, r.target_date) not in existing]
    if not todo:
        print(f"Nothing to backfill before {cutoff}. All past target dates already settled.")
        return

    print(f"Backfilling {len(todo)} settlement(s) before {cutoff}...")
    new_rows = []
    for station, td in sorted(todo):
        try:
            high_f = fetch_actual_high_f(station, td)
        except Exception as e:
            print(f"  {station} {td}: fetch failed: {e}")
            continue
        if high_f is None:
            print(f"  {station} {td}: no archive observation available")
            continue
        print(f"  {station} {td}: actual high = {high_f:.1f}F")
        new_rows.append({
            "station": station,
            "target_date": td,
            "final_high_f": round(high_f, 1),
            "source": "OPEN_METEO_ARCHIVE",
            "resolved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "notes": "auto-backfilled from open-meteo archive (proxy, verify vs official source)",
        })

    if not new_rows:
        print("No settlements written.")
        return
    if args.dry_run:
        print(f"[dry-run] would write {len(new_rows)} settlement row(s).")
        return

    out = pd.concat([settle, pd.DataFrame(new_rows)], ignore_index=True)
    out.to_csv(settle_path, index=False)
    print(f"Wrote {len(new_rows)} settlement(s) to {settle_path}")


if __name__ == "__main__":
    main()
