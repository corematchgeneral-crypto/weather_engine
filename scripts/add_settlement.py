"""Record an official settlement outcome for a station + date.

For traded ForecastEx contracts, enter the OFFICIAL whole-degree value shown in
the Weather Underground daily summary table ("Actual" column) for the contract's
station and date. Pass --final-high-f and/or --final-low-f depending on the
contract underlying (HIGH/LOW/AVG; AVG needs both).

Example:
  python -m scripts.add_settlement --station KMDW --target-date 2026-06-17 \\
      --final-high-f 84 --source "Weather Underground (KMDW)"
"""
from pathlib import Path
import argparse
import pandas as pd

COLUMNS = ["station", "target_date", "final_high_f", "final_low_f", "source", "resolved_at_utc", "notes"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", required=True)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--final-high-f", type=float, default=None, help="official daily HIGH (deg F)")
    parser.add_argument("--final-low-f", type=float, default=None, help="official daily LOW (deg F)")
    parser.add_argument("--source", required=True, help='e.g. "Weather Underground (KMDW)"')
    parser.add_argument("--notes", required=False, default="")
    args = parser.parse_args()

    if args.final_high_f is None and args.final_low_f is None:
        parser.error("provide at least one of --final-high-f / --final-low-f")

    base = Path(__file__).parent.parent
    path = base / "data" / "settlements.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    td = pd.to_datetime(args.target_date).date()
    existing = df[(df["station"] == args.station) & (pd.to_datetime(df["target_date"]).dt.date == td)]
    if not existing.empty:
        print("Settlement for station+target_date already exists; overwriting")
        df = df[~((df["station"] == args.station) & (pd.to_datetime(df["target_date"]).dt.date == td))]

    new = {
        "station": args.station,
        "target_date": td,
        "final_high_f": args.final_high_f if args.final_high_f is not None else "",
        "final_low_f": args.final_low_f if args.final_low_f is not None else "",
        "source": args.source,
        "resolved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "notes": args.notes,
    }
    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"Settlement added for {args.station} {td}: high={args.final_high_f} low={args.final_low_f}")


if __name__ == "__main__":
    main()
