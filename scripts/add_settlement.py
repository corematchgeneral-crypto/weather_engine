from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", required=True)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--final-high-f", required=True, type=float)
    parser.add_argument("--source", required=True)
    parser.add_argument("--notes", required=False, default="")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    path = base / "data" / "settlements.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["station", "target_date", "final_high_f", "source", "resolved_at_utc", "notes"])
    td = pd.to_datetime(args.target_date).date()
    # Check duplicates
    existing = df[(df["station"] == args.station) & (pd.to_datetime(df["target_date"]).dt.date == td)]
    if not existing.empty:
        print("Settlement for station+target_date already exists; merging/overwriting")
        df = df[~((df["station"] == args.station) & (pd.to_datetime(df["target_date"]).dt.date == td))]

    new = {"station": args.station, "target_date": td, "final_high_f": args.final_high_f, "source": args.source, "resolved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "notes": args.notes}
    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    df.to_csv(path, index=False)
    print("Settlement added")


if __name__ == "__main__":
    main()
