#!/usr/bin/env python3
"""Fit calibration parameters from accumulated forecast errors.

Estimates per-station bias/error_std plus the global lead_time_slope and
ensemble_spread_inflation, and prints them. With --apply it writes the
per-station values into data/calibration.csv (which config.yaml already reads),
and saves the suggested global defaults to data/fitted_defaults.yaml for you to
paste into config.yaml's `defaults:` block.

Run after you have collected several days of forecasts + settlements
(use scripts/backfill_settlements.py to fill outcomes).
"""
from pathlib import Path
import argparse

from src.config import load_config
from src.calibration_fit import fit_calibration, write_station_calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="write per-station values to data/calibration.csv and save fitted defaults")
    parser.add_argument("--min-station-samples", type=int, default=8)
    parser.add_argument("--min-per-lead", type=int, default=5)
    parser.add_argument("--min-spread-samples", type=int, default=10)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    data = base / "data"
    cfg = load_config(base / "config.yaml")

    res = fit_calibration(
        data / "forecast_snapshots.csv",
        data / "settlements.csv",
        ref_days=float(cfg.defaults.get("lead_time_ref_days", 1.0)),
        min_station_samples=args.min_station_samples,
        min_per_lead=args.min_per_lead,
        min_spread_samples=args.min_spread_samples,
        current_defaults=cfg.defaults,
    )

    if "error" in res:
        print("Cannot fit:", res["error"])
        return

    print("=" * 60)
    print("PER-STATION CALIBRATION")
    print("=" * 60)
    for r in res["station_calibration"]:
        flag = "" if r["enough_data"] else "  (low data)"
        print(f"  {r['station']:6s} bias={r['station_bias_f']:>7}  error_std={str(r['error_std_f']):>6}  (n={r['n']}){flag}")

    print()
    print("=" * 60)
    print("SUGGESTED GLOBAL DEFAULTS")
    print("=" * 60)
    for k, v in res["suggested_defaults"].items():
        print(f"  {k}: {v}")
    print("  diagnostics:", res["diagnostics"])
    for n in res["notes"]:
        print("  note:", n)

    if args.apply:
        written = write_station_calibration(res["station_calibration"], data / "calibration.csv")
        print()
        print(f"Wrote {written} station row(s) to {data / 'calibration.csv'}")
        defaults_path = data / "fitted_defaults.yaml"
        with defaults_path.open("w") as fh:
            fh.write("# Suggested defaults fitted from historical forecast errors.\n")
            fh.write("# Paste these into config.yaml under `defaults:`.\n")
            for k, v in res["suggested_defaults"].items():
                fh.write(f"{k}: {v}\n")
        print(f"Wrote suggested global defaults to {defaults_path}")
    else:
        print("\n(re-run with --apply to write calibration.csv and fitted_defaults.yaml)")


if __name__ == "__main__":
    main()
