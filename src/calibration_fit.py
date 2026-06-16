"""Fit model calibration parameters from historical forecast errors.

Given accumulated forecast snapshots and settled outcomes, this estimates:

- per-station bias and base error std (``error_std_f`` at the reference lead),
- the global ``lead_time_slope`` (how fast error grows with lead time),
- the global ``ensemble_spread_inflation`` (how model disagreement maps to
  realized error).

Everything degrades gracefully: when there is too little data for a parameter,
the current configured default is kept and a note is returned. The point is
that the fit improves automatically as more days accumulate.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import math
import statistics
import numpy as np
import pandas as pd


def summarize_forecast_errors(errors: List[Optional[float]]) -> Optional[Dict[str, Any]]:
    """Summarize a list of (forecast - actual) errors in deg F.

    Returns mean error, the bias correction to apply (``station_bias_f`` =
    -mean_error, so ``forecast + bias`` is unbiased), and the error std (used as
    ``error_std_f``). Returns None if there are no usable errors.
    """
    errs = [float(e) for e in errors if e is not None and not (isinstance(e, float) and math.isnan(e))]
    if not errs:
        return None
    mean_err = statistics.fmean(errs)
    std = statistics.stdev(errs) if len(errs) >= 2 else float("nan")
    return {
        "n": len(errs),
        "mean_error_f": round(mean_err, 3),
        "mae_f": round(statistics.fmean([abs(e) for e in errs]), 3),
        "station_bias_f": round(-mean_err, 3),
        "error_std_f": round(std, 3) if not math.isnan(std) else "",
    }


def _read_csv_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#") if Path(path).exists() else pd.DataFrame()


def _prepare(forecast_snapshots_path: Path, settlements_path: Path):
    fs = _read_csv_optional(forecast_snapshots_path)
    settle = _read_csv_optional(settlements_path)
    if fs.empty:
        return None, "No forecast snapshots found"
    if settle.empty or not {"station", "target_date", "final_high_f"}.issubset(settle.columns):
        return None, "No settlements found"

    fs = fs.copy()
    fs["target_date"] = pd.to_datetime(fs["target_date"], errors="coerce").dt.date
    fs["predicted_high_f"] = pd.to_numeric(fs.get("predicted_high_f"), errors="coerce")
    fs = fs[fs["target_date"].notna() & fs["predicted_high_f"].notna()]

    run_col = "forecast_run_time_utc" if "forecast_run_time_utc" in fs.columns else "timestamp_utc"
    run_dates = pd.to_datetime(fs.get(run_col), errors="coerce", utc=True).dt.date
    fs["lead_days"] = [
        (td - rd).days if (pd.notna(rd) and td is not None) else np.nan
        for td, rd in zip(fs["target_date"], run_dates)
    ]

    settle = settle.copy()
    settle["target_date"] = pd.to_datetime(settle["target_date"], errors="coerce").dt.date
    settle["final_high_f"] = pd.to_numeric(settle["final_high_f"], errors="coerce")
    settle = settle[["station", "target_date", "final_high_f"]].dropna(subset=["final_high_f"])
    return (fs, settle), None


def fit_calibration(
    forecast_snapshots_path: Path,
    settlements_path: Path,
    ref_days: float = 1.0,
    min_station_samples: int = 8,
    min_per_lead: int = 5,
    min_spread_samples: int = 10,
    current_defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    current_defaults = current_defaults or {}
    prep, err = _prepare(forecast_snapshots_path, settlements_path)
    if err:
        return {"error": err}
    fs, settle = prep
    notes: List[str] = []

    # ---- Ensemble-mean errors per (station, target_date) ----
    ens = (
        fs.groupby(["station", "target_date"])
        .agg(predicted_high_f=("predicted_high_f", "mean"),
             spread_f=("predicted_high_f", lambda x: x.std(ddof=1) if x.count() >= 2 else np.nan),
             n_models=("predicted_high_f", "count"),
             lead_days=("lead_days", "min"))
        .reset_index()
        .merge(settle, how="inner", on=["station", "target_date"])
    )
    if ens.empty:
        return {"error": "No settled forecasts (no station+date overlap)"}
    ens["error_f"] = ens["predicted_high_f"] - ens["final_high_f"]

    # ---- Per-station bias + base std (at near-reference lead) ----
    station_rows = []
    for station, g in ens.groupby("station"):
        err_all = g["error_f"].dropna()
        if err_all.empty:
            continue
        bias = -float(err_all.mean())
        near = g[g["lead_days"] <= ref_days]["error_f"].dropna()
        base_src = near if len(near) >= max(2, min_station_samples // 2) else err_all
        base_std = float(base_src.std(ddof=1)) if len(base_src) >= 2 else np.nan
        station_rows.append({
            "station": station,
            "n": int(len(err_all)),
            "station_bias_f": round(bias, 3),
            "error_std_f": round(base_std, 3) if not np.isnan(base_std) else "",
            "enough_data": len(err_all) >= min_station_samples,
        })
    station_calib = pd.DataFrame(station_rows)

    # ---- Global lead-time slope ----
    # De-bias per station, then look at error std as a function of lead days.
    ens = ens.merge(
        ens.groupby("station")["error_f"].mean().rename("station_mean_err"),
        on="station", how="left",
    )
    ens["debiased_err"] = ens["error_f"] - ens["station_mean_err"]

    lead_stats = (
        ens.dropna(subset=["lead_days"])
        .groupby("lead_days")["debiased_err"]
        .agg(["count", "std"])
        .reset_index()
    )
    lead_stats = lead_stats[lead_stats["count"] >= min_per_lead]

    fitted_slope = current_defaults.get("lead_time_slope", 0.15)
    slope_diag = {}
    if len(lead_stats) >= 2 and (lead_stats["lead_days"] <= ref_days).any():
        # base std = std at/just below the reference lead
        base_row = lead_stats[lead_stats["lead_days"] <= ref_days].sort_values("lead_days").iloc[-1]
        base_std_global = float(base_row["std"]) if base_row["std"] and base_row["std"] > 0 else np.nan
        if not np.isnan(base_std_global) and base_std_global > 0:
            x = (lead_stats["lead_days"] - ref_days).to_numpy(dtype=float)
            y = (lead_stats["std"].to_numpy(dtype=float) / base_std_global) - 1.0
            mask = x > 0
            if mask.sum() >= 1 and np.sum(x[mask] ** 2) > 0:
                slope = float(np.sum(x[mask] * y[mask]) / np.sum(x[mask] ** 2))  # regression through origin
                fitted_slope = max(0.0, round(slope, 3))
                slope_diag = {"base_std_global": round(base_std_global, 3),
                              "lead_points": int(len(lead_stats))}
            else:
                notes.append("Not enough multi-lead data to fit lead_time_slope; kept default.")
        else:
            notes.append("Could not establish a base std at reference lead; kept default slope.")
    else:
        notes.append("Need >=2 lead buckets (incl. one at/below ref) to fit slope; kept default.")

    # ---- Global ensemble spread inflation ----
    fitted_inflation = current_defaults.get("ensemble_spread_inflation", 1.0)
    infl_diag = {}
    spread_df = ens.dropna(subset=["spread_f"])
    spread_df = spread_df[spread_df["spread_f"] > 0]
    if len(spread_df) >= min_spread_samples:
        realized_std = float(spread_df["debiased_err"].std(ddof=1))
        mean_spread = float(spread_df["spread_f"].mean())
        if mean_spread > 0:
            fitted_inflation = round(max(0.1, realized_std / mean_spread), 3)
            infl_diag = {"realized_err_std": round(realized_std, 3),
                         "mean_spread": round(mean_spread, 3),
                         "n": int(len(spread_df))}
    else:
        notes.append(f"Need >={min_spread_samples} multi-model settled samples to fit "
                     "ensemble_spread_inflation; kept default.")

    suggested_defaults = {
        "lead_time_slope": fitted_slope,
        "ensemble_spread_inflation": fitted_inflation,
    }

    return {
        "station_calibration": station_calib.to_dict(orient="records"),
        "suggested_defaults": suggested_defaults,
        "diagnostics": {"lead_slope": slope_diag, "ensemble_inflation": infl_diag,
                        "n_settled_forecasts": int(len(ens))},
        "notes": notes,
    }


def write_station_calibration(station_calib: List[dict], calibration_csv: Path) -> int:
    """Write fitted per-station bias/std to calibration.csv (the file config reads)."""
    rows = [
        {"station": r["station"], "station_bias_f": r["station_bias_f"], "error_std_f": r["error_std_f"]}
        for r in station_calib if r.get("error_std_f") not in ("", None)
    ]
    if not rows:
        return 0
    pd.DataFrame(rows).to_csv(calibration_csv, index=False)
    return len(rows)
