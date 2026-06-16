"""Forecast-accuracy and signal-win-rate tracking over time.

This module answers the question "how accurate is the engine?" using two
independent lenses:

1. Forecast accuracy (does not need any market prices): compares the predicted
   daily high against the settled actual high, broken down by station, by
   individual weather model, and by lead time. It also derives a *suggested
   calibration* (per-station bias + error std) that can be fed back into
   ``data/calibration.csv`` to self-tune the model.

2. Signal win-rate over time: reads the evaluated trades produced by
   ``historical_evaluation`` and builds cumulative and rolling win-rate series.

Everything is written to CSVs under ``data/`` so it accumulates as you collect
more days of forecasts and settlements.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import math
import numpy as np
import pandas as pd


def _read_csv_optional(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path, comment="#")


def _accuracy_stats(group: pd.DataFrame) -> pd.Series:
    err = group["error_f"]
    return pd.Series({
        "n": int(err.notna().sum()),
        "mae_f": float(err.abs().mean()) if err.notna().any() else np.nan,
        "bias_f": float(err.mean()) if err.notna().any() else np.nan,
        "rmse_f": float(math.sqrt((err ** 2).mean())) if err.notna().any() else np.nan,
    })


def compute_forecast_accuracy(
    forecast_snapshots_path: Path,
    settlements_path: Path,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compare predicted highs to settled actual highs.

    ``error_f`` is defined as ``predicted_high_f - final_high_f`` (positive =
    forecast ran hot). The suggested per-station calibration uses
    ``station_bias_f = -mean(error)`` and ``error_std_f = std(error)`` computed
    on the ensemble-mean forecast.
    """
    fs = _read_csv_optional(forecast_snapshots_path)
    settle = _read_csv_optional(settlements_path)

    if fs.empty:
        return {"error": "No forecast snapshots to evaluate"}
    if settle.empty or not {"station", "target_date", "final_high_f"}.issubset(settle.columns):
        return {"error": "No settlements to evaluate"}

    fs = fs.copy()
    fs["target_date"] = pd.to_datetime(fs["target_date"], errors="coerce").dt.date
    fs["predicted_high_f"] = pd.to_numeric(fs.get("predicted_high_f"), errors="coerce")
    fs = fs[fs["target_date"].notna() & fs["predicted_high_f"].notna()]

    settle = settle.copy()
    settle["target_date"] = pd.to_datetime(settle["target_date"], errors="coerce").dt.date
    settle["final_high_f"] = pd.to_numeric(settle["final_high_f"], errors="coerce")
    settle = settle[["station", "target_date", "final_high_f"]].dropna(subset=["final_high_f"])

    # lead time per snapshot row (days between forecast run and target date)
    run_col = "forecast_run_time_utc" if "forecast_run_time_utc" in fs.columns else "timestamp_utc"
    if run_col in fs.columns:
        run_dates = pd.to_datetime(fs[run_col], errors="coerce", utc=True).dt.date
        fs["lead_days"] = [
            (td - rd).days if (pd.notna(rd) and td is not None) else np.nan
            for td, rd in zip(fs["target_date"], run_dates)
        ]
    else:
        fs["lead_days"] = np.nan

    # ---- Per-model (per-source) accuracy ----
    merged_src = fs.merge(settle, how="inner", on=["station", "target_date"])
    if merged_src.empty:
        return {"error": "No settled forecasts to evaluate (no station+date overlap)"}
    merged_src["error_f"] = merged_src["predicted_high_f"] - merged_src["final_high_f"]

    by_source = (
        merged_src.groupby(["station", "source_name"]).apply(_accuracy_stats).reset_index()
        if "source_name" in merged_src.columns
        else pd.DataFrame()
    )

    by_lead = merged_src.groupby("lead_days").apply(_accuracy_stats).reset_index()

    # ---- Ensemble-mean accuracy (what the signal actually uses) ----
    ens = (
        fs.groupby(["station", "target_date"])
        .agg(predicted_high_f=("predicted_high_f", "mean"),
             n_models=("predicted_high_f", "count"),
             lead_days=("lead_days", "min"))
        .reset_index()
        .merge(settle, how="inner", on=["station", "target_date"])
    )
    ens["error_f"] = ens["predicted_high_f"] - ens["final_high_f"]

    by_station = ens.groupby("station").apply(_accuracy_stats).reset_index()

    overall = _accuracy_stats(ens).to_dict()

    # ---- Suggested calibration (feed back into data/calibration.csv) ----
    calib_rows = []
    for station, g in ens.groupby("station"):
        err = g["error_f"].dropna()
        if err.empty:
            continue
        suggested_bias = -float(err.mean())  # mu = predicted + bias => unbias forecast
        suggested_std = float(err.std(ddof=1)) if len(err) >= 2 else np.nan
        calib_rows.append({
            "station": station,
            "n": int(len(err)),
            "station_bias_f": round(suggested_bias, 3),
            "error_std_f": round(suggested_std, 3) if not math.isnan(suggested_std) else "",
        })
    suggested_calib = pd.DataFrame(calib_rows)

    # ---- write outputs ----
    base = data_dir or Path(forecast_snapshots_path).parent
    base = Path(base)
    by_station_path = base / "forecast_accuracy_by_station.csv"
    by_source_path = base / "forecast_accuracy_by_source.csv"
    by_lead_path = base / "forecast_accuracy_by_lead.csv"
    suggested_calib_path = base / "calibration_suggested.csv"

    by_station.to_csv(by_station_path, index=False)
    if not by_source.empty:
        by_source.to_csv(by_source_path, index=False)
    by_lead.to_csv(by_lead_path, index=False)
    if not suggested_calib.empty:
        suggested_calib.to_csv(suggested_calib_path, index=False)

    return {
        "overall": overall,
        "by_station": by_station.to_dict(orient="records"),
        "by_source": by_source.to_dict(orient="records") if not by_source.empty else [],
        "by_lead": by_lead.to_dict(orient="records"),
        "suggested_calibration": suggested_calib.to_dict(orient="records"),
        "paths": {
            "by_station": str(by_station_path),
            "by_source": str(by_source_path),
            "by_lead": str(by_lead_path),
            "calibration_suggested": str(suggested_calib_path),
        },
    }


def compute_winrate_timeseries(
    evaluated_trades_path: Path,
    rolling_window: int = 20,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build cumulative and rolling signal win-rate over time.

    Consumes ``evaluated_trades.csv`` (produced by ``historical_evaluation``),
    which is the single source of truth for per-trade PnL.
    """
    df = _read_csv_optional(evaluated_trades_path)
    if df.empty:
        return {"error": "No evaluated trades found. Run historical evaluation first."}

    df = df.copy()
    if "signal" not in df.columns or "pnl" not in df.columns:
        return {"error": "evaluated_trades.csv missing required columns (signal, pnl)"}

    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    buys = df[df["signal"].isin(["BUY_YES", "BUY_NO"]) & df["pnl"].notna()].copy()
    if buys.empty:
        return {"error": "No settled BUY signals to build a win-rate series"}

    sort_col = "target_date" if "target_date" in buys.columns else "generated_at_utc"
    buys["_sort"] = pd.to_datetime(buys[sort_col], errors="coerce")
    buys = buys.sort_values("_sort").reset_index(drop=True)

    buys["win"] = (buys["pnl"] > 0).astype(int)
    buys["trade_index"] = np.arange(1, len(buys) + 1)
    buys["cumulative_win_rate"] = buys["win"].expanding().mean()
    buys["cumulative_pnl"] = buys["pnl"].cumsum()
    buys["rolling_win_rate"] = buys["win"].rolling(window=rolling_window, min_periods=1).mean()

    out_cols = [
        "trade_index", sort_col, "station", "signal", "confidence_label",
        "pnl", "win", "cumulative_win_rate", "rolling_win_rate", "cumulative_pnl",
    ]
    out_cols = [c for c in out_cols if c in buys.columns]
    ts = buys[out_cols]

    base = data_dir or Path(evaluated_trades_path).parent
    ts_path = Path(base) / "signal_winrate_timeseries.csv"
    ts.to_csv(ts_path, index=False)

    summary = {
        "n_settled_trades": int(len(buys)),
        "overall_win_rate": float(buys["win"].mean()),
        "rolling_window": rolling_window,
        "latest_rolling_win_rate": float(buys["rolling_win_rate"].iloc[-1]),
        "total_pnl": float(buys["pnl"].sum()),
        "timeseries_path": str(ts_path),
    }
    return summary


if __name__ == "__main__":
    base = Path(__file__).parent.parent / "data"
    acc = compute_forecast_accuracy(base / "forecast_snapshots.csv", base / "settlements.csv", data_dir=base)
    print("FORECAST ACCURACY:", acc)
    wr = compute_winrate_timeseries(base / "evaluated_trades.csv", data_dir=base)
    print("WIN-RATE TIMESERIES:", wr)
