from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime


def _read_csv_optional(path: Path, parse_dates=None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    # Read without parse_dates first to avoid pandas error when requested columns are missing
    df = pd.read_csv(path, keep_default_na=True)
    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    # fallback: leave as-is if parse fails
                    pass
    return df


def evaluate_signals(signals_path: Path, market_snapshots_path: Path, settlements_path: Path, fee_buffer: float = 0.03, data_dir: Path = None) -> Dict[str, Any]:
    signals = _read_csv_optional(signals_path, parse_dates=["generated_at_utc", "timestamp_utc"]) 
    market = _read_csv_optional(market_snapshots_path, parse_dates=["timestamp_utc"]) 
    settlements = _read_csv_optional(settlements_path, parse_dates=["resolved_at_utc"]) 

    if signals.empty:
        return {"error": "No signals to evaluate"}

    # Normalize target_date to date; coerce malformed values to NaT then drop
    signals["target_date"] = pd.to_datetime(signals["target_date"], errors="coerce").dt.date
    # drop rows where target_date could not be parsed
    signals = signals[signals["target_date"].notna()].copy()
    if signals.empty:
        return {"error": "No valid signals with parseable target_date"}
    if not settlements.empty and "target_date" in settlements.columns:
        settlements["target_date"] = pd.to_datetime(settlements["target_date"], errors="coerce").dt.date

    # Require settlements with station+target_date and at least one observed value
    has_value_col = {"final_high_f", "final_low_f"} & set(settlements.columns)
    if settlements.empty or not {"station", "target_date"}.issubset(settlements.columns) or not has_value_col:
        return {"error": "No settlements to evaluate"}

    # Merge signals with settlements on station+target_date
    merged = signals.merge(settlements, how="left", on=["station", "target_date"], suffixes=("", "_settlement"))

    # Ensure expected columns exist with defaults (back-compat with older signal files)
    if "direction" not in merged.columns:
        merged["direction"] = "ABOVE"
    else:
        merged["direction"] = merged["direction"].fillna("ABOVE")
    if "comparison" not in merged.columns:
        merged["comparison"] = merged["direction"]
    else:
        merged["comparison"] = merged["comparison"].fillna(merged["direction"])
    if "unit" not in merged.columns:
        merged["unit"] = "F"
    else:
        merged["unit"] = merged["unit"].fillna("F")
    if "threshold_high" not in merged.columns:
        merged["threshold_high"] = np.nan
    if "underlying" not in merged.columns:
        merged["underlying"] = "HIGH"
    else:
        merged["underlying"] = merged["underlying"].fillna("HIGH")
    if "final_high_f" not in merged.columns:
        merged["final_high_f"] = np.nan
    if "final_low_f" not in merged.columns:
        merged["final_low_f"] = np.nan

    # Coerce numeric-like columns before computing outcomes
    num_cols = ["model_prob_yes", "model_prob_no", "yes_ask", "no_ask", "threshold_f", "threshold_high", "final_high_f", "final_low_f"]
    for c in num_cols:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    # Observed settled value depends on the contract's underlying (HIGH/LOW/AVG).
    # Stored final values are in deg F (the unit-conversion lives in the bounds).
    def _observed_value(r):
        u = str(r.get("underlying", "HIGH")).upper()
        hi, lo = r.get("final_high_f"), r.get("final_low_f")
        if u == "LOW":
            return lo
        if u == "AVG":
            return (hi + lo) / 2.0 if (pd.notna(hi) and pd.notna(lo)) else np.nan
        return hi

    merged["observed_value_f"] = merged.apply(_observed_value, axis=1)
    # Only evaluate signals where the needed observed value exists
    merged = merged[merged["observed_value_f"].notna()]
    if merged.empty:
        return {"error": "No settled signals to evaluate"}

    # YES outcome via the same bounds the model used: YES iff a_f <= observed_F < b_f
    from src.probability_model import contract_bounds_f

    def _outcome(r):
        obs, thr = r.get("observed_value_f"), r.get("threshold_f")
        if pd.isna(obs) or pd.isna(thr):
            return np.nan
        thr_hi = r.get("threshold_high")
        thr_hi = float(thr_hi) if pd.notna(thr_hi) else None
        try:
            a_f, b_f = contract_bounds_f(
                str(r.get("comparison", "ABOVE")).upper(), float(thr), thr_hi,
                str(r.get("unit", "F")).upper(), True,
            )
        except Exception:
            return np.nan
        return 1 if (a_f <= float(obs) < b_f) else 0

    merged["actual_yes_outcome"] = merged.apply(_outcome, axis=1)

    # Compute PnL per spec
    def compute_pnl(row):
        sig = row.get("signal")
        entry_yes = row.get("yes_ask")
        entry_no = row.get("no_ask")
        outc = row.get("actual_yes_outcome")
        if pd.isna(outc):
            return np.nan
        if sig == "BUY_YES":
            entry = entry_yes
            if pd.isna(entry):
                return np.nan
            return (1 - entry) if outc == 1 else (-entry)
        elif sig == "BUY_NO":
            entry = entry_no
            if pd.isna(entry):
                return np.nan
            # NO wins when final_high_f <= threshold
            return (1 - entry) if outc == 0 else (-entry)
        else:
            return np.nan

    merged["pnl"] = merged.apply(compute_pnl, axis=1)

    # Compute ROI per trade (pnl / stake) for buy trades where stake > 0
    def compute_roi(row):
        sig = row.get("signal")
        entry = row.get("yes_ask") if sig == "BUY_YES" else (row.get("no_ask") if sig == "BUY_NO" else None)
        if pd.isna(entry) or entry == 0 or pd.isna(row.get("pnl")):
            return np.nan
        return row.get("pnl") / entry

    merged["roi"] = merged.apply(compute_roi, axis=1)

    # For each settled signal, compute CLV (closing snapshot after signal and before settlement)
    if not market.empty:
        market["target_date"] = pd.to_datetime(market["target_date"]).dt.date

    clv_values = []
    closing_yes_list = []
    closing_no_list = []
    closing_ts_list = []
    clv_available_list = []

    for _, s in merged.iterrows():
        station = s.get("station")
        td = s.get("target_date")
        gen_ts = pd.to_datetime(s.get("generated_at_utc")) if not pd.isna(s.get("generated_at_utc")) else pd.to_datetime(s.get("timestamp_utc"))
        resolved_ts = pd.to_datetime(s.get("resolved_at_utc")) if not pd.isna(s.get("resolved_at_utc")) else None

        # Filter market snapshots for same station+target_date and within (gen_ts, resolved_ts]
        if market.empty:
            clv_available_list.append(False)
            clv_values.append(np.nan)
            closing_yes_list.append(np.nan)
            closing_no_list.append(np.nan)
            closing_ts_list.append(pd.NaT)
            continue

        cand = market[(market["station"] == station) & (market["target_date"] == td)].copy()
        if cand.empty:
            clv_available_list.append(False)
            clv_values.append(np.nan)
            closing_yes_list.append(np.nan)
            closing_no_list.append(np.nan)
            closing_ts_list.append(pd.NaT)
            continue

        cand["timestamp_utc"] = pd.to_datetime(cand["timestamp_utc"]) 
        cand = cand[cand["timestamp_utc"] > gen_ts]
        if resolved_ts is not None:
            cand = cand[cand["timestamp_utc"] <= resolved_ts]

        if cand.empty:
            clv_available_list.append(False)
            clv_values.append(np.nan)
            closing_yes_list.append(np.nan)
            closing_no_list.append(np.nan)
            closing_ts_list.append(pd.NaT)
            continue

        last = cand.sort_values("timestamp_utc").iloc[-1]
        closing_yes = last.get("yes_ask")
        closing_no = last.get("no_ask")
        closing_ts = last.get("timestamp_utc")

        mp_yes = s.get("model_prob_yes")
        mp_no = s.get("model_prob_no")
        clv_edge = None
        if s.get("signal") == "BUY_YES":
            if pd.notna(mp_yes) and pd.notna(closing_yes):
                clv_edge = float(mp_yes) - float(closing_yes) - fee_buffer
        elif s.get("signal") == "BUY_NO":
            if pd.notna(mp_no) and pd.notna(closing_no):
                clv_edge = float(mp_no) - float(closing_no) - fee_buffer

        clv_available_list.append(True)
        clv_values.append(clv_edge)
        closing_yes_list.append(closing_yes)
        closing_no_list.append(closing_no)
        closing_ts_list.append(closing_ts)

    merged["closing_yes"] = closing_yes_list
    merged["closing_no"] = closing_no_list
    merged["closing_timestamp"] = closing_ts_list
    merged["clv_available"] = clv_available_list
    merged["clv_value"] = clv_values

    # Focus on BUY signals for trading metrics
    buys = merged[merged["signal"].isin(["BUY_YES", "BUY_NO"])].copy()
    n_settled = len(buys)
    pnl_series = buys["pnl"].dropna()
    total_pnl = float(pnl_series.sum()) if not pnl_series.empty else 0.0
    avg_pnl = float(pnl_series.mean()) if not pnl_series.empty else 0.0
    win_rate = float((pnl_series > 0).mean()) if not pnl_series.empty else 0.0
    total_stake = buys.apply(lambda r: r.get("yes_ask") if r.get("signal") == "BUY_YES" else r.get("no_ask"), axis=1).dropna().sum()
    roi_on_stake = float(total_pnl / total_stake) if total_stake and total_stake > 0 else None

    clv_vals = buys[buys["clv_available"] == True]["clv_value"].dropna()
    avg_clv = float(clv_vals.mean()) if not clv_vals.empty else None
    clv_positive_rate = float((clv_vals > 0).mean()) if not clv_vals.empty else None

    # Brier score across settled signals (robust to missing/invalid values)
    if "model_prob_yes" in merged.columns and "actual_yes_outcome" in merged.columns:
        valid_mask = merged["model_prob_yes"].notna() & merged["actual_yes_outcome"].notna()
        if valid_mask.any():
            brier = float(((merged.loc[valid_mask, "model_prob_yes"] - merged.loc[valid_mask, "actual_yes_outcome"]) ** 2).mean())
        else:
            brier = float(np.nan)
    else:
        brier = float(np.nan)

    buys_sorted = buys.sort_values(by="generated_at_utc")
    pnl_seq = buys_sorted["pnl"].fillna(0).tolist()
    cum = np.cumsum(pnl_seq)
    if len(cum) > 0:
        running_max = np.maximum.accumulate(cum)
        drawdown = float((running_max - cum).max())
    else:
        drawdown = 0.0

    streak = 0
    max_streak = 0
    for v in pnl_seq:
        if v <= 0:
            streak += 1
            if streak > max_streak:
                max_streak = streak
        else:
            streak = 0

    metrics = {
        "total_settled_buy_signals": int(n_settled),
        "win_rate": win_rate,
        "average_pnl_per_contract": avg_pnl,
        "total_pnl": total_pnl,
        "roi_on_stake": roi_on_stake,
        "average_clv": avg_clv,
        "clv_positive_rate": clv_positive_rate,
        "brier_score": brier,
        "max_drawdown": drawdown,
        "longest_loss_streak": int(max_streak),
    }

    group = buys.groupby(["station", "signal", "confidence_label"]).agg(
        n_trades=("signal", "count"),
        win_rate=("pnl", lambda x: float((x > 0).mean()) if len(x) > 0 else np.nan),
        avg_pnl=("pnl", lambda x: float(x.mean()) if len(x) > 0 else np.nan),
        total_pnl=("pnl", lambda x: float(x.sum()) if len(x) > 0 else 0.0),
        avg_clv=("clv_value", lambda x: float(x.dropna().mean()) if len(x.dropna()) > 0 else np.nan),
        clv_available_pct=("clv_value", lambda x: float(x.notna().mean()) if len(x) > 0 else np.nan),
    ).reset_index()

    merged["prob_bucket"] = pd.cut(merged["model_prob_yes"], bins=np.linspace(0.0, 1.0, 11), include_lowest=True)
    calib = merged.groupby("prob_bucket").agg(
        n=("model_prob_yes", "count"),
        avg_model_prob=("model_prob_yes", lambda x: float(x.mean()) if len(x) > 0 else np.nan),
        actual_yes_rate=("actual_yes_outcome", lambda x: float(x.mean()) if len(x) > 0 else np.nan),
    ).reset_index()
    calib["calibration_error"] = calib["actual_yes_rate"] - calib["avg_model_prob"]

    base = Path(__file__).parent.parent
    out_dir = Path(data_dir) if data_dir is not None else base / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluated_path = out_dir / "evaluated_trades.csv"
    metrics_by_group_path = out_dir / "historical_metrics_by_group.csv"
    calib_path = out_dir / "calibration_buckets.csv"

    merged.to_csv(evaluated_path, index=False)
    group.to_csv(metrics_by_group_path, index=False)
    calib.to_csv(calib_path, index=False)

    return {"metrics": metrics, "evaluated_trades_path": str(evaluated_path), "metrics_by_group_path": str(metrics_by_group_path), "calibration_buckets_path": str(calib_path)}


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    res = evaluate_signals(base / "data" / "signals.csv", base / "data" / "market_snapshots.csv", base / "data" / "settlements.csv")
    print(res)
