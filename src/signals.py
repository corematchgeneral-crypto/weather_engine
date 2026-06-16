from pathlib import Path
import pandas as pd
import math
import hashlib
import json
from src.market_data import load_market_csv
from src.probability_model import model_probability_above
from src.config import load_config
from src.market_validation import validate_market_row
from datetime import datetime, timezone


FEE_BUFFER_DEFAULT = 0.03
MIN_EDGE_DEFAULT = 0.12
MODEL_VERSION = "1.0"


def _hash_str(*parts) -> str:
    s = "|".join(["" if p is None else str(p) for p in parts])
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def confidence_label(edge: float) -> str:
    if edge is None:
        return "NONE"
    if edge >= 0.15:
        return "STRONG"
    if edge >= 0.12:
        return "MEDIUM"
    if edge >= 0.05:
        return "WEAK"
    return "NONE"


def _ensure_csv(path: Path, columns: list[str]):
    if not path.exists():
        df = pd.DataFrame(columns=columns)
        df.to_csv(path, index=False)


def generate_signals(market_csv_path: str | Path, forecasts_df: pd.DataFrame, fee_buffer: float = None, minimum_edge: float = None, integer_settlement_mode: bool = True, min_volume: float | None = None, data_dir: str | Path | None = None) -> pd.DataFrame:
    cfg = load_config()
    if fee_buffer is None:
        fee_buffer = cfg.defaults.get("fee_buffer", FEE_BUFFER_DEFAULT)
    if minimum_edge is None:
        minimum_edge = cfg.defaults.get("minimum_edge", MIN_EDGE_DEFAULT)
    if min_volume is None:
        min_volume = cfg.defaults.get("min_market_volume", 0)

    market_df = load_market_csv(market_csv_path)
    # Ensure dates and timestamps
    market_df["target_date"] = pd.to_datetime(market_df["target_date"]).dt.date
    market_df["timestamp_utc"] = pd.to_datetime(market_df["timestamp_utc"], utc=True)

    # Prepare archive paths (data_dir overridable for tests / alternate datasets)
    base = Path(__file__).parent.parent
    data_root = Path(data_dir) if data_dir is not None else base / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    market_archive = data_root / "market_snapshots.csv"
    forecast_archive = data_root / "forecast_snapshots.csv"
    signals_path = data_root / "signals.csv"

    # Ensure archive files exist (with headers)
    _ensure_csv(market_archive, ["market_snapshot_id", "timestamp_utc", "market", "city", "station", "target_date", "threshold_f", "yes_ask", "no_ask", "yes_bid", "no_bid", "volume", "source_file", "created_at_utc"])
    _ensure_csv(forecast_archive, ["forecast_snapshot_id", "timestamp_utc", "station", "target_date", "source_name", "predicted_high_f", "predicted_low_f", "precip_probability", "cloud_cover", "wind_speed", "forecast_run_time_utc", "raw_metadata", "created_at_utc"])
    _ensure_csv(signals_path, ["signal_id", "market_snapshot_id", "forecast_bundle_id", "model_version", "generated_at_utc"])

    # Archive market snapshots (append unique by deterministic id)
    market_to_archive = []
    for _, r in market_df.iterrows():
        ts = r.get("timestamp_utc")
        ts_iso = pd.to_datetime(ts).isoformat() if not pd.isna(ts) else datetime.now(timezone.utc).isoformat()
        msid = _hash_str(ts_iso, r.get("market"), r.get("station"), r.get("target_date"), r.get("threshold_f"), r.get("yes_ask"), r.get("no_ask"), r.get("yes_bid"), r.get("no_bid"), r.get("volume"))
        market_to_archive.append({
            "market_snapshot_id": msid,
            "timestamp_utc": ts_iso,
            "market": r.get("market"),
            "city": r.get("city"),
            "station": r.get("station"),
            "target_date": r.get("target_date"),
            "threshold_f": r.get("threshold_f"),
            "yes_ask": r.get("yes_ask"),
            "no_ask": r.get("no_ask"),
            "yes_bid": r.get("yes_bid"),
            "no_bid": r.get("no_bid"),
            "volume": r.get("volume"),
            "source_file": Path(market_csv_path).name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    if market_to_archive:
        existing_market = pd.read_csv(market_archive)
        existing_ids = set(existing_market.get("market_snapshot_id", []))
        new_market = [m for m in market_to_archive if m["market_snapshot_id"] not in existing_ids]
        if new_market:
            pd.concat([existing_market, pd.DataFrame(new_market)], ignore_index=True).to_csv(market_archive, index=False)

    # Archive forecast snapshots (from provided forecasts_df)
    forecast_to_archive = []
    fdf = forecasts_df.copy()
    if not fdf.empty and "target_date" in fdf.columns:
        fdf["target_date"] = pd.to_datetime(fdf["target_date"]).dt.date
        for _, fr in fdf.iterrows():
            fr_ts = fr.get("run_time_utc")
            fr_ts_iso = pd.to_datetime(fr_ts).isoformat() if not pd.isna(fr_ts) else datetime.now(timezone.utc).isoformat()
            fsid = _hash_str(fr_ts_iso, fr.get("station"), fr.get("target_date"), fr.get("forecast_source"), fr.get("predicted_high_f"), fr.get("predicted_low_f"))
            forecast_to_archive.append({
                "forecast_snapshot_id": fsid,
                "timestamp_utc": fr_ts_iso,
                "station": fr.get("station"),
                "target_date": fr.get("target_date"),
                "source_name": fr.get("forecast_source"),
                "predicted_high_f": fr.get("predicted_high_f"),
                "predicted_low_f": fr.get("predicted_low_f"),
                "precip_probability": fr.get("precip_probability"),
                "cloud_cover": fr.get("cloud_cover"),
                "wind_speed": fr.get("wind_speed"),
                "forecast_run_time_utc": fr_ts_iso,
                "raw_metadata": fr.get("raw_json_path_or_summary") if "raw_json_path_or_summary" in fr.index else None,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            })

        if forecast_to_archive:
            existing_forecast = pd.read_csv(forecast_archive)
            existing_ids = set(existing_forecast.get("forecast_snapshot_id", []))
            new_forecasts = [f for f in forecast_to_archive if f["forecast_snapshot_id"] not in existing_ids]
            if new_forecasts:
                pd.concat([existing_forecast, pd.DataFrame(new_forecasts)], ignore_index=True).to_csv(forecast_archive, index=False)

    # Aggregate forecasts per station+target_date into a bundle.
    # Mean across models = point forecast; std across models = ensemble spread
    # (a live measure of forecast uncertainty fed into the probability model).
    if not fdf.empty and "target_date" in fdf.columns:
        agg_specs = {}
        if "predicted_high_f" in fdf.columns:
            agg_specs["predicted_high_f"] = ("predicted_high_f", "mean")
            agg_specs["predicted_high_std_f"] = ("predicted_high_f", "std")
            agg_specs["n_models"] = ("predicted_high_f", "count")
        if "predicted_low_f" in fdf.columns:
            agg_specs["predicted_low_f"] = ("predicted_low_f", "mean")
            agg_specs["predicted_low_std_f"] = ("predicted_low_f", "std")
        if "precip_probability" in fdf.columns:
            agg_specs["precip_probability"] = ("precip_probability", "mean")
        if "cloud_cover" in fdf.columns:
            agg_specs["cloud_cover"] = ("cloud_cover", "mean")
        if "wind_speed" in fdf.columns:
            agg_specs["wind_speed"] = ("wind_speed", "mean")
        if "forecast_source" in fdf.columns:
            agg_specs["forecast_sources"] = (
                "forecast_source",
                lambda x: ",".join(sorted(set([str(v) for v in x if pd.notna(v)]))),
            )
        if "run_time_utc" in fdf.columns:
            agg_specs["forecast_run_time_utc"] = ("run_time_utc", "max")
        agg = fdf.groupby(["station", "target_date"]).agg(**agg_specs).reset_index()
    else:
        agg = pd.DataFrame()

    # Build a map of forecast snapshot ids by station+target_date to create a bundle id
    forecast_map = {}
    if not fdf.empty:
        # Build index of forecast_snapshot_id per row
        temp_forecasts = pd.read_csv(forecast_archive)
        for _, fr in temp_forecasts.iterrows():
            key = (fr.get("station"), fr.get("target_date"))
            forecast_map.setdefault(key, []).append(fr.get("forecast_snapshot_id"))

    # Merge aggregated forecasts onto market rows for signal calculation
    if not agg.empty:
        merged = market_df.merge(agg, how="left", on=["station", "target_date"])
    else:
        merged = market_df.copy()

    rows = []
    generated_at = datetime.now(timezone.utc).isoformat()
    for _, r in merged.iterrows():
        station = r.get("station")
        timestamp = r.get("timestamp_utc")
        if pd.isna(timestamp):
            timestamp = datetime.now(timezone.utc)
        threshold = r.get("threshold_f")
        yes_ask = float(r["yes_ask"]) if not pd.isna(r.get("yes_ask")) else None
        no_ask = float(r["no_ask"]) if not pd.isna(r.get("no_ask")) else None
        yes_bid = float(r["yes_bid"]) if not pd.isna(r.get("yes_bid")) else None
        no_bid = float(r["no_bid"]) if not pd.isna(r.get("no_bid")) else None
        volume = float(r["volume"]) if (not pd.isna(r.get("volume"))) else 0.0

        # Market validation using centralized rules
        validation = validate_market_row(r)
        issues = validation.get("issues", [])
        warnings = validation.get("warnings", [])

        # Contract semantics: underlying (HIGH/LOW/AVG), comparison, unit, range
        underlying = str(r.get("underlying")).upper() if ("underlying" in r.index and pd.notna(r.get("underlying"))) else "HIGH"
        direction = str(r.get("direction")).upper() if ("direction" in r.index and pd.notna(r.get("direction"))) else "ABOVE"
        comparison = str(r.get("comparison")).upper() if ("comparison" in r.index and pd.notna(r.get("comparison"))) else direction
        unit = str(r.get("unit")).upper() if ("unit" in r.index and pd.notna(r.get("unit"))) else "F"
        threshold_high = r.get("threshold_high") if ("threshold_high" in r.index and pd.notna(r.get("threshold_high"))) else None
        if underlying not in ("HIGH", "LOW", "AVG"):
            underlying = "HIGH"
        if comparison not in ("ABOVE", "BELOW", "ATLEAST", "ATMOST", "EQUALS", "RANGE"):
            comparison = "ABOVE"
        if unit not in ("C", "F"):
            unit = "F"

        # Select the predicted value + ensemble spread for the contract's underlying.
        pred_high = r.get("predicted_high_f")
        pred_low = r.get("predicted_low_f") if "predicted_low_f" in r.index else None
        std_high = r.get("predicted_high_std_f") if "predicted_high_std_f" in r.index else None
        std_low = r.get("predicted_low_std_f") if "predicted_low_std_f" in r.index else None
        if underlying == "LOW":
            predicted_value = pred_low
            ens_std_raw = std_low
        elif underlying == "AVG":
            predicted_value = ((pred_high + pred_low) / 2.0) if (pd.notna(pred_high) and pd.notna(pred_low)) else None
            ens_std_raw = None  # per-model avg spread not tracked; fall back to climatology
        else:  # HIGH
            predicted_value = pred_high
            ens_std_raw = std_high
        predicted_high = predicted_value  # name kept for downstream/output compatibility

        # Lead time: hours from the market snapshot to the target day.
        target_d = r.get("target_date")
        hours_until_target = 24.0  # sensible ~1-day fallback
        try:
            if pd.notna(timestamp) and target_d is not None:
                snap_date = pd.to_datetime(timestamp).date()
                lead_days = (target_d - snap_date).days
                hours_until_target = max(0.0, float(lead_days) * 24.0)
        except Exception:
            hours_until_target = 24.0

        # Ensemble spread (std of per-model predicted values for this underlying).
        ens_std = float(ens_std_raw) if (ens_std_raw is not None and not pd.isna(ens_std_raw)) else None
        n_models_raw = r.get("n_models") if "n_models" in r.index else None
        n_models = int(n_models_raw) if (n_models_raw is not None and not pd.isna(n_models_raw)) else None

        model = None
        if predicted_high is not None and not pd.isna(predicted_high) and threshold is not None and not pd.isna(threshold):
            model = model_probability_above(
                predicted_high,
                threshold,
                hours_until_target,
                station,
                integer_settlement_mode=integer_settlement_mode,
                ensemble_std_f=ens_std,
                comparison=comparison,
                unit=unit,
                threshold_high_f=(float(threshold_high) if threshold_high is not None else None),
                cfg=cfg,
            )
        model_prob_yes = model["model_prob_yes"] if model else None
        model_prob_no = model["model_prob_no"] if model else None
        effective_sigma_f = model["error_std_f"] if model else None
        model_method = model["method"] if model else None
        yes_edge = model_prob_yes - yes_ask - fee_buffer if (model_prob_yes is not None and yes_ask is not None) else None
        no_edge = model_prob_no - no_ask - fee_buffer if (model_prob_no is not None and no_ask is not None) else None

        # Liquidity filter
        low_liquidity = (volume < min_volume)

        best_side = None
        signal = "NO_TRADE"
        reasons = []
        if issues:
            reasons.append("SANITY:" + ",".join(issues))
            signal = "NO_TRADE"
        elif low_liquidity:
            reasons.append(f"LOW_LIQUIDITY(volume={volume} < min={min_volume})")
            signal = "NO_TRADE"
        else:
            if yes_edge is not None and yes_edge >= minimum_edge and yes_ask is not None:
                signal = "BUY_YES"
                best_side = "YES"
                reasons.append(f"YES_EDGE={yes_edge:.4f}>=min({minimum_edge})")
            elif no_edge is not None and no_edge >= minimum_edge and no_ask is not None:
                signal = "BUY_NO"
                best_side = "NO"
                reasons.append(f"NO_EDGE={no_edge:.4f}>=min({minimum_edge})")
            else:
                reasons.append("NO_EDGE")

        max_loss = yes_ask if signal == "BUY_YES" else (no_ask if signal == "BUY_NO" else None)
        max_profit = (1 - yes_ask) if signal == "BUY_YES" else ((1 - no_ask) if signal == "BUY_NO" else None)
        reward_risk = (max_profit / max_loss) if (max_profit is not None and max_loss and max_loss > 0) else None
        ev_yes = (model_prob_yes * 1 - yes_ask) if (model_prob_yes is not None and yes_ask is not None) else None
        ev_no = (model_prob_no * 1 - no_ask) if (model_prob_no is not None and no_ask is not None) else None
        breakeven_probability = (yes_ask + fee_buffer) if yes_ask is not None else None
        conf_edge = None
        if yes_edge is not None and no_edge is not None:
            conf_edge = max(yes_edge, no_edge)
        else:
            conf_edge = yes_edge if yes_edge is not None else (no_edge if no_edge is not None else None)
        conf = confidence_label(conf_edge)

        # Forecast bundle id for aggregated forecasts
        bundle_key = (station, r.get("target_date"))
        bundle_ids = forecast_map.get(bundle_key, [])
        forecast_bundle_id = None
        if bundle_ids:
            forecast_bundle_id = _hash_str("|".join(sorted(bundle_ids)))

        # Build market snapshot id for this row (same logic as archived)
        ts_iso = pd.to_datetime(r.get("timestamp_utc")).isoformat() if not pd.isna(r.get("timestamp_utc")) else datetime.now(timezone.utc).isoformat()
        market_snapshot_id = _hash_str(ts_iso, r.get("market"), station, r.get("target_date"), r.get("threshold_f"), r.get("yes_ask"), r.get("no_ask"), r.get("yes_bid"), r.get("no_bid"), r.get("volume"))

        # Build deterministic signal id
        signal_id = _hash_str(market_snapshot_id, forecast_bundle_id, MODEL_VERSION, minimum_edge, fee_buffer, integer_settlement_mode, station, r.get("target_date"), r.get("threshold_f"), signal, yes_ask, no_ask)

        # Append validation warnings
        if warnings:
            reasons.append("WARN:" + ",".join(warnings))

        reason_text = "; ".join(reasons)

        rows.append({
            "signal_id": signal_id,
            "market_snapshot_id": market_snapshot_id,
            "forecast_bundle_id": forecast_bundle_id,
            "model_version": MODEL_VERSION,
            "generated_at_utc": generated_at,
            "timestamp_utc": ts_iso,
            "market": r.get("market"),
            "city": r.get("city"),
            "station": station,
            "target_date": r.get("target_date"),
            "threshold_f": r.get("threshold_f"),
            "threshold_high": threshold_high,
            "unit": unit,
            "comparison": comparison,
            "underlying": underlying,
            "direction": direction,
            "yes_ask": yes_ask,
            "no_ask": no_ask,
            "predicted_high_f": r.get("predicted_high_f"),
            "predicted_value_f": predicted_high,
            "predicted_high_std_f": ens_std,
            "n_models": n_models,
            "hours_until_target": hours_until_target,
            "effective_sigma_f": effective_sigma_f,
            "model_method": model_method,
            "model_prob_yes": model_prob_yes,
            "model_prob_no": model_prob_no,
            "yes_edge": yes_edge,
            "no_edge": no_edge,
            "best_side": best_side,
            "signal": signal,
            "confidence_label": conf,
            "max_loss": max_loss,
            "max_profit": max_profit,
            "reward_risk": reward_risk,
            "expected_value_yes": ev_yes,
            "expected_value_no": ev_no,
            "breakeven_probability": breakeven_probability,
            # During live generation, CLV is not calculated
            "closing_yes": None,
            "closing_no": None,
            "closing_timestamp": None,
            "closing_line_value": None,
            "clv_available": False,
            "clv_note": "CLV is calculated only during historical evaluation, not during live signal generation.",
            "forecast_sources": r.get("forecast_sources") if "forecast_sources" in r.index else None,
            "signal_reason": reason_text,
            "notes": r.get("notes"),
        })

    out = pd.DataFrame(rows)

    # Remove exact duplicate rows within this run by signal_id
    if not out.empty:
        out = out.drop_duplicates(subset=["signal_id"])

    # Append non-duplicate signals to data/signals.csv
    if signals_path.exists():
        existing = pd.read_csv(signals_path, comment="#")
    else:
        existing = pd.DataFrame()

    existing_ids = set(existing.get("signal_id", [])) if not existing.empty else set()
    new_rows = [r for _, r in out.iterrows() if r.get("signal_id") not in existing_ids]
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        if existing.empty:
            df_new.to_csv(signals_path, index=False)
        else:
            pd.concat([existing, df_new], ignore_index=True).to_csv(signals_path, index=False)

    return out
