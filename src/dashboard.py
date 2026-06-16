import sys
import os
# Ensure the repo root is importable when launched via `streamlit run src/dashboard.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta

from src.weather_data import fetch_forecasts_for_stations
from src.signals import generate_signals
from src.config import load_config
from src.probability_model import effective_sigma, _norm_cdf
from src.historical_evaluation import evaluate_signals
from src.forecast_accuracy import compute_forecast_accuracy, compute_winrate_timeseries


st.set_page_config(page_title="Weather Signal Engine", layout="wide")
st.title("Weather Signal Engine - ForecastEx / IBKR ForecastTrader MVP")

BASE = Path(__file__).parent.parent
DATA = BASE / "data"
cfg = load_config(BASE / "config.yaml")
stations = list(cfg.stations.keys())


def _sigma_params():
    d = cfg.defaults or {}
    return {
        "lead_time_slope": float(d.get("lead_time_slope", 0.15)),
        "lead_time_ref_days": float(d.get("lead_time_ref_days", 1.0)),
        "lead_time_cap_mult": float(d.get("lead_time_cap_mult", 3.0)),
        "ensemble_spread_inflation": float(d.get("ensemble_spread_inflation", 1.0)),
        "ensemble_blend": str(d.get("ensemble_blend", "max")),
        "min_sigma_f": float(d.get("min_sigma_f", 1.0)),
    }


def _read(path, **kw):
    return pd.read_csv(path, **kw) if Path(path).exists() else pd.DataFrame()


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.header("Controls")
minimum_edge = st.sidebar.number_input("minimum_edge", value=float(cfg.defaults.get("minimum_edge", 0.12)), step=0.01)
fee_buffer = st.sidebar.number_input("fee_buffer", value=float(cfg.defaults.get("fee_buffer", 0.03)), step=0.01)
integer_settlement_mode = st.sidebar.checkbox("integer_settlement_mode", value=bool(cfg.defaults.get("integer_settlement_mode", True)))
start = st.sidebar.date_input("start_date", date.today())
end = st.sidebar.date_input("end_date", date.today() + timedelta(days=2))

st.sidebar.markdown("---")
if st.sidebar.button("Fetch forecasts and compute signals"):
    with st.spinner("Fetching multi-model forecasts..."):
        st.session_state["forecasts"] = fetch_forecasts_for_stations(stations, start, end)
        st.success("Forecasts fetched")

forecasts = st.session_state.get("forecasts", pd.DataFrame())

market_path = DATA / "market_prices_sample.csv"

tab_signals, tab_accuracy = st.tabs(["Signals", "Accuracy over time"])

# --------------------------------------------------------------------------
# Signals tab
# --------------------------------------------------------------------------
with tab_signals:
    st.header("Market Input")
    market_df = _read(market_path, parse_dates=["timestamp_utc", "target_date"])
    st.dataframe(market_df, use_container_width=True)

    st.header("Signals")
    if st.button("Run signal engine"):
        if forecasts.empty:
            st.error("No forecasts available. Fetch forecasts first.")
        else:
            out = generate_signals(market_path, forecasts, fee_buffer=fee_buffer,
                                   minimum_edge=minimum_edge, integer_settlement_mode=integer_settlement_mode)
            st.session_state["signals"] = out
            st.success(f"Signals generated: {len(out)}")

    signals = st.session_state.get("signals", pd.DataFrame())
    if not signals.empty:
        st.subheader("Ranked signals by best edge")
        signals = signals.copy()
        signals["best_edge"] = signals[["yes_edge", "no_edge"]].max(axis=1)
        st.dataframe(signals.sort_values(by="best_edge", ascending=False), use_container_width=True)

    st.header("Probability curve (v2 model)")
    if not market_df.empty:
        selected = st.selectbox("Select market row", options=market_df.index.tolist())
        row = market_df.loc[selected]
        station = row["station"]
        target_date = pd.to_datetime(row["target_date"]).date()
        thresh = float(row["threshold_f"])
        direction = str(row["direction"]).upper() if ("direction" in market_df.columns and pd.notna(row.get("direction"))) else "ABOVE"
        underlying = str(row["underlying"]).upper() if ("underlying" in market_df.columns and pd.notna(row.get("underlying"))) else "HIGH"

        # ensemble mean + spread from fetched forecasts for this station/date/underlying
        ph, ens_std = None, None
        if not forecasts.empty:
            fr = forecasts[(forecasts["station"] == station) &
                           (pd.to_datetime(forecasts["target_date"]).dt.date == target_date)]
            if not fr.empty:
                if underlying == "LOW" and "predicted_low_f" in fr.columns:
                    series = fr["predicted_low_f"]
                elif underlying == "AVG" and "predicted_low_f" in fr.columns:
                    series = (fr["predicted_high_f"] + fr["predicted_low_f"]) / 2.0
                else:
                    series = fr["predicted_high_f"]
                ph = float(series.mean())
                ens_std = float(series.std(ddof=1)) if len(series) >= 2 else None

        if ph is not None:
            station_cfg = cfg.stations[station]
            mu = ph + station_cfg.station_bias_f
            lead_days = max(0, (target_date - date.today()).days)
            sig_info = effective_sigma(base_std_f=station_cfg.error_std_f,
                                       hours_until_target_day=lead_days * 24.0,
                                       ensemble_std_f=ens_std, **_sigma_params())
            sigma = sig_info["sigma"]
            boundary = (0.5 if direction == "ABOVE" else -0.5) if integer_settlement_mode else 0.0
            eff_thr = thresh + boundary

            def _p_yes(x):
                cdf = _norm_cdf((eff_thr - x) / sigma)
                return (1.0 - cdf) if direction == "ABOVE" else cdf

            xs = np.arange(mu - 12, mu + 12, 0.5)
            probs = np.array([_p_yes(x) for x in xs])
            chart_df = pd.DataFrame({"temp": xs, "prob_yes": probs}).set_index("temp")
            st.line_chart(chart_df)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"Predicted {underlying} (mean)", f"{ph:.1f} F")
            c2.metric("Ensemble spread", f"{ens_std:.2f} F" if ens_std is not None else "n/a")
            c3.metric(f"Effective sigma (lead {lead_days}d)", f"{sigma:.2f} F")
            c4.metric(f"P(YES) {direction} {thresh:g}", f"{_p_yes(mu):.1%}")
        else:
            st.info("No forecast found for this row. Fetch forecasts covering the target date.")

# --------------------------------------------------------------------------
# Accuracy tab
# --------------------------------------------------------------------------
with tab_accuracy:
    st.header("How accurate is the engine?")
    st.caption("Forecast accuracy needs only forecasts + settlements. Win-rate also needs market prices/signals. "
               "Use scripts/backfill_settlements.py to auto-fill past outcomes, then refresh below.")

    if st.button("Recompute accuracy report"):
        with st.spinner("Evaluating settled signals and forecasts..."):
            evaluate_signals(DATA / "signals.csv", DATA / "market_snapshots.csv", DATA / "settlements.csv")
            compute_forecast_accuracy(DATA / "forecast_snapshots.csv", DATA / "settlements.csv", data_dir=DATA)
            compute_winrate_timeseries(DATA / "evaluated_trades.csv", data_dir=DATA)
        st.success("Recomputed. See results below.")

    # ---- Forecast accuracy ----
    st.subheader("Forecast accuracy (predicted high vs actual)")
    by_station = _read(DATA / "forecast_accuracy_by_station.csv")
    by_lead = _read(DATA / "forecast_accuracy_by_lead.csv")
    by_source = _read(DATA / "forecast_accuracy_by_source.csv")

    if by_station.empty:
        st.info("No forecast-accuracy data yet. Collect forecasts + settlements, then recompute.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Mean absolute error (F) by station**")
            st.bar_chart(by_station.set_index("station")["mae_f"])
            st.dataframe(by_station, use_container_width=True)
        with c2:
            if not by_lead.empty:
                st.markdown("**Error vs lead time** (should grow with lead days)")
                ld = by_lead.dropna(subset=["lead_days"]).set_index("lead_days")[["mae_f", "rmse_f"]]
                st.line_chart(ld)
        if not by_source.empty:
            st.markdown("**Which forecast model is most accurate?**")
            st.dataframe(by_source.sort_values(["station", "mae_f"]), use_container_width=True)

    # ---- Suggested calibration ----
    sug = _read(DATA / "calibration_suggested.csv")
    if not sug.empty:
        st.subheader("Suggested calibration")
        st.caption("Copy these per-station values into data/calibration.csv (or run scripts/fit_calibration.py --apply).")
        st.dataframe(sug, use_container_width=True)

    # ---- Win-rate over time ----
    st.subheader("Signal win-rate over time")
    ts = _read(DATA / "signal_winrate_timeseries.csv")
    if ts.empty:
        st.info("No settled BUY signals yet. Generate signals, backfill settlements, then recompute.")
    else:
        idx = "trade_index" if "trade_index" in ts.columns else ts.columns[0]
        cols = [c for c in ["cumulative_win_rate", "rolling_win_rate"] if c in ts.columns]
        if cols:
            st.line_chart(ts.set_index(idx)[cols])
        if "cumulative_pnl" in ts.columns:
            st.markdown("**Cumulative PnL (per 1-unit stake)**")
            st.line_chart(ts.set_index(idx)["cumulative_pnl"])
        wins = int((ts["win"] == 1).sum()) if "win" in ts.columns else 0
        n = len(ts)
        m1, m2, m3 = st.columns(3)
        m1.metric("Settled trades", n)
        m2.metric("Overall win rate", f"{(wins / n):.1%}" if n else "n/a")
        if "cumulative_pnl" in ts.columns and n:
            m3.metric("Total PnL (1u)", f"{ts['cumulative_pnl'].iloc[-1]:.3f}")
        st.dataframe(ts, use_container_width=True)

    # ---- Calibration buckets ----
    calib = _read(DATA / "calibration_buckets.csv")
    if not calib.empty:
        st.subheader("Probability calibration buckets")
        st.caption("Model probability vs realized YES rate. Well-calibrated => avg_model_prob ~ actual_yes_rate.")
        st.dataframe(calib, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.write("Signals are research-only. This app does not place trades.")
