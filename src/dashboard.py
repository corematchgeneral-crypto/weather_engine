import streamlit as st
import pandas as pd
from pathlib import Path
from src.weather_data import fetch_forecasts_for_stations
from src.signals import generate_signals
from src.config import load_config
from datetime import date, timedelta


st.set_page_config(page_title="Weather Signal Engine", layout="wide")

st.title("Weather Signal Engine - ForecastEx / IBKR ForecastTrader MVP")

cfg = load_config(Path(__file__).parent.parent / "config.yaml")
stations = list(cfg.stations.keys())

st.sidebar.header("Controls")
minimum_edge = st.sidebar.number_input("minimum_edge", value=float(cfg.defaults.get("minimum_edge", 0.10)), step=0.01)
fee_buffer = st.sidebar.number_input("fee_buffer", value=float(cfg.defaults.get("fee_buffer", 0.02)), step=0.01)
integer_settlement_mode = st.sidebar.checkbox("integer_settlement_mode", value=bool(cfg.defaults.get("integer_settlement_mode", True)))
selected_station = st.sidebar.selectbox("station", stations)
start = st.sidebar.date_input("start_date", date.today())
end = st.sidebar.date_input("end_date", date.today() + timedelta(days=2))

st.sidebar.markdown("---")
if st.sidebar.button("Fetch forecasts and compute signals"):
    with st.spinner("Fetching forecasts..."):
        forecasts = fetch_forecasts_for_stations(stations, start, end)
        st.session_state["forecasts"] = forecasts
        st.success("Forecasts fetched")

forecasts = st.session_state.get("forecasts") if "forecasts" in st.session_state else pd.DataFrame()

st.header("Market Input")
market_path = Path(__file__).parent.parent / "data" / "market_prices_sample.csv"
market_df = pd.read_csv(market_path, parse_dates=["timestamp_utc", "target_date"]) if market_path.exists() else pd.DataFrame()
st.dataframe(market_df)

st.header("Signals")
if st.button("Run signal engine"):
    if forecasts.empty:
        st.error("No forecasts available. Fetch forecasts first.")
    else:
        out = generate_signals(market_path, forecasts, fee_buffer=fee_buffer, minimum_edge=minimum_edge, integer_settlement_mode=integer_settlement_mode)
        st.session_state["signals"] = out
        st.success("Signals generated")

signals = st.session_state.get("signals") if "signals" in st.session_state else pd.DataFrame()
if not signals.empty:
    st.subheader("Ranked signals by best edge")
    signals["best_edge"] = signals[["yes_edge", "no_edge"]].max(axis=1)
    st.dataframe(signals.sort_values(by="best_edge", ascending=False))

st.header("Probability curve")
selected = st.selectbox("Select market row", options=market_df.index.tolist() if not market_df.empty else [])
if not market_df.empty and not forecasts.empty:
    row = market_df.loc[selected]
    station = row["station"]
    target_date = pd.to_datetime(row["target_date"]).date()
    f_row = forecasts[(forecasts["station"] == station) & (forecasts["target_date"] == pd.to_datetime(target_date))]
    if not f_row.empty:
        ph = float(f_row.iloc[0]["predicted_high_f"])
        thresh = float(row["threshold_f"])
        # show probability at thresholds range
        import numpy as np
        from scipy.stats import norm
        station_cfg = cfg.stations[station]
        mu = ph + station_cfg.station_bias_f
        sigma = station_cfg.error_std_f
        xs = np.arange(mu - 10, mu + 10, 0.5)
        probs = 1 - norm.cdf((thresh - xs) / sigma)
        df = pd.DataFrame({"temp": xs, "prob_yes": probs})
        st.line_chart(df.set_index("temp"))
        st.markdown(f"Predicted high: **{ph:.2f} F** — Model mu: **{mu:.2f}**, sigma: **{sigma:.2f}**")
    else:
        st.info("No forecast found for selected market row. Fetch forecasts for full date range.")

st.sidebar.markdown("---")
st.sidebar.write("Signals are research-only. This app does not place trades.")

