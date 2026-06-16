import pandas as pd
from src.signals import generate_signals
from pathlib import Path


def test_signal_triggers_only_with_edge(tmp_path):
    # create minimal market csv
    csv = tmp_path / "market.csv"
    csv.write_text("timestamp_utc,market,city,station,target_date,threshold_f,yes_ask,no_ask,yes_bid,no_bid,volume,notes\n2026-06-16T10:30:00Z,Test,TestCity,KMDW,2026-06-17,70,0.10,0.9,0.0,0.0,10,\n")
    # create forecast df
    import pandas as pd
    forecasts = pd.DataFrame([{"station":"KMDW","target_date":pd.to_datetime("2026-06-17").date(),"predicted_high_f":85.0}])
    out = generate_signals(csv, forecasts, fee_buffer=0.0, minimum_edge=0.10, integer_settlement_mode=True, min_volume=0, data_dir=tmp_path)
    assert not out.empty
    # Should signal BUY_YES because model_prob near 1 and yes_ask 0.1 -> edge ~0.9
    assert out.iloc[0]["signal"] == "BUY_YES"


def test_missing_asks_do_not_crash(tmp_path):
    csv = tmp_path / "market.csv"
    csv.write_text("timestamp_utc,market,city,station,target_date,threshold_f,yes_ask,no_ask,yes_bid,no_bid,volume,notes\n2026-06-16T10:30:00Z,Test,TestCity,KMDW,2026-06-17,70,,0.9,,,,\n")
    forecasts = pd.DataFrame([{"station":"KMDW","target_date":pd.to_datetime("2026-06-17").date(),"predicted_high_f":65.0}])
    out = generate_signals(csv, forecasts, fee_buffer=0.0, minimum_edge=0.10, integer_settlement_mode=True, min_volume=0, data_dir=tmp_path)
    assert not out.empty
    # no yes_ask, so cannot BUY_YES; expected signal should be NO_TRADE or BUY_NO depending on edge
    assert out.iloc[0]["signal"] in ("NO_TRADE", "BUY_NO")
