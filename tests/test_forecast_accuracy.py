import pandas as pd
import pytest

from src.forecast_accuracy import compute_forecast_accuracy, compute_winrate_timeseries


def _write_forecasts(path):
    # Two models for one station+date; ensemble mean = 75.0
    rows = [
        {"forecast_snapshot_id": "a", "timestamp_utc": "2026-06-10T00:00:00+00:00",
         "station": "KMDW", "target_date": "2026-06-12", "source_name": "gfs_seamless",
         "predicted_high_f": 74.0, "forecast_run_time_utc": "2026-06-10T00:00:00+00:00"},
        {"forecast_snapshot_id": "b", "timestamp_utc": "2026-06-10T00:00:00+00:00",
         "station": "KMDW", "target_date": "2026-06-12", "source_name": "ecmwf_ifs025",
         "predicted_high_f": 76.0, "forecast_run_time_utc": "2026-06-10T00:00:00+00:00"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_settlements(path):
    pd.DataFrame([
        {"station": "KMDW", "target_date": "2026-06-12", "final_high_f": 73.0,
         "source": "TEST", "resolved_at_utc": "2026-06-12T07:00:00+00:00"},
    ]).to_csv(path, index=False)


def test_forecast_accuracy_basic(tmp_path):
    fs = tmp_path / "forecast_snapshots.csv"
    se = tmp_path / "settlements.csv"
    _write_forecasts(fs)
    _write_settlements(se)

    res = compute_forecast_accuracy(fs, se, data_dir=tmp_path)
    assert "error" not in res
    ov = res["overall"]
    # ensemble mean 75 vs actual 73 -> error +2.0
    assert ov["bias_f"] == pytest.approx(2.0, abs=1e-9)
    assert ov["mae_f"] == pytest.approx(2.0, abs=1e-9)

    # suggested calibration should recommend bias = -2.0 to unbias the forecast
    sc = {r["station"]: r for r in res["suggested_calibration"]}
    assert sc["KMDW"]["station_bias_f"] == pytest.approx(-2.0, abs=1e-9)

    # lead time = 2 days (2026-06-10 -> 2026-06-12)
    assert (tmp_path / "forecast_accuracy_by_lead.csv").exists()


def test_winrate_timeseries(tmp_path):
    ev = tmp_path / "evaluated_trades.csv"
    pd.DataFrame([
        {"target_date": "2026-06-11", "station": "KMDW", "signal": "BUY_YES",
         "confidence_label": "STRONG", "pnl": 0.5},
        {"target_date": "2026-06-12", "station": "KMDW", "signal": "BUY_NO",
         "confidence_label": "MEDIUM", "pnl": -0.4},
        {"target_date": "2026-06-13", "station": "KLGA", "signal": "BUY_YES",
         "confidence_label": "STRONG", "pnl": 0.6},
        {"target_date": "2026-06-14", "station": "KLGA", "signal": "NO_TRADE", "pnl": None},
    ]).to_csv(ev, index=False)

    res = compute_winrate_timeseries(ev, data_dir=tmp_path)
    assert "error" not in res
    assert res["n_settled_trades"] == 3
    assert res["overall_win_rate"] == pytest.approx(2.0 / 3.0, abs=1e-9)
    assert (tmp_path / "signal_winrate_timeseries.csv").exists()
