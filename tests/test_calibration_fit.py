import pandas as pd
import pytest

from src.calibration_fit import fit_calibration, write_station_calibration


def _make_dataset(tmp_path, n_dates=12, models=("gfs_seamless", "ecmwf_ifs025"), error=2.0):
    """One station, several dates; ensemble mean runs `error` degrees hot."""
    base_date = pd.Timestamp("2026-05-01")
    fs_rows = []
    se_rows = []
    for i in range(n_dates):
        td = (base_date + pd.Timedelta(days=i)).date()
        run = (base_date + pd.Timedelta(days=i - 1))  # lead = 1 day
        actual = 70.0 + (i % 5)
        # models centered so the mean = actual + error
        preds = [actual + error - 0.5, actual + error + 0.5]
        for m, p in zip(models, preds):
            fs_rows.append({
                "forecast_snapshot_id": f"{i}_{m}", "timestamp_utc": run.isoformat(),
                "station": "KMDW", "target_date": str(td), "source_name": m,
                "predicted_high_f": p, "forecast_run_time_utc": run.isoformat(),
            })
        se_rows.append({"station": "KMDW", "target_date": str(td), "final_high_f": actual,
                        "source": "TEST", "resolved_at_utc": run.isoformat()})

    fs = tmp_path / "forecast_snapshots.csv"
    se = tmp_path / "settlements.csv"
    pd.DataFrame(fs_rows).to_csv(fs, index=False)
    pd.DataFrame(se_rows).to_csv(se, index=False)
    return fs, se


def test_fit_station_bias(tmp_path):
    fs, se = _make_dataset(tmp_path, error=2.0)
    res = fit_calibration(fs, se, min_station_samples=4)
    assert "error" not in res
    sc = {r["station"]: r for r in res["station_calibration"]}
    # forecast runs +2 hot -> bias should be about -2 to unbias
    assert sc["KMDW"]["station_bias_f"] == pytest.approx(-2.0, abs=1e-6)
    assert "lead_time_slope" in res["suggested_defaults"]
    assert "ensemble_spread_inflation" in res["suggested_defaults"]


def test_insufficient_data_keeps_defaults(tmp_path):
    fs, se = _make_dataset(tmp_path, n_dates=3)
    res = fit_calibration(fs, se, min_station_samples=8, min_per_lead=5, min_spread_samples=10,
                          current_defaults={"lead_time_slope": 0.15, "ensemble_spread_inflation": 1.0})
    assert res["suggested_defaults"]["lead_time_slope"] == 0.15
    assert res["suggested_defaults"]["ensemble_spread_inflation"] == 1.0
    assert res["notes"]


def test_write_station_calibration(tmp_path):
    calib = tmp_path / "calibration.csv"
    n = write_station_calibration(
        [{"station": "KMDW", "station_bias_f": -2.0, "error_std_f": 2.5}], calib,
    )
    assert n == 1
    df = pd.read_csv(calib)
    assert list(df.columns) == ["station", "station_bias_f", "error_std_f"]
    assert df.iloc[0]["station"] == "KMDW"
