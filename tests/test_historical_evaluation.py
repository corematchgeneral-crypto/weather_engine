import pandas as pd
import pytest

from src.historical_evaluation import evaluate_signals


def _signals():
    common = {
        "station": "KMDW", "target_date": "2026-06-12",
        "generated_at_utc": "2026-06-11T00:00:00+00:00",
        "timestamp_utc": "2026-06-11T00:00:00+00:00",
        "confidence_label": "STRONG", "model_prob_no": 0.1,
    }
    return pd.DataFrame([
        # ABOVE 72, observed high 75 -> YES wins; BUY_YES @0.4 -> pnl +0.6
        {**common, "threshold_f": 72, "direction": "ABOVE", "underlying": "HIGH",
         "signal": "BUY_YES", "yes_ask": 0.4, "no_ask": 0.6, "model_prob_yes": 0.9},
        # BELOW 80, observed high 75 -> YES (below) wins; BUY_YES @0.3 -> pnl +0.7
        {**common, "threshold_f": 80, "direction": "BELOW", "underlying": "HIGH",
         "signal": "BUY_YES", "yes_ask": 0.3, "no_ask": 0.7, "model_prob_yes": 0.8},
        # ABOVE 80, observed high 75 -> NO; BUY_YES @0.5 loses -> pnl -0.5
        {**common, "threshold_f": 80, "direction": "ABOVE", "underlying": "HIGH",
         "signal": "BUY_YES", "yes_ask": 0.5, "no_ask": 0.5, "model_prob_yes": 0.6},
    ])


def test_direction_aware_outcomes(tmp_path):
    sig = tmp_path / "signals.csv"
    mkt = tmp_path / "market_snapshots.csv"
    settle = tmp_path / "settlements.csv"
    _signals().to_csv(sig, index=False)
    pd.DataFrame(columns=["market_snapshot_id", "timestamp_utc", "station", "target_date", "yes_ask", "no_ask"]).to_csv(mkt, index=False)
    pd.DataFrame([{"station": "KMDW", "target_date": "2026-06-12", "final_high_f": 75,
                   "source": "TEST", "resolved_at_utc": "2026-06-12T07:00:00+00:00"}]).to_csv(settle, index=False)

    res = evaluate_signals(sig, mkt, settle, data_dir=tmp_path)
    assert "error" not in res, res
    m = res["metrics"]
    assert m["total_settled_buy_signals"] == 3
    # two winners (0.6, 0.7), one loser (-0.5) -> win rate 2/3, total pnl 0.8
    assert m["win_rate"] == pytest.approx(2 / 3, abs=1e-9)
    assert m["total_pnl"] == pytest.approx(0.8, abs=1e-9)


def test_low_underlying_uses_final_low(tmp_path):
    sig = tmp_path / "signals.csv"
    mkt = tmp_path / "market_snapshots.csv"
    settle = tmp_path / "settlements.csv"
    pd.DataFrame([{
        "station": "KMDW", "target_date": "2026-06-12", "threshold_f": 60,
        "direction": "ABOVE", "underlying": "LOW", "signal": "BUY_YES",
        "yes_ask": 0.5, "no_ask": 0.5, "model_prob_yes": 0.7, "model_prob_no": 0.3,
        "confidence_label": "STRONG", "generated_at_utc": "2026-06-11T00:00:00+00:00",
        "timestamp_utc": "2026-06-11T00:00:00+00:00",
    }]).to_csv(sig, index=False)
    pd.DataFrame(columns=["station", "target_date", "timestamp_utc", "yes_ask", "no_ask"]).to_csv(mkt, index=False)
    # low of 62 > threshold 60 -> ABOVE LOW resolves YES; BUY_YES @0.5 -> +0.5
    pd.DataFrame([{"station": "KMDW", "target_date": "2026-06-12", "final_high_f": 80,
                   "final_low_f": 62, "source": "TEST", "resolved_at_utc": "2026-06-12T07:00:00+00:00"}]).to_csv(settle, index=False)

    res = evaluate_signals(sig, mkt, settle, data_dir=tmp_path)
    assert "error" not in res, res
    assert res["metrics"]["total_pnl"] == pytest.approx(0.5, abs=1e-9)
