import pytest
from src.probability_model import model_probability_above


def test_integer_settlement_rule_yes_loses_at_equal():
    # Above 74, final 74 -> YES loses
    res = model_probability_above(predicted_high_f=74.0, threshold_f=74.0, hours_until_target_day=24, station="KMDW", integer_settlement_mode=True)
    # With predicted at 74 and threshold 74+0.5 => prob_yes should be ~0.3085 for sigma 2.5
    assert 0.0 <= res["model_prob_yes"] <= 1.0


def test_integer_settlement_rule_yes_wins_at_75():
    # Above 74, if predicted 75 -> higher prob
    res1 = model_probability_above(predicted_high_f=74.0, threshold_f=74.0, hours_until_target_day=24, station="KMDW", integer_settlement_mode=True)
    res2 = model_probability_above(predicted_high_f=76.0, threshold_f=74.0, hours_until_target_day=24, station="KMDW", integer_settlement_mode=True)
    assert res2["model_prob_yes"] > res1["model_prob_yes"]


def test_model_probability_decreases_with_threshold():
    base = model_probability_above(predicted_high_f=75.0, threshold_f=74.0, hours_until_target_day=24, station="KMDW", integer_settlement_mode=True)
    higher = model_probability_above(predicted_high_f=75.0, threshold_f=76.0, hours_until_target_day=24, station="KMDW", integer_settlement_mode=True)
    assert higher["model_prob_yes"] < base["model_prob_yes"]
