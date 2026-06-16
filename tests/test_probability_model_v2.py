import math
import pytest

from src.probability_model import model_probability_above, effective_sigma


# ---------------------------------------------------------------------------
# Pure effective_sigma logic (no scipy needed)
# ---------------------------------------------------------------------------

def test_sigma_grows_with_lead_time():
    near = effective_sigma(base_std_f=2.5, hours_until_target_day=24, ensemble_std_f=None)
    far = effective_sigma(base_std_f=2.5, hours_until_target_day=24 * 7, ensemble_std_f=None)
    assert far["sigma"] > near["sigma"]


def test_sigma_lead_multiplier_is_capped():
    s = effective_sigma(
        base_std_f=2.0, hours_until_target_day=24 * 100, ensemble_std_f=None,
        lead_time_cap_mult=3.0,
    )
    # capped at base * 3.0
    assert s["sigma"] == pytest.approx(6.0, abs=1e-9)


def test_sigma_max_blend_uses_larger_of_climatology_and_spread():
    s = effective_sigma(
        base_std_f=2.0, hours_until_target_day=24, ensemble_std_f=5.0,
        ensemble_blend="max", lead_time_ref_days=1.0,
    )
    assert s["sigma"] == pytest.approx(5.0, abs=1e-9)


def test_sigma_quad_blend_combines_in_quadrature():
    s = effective_sigma(
        base_std_f=3.0, hours_until_target_day=24, ensemble_std_f=4.0,
        ensemble_blend="quad", lead_time_ref_days=1.0,
    )
    assert s["sigma"] == pytest.approx(5.0, abs=1e-9)  # sqrt(3^2 + 4^2)


def test_sigma_floor_applied():
    s = effective_sigma(base_std_f=0.1, hours_until_target_day=24, ensemble_std_f=None, min_sigma_f=1.0)
    assert s["sigma"] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Full model behavior (needs config + numpy; no scipy required)
# ---------------------------------------------------------------------------

def test_v2_longer_lead_pulls_probability_toward_half():
    # Forecast well above threshold -> high prob_yes; more lead time should
    # widen sigma and pull prob_yes down toward 0.5 (less certainty).
    near = model_probability_above(80.0, 74.0, hours_until_target_day=24, station="KMDW",
                                   integer_settlement_mode=True, method="normal_error_v2")
    far = model_probability_above(80.0, 74.0, hours_until_target_day=24 * 10, station="KMDW",
                                  integer_settlement_mode=True, method="normal_error_v2")
    assert 0.5 < far["model_prob_yes"] < near["model_prob_yes"]


def test_v2_ensemble_disagreement_reduces_confidence():
    tight = model_probability_above(80.0, 74.0, hours_until_target_day=24, station="KMDW",
                                    ensemble_std_f=0.5, method="normal_error_v2")
    wide = model_probability_above(80.0, 74.0, hours_until_target_day=24, station="KMDW",
                                   ensemble_std_f=8.0, method="normal_error_v2")
    assert wide["error_std_f"] > tight["error_std_f"]
    assert wide["model_prob_yes"] < tight["model_prob_yes"]


def test_v1_ignores_lead_time():
    a = model_probability_above(80.0, 74.0, hours_until_target_day=24, station="KMDW",
                                method="normal_error_v1")
    b = model_probability_above(80.0, 74.0, hours_until_target_day=24 * 30, station="KMDW",
                                method="normal_error_v1")
    assert a["model_prob_yes"] == pytest.approx(b["model_prob_yes"], abs=1e-12)


def test_probabilities_are_valid():
    res = model_probability_above(74.0, 74.0, hours_until_target_day=48, station="KMDW")
    assert 0.0 <= res["model_prob_yes"] <= 1.0
    assert res["model_prob_no"] == pytest.approx(1.0 - res["model_prob_yes"], abs=1e-12)


def test_below_direction_flips_probability():
    # "exceed 74" with a hot forecast -> YES likely; "be below 74" -> YES unlikely
    above = model_probability_above(80.0, 74.0, 24, "KMDW", direction="ABOVE")
    below = model_probability_above(80.0, 74.0, 24, "KMDW", direction="BELOW")
    assert above["model_prob_yes"] > 0.5
    assert below["model_prob_yes"] < 0.5
    # cold forecast -> "be below 74" should be likely YES
    below_cold = model_probability_above(60.0, 74.0, 24, "KMDW", direction="BELOW")
    assert below_cold["model_prob_yes"] > 0.5


def test_below_integer_boundary():
    # "be below 74": with integer settlement, YES needs observed <= 73 (true < 73.5)
    res = model_probability_above(73.0, 74.0, 24, "KMDW", direction="BELOW", integer_settlement_mode=True)
    assert res["effective_threshold_f"] == pytest.approx(73.5, abs=1e-9)


def test_invalid_direction_raises():
    with pytest.raises(ValueError):
        model_probability_above(74.0, 74.0, 24, "KMDW", direction="SIDEWAYS")
