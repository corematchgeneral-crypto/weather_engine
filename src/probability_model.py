"""Probabilistic model for daily-high binary contracts.

Two methods are provided:

- ``normal_error_v1`` (legacy): a single fixed per-station error standard
  deviation, ignoring forecast lead time and model disagreement.
- ``normal_error_v2`` (default): the effective standard deviation grows with
  forecast lead time and widens when multiple weather models disagree
  (ensemble spread). This produces more honest probabilities -- it stops the
  model from being over-confident on far-out dates, which is the main way a
  naive model leaks money on these markets.

Both model the eventual settled high temperature as a normal distribution
around the (bias-corrected) forecast and integrate the tail beyond the
contract threshold.
"""

from typing import Optional, TYPE_CHECKING
from pathlib import Path
import math

if TYPE_CHECKING:  # for type hints only; avoids importing pydantic/yaml at module load
    from src.config import Config


# ---------------------------------------------------------------------------
# Effective-sigma model (the core of v2)
# ---------------------------------------------------------------------------

def effective_sigma(
    base_std_f: float,
    hours_until_target_day: Optional[float],
    ensemble_std_f: Optional[float],
    *,
    lead_time_slope: float = 0.15,
    lead_time_ref_days: float = 1.0,
    lead_time_cap_mult: float = 3.0,
    ensemble_spread_inflation: float = 1.0,
    ensemble_blend: str = "max",
    min_sigma_f: float = 1.0,
) -> dict:
    """Compute the effective forecast-error standard deviation.

    Components
    ----------
    1. Climatological error scaled by lead time. ``base_std_f`` is the typical
       error of the (bias-corrected) forecast at the reference lead time
       (about 1 day). Uncertainty grows roughly linearly with additional lead
       days, capped so it does not explode for very distant dates::

           mult = 1 + slope * max(0, lead_days - ref_days)   (capped at cap_mult)
           sigma_clim = base_std_f * mult

    2. Live inter-model disagreement (``ensemble_std_f`` = standard deviation of
       the per-model predicted highs), optionally inflated because ensemble
       spread tends to under-disperse relative to true error.

    Blending
    --------
    - ``"max"`` (default, conservative): use climatology as a floor and let
      live disagreement widen sigma when models strongly disagree. Avoids
      double-counting and never shrinks below the climatological estimate.
    - ``"quad"``: combine the two components in quadrature (treats them as
      independent sources of uncertainty).

    A hard floor ``min_sigma_f`` is always applied to avoid overconfident
    probabilities near 0 or 1.
    """
    if hours_until_target_day is None or hours_until_target_day < 0:
        lead_days = lead_time_ref_days
    else:
        lead_days = hours_until_target_day / 24.0

    lead_mult = 1.0 + lead_time_slope * max(0.0, lead_days - lead_time_ref_days)
    lead_mult = min(lead_mult, lead_time_cap_mult)
    sigma_clim = base_std_f * lead_mult

    spread_component = None
    if ensemble_std_f is not None and ensemble_std_f > 0:
        spread_component = ensemble_spread_inflation * ensemble_std_f

    if spread_component is None:
        sigma = sigma_clim
    elif ensemble_blend == "quad":
        sigma = math.sqrt(sigma_clim ** 2 + spread_component ** 2)
    else:  # "max"
        sigma = max(sigma_clim, spread_component)

    sigma = max(sigma, min_sigma_f)

    return {
        "sigma": float(sigma),
        "sigma_climatological": float(sigma_clim),
        "lead_days": float(lead_days),
        "lead_multiplier": float(lead_mult),
        "ensemble_std_f": float(ensemble_std_f) if ensemble_std_f is not None else None,
        "ensemble_blend": ensemble_blend,
    }


def _sigma_params_from_config(cfg: "Config") -> dict:
    d = cfg.defaults or {}
    return {
        "lead_time_slope": float(d.get("lead_time_slope", 0.15)),
        "lead_time_ref_days": float(d.get("lead_time_ref_days", 1.0)),
        "lead_time_cap_mult": float(d.get("lead_time_cap_mult", 3.0)),
        "ensemble_spread_inflation": float(d.get("ensemble_spread_inflation", 1.0)),
        "ensemble_blend": str(d.get("ensemble_blend", "max")),
        "min_sigma_f": float(d.get("min_sigma_f", 1.0)),
    }


# ---------------------------------------------------------------------------
# Public model entry point
# ---------------------------------------------------------------------------

def model_probability_above(
    predicted_high_f: float,
    threshold_f: float,
    hours_until_target_day: Optional[float],
    station: str,
    integer_settlement_mode: bool = True,
    ensemble_std_f: Optional[float] = None,
    method: str = "normal_error_v2",
    cfg: Optional["Config"] = None,
) -> dict:
    """Probability that the settled daily high exceeds ``threshold_f``.

    Parameters
    ----------
    predicted_high_f:
        Point forecast (typically the ensemble mean across models) in degrees F.
    threshold_f:
        Contract threshold ("Above X").
    hours_until_target_day:
        Hours from the forecast/snapshot time to the target day. Used by v2 to
        scale uncertainty with lead time. Ignored by v1.
    station:
        Station id used to look up the bias and base error std.
    integer_settlement_mode:
        If True, model integer settlement rounding by comparing to
        ``threshold_f + 0.5``.
    ensemble_std_f:
        Standard deviation of the per-model predicted highs (live disagreement).
        Used by v2 only. ``None`` falls back to climatology alone.
    method:
        ``"normal_error_v2"`` (default) or ``"normal_error_v1"`` (legacy).
    cfg:
        Optional preloaded config (avoids re-reading YAML on every call).
    """
    if cfg is None:
        from src.config import load_config  # lazy: keeps pure sigma logic import-light
        cfg = load_config(Path(__file__).parent.parent / "config.yaml")
    station_cfg = cfg.stations.get(station)
    if not station_cfg:
        raise KeyError(f"Station config not found: {station}")

    bias = station_cfg.station_bias_f
    base_std = station_cfg.error_std_f

    mu = predicted_high_f + bias

    if integer_settlement_mode:
        # Settlement compares to an integer observation; model rounding by
        # shifting the threshold by 0.5 (a contract "Above 74" loses at 74).
        effective_threshold = threshold_f + 0.5
    else:
        effective_threshold = threshold_f

    if method == "normal_error_v1":
        sigma = base_std
        sigma_info = {
            "sigma": float(sigma),
            "sigma_climatological": float(sigma),
            "lead_days": None,
            "lead_multiplier": 1.0,
            "ensemble_std_f": None,
            "ensemble_blend": None,
        }
    else:
        method = "normal_error_v2"
        sigma_info = effective_sigma(
            base_std_f=base_std,
            hours_until_target_day=hours_until_target_day,
            ensemble_std_f=ensemble_std_f,
            **_sigma_params_from_config(cfg),
        )
        sigma = sigma_info["sigma"]

    z = (effective_threshold - mu) / sigma
    from scipy.stats import norm  # local import: keeps pure sigma logic importable without scipy
    prob_yes = float(1.0 - norm.cdf(z))
    prob_no = float(1.0 - prob_yes)

    return {
        "model_prob_yes": prob_yes,
        "model_prob_no": prob_no,
        "predicted_high_f": float(predicted_high_f),
        "effective_threshold_f": float(effective_threshold),
        "error_std_f": float(sigma),
        "base_error_std_f": float(base_std),
        "station_bias_f": float(bias),
        "lead_days": sigma_info["lead_days"],
        "lead_multiplier": sigma_info["lead_multiplier"],
        "ensemble_std_f": sigma_info["ensemble_std_f"],
        "method": method,
    }
