from typing import Tuple
from pathlib import Path
import math
from scipy.stats import norm
from src.config import load_config


def model_probability_above(predicted_high_f: float, threshold_f: float, hours_until_target_day: float, station: str, integer_settlement_mode: bool = True) -> dict:
    cfg = load_config(Path(__file__).parent.parent / "config.yaml")
    station_cfg = cfg.stations.get(station)
    if not station_cfg:
        raise KeyError(f"Station config not found: {station}")
    bias = station_cfg.station_bias_f
    error_std = station_cfg.error_std_f
    # For version 1, ignore hours_until_target_day but keep parameter for future updates
    mu = predicted_high_f + bias
    if integer_settlement_mode:
        # Settlement compares to integer; model rounding by using threshold + 0.5
        effective_threshold = threshold_f + 0.5
    else:
        effective_threshold = threshold_f
    # model_prob_yes = P(final_high > effective_threshold)
    # For continuous normal, P(X > t) = 1 - CDF((t - mu)/sigma)
    z = (effective_threshold - mu) / error_std
    prob_yes = 1.0 - norm.cdf(z)
    prob_no = 1.0 - prob_yes
    return {
        "model_prob_yes": float(prob_yes),
        "model_prob_no": float(prob_no),
        "predicted_high_f": float(predicted_high_f),
        "error_std_f": float(error_std),
        "station_bias_f": float(bias),
        "method": "normal_error_v1",
    }
