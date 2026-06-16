from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class MarketRow(BaseModel):
    timestamp_utc: datetime
    market: str
    city: str
    station: str
    target_date: datetime
    threshold_f: float
    yes_ask: Optional[float]
    no_ask: Optional[float]
    yes_bid: Optional[float]
    no_bid: Optional[float]
    volume: Optional[float]
    notes: Optional[str]


class ForecastRow(BaseModel):
    station: str
    target_date: datetime
    forecast_source: str
    run_time_utc: datetime
    predicted_high_f: float
    predicted_low_f: Optional[float]
    precip_probability: Optional[float]
    cloud_cover: Optional[float]
    wind_speed: Optional[float]
    raw_json_path_or_summary: Optional[str]


class ProbabilityResult(BaseModel):
    model_prob_yes: float
    model_prob_no: float
    predicted_high_f: float
    error_std_f: float
    station_bias_f: float
    method: str


class SignalRow(BaseModel):
    timestamp_utc: datetime
    market: str
    city: str
    station: str
    target_date: datetime
    threshold_f: float
    yes_ask: Optional[float]
    no_ask: Optional[float]
    predicted_high_f: Optional[float]
    model_prob_yes: Optional[float]
    model_prob_no: Optional[float]
    yes_edge: Optional[float]
    no_edge: Optional[float]
    best_side: Optional[str]
    signal: str
    confidence_label: str
    max_loss: Optional[float]
    max_profit: Optional[float]
    reward_risk: Optional[float]
    notes: Optional[str]
