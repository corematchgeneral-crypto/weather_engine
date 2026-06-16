from typing import Dict, Any
import pandas as pd


def _to_float_or_none(v):
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return None


def validate_market_row(row: pd.Series) -> Dict[str, Any]:
    """
    Validate a market snapshot row and return a dict with:
    - is_valid: bool (no hard errors)
    - issues: list of strings (hard issues that should prevent signaling)
    - warnings: list of strings (soft issues to note)

    Rules implemented:
    - yes_ask, no_ask, yes_bid, no_bid must be in [0,1] when present -> issue
    - yes_ask >= yes_bid when both present -> issue
    - no_ask >= no_bid when both present -> issue
    - yes_ask + no_ask should normally be >= ~1.00 -> warning if < 0.99
    - yes_bid + no_bid should normally be <= ~1.00 -> warning if > 1.02
    """
    issues = []
    warnings = []

    yes_ask = _to_float_or_none(row.get("yes_ask"))
    no_ask = _to_float_or_none(row.get("no_ask"))
    yes_bid = _to_float_or_none(row.get("yes_bid"))
    no_bid = _to_float_or_none(row.get("no_bid"))

    # Bounds checks
    if yes_ask is not None and not (0.0 <= yes_ask <= 1.0):
        issues.append("YES_ASK_OOB")
    if no_ask is not None and not (0.0 <= no_ask <= 1.0):
        issues.append("NO_ASK_OOB")
    if yes_bid is not None and not (0.0 <= yes_bid <= 1.0):
        issues.append("YES_BID_OOB")
    if no_bid is not None and not (0.0 <= no_bid <= 1.0):
        issues.append("NO_BID_OOB")

    # Ask vs bid relationships
    if yes_ask is not None and yes_bid is not None and yes_ask < yes_bid:
        issues.append("YES_ASK_LT_BID")
    if no_ask is not None and no_bid is not None and no_ask < no_bid:
        issues.append("NO_ASK_LT_BID")

    # Both asks missing is a hard issue
    if yes_ask is None and no_ask is None:
        issues.append("NO_ASKS")

    # Sum checks (soft warnings)
    asks_sum = 0.0
    if yes_ask is not None:
        asks_sum += yes_ask
    if no_ask is not None:
        asks_sum += no_ask
    if asks_sum > 0 and asks_sum < 0.99:
        warnings.append("ASKS_SUM_LT_1")

    bids_sum = 0.0
    if yes_bid is not None:
        bids_sum += yes_bid
    if no_bid is not None:
        bids_sum += no_bid
    if bids_sum > 1.02:
        warnings.append("BIDS_SUM_GT_1")

    is_valid = len(issues) == 0
    return {"is_valid": is_valid, "issues": issues, "warnings": warnings}
