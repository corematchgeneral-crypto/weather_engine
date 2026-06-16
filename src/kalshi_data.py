"""Fetch open markets from Kalshi and normalize to the common cross-venue schema.

Uses Kalshi's public trade API v2 market-data endpoint. Read-only market data is
generally accessible without trading auth; if your account/region requires a
token, set KALSHI_API_TOKEN in the environment and it will be sent as a bearer
header.

Prices on Kalshi are in cents (1..99); we convert to dollars (0..1). yes_ask /
no_ask are the prices you PAY to buy each side.

Requires internet (run on your own machine). Untested against the live API in
this environment -- if it errors, paste the message and it will be fixed.
"""

from typing import List, Dict, Any, Optional
import os

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


def _to_dollars(v) -> Optional[float]:
    """Kalshi v2 prices already come as dollar strings (e.g. '0.4700')."""
    try:
        if v is None or v == "":
            return None
        return round(float(v), 4)
    except Exception:
        return None


def normalize_kalshi_market(m: Dict[str, Any]) -> Dict[str, Any]:
    ticker = m.get("ticker", "")
    title = (m.get("title") or "").strip()
    sub = (m.get("yes_sub_title") or "").strip()
    # Append the strike sub-title only if it's a real strike (not a parlay leg list)
    if sub and "," not in sub and sub.lower() not in title.lower():
        title = f"{title} {sub}".strip()
    if not title:
        title = ticker
    return {
        "venue": "kalshi",
        "id": ticker,
        "title": title,
        "url": f"https://kalshi.com/markets/{ticker}" if ticker else None,
        "yes_ask": _to_dollars(m.get("yes_ask_dollars")),
        "no_ask": _to_dollars(m.get("no_ask_dollars")),
        "yes_bid": _to_dollars(m.get("yes_bid_dollars")),
        "no_bid": _to_dollars(m.get("no_bid_dollars")),
        "volume": m.get("volume_fp"),
        "end_date": m.get("close_time"),
        "category": m.get("event_ticker"),
    }


def _is_multivariate(m: Dict[str, Any]) -> bool:
    """Skip multivariate / parlay markets (their sub-titles are leg lists, not questions)."""
    if m.get("mve_selected_legs"):
        return True
    et = str(m.get("event_ticker") or "")
    return et.startswith("KXMVE") or bool(m.get("mve_collection_ticker"))


def fetch_open_markets(max_pages: int = 20, page_size: int = 200, timeout: int = 20) -> List[Dict[str, Any]]:
    """Fetch open Kalshi markets (paginated) and normalize them."""
    import requests  # lazy
    headers = {"Accept": "application/json"}
    token = os.environ.get("KALSHI_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    out: List[Dict[str, Any]] = []
    cursor = None
    for _ in range(max_pages):
        params = {"status": "open", "limit": page_size}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{KALSHI_BASE}/markets", params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        markets = data.get("markets", []) if isinstance(data, dict) else []
        if not markets:
            break
        for m in markets:
            if _is_multivariate(m):
                continue
            nm = normalize_kalshi_market(m)
            if nm["yes_ask"] is None and nm["no_ask"] is None:
                continue
            out.append(nm)
        cursor = data.get("cursor")
        if not cursor:
            break
    return out


def fetch_raw_sample(limit: int = 5, timeout: int = 20) -> List[Dict[str, Any]]:
    """Return a few RAW Kalshi market dicts (for inspecting field names)."""
    import requests  # lazy
    headers = {"Accept": "application/json"}
    token = os.environ.get("KALSHI_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(f"{KALSHI_BASE}/markets", params={"status": "open", "limit": limit}, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return (data.get("markets", []) if isinstance(data, dict) else [])[:limit]
