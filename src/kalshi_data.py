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


def _cents_to_dollars(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return round(float(v) / 100.0, 4)
    except Exception:
        return None


def normalize_kalshi_market(m: Dict[str, Any]) -> Dict[str, Any]:
    ticker = m.get("ticker", "")
    # Build a human-readable title from the market's question + the yes side label.
    title = m.get("title") or ""
    sub = m.get("yes_sub_title") or m.get("subtitle") or ""
    if sub and sub.lower() not in title.lower():
        title = f"{title} {sub}".strip()
    if not title:
        title = ticker
    return {
        "venue": "kalshi",
        "id": ticker,
        "title": title,
        "url": f"https://kalshi.com/markets/{ticker}" if ticker else None,
        "yes_ask": _cents_to_dollars(m.get("yes_ask")),
        "no_ask": _cents_to_dollars(m.get("no_ask")),
        "yes_bid": _cents_to_dollars(m.get("yes_bid")),
        "no_bid": _cents_to_dollars(m.get("no_bid")),
        "volume": m.get("volume"),
        "end_date": m.get("close_time"),
        "category": m.get("category") or m.get("event_ticker"),
    }


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
        out.extend(normalize_kalshi_market(m) for m in markets)
        cursor = data.get("cursor")
        if not cursor:
            break
    return out
