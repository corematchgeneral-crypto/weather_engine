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
    """Skip multivariate / parlay / provisional markets (auto-generated sports junk)."""
    if m.get("mve_selected_legs") or m.get("is_provisional"):
        return True
    et = str(m.get("event_ticker") or "")
    return et.startswith("KXMVE") or bool(m.get("mve_collection_ticker"))


def fetch_open_markets(max_pages: int = 40, page_size: int = 200, timeout: int = 30,
                       target: int = 2000, verbose: bool = True,
                       skip_categories=("Sports",)) -> List[Dict[str, Any]]:
    """Fetch open Kalshi markets via the /events endpoint (has categories).

    Paging the raw /markets list is ~99% provisional sports parlays, so instead
    we pull events (which carry a `category`), skip Sports + multivariate, and
    take their nested markets. This surfaces the Politics/Economics/Crypto/World
    markets that actually overlap Polymarket.
    """
    import requests  # lazy
    headers = {"Accept": "application/json"}
    token = os.environ.get("KALSHI_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    skip = {c.lower() for c in skip_categories}
    out: List[Dict[str, Any]] = []
    cursor = None
    scanned = 0
    for page in range(max_pages):
        params = {"status": "open", "limit": page_size, "with_nested_markets": "true"}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{KALSHI_BASE}/events", params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        events = data.get("events", []) if isinstance(data, dict) else []
        if not events:
            break
        scanned += len(events)
        for ev in events:
            et = str(ev.get("event_ticker") or "")
            cat = str(ev.get("category") or "")
            if et.startswith("KXMVE") or cat.lower() in skip:
                continue
            ev_title = (ev.get("title") or ev.get("sub_title") or "").strip()
            for m in ev.get("markets", []) or []:
                if _is_multivariate(m):
                    continue
                nm = normalize_kalshi_market(m)
                strike = (m.get("yes_sub_title") or "").strip()
                title = ev_title
                if strike and "," not in strike and strike.lower() not in title.lower():
                    title = f"{title} {strike}".strip()
                if title:
                    nm["title"] = title
                nm["category"] = cat
                if nm["yes_ask"] is not None or nm["no_ask"] is not None:
                    out.append(nm)
        if verbose:
            print(f"    kalshi events page {page + 1}: scanned {scanned} events, kept {len(out)} markets")
        cursor = data.get("cursor")
        if not cursor or len(out) >= target:
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
