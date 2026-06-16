"""Pull a Polymarket temperature event (full bucket ladder + live prices).

Given a Polymarket event URL or slug, this queries the public Gamma API and
returns one row per bucket contract in the same schema the signal engine uses
(comparison / unit / threshold / yes_ask / no_ask / ...). No API key required.

Bucket label parsing handles the shapes Polymarket uses for temperature:
  "28°C"            -> EQUALS 28 C
  "34°C or higher"  -> ATLEAST 34 C
  "25°C or below"   -> ATMOST 25 C
  "72-73°F"         -> RANGE 72 73 F
"""

from typing import Optional, List, Dict, Any
import json
import re
from datetime import datetime, timezone

GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"

_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}


def slug_from_url(url_or_slug: str) -> str:
    """Accept a full Polymarket event URL or a bare slug; return the slug."""
    s = url_or_slug.strip().rstrip("/")
    if "/event/" in s:
        s = s.split("/event/", 1)[1]
    if "?" in s:
        s = s.split("?", 1)[0]
    return s


def fetch_event(slug: str, timeout: int = 20) -> Dict[str, Any]:
    """Fetch a single event by slug from the Gamma API."""
    import requests  # lazy: keeps the pure parsing helpers importable without requests
    r = requests.get(GAMMA_EVENTS_URL, params={"slug": slug}, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    events = data if isinstance(data, list) else data.get("events", [data])
    if not events:
        raise ValueError(f"No Polymarket event found for slug {slug!r}")
    return events[0]


def parse_event_title(title: str, slug: str = "") -> Dict[str, Any]:
    """Extract city, underlying (HIGH/LOW) and target_date from the event title."""
    underlying = "LOW" if re.search(r"lowest", title, re.I) else "HIGH"

    city = None
    m = re.search(r"temperature in (.+?) on ", title, re.I)
    if m:
        city = m.group(1).strip()

    target_date = None
    m = re.search(r"on\s+([A-Za-z]+)\s+(\d{1,2}),?\s*(\d{4})", title)
    if m and m.group(1).lower() in _MONTHS:
        target_date = datetime(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2))).date()
    if target_date is None and slug:  # fallback: parse slug like ...-june-16-2026
        m = re.search(r"([a-z]+)-(\d{1,2})-(\d{4})", slug)
        if m and m.group(1) in _MONTHS:
            target_date = datetime(int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2))).date()
    return {"city": city, "underlying": underlying, "target_date": target_date}


def parse_bucket_label(label: str) -> Optional[Dict[str, Any]]:
    """Parse a bucket label into (comparison, threshold, threshold_high, unit)."""
    if not label:
        return None
    text = label.replace("\u00b0", "").strip()  # drop degree symbol
    unit = "F" if re.search(r"F\b", text) else "C"
    low = bool(re.search(r"or below|or lower|below", text, re.I))
    high = bool(re.search(r"or higher|or above|above", text, re.I))

    if high:
        m = re.search(r"-?\d+", text)
        return {"comparison": "ATLEAST", "threshold": int(m.group()), "threshold_high": None, "unit": unit} if m else None
    if low:
        m = re.search(r"-?\d+", text)
        return {"comparison": "ATMOST", "threshold": int(m.group()), "threshold_high": None, "unit": unit} if m else None

    # Range like "72-73" / "72 to 73" (match the dash as a separator, not a sign)
    rng = re.search(r"(-?\d+)\s*(?:to|[-\u2013])\s*(-?\d+)", text)
    if rng:
        a, b = int(rng.group(1)), int(rng.group(2))
        return {"comparison": "RANGE", "threshold": min(a, b), "threshold_high": max(a, b), "unit": unit}

    m = re.search(r"-?\d+", text)
    return {"comparison": "EQUALS", "threshold": int(m.group()), "threshold_high": None, "unit": unit} if m else None


def _market_prices(market: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Return {'yes': ask, 'no': ask} from a market's outcome prices.

    Uses outcomePrices (mid/last). These approximate the ask; real asks include
    a small spread, so treat resulting edges as slightly optimistic.
    """
    outcomes = market.get("outcomes")
    prices = market.get("outcomePrices")
    try:
        outcomes = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
        prices = json.loads(prices) if isinstance(prices, str) else prices
    except Exception:
        return {"yes": None, "no": None}
    if not outcomes or not prices:
        return {"yes": None, "no": None}
    mp = {str(o).strip().lower(): float(p) for o, p in zip(outcomes, prices)}
    return {"yes": mp.get("yes"), "no": mp.get("no")}


def event_to_rows(event: Dict[str, Any], station: Optional[str] = None) -> List[dict]:
    """Convert a Gamma event into market-CSV rows (one per bucket)."""
    slug = event.get("slug", "")
    meta = parse_event_title(event.get("title", ""), slug)
    now_iso = datetime.now(timezone.utc).isoformat()
    rows = []
    for mk in event.get("markets", []) or []:
        label = mk.get("groupItemTitle") or mk.get("question") or ""
        bucket = parse_bucket_label(label)
        if not bucket:
            continue
        prices = _market_prices(mk)
        vol = mk.get("volumeNum") or mk.get("volume") or 0
        try:
            vol = float(vol)
        except Exception:
            vol = 0.0
        rows.append({
            "timestamp_utc": now_iso,
            "market": event.get("title", "")[:80],
            "city": meta["city"],
            "station": station or "",
            "target_date": meta["target_date"].isoformat() if meta["target_date"] else "",
            "threshold_f": bucket["threshold"],
            "threshold_high": bucket["threshold_high"] if bucket["threshold_high"] is not None else "",
            "unit": bucket["unit"],
            "comparison": bucket["comparison"],
            "underlying": meta["underlying"],
            "direction": "ABOVE",
            "yes_ask": prices["yes"] if prices["yes"] is not None else "",
            "no_ask": prices["no"] if prices["no"] is not None else "",
            "yes_bid": "",
            "no_bid": "",
            "volume": vol,
            "notes": str(label).strip(),
        })
    return rows


def city_to_station_map(cfg) -> Dict[str, str]:
    """Build {city_name_lower: station_key} from config so we can match by city."""
    return {sc.city.strip().lower(): key for key, sc in cfg.stations.items()}


# Polymarket city slug -> our config station key. Used by the multi-city scanner
# to build event slugs deterministically and resolve the forecast station.
CITY_SLUGS: Dict[str, str] = {
    "paris": "PARIS", "london": "LONDON", "munich": "MUNICH", "amsterdam": "AMSTERDAM",
    "madrid": "MADRID", "seattle": "SEATTLE", "ankara": "ANKARA", "moscow": "MOSCOW",
    "milan": "MILAN", "nyc": "KLGA", "jeddah": "JEDDAH", "atlanta": "KATL",
    "warsaw": "WARSAW", "los-angeles": "KLAX", "toronto": "TORONTO",
    "san-francisco": "KSFO", "miami": "MIAMI", "houston": "HOUSTON",
    "shanghai": "SHANGHAI", "austin": "AUSTIN", "dallas": "DALLAS",
    "sao-paulo": "SAOPAULO", "buenos-aires": "BUENOSAIRES", "chicago": "KMDW",
    "denver": "DENVER", "seoul": "SEOUL", "hong-kong": "HONGKONG", "taipei": "TAIPEI",
    "chongqing": "CHONGQING", "kuala-lumpur": "KUALALUMPUR", "helsinki": "HELSINKI",
    "beijing": "BEIJING", "shenzhen": "SHENZHEN", "chengdu": "CHENGDU",
    "wellington": "WELLINGTON", "tokyo": "TOKYO", "lucknow": "LUCKNOW", "tel-aviv": "TELAVIV",
}

_MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july",
                "august", "september", "october", "november", "december"]


def build_event_slug(kind: str, city_slug: str, d) -> str:
    """Construct a Polymarket event slug, e.g.
    highest-temperature-in-paris-on-june-16-2026
    """
    month = _MONTH_NAMES[d.month - 1]
    return f"{kind}-temperature-in-{city_slug}-on-{month}-{d.day}-{d.year}"


def fetch_event_safe(slug: str, timeout: int = 20) -> Optional[Dict[str, Any]]:
    """Fetch an event by slug, returning None instead of raising (for scanning)."""
    try:
        return fetch_event(slug, timeout=timeout)
    except Exception:
        return None
