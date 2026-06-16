"""Cross-venue relative-value / arbitrage detection.

Pure, offline-testable logic for:
  - normalizing a binary market from any venue into a common schema,
  - matching equivalent markets across venues (Polymarket <-> Kalshi),
  - scoring the cross-venue price gap and any arbitrage.

A "common market" is a dict with keys:
  venue, id, title, url, yes_ask, no_ask, yes_bid, no_bid, volume, end_date, category

Prices are in dollars (0..1). yes_ask/no_ask are what you PAY to buy that side.

IMPORTANT (read before trading): two markets that look equivalent can resolve
differently (different data source, station, cutoff, or wording). This module
proposes *candidates*; a human must confirm the resolution criteria match before
treating a gap as a real edge. Polymarket prices are mids (no separate ask), so
arbitrage numbers using them are optimistic.
"""

from typing import List, Dict, Any, Optional, Tuple
import re

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "be", "is",
    "will", "by", "this", "that", "with", "than", "above", "below", "over", "under",
    "more", "less", "have", "has", "it", "as", "from", "market", "?", "yes", "no",
}


def normalize_tokens(title: str) -> set:
    """Lowercase, strip punctuation, drop stopwords -> a set of meaningful tokens."""
    if not title:
        return set()
    t = re.sub(r"[^a-z0-9\s]", " ", title.lower())
    return {w for w in t.split() if w and w not in _STOPWORDS}


def extract_numbers(title: str) -> set:
    """Numbers in a title (thresholds, counts). Used as a hard guard so we don't
    match 'above 70' with 'above 80'."""
    return set(re.findall(r"\d+", title or ""))


def title_similarity(a: str, b: str) -> float:
    """Jaccard similarity of meaningful tokens, with a hard penalty when both
    titles contain numbers that don't overlap (different thresholds)."""
    ta, tb = normalize_tokens(a), normalize_tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    jac = inter / union if union else 0.0
    na, nb = extract_numbers(a), extract_numbers(b)
    if na and nb and not (na & nb):
        jac *= 0.3  # numbers disagree -> very unlikely the same contract
    return jac


def find_matches(
    markets_a: List[Dict[str, Any]],
    markets_b: List[Dict[str, Any]],
    min_similarity: float = 0.6,
) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    """Greedy best-match markets across two venues by title similarity.

    Returns list of (market_a, market_b, similarity), each market used at most once,
    sorted by similarity descending.
    """
    scored = []
    for ma in markets_a:
        for mb in markets_b:
            sim = title_similarity(ma.get("title", ""), mb.get("title", ""))
            if sim >= min_similarity:
                scored.append((sim, ma, mb))
    scored.sort(key=lambda x: x[0], reverse=True)

    used_a, used_b, out = set(), set(), []
    for sim, ma, mb in scored:
        ka, kb = id(ma), id(mb)
        if ka in used_a or kb in used_b:
            continue
        used_a.add(ka)
        used_b.add(kb)
        out.append((ma, mb, sim))
    return out


def _f(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def evaluate_pair(a: Dict[str, Any], b: Dict[str, Any], fee: float = 0.0) -> Dict[str, Any]:
    """Compute the cross-venue value gap and arbitrage for a matched pair.

    - value_gap: difference in implied YES probability between the two venues.
    - arb_profit: best riskless profit from buying YES on one venue and NO on the
      other (1 - (yes_ask_X + no_ask_Y) - fees), if positive.
    """
    ya, na = _f(a.get("yes_ask")), _f(a.get("no_ask"))
    yb, nb = _f(b.get("yes_ask")), _f(b.get("no_ask"))

    # implied YES price per venue (prefer ask; fall back to 1 - no_ask)
    yes_a = ya if ya is not None else (1 - na if na is not None else None)
    yes_b = yb if yb is not None else (1 - nb if nb is not None else None)
    value_gap = abs(yes_a - yes_b) if (yes_a is not None and yes_b is not None) else None
    cheaper_yes_venue = None
    if yes_a is not None and yes_b is not None:
        cheaper_yes_venue = a.get("venue") if yes_a < yes_b else b.get("venue")

    # arbitrage: YES on one venue + NO on the other should cover all outcomes
    arb_legs = []
    if ya is not None and nb is not None:
        arb_legs.append((ya + nb, f"BUY YES@{a.get('venue')} + NO@{b.get('venue')}"))
    if yb is not None and na is not None:
        arb_legs.append((yb + na, f"BUY YES@{b.get('venue')} + NO@{a.get('venue')}"))
    arb_profit, arb_desc = None, None
    if arb_legs:
        cost, desc = min(arb_legs, key=lambda x: x[0])
        arb_profit = round(1.0 - cost - fee, 4)
        arb_desc = desc

    return {
        "title_a": a.get("title"),
        "title_b": b.get("title"),
        "venue_a": a.get("venue"),
        "venue_b": b.get("venue"),
        "yes_a": round(yes_a, 4) if yes_a is not None else None,
        "yes_b": round(yes_b, 4) if yes_b is not None else None,
        "value_gap": round(value_gap, 4) if value_gap is not None else None,
        "cheaper_yes_venue": cheaper_yes_venue,
        "arb_profit": arb_profit,
        "arb_desc": arb_desc,
        "url_a": a.get("url"),
        "url_b": b.get("url"),
        "volume_a": a.get("volume"),
        "volume_b": b.get("volume"),
    }


def scan_pairs(
    markets_a: List[Dict[str, Any]],
    markets_b: List[Dict[str, Any]],
    min_similarity: float = 0.6,
    fee: float = 0.0,
) -> List[Dict[str, Any]]:
    """Match across venues and evaluate every pair, sorted by arb then value gap."""
    results = []
    for ma, mb, sim in find_matches(markets_a, markets_b, min_similarity):
        ev = evaluate_pair(ma, mb, fee=fee)
        ev["similarity"] = round(sim, 3)
        results.append(ev)
    results.sort(key=lambda r: (r.get("arb_profit") or -9, r.get("value_gap") or 0), reverse=True)
    return results
