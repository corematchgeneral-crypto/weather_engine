#!/usr/bin/env python3
"""Option A -- Cross-venue scanner (Polymarket <-> Kalshi).

Pulls open markets from both venues, matches equivalent contracts by title, and
ranks cross-venue price gaps and arbitrage opportunities.

Usage:
  python -m scripts.scan_cross_venue
  python -m scripts.scan_cross_venue --min-similarity 0.7 --min-gap 0.05 --top 40
  python -m scripts.scan_cross_venue --fee 0.02 --min-arb 0.01

READ THIS: A flagged pair is a CANDIDATE, not a confirmed trade. Two markets can
read alike but resolve on different sources/dates/wording. ALWAYS open both URLs
and confirm the resolution criteria match before trading. Polymarket prices are
mids (no separate ask), so arbitrage figures are optimistic.

Requires internet (run on your own machine).
"""
from pathlib import Path
import argparse

from src.polymarket_data import fetch_active_markets
from src.kalshi_data import fetch_open_markets
from src.cross_venue import scan_pairs


def _fmt(x, p="{:.3f}"):
    return "n/a" if x is None else (p.format(x) if isinstance(x, (int, float)) else str(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-similarity", type=float, default=0.6, help="title match threshold (0-1)")
    parser.add_argument("--min-gap", type=float, default=0.04, help="min cross-venue YES price gap to show")
    parser.add_argument("--min-arb", type=float, default=None, help="if set, only show pairs with arb_profit >= this")
    parser.add_argument("--fee", type=float, default=0.0, help="per-leg fee assumption (dollars)")
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    print("Fetching Polymarket active markets ...")
    try:
        pm = fetch_active_markets(max_pages=args.max_pages)
    except Exception as e:
        print(f"  Polymarket fetch failed: {e}")
        pm = []
    print(f"  Polymarket markets: {len(pm)}")

    print("Fetching Kalshi open markets ...")
    try:
        kl = fetch_open_markets(max_pages=args.max_pages)
    except Exception as e:
        print(f"  Kalshi fetch failed: {e} (if 401/403, set KALSHI_API_TOKEN)")
        kl = []
    print(f"  Kalshi markets: {len(kl)}")

    if not pm or not kl:
        print("Need markets from BOTH venues to compare. Aborting.")
        return

    print(f"Matching (similarity >= {args.min_similarity}) ...")
    results = scan_pairs(pm, kl, min_similarity=args.min_similarity, fee=args.fee)

    # Filter to interesting rows
    shown = []
    for r in results:
        arb = r.get("arb_profit")
        gap = r.get("value_gap") or 0
        if args.min_arb is not None:
            if arb is not None and arb >= args.min_arb:
                shown.append(r)
        elif (arb is not None and arb > 0) or gap >= args.min_gap:
            shown.append(r)

    print(f"\n{len(shown)} candidate(s) (of {len(results)} matched pairs):\n")
    for r in shown[: args.top]:
        arb = r.get("arb_profit")
        arb_str = f"ARB +{arb:.1%}  ({r.get('arb_desc')})" if (arb is not None and arb > 0) else ""
        print(f"sim={_fmt(r.get('similarity'))}  gap={_fmt(r.get('value_gap'))}  "
              f"YES {r.get('venue_a')}={_fmt(r.get('yes_a'),'{:.2f}')} vs "
              f"{r.get('venue_b')}={_fmt(r.get('yes_b'),'{:.2f}')}  "
              f"cheaper YES@{r.get('cheaper_yes_venue')}  {arb_str}")
        print(f"    A: {str(r.get('title_a'))[:80]}  -> {r.get('url_a')}")
        print(f"    B: {str(r.get('title_b'))[:80]}  -> {r.get('url_b')}")

    print("\nReminder: confirm BOTH markets resolve on the SAME criteria before trading. "
          "Polymarket prices are mids, so arb figures are optimistic; size for fees + slippage.")


if __name__ == "__main__":
    main()
