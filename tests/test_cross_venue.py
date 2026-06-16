import pytest

from src.cross_venue import (
    normalize_tokens, extract_numbers, title_similarity, find_matches,
    evaluate_pair, scan_pairs,
)


def test_normalize_and_numbers():
    assert "bitcoin" in normalize_tokens("Will Bitcoin close above 100000?")
    assert "the" not in normalize_tokens("the the the bitcoin")
    assert extract_numbers("High above 80 on June 30") == {"80", "30"}


def test_similarity_number_guard():
    # same wording, different threshold -> penalized below match threshold
    assert title_similarity("High temp NYC above 80", "High temp NYC above 70") < 0.3
    # genuinely similar -> high
    assert title_similarity("Will Bitcoin be above 100000 on June 30",
                            "Bitcoin above 100000 by June 30") > 0.5


def test_arbitrage_detection():
    a = [{"venue": "polymarket", "title": "Will X win the election", "yes_ask": 0.40, "no_ask": 0.62, "url": "a"}]
    b = [{"venue": "kalshi", "title": "Will X win election", "yes_ask": 0.47, "no_ask": 0.55, "url": "b"}]
    res = scan_pairs(a, b, min_similarity=0.5)
    assert len(res) == 1
    r = res[0]
    # buy YES@poly(0.40) + NO@kalshi(0.55) = 0.95 -> 5% arb
    assert r["arb_profit"] == pytest.approx(0.05, abs=1e-9)
    assert r["arb_desc"] == "BUY YES@polymarket + NO@kalshi"
    assert r["cheaper_yes_venue"] == "polymarket"


def test_no_arbitrage_when_prices_align():
    a = [{"venue": "polymarket", "title": "Will it rain tomorrow", "yes_ask": 0.55, "no_ask": 0.47}]
    b = [{"venue": "kalshi", "title": "Will it rain tomorrow", "yes_ask": 0.56, "no_ask": 0.46}]
    r = scan_pairs(a, b, min_similarity=0.5)[0]
    assert r["arb_profit"] <= 0  # 0.55+0.46=1.01 and 0.56+0.47=1.03 -> no riskless profit


def test_value_gap_only():
    a = [{"venue": "polymarket", "title": "Team A wins title", "yes_ask": 0.30, "no_ask": 0.72}]
    b = [{"venue": "kalshi", "title": "Team A wins title", "yes_ask": 0.40, "no_ask": 0.62}]
    r = scan_pairs(a, b, min_similarity=0.5)[0]
    assert r["value_gap"] == pytest.approx(0.10, abs=1e-9)
    assert r["cheaper_yes_venue"] == "polymarket"


def test_greedy_unique_matching():
    a = [{"venue": "polymarket", "title": "Bitcoin above 100000 June 30", "yes_ask": 0.5, "no_ask": 0.5}]
    b = [
        {"venue": "kalshi", "title": "Bitcoin above 100000 by June 30", "yes_ask": 0.5, "no_ask": 0.5},
        {"venue": "kalshi", "title": "Completely unrelated market", "yes_ask": 0.5, "no_ask": 0.5},
    ]
    matches = find_matches(a, b, min_similarity=0.5)
    assert len(matches) == 1
    assert "by June 30" in matches[0][1]["title"]
