from src.polymarket_data import (
    slug_from_url, parse_bucket_label, parse_event_title, event_to_rows,
)


def test_slug_from_url():
    assert slug_from_url("https://polymarket.com/event/highest-temperature-in-paris-on-june-16-2026") == \
        "highest-temperature-in-paris-on-june-16-2026"
    assert slug_from_url("highest-temperature-in-paris-on-june-16-2026/") == \
        "highest-temperature-in-paris-on-june-16-2026"
    assert slug_from_url("https://polymarket.com/event/foo?tid=1") == "foo"


def test_parse_bucket_label():
    assert parse_bucket_label("28°C") == {"comparison": "EQUALS", "threshold": 28, "threshold_high": None, "unit": "C"}
    assert parse_bucket_label("34°C or higher")["comparison"] == "ATLEAST"
    assert parse_bucket_label("25°C or below")["comparison"] == "ATMOST"
    r = parse_bucket_label("72-73°F")
    assert r == {"comparison": "RANGE", "threshold": 72, "threshold_high": 73, "unit": "F"}
    assert parse_bucket_label("90-91°F")["threshold"] == 90
    assert parse_bucket_label("-5°C")["threshold"] == -5


def test_parse_event_title():
    m = parse_event_title("Highest temperature in Paris on June 16, 2026?", "highest-temperature-in-paris-on-june-16-2026")
    assert m["city"] == "Paris"
    assert m["underlying"] == "HIGH"
    assert str(m["target_date"]) == "2026-06-16"
    low = parse_event_title("Lowest temperature in Shanghai on June 16, 2026?")
    assert low["underlying"] == "LOW" and low["city"] == "Shanghai"


def test_event_to_rows():
    event = {
        "slug": "highest-temperature-in-paris-on-june-16-2026",
        "title": "Highest temperature in Paris on June 16, 2026?",
        "markets": [
            {"groupItemTitle": "28°C", "outcomes": '["Yes","No"]', "outcomePrices": '["0.93","0.08"]', "volumeNum": 11656},
            {"groupItemTitle": "25°C or below", "outcomes": '["Yes","No"]', "outcomePrices": '["0.001","0.999"]', "volumeNum": 19422},
        ],
    }
    rows = event_to_rows(event, station="PARIS")
    assert len(rows) == 2
    assert rows[0]["comparison"] == "EQUALS" and rows[0]["yes_ask"] == 0.93 and rows[0]["unit"] == "C"
    assert rows[0]["station"] == "PARIS" and rows[0]["city"] == "Paris"
    assert rows[1]["comparison"] == "ATMOST"



def test_build_event_slug():
    from datetime import date
    from src.polymarket_data import build_event_slug, CITY_SLUGS
    assert build_event_slug("highest", "paris", date(2026, 6, 16)) == \
        "highest-temperature-in-paris-on-june-16-2026"
    assert build_event_slug("lowest", "hong-kong", date(2026, 6, 17)) == \
        "lowest-temperature-in-hong-kong-on-june-17-2026"
    # every scanner city maps to a station key
    assert all(isinstance(v, str) and v for v in CITY_SLUGS.values())
    assert len(CITY_SLUGS) >= 30
