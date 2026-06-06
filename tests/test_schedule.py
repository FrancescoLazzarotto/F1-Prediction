"""Tests for schedule resolution utilities."""

import pytest


def test_resolve_event_by_round(monkeypatch):
    """resolve_event with year+round should return correct tuple."""
    import pandas as pd
    from f1predict.data import schedule as S

    fake_schedule = pd.DataFrame({
        "RoundNumber": [1, 2, 3],
        "EventName": ["Bahrain", "Saudi Arabia", "Australia"],
        "OfficialEventName": ["Formula 1 Gulf Air Bahrain Grand Prix",
                              "Formula 1 Saudi Arabian Grand Prix",
                              "Formula 1 Australian Grand Prix"],
        "Location": ["Sakhir", "Jeddah", "Melbourne"],
        "Country": ["Bahrain", "Saudi Arabia", "Australia"],
        "CircuitShortName": ["Bahrain", "Jeddah", "Melbourne"],
        "EventFormat": ["conventional", "conventional", "conventional"],
        "EventDate": pd.to_datetime(["2024-03-02", "2024-03-09", "2024-03-24"]),
    })

    monkeypatch.setattr(S, "get_schedule", lambda year: fake_schedule)

    y, r, info = S.resolve_event(2024, 2, None, False)
    assert y == 2024
    assert r == 2
    assert info["name"] == "Saudi Arabia"


def test_resolve_event_by_gp_name(monkeypatch):
    import pandas as pd
    from f1predict.data import schedule as S

    fake_schedule = pd.DataFrame({
        "RoundNumber": [1, 2, 3],
        "EventName": ["Bahrain", "Saudi Arabia", "Australia"],
        "OfficialEventName": ["Grand Prix Bahrain", "Saudi GP", "Australian GP"],
        "Location": ["Sakhir", "Jeddah", "Melbourne"],
        "Country": ["Bahrain", "Saudi Arabia", "Australia"],
        "CircuitShortName": ["Bahrain", "Jeddah", "Melbourne"],
        "EventFormat": ["conventional"] * 3,
        "EventDate": pd.to_datetime(["2024-03-02", "2024-03-09", "2024-03-24"]),
    })
    monkeypatch.setattr(S, "get_schedule", lambda year: fake_schedule)

    y, r, info = S.resolve_event(2024, None, "australia", False)
    assert r == 3
    assert "Australia" in info["name"]


def test_resolve_event_no_args():
    from f1predict.data.schedule import resolve_event
    with pytest.raises(ValueError):
        resolve_event(2024, None, None, False)
