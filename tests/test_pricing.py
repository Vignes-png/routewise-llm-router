"""Tests for app/pricing.py (cost ESTIMATES). No network or API key involved."""

from datetime import date

import pytest

from app import pricing
from app.pricing import PRICING, PRICING_CHECKED_ON, PRICING_SOURCE, estimate_cost_usd

FLASH_LITE = "gemini/gemini-3.5-flash-lite"
FLASH_36 = "gemini/gemini-3.6-flash"


def test_flash_lite_rates():
    # 1M input tokens at $0.30 + 1M output tokens at $2.50
    assert estimate_cost_usd(FLASH_LITE, 1_000_000, 1_000_000, date(2026, 11, 1)) == pytest.approx(2.80)


def test_small_request_cost():
    # 3 in + 4 out on flash-lite: (3*0.30 + 4*2.50) / 1e6
    assert estimate_cost_usd(FLASH_LITE, 3, 4, date(2026, 11, 1)) == pytest.approx(1.09e-5)


def test_flash_36_rate_through_end_of_2026():
    assert estimate_cost_usd(FLASH_36, 1_000_000, 1_000_000, date(2026, 12, 31)) == pytest.approx(0.75 + 3.75)


def test_flash_36_rate_doubles_in_2027():
    assert estimate_cost_usd(FLASH_36, 1_000_000, 1_000_000, date(2027, 1, 1)) == pytest.approx(1.50 + 7.50)


def test_zero_tokens_is_zero_cost_not_unknown():
    assert estimate_cost_usd(FLASH_LITE, 0, 0, date(2026, 11, 1)) == 0.0


@pytest.mark.parametrize(
    "prompt_tokens, completion_tokens",
    [(None, 10), (10, None), (None, None), (-1, 5), (5, -1)],
)
def test_unknown_when_usage_missing_or_invalid(prompt_tokens, completion_tokens):
    assert estimate_cost_usd(FLASH_LITE, prompt_tokens, completion_tokens, date(2026, 11, 1)) is None


def test_unknown_model_gives_none():
    assert estimate_cost_usd("gemini/not-in-the-table", 10, 10, date(2026, 11, 1)) is None


def test_date_outside_every_period_gives_none(monkeypatch):
    only_future = {"gemini/x": [pricing.PricePeriod(1.0, 1.0, valid_from=date(2030, 1, 1))]}
    monkeypatch.setattr(pricing, "PRICING", only_future)
    assert estimate_cost_usd("gemini/x", 10, 10, date(2026, 11, 1)) is None


def test_source_and_check_date_are_documented():
    assert PRICING_SOURCE.startswith("https://")
    date.fromisoformat(PRICING_CHECKED_ON)  # must be a valid ISO date


def test_every_configured_model_has_a_price():
    from app.config import MODEL_BY_TIER

    for model in MODEL_BY_TIER.values():
        assert model in PRICING, f"add a price entry for {model}"