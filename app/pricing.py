"""Configurable price table used ONLY to ESTIMATE the cost of a model call.

These numbers are not a bill. Real charges can differ: you may be on the free
tier (billed $0), use batch/flex/priority modes, caching or grounding, and
Google can change prices at any time. Edit PRICING below when prices change,
and update PRICING_CHECKED_ON at the same time.
"""

from dataclasses import dataclass
from datetime import date, datetime, timezone

# Where the rates below came from, and when they were read.
PRICING_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"
PRICING_CHECKED_ON = "2026-10-28"
PRICING_BASIS = (
    "Gemini Developer API, Standard mode, paid tier, text input/output, "
    "USD per 1M tokens; output price includes thinking tokens."
)


@dataclass(frozen=True)
class PricePeriod:
    """A rate that applies from `valid_from` to `valid_until` (both inclusive, UTC dates).

    None means "no limit" on that side.
    """

    input_per_million: float
    output_per_million: float
    valid_from: date | None = None
    valid_until: date | None = None


# Keys are the LiteLLM model ids used in app/config.py (MODEL_BY_TIER).
# A model that is not listed here simply gets an UNKNOWN cost (None).
PRICING: dict[str, list[PricePeriod]] = {
    "gemini/gemini-3.5-flash-lite": [
        PricePeriod(input_per_million=0.30, output_per_million=2.50),
    ],
    # Google's page lists a launch rate through 2026-12-31 and double that rate
    # starting 2027-01-01.
    "gemini/gemini-3.6-flash": [
        PricePeriod(0.75, 3.75, valid_until=date(2026, 12, 31)),
        PricePeriod(1.50, 7.50, valid_from=date(2027, 1, 1)),
    ],
}


def find_price(model: str, on_date: date) -> PricePeriod | None:
    """Return the rate that applies to `model` on `on_date`, or None if unknown."""
    for period in PRICING.get(model, []):
        starts_ok = period.valid_from is None or on_date >= period.valid_from
        ends_ok = period.valid_until is None or on_date <= period.valid_until
        if starts_ok and ends_ok:
            return period
    return None


def estimate_cost_usd(
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    on_date: date | None = None,
) -> float | None:
    """Estimated cost in USD, or None when it cannot be calculated honestly.

    None is returned when the model has no known rate for that date, or when
    the provider did not report BOTH token counts. Nothing is guessed.
    """
    if prompt_tokens is None or completion_tokens is None:
        return None
    if prompt_tokens < 0 or completion_tokens < 0:
        return None

    day = on_date or datetime.now(timezone.utc).date()
    price = find_price(model, day)
    if price is None:
        return None

    cost = (
        prompt_tokens * price.input_per_million
        + completion_tokens * price.output_per_million
    ) / 1_000_000
    return round(cost, 8)