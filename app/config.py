"""Central configuration for RouteWise: model mapping, timeout and API key."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Read the local .env file (if it exists) into environment variables.
# Variables that are already set in the environment are NOT overridden.
load_dotenv(PROJECT_ROOT / ".env")

# The ONLY place that maps a routing tier to a concrete model.
# To change a model later, edit this dictionary and nothing else.
MODEL_BY_TIER: dict[str, str] = {
    "cheap": "gemini/gemini-3.5-flash-lite",
    "strong": "gemini/gemini-3.6-flash",
}

# Seconds to wait for the provider before giving up (no automatic retries).
REQUEST_TIMEOUT_SECONDS = 60


def get_gemini_api_key() -> str | None:
    """Return the Gemini API key from the environment, or None if it is missing/blank."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key or None


def get_database_path() -> Path:
    """Return the SQLite file used for observability.

    Defaults to routewise.db in the project root. Set ROUTEWISE_DB_PATH in the
    environment (or .env) to use a different file. Read at call time so tests
    can point it at a temporary file.
    """
    override = os.environ.get("ROUTEWISE_DB_PATH", "").strip()
    return Path(override) if override else PROJECT_ROOT / "routewise.db"