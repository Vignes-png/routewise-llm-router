"""SQLite observability for POST /chat.

One row is written per /chat request that reaches the endpoint, including
failures. The table stores METADATA ONLY: never the prompt, the generated
answer, the API key, or raw provider error text. Failures are stored as a short
fixed `error_category` label.

Writing a row must never break a request, so record_chat_event() catches every
error and just logs it.
"""

import logging
import sqlite3
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_database_path

logger = logging.getLogger("routewise.observability")

# The ONLY values allowed in the error_category column.
ERROR_CATEGORIES = (
    "routing_model_unavailable",  # embedding model could not be loaded
    "routing_error",              # unexpected error while routing
    "missing_credentials",        # GEMINI_API_KEY not configured
    "provider_auth",              # provider rejected the key
    "rate_limit",                 # provider rate limit / quota
    "timeout",                    # provider did not answer in time
    "model_not_found",            # provider does not know the model id
    "bad_request",                # provider rejected the request
    "provider_error",             # any other provider problem
    "empty_response",             # provider answered with no text
    "unexpected_error",           # a bug on our side
)


def new_request_id() -> str:
    return uuid.uuid4().hex


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class ChatEvent:
    """Everything we remember about one /chat request. Field names = column names.

    The defaults describe the WORST case (an unexpected failure). The endpoint
    overwrites them as it learns more, so if something unforeseen happens the
    row still says "failure" instead of silently looking like a success.
    """

    request_id: str = field(default_factory=new_request_id)
    created_at: str = field(default_factory=utc_now_iso)  # UTC, request start
    status: str = "failure"                    # "success" or "failure"
    error_category: str | None = "unexpected_error"
    http_status: int | None = 500
    routing_method: str | None = None
    task_category: str | None = None
    similarity_score: float | None = None
    model_tier: str | None = None
    model: str | None = None
    latency_ms: int | None = None              # model call only; None if never called
    prompt_tokens: int | None = None           # only if the provider reported them
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None    # ESTIMATE, not a bill; None = unknown
    pricing_checked_on: str | None = None      # date of the rates used for the estimate


COLUMNS = [f.name for f in fields(ChatEvent)]

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chat_requests (
    request_id          TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    status              TEXT NOT NULL,
    error_category      TEXT,
    http_status         INTEGER,
    routing_method      TEXT,
    task_category       TEXT,
    similarity_score    REAL,
    model_tier          TEXT,
    model               TEXT,
    latency_ms          INTEGER,
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    total_tokens        INTEGER,
    estimated_cost_usd  REAL,
    pricing_checked_on  TEXT
)
"""

INSERT_SQL = (
    f"INSERT INTO chat_requests ({', '.join(COLUMNS)}) "
    f"VALUES ({', '.join(':' + c for c in COLUMNS)})"
)


def record_chat_event(event: ChatEvent, db_path: Path | str | None = None) -> bool:
    """Save one event. Returns True if saved, False if not. NEVER raises."""
    try:
        path = Path(db_path) if db_path else get_database_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=5)) as conn:
            conn.execute(CREATE_TABLE_SQL)
            conn.execute(INSERT_SQL, asdict(event))
            conn.commit()
        return True
    except Exception as exc:
        # SQLite's own messages describe the problem ("database is locked",
        # "unable to open database file") and never contain our row values, so
        # they are safe to log. For any other error type, log only the type.
        if isinstance(exc, sqlite3.Error):
            logger.warning("Could not record chat event (%s): %s", type(exc).__name__, exc)
        else:
            logger.warning("Could not record chat event (%s)", type(exc).__name__)
        return False


def fetch_recent_events(limit: int = 10, db_path: Path | str | None = None) -> list[dict]:
    """Return the newest rows (newest first) as dictionaries. Used by tests and the CLI below."""
    path = Path(db_path) if db_path else get_database_path()
    if not path.exists():
        return []
    with closing(sqlite3.connect(path, timeout=5)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(CREATE_TABLE_SQL)
        rows = conn.execute(
            "SELECT * FROM chat_requests ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


if __name__ == "__main__":
    # Quick look at the newest rows:  python -m app.observability
    events = fetch_recent_events(10)
    if not events:
        print(f"No rows yet in {get_database_path()}")
    for e in events:
        cost = "unknown" if e["estimated_cost_usd"] is None else f"${e['estimated_cost_usd']:.6f} (estimate)"
        print(
            f"{e['created_at']}  {e['status']:<7} http={e['http_status']}  "
            f"error={e['error_category']}  tier={e['model_tier']}  model={e['model']}  "
            f"latency_ms={e['latency_ms']}  tokens={e['prompt_tokens']}/{e['completion_tokens']}/{e['total_tokens']}  "
            f"cost={cost}  id={e['request_id']}"
        )