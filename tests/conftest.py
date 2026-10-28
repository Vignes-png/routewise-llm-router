"""Shared pytest setup."""

import pytest


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    """Send EVERY test's observability writes to a throwaway SQLite file.

    Without this, running the test suite would write rows into your real
    routewise.db. The file lives in pytest's temporary folder per test.
    """
    monkeypatch.setenv("ROUTEWISE_DB_PATH", str(tmp_path / "test_routewise.db"))