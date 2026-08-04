"""Tracks when the last digest report was generated, so each periodic
report only covers clusters that are new or updated since then.

Separate single-row table in the same DB rather than a field on
rumor_clusters - this is report-cadence bookkeeping, not rumor data.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "rumors.db"

# No report has run yet before this - using it as "since" means the
# first-ever report covers every cluster currently stored
_EPOCH = "1970-01-01T00:00:00+00:00"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS report_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_report_at TEXT NOT NULL
);
"""


@contextmanager
def _connect(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_last_report_time(path: Path = DB_PATH) -> str:
    with _connect(path) as conn:
        row = conn.execute("SELECT last_report_at FROM report_state WHERE id = 1").fetchone()
    return row[0] if row else _EPOCH


def set_last_report_time(when: str | None = None, path: Path = DB_PATH) -> None:
    when = when or datetime.now(timezone.utc).isoformat()
    with _connect(path) as conn:
        conn.execute(
            """
            INSERT INTO report_state (id, last_report_at) VALUES (1, ?)
            ON CONFLICT(id) DO UPDATE SET last_report_at = excluded.last_report_at
            """,
            (when,),
        )
