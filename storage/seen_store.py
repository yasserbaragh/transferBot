"""SQLite-backed record of which articles have already been through the
pipeline, keyed by link.

This is purely a dedup gate to stop the same RSS entries (feeds keep
recent history, so every fetch re-returns old ones) from being sent to
the LLM extraction step more than once. It is not the structured rumor
storage described in CLAUDE.md's step 5 - that's a separate concern for
later, once clustering/confidence scoring exist.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ingestion.rss_fetcher import RawArticle

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "seen_articles.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_articles (
    link TEXT PRIMARY KEY,
    first_seen TEXT NOT NULL
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


def filter_new(articles: list[RawArticle], path: Path = DB_PATH) -> list[RawArticle]:
    """Return only the articles whose link isn't already recorded as seen."""
    with _connect(path) as conn:
        seen = {row[0] for row in conn.execute("SELECT link FROM seen_articles")}
    return [a for a in articles if a.link and a.link not in seen]


def mark_seen(articles: list[RawArticle], path: Path = DB_PATH) -> None:
    """Record articles as seen so future runs skip them."""
    now = datetime.now(timezone.utc).isoformat()
    with _connect(path) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO seen_articles (link, first_seen) VALUES (?, ?)",
            [(a.link, now) for a in articles if a.link],
        )
