"""Structured storage for extracted transfer rumors.

Storage-only for now: every extracted rumor is saved as its own new
cluster with a single sighting. Matching a new rumor against an
existing cluster ("same rumor, new source") isn't implemented yet -
that's the next piece, wired in on top of this storage.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ingestion.llm_filter import ExtractedRumor

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "rumors.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rumor_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player TEXT,
    clubs TEXT NOT NULL,
    representative_text TEXT NOT NULL,
    embedding BLOB,
    first_seen TEXT NOT NULL,
    last_updated TEXT NOT NULL,
    confidence REAL NOT NULL,
    tiers_seen TEXT NOT NULL,
    outcome TEXT
);

CREATE TABLE IF NOT EXISTS rumor_sightings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER NOT NULL REFERENCES rumor_clusters(id),
    source_name TEXT NOT NULL,
    source_tier TEXT NOT NULL,
    link TEXT NOT NULL,
    title TEXT NOT NULL,
    fee TEXT,
    wage TEXT,
    llm_confidence REAL NOT NULL,
    seen_at TEXT NOT NULL
);
"""
# embedding: unused placeholder until the clustering piece is wired in -
#   will hold a serialized vector for similarity matching.
# outcome: unused placeholder for later model training - meant to record
#   whether the rumor actually happened (e.g. "confirmed" / "denied") once
#   that's known. Nothing writes to this column yet.


@contextmanager
def _connect(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_rumor(rumor: ExtractedRumor, path: Path = DB_PATH) -> int:
    """Save an extracted rumor as a brand-new cluster with one sighting.

    No matching against existing clusters yet - every call creates a new
    cluster. That comes with the clustering piece, built on top of this.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _connect(path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO rumor_clusters
                (player, clubs, representative_text, first_seen,
                 last_updated, confidence, tiers_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rumor.player,
                json.dumps(rumor.clubs),
                rumor.title,
                now,
                now,
                rumor.confidence,
                json.dumps([rumor.source_tier]),
            ),
        )
        cluster_id = cursor.lastrowid

        conn.execute(
            """
            INSERT INTO rumor_sightings
                (cluster_id, source_name, source_tier, link, title,
                 fee, wage, llm_confidence, seen_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cluster_id,
                rumor.source_name,
                rumor.source_tier,
                rumor.link,
                rumor.title,
                rumor.fee,
                rumor.wage,
                rumor.confidence,
                now,
            ),
        )
    return cluster_id


def save_rumors(rumors: list[ExtractedRumor], path: Path = DB_PATH) -> list[int]:
    return [save_rumor(r, path=path) for r in rumors]
