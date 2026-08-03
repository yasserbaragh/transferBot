"""Structured storage for extracted transfer rumors.

Every saved rumor is embedded and compared against recently-updated
clusters (clustering/matcher.py). A close enough match gets attached as
a new sighting on the existing cluster ("same rumor, new source"); no
close match creates a brand-new cluster ("genuinely new rumor").
"""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from clustering.embeddings import embed_rumor
from clustering.matcher import find_best_match
from ingestion.llm_filter import ExtractedRumor
from scoring.credibility import compute_confidence

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "rumors.db"

# How far back to look for clusters a new rumor could attach to. Old,
# stale rumors shouldn't keep matching new unrelated ones just because
# they're semantically similar (e.g. the same player linked to the same
# club in different transfer windows)
RECENT_WINDOW_DAYS = 14

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
# outcome: unused placeholder for later model training - meant to record
# whether the rumor actually happened (e.g. "confirmed" / "denied") once
# that's known. Nothing writes to this column yet.


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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pack_embedding(embedding: np.ndarray) -> bytes:
    return embedding.astype(np.float32).tobytes()


def _unpack_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _recent_cluster_embeddings(conn: sqlite3.Connection) -> list[tuple[int, np.ndarray]]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=RECENT_WINDOW_DAYS)).isoformat()
    rows = conn.execute(
        """
        SELECT id, embedding FROM rumor_clusters
        WHERE last_updated >= ? AND embedding IS NOT NULL
        """,
        (cutoff,),
    ).fetchall()
    return [(row[0], _unpack_embedding(row[1])) for row in rows]


def _insert_sighting(conn: sqlite3.Connection, cluster_id: int, rumor: ExtractedRumor, now: str) -> None:
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


def _sightings_for_cluster(conn: sqlite3.Connection, cluster_id: int) -> list[tuple[str, str, float]]:
    rows = conn.execute(
        "SELECT source_name, source_tier, llm_confidence FROM rumor_sightings WHERE cluster_id = ?",
        (cluster_id,),
    ).fetchall()
    return [(row[0], row[1], row[2]) for row in rows]


def _recompute_confidence(conn: sqlite3.Connection, cluster_id: int) -> None:
    confidence = compute_confidence(_sightings_for_cluster(conn, cluster_id))
    conn.execute("UPDATE rumor_clusters SET confidence = ? WHERE id = ?", (confidence, cluster_id))


def _create_cluster(conn: sqlite3.Connection, rumor: ExtractedRumor, embedding: np.ndarray) -> int:
    now = _now()
    cursor = conn.execute(
        """
        INSERT INTO rumor_clusters
            (player, clubs, representative_text, embedding, first_seen,
             last_updated, confidence, tiers_seen)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            rumor.player,
            json.dumps(rumor.clubs),
            rumor.title,
            _pack_embedding(embedding),
            now,
            now,
            rumor.confidence,
            json.dumps([rumor.source_tier]),
        ),
    )
    cluster_id = cursor.lastrowid
    _insert_sighting(conn, cluster_id, rumor, now)
    _recompute_confidence(conn, cluster_id)
    return cluster_id


def _attach_sighting(conn: sqlite3.Connection, cluster_id: int, rumor: ExtractedRumor) -> None:
    now = _now()
    row = conn.execute(
        "SELECT tiers_seen FROM rumor_clusters WHERE id = ?", (cluster_id,)
    ).fetchone()
    tiers_seen = json.loads(row[0])
    tiers_seen.append(rumor.source_tier)
    conn.execute(
        "UPDATE rumor_clusters SET last_updated = ?, tiers_seen = ? WHERE id = ?",
        (now, json.dumps(tiers_seen), cluster_id),
    )
    _insert_sighting(conn, cluster_id, rumor, now)
    _recompute_confidence(conn, cluster_id)


def save_rumor(rumor: ExtractedRumor, path: Path = DB_PATH) -> tuple[int, bool]:
    """Save a rumor, matching it against recent clusters first.

    Returns (cluster_id, is_new_cluster) - is_new_cluster is False when
    the rumor was attached to an existing cluster as a new sighting
    instead of starting one.
    """
    embedding = embed_rumor(rumor)
    with _connect(path) as conn:
        candidates = _recent_cluster_embeddings(conn)
        match = find_best_match(embedding, candidates)
        if match is not None:
            cluster_id, _score = match
            _attach_sighting(conn, cluster_id, rumor)
            return cluster_id, False
        cluster_id = _create_cluster(conn, rumor, embedding)
        return cluster_id, True


def save_rumors(rumors: list[ExtractedRumor], path: Path = DB_PATH) -> list[tuple[int, bool]]:
    return [save_rumor(r, path=path) for r in rumors]


@dataclass
class SightingSummary:
    source_name: str
    source_tier: str
    link: str
    title: str
    fee: str | None
    wage: str | None


@dataclass
class ClusterDigest:
    id: int
    player: str | None
    clubs: list[str]
    confidence: float
    tiers_seen: list[str]
    first_seen: str
    last_updated: str
    sightings: list[SightingSummary] = field(default_factory=list)


def get_clusters_since(since: str, path: Path = DB_PATH) -> list[ClusterDigest]:
    """Clusters first created or last updated at/after `since` (ISO
    timestamp), each with its sightings - the set a periodic report
    should cover. Ordered by confidence, most credible first.
    """
    with _connect(path) as conn:
        cluster_rows = conn.execute(
            """
            SELECT id, player, clubs, confidence, tiers_seen, first_seen, last_updated
            FROM rumor_clusters
            WHERE last_updated >= ?
            ORDER BY confidence DESC
            """,
            (since,),
        ).fetchall()

        clusters = []
        for row in cluster_rows:
            cluster_id = row[0]
            sighting_rows = conn.execute(
                """
                SELECT source_name, source_tier, link, title, fee, wage
                FROM rumor_sightings WHERE cluster_id = ?
                ORDER BY seen_at
                """,
                (cluster_id,),
            ).fetchall()
            clusters.append(
                ClusterDigest(
                    id=cluster_id,
                    player=row[1],
                    clubs=json.loads(row[2]),
                    confidence=row[3],
                    tiers_seen=json.loads(row[4]),
                    first_seen=row[5],
                    last_updated=row[6],
                    sightings=[
                        SightingSummary(
                            source_name=s[0], source_tier=s[1], link=s[2],
                            title=s[3], fee=s[4], wage=s[5],
                        )
                        for s in sighting_rows
                    ],
                )
            )
        return clusters
