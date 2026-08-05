"""Structured storage for extracted transfer rumors.

Every saved rumor is embedded and compared against recently-updated
clusters (clustering/matcher.py). A close enough match gets attached as
a new sighting on the existing cluster ("same rumor, new source"); no
close match creates a brand-new cluster ("genuinely new rumor").
"""

import json
import sqlite3
import unicodedata
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
    players TEXT NOT NULL,
    current_club TEXT,
    linked_clubs TEXT NOT NULL,
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


def _migrate_legacy_schema(path: Path) -> None:
    """Move an incompatible DB aside instead of altering it in place -
    there's nothing sane to migrate row-by-row, so this keeps the old
    data around as a backup instead of losing it."""
    if not path.exists():
        return
    conn = sqlite3.connect(path)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(rumor_clusters)").fetchall()}
    finally:
        conn.close()
    if columns and "players" not in columns:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        backup = path.with_name(f"{path.stem}.legacy-{stamp}{path.suffix}")
        path.rename(backup)
        print(f"[migrate] old rumor_clusters schema detected - moved {path.name} to {backup.name}, starting fresh")


@contextmanager
def _connect(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_schema(path)
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _normalize_key(text: str) -> str:
    stripped = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return stripped.lower().strip()


def _merge_names(existing: list[str], new: list[str]) -> list[str]:
    """Union two name lists, treating case/accent variants of the same
    name (e.g. "Bruno Guimaraes" vs "Bruno Guimarães") as duplicates."""
    seen = {_normalize_key(x) for x in existing}
    merged = list(existing)
    for item in new:
        key = _normalize_key(item)
        if key and key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


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
            (players, current_club, linked_clubs, representative_text, embedding,
             first_seen, last_updated, confidence, tiers_seen)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            json.dumps(rumor.players),
            rumor.current_club,
            json.dumps(rumor.linked_clubs),
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
        "SELECT tiers_seen, players, current_club, linked_clubs FROM rumor_clusters WHERE id = ?",
        (cluster_id,),
    ).fetchone()
    tiers_seen = json.loads(row[0])
    tiers_seen.append(rumor.source_tier)

    # A new sighting can name the player slightly differently, or only
    # mention a subset of the clubs already on record - merge rather than
    # overwrite so the cluster keeps everything reported about it so far.
    players = _merge_names(json.loads(row[1]), rumor.players)
    current_club = row[2] or rumor.current_club
    linked_clubs = _merge_names(json.loads(row[3]), rumor.linked_clubs)

    conn.execute(
        """
        UPDATE rumor_clusters
        SET last_updated = ?, tiers_seen = ?, players = ?, current_club = ?, linked_clubs = ?
        WHERE id = ?
        """,
        (now, json.dumps(tiers_seen), json.dumps(players), current_club, json.dumps(linked_clubs), cluster_id),
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
    players: list[str]
    current_club: str | None
    linked_clubs: list[str]
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
            SELECT id, players, current_club, linked_clubs, confidence, tiers_seen, first_seen, last_updated
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
                    players=json.loads(row[1]),
                    current_club=row[2],
                    linked_clubs=json.loads(row[3]),
                    confidence=row[4],
                    tiers_seen=json.loads(row[5]),
                    first_seen=row[6],
                    last_updated=row[7],
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
