"""Local embedding generation for rumor clustering (pipeline step 3).

Runs entirely on-device via sentence-transformers - no API calls, no
per-request cost - since this step runs on every extracted rumor.
Just embedding generation here; comparing embeddings against existing
clusters to decide "new cluster" vs "attach as sighting" is the next
piece, built on top of this.
"""

import os
import unicodedata

import numpy as np
from sentence_transformers import SentenceTransformer

from ingestion.llm_filter import ExtractedRumor
from ingestion.llm_filter import extract_rumors
from ingestion.relevance_filter import filter_transfer_related
from ingestion.rss_fetcher import fetch_all_sources

# Multilingual so a Spanish/German/Italian/French report and an English
# report of the same rumor still embed close enough to cluster together -
# rumor.title (still in its source language) feeds into _rumor_text below
# alongside the already English-normalized players/clubs, so the model
# needs to handle more than just English. Small, fast, good enough for
# short news snippets. Override via env var without touching code if a
# different local model is ever preferred.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    # Loaded lazily and cached at module level - loading the model is
    # the expensive part (~1s), so we pay that once per process, not
    # once per rumor.
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _strip_accents(text: str) -> str:
    # So "Guimaraes" and "Guimarães" (or "Vinicius"/"Vinícius") embed close
    # enough to cluster together instead of fragmenting into separate
    # rumors just because different outlets spell the name differently.
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def _rumor_text(rumor: ExtractedRumor) -> str:
    players = _strip_accents(" & ".join(rumor.players))
    club_parts = ([rumor.current_club] if rumor.current_club else []) + rumor.linked_clubs
    clubs = _strip_accents(" to ".join(club_parts))
    parts = [players, clubs, rumor.title]
    return " - ".join(p for p in parts if p)


def embed_text(text: str) -> np.ndarray:
    return _get_model().encode(text, normalize_embeddings=True)


def embed_rumor(rumor: ExtractedRumor) -> np.ndarray:
    return embed_text(_rumor_text(rumor))


if __name__ == "__main__":
    candidates = filter_transfer_related(fetch_all_sources())
    rumors = extract_rumors(candidates)
    print(f"{len(rumors)} confirmed transfer rumors, embedding...\n")

    for r in rumors[:10]:
        vec = embed_rumor(r)
        print(f"[{r.source_tier}] {_rumor_text(r)}")
        print(f"  embedding shape={vec.shape} first5={vec[:5]}\n")
