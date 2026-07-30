"""Local embedding generation for rumor clustering (pipeline step 3).

Runs entirely on-device via sentence-transformers - no API calls, no
per-request cost - since this step runs on every extracted rumor.
Just embedding generation here; comparing embeddings against existing
clusters to decide "new cluster" vs "attach as sighting" is the next
piece, built on top of this.
"""

import os

import numpy as np
from sentence_transformers import SentenceTransformer

from ingestion.llm_filter import ExtractedRumor

# Small, fast, good enough for short news snippets. Override via env var
# without touching code if a different local model is ever preferred.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    # Loaded lazily and cached at module level - loading the model is
    # the expensive part (~1s), so we pay that once per process, not
    # once per rumor.
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _rumor_text(rumor: ExtractedRumor) -> str:
    parts = [rumor.player or "", " to ".join(rumor.clubs), rumor.title]
    return " - ".join(p for p in parts if p)


def embed_text(text: str) -> np.ndarray:
    return _get_model().encode(text, normalize_embeddings=True)


def embed_rumor(rumor: ExtractedRumor) -> np.ndarray:
    return embed_text(_rumor_text(rumor))


if __name__ == "__main__":
    sample = ExtractedRumor(
        source_name="test",
        source_tier="tier1",
        link="https://example.com",
        title="Player X close to Club Y move, medical scheduled",
        is_transfer_rumor=True,
        player="Player X",
        clubs=["Club Y"],
        confidence=0.9,
    )
    vec = embed_rumor(sample)
    print(f"embedding shape: {vec.shape}, dtype: {vec.dtype}")
    print(vec[:5])
