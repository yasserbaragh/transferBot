"""Decides whether a new rumor's embedding matches an existing cluster
("same rumor, new source" / "update") or is far enough from all of them
to be a genuinely new rumor.

Brute-force cosine similarity against recent clusters - no vector DB
needed at the volume this runs at (a handful of new rumors per 30-min
run, compared against a bounded recent-cluster window).
"""

import os

import numpy as np

# Cosine similarity threshold above which a new rumor is considered the
# same story as an existing cluster. Embeddings from clustering/embeddings.py
# are pre-normalized, so cosine similarity is just a dot product.
# This is the one knob that controls dedup aggressiveness - tune once
# real data shows how similar same-story vs different-story pairs
# actually score. Override via env var without touching code.
SIMILARITY_THRESHOLD = float(os.environ.get("CLUSTER_SIMILARITY_THRESHOLD", "0.72"))


def find_best_match(
    new_embedding: np.ndarray,
    candidates: list[tuple[int, np.ndarray]],
) -> tuple[int, float] | None:
    """Compare new_embedding against (cluster_id, embedding) candidates.

    Returns the (cluster_id, score) of the closest candidate if its score
    is at or above SIMILARITY_THRESHOLD, else None (meaning: no existing
    cluster is close enough, this is a new rumor).
    """
    best_id = None
    best_score = -1.0
    for cluster_id, embedding in candidates:
        score = float(np.dot(new_embedding, embedding))
        if score > best_score:
            best_id, best_score = cluster_id, score

    if best_id is not None and best_score >= SIMILARITY_THRESHOLD:
        return best_id, best_score
    return None
