"""Computes a cluster's overall confidence from its sightings so far
(pipeline step 4). Isolated from storage so the formula can be tuned
independently of the DB layer.

Read as: how sure are we this rumor is real, given who's reporting it
and how many independent sources agree?
"""

# Tunable weights. Independent tier1 corroboration matters most, tier2
# less so, and both are counted by distinct source, not raw sighting
# count - an outlet re-posting the same rumor three times shouldn't look
# like three independent confirmations. Caps keep a handful of sources
# from pushing confidence past 1.0 on their own.
TIER1_SOURCE_WEIGHT = 0.15
TIER2_SOURCE_WEIGHT = 0.05
MAX_TIER1_BOOST = 0.45
MAX_TIER2_BOOST = 0.15


def compute_confidence(sightings: list[tuple[str, str, float]]) -> float:
    """sightings: (source_name, source_tier, llm_confidence) for every
    sighting attached to a cluster so far (including the newest one).

    Base confidence is the average of each sighting's own LLM-extraction
    confidence (how sure the extraction step was this is a genuine
    rumor). Distinct corroborating sources add a boost on top - a tier1
    source (e.g. a trusted journalist outlet) counts for more per source
    than a tier2 aggregator, so a tier1+tier2 mix ends up more credible
    than several tier2s alone.
    """
    if not sightings:
        return 0.0

    avg_llm_confidence = sum(s[2] for s in sightings) / len(sightings)

    tier1_sources = {s[0] for s in sightings if s[1] == "tier1"}
    tier2_sources = {s[0] for s in sightings if s[1] == "tier2"}

    boost = min(len(tier1_sources) * TIER1_SOURCE_WEIGHT, MAX_TIER1_BOOST)
    boost += min(len(tier2_sources) * TIER2_SOURCE_WEIGHT, MAX_TIER2_BOOST)

    return min(1.0, avg_llm_confidence + boost)
