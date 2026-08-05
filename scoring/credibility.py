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
SINGLE_SOURCE_FLOOR = 0.30
TIER1_SOURCE_WEIGHT = 0.25
TIER2_SOURCE_WEIGHT = 0.08
MAX_TIER1_BOOST = 0.55
MAX_TIER2_BOOST = 0.25

# A done deal is normally announced once, definitively - it doesn't
# naturally accumulate repeat coverage across our configured feeds the
# way a developing rumor does, so it shouldn't need the same
# multi-source corroboration to be treated as credible. One sighting
# tagged "confirmed" (club/official announcement language) raises the
# floor straight to the "Confident" bucket instead of "Weak".
CONFIRMED_FLOOR = 0.80

# Buckets for display - see confidence_label().
CONFIDENT_THRESHOLD = 0.75
FIFTY_FIFTY_THRESHOLD = 0.45


def compute_confidence(sightings: list[tuple[str, str, float, str]]) -> float:
    """sightings: (source_name, source_tier, llm_confidence, status) for
    every sighting attached to a cluster so far (including the newest
    one). status is "rumor" or "confirmed", as classified by the LLM
    extraction step.

    Confidence is driven by corroboration - how many *distinct* sources,
    and which tiers, are reporting the same rumor - not by the LLM's own
    self-rated confidence. That self-rating measures "is this text really
    about a transfer" (extraction confidence), not "is this rumor
    credible", and models tend to report it near 1.0 for almost anything.
    It's only used here as a ceiling: a very low extraction confidence
    caps how high corroboration alone can push the score, since a shaky
    extraction shouldn't read as a well-sourced rumor no matter how many
    sightings pile up.
    """
    if not sightings:
        return 0.0

    tier1_sources = {s[0] for s in sightings if s[1] == "tier1"}
    tier2_sources = {s[0] for s in sightings if s[1] == "tier2"}

    corroboration = SINGLE_SOURCE_FLOOR
    corroboration += min(len(tier1_sources) * TIER1_SOURCE_WEIGHT, MAX_TIER1_BOOST)
    corroboration += min(len(tier2_sources) * TIER2_SOURCE_WEIGHT, MAX_TIER2_BOOST)

    if any(s[3] == "confirmed" for s in sightings):
        corroboration = max(corroboration, CONFIRMED_FLOOR)

    avg_llm_confidence = sum(s[2] for s in sightings) / len(sightings)
    extraction_ceiling = avg_llm_confidence + 0.30

    return min(1.0, corroboration, extraction_ceiling)


def confidence_label(score: float) -> str:
    """Coarse, human-readable bucket for a confidence score - deliberately
    not a percentage, since the precision that implies isn't real."""
    if score >= CONFIDENT_THRESHOLD:
        return "Confident"
    if score >= FIFTY_FIFTY_THRESHOLD:
        return "50/50"
    return "Weak"
