"""Turns new/updated rumor clusters into a readable digest
(pipeline step 6) - template-based, no LLM call.

build_digest(clusters, since) -> str is the contract: takes the clusters
a report should cover plus the timestamp the report is covering since,
returns the digest text. Swapping this for an LLM-written version later
(e.g. one call turning the same cluster data into more natural prose) is
a drop-in replacement - callers don't need to change, only this
function's internals would.
"""

import os

from scoring.credibility import confidence_label
from storage.rumor_store import ClusterDigest
from storage.report_state import get_last_report_time
from storage.rumor_store import get_clusters_since


# Clusters below this label are excluded from the digest entirely - a
# single tier2 source with nothing corroborating it is noise at report
# time, even though it's still tracked in the DB in case it gets
# corroborated later. Override via env var: "Weak" to include everything.
_LABEL_RANK = {"Weak": 0, "50/50": 1, "Confident": 2}
MIN_DIGEST_LABEL = os.environ.get("DIGEST_MIN_CONFIDENCE", "50/50")


def _passes_floor(cluster: ClusterDigest) -> bool:
    return _LABEL_RANK[confidence_label(cluster.confidence)] >= _LABEL_RANK[MIN_DIGEST_LABEL]


def _format_players(cluster: ClusterDigest) -> str:
    return " & ".join(cluster.players) if cluster.players else "Unknown player"


def _format_clubs(cluster: ClusterDigest) -> str:
    if cluster.current_club and cluster.linked_clubs:
        return f"{cluster.current_club} -> {', '.join(cluster.linked_clubs)}"
    if cluster.linked_clubs:
        return f"linked with {', '.join(cluster.linked_clubs)}"
    if cluster.current_club:
        return cluster.current_club
    return "club(s) unclear"


def _format_sources(cluster: ClusterDigest) -> str:
    # Distinct outlets, not one line per sighting - the same outlet
    # posting/re-crawling the same rumor three times shouldn't print three
    # times, and the tier label isn't useful to a reader here (it already
    # fed into the confidence bucket above).
    seen: list[str] = []
    for s in cluster.sightings:
        if s.source_name not in seen:
            seen.append(s.source_name)
    return ", ".join(seen)


def _format_fees(cluster: ClusterDigest) -> str | None:
    fees = []
    for s in cluster.sightings:
        if s.fee and s.fee not in fees:
            fees.append(s.fee)
    return ", ".join(fees) if fees else None


def _latest_link(cluster: ClusterDigest) -> str | None:
    return cluster.sightings[-1].link if cluster.sightings else None


def _format_cluster(cluster: ClusterDigest) -> str:
    lines = [
        f"**{_format_players(cluster)}** - {_format_clubs(cluster)}",
        f"{confidence_label(cluster.confidence)} | Sources: {_format_sources(cluster)}",
    ]
    fees = _format_fees(cluster)
    if fees:
        lines.append(f"Reported fee: {fees}")
    link = _latest_link(cluster)
    if link:
        lines.append(link)
    return "\n".join(lines)


def build_digest(clusters: list[ClusterDigest], since: str) -> str:
    relevant = [c for c in clusters if _passes_floor(c)]
    if not relevant:
        return "No new or updated transfer rumors since the last report."

    # A cluster first seen at/after `since` wasn't in any earlier report -
    # it's brand new this cycle. Anything older got a new sighting since
    # the last report, i.e. it's an update to something already reported.
    new_clusters = [c for c in relevant if c.first_seen >= since]
    updated_clusters = [c for c in relevant if c.first_seen < since]

    sections = [f"**Transfer rumor digest - {len(relevant)} update(s)**"]
    if new_clusters:
        sections.append("__New rumors__\n\n" + "\n\n".join(_format_cluster(c) for c in new_clusters))
    if updated_clusters:
        sections.append("__Updated rumors__\n\n" + "\n\n".join(_format_cluster(c) for c in updated_clusters))
    return "\n\n".join(sections)


if __name__ == "__main__":
    since = get_last_report_time()
    clusters = get_clusters_since(since)
    print(f"{len(clusters)} cluster(s) updated since {since}\n")
    print(build_digest(clusters, since))
