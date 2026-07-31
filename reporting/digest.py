"""Turns new/updated rumor clusters into a readable digest
(pipeline step 6) - template-based, no LLM call.

build_digest(clusters) -> str is the contract: takes the clusters a
report should cover, returns the digest text. Swapping this for an
LLM-written version later (e.g. one call turning the same cluster data
into more natural prose) is a drop-in replacement - callers don't need
to change, only this function's internals would.
"""

from storage.rumor_store import ClusterDigest


def _format_sources(cluster: ClusterDigest) -> str:
    return ", ".join(f"{s.source_name} ({s.source_tier})" for s in cluster.sightings)


def _format_fees(cluster: ClusterDigest) -> str | None:
    fees = []
    for s in cluster.sightings:
        if s.fee and s.fee not in fees:
            fees.append(s.fee)
    return ", ".join(fees) if fees else None


def _format_cluster(cluster: ClusterDigest) -> str:
    player = cluster.player or "Unknown player"
    clubs = " -> ".join(cluster.clubs) if cluster.clubs else "unknown club(s)"

    lines = [
        f"**{player}** - {clubs}",
        f"Confidence: {cluster.confidence:.0%} | Sources: {_format_sources(cluster)}",
    ]
    fees = _format_fees(cluster)
    if fees:
        lines.append(f"Reported fee: {fees}")
    return "\n".join(lines)


def build_digest(clusters: list[ClusterDigest]) -> str:
    if not clusters:
        return "No new or updated transfer rumors since the last report."

    header = f"**Transfer rumor digest - {len(clusters)} update(s)**"
    body = "\n\n".join(_format_cluster(c) for c in clusters)
    return f"{header}\n\n{body}"


if __name__ == "__main__":
    from storage.report_state import get_last_report_time
    from storage.rumor_store import get_clusters_since

    since = get_last_report_time()
    clusters = get_clusters_since(since)
    print(f"{len(clusters)} cluster(s) updated since {since}\n")
    print(build_digest(clusters))
