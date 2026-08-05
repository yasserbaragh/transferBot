"""Runs the fetch -> dedup -> keyword filter -> LLM extraction ->
embedding/clustering -> storage -> report -> delivery pipeline on a
recurring interval (CLAUDE.md steps 1-7).
"""

from apscheduler.schedulers.blocking import BlockingScheduler

from delivery.discord_webhook import send_digest
from ingestion.llm_filter import extract_rumors
from ingestion.relevance_filter import filter_transfer_related
from ingestion.rss_fetcher import fetch_all_sources
from reporting.digest import build_digest
from storage.report_state import get_last_report_time, set_last_report_time
from storage.rumor_store import get_clusters_since, save_rumors
from storage.seen_store import filter_new, mark_seen

RUN_INTERVAL_MINUTES = 30


def run_pipeline() -> None:
    all_articles = fetch_all_sources()
    new_articles = filter_new(all_articles)
    mark_seen(new_articles)

    print(f"\n{len(all_articles)} fetched, {len(new_articles)} new since last run")

    if not new_articles:
        print("nothing new - skipping LLM step\n")
    else:
        candidates = filter_transfer_related(new_articles)
        print(f"{len(candidates)} passed the keyword filter, sending to Gemini...")

        rumors = extract_rumors(candidates)
        print(f"{len(rumors)} confirmed transfer rumors, embedding + matching...\n")

        results = save_rumors(rumors)
        for r, (cluster_id, is_new) in zip(rumors, results):
            label = "NEW cluster" if is_new else f"attached to cluster {cluster_id}"
            print(
                f"[{r.source_tier}] {r.player} -> {r.clubs} "
                f"(fee={r.fee}, conf={r.confidence:.2f}) - {label} - {r.title}"
            )

    _deliver_report()


def _deliver_report() -> None:
    # Reads from get_last_report_time() rather than "did this run find
    # anything new" so a failed send just gets retried next run instead
    # of silently dropping those clusters' updates.
    since = get_last_report_time()
    clusters = get_clusters_since(since)
    if not clusters:
        print("no cluster updates since last report - skipping delivery\n")
        return

    digest = build_digest(clusters)
    if send_digest(digest):
        set_last_report_time()
        print(f"report sent - {len(clusters)} cluster(s) - last_report_at updated\n")
    else:
        print("report delivery FAILED - last_report_at not updated, will retry next run\n")


if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(run_pipeline, "interval", minutes=RUN_INTERVAL_MINUTES)

    print(f"Running once now, then every {RUN_INTERVAL_MINUTES} minutes. Ctrl+C to stop.")
    run_pipeline()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")
