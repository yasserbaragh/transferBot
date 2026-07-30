"""Runs the fetch -> dedup -> keyword filter -> LLM extraction ->
embedding/clustering -> storage pipeline on a recurring interval.

v1: no report generation/delivery yet (CLAUDE.md steps 6-7) - this just
proves the pipeline can run unattended, only spending LLM calls on
articles it hasn't seen before.
"""

from apscheduler.schedulers.blocking import BlockingScheduler

from ingestion.llm_filter import extract_rumors
from ingestion.relevance_filter import filter_transfer_related
from ingestion.rss_fetcher import fetch_all_sources
from storage.rumor_store import save_rumors
from storage.seen_store import filter_new, mark_seen

RUN_INTERVAL_MINUTES = 30


def run_pipeline() -> None:
    all_articles = fetch_all_sources()
    new_articles = filter_new(all_articles)
    mark_seen(new_articles)

    print(f"\n{len(all_articles)} fetched, {len(new_articles)} new since last run")

    if not new_articles:
        print("nothing new - skipping LLM step\n")
        return

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


if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(run_pipeline, "interval", minutes=RUN_INTERVAL_MINUTES)

    print(f"Running once now, then every {RUN_INTERVAL_MINUTES} minutes. Ctrl+C to stop.")
    run_pipeline()

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")
