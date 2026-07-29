"""Pulls raw articles from the configured RSS sources.

This is step 1 of the pipeline only: fetch + normalize. No extraction,
embedding, or storage happens here yet.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import yaml
from dateutil import parser as dateutil_parser

SOURCES_PATH = Path(__file__).resolve().parent.parent / "config" / "sources.yaml"

ALLOWED_TIERS = {"tier1", "tier2"}


@dataclass
class RawArticle:
    source_name: str
    source_tier: str
    title: str
    link: str
    published: datetime | None
    summary: str


def load_sources(path: Path = SOURCES_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    sources = config["sources"]

    for source in sources:
        tier = source.get("tier")
        if tier not in ALLOWED_TIERS:
            raise ValueError(
                f"source {source.get('name')!r} has invalid tier {tier!r}, "
                f"expected one of {sorted(ALLOWED_TIERS)}"
            )

    return sources


def _parse_published(entry) -> datetime | None:
    raw = entry.get("published") or entry.get("updated")
    if not raw:
        return None
    try:
        dt = dateutil_parser.parse(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def fetch_source(source: dict) -> list[RawArticle]:
    parsed = feedparser.parse(source["url"])

    status = getattr(parsed, "status", None)
    bozo = getattr(parsed, "bozo", 0)
    bozo_exc = getattr(parsed, "bozo_exception", None)
    print(
        f"[fetch] {source['name']}: status={status} bozo={bozo} "
        f"entries={len(parsed.entries)}"
        + (f" bozo_exception={bozo_exc!r}" if bozo else "")
    )
    if not parsed.entries:
        print(f"[warn] {source['name']}: 0 entries returned, check URL/blocking/UA")

    articles = []
    for entry in parsed.entries:
        articles.append(
            RawArticle(
                source_name=source["name"],
                source_tier=source["tier"],
                title=entry.get("title", "").strip(),
                link=entry.get("link", "").strip(),
                published=_parse_published(entry),
                summary=entry.get("summary", "").strip(),
            )
        )
    return articles


def fetch_all_sources(path: Path = SOURCES_PATH) -> list[RawArticle]:
    sources = load_sources(path)
    all_articles = []
    for source in sources:
        if str(source["url"]).startswith("TODO"):
            print(f"[skip] {source['name']}: no URL configured yet")
            continue
        try:
            all_articles.extend(fetch_source(source))
        except Exception as exc:
            print(f"[error] {source['name']}: {exc}")
    return all_articles


if __name__ == "__main__":
    articles = fetch_all_sources()
    print(f"Fetched {len(articles)} articles total\n")
    for a in articles[:10]:
        print(f"[{a.source_tier}] {a.source_name}: {a.title} ({a.published})")
