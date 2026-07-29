"""Keyword-based relevance pre-filter.

Cheap first pass to drop articles that obviously aren't transfer rumors
(match reports, previews, off-pitch stories) before they reach the more
expensive LLM extraction step. Tuned for high recall over precision: when
a headline is ambiguous, keep it and let extraction make the final call.
"""

import re

from ingestion.rss_fetcher import RawArticle
from ingestion.rss_fetcher import fetch_all_sources

# Word/phrase keywords - each entry is matched with \b on both ends, so
# spaces inside a phrase (e.g. "in talks") are fine but no partial-word
# matches (e.g. "sign" won't match "signal").
TRANSFER_KEYWORDS = [
    r"transfer",
    r"sign(?:s|ed|ing)?",
    r"loan",
    r"medical",
    r"here we go",
    r"personal terms",
    r"agree(?:s|d)? (?:a )?(?:fee|deal|terms)",
    r"fee",
    r"million",
    r"bid",
    r"offer",
    r"rumou?r",
    r"link(?:ed|s)?",
    r"target(?:ed|ing)?",
    r"swoop",
    r"deal",
    r"release clause",
    r"buy-?back",
    r"buyout",
    r"in talks",
    r"interested in",
    r"want(?:s|ed)? to sign",
    r"contract extension",
    r"new contract",
    r"unveiled",
    r"confirmed",
]

_WORD_PATTERN = re.compile(r"\b(?:" + "|".join(TRANSFER_KEYWORDS) + r")\b", re.IGNORECASE)
# Currency amounts (£25m, €40 million, $10m) - symbol isn't a word char so
# it can't be wrapped in \b like the keywords above.
_MONEY_PATTERN = re.compile(r"[£€$]\s?\d")


def is_transfer_related(article: RawArticle) -> bool:
    text = f"{article.title} {article.summary}"
    return bool(_WORD_PATTERN.search(text) or _MONEY_PATTERN.search(text))


def filter_transfer_related(articles: list[RawArticle]) -> list[RawArticle]:
    return [a for a in articles if is_transfer_related(a)]


if __name__ == "__main__":
    all_articles = fetch_all_sources()
    kept = filter_transfer_related(all_articles)
    dropped = len(all_articles) - len(kept)

    print(f"\n{len(all_articles)} fetched, {len(kept)} kept, {dropped} dropped\n")
    for a in kept[:10]:
        print(f"[{a.source_tier}] {a.source_name}: {a.title}")
