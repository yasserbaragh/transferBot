"""LLM-based transfer-rumor extraction and filter (Gemini).

Takes the articles that already passed the keyword pre-filter
(relevance_filter.py) and asks Gemini to (a) confirm each one is
genuinely a transfer rumor and (b) pull structured fields out of the
ones that are.

Cost choice: articles are batched into a single request (BATCH_SIZE per
call) instead of one call per article, and the model defaults to
Gemini's cheapest "flash-lite" tier. Request count against a free/low
tier's per-minute and per-day quota is the binding cost constraint here,
not token volume - these articles are headline + summary sized, so
batching several into one prompt is far cheaper than paying the fixed
per-request overhead N times.
"""

import json
import os
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from ingestion.relevance_filter import filter_transfer_related
from ingestion.rss_fetcher import RawArticle, fetch_all_sources

# Confirm this against your Gemini account's current model/pricing list
# before relying on it - model names and tiers change over time and this
# is just the cheapest tier known at the time this was written. Override
# via the GEMINI_MODEL env var without touching code.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

# Articles per request. Keeps each prompt small while cutting the number
# of calls (and therefore quota usage) roughly BATCH_SIZE-fold vs one
# call per article.
BATCH_SIZE = 20

_RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "index": {"type": "integer"},
            "is_transfer_rumor": {"type": "boolean"},
            "player": {"type": "string", "nullable": True},
            "clubs": {"type": "array", "items": {"type": "string"}},
            "fee": {"type": "string", "nullable": True},
            "wage": {"type": "string", "nullable": True},
            "confidence": {"type": "number"},
        },
        "required": ["index", "is_transfer_rumor", "clubs", "confidence"],
    },
}

_PROMPT_TEMPLATE = """You are filtering football news for genuine transfer rumors/news.

For each numbered article below, decide if it is actually about a
transfer, loan, signing, contract extension, or transfer bid/rumor -
not a match report, injury update, off-pitch story, or unrelated club
news.

Return one object per article with:
- index: the article's number below
- is_transfer_rumor: true/false
- player: the main player's name if mentioned, else null
- clubs: list of club names involved (empty list if none)
- fee: reported transfer fee if mentioned, as written in the text, else null
- wage: reported wage/salary if mentioned, else null
- confidence: 0.0-1.0, how confident you are this is a genuine transfer rumor

Articles:
{articles}
"""


@dataclass
class ExtractedRumor:
    source_name: str
    source_tier: str
    link: str
    title: str
    is_transfer_rumor: bool
    player: str | None = None
    clubs: list[str] = field(default_factory=list)
    fee: str | None = None
    wage: str | None = None
    confidence: float = 0.0


def _format_articles(articles: list[RawArticle]) -> str:
    return "\n\n".join(f"{i}. {a.title}\n{a.summary}" for i, a in enumerate(articles))


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    return genai.Client(api_key=api_key)


def _extract_batch(client: genai.Client, batch: list[RawArticle]) -> list[ExtractedRumor]:
    prompt = _PROMPT_TEMPLATE.format(articles=_format_articles(batch))

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
        ),
    )

    try:
        results = json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as exc:
        print(f"[error] batch of {len(batch)}: unparseable model response: {exc}")
        return []

    extracted = []
    for item in results:
        idx = item.get("index")
        if idx is None or not (0 <= idx < len(batch)):
            continue
        article = batch[idx]
        extracted.append(
            ExtractedRumor(
                source_name=article.source_name,
                source_tier=article.source_tier,
                link=article.link,
                title=article.title,
                is_transfer_rumor=bool(item.get("is_transfer_rumor", False)),
                player=item.get("player"),
                clubs=item.get("clubs") or [],
                fee=item.get("fee"),
                wage=item.get("wage"),
                confidence=float(item.get("confidence", 0.0)),
            )
        )
    return extracted


def extract_rumors(articles: list[RawArticle]) -> list[ExtractedRumor]:
    client = _client()
    all_extracted = []
    for start in range(0, len(articles), BATCH_SIZE):
        batch = articles[start : start + BATCH_SIZE]
        try:
            all_extracted.extend(_extract_batch(client, batch))
        except Exception as exc:
            print(f"[error] batch starting at index {start}: {exc}")
    return [e for e in all_extracted if e.is_transfer_rumor]


if __name__ == "__main__":
    candidates = filter_transfer_related(fetch_all_sources())
    print(f"{len(candidates)} articles passed the keyword filter, sending to Gemini...\n")

    rumors = extract_rumors(candidates)
    print(f"{len(rumors)} confirmed transfer rumors\n")
    for r in rumors[:10]:
        print(
            f"[{r.source_tier}] {r.player} -> {r.clubs} "
            f"(fee={r.fee}, conf={r.confidence:.2f}) - {r.title}"
        )
