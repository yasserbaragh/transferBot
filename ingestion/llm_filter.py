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
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google import genai
from google.genai import types

from ingestion.relevance_filter import filter_transfer_related
from ingestion.rss_fetcher import RawArticle, fetch_all_sources

# Loads GEMINI_API_KEY (and anything else in .env) into the process
# environment. Called at import time so it's in place before _client()
# reads os.environ, regardless of which script imports this module.
load_dotenv()

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
            "players": {"type": "array", "items": {"type": "string"}},
            "current_club": {"type": "string", "nullable": True},
            "linked_clubs": {"type": "array", "items": {"type": "string"}},
            "fee": {"type": "string", "nullable": True},
            "wage": {"type": "string", "nullable": True},
            "confidence": {"type": "number"},
        },
        "required": ["index", "is_transfer_rumor", "players", "linked_clubs", "confidence"],
    },
}

_PROMPT_TEMPLATE = """You are filtering football news for genuine transfer rumors/news.

For each numbered article below, decide if it is actually about a
transfer, loan, signing, contract extension, or transfer bid/rumor for
one or more NAMED players - not a match report, injury update,
off-pitch story, unrelated club news, or a general roundup/preview that
never names a specific player. If the article doesn't name a specific
player, set is_transfer_rumor to false even if it's otherwise
transfer-related (e.g. "5 players Arsenal could sign" with no names
given yet is not a rumor for our purposes).

Articles may be written in English, Spanish, German, Italian, or French.
Read each one in its original language, but always respond in English:
- players / current_club / linked_clubs: use the common English form of
  player and club names (e.g. "Bayern München" -> "Bayern Munich", "FC
  Bayern" -> "Bayern Munich"), not a transliteration or the source
  language's own spelling, so the same person/club is named consistently
  no matter which source reported it
- fee / wage: normalize to a plain figure with its currency symbol
  (e.g. German "80,5 Mio. €" -> "€80.5m"), not the source's original
  number formatting

Return one object per article with:
- index: the article's number below
- is_transfer_rumor: true/false
- players: every named player the rumor is about, as a list (almost
  always one name; more than one only for a genuine joint/multi-player
  deal covered in the same article)
- current_club: the player's club at the time of the article, if
  stated, else null
- linked_clubs: clubs reported as interested in, bidding for, or the
  likely/agreed destination for the player - never include
  current_club in this list
- fee: reported transfer fee if mentioned, as written in the text, else null
- wage: reported wage/salary if mentioned, else null
- confidence: 0.0-1.0, how confident you are this is a genuine transfer
  rumor about the named player(s)

Articles:
{articles}
"""

# Some source feeds contain mis-decoded currency symbols (mojibake), so a
# fee sometimes arrives as e.g. "#80m" or "•100m" instead of "£80m". We
# can't reliably guess which currency symbol was intended, so rather than
# display a wrong glyph we just strip a garbled leading symbol and keep the
# digits - "#80m" -> "80m". Symbols we trust (£/€/$) are left alone.
_GARBLED_FEE_PREFIX_RE = re.compile(r"^[^\w£€$]+(?=\d)")


def _sanitize_fee(fee: str | None) -> str | None:
    if not fee:
        return fee
    cleaned = _GARBLED_FEE_PREFIX_RE.sub("", fee.strip())
    return cleaned or None


@dataclass
class ExtractedRumor:
    source_name: str
    source_tier: str
    link: str
    title: str
    is_transfer_rumor: bool
    players: list[str] = field(default_factory=list)
    current_club: str | None = None
    linked_clubs: list[str] = field(default_factory=list)
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
                players=item.get("players") or [],
                current_club=item.get("current_club"),
                linked_clubs=item.get("linked_clubs") or [],
                fee=_sanitize_fee(item.get("fee")),
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
    # Belt-and-suspenders alongside the prompt instruction: a rumor with no
    # named player isn't actionable for this pipeline, so drop it here too
    # even if the model marked it is_transfer_rumor=true anyway.
    return [e for e in all_extracted if e.is_transfer_rumor and e.players]


if __name__ == "__main__":
    candidates = filter_transfer_related(fetch_all_sources())
    print(f"{len(candidates)} articles passed the keyword filter, sending to Gemini...\n")

    rumors = extract_rumors(candidates)
    print(f"{len(rumors)} confirmed transfer rumors\n")
    for r in rumors[:10]:
        print(
            f"[{r.source_tier}] {' & '.join(r.players)} - {r.current_club} -> {r.linked_clubs} "
            f"(fee={r.fee}, conf={r.confidence:.2f}) - {r.title}"
        )
