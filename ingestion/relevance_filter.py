"""Keyword-based relevance pre-filter.

Cheap first pass to drop articles that obviously aren't transfer rumors
(match reports, previews, off-pitch stories) before they reach the more
expensive LLM extraction step. Tuned for high recall over precision: when
a headline is ambiguous, keep it and let extraction make the final call.

One keyword list per source language - a regex built from English words
can't do relevance filtering on a Spanish/German/Italian/French article,
so each language gets its own list of the same kind of transfer-market
vocabulary (signing/loan/fee/bid/rumor equivalents) rather than sharing
the English one.
"""

import re

from ingestion.rss_fetcher import RawArticle
from ingestion.rss_fetcher import fetch_all_sources

# Word/phrase keywords - each entry is matched with \b on both ends, so
# spaces inside a phrase (e.g. "in talks") are fine but no partial-word
# matches (e.g. "sign" won't match "signal").
TRANSFER_KEYWORDS_EN = [
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

TRANSFER_KEYWORDS_ES = [
    r"fich\w*",  # fichaje(s), ficha, fichó, fichar - transfer/signing
    r"traspas\w*",  # traspaso(s), traspasar
    r"cesi[oó]n(?:es)?",
    r"ced\w*",  # cedido, cede, ceder - loaned out
    r"pr[eé]stamo(?:s)?",
    r"ofert\w*",  # oferta(s), ofertar
    r"puj\w*",  # puja(s), pujar - bid
    r"acuerdo(?:s)?",
    r"interesad[oa]s? en",
    r"negocia\w*",  # negocia, negociación, negociando
    r"cl[aá]usula de rescisi[oó]n",
    r"renov\w*",  # renovación, renovar, renovado
    r"renuev\w*",  # renueva, renuevan - irregular stem of renovar
    r"agente libre",
    r"mill[oó]n(?:es)?",
    r"se une a",
    r"firm\w*",  # firma, firmó, firmar - sign
    r"rumor(?:es)? de fichaje",
]

TRANSFER_KEYWORDS_DE = [
    r"wechsel\w*",  # Wechsel, wechselt, Wechselgerüchte (compound)
    r"transfer\w*",  # Transfer, Transfersumme, Transfergerüchte (compound)
    r"leih\w*",  # Leihe, leihen
    r"geliehen",
    r"verliehen",
    r"abl[oö]se\w*",  # Ablöse, Ablösesumme, ablösefrei (compound)
    r"verpflicht\w*",  # verpflichten, verpflichtet, Verpflichtung(en)
    r"vertragsverl[aä]ngerung",
    r"neuer vertrag",
    r"gebot",
    r"angebot",
    r"interesse an",
    r"million\w*",  # Million, Millionen
    r"unterschreib\w*",  # unterschreibt, unterschreiben
]

TRANSFER_KEYWORDS_IT = [
    r"trasferi\w*",  # trasferimento(i), trasferisce, si trasferisce
    r"calciomercato",
    r"prestito(?:i)?",  # kept literal, not stemmed - "prest*" would
    # also match "prestazione" (performance), a very common unrelated
    # football word
    r"cifra(?:e)?",
    r"clausola rescissoria",
    r"rinnov\w*",  # rinnovo, rinnova, rinnovare, rinnovato
    r"trattativ\w*",  # trattativa, trattative
    r"offert\w*",  # offerta, offerte
    r"interessat[oa]e? a",
    r"firma\w*",  # firma, firmare, firmato
    r"milion\w*",  # milione, milioni
    r"svincolat\w*",
    r"passa all?a?",  # passa al/alla <club>
]

TRANSFER_KEYWORDS_FR = [
    r"transfert\w*",
    r"mercato",
    r"pr[eê]t[eé]?s?",  # prêt(s), prêté(e) - also matches the adjective
    # "prêt(e)" (ready); accepted, same recall-over-precision tradeoff
    # as the rest of this list
    r"indemnit[eé]s? de transfert",
    r"sign\w*",  # signe, signer, signature, signé
    r"prolongation\w* de contrat",
    r"nouveau contrat",
    r"offre(?:s)?",
    r"int[eé]ress[eé]s? par",
    r"n[eé]gociation(?:s)?",
    r"million(?:s)?",
    r"agent libre",
    r"s'engage (?:avec|pour)",
]

# Currency amounts (£25m, €40 million, $10m) - symbol isn't a word char so
# it can't be wrapped in \b like the keywords above. Shared across every
# language since digits + a currency symbol read the same everywhere.
_MONEY_PATTERN = re.compile(r"[£€$]\s?\d")


def _compile(keywords: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(keywords) + r")\b", re.IGNORECASE)


_WORD_PATTERNS = {
    "en": _compile(TRANSFER_KEYWORDS_EN),
    "es": _compile(TRANSFER_KEYWORDS_ES),
    "de": _compile(TRANSFER_KEYWORDS_DE),
    "it": _compile(TRANSFER_KEYWORDS_IT),
    "fr": _compile(TRANSFER_KEYWORDS_FR),
}


def is_transfer_related(article: RawArticle) -> bool:
    text = f"{article.title} {article.summary}"
    pattern = _WORD_PATTERNS.get(article.language)
    if pattern is None:
        # No keyword list for this language yet - matching nothing would
        # mean "we can't filter this," not "it's irrelevant," so pass it
        # through to the LLM instead of silently dropping it.
        return True
    return bool(pattern.search(text) or _MONEY_PATTERN.search(text))


def filter_transfer_related(articles: list[RawArticle]) -> list[RawArticle]:
    return [a for a in articles if is_transfer_related(a)]


if __name__ == "__main__":
    all_articles = fetch_all_sources()
    kept = filter_transfer_related(all_articles)
    dropped = len(all_articles) - len(kept)

    print(f"\n{len(all_articles)} fetched, {len(kept)} kept, {dropped} dropped\n")
    for a in kept[:10]:
        print(f"[{a.source_tier}] {a.source_name}: {a.title}")
