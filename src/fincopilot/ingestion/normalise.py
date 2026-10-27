"""Descriptor normalisation shared by ingestion and the ML feature pipeline.

Keep this deterministic and dependency-free: the exact same function is used at training time
(in the Kaggle notebooks) and at inference time inside the app.
"""

from __future__ import annotations

import re
import unicodedata

_NOISE_TOKENS = re.compile(
    r"\b(POS|PURCHASE|PAYMENT|CARD|DEBIT|CREDIT|TXN|TRANSACTION|REF|AUTH|VISA|MASTERCARD|"
    r"MC|ONLINE|ECOM|E-COM|INTL|TERMINAL|TRX|NEFT|IMPS|UPI|ACH|SEPA|BKASH|NAGAD)\b",
    re.IGNORECASE,
)
_LONG_DIGITS = re.compile(r"\d{5,}")
_SHORT_DIGITS = re.compile(r"\b\d{1,4}\b")
_MASKED = re.compile(r"[xX*#]{3,}\d*")
_DATE_LIKE = re.compile(r"\b\d{1,2}[/.-]\d{1,2}([/.-]\d{2,4})?\b")
_NON_ALNUM = re.compile(r"[^a-z0-9<> ]+")
_SPACES = re.compile(r"\s+")

_MERCHANT_ALIASES: dict[str, str] = {
    "uber bv": "uber",
    "uber trip": "uber",
    "uber eats": "uber eats",
    "amzn": "amazon",
    "amazon mktp": "amazon",
    "amazon prime": "amazon prime",
    "netflix com": "netflix",
    "spotify ab": "spotify",
    "google play": "google play",
    "apple com bill": "apple",
    "mcd": "mcdonalds",
}


def normalise_descriptor(raw: str) -> str:
    """Lower-case, strip reference numbers/dates/noise, collapse whitespace.

    >>> normalise_descriptor("POS 4521 DHK*UBERBD 22/10 REF 889123445")
    'dhk uberbd'
    """
    s = unicodedata.normalize("NFKC", raw)
    s = _DATE_LIKE.sub(" ", s)
    s = _MASKED.sub(" <masked> ", s)
    s = _LONG_DIGITS.sub(" <num> ", s)
    s = _NOISE_TOKENS.sub(" ", s)
    s = _SHORT_DIGITS.sub(" ", s)
    s = s.lower()
    s = _NON_ALNUM.sub(" ", s)
    s = _SPACES.sub(" ", s).strip()
    return s


def canonical_merchant(normalised: str) -> str:
    """Best-effort merchant key: drop placeholder tokens, take the first 3 words, apply aliases."""
    words = [w for w in normalised.split() if not w.startswith("<")]
    key = " ".join(words[:3])
    for alias, canon in _MERCHANT_ALIASES.items():
        if key.startswith(alias):
            return canon
    return key or "unknown"
