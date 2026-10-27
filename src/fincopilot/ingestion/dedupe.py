from __future__ import annotations

import hashlib

from fincopilot.ingestion.base import ParsedTransaction
from fincopilot.ingestion.normalise import normalise_descriptor


def dedupe_hash(txn: ParsedTransaction) -> str:
    """Stable hash used to make re-imports of the same statement idempotent.

    Uses the *normalised* descriptor so cosmetic differences between exports (extra spaces,
    different casing) don't produce duplicates.
    """
    key = "|".join(
        [
            txn.date.isoformat(),
            f"{txn.amount:.2f}",
            txn.direction,
            normalise_descriptor(txn.raw_descriptor),
        ]
    )
    return hashlib.sha256(key.encode()).hexdigest()


def split_new_and_duplicate(
    parsed: list[ParsedTransaction], existing_hashes: set[str]
) -> tuple[list[tuple[ParsedTransaction, str]], int]:
    """Return (new rows with their hash, duplicate count). Also de-dupes within the batch."""
    seen = set(existing_hashes)
    new: list[tuple[ParsedTransaction, str]] = []
    dup = 0
    for t in parsed:
        h = dedupe_hash(t)
        if h in seen:
            dup += 1
            continue
        seen.add(h)
        new.append((t, h))
    return new, dup
