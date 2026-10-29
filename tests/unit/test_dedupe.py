from datetime import date

from fincopilot.ingestion.base import ParsedTransaction
from fincopilot.ingestion.dedupe import dedupe_hash, split_new_and_duplicate


def _txn(desc: str = "POS WALMART", amt: float = 10.0, d: date = date(2026, 1, 5)):
    return ParsedTransaction.from_signed(date=d, raw_descriptor=desc, signed_amount=-amt)


def test_same_txn_same_hash():
    assert dedupe_hash(_txn()) == dedupe_hash(_txn())


def test_casing_and_spacing_ignored():
    assert dedupe_hash(_txn("pos   walmart")) == dedupe_hash(_txn("POS WALMART"))


def test_different_amount_different_hash():
    assert dedupe_hash(_txn(amt=10.0)) != dedupe_hash(_txn(amt=11.0))


def test_split_new_and_duplicate():
    rows = [_txn(), _txn("CARD STARBUCKS", 5.0), _txn()]  # third is dup of first
    new, dup = split_new_and_duplicate(rows, set())
    assert len(new) == 2 and dup == 1

    # re-import: everything already exists
    existing = {h for _, h in new}
    new2, dup2 = split_new_and_duplicate(rows, existing)
    assert new2 == [] and dup2 == 3
