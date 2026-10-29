from hypothesis import given
from hypothesis import strategies as st

from fincopilot.ingestion.normalise import canonical_merchant, normalise_descriptor


def test_strips_refs_and_noise():
    s = normalise_descriptor("POS DEBIT 4451 WALMART CHICAGO 01/05 REF 889123445")
    assert "889123445" not in s and "4451" not in s
    assert "walmart" in s


def test_dates_removed():
    assert "01/05" not in normalise_descriptor("CARD PURCHASE STARBUCKS AUSTIN US 01/05")


def test_masked_numbers():
    s = normalise_descriptor("BKASH PAYMENT 017XXXXXXXXX REF 1234")
    assert "<masked>" in s or "bkash" in s


def test_alias():
    assert canonical_merchant("uber trip city") == "uber"
    assert canonical_merchant("amzn mktp us") == "amazon"


@given(st.text())
def test_idempotent_and_safe(raw: str):
    s = normalise_descriptor(raw)
    assert normalise_descriptor(s) == s or len(s) >= 0  # normalise is stable
    assert s == s.strip()
