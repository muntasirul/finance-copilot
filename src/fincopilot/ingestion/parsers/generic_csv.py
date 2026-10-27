"""Heuristic CSV parser that maps arbitrary bank column names onto the canonical schema."""

from __future__ import annotations

import csv
import re
from datetime import date, datetime
from pathlib import Path

from fincopilot.ingestion.base import ParsedTransaction, ParseError, StatementParser

DATE_COLS = ("date", "transaction date", "txn date", "posting date", "value date", "booked")
DESC_COLS = ("description", "details", "narration", "particulars", "memo", "payee", "merchant")
AMOUNT_COLS = ("amount", "transaction amount", "value")
DEBIT_COLS = ("debit", "withdrawal", "money out", "paid out", "dr")
CREDIT_COLS = ("credit", "deposit", "money in", "paid in", "cr")
BALANCE_COLS = ("balance", "running balance", "closing balance")

DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d %b %Y", "%d-%b-%Y", "%Y/%m/%d")
_NUM = re.compile(r"[^0-9.\-]")


def _find(header: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {h.lower().strip(): h for h in header}
    for c in candidates:
        if c in lowered:
            return lowered[c]
    for h_low, h in lowered.items():
        if any(c in h_low for c in candidates):
            return h
    return None


def parse_date(s: str) -> date:
    s = s.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ParseError(f"unrecognised date: {s!r}")


def parse_number(s: str | None) -> float | None:
    if s is None or not s.strip():
        return None
    neg = "(" in s and ")" in s
    cleaned = _NUM.sub("", s.replace(",", ""))
    if cleaned in ("", "-", "."):
        return None
    v = float(cleaned)
    return -abs(v) if neg else v


class GenericCSVParser(StatementParser):
    name = "generic_csv"

    def _header(self, path: Path) -> list[str] | None:
        if path.suffix.lower() != ".csv":
            return None
        with path.open(newline="", encoding="utf-8-sig") as f:
            try:
                return next(csv.reader(f))
            except StopIteration:
                return None

    def detect(self, path: Path) -> float:
        header = self._header(path)
        if not header:
            return 0.0
        has_date = _find(header, DATE_COLS) is not None
        has_desc = _find(header, DESC_COLS) is not None
        has_amt = _find(header, AMOUNT_COLS) is not None or (
            _find(header, DEBIT_COLS) is not None and _find(header, CREDIT_COLS) is not None
        )
        return 0.6 if (has_date and has_desc and has_amt) else 0.0

    def parse(self, path: Path) -> list[ParsedTransaction]:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            d_col, desc_col = _find(header, DATE_COLS), _find(header, DESC_COLS)
            amt_col = _find(header, AMOUNT_COLS)
            dr_col, cr_col = _find(header, DEBIT_COLS), _find(header, CREDIT_COLS)
            bal_col = _find(header, BALANCE_COLS)
            if not (d_col and desc_col):
                raise ParseError("missing date/description columns")

            out: list[ParsedTransaction] = []
            for row in reader:
                if not row.get(desc_col):
                    continue
                if amt_col and parse_number(row.get(amt_col)) is not None:
                    signed = parse_number(row[amt_col]) or 0.0
                elif dr_col or cr_col:
                    dr = parse_number(row.get(dr_col or "")) or 0.0
                    cr = parse_number(row.get(cr_col or "")) or 0.0
                    signed = cr - abs(dr)
                else:
                    raise ParseError("no amount columns")
                if signed == 0:
                    continue
                out.append(
                    ParsedTransaction.from_signed(
                        date=parse_date(row[d_col]),
                        raw_descriptor=row[desc_col],
                        signed_amount=signed,
                        balance_after=parse_number(row.get(bal_col)) if bal_col else None,
                    )
                )
            return out
