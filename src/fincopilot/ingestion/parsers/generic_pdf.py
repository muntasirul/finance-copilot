"""Generic PDF statement parser built on pdfplumber table extraction.

Strategy: extract every table on every page, treat the first row whose cells look like a
statement header as the schema, and reuse the CSV column heuristics. Scanned PDFs (no text layer)
are flagged with ``needs_ocr`` so the UI can offer an OCR fallback.
"""

from __future__ import annotations

from pathlib import Path

from fincopilot.ingestion.base import ParsedTransaction, ParseError, StatementParser
from fincopilot.ingestion.parsers.generic_csv import (
    AMOUNT_COLS,
    BALANCE_COLS,
    CREDIT_COLS,
    DATE_COLS,
    DEBIT_COLS,
    DESC_COLS,
    _find,
    parse_date,
    parse_number,
)


class GenericPDFParser(StatementParser):
    name = "generic_pdf"
    needs_ocr: bool = False

    def _tables(self, path: Path) -> list[list[list[str]]]:
        import pdfplumber

        tables: list[list[list[str]]] = []
        with pdfplumber.open(path) as pdf:
            text_chars = 0
            for page in pdf.pages:
                text_chars += len(page.extract_text() or "")
                for t in page.extract_tables():
                    tables.append([[(c or "").strip() for c in row] for row in t if row])
            self.needs_ocr = text_chars < 50
        return tables

    def detect(self, path: Path) -> float:
        if path.suffix.lower() != ".pdf":
            return 0.0
        try:
            tables = self._tables(path)
        except Exception:  # noqa: BLE001
            return 0.0
        for t in tables:
            if t and _find(t[0], DATE_COLS) and _find(t[0], DESC_COLS):
                return 0.55
        return 0.0

    def parse(self, path: Path) -> list[ParsedTransaction]:
        out: list[ParsedTransaction] = []
        header: list[str] | None = None
        for table in self._tables(path):
            for row in table:
                if _find(row, DATE_COLS) and _find(row, DESC_COLS):
                    header = row
                    continue
                if header is None or len(row) != len(header):
                    continue
                rec = dict(zip(header, row, strict=True))
                d_col, desc_col = _find(header, DATE_COLS), _find(header, DESC_COLS)
                if not (d_col and desc_col) or not rec.get(desc_col):
                    continue
                try:
                    dt = parse_date(rec[d_col])
                except ParseError:
                    continue  # continuation line / footer
                amt_col = _find(header, AMOUNT_COLS)
                dr_col, cr_col = _find(header, DEBIT_COLS), _find(header, CREDIT_COLS)
                bal_col = _find(header, BALANCE_COLS)
                if amt_col and parse_number(rec.get(amt_col)) is not None:
                    signed = parse_number(rec[amt_col]) or 0.0
                else:
                    dr = parse_number(rec.get(dr_col or "")) or 0.0
                    cr = parse_number(rec.get(cr_col or "")) or 0.0
                    signed = cr - abs(dr)
                if signed == 0:
                    continue
                out.append(
                    ParsedTransaction.from_signed(
                        date=dt,
                        raw_descriptor=rec[desc_col],
                        signed_amount=signed,
                        balance_after=parse_number(rec.get(bal_col)) if bal_col else None,
                    )
                )
        if not out and self.needs_ocr:
            raise ParseError("no text layer found; OCR required")
        return out
