from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from fincopilot.db.models import Direction, ImportBatch, LabelSource, Merchant, Transaction
from fincopilot.ingestion.dedupe import split_new_and_duplicate
from fincopilot.ingestion.normalise import canonical_merchant, normalise_descriptor
from fincopilot.ingestion.registry import ParserRegistry
from fincopilot.ml.categoriser.infer import Categoriser


@dataclass
class ImportResult:
    batch_id: str
    parser: str
    rows_total: int
    rows_new: int
    rows_duplicate: int


def import_statement(
    session: Session,
    registry: ParserRegistry,
    categoriser: Categoriser,
    account_id: str,
    path: Path,
    parser_name: str | None = None,
) -> ImportResult:
    """Parse a statement file, de-duplicate, categorise, and persist. Idempotent on re-import."""
    if parser_name:
        parser = registry.get(parser_name)
    else:
        detected = registry.detect(path)
        if detected is None:
            raise ValueError(f"no parser could handle {path.name}")
        parser = detected[0]

    batch = ImportBatch(account_id=account_id, filename=path.name, parser=parser.name)
    session.add(batch)
    session.flush()

    try:
        parsed = parser.parse(path)
        existing = set(
            session.scalars(
                select(Transaction.dedupe_hash).where(Transaction.account_id == account_id)
            )
        )
        new_rows, dup = split_new_and_duplicate(parsed, existing)

        normalised = [normalise_descriptor(t.raw_descriptor) for t, _ in new_rows]
        preds = categoriser.predict(
            normalised, [t.amount for t, _ in new_rows], [t.date.weekday() for t, _ in new_rows]
        )

        merchants = _merchant_cache(session)
        for (t, h), norm, pred in zip(new_rows, normalised, preds, strict=True):
            m_key = canonical_merchant(norm)
            merchant = merchants.get(m_key)
            if merchant is None:
                merchant = Merchant(canonical_name=m_key)
                session.add(merchant)
                session.flush()
                merchants[m_key] = merchant
            session.add(
                Transaction(
                    account_id=account_id,
                    import_batch_id=batch.id,
                    merchant_id=merchant.id,
                    date=t.date,
                    posted_date=t.posted_date,
                    raw_descriptor=t.raw_descriptor,
                    normalised_descriptor=norm,
                    amount=t.amount,
                    direction=Direction(t.direction),
                    balance_after=t.balance_after,
                    currency=t.currency,
                    subcategory=pred.label,
                    category_confidence=pred.confidence,
                    label_source=LabelSource.model,
                    model_version=categoriser.version,
                    needs_review=pred.needs_review,
                    dedupe_hash=h,
                )
            )
        batch.status, batch.rows_total = "done", len(parsed)
        batch.rows_new, batch.rows_duplicate = len(new_rows), dup
    except Exception as e:
        batch.status, batch.error = "failed", str(e)
        raise
    finally:
        session.flush()

    return ImportResult(batch.id, parser.name, len(parsed), len(new_rows), dup)


def _merchant_cache(session: Session) -> dict[str, Merchant]:
    return {m.canonical_name: m for m in session.scalars(select(Merchant))}
