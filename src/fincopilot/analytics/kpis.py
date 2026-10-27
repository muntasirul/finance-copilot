"""SQL-backed KPIs consumed by the dashboard and (read-only) by the LLM assistant."""

from __future__ import annotations

from typing import Any

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from fincopilot.db.models import Direction, Transaction
from fincopilot.taxonomy import SUB_TO_PARENT

_month = func.strftime("%Y-%m", Transaction.date)
_debit = case((Transaction.direction == Direction.debit, Transaction.amount), else_=0.0)
_credit = case((Transaction.direction == Direction.credit, Transaction.amount), else_=0.0)


def monthly_summary(session: Session, account_id: str | None = None) -> list[dict[str, Any]]:
    q = select(
        _month.label("month"),
        func.sum(_credit).label("income"),
        func.sum(_debit).label("spend"),
    )
    if account_id:
        q = q.where(Transaction.account_id == account_id)
    q = q.group_by(_month).order_by(_month)
    rows = session.execute(q).all()
    return [
        {
            "month": m,
            "income": round(i or 0, 2),
            "spend": round(s or 0, 2),
            "net": round((i or 0) - (s or 0), 2),
            "savings_rate": round(((i or 0) - (s or 0)) / i, 4) if i else None,
        }
        for m, i, s in rows
    ]


def spend_by_category(session: Session, month: str | None = None) -> list[dict[str, Any]]:
    q = select(Transaction.subcategory, func.sum(Transaction.amount).label("spend")).where(
        Transaction.direction == Direction.debit
    )
    if month:
        q = q.where(_month == month)
    q = q.group_by(Transaction.subcategory).order_by(func.sum(Transaction.amount).desc())
    return [
        {"subcategory": s, "category": SUB_TO_PARENT.get(s, "other"), "spend": round(v, 2)}
        for s, v in session.execute(q).all()
    ]


def top_merchants(session: Session, limit: int = 10) -> list[dict[str, Any]]:
    q = (
        select(
            Transaction.merchant_id,
            func.count().label("n"),
            func.sum(Transaction.amount).label("spend"),
        )
        .where(Transaction.direction == Direction.debit)
        .group_by(Transaction.merchant_id)
        .order_by(func.sum(Transaction.amount).desc())
        .limit(limit)
    )
    return [
        {"merchant_id": m, "count": n, "spend": round(s, 2)} for m, n, s in session.execute(q).all()
    ]


def review_queue_size(session: Session) -> int:
    return session.scalar(select(func.count()).where(Transaction.needs_review.is_(True))) or 0
