from __future__ import annotations

import enum
import uuid
from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class Direction(enum.StrEnum):
    debit = "debit"
    credit = "credit"


class LabelSource(enum.StrEnum):
    model = "model"
    user = "user"
    rule = "rule"
    import_ = "import"


class AlertKind(enum.StrEnum):
    duplicate_charge = "duplicate_charge"
    amount_spike = "amount_spike"
    new_merchant_burst = "new_merchant_burst"
    subscription_price_change = "subscription_price_change"
    unusual_pattern = "unusual_pattern"


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(120))
    institution: Mapped[str | None] = mapped_column(String(120))
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    transactions: Mapped[list[Transaction]] = relationship(back_populates="account")


class ImportBatch(Base):
    __tablename__ = "import_batches"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"))
    filename: Mapped[str] = mapped_column(String(255))
    parser: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(20), default="pending")
    rows_total: Mapped[int] = mapped_column(Integer, default=0)
    rows_new: Mapped[int] = mapped_column(Integer, default=0)
    rows_duplicate: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Merchant(Base):
    __tablename__ = "merchants"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    canonical_name: Mapped[str] = mapped_column(String(120), unique=True)
    default_subcategory: Mapped[str | None] = mapped_column(String(40))


class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = (
        UniqueConstraint("account_id", "dedupe_hash", name="uq_txn_dedupe"),
        Index("ix_txn_account_date", "account_id", "date"),
        Index("ix_txn_subcategory", "subcategory"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"))
    import_batch_id: Mapped[str | None] = mapped_column(ForeignKey("import_batches.id"))
    merchant_id: Mapped[str | None] = mapped_column(ForeignKey("merchants.id"))

    date: Mapped[date] = mapped_column(Date)
    posted_date: Mapped[date | None] = mapped_column(Date)
    raw_descriptor: Mapped[str] = mapped_column(Text)
    normalised_descriptor: Mapped[str] = mapped_column(Text)
    amount: Mapped[float] = mapped_column(Float)  # always positive; see direction
    direction: Mapped[Direction] = mapped_column(Enum(Direction))
    balance_after: Mapped[float | None] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(3), default="USD")

    subcategory: Mapped[str] = mapped_column(String(40), default="uncategorised")
    category_confidence: Mapped[float | None] = mapped_column(Float)
    label_source: Mapped[LabelSource] = mapped_column(Enum(LabelSource), default=LabelSource.model)
    model_version: Mapped[str | None] = mapped_column(String(40))
    needs_review: Mapped[bool] = mapped_column(default=False)

    dedupe_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    account: Mapped[Account] = relationship(back_populates="transactions")
    merchant: Mapped[Merchant | None] = relationship()


class Label(Base):
    """Every user correction is retained for incremental retraining."""

    __tablename__ = "labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(ForeignKey("transactions.id"))
    old_subcategory: Mapped[str] = mapped_column(String(40))
    new_subcategory: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"))
    merchant_id: Mapped[str] = mapped_column(ForeignKey("merchants.id"))
    cadence_days: Mapped[int] = mapped_column(Integer)
    typical_amount: Mapped[float] = mapped_column(Float)
    last_seen: Mapped[date] = mapped_column(Date)
    next_expected: Mapped[date] = mapped_column(Date)
    price_change_pct: Mapped[float | None] = mapped_column(Float)
    active: Mapped[bool] = mapped_column(default=True)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    transaction_id: Mapped[str | None] = mapped_column(ForeignKey("transactions.id"))
    kind: Mapped[AlertKind] = mapped_column(Enum(AlertKind))
    score: Mapped[float] = mapped_column(Float)
    explanation: Mapped[str] = mapped_column(Text)
    dismissed: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Budget(Base):
    __tablename__ = "budgets"
    __table_args__ = (UniqueConstraint("subcategory", "month", name="uq_budget_month"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subcategory: Mapped[str] = mapped_column(String(40))
    month: Mapped[str] = mapped_column(String(7))  # YYYY-MM
    amount: Mapped[float] = mapped_column(Float)


class Forecast(Base):
    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"))
    generated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    target_date: Mapped[date] = mapped_column(Date)
    balance_p10: Mapped[float] = mapped_column(Float)
    balance_p50: Mapped[float] = mapped_column(Float)
    balance_p90: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(40))


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    action: Mapped[str] = mapped_column(String(60))
    detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
