from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from fincopilot.taxonomy import SUBCATEGORIES


class AccountCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    institution: str | None = None
    currency: str = Field(default="USD", min_length=3, max_length=3)


class AccountOut(AccountCreate):
    id: str

    model_config = {"from_attributes": True}


class ImportOut(BaseModel):
    batch_id: str
    parser: str
    rows_total: int
    rows_new: int
    rows_duplicate: int


class TransactionOut(BaseModel):
    id: str
    account_id: str
    date: date
    raw_descriptor: str
    normalised_descriptor: str
    amount: float
    direction: str
    balance_after: float | None
    subcategory: str
    category_confidence: float | None
    label_source: str
    needs_review: bool

    model_config = {"from_attributes": True}


class TransactionPatch(BaseModel):
    subcategory: str

    def validate_label(self) -> str:
        if self.subcategory not in SUBCATEGORIES:
            raise ValueError(f"unknown subcategory {self.subcategory!r}")
        return self.subcategory


class Page(BaseModel):
    items: list[TransactionOut]
    total: int
    limit: int
    offset: int
