from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fincopilot.analytics import kpis
from fincopilot.api.deps import AppState, get_db, get_state, require_token
from fincopilot.api.schemas import (
    AccountCreate,
    AccountOut,
    ImportOut,
    Page,
    TransactionOut,
    TransactionPatch,
)
from fincopilot.db.models import Account, Label, LabelSource, Transaction
from fincopilot.ingestion.service import import_statement
from fincopilot.taxonomy import TAXONOMY

router = APIRouter(dependencies=[Depends(require_token)])
DB = Annotated[Session, Depends(get_db)]
State = Annotated[AppState, Depends(get_state)]


@router.get("/health", tags=["meta"])
def health(state: State) -> dict[str, str]:
    return {"status": "ok", "categoriser": state.categoriser.version}


@router.get("/taxonomy", tags=["meta"])
def taxonomy() -> dict[str, list[str]]:
    return TAXONOMY


@router.get("/parsers", tags=["meta"])
def parsers(state: State) -> list[str]:
    return state.registry.names()


@router.post("/accounts", response_model=AccountOut, status_code=201, tags=["accounts"])
def create_account(body: AccountCreate, db: DB) -> Account:
    acc = Account(**body.model_dump())
    db.add(acc)
    db.flush()
    return acc


@router.get("/accounts", response_model=list[AccountOut], tags=["accounts"])
def list_accounts(db: DB) -> list[Account]:
    return list(db.scalars(select(Account).order_by(Account.created_at)))


@router.post("/accounts/{account_id}/imports", response_model=ImportOut, tags=["imports"])
def upload_statement(
    account_id: str,
    db: DB,
    state: State,
    file: UploadFile = File(...),
    parser: str | None = None,
) -> ImportOut:
    if db.get(Account, account_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "account not found")
    suffix = Path(file.filename or "statement").suffix or ".csv"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        res = import_statement(db, state.registry, state.categoriser, account_id, tmp_path, parser)
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)
    return ImportOut(**res.__dict__)


@router.get("/transactions", response_model=Page, tags=["transactions"])
def list_transactions(
    db: DB,
    account_id: str | None = None,
    subcategory: str | None = None,
    needs_review: bool | None = None,
    search: str | None = None,
    limit: int = Query(50, le=500),
    offset: int = 0,
) -> Page:
    q = select(Transaction)
    if account_id:
        q = q.where(Transaction.account_id == account_id)
    if subcategory:
        q = q.where(Transaction.subcategory == subcategory)
    if needs_review is not None:
        q = q.where(Transaction.needs_review.is_(needs_review))
    if search:
        q = q.where(Transaction.normalised_descriptor.contains(search.lower()))
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    items = db.scalars(
        q.order_by(Transaction.date.desc(), Transaction.created_at.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return Page(
        items=[TransactionOut.model_validate(t) for t in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.patch("/transactions/{txn_id}", response_model=TransactionOut, tags=["transactions"])
def relabel(txn_id: str, body: TransactionPatch, db: DB) -> Transaction:
    """User correction: update the row and keep the label for incremental retraining."""
    txn = db.get(Transaction, txn_id)
    if txn is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "transaction not found")
    try:
        new = body.validate_label()
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e
    db.add(Label(transaction_id=txn.id, old_subcategory=txn.subcategory, new_subcategory=new))
    txn.subcategory, txn.label_source = new, LabelSource.user
    txn.needs_review, txn.category_confidence = False, 1.0
    db.flush()
    return txn


@router.get("/review-queue", response_model=Page, tags=["transactions"])
def review_queue(db: DB, limit: int = Query(20, le=200)) -> Page:
    q = (
        select(Transaction)
        .where(Transaction.needs_review.is_(True))
        .order_by(Transaction.category_confidence.asc())
        .limit(limit)
    )
    items = db.scalars(q).all()
    return Page(
        items=[TransactionOut.model_validate(t) for t in items],
        total=kpis.review_queue_size(db),
        limit=limit,
        offset=0,
    )


@router.get("/analytics/monthly", tags=["analytics"])
def monthly(db: DB, account_id: str | None = None) -> list[dict[str, object]]:
    return kpis.monthly_summary(db, account_id)


@router.get("/analytics/categories", tags=["analytics"])
def categories(db: DB, month: str | None = None) -> list[dict[str, object]]:
    return kpis.spend_by_category(db, month)


@router.get("/analytics/merchants", tags=["analytics"])
def merchants(db: DB, limit: int = Query(10, le=100)) -> list[dict[str, object]]:
    return kpis.top_merchants(db, limit)
