from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class ParsedTransaction(BaseModel):
    """Bank-agnostic transaction produced by any parser."""

    date: date
    posted_date: date | None = None
    raw_descriptor: str = Field(min_length=1)
    amount: float = Field(gt=0)
    direction: str = Field(pattern="^(debit|credit)$")
    balance_after: float | None = None
    currency: str = "USD"

    @field_validator("raw_descriptor")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @classmethod
    def from_signed(cls, *, date: date, raw_descriptor: str, signed_amount: float, **kw: object):
        """Convenience: build from a signed amount (negative = debit)."""
        return cls(
            date=date,
            raw_descriptor=raw_descriptor,
            amount=abs(signed_amount),
            direction="debit" if signed_amount < 0 else "credit",
            **kw,  # type: ignore[arg-type]
        )


class StatementParser(ABC):
    """Plugin contract. Implementations must be stateless and side-effect free."""

    name: str = "base"
    institution: str | None = None

    @abstractmethod
    def detect(self, path: Path) -> float:
        """Return a confidence in [0, 1] that this parser handles ``path``."""

    @abstractmethod
    def parse(self, path: Path) -> list[ParsedTransaction]: ...


class ParseError(RuntimeError):
    pass
