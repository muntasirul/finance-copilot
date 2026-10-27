"""Feature construction for the categoriser — identical at train (Kaggle) and inference (app)."""

from __future__ import annotations

import math

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def amount_bucket(amount: float) -> str:
    """Log-spaced bucket so the text model can condition on magnitude without exact values."""
    if amount <= 0:
        return "amt_0"
    b = min(int(math.log10(amount) * 2), 11)  # 0..11 -> up to ~10^5.5
    return f"amt_{b}"


def build_text(normalised_descriptor: str, amount: float, weekday: int) -> str:
    """Single string fed to the tokenizer: descriptor + side-information tokens."""
    return f"{normalised_descriptor} [SEP] {amount_bucket(amount)} {WEEKDAYS[weekday % 7]}"
