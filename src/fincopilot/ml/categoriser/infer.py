"""Inference wrapper. Two backends:

* ``OnnxCategoriser`` — the real model exported from the Kaggle notebook (tokenizer + ONNX).
* ``SklearnCategoriser`` — TF-IDF + LogisticRegression baseline pickled by ``train.py``.
* ``NullCategoriser`` — used when no model is present; everything is ``uncategorised``.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from fincopilot.config import settings
from fincopilot.ml.categoriser.features import build_text
from fincopilot.taxonomy import ID2LABEL, UNCATEGORISED


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    top3: tuple[str, ...]
    needs_review: bool


class Categoriser(Protocol):
    version: str

    def predict(
        self, normalised: list[str], amounts: list[float], weekdays: list[int]
    ) -> list[Prediction]: ...


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)  # type: ignore[no-any-return]


def _to_predictions(probs: np.ndarray, threshold: float) -> list[Prediction]:
    out: list[Prediction] = []
    for row in probs:
        idx = np.argsort(row)[::-1][:3]
        conf = float(row[idx[0]])
        label = ID2LABEL[int(idx[0])] if conf >= threshold else UNCATEGORISED
        out.append(
            Prediction(
                label=label,
                confidence=conf,
                top3=tuple(ID2LABEL[int(i)] for i in idx),
                needs_review=conf < threshold,
            )
        )
    return out


class NullCategoriser:
    version = "none"

    def predict(
        self, normalised: list[str], amounts: list[float], weekdays: list[int]
    ) -> list[Prediction]:
        return [Prediction(UNCATEGORISED, 0.0, (UNCATEGORISED,), True) for _ in normalised]


class SklearnCategoriser:
    """Baseline model: a scikit-learn Pipeline with ``predict_proba`` over ``build_text``."""

    def __init__(self, path: Path, threshold: float):
        with (path / "pipeline.pkl").open("rb") as f:
            self.pipeline = pickle.load(f)
        meta = json.loads((path / "meta.json").read_text())
        self.version = meta.get("version", "sklearn-unknown")
        self.threshold = threshold

    def predict(
        self, normalised: list[str], amounts: list[float], weekdays: list[int]
    ) -> list[Prediction]:
        if not normalised:
            return []
        texts = [build_text(n, a, w) for n, a, w in zip(normalised, amounts, weekdays, strict=True)]
        probs = self.pipeline.predict_proba(texts)
        full = np.zeros((len(texts), len(ID2LABEL)))
        full[:, self.pipeline.classes_] = probs
        return _to_predictions(full, self.threshold)


class OnnxCategoriser:
    def __init__(self, path: Path, threshold: float, batch_size: int = 64):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.session = ort.InferenceSession(
            str(path / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        meta = json.loads((path / "meta.json").read_text())
        self.version = meta["version"]
        self.max_length = int(meta.get("max_length", 48))
        self.threshold = threshold
        self.batch_size = batch_size

    def predict(
        self, normalised: list[str], amounts: list[float], weekdays: list[int]
    ) -> list[Prediction]:
        texts = [build_text(n, a, w) for n, a, w in zip(normalised, amounts, weekdays, strict=True)]
        probs: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tokenizer(
                texts[i : i + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            feeds = {k: v.astype(np.int64) for k, v in enc.items() if k in self.input_names}
            logits = self.session.run(None, feeds)[0]
            probs.append(_softmax(logits))
        return _to_predictions(np.concatenate(probs), self.threshold) if probs else []


def load_categoriser(models_dir: Path | None = None) -> Categoriser:
    """Prefer ONNX transformer, then sklearn baseline, then the null model."""
    root = (models_dir or settings.models_dir) / "categoriser"
    thr = settings.categoriser_confidence_threshold
    if (root / "model.onnx").exists():
        return OnnxCategoriser(root, thr)
    if (root / "pipeline.pkl").exists():
        return SklearnCategoriser(root, thr)
    return NullCategoriser()
