"""Local CPU baseline: TF-IDF (char n-grams) + Logistic Regression.

This is the model the app ships with until the Kaggle-trained DistilBERT ONNX export is
dropped into ``models/categoriser/``. It also provides the "baseline to beat" numbers.

Usage:
    python -m fincopilot.ml.categoriser.train --data data/processed --out models/categoriser
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, top_k_accuracy_score
from sklearn.pipeline import Pipeline

from fincopilot.ingestion.normalise import normalise_descriptor
from fincopilot.ml.categoriser.features import build_text
from fincopilot.taxonomy import LABEL2ID


def load_split(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(data_dir / "synthetic" / "transactions.parquet")
    # split by account so we measure generalisation to unseen users (~25%)
    accounts = sorted(df["account_id"].unique())
    test_accounts = set(accounts[:: max(1, len(accounts) // 4)])
    test = df[df["account_id"].isin(test_accounts)]
    train = df[~df["account_id"].isin(test_accounts)]
    return train, test


def to_text(df: pd.DataFrame) -> list[str]:
    return [
        build_text(normalise_descriptor(r.raw_descriptor), r.amount, pd.Timestamp(r.date).weekday())
        for r in df.itertuples()
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("models/categoriser"))
    args = ap.parse_args()

    train, test = load_split(args.data)
    x_tr, x_te = to_text(train), to_text(test)
    y_tr = train["subcategory"].map(LABEL2ID).to_numpy()
    y_te = test["subcategory"].map(LABEL2ID).to_numpy()

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
        ]
    )
    t0 = time.perf_counter()
    pipe.fit(x_tr, y_tr)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    probs = pipe.predict_proba(x_te)
    ms_per_row = (time.perf_counter() - t0) * 1000 / max(1, len(x_te))
    pred = pipe.classes_[probs.argmax(1)]

    metrics = {
        "macro_f1": float(f1_score(y_te, pred, average="macro")),
        "accuracy": float((pred == y_te).mean()),
        "top3_accuracy": float(top_k_accuracy_score(y_te, probs, k=3, labels=pipe.classes_)),
        "ms_per_row": ms_per_row,
        "fit_seconds": fit_s,
        "n_train": len(x_tr),
        "n_test": len(x_te),
    }
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_te, pred, zero_division=0))

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "pipeline.pkl").open("wb") as f:
        pickle.dump(pipe, f)
    (args.out / "meta.json").write_text(
        json.dumps({"version": "tfidf-lr-0.1", "metrics": metrics, "classes": len(LABEL2ID)})
    )
    np.save(args.out / "test_probs.npy", probs)


if __name__ == "__main__":
    main()
