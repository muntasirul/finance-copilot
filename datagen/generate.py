"""Synthetic transaction generator.

Produces ``transactions.parquet`` — labelled, multi-persona, multi-bank-format — plus a
``kaggle_dataset/`` folder (data + dataset-metadata.json) ready for ``kaggle datasets create``.

    python -m datagen.generate --out data/processed/synthetic --personas 12 --months 24 --seed 42
    python -m datagen.generate --out data/processed/synthetic --kaggle-out kaggle_dataset
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fincopilot.taxonomy import SUB_TO_PARENT

HERE = Path(__file__).parent


def _load(name: str) -> dict[str, Any]:
    return yaml.safe_load((HERE / name).read_text())


def fmt_descriptor(rng: random.Random, template: str, merchant: str, dt: date) -> str:
    merchant_up = merchant.upper()
    return (
        template.replace("{merchant_short}", merchant_up[:12])
        .replace("{merchant}", merchant_up)
        .replace("{ref}", f"{rng.randint(10**8, 10**9 - 1)}")
        .replace("{terminal}", f"{rng.randint(1000, 9999)}")
        .replace("{num4}", f"{rng.randint(1000, 9999)}")
        .replace("{masked}", f"{rng.randint(10**9, 10**10 - 1)}")
        .replace("{date}", dt.strftime("%d/%m"))
    )


class Gen:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.personas = _load("personas.yaml")
        self.catalog = _load("merchants.yaml")
        self.templates = _load("templates.yaml")
        self.rows: list[dict[str, Any]] = []
        self.sub_to_cat = dict(SUB_TO_PARENT)
        # merchant name -> (taxonomy subcategory, merchant dict) for persona subscription
        # lookups; pseudo-key merchants declare their real subcategory via `sub`
        self.merchant_by_name: dict[str, tuple[str, dict[str, Any]]] = {
            m["name"]: (m.get("sub", sub), m) for sub, ms in self.catalog.items() for m in ms
        }

    # ---- helpers ---------------------------------------------------------
    def _amount(self, lo: float, hi: float) -> float:
        return round(self.rng.uniform(lo, hi), 2)

    def _merchant(self, sub: str) -> dict[str, Any]:
        pool = self.catalog[sub]
        return self.rng.choices(pool, weights=[m["weight"] for m in pool], k=1)[0]

    def _bank(self) -> dict[str, Any]:
        return self.rng.choice(list(self.templates["banks"].values()))

    def _add(
        self,
        *,
        account: str,
        persona: str,
        dt: date,
        merchant_key: str,
        sub: str,
        amount: float,
        direction: str,
        balance: float,
        bank: dict[str, Any],
        recurring: bool = False,
        anomaly: str | None = None,
    ) -> float:
        desc = fmt_descriptor(self.rng, self.rng.choice(bank["templates"]), merchant_key, dt)
        balance = round(balance - amount if direction == "debit" else balance + amount, 2)
        self.rows.append(
            {
                "txn_id": uuid.uuid4().hex,
                "account_id": account,
                "persona": persona,
                "date": dt.isoformat(),
                "raw_descriptor": desc,
                "amount": amount,
                "direction": direction,
                "balance_after": balance,
                "merchant_id": merchant_key,
                "subcategory": sub,
                "category": self.sub_to_cat.get(sub, "other"),
                "is_recurring": recurring,
                "anomaly_type": anomaly or "",
            }
        )
        return balance

    # ---- one persona ------------------------------------------------------
    def generate_persona(self, persona: str, months: int) -> None:
        p = self.personas[persona]
        account = f"acct_{persona}_{uuid.uuid4().hex[:6]}"
        bank = self._bank()
        balance = float(self.rng.uniform(400, 2500))
        start = date.today() - timedelta(days=months * 31)
        end = date.today() - timedelta(days=1)

        # daily debit spend
        n_months = (end - start).days / 30
        lo, hi = p["spend_monthly"]
        monthly_budget = self.rng.uniform(lo, hi)
        n_txn = int(n_months * self.rng.uniform(*p["txn_per_month"]))
        days = sorted(self.rng.sample(range((end - start).days), k=min(n_txn, (end - start).days)))
        per_txn = monthly_budget / max(1, p["txn_per_month"][1] // 2)
        for i in days:
            dt = start + timedelta(days=i)
            amt = round(float(np.random.lognormal(np.log(per_txn), p["daily_spend_sigma"])), 2)
            amt = float(np.clip(amt, 1.5, 400))
            # random daily spend only over taxonomy subcategories (catalog also contains
            # pseudo-keys like icloud/adobe_cc that exist solely for persona subscriptions)
            candidates = [s for s in self.catalog if s in self.sub_to_cat]
            sub = self.rng.choices(
                candidates,
                weights=[sum(m["weight"] for m in self.catalog[s]) for s in candidates],
            )[0]
            if "subscription" in self._merchant(sub):
                sub = "groceries"  # never draw subs for random daily spend
            m = self._merchant(sub)
            balance = self._add(
                account=account,
                persona=persona,
                dt=dt,
                merchant_key=m["name"],
                sub=sub,
                amount=amt,
                direction="debit",
                balance=balance,
                bank=bank,
            )

        # income + rent + utilities + subscriptions (recurring)
        for i in range(months):
            base = start + timedelta(days=i * 30)
            inc = p["income"]
            d_inc = base + timedelta(
                days=(inc["day_of_month"] or 15)
                - 1
                + self.rng.randint(-inc["jitter_days"], inc["jitter_days"])
            )
            balance = self._add(
                account=account,
                persona=persona,
                dt=min(d_inc, end),
                merchant_key=inc["merchant"],
                sub=inc["subcategory"],
                amount=inc["amount"],
                direction="credit",
                balance=balance,
                bank=bank,
            )

            d_rent = base + timedelta(days=p["rent"]["day_of_month"] - 1)
            balance = self._add(
                account=account,
                persona=persona,
                dt=min(d_rent, end),
                merchant_key=p["rent"]["subcategory"],
                sub=p["rent"]["subcategory"],
                amount=p["rent"]["amount"],
                direction="debit",
                balance=balance,
                bank=bank,
                recurring=True,
            )

            for util, amt in p.get("utilities", {}).items():
                balance = self._add(
                    account=account,
                    persona=persona,
                    dt=base + timedelta(days=8),
                    merchant_key=util,
                    sub=util,
                    amount=amt,
                    direction="debit",
                    balance=balance,
                    bank=bank,
                    recurring=True,
                )

            for sub_key in p.get("subscriptions", []):
                # subscription entries may name a merchant (e.g. "netflix", "amazon prime"
                # written as amazon_prime) or a catalog pseudo-key (e.g. "gym_membership" →
                # its first merchant, which declares its real taxonomy subcategory via `sub`)
                name = sub_key.replace("_", " ")
                if name in self.merchant_by_name:
                    real_sub, m = self.merchant_by_name[name]
                else:
                    m = self.catalog[sub_key][0]
                    real_sub = m.get("sub", sub_key)
                balance = self._add(
                    account=account,
                    persona=persona,
                    dt=base + timedelta(days=self.rng.randint(10, 22)),
                    merchant_key=m["name"],
                    sub=real_sub,
                    amount=m["amount"][0],
                    direction="debit",
                    balance=balance,
                    bank=bank,
                    recurring=True,
                )

        # inject labelled anomalies (~1.5% of rows)
        n_rows = sum(1 for r in self.rows if r["account_id"] == account)
        for _ in range(max(3, int(n_rows * 0.015))):
            kind = self.rng.choice(["duplicate_charge", "amount_spike", "new_merchant_burst"])
            dt = start + timedelta(days=self.rng.randint(0, (end - start).days - 1))
            if kind == "duplicate_charge" and self.rows:
                src = self.rng.choice(
                    [
                        r
                        for r in self.rows
                        if r["account_id"] == account and r["direction"] == "debit"
                    ]
                )
                balance = self._add(
                    account=account,
                    persona=persona,
                    dt=date.fromisoformat(src["date"]) + timedelta(hours=6),
                    merchant_key=src["merchant_id"],
                    sub=src["subcategory"],
                    amount=src["amount"],
                    direction="debit",
                    balance=balance,
                    bank=bank,
                    anomaly=kind,
                )
            elif kind == "amount_spike":
                m = self._merchant("online_marketplace")
                balance = self._add(
                    account=account,
                    persona=persona,
                    dt=dt,
                    merchant_key=m["name"],
                    sub="online_marketplace",
                    amount=round(per_txn * self.rng.uniform(8, 15), 2),
                    direction="debit",
                    balance=balance,
                    bank=bank,
                    anomaly=kind,
                )
            else:
                fake = f"merchant_{uuid.uuid4().hex[:8]}"
                for j in range(self.rng.randint(3, 6)):
                    balance = self._add(
                        account=account,
                        persona=persona,
                        dt=dt + timedelta(days=j),
                        merchant_key=fake,
                        sub="online_marketplace",
                        amount=self._amount(15, 60),
                        direction="debit",
                        balance=balance,
                        bank=bank,
                        anomaly=kind,
                    )

    # ---- driver -----------------------------------------------------------
    def run(self, out: Path, kaggle_out: Path | None, months: int) -> None:
        for persona, p in self.personas.items():
            for _ in range(p["count"]):
                self.generate_persona(persona, months)
        df = pd.DataFrame(self.rows).sort_values(["account_id", "date"]).reset_index(drop=True)
        out.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out / "transactions.parquet", index=False)
        df.to_csv(out / "transactions.csv", index=False)
        print(f"wrote {len(df):,} transactions → {out}/transactions.parquet")
        print(df["category"].value_counts().head(10).to_string())

        if kaggle_out:
            kaggle_out.mkdir(parents=True, exist_ok=True)
            df.to_parquet(kaggle_out / "transactions.parquet", index=False)
            (kaggle_out / "dataset-metadata.json").write_text(
                json.dumps(
                    {
                        "title": "fincopilot-synthetic-transactions",
                        "id": "YOUR_KAGGLE_USERNAME/fincopilot-synthetic-transactions",
                        "licenses": [{"name": "CC0-1.0"}],
                    },
                    indent=2,
                )
            )
            print(f"kaggle dataset staged at {kaggle_out}/ — push with:")
            print("  kaggle datasets create -p kaggle_dataset --dir-mode zip")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/processed/synthetic"))
    ap.add_argument("--kaggle-out", type=Path, default=None)
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--personas", type=int, default=12, help="ignored; count comes from personas.yaml"
    )
    args = ap.parse_args()
    np.random.seed(args.seed)
    Gen(args.seed).run(args.out, args.kaggle_out, args.months)


if __name__ == "__main__":
    main()
