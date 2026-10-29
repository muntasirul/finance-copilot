"""Download the public datasets used for pre-training and evaluation.

Run:
    python scripts/download_datasets.py            # all datasets
    python scripts/download_datasets.py --list     # print what's configured

Requires a Kaggle API token at ~/.kaggle/kaggle.json
(https://www.kaggle.com/settings → Create New Token). Missing credentials are reported
gracefully so CI never hard-fails.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

RAW = Path("data/raw")

# slug -> (kind, dest_subdir, note)
DATASETS: dict[str, tuple[str, str, str]] = {
    # Merchant-string / transaction categorisation corpora
    "priyamchoksi/credit-card-transactions-dataset": (
        "dataset",
        "credit_card_transactions",
        "1.3M simulated card transactions with merchant + category",
    ),
    "apoorvats/transaction-data": (
        "dataset",
        "transaction_data",
        "retail bank transactions with descriptions",
    ),
    # IEEE-CIS fraud (useful later for anomaly work — large, download if you want it)
    # "ieee-fraud-detection": ("competition", "ieee_fraud", "Kaggle competition files"),
}

MCC_CSV = "mcc_codes.csv"


def have_credentials() -> bool:
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def download(slug: str, kind: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = (
        ["kaggle", "competitions", "files", "-c", slug, "-p", str(dest), "-d"]
        if kind == "competition"
        else ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest)]
    )
    print(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr.strip())
        return False
    for z in dest.glob("*.zip"):
        with zipfile.ZipFile(z) as f:
            f.extractall(dest)
        z.unlink()
    print(f"  → {dest}")
    return True


def write_mcc(dest: Path) -> None:
    """Ship a small built-in MCC slice so the repo is self-contained without network."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / MCC_CSV).write_text(
        "mcc,description,category\n"
        "5411,Grocery Stores Supermarkets,groceries\n"
        "5812,Eating Places Restaurants,restaurants\n"
        "5814,Fast Food Restaurants,restaurants\n"
        "5499,Misc Food Stores,groceries\n"
        "5541,Service Stations,fuel\n"
        "4111,Local Suburban Commuter Transport,public_transit\n"
        "4121,Taxicabs Limousines,ride_hailing\n"
        "5651,Family Clothing Stores,clothing\n"
        "5732,Electronics Stores,electronics\n"
        "5942,Book Stores,books\n"
        "5912,Drug Stores Pharmacies,pharmacy\n"
        "5941,Sporting Goods Stores,entertainment\n"
        "7997,Membership Clubs,gym\n"
        "5999,Miscellaneous Specialty Retail,online_marketplace\n"
        "6011,Automated Cash Disburse,cash_withdrawal\n"
        "6300,Insurance Sales,insurance_other\n"
        "8211,Elementary Secondary Schools,education\n"
        "8299,Schools Educational Services,education\n"
        "5813,Drinking Places,restaurants\n"
    )
    print(f"  → {dest / MCC_CSV} (built-in MCC taxonomy)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for slug, (kind, _sub, note) in DATASETS.items():
            print(f"{slug:55s} [{kind}] {note}")
        return

    write_mcc(RAW / "taxonomy")

    if not have_credentials():
        print(
            "\n⚠  ~/.kaggle/kaggle.json not found — skipping Kaggle downloads.\n"
            "   Create a token at https://www.kaggle.com/settings → 'Create New Token', then:\n"
            "   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/\n"
            "   chmod 600 ~/.kaggle/kaggle.json\n"
            "   (The project still works fully on synthetic data.)"
        )
        return

    if not shutil.which("kaggle"):
        print("kaggle CLI not on PATH — run: uv pip install kaggle")
        return

    ok = 0
    for slug, (kind, sub, _note) in DATASETS.items():
        ok += download(slug, kind, RAW / sub)
    print(f"\n{ok}/{len(DATASETS)} datasets downloaded to {RAW}/")


if __name__ == "__main__":
    main()
