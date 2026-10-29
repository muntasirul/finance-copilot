from __future__ import annotations

import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fincopilot.config import Settings
from fincopilot.main import create_app


@pytest.fixture()
def cfg(tmp_path: Path) -> Settings:
    return Settings(app_dir=tmp_path / "app", session_token="test-token")


@pytest.fixture()
def client(cfg: Settings, tmp_path: Path) -> TestClient:
    app = create_app(cfg, db_path=":memory:")
    with TestClient(app) as c:
        c.headers["Authorization"] = "Bearer test-token"
        yield c


@pytest.fixture()
def csv_statement(tmp_path: Path) -> Path:
    p = tmp_path / "stmt.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Description", "Debit", "Credit", "Balance"])
        w.writerow(["2026-01-01", "ACME PAYROLL", "", "4200.00", "4200.00"])
        w.writerow(["2026-01-05", "POS DEBIT 4451 WALMART CHICAGO 01/05", "84.20", "", "4115.80"])
        w.writerow(
            ["2026-01-06", "CARD PURCHASE STARBUCKS AUSTIN US 889123445", "6.10", "", "4109.70"]
        )
        w.writerow(["2026-01-10", "POS DEBIT 4451 WALMART CHICAGO 01/05", "84.20", "", "4025.50"])
    return p
