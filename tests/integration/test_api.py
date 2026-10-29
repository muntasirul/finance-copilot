from __future__ import annotations


def test_health_requires_token(cfg):
    from fastapi.testclient import TestClient

    from fincopilot.main import create_app

    with TestClient(create_app(cfg, db_path=":memory:")) as c:
        assert c.get("/api/health").status_code == 401
        r = c.get("/api/health", headers={"Authorization": "Bearer test-token"})
        assert r.status_code == 200 and r.json()["status"] == "ok"


def test_import_and_categorise(client, csv_statement):
    acc = client.post("/api/accounts", json={"name": "checking"}).json()
    with csv_statement.open("rb") as f:
        r = client.post(
            f"/api/accounts/{acc['id']}/imports",
            files={"file": ("stmt.csv", f, "text/csv")},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["rows_total"] == 4 and body["rows_new"] == 4 and body["rows_duplicate"] == 0

    txns = client.get(f"/api/transactions?account_id={acc['id']}").json()
    assert txns["total"] == 4
    # whichever categoriser is loaded must label every row with a valid subcategory
    assert all(t["subcategory"] for t in txns["items"])

    # Re-import is idempotent
    with csv_statement.open("rb") as f:
        r2 = client.post(f"/api/accounts/{acc['id']}/imports", files={"file": ("stmt.csv", f)})
    assert r2.json()["rows_new"] == 0 and r2.json()["rows_duplicate"] == 4


def test_relabel_creates_label_and_clears_review(client, csv_statement):
    acc = client.post("/api/accounts", json={"name": "c"}).json()
    with csv_statement.open("rb") as f:
        client.post(f"/api/accounts/{acc['id']}/imports", files={"file": ("s.csv", f)})

    target = client.get(f"/api/transactions?account_id={acc['id']}").json()["items"][0]
    r = client.patch(f"/api/transactions/{target['id']}", json={"subcategory": "groceries"})
    assert r.status_code == 200
    updated = r.json()
    assert updated["subcategory"] == "groceries"
    assert updated["label_source"] == "user"
    assert not updated["needs_review"]


def test_analytics_endpoints(client, csv_statement):
    acc = client.post("/api/accounts", json={"name": "c"}).json()
    with csv_statement.open("rb") as f:
        client.post(f"/api/accounts/{acc['id']}/imports", files={"file": ("s.csv", f)})
    monthly = client.get(f"/api/analytics/monthly?account_id={acc['id']}").json()
    assert monthly[0]["income"] == 4200.0
    assert monthly[0]["spend"] == 174.50
    cats = client.get("/api/analytics/categories").json()
    assert sum(c["spend"] for c in cats) == 174.50  # all debits are accounted for
