"""Two-level category taxonomy shared by the data generator, models, and the app."""

from __future__ import annotations

TAXONOMY: dict[str, list[str]] = {
    "income": ["salary", "freelance", "refund", "interest", "transfer_in", "other_income"],
    "housing": [
        "rent",
        "mortgage",
        "utilities_electric",
        "utilities_gas",
        "utilities_water",
        "internet",
        "phone",
    ],
    "food": ["groceries", "restaurants", "cafes", "food_delivery"],
    "transport": ["fuel", "ride_hailing", "public_transit", "parking", "vehicle_maintenance"],
    "shopping": ["clothing", "electronics", "home_goods", "online_marketplace"],
    "health": ["pharmacy", "doctor", "hospital", "gym", "insurance_health"],
    "entertainment": ["streaming", "gaming", "events", "books"],
    "education": ["tuition", "courses", "books_edu"],
    "finance": ["loan_payment", "credit_card_payment", "bank_fee", "investment", "insurance_other"],
    "family": ["childcare", "pets", "gifts", "charity"],
    "travel": ["flights", "hotels", "travel_other"],
    "transfers": ["transfer_out", "cash_withdrawal", "p2p"],
    "other": ["uncategorised"],
}

SUBCATEGORIES: list[str] = [s for subs in TAXONOMY.values() for s in subs]
SUB_TO_PARENT: dict[str, str] = {s: p for p, subs in TAXONOMY.items() for s in subs}
LABEL2ID: dict[str, int] = {s: i for i, s in enumerate(SUBCATEGORIES)}
ID2LABEL: dict[int, str] = {i: s for s, i in LABEL2ID.items()}
UNCATEGORISED = "uncategorised"

assert len(SUBCATEGORIES) == len(set(SUBCATEGORIES)), "duplicate subcategory names"
