from fincopilot.ingestion.parsers.generic_csv import GenericCSVParser, parse_number


def test_detect_and_parse(csv_statement):
    p = GenericCSVParser()
    assert p.detect(csv_statement) >= 0.5
    txns = p.parse(csv_statement)
    assert len(txns) == 4
    inc = txns[0]
    assert inc.direction == "credit" and inc.amount == 4200.0
    debit = txns[1]
    assert debit.direction == "debit" and debit.amount == 84.20
    assert debit.balance_after == 4115.80


def test_parse_number_formats():
    assert parse_number("(12.30)") == -12.30
    assert parse_number("1,234.56") == 1234.56
    assert parse_number("") is None
    assert parse_number(None) is None
