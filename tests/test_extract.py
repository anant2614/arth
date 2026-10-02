import pytest

from arth.extract import detect_channel, extract, looks_like_person, parse_amount, parse_date


@pytest.mark.parametrize("text,amount", [
    ("Rs.450 debited from A/c XX1234 to VPA zomatoonline@paytm on 01-10-26", 450.0),
    ("INR 1,02,345.50 credited to your A/c", 102345.50),
    ("₹ 99.5 spent at DMART", 99.5),
    ("Sent Rs.500.00 from Kotak Bank AC X9876", 500.0),
    # balance must not be mistaken for the amount
    ("Avl Bal INR 12,000.00. Rs 250 debited for UPI", 250.0),
    ("Dear UPI user A/C X5549 debited by 5,697.09 on date 30 May 2026", 5697.09),
    ("UPI/DR/402155/SWIGGY/YESB/swiggy@ybl/Pay", None),
])
def test_amount(text, amount):
    assert parse_amount(text) == amount


@pytest.mark.parametrize("text,iso", [
    ("on 01-10-26", "2026-10-01"),
    ("on 30/09/2026", "2026-09-30"),
    ("on 30-Sep-26", "2026-09-30"),
    ("on 15SEP26.UPI", "2026-09-15"),
    ("on 2026-10-03", "2026-10-03"),
    ("on 05 Oct 2026", "2026-10-05"),
    ("on 31-02-26", None),  # invalid calendar date
])
def test_date(text, iso):
    assert parse_date(text) == iso


def test_balance_and_account():
    ex = extract("Your A/c XX5678 is credited with INR 85,000.00 on 30-Sep-26 by NEFT from ACME TECHNOLOGIES "
                 "PVT LTD. Avl Bal INR 1,02,345.50")
    assert ex.amount == 85000.0
    assert ex.balance == 102345.50
    assert ex.account_suffix == "5678"
    assert ex.direction == "credit"
    assert ex.channel == "NEFT"
    assert ex.counterparty_type == "merchant"


@pytest.mark.parametrize("text,channel", [
    ("UPI/DR/402155/SWIGGY/YESB/swiggy@ybl/Pay", "UPI"),
    ("NACH DR BSE STAR MF 0012345", "NACH"),
    ("ACH D- TP ACH BAJAJFINANCE-1234567", "NACH"),
    ("IMPS/P2A/402166/RAMESH K/rent oct", "IMPS"),
    ("NEFT CR-HDFC0000001-ACME TECHNOLOGIES PVT LTD-SALARY SEP", "NEFT"),
    ("RTGS CR-HDFC0000060-RELIANCE INDUSTRIES LTD", "RTGS"),
    ("ATW-512345XXXXXX1234-S1ANMU12-MUMBAI", "ATM"),
    ("POS 4591XXXXXXXX1234 DMART AVENUE", "CARD"),
    ("BIL/BPAY/001234/BESCOM", "BILLPAY"),
    ("CHQ RTN CHGS", "CHEQUE"),
    ("CASH DEP PUNE BRANCH", "CASH"),
    ("INT.PD:01-07-2026 TO 30-09-2026", "INTEREST"),
    ("SMS CHGS QTR SEP26", "CHARGES"),
    ("MB/TPT/SOMEONE/rent", "NETBANKING"),
    ("hello world", "OTHER"),
])
def test_channel(text, channel):
    assert detect_channel(text) == channel


def test_prd_examples():
    ex = extract("Rs.450 debited from A/c XX1234 to VPA zomatoonline@paytm on 01-10-26")
    assert (ex.amount, ex.date, ex.account_suffix, ex.channel, ex.direction) == (450.0, "2026-10-01", "1234", "UPI", "debit")
    assert ex.merchant == "Zomato" and ex.counterparty_type == "merchant"

    ex = extract("UPI/DR/402155/SWIGGY/YESB/swiggy@ybl/Pay")
    assert ex.merchant == "Swiggy" and ex.merchant_concept == "food_delivery"

    ex = extract("NACH DR BSE STAR MF 0012345")
    assert ex.channel == "NACH" and ex.merchant_concept == "investment"

    ex = extract("IMPS/P2A/402166/RAMESH K/rent oct")
    assert ex.counterparty == "RAMESH K" and ex.counterparty_type == "person" and ex.remark == "rent oct"


def test_counterparty_types():
    assert extract("UPI-RAHUL SHARMA-rahul.s@okhdfcbank-HDFC0001234-412345678901-dinner").counterparty_type == "person"
    assert extract("UPI/DR/1/EXCITEL BROADBAND/YESB/excitel@ybl/plan").counterparty_type != "person"
    assert extract("NEFT DR-SBIN0004321-GANESH STEEL TRADERS-NETBANK-inv 778").counterparty_type == "merchant"
    assert extract("MB/TPT/OWN A/C/sweep").counterparty_type == "self"
    assert extract("INT.PD:01-07-2026 TO 30-09-2026").counterparty_type == "institution"


def test_caller_direction_wins():
    assert extract("IMPS/P2A/402166/RAMESH K/rent oct", direction="debit").direction == "debit"


def test_person_lexicon():
    assert looks_like_person("Priya Nair")
    assert looks_like_person("MUMMY")
    assert not looks_like_person("CHAI POINT")
    assert not looks_like_person("LAKSHMI TEXTILES")  # business word wins
    assert not looks_like_person(None)


def test_public_fields_exclude_pii():
    ex = extract("UPI-RAHUL SHARMA-rahul.s@okhdfcbank-HDFC0001234-412345678901-dinner")
    pub = ex.public()
    assert "counterparty" not in pub and "vpas" not in pub and "remark" not in pub
