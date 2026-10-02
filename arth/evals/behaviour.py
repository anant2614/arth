"""Behavioural evals: small, named, human-readable expectations.

Each case is one narration plus the answer a careful human would give. They pin
down product promises (the PRD's own examples, the hard rent/salary/EMI cases that
history features exist for, refunds vs purchases) so regressions show up by name.

    python -m arth.evals.behaviour --models-dir models
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from ..engine import Engine, StudentBundle

MONTHLY_FIXED = {"same_counterparty_count": 6, "amount_cv": 0.01, "median_gap_days": 30}
ONE_OFF = {"same_counterparty_count": 1, "amount_cv": None, "median_gap_days": None}


@dataclass(frozen=True)
class Case:
    name: str
    text: str
    expected: str | bool
    question: str = "personal_finance"
    amount: float | None = None
    direction: str | None = None
    history: dict | None = None


CASES: list[Case] = [
    # PRD examples
    Case("prd_zomato_sms", "Rs.450 debited from A/c XX1234 to VPA zomatoonline@paytm on 01-10-26", "food_dining"),
    Case("prd_swiggy_upi", "UPI/DR/402155/SWIGGY/YESB/swiggy@ybl/Pay", "food_dining", amount=320, direction="debit"),
    Case("prd_bse_star_mf", "NACH DR BSE STAR MF 0012345", "investments", amount=5000, direction="debit"),
    Case("prd_rent_to_person", "IMPS/P2A/402166/RAMESH K/rent oct", "rent", amount=18000, direction="debit",
         history=MONTHLY_FIXED),
    # salary vs other credits
    Case("salary_neft", "NEFT CR-HDFC0000001-ACME TECHNOLOGIES PVT LTD-SALARY SEP", "salary", amount=85000,
         direction="credit"),
    Case("salary_is_salary", "NEFT CR-HDFC0000001-ACME TECHNOLOGIES PVT LTD-SALARY SEP", True, "is_salary",
         amount=85000, direction="credit"),
    Case("p2p_in_not_salary", "UPI/CR/412345678901/ANIL KUMAR/SBIN/anil.k@oksbi/dinner share", False, "is_salary",
         amount=600, direction="credit"),
    Case("p2p_in_dinner", "UPI/CR/412345678901/ANIL KUMAR/SBIN/anil.k@oksbi/dinner share", "transfers",
         amount=600, direction="credit", history=ONE_OFF),
    # obligations
    Case("emi_bajaj", "ACH D- TP ACH BAJAJFINANCE-1234567", "emi_loans", amount=4500, direction="debit"),
    Case("emi_is_emi", "ACH D- TP ACH BAJAJFINANCE-1234567", True, "is_emi", amount=4500, direction="debit"),
    Case("cc_bill_cred", "UPI/DR/412345678901/CRED/AXIS/cred.club@axisbank/payment", "credit_card_payment",
         amount=23000, direction="debit"),
    Case("bounce_charges", "NACH RTN CHGS BAJAJ FINANCE", "fees_charges", amount=590, direction="debit"),
    # cash, interest, charges
    Case("atm", "ATW-512345XXXXXX1234-S1ANMU12-MUMBAI", "cash", amount=5000, direction="debit"),
    Case("interest", "INT.PD:01-07-2026 TO 30-09-2026", "other_income", amount=812, direction="credit"),
    Case("sms_charges", "SMS CHGS QTR SEP26", "fees_charges", amount=17.7, direction="debit"),
    # merchants, known and unknown
    Case("groceries_dmart_card", "Spent Rs 1,299 on HDFC Bank Card x4321 at DMART AVENUE on 2026-10-03", "groceries"),
    Case("fuel_iocl", "POS 4591XXXXXXXX1234 INDIAN OIL PETROL PUMP", "transport", amount=2000, direction="debit"),
    Case("bill_bescom", "BIL/BPAY/001234/BESCOM", "bills_utilities", amount=1450, direction="debit"),
    Case("unknown_pharmacy", "UPI/DR/412345678901/SHREE BALAJI MEDICALS/YESB/balajimed@okaxis/medicines", "health",
         amount=340, direction="debit"),
    Case("unknown_school", "NEFT DR-SBIN0001234-ST XAVIERS PUBLIC SCHOOL-term fees", "education", amount=24000,
         direction="debit"),
    Case("vendor_traders", "NEFT DR-SBIN0004321-GANESH STEEL TRADERS-NETBANK-inv 778", "business_expense",
         amount=56000, direction="debit"),
    # refunds are credits from merchants, not purchases
    Case("refund_amazon", "UPI/CR/412345678901/AMAZON SELLER SERVICES/YESB/amazon@apl/Refund", "refunds",
         amount=899, direction="credit"),
    Case("refund_is_refund", "UPI/CR/412345678901/AMAZON SELLER SERVICES/YESB/amazon@apl/Refund", True, "is_refund",
         amount=899, direction="credit"),
    # transfers
    Case("self_transfer", "MB/TPT/OWN A/C/sweep to savings", "transfers", amount=50000, direction="debit"),
    Case("p2p_out_mummy", "UPI/DR/412345678901/MUMMY/SBIN/sunita.devi55@oksbi/mummy ko", True, "is_p2p",
         amount=3000, direction="debit", history=ONE_OFF),
]


def run_cases(bundle: StudentBundle, cases: list[Case] = CASES) -> list[dict]:
    eng = Engine(bundle)
    out = []
    for c in cases:
        qm = bundle.builtin[c.question]
        res = eng.categorize([{"text": c.text, "amount": c.amount, "direction": c.direction, "history": c.history}],
                             [qm.question])[0]
        a = res["answers"][c.question]
        out.append({"case": c.name, "expected": c.expected, "got": a["value"], "ok": a["value"] == c.expected,
                    "auto": a["auto"], "route": a["route"]})
    return out


def main(argv: list[str] | None = None) -> None:
    from ..registry import Registry

    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args(argv)
    res = run_cases(Registry(args.models_dir).load())
    for r in res:
        print(f"{'PASS' if r['ok'] else 'FAIL'}  {r['case']:24s} expected={r['expected']!s:20s} got={r['got']!s:20s} "
              f"auto={r['auto']} route={r['route']}")
    print(f"{sum(r['ok'] for r in res)}/{len(res)} passed")


if __name__ == "__main__":
    main()
