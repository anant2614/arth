"""Fine-grained transaction concepts and the built-in taxonomies derived from them.

Every training and gold line carries one *concept*. Caller-facing taxonomies are
views over concepts: a mapping from concept to option. This lets one labelled
narration be labelled under many taxonomies (PRD: "varied taxonomies") without
re-annotating, and lets us hold out whole taxonomies to test zero-shot transfer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CONCEPTS: dict[str, str] = {
    "food_delivery": "Food ordered online for delivery from apps like Swiggy or Zomato",
    "dining": "Eating out at restaurants, cafes, bars or dhabas",
    "groceries": "Supermarkets, kirana stores and quick-commerce grocery apps",
    "shopping": "Online and retail shopping for clothes, electronics and household goods",
    "fuel": "Petrol, diesel or CNG at a fuel station",
    "cab_ride": "Local commute: cabs, autos, bike taxis, metro and FASTag tolls",
    "travel": "Flights, trains, intercity buses and hotel stays",
    "entertainment": "Streaming subscriptions, movies, events and gaming",
    "telecom": "Mobile recharge, postpaid, broadband and DTH bills",
    "utilities": "Electricity, piped gas, water and cooking gas bills",
    "rent": "Monthly house or shop rent paid to a landlord or rent platform",
    "salary": "Salary credited by an employer",
    "emi": "Fixed loan instalment debited to a lender",
    "credit_card_bill": "Payment towards a credit card bill",
    "investment": "Mutual fund SIP, stocks, NPS, PPF, gold or fixed deposit",
    "insurance": "Life, health or motor insurance premium",
    "tax": "Income tax, advance tax, GST, TDS or property tax payment",
    "p2p_out": "Money sent to a friend or family member",
    "p2p_in": "Money received from a friend or family member",
    "self_transfer": "Transfer between the account holder's own accounts",
    "atm_cash": "Cash withdrawn from an ATM",
    "cash_deposit": "Cash deposited at a branch or cash deposit machine",
    "interest": "Interest credited on savings or deposits",
    "refund": "Refund, reversal or cashback from a merchant",
    "bank_charges": "Bank fees: SMS charges, card fees, minimum balance penalty",
    "bounce": "Cheque, NACH or ECS return and its penalty",
    "medical": "Pharmacy, hospital, clinic or diagnostics",
    "education": "School or college fees, coaching and online courses",
    "wallet_load": "Top-up of a prepaid wallet",
    "business_income": "Payment received from a customer or a payment gateway settlement",
    "vendor_payment": "Payment to a supplier, vendor or contractor for business",
    "dividend": "Dividend credited by a company",
}

CONCEPT_LIST: list[str] = list(CONCEPTS)


@dataclass(frozen=True)
class BuiltinTaxonomy:
    """A choice taxonomy defined as a view over concepts."""

    name: str
    options: dict[str, str]  # option -> description
    mapping: dict[str, str]  # concept -> option
    heldout: bool = False  # never used to train the general student

    def label(self, concept: str) -> str:
        return self.mapping[concept]


@dataclass(frozen=True)
class BuiltinBool:
    name: str
    description: str
    positives: frozenset[str] = field(default_factory=frozenset)
    heldout: bool = False

    def label(self, concept: str) -> bool:
        return concept in self.positives


def _invert(groups: dict[str, list[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for option, concepts in groups.items():
        for c in concepts:
            assert c in CONCEPTS, c
            assert c not in out, f"{c} mapped twice"
            out[c] = option
    missing = set(CONCEPTS) - set(out)
    assert not missing, f"unmapped concepts: {missing}"
    return out


PERSONAL_FINANCE = BuiltinTaxonomy(
    name="personal_finance",
    options={
        "food_dining": "Food delivery, restaurants and cafes",
        "groceries": "Groceries and daily essentials",
        "shopping": "Shopping for clothes, electronics and household items",
        "transport": "Fuel, cabs, autos and local commute",
        "travel": "Flights, trains, buses and hotels",
        "entertainment": "Subscriptions, movies and leisure",
        "bills_utilities": "Electricity, gas, water, mobile and internet bills",
        "rent": "House rent",
        "salary": "Salary from employer",
        "other_income": "Business receipts, interest and dividends",
        "emi_loans": "Loan EMIs",
        "credit_card_payment": "Credit card bill payments",
        "investments": "Mutual funds, stocks and deposits",
        "insurance": "Insurance premiums",
        "taxes": "Tax payments",
        "transfers": "Transfers to or from people, own accounts and wallets",
        "cash": "ATM withdrawals and cash deposits",
        "fees_charges": "Bank charges, penalties and bounces",
        "health": "Medical and pharmacy",
        "education": "Fees and courses",
        "refunds": "Refunds and cashbacks",
        "business_expense": "Payments to suppliers and vendors",
    },
    mapping=_invert({
        "food_dining": ["food_delivery", "dining"],
        "groceries": ["groceries"],
        "shopping": ["shopping"],
        "transport": ["fuel", "cab_ride"],
        "travel": ["travel"],
        "entertainment": ["entertainment"],
        "bills_utilities": ["telecom", "utilities"],
        "rent": ["rent"],
        "salary": ["salary"],
        "other_income": ["business_income", "dividend", "interest"],
        "emi_loans": ["emi"],
        "credit_card_payment": ["credit_card_bill"],
        "investments": ["investment"],
        "insurance": ["insurance"],
        "taxes": ["tax"],
        "transfers": ["p2p_out", "p2p_in", "self_transfer", "wallet_load"],
        "cash": ["atm_cash", "cash_deposit"],
        "fees_charges": ["bank_charges", "bounce"],
        "health": ["medical"],
        "education": ["education"],
        "refunds": ["refund"],
        "business_expense": ["vendor_payment"],
    }),
)

# Held out from student training: tests bring-your-own-taxonomy transfer.
GST_LEDGER = BuiltinTaxonomy(
    name="gst_ledger",
    heldout=True,
    options={
        "sales_receipts": "Sales and receipts from customers or payment gateways",
        "purchases": "Purchases from suppliers, vendors and contractors",
        "office_expenses": "Office supplies, equipment and general shopping",
        "rent_expense": "Rent for office, shop or premises",
        "salary_wages": "Salaries and wages",
        "bank_charges": "Bank charges, fees and bounce penalties",
        "interest_dividend_income": "Interest and dividend income",
        "loan_repayment": "Loan EMIs and credit card repayments",
        "telephone_internet": "Telephone, mobile and internet expenses",
        "power_fuel_utilities": "Electricity, gas and water",
        "travel_conveyance": "Travel, conveyance, fuel and local transport",
        "staff_welfare": "Meals, refreshments and groceries for staff",
        "duties_taxes": "GST, TDS, income tax and other government dues",
        "insurance_expense": "Insurance premiums",
        "investments": "Investments, deposits and securities",
        "drawings": "Owner's personal withdrawals, transfers and personal spends",
        "other_receipts": "Capital introduced, cash deposits, refunds and other receipts",
    },
    mapping=_invert({
        "sales_receipts": ["business_income"],
        "purchases": ["vendor_payment"],
        "office_expenses": ["shopping"],
        "rent_expense": ["rent"],
        "salary_wages": ["salary"],
        "bank_charges": ["bank_charges", "bounce"],
        "interest_dividend_income": ["interest", "dividend"],
        "loan_repayment": ["emi", "credit_card_bill"],
        "telephone_internet": ["telecom"],
        "power_fuel_utilities": ["utilities"],
        "travel_conveyance": ["travel", "cab_ride", "fuel"],
        "staff_welfare": ["food_delivery", "dining", "groceries"],
        "duties_taxes": ["tax"],
        "insurance_expense": ["insurance"],
        "investments": ["investment"],
        "drawings": ["p2p_out", "self_transfer", "atm_cash", "wallet_load",
                     "entertainment", "medical", "education"],
        "other_receipts": ["p2p_in", "cash_deposit", "refund"],
    }),
)

LENDER_FLAGS = BuiltinTaxonomy(
    name="lender_flags",
    options={
        "income": "Regular income such as salary or business receipts",
        "obligation": "Debt obligations such as EMIs and card bills",
        "bounce": "Returned cheque or failed auto-debit and penalty",
        "cash": "Cash withdrawal or deposit",
        "savings": "Investments and insurance",
        "transfer": "Transfers with people or own accounts",
        "spend": "Everyday spending",
        "other": "Interest, refunds, taxes and fees",
    },
    mapping=_invert({
        "income": ["salary", "business_income"],
        "obligation": ["emi", "credit_card_bill"],
        "bounce": ["bounce"],
        "cash": ["atm_cash", "cash_deposit"],
        "savings": ["investment", "insurance"],
        "transfer": ["p2p_out", "p2p_in", "self_transfer", "wallet_load"],
        "spend": ["food_delivery", "dining", "groceries", "shopping", "fuel", "cab_ride",
                  "travel", "entertainment", "telecom", "utilities", "rent", "medical",
                  "education", "vendor_payment"],
        "other": ["interest", "dividend", "refund", "tax", "bank_charges"],
    }),
)

COARSE = BuiltinTaxonomy(
    name="coarse",
    options={
        "income": "Money coming in as earnings",
        "expense": "Spending on goods, services and bills",
        "transfer": "Moving money between people or accounts",
        "investment": "Saving and investing",
        "debt": "Repaying loans and cards",
    },
    mapping=_invert({
        "income": ["salary", "business_income", "interest", "dividend", "refund", "p2p_in"],
        "expense": ["food_delivery", "dining", "groceries", "shopping", "fuel", "cab_ride",
                    "travel", "entertainment", "telecom", "utilities", "rent", "medical",
                    "education", "vendor_payment", "tax", "bank_charges", "bounce", "insurance"],
        "transfer": ["p2p_out", "self_transfer", "wallet_load", "atm_cash", "cash_deposit"],
        "investment": ["investment"],
        "debt": ["emi", "credit_card_bill"],
    }),
)

BUILTIN_TAXONOMIES: dict[str, BuiltinTaxonomy] = {
    t.name: t for t in (PERSONAL_FINANCE, GST_LEDGER, LENDER_FLAGS, COARSE)
}

BUILTIN_BOOLS: dict[str, BuiltinBool] = {
    b.name: b
    for b in (
        BuiltinBool("is_salary", "Salary credited by an employer", frozenset({"salary"})),
        BuiltinBool("is_emi", "Loan EMI or instalment debit", frozenset({"emi"})),
        BuiltinBool("is_p2p", "Transfer with an individual person",
                    frozenset({"p2p_out", "p2p_in"})),
        BuiltinBool("is_refund", "Refund, reversal or cashback", frozenset({"refund"})),
        BuiltinBool("is_bounce", "Bounced cheque or failed auto-debit", frozenset({"bounce"})),
        BuiltinBool("is_investment", "Investment into mutual funds, stocks or deposits",
                    frozenset({"investment"})),
        BuiltinBool("is_cash", "Cash withdrawal or deposit",
                    frozenset({"atm_cash", "cash_deposit"})),
        BuiltinBool("is_tax", "Tax payment to the government", frozenset({"tax"})),
        BuiltinBool("is_rent", "House rent payment", frozenset({"rent"}), heldout=True),
        BuiltinBool("is_business_income", "Payment received from a customer for business",
                    frozenset({"business_income"}), heldout=True),
    )
}

# Paraphrase pool: alternative descriptions per concept, used when generating varied
# training taxonomies so the general student learns meaning rather than one wording.
PARAPHRASES: dict[str, list[str]] = {
    "food_delivery": ["Online food orders", "Swiggy, Zomato and food apps", "Meals delivered home"],
    "dining": ["Restaurants and cafes", "Eating out", "Hotels, dhabas and bars"],
    "groceries": ["Grocery shopping", "Kirana and supermarket", "Daily essentials and vegetables"],
    "shopping": ["Online shopping", "Clothing, gadgets and home items", "E-commerce and retail purchases"],
    "fuel": ["Petrol and diesel", "Fuel station", "Vehicle fuel and CNG"],
    "cab_ride": ["Cabs and autos", "Local commute and tolls", "Uber, Ola, Rapido, metro"],
    "travel": ["Flights, trains and hotels", "Trips and holidays", "Intercity travel bookings"],
    "entertainment": ["Movies and streaming", "OTT subscriptions and games", "Leisure and fun"],
    "telecom": ["Mobile and internet", "Phone recharge and broadband", "DTH and telecom bills"],
    "utilities": ["Electricity and water bills", "Utility bills", "Power, gas and water"],
    "rent": ["House rent", "Monthly rent to landlord", "Rent for flat or shop"],
    "salary": ["Salary", "Monthly pay from employer", "Wages credited by company"],
    "emi": ["Loan EMI", "Instalment to lender", "Loan repayment debit"],
    "credit_card_bill": ["Credit card bill", "Card dues payment", "Paying off credit card"],
    "investment": ["Investments", "Mutual funds and stocks", "SIP, NPS, PPF and deposits"],
    "insurance": ["Insurance premium", "Life and health insurance", "Policy premiums"],
    "tax": ["Taxes", "Government tax payments", "GST, TDS and income tax"],
    "p2p_out": ["Sent to friends or family", "Money given to a person", "Personal transfer out"],
    "p2p_in": ["Received from friends or family", "Money from a person", "Personal transfer in"],
    "self_transfer": ["Own account transfer", "Moving money between my accounts", "Self transfer"],
    "atm_cash": ["ATM withdrawal", "Cash taken out", "Cash from ATM"],
    "cash_deposit": ["Cash deposit", "Cash paid into account", "CDM deposit"],
    "interest": ["Interest earned", "Savings interest", "Deposit interest credit"],
    "refund": ["Refunds", "Reversals and cashback", "Money returned by merchant"],
    "bank_charges": ["Bank charges", "Fees levied by bank", "Service charges and penalties"],
    "bounce": ["Bounced payments", "Cheque or auto-debit return", "Dishonour and return charges"],
    "medical": ["Medical expenses", "Doctor, pharmacy and hospital", "Healthcare"],
    "education": ["Education", "School and college fees", "Courses and tuition"],
    "wallet_load": ["Wallet top-up", "Adding money to wallet", "Prepaid wallet load"],
    "business_income": ["Business receipts", "Customer payments", "Sales collections"],
    "vendor_payment": ["Supplier payments", "Vendor and contractor bills", "Business purchases"],
    "dividend": ["Dividends", "Dividend income", "Company dividend credit"],
}
