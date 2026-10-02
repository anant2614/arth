"""Synthetic data pool: Indian bank SMS, statement narrations and AA descriptions.

Templates follow the public formats used by open-source Indian SMS parsers (bank SMS
phrasing, UPI/IMPS/NEFT/NACH narration layouts). Every line carries its concept.
This pool is for training only; it is never used to judge a model (the gold set is).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from importlib import resources
from datetime import date, timedelta

from .gazetteer import Gazetteer

BANKS = ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK", "YES", "PNB", "BOB", "IDFC", "INDUSIND",
         "CANARA", "UNION", "FEDERAL", "AU", "PAYTMPB"]
IFSC_PREFIX = {"HDFC": "HDFC", "ICICI": "ICIC", "SBI": "SBIN", "AXIS": "UTIB", "KOTAK": "KKBK", "YES": "YESB",
               "PNB": "PUNB", "BOB": "BARB", "IDFC": "IDFB", "INDUSIND": "INDB", "CANARA": "CNRB",
               "UNION": "UBIN", "FEDERAL": "FDRL", "AU": "AUBL", "PAYTMPB": "PYTM"}
PERSONAL_HANDLES = ["okhdfcbank", "okaxis", "oksbi", "okicici", "ybl", "ibl", "axl", "paytm", "apl", "naviaxis"]
_NAMES = json.loads(resources.files("arth.data").joinpath("names.json").read_text())
FIRST = [n.title() for n in _NAMES["first"]]
LAST = [n.title() for n in _NAMES["last"]] + ["K", "M", "S", "R"]
PLACES = ["MUMBAI", "PUNE", "BANGALORE", "BENGALURU", "DELHI", "NOIDA", "GURGAON", "HYDERABAD", "CHENNAI",
          "KOLKATA", "AHMEDABAD", "JAIPUR", "LUCKNOW", "INDORE", "KOCHI", "SURAT", "NAGPUR", "BHOPAL", "PATNA"]
DEITIES = ["SHREE GANESH", "SAI", "BALAJI", "LAXMI", "DURGA", "KRISHNA", "HANUMAN", "JAI AMBE", "OM SAI", "MAHALAXMI"]

UNKNOWN_MERCHANTS = {
    "food_delivery": ["{X} CLOUD KITCHEN", "{X} TIFFIN SERVICE", "{X} FOODS ONLINE"],
    "dining": ["{X} RESTAURANT", "HOTEL {X}", "{X} CAFE", "{X} DHABA", "{X} BAR AND KITCHEN", "{X} SWEETS",
               "{X} BAKERY", "UDUPI {X}", "{X} BREWPUB", "{X} BREWERY", "{X} BIRYANI HOUSE", "{X} PIZZERIA",
               "{X} BISTRO", "{X} DOSA CORNER", "{X} TEA STALL", "{X} MESS", "{X} CANTEEN", "{X} JUICE CENTRE",
               "{X} ICE CREAM PARLOUR", "{X} FOOD COURT", "{P} SOCIAL", "{X} LOUNGE", "{X} FAMILY RESTAURANT"],
    "groceries": ["{X} KIRANA STORE", "{D} GENERAL STORES", "{P} SUPER MART", "{X} PROVISION STORE",
                  "{X} FRUITS AND VEGETABLES", "{D} DAIRY", "{X} BHANDAR"],
    "shopping": ["{X} ELECTRONICS", "{X} GARMENTS", "{X} TEXTILES", "{X} FOOTWEAR", "{P} MOBILE GALLERY",
                 "{X} FASHION", "{X} HOME DECOR", "{X} OPTICALS", "{X} GIFT CENTRE", "{X} JEWELLERS",
                 "{X} FURNITURE", "{X} BOOK DEPOT", "{X} STATIONERY", "{X} TOYS", "{X} SAREES", "{X} TAILORS",
                 "{X} HARDWARE", "{X} SPORTS", "{X} WATCH CO", "{X} COLLECTION"],
    "fuel": ["{X} FILLING STATION", "{X} SERVICE STATION", "{D} PETROL PUMP", "{X} FUELS"],
    "cab_ride": ["{X} TOURS AND CABS", "{X} AUTO", "{P} CAB SERVICE"],
    "entertainment": ["{X} CINEMAS", "{X} GAMING ZONE", "{X} MULTIPLEX", "{X} BOWLING", "{X} WATER PARK",
                      "{X} AMUSEMENT PARK", "{X} EVENTS", "{X} CLUB", "{X} THEATRE", "{X} FUN WORLD"],
    "medical": ["{X} MEDICALS", "{X} PHARMACY", "{X} CLINIC", "{X} DIAGNOSTICS", "{X} HOSPITAL",
                "{D} CHEMISTS", "DR {X} DENTAL CLINIC", "{X} HOSPITALS", "{X} NURSING HOME", "{X} EYE CARE",
                "{X} PATH LAB", "{X} MEDICAL STORE", "{X} AYURVEDA", "{X} PHYSIOTHERAPY", "{X} HEALTHCARE"],
    "education": ["{X} PUBLIC SCHOOL", "{X} ACADEMY", "{X} COACHING CLASSES", "{X} INSTITUTE", "{P} UNIVERSITY",
                  "{X} INTERNATIONAL SCHOOL", "{X} COACHING", "{X} TUTORIALS", "{X} COLLEGE", "{X} VIDYALAYA",
                  "{X} CONVENT SCHOOL", "{X} LEARNING CENTRE", "{X} MUSIC CLASSES", "{X} DANCE ACADEMY"],
    "vendor_payment": ["{X} TRADERS", "{X} ENTERPRISES", "{X} AGENCIES", "{X} INDUSTRIES", "{X} SUPPLIERS",
                       "{X} WHOLESALE", "{X} PACKAGING", "{X} LOGISTICS", "{X} STEEL TRADERS", "{X} TEXTILE MILLS",
                       "{X} DISTRIBUTORS", "{X} PRINTERS", "{X} TRANSPORT CO", "{X} CHEMICALS", "{X} PLASTICS",
                       "{X} HARDWARE AND PAINTS", "{X} CONSTRUCTION", "{X} ELECTRICALS"],
    "business_income": ["{X} TECHNOLOGIES PVT LTD", "{X} SOLUTIONS LLP", "{X} RETAIL PVT LTD",
                        "{X} INFRA PROJECTS", "{X} EXPORTS"],
    "salary": ["{X} TECHNOLOGIES PVT LTD", "{X} SOFTWARE SERVICES", "{X} INDIA PVT LTD", "{X} CONSULTING LLP",
               "{X} SYSTEMS LTD", "{X} HOSPITALS LTD"],
    "insurance": ["{X} GENERAL INSURANCE", "{X} LIFE INSURANCE"],
    "telecom": ["{X} BROADBAND", "{X} CABLE NETWORK", "{X} FIBERNET", "{X} DIGITAL TV", "{X} NETWORKS"],
    "utilities": ["{P} MUNICIPAL WATER", "{X} GAS AGENCY", "{P} VIDYUT VITRAN", "{P} ELECTRICITY BOARD"],
    "travel": ["HOTEL {X} INN", "{X} TOURS AND TRAVELS", "{X} TRAVELS", "{X} RESORT", "{X} HOLIDAYS",
               "{X} TOURS", "{X} HOMESTAY", "{X} LODGE", "{X} BUS SERVICE"],
    "emi": ["{X} FINANCE LTD", "{X} FINSERV", "{X} CAPITAL LTD", "{X} HOUSING FINANCE"],
}
BRAND_WORDS = ["SUNRISE", "NEW", "GALAXY", "ROYAL", "STAR", "GREEN", "LOTUS", "KRISHNA", "GOLDEN", "SILVER",
               "PARAMOUNT", "APEX", "OM", "SAI", "BHARAT", "NATIONAL", "CITY", "METRO", "PRIME", "UNIQUE",
               "AGARWAL", "GUPTA", "MEHTA", "BALAJI", "VINAYAKA", "ANNAPURNA", "SAGAR", "RAJ", "SHIV", "JYOTI"]

REMARKS = {
    "rent": ["rent", "rent oct", "house rent", "flat rent", "kiraya", "ghar ka kiraya", "rent sep", "room rent",
             "rent for october", "monthly rent", "rent nov", "pg rent", "shop rent", ""],
    "p2p_out": ["", "", "dinner", "split", "mummy ko", "for papa", "loan return", "udhaar", "chai paani",
                "birthday gift", "trip share", "movie", "petrol share", "bhai ke liye", "pocket money", "lunch",
                "gift", "paid for tickets", "shaadi gift", "ghar kharcha", "hospital ke liye", "sent"],
    "p2p_in": ["", "", "returned", "dinner share", "from papa", "udhaar wapas", "gift", "trip", "thanks",
               "money back", "share", "mummy", "rakhi gift", "repay"],
    "self_transfer": ["self", "own account", "to savings", "transfer to self", "sweep", "fd funding", "self trf"],
    "vendor_payment": ["invoice", "inv 1043", "material", "bill no 223", "labour", "advance", "supply",
                       "purchase", "raw material", "goods", "payment against invoice", "transport charges"],
    "business_income": ["invoice", "inv", "payment", "order payment", "advance", "against bill", "settlement",
                        "consulting fee", "project payment", ""],
    "education": ["fees", "term fees", "tuition fee", "admission fee", "school fees", "exam fee", "course fee",
                  "sem fee", "bus fee", ""],
    "medical": ["medicines", "consultation", "opd", "lab test", "pharmacy", "hospital bill", ""],
    "insurance": ["premium", "policy renewal", "health insurance", "term plan", ""],
    "utilities": ["bill", "electricity bill", "bijli bill", "gas bill", "water bill", ""],
    "telecom": ["recharge", "postpaid bill", "broadband", "dth recharge", ""],
    "refund": ["Refund", "refund", "return refund", "order cancelled", "reversal", "cashback", ""],
    "salary": ["SALARY", "SAL", "SALARY FOR SEP", "SALARY SEP 2026", "SALARY CREDIT", "PAYROLL", "STIPEND",
               "SAL OCT26", "MONTHLY SALARY"],
}


def _rand_ref(r: random.Random, n: int = 12) -> str:
    return "".join(r.choice("0123456789") for _ in range(n))


def _person(r: random.Random) -> str:
    return f"{r.choice(FIRST)} {r.choice(LAST)}"


def _fmt_amount(r: random.Random, a: float) -> str:
    style = r.random()
    if a == int(a) and style < 0.4:
        return f"{int(a)}"
    if style < 0.7:
        return f"{a:.2f}"
    # Indian grouping 1,23,456.00
    s = f"{int(a)}"
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups + [tail])
    return f"{s}.{int(round((a % 1) * 100)):02d}"


def _fmt_date(r: random.Random, d: date) -> str:
    return r.choice([d.strftime("%d-%m-%y"), d.strftime("%d/%m/%Y"), d.strftime("%d-%b-%y"),
                     d.strftime("%d%b%y").upper(), d.strftime("%Y-%m-%d"), d.strftime("%d-%m-%Y"),
                     d.strftime("%d %b %Y")])


@dataclass
class Spec:
    direction: str
    amount: tuple[float, float]
    channels: list[str]
    cp: str  # merchant | person | self | institution | employer | customer | lender
    weight: float


SPECS: dict[str, Spec] = {
    "food_delivery": Spec("debit", (89, 1800), ["upi", "card", "aa"], "merchant", 5),
    "dining": Spec("debit", (60, 6000), ["upi", "card", "aa"], "merchant", 4),
    "groceries": Spec("debit", (40, 8000), ["upi", "card", "aa"], "merchant", 5),
    "shopping": Spec("debit", (149, 60000), ["upi", "card", "aa", "billpay"], "merchant", 5),
    "fuel": Spec("debit", (100, 6000), ["upi", "card", "aa"], "merchant", 3),
    "cab_ride": Spec("debit", (30, 2500), ["upi", "card", "aa"], "merchant", 3),
    "travel": Spec("debit", (300, 45000), ["upi", "card", "aa", "netbanking"], "merchant", 3),
    "entertainment": Spec("debit", (49, 2500), ["upi", "card", "aa", "nach"], "merchant", 3),
    "telecom": Spec("debit", (149, 3500), ["upi", "card", "aa", "billpay", "nach"], "merchant", 3),
    "utilities": Spec("debit", (150, 9000), ["upi", "aa", "billpay", "netbanking"], "merchant", 3),
    "rent": Spec("debit", (6000, 85000), ["upi", "imps", "neft", "aa", "netbanking"], "person", 4),
    "salary": Spec("credit", (12000, 350000), ["neft", "aa", "imps"], "employer", 4),
    "emi": Spec("debit", (999, 95000), ["nach", "aa"], "lender", 4),
    "credit_card_bill": Spec("debit", (500, 150000), ["upi", "netbanking", "aa", "billpay"], "card", 3),
    "investment": Spec("debit", (500, 100000), ["nach", "upi", "netbanking", "aa"], "merchant", 4),
    "insurance": Spec("debit", (500, 60000), ["nach", "upi", "aa", "billpay", "netbanking"], "merchant", 2),
    "tax": Spec("debit", (500, 300000), ["netbanking", "aa", "upi"], "tax", 2),
    "p2p_out": Spec("debit", (50, 40000), ["upi", "imps", "aa", "netbanking"], "person", 6),
    "p2p_in": Spec("credit", (50, 40000), ["upi", "imps", "aa", "neft"], "person", 5),
    "self_transfer": Spec("debit", (1000, 200000), ["imps", "neft", "netbanking", "upi", "aa"], "self", 2),
    "atm_cash": Spec("debit", (100, 20000), ["atm"], "institution", 3),
    "cash_deposit": Spec("credit", (500, 100000), ["cash"], "institution", 2),
    "interest": Spec("credit", (1, 25000), ["interest"], "institution", 2),
    "refund": Spec("credit", (20, 25000), ["upi", "card", "aa"], "merchant", 3),
    "bank_charges": Spec("debit", (5, 1200), ["charges"], "institution", 2),
    "bounce": Spec("debit", (150, 1200), ["bounce"], "institution", 2),
    "medical": Spec("debit", (40, 80000), ["upi", "card", "aa", "neft"], "merchant", 3),
    "education": Spec("debit", (300, 150000), ["upi", "netbanking", "aa", "card", "neft"], "merchant", 2),
    "wallet_load": Spec("debit", (100, 10000), ["upi", "card", "aa"], "merchant", 2),
    "business_income": Spec("credit", (500, 500000), ["neft", "upi", "imps", "aa", "rtgs"], "customer", 4),
    "vendor_payment": Spec("debit", (1000, 500000), ["neft", "rtgs", "imps", "upi", "aa", "netbanking"], "vendor", 4),
    "dividend": Spec("credit", (10, 50000), ["nach", "neft", "aa"], "merchant", 1),
}

LENDERS = ["Bajaj Finance", "HDB Financial", "Tata Capital", "Home Credit", "Muthoot Finance",
           "LIC Housing Finance", "Fullerton", "Kissht"]
BANK_LOAN_NAMES = ["HDFC BANK LTD LOAN", "ICICI BANK LOAN", "SBI HOME LOAN", "AXIS BANK RETAIL LOAN",
                   "KOTAK MAHINDRA PRIME", "IDFC FIRST BANK LOAN", "PERSONAL LOAN EMI", "LOAN REC"]
DIVIDEND_COS = ["INFOSYS LTD", "TCS LTD", "ITC LTD", "HDFC BANK LTD", "RELIANCE INDUSTRIES", "COAL INDIA",
                "HINDUSTAN UNILEVER", "WIPRO LTD", "NTPC LTD", "ONGC"]


class Generator:
    def __init__(self, seed: int = 0, gaz: Gazetteer | None = None, merchant_holdout: set[str] | None = None):
        self.r = random.Random(seed)
        self.gaz = gaz or Gazetteer.default()
        self.holdout = merchant_holdout or set()
        self.by_concept: dict[str, list[dict]] = {}
        for m in self.gaz.by_name.values():
            if m["name"] not in self.holdout:
                self.by_concept.setdefault(m["concept"], []).append(m)
        self.vpa_of: dict[str, list[str]] = {}
        for h, name in self.gaz.vpa_handles.items():
            self.vpa_of.setdefault(name, []).append(h)

    # ---- counterparties -------------------------------------------------
    def _unknown_merchant(self, concept: str) -> str:
        pats = UNKNOWN_MERCHANTS.get(concept) or UNKNOWN_MERCHANTS["shopping"]
        r = self.r
        return r.choice(pats).format(X=r.choice(BRAND_WORDS), D=r.choice(DEITIES), P=r.choice(PLACES))

    def _merchant(self, concept: str) -> tuple[str, str | None]:
        """(display name, vpa or None)."""
        r = self.r
        known = self.by_concept.get(concept, [])
        if known and r.random() < 0.7:
            m = r.choice(known)
            display = r.choice(m["aliases"] + [m["name"].upper()])
            handles = self.vpa_of.get(m["name"])
            vpa = None
            if handles:
                vpa = f"{r.choice(handles)}@{r.choice(['ybl', 'paytm', 'axisbank', 'icici', 'hdfcbank', 'okicici'])}"
            return display, vpa
        name = self._unknown_merchant(concept)
        slug = "".join(ch for ch in name.lower() if ch.isalnum())[:14]
        vpa = r.choice([f"paytmqr{_rand_ref(r, 10)}@paytm", f"bharatpe.{_rand_ref(r, 10)}@fbpe",
                        f"{slug}@okaxis", f"{slug}.{_rand_ref(r, 4)}@icici", f"q{_rand_ref(r, 9)}@ybl"])
        return name, vpa

    def _person_vpa(self, name: str) -> str:
        r = self.r
        first = name.split()[0].lower()
        last = name.split()[-1].lower()
        local = r.choice([f"{first}.{last}", f"{first}{r.randint(1, 999)}", f"9{_rand_ref(r, 9)}",
                          f"{first}{last[:1]}", f"{first}_{last}"])
        return f"{local}@{r.choice(PERSONAL_HANDLES)}"

    def counterparty(self, concept: str) -> tuple[str, str | None]:
        spec, r = SPECS[concept], self.r
        if concept == "emi":
            if r.random() < 0.6:
                m = self.gaz.by_name[r.choice(LENDERS)]
                return r.choice(m["aliases"]), None
            return r.choice(BANK_LOAN_NAMES + [self._unknown_merchant("emi")]), None
        if concept == "dividend":
            return r.choice(DIVIDEND_COS), None
        if concept == "credit_card_bill":
            if r.random() < 0.35:
                return "CRED", f"cred.club@axisbank"
            bank = r.choice(["HDFC", "ICICI", "SBI CARD", "AXIS", "KOTAK", "AMEX", "ONECARD", "IDFC FIRST"])
            return r.choice([f"{bank} CREDIT CARD", f"{bank} CARD PAYMENT", f"CC {bank} XX{r.randint(1000, 9999)}",
                             f"{bank} CC BILLPAY"]), None
        if concept == "tax":
            return r.choice(["INCOME TAX", "CBDT TAX PAYMENT", "GST PMT", "GSTN CHALLAN", "TIN NSDL TDS",
                             "ADVANCE TAX", "OLTAS CHALLAN 280", "MUNICIPAL CORP PROPERTY TAX", "BBMP PROPERTY TAX",
                             "SELF ASSESSMENT TAX"]), None
        if concept == "refund":  # refunds come from the merchants people buy from
            return self._merchant(r.choice(["shopping", "shopping", "food_delivery", "travel", "groceries",
                                            "entertainment", "cab_ride", "telecom", "medical"]))
        if spec.cp == "merchant":
            return self._merchant(concept)
        if spec.cp == "employer":
            return self._unknown_merchant("salary"), None
        if spec.cp == "customer":
            if r.random() < 0.3:
                m = self.gaz.by_name[r.choice(["Razorpay", "Cashfree", "PayU", "Paytm Settlement"])]
                return r.choice(m["aliases"]), None
            if r.random() < 0.3:
                name = _person(r)
                return name.upper(), self._person_vpa(name)
            return self._unknown_merchant("business_income"), None
        if spec.cp == "vendor":
            if r.random() < 0.25:  # contractors are often individuals
                name = _person(r)
                return name.upper(), self._person_vpa(name)
            name = self._unknown_merchant("vendor_payment")
            return name, f"{''.join(c for c in name.lower() if c.isalnum())[:12]}@okaxis"
        if spec.cp == "person":
            name = _person(r)
            if concept in {"p2p_out", "p2p_in"} and r.random() < 0.12:  # "MUMMY", "Bhai" as the payee label
                return r.choice(_NAMES["kin"]).upper(), self._person_vpa(name)
            return (name.upper() if r.random() < 0.6 else name), self._person_vpa(name)
        if spec.cp == "self":
            return r.choice(["SELF", "OWN A/C", "TO SELF", "SWEEP TO FD", _person(r).upper()]), None
        return "", None

    # ---- rendering ------------------------------------------------------
    def render(self, concept: str, bank: str, channel: str, amount: float, d: date, cp: str,
               vpa: str | None, remark: str) -> str:
        r = self.r
        amt = _fmt_amount(r, amount)
        ds = _fmt_date(r, d)
        acct = f"{r.randint(1000, 9999)}"
        ref = _rand_ref(r)
        ifsc = f"{IFSC_PREFIX.get(bank, 'HDFC')}0{_rand_ref(r, 6)}"
        cur = r.choice(["Rs.", "Rs", "INR", "Rs ", "INR "])
        bal = _fmt_amount(r, round(r.uniform(500, 300000), 2))
        bank_code = r.choice(["YESB", "HDFC", "ICIC", "SBIN", "UTIB", "PYTM", "KKBK"])
        direction = SPECS[concept].direction
        dr = direction == "debit"
        rem = remark or r.choice(["Pay", "UPI", "Payment from Ph", "NA", "Sent via app", ""])

        if channel == "upi":
            if dr:
                t = r.choice([
                    f"UPI/DR/{ref}/{cp}/{bank_code}/{vpa}/{rem}",
                    f"UPI-{cp}-{vpa}-{ifsc}-{ref}-{rem}",
                    f"UPI:{ref}:DR:{cp}:{vpa}:{rem}",
                    f"UPI-DR-{ref}-{cp}-{rem}",
                    f"UPI/{ref}/{cp}/{vpa}/{rem}",
                    f"UPI/P2{'M' if SPECS[concept].cp == 'merchant' else 'A'}/{ref}/{cp}/{rem}",
                    f"{cur}{amt} debited from A/c XX{acct} to VPA {vpa} on {ds}",
                    f"Sent {cur}{amt} From {bank} Bank A/C *{acct} To {cp} On {ds} Ref {ref}",
                    f"Dear UPI user A/C X{acct} debited by {amt} on date {ds} trf to {cp} Refno {ref}. -{bank}",
                    f"INR {amt} debited A/c no. XX{acct} {ds} UPI/P2M/{ref}/{cp} Not you? Call {bank}",
                    f"Sent {cur}{amt} from {bank} Bank AC X{acct} to {vpa} on {ds}.UPI Ref {ref}",
                    f"{bank} Bank Acct XX{acct} debited for {cur} {amt} on {ds}; {cp} credited. UPI:{ref}",
                    f"Your a/c no. XXXXXXXX{acct} is debited for {cur}{amt} on {ds} and credited to {vpa} (UPI Ref no {ref})",
                    f"Paid {cur}{amt} to {cp} from {bank} A/c XX{acct}. UPI Ref {ref}. Avl Bal {cur}{bal}",
                ])
            else:
                t = r.choice([
                    f"UPI/CR/{ref}/{cp}/{bank_code}/{vpa}/{rem}",
                    f"UPI-{cp}-{vpa}-{ifsc}-{ref}-{rem}",
                    f"UPI:{ref}:CR:{cp}:{vpa}:{rem}",
                    f"{cur}{amt} credited to a/c XX{acct} on {ds} by a/c linked to VPA {vpa} (UPI Ref No {ref}).",
                    f"Money Received - INR {amt} in your {bank} Bank AC X{acct} from {vpa} on {ds}.UPI Ref:{ref}",
                    f"Dear Customer, your A/c X{acct} is credited by {cur}{amt} on {ds} from {cp} UPI Ref {ref}",
                    f"Received {cur}{amt} in your A/c XX{acct} from {cp} on {ds}. UPI Ref {ref}. Avl Bal {cur}{bal}",
                ])
            if concept == "refund":
                t = r.choice([
                    f"UPI/REV/{ref}/{cp}/refund", f"REV-UPI/{ref}/{cp}/{bank_code}",
                    f"Refund of {cur}{amt} from {cp} credited to A/c XX{acct} on {ds}",
                    f"{cur}{amt} credited to A/c XX{acct} on {ds} as reversal of UPI txn to {cp} Ref {ref}",
                    f"UPI/CR/{ref}/{cp}/{bank_code}/{vpa}/Refund", f"Cashback of {cur}{amt} from {cp} credited",
                ])
            return t
        if channel == "card":
            card = f"{r.randint(400000, 559999)}XXXXXX{acct}"
            if concept == "refund":
                return r.choice([f"POS REFUND {card} {cp}", f"ECOM REFUND/{cp}/{ds}",
                                 f"{cur}{amt} refunded to your {bank} Card XX{acct} by {cp} on {ds}"])
            return r.choice([
                f"POS {card} {cp}", f"PCD/{acct}/{cp}/{r.choice(PLACES)}", f"ECOM PUR/{cp}/{ds}",
                f"Spent {cur}{amt} On {bank} Bank Card {acct} At {cp} On {ds}",
                f"Txn of INR {amt} done on Card XX{acct} at {cp} on {ds}. Avl Lmt: INR {bal}",
                f"{cur}{amt} spent on your {bank} Credit Card ending {acct} at {cp} on {ds}",
                f"VPS/{cp}/{ds}/{ref[:6]}", f"IPS/{cp}/{r.choice(PLACES)}",
            ])
        if channel == "aa":
            return r.choice([
                f"{cp} {remark}".strip(), f"PAYMENT TO {cp}" if dr else f"PAYMENT FROM {cp}",
                f"UPI {cp} {remark}".strip(), f"{cp}", f"{'TO' if dr else 'BY'} {cp} {ref[:8]}",
                f"{cp.title()} - {remark or ('purchase' if dr else 'credit')}",
            ])
        if channel == "imps":
            return r.choice([
                f"IMPS/P2A/{ref}/{cp}/{rem}", f"IMPS-{ref}-{cp}-{bank_code}-{rem}",
                f"MMT/IMPS/{ref}/{rem}/{cp}/{bank_code}",
                f"{cur}{amt} {'debited' if dr else 'credited'} from A/c XX{acct} via IMPS {'to' if dr else 'from'} {cp} Ref {ref}",
            ])
        if channel in ("neft", "rtgs"):
            ch = channel.upper()
            if dr:
                return r.choice([
                    f"{ch}/{ref}/{cp}/{rem}", f"{ch} DR-{ifsc}-{cp}-NETBANK, MUM-{ref}-{rem}",
                    f"{ch}-{ifsc}-{cp}-{rem}",
                ])
            return r.choice([
                f"{ch} CR-{ifsc}-{cp}-{rem}-{ref}", f"BY TRANSFER-{ch}*{ifsc}*{ref}*{cp}",
                f"{ch}/{ref}/{cp}/{rem}",
                f"Dear Customer, INR {amt} credited to your A/c No XX{acct} on {ds} through {ch} with UTR {ref} by {cp}",
                f"Your A/c XX{acct} is credited with INR {amt} on {ds} by {ch} from {cp}. Avl Bal INR {bal}",
            ])
        if channel == "netbanking":
            return r.choice([
                f"MB/TPT/{cp}/{rem}", f"INF/INFT/{ref}/{rem}/{cp}", f"IB FUNDS TRANSFER DR-{acct}-{cp}",
                f"TO TRANSFER-INB {cp} {rem}".strip(), f"NETBANKING/{cp}/{ref}",
                f"BIL/ONL/{ref}/{cp}/{rem}",
            ])
        if channel == "billpay":
            return r.choice([f"BIL/BPAY/{ref}/{cp}", f"BBPS/{cp}/{ref}", f"BIL/ONL/{ref}/{cp}/{rem}",
                             f"Payment of {cur}{amt} to {cp} via BBPS successful. Ref {ref}"])
        if channel == "nach":
            mandate = _rand_ref(r, r.choice([7, 10, 15]))
            if concept == "dividend":
                return r.choice([f"ACH C- {cp} DIVIDEND {mandate}", f"NACH CR {cp} DIV {d.year}",
                                 f"{cp} INTERIM DIVIDEND {d.year}"])
            return r.choice([
                f"NACH DR {cp} {mandate}", f"ACH D- {cp}-{mandate}", f"ECS/{cp}/{mandate}",
                f"Your A/c XX{acct} has been debited for {cur}{amt} on {ds} towards NACH mandate of {cp}.",
                f"AUTOPAY SI {cp} {mandate}", f"NACH-{cp}-{mandate}",
                f"{cur}{amt} debited from A/c XX{acct} on {ds} for E-Mandate {cp}",
            ])
        if channel == "atm":
            card = f"{r.randint(400000, 559999)}XXXXXX{acct}"
            term = f"S1{r.choice(['A', 'B', 'C'])}N{r.choice(['MU', 'DL', 'BL', 'HY', 'CH'])}{r.randint(10, 99)}"
            return r.choice([
                f"ATW-{card}-{term}-{r.choice(PLACES)}", f"NWD-{card}-{term}-{r.choice(PLACES)}",
                f"ATM WDL {r.choice(PLACES)} {term}",
                f"{cur}{amt} withdrawn at ATM {term} from A/c XX{acct} on {ds}. Avl Bal {cur}{bal}",
                f"CASH WDL {term} {r.choice(PLACES)}", f"EAW-{card}-{term}",
            ])
        if channel == "cash":
            return r.choice([f"CASH DEP {r.choice(PLACES)} BRANCH", f"BY CASH-{r.choice(PLACES)}",
                             f"CDM CASH DEPOSIT {term_id(r)}",
                             f"{cur}{amt} deposited in cash to A/c XX{acct} on {ds} at {r.choice(PLACES)} branch",
                             f"CASH DEPOSIT BY SELF {r.choice(PLACES)}"])
        if channel == "interest":
            return r.choice([f"INT.PD:{d.strftime('%d-%m-%Y')} TO {(d + timedelta(days=90)).strftime('%d-%m-%Y')}",
                             "CREDIT INTEREST", f"INT CR {d.strftime('%b%y').upper()}",
                             f"Interest credited {cur}{amt} to A/c XX{acct} on {ds}",
                             f"SB INTEREST {d.strftime('%b %Y')}", f"FD INT {_rand_ref(r, 8)}"])
        if channel == "charges":
            return r.choice(["SMS CHGS QTR SEP26", "DEBIT CARD ANNUAL FEE+GST", "MIN BAL CHGS",
                             "ATM WDL CHGS EXCEEDED FREE TXN", "GST ON CHGS", "CONSOLIDATED CHGS FOR A/C",
                             f"{cur}{amt} debited towards SMS alert charges from A/c XX{acct}",
                             "NON MAINTENANCE CHGS", "CHQ BOOK ISSUE CHGS", "IMPS CHG", "DC AMC CHGS",
                             f"Annual fee {cur}{amt} charged on Card XX{acct}"])
        if channel == "bounce":
            lender = self.r.choice(LENDERS + ["MUTUAL FUND SIP", "LIC PREMIUM"])
            return r.choice([f"NACH RTN CHGS {lender.upper()}", f"ECS RETURN {lender.upper()} INSUFFICIENT FUNDS",
                             "CHQ RTN CHGS", f"I/W CHQ RET-{_rand_ref(r, 6)} FUNDS INSUFFICIENT",
                             "ACH DEBIT RETURN CHARGES", f"NACH BOUNCE CHARGES {lender.upper()}",
                             f"Mandate debit of {cur}{amt} to {lender} failed due to insufficient balance. Charges levied",
                             "OUTWARD RETURN CHGS+GST", f"EMI BOUNCE CHGS {_rand_ref(r, 8)}"])
        raise ValueError(channel)

    # ---- one line -------------------------------------------------------
    def history(self, concept: str, amount_fixed: bool) -> dict | None:
        r = self.r
        if r.random() < 0.6:
            return None
        recurring = {"rent", "salary", "emi", "investment", "insurance", "entertainment", "telecom"}
        if concept in recurring:
            return {"same_counterparty_count": r.randint(3, 24),
                    "amount_cv": round(r.uniform(0, 0.02) if amount_fixed else r.uniform(0.0, 0.12), 3),
                    "median_gap_days": r.choice([30, 30, 31, 29, 28, 30.5])}
        if concept in {"p2p_out", "p2p_in"}:
            if r.random() < 0.15:  # monthly pocket money: looks like rent, isn't
                return {"same_counterparty_count": r.randint(3, 12), "amount_cv": round(r.uniform(0, 0.05), 3),
                        "median_gap_days": 30}
            return {"same_counterparty_count": r.randint(1, 6), "amount_cv": round(r.uniform(0.2, 1.2), 3),
                    "median_gap_days": r.choice([None, 4, 9, 17, 45, 63])}
        if concept in {"food_delivery", "groceries", "cab_ride", "fuel", "dining"}:
            return {"same_counterparty_count": r.randint(2, 40), "amount_cv": round(r.uniform(0.2, 0.9), 3),
                    "median_gap_days": r.choice([2, 3, 5, 7, 10])}
        return {"same_counterparty_count": r.randint(1, 4), "amount_cv": round(r.uniform(0.1, 1.0), 3),
                "median_gap_days": r.choice([None, 20, 45, 90])}

    def line(self, concept: str, idx: int) -> dict:
        r = self.r
        spec = SPECS[concept]
        bank = r.choice(BANKS)
        channel = r.choice(spec.channels)
        lo, hi = spec.amount
        amount = round(r.uniform(lo, hi) if r.random() < 0.5 else min(hi, lo * (hi / lo) ** r.random()), 2)
        if concept in {"rent", "salary", "emi", "investment", "self_transfer"} or r.random() < 0.2:
            amount = float(round(amount, -2) or amount)
        d = date(2026, 1, 1) + timedelta(days=r.randint(0, 300))
        cp, vpa = self.counterparty(concept)
        if vpa is None and channel == "upi":
            vpa = self._person_vpa(_person(r)) if spec.cp in {"person", "self"} else \
                f"{''.join(c for c in cp.lower() if c.isalnum())[:12] or 'merchant'}@ybl"
        remark = r.choice(REMARKS.get(concept, [""]))
        if concept in {"rent", "p2p_out"} and r.random() < 0.25:
            remark = ""  # genuinely ambiguous lines
        text = self.render(concept, bank, channel, amount, d, cp, vpa, remark)
        if r.random() < 0.15:
            text = text.upper()
        elif r.random() < 0.05:
            text = text.lower()
        return {
            "id": f"s{idx:06d}", "text": text, "amount": amount, "direction": spec.direction,
            "date": d.isoformat(), "bank": bank, "concept": concept,
            "history": self.history(concept, amount_fixed=concept in {"rent", "emi", "investment"}),
            "source": "synthetic",
        }

    def generate(self, n: int) -> list[dict]:
        concepts = list(SPECS)
        weights = [SPECS[c].weight for c in concepts]
        # Every concept gets at least 1% of lines, the rest by weight.
        floor = max(1, n // (100))
        plan = [c for c in concepts for _ in range(floor)]
        plan += self.r.choices(concepts, weights=weights, k=max(0, n - len(plan)))
        self.r.shuffle(plan)
        return [self.line(c, i) for i, c in enumerate(plan[:n])]


def term_id(r: random.Random) -> str:
    return f"CDM{r.randint(1000, 9999)}"


def generate(n: int, seed: int = 0, merchant_holdout: set[str] | None = None) -> list[dict]:
    return Generator(seed, merchant_holdout=merchant_holdout).generate(n)
