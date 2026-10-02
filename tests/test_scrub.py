import random
import re
from datetime import date

import pytest

from arth.gold import DEFAULT_GOLD, load_jsonl
from arth.prepare import prepare
from arth.scrub import hash_id, scrub
from arth.synth import FIRST, LAST, Generator


def test_placeholders():
    out = scrub("Rs 2,000 credited to a/c **4567 from Priya Nair (priya.nair@oksbi). Call 9123456789. "
                "PAN ABCDE1234F Aadhaar 1234 5678 9012 mail me at priya@example.com").text
    for leaked in ("Priya", "Nair", "4567", "9123456789", "ABCDE1234F", "1234 5678 9012", "priya@example.com",
                   "priya.nair@oksbi"):
        assert leaked.lower() not in out.lower(), leaked
    for ph in ("<NAME>", "<ACCT>", "<PHONE>", "<PAN>", "<AADHAAR>", "<VPA>", "<EMAIL>"):
        assert ph in out
    assert "2,000" in out  # amounts are kept


def test_merchants_are_kept():
    out = scrub("UPI/DR/402155/SWIGGY/YESB/swiggy@ybl/Pay").text
    assert "SWIGGY" in out and "swiggy@ybl" in out
    out = scrub("UPI/DR/615612342828/EXCITEL BROADBAND/YESB/excitel@ybl/6 month plan").text
    assert "EXCITEL BROADBAND" in out


def test_card_and_ifsc_and_refs():
    out = scrub("POS 4591XXXXXXXX1234 DMART AVENUE").text
    assert out == "POS <CARD> DMART AVENUE"
    out = scrub("NEFT CR-HDFC0000001-ACME TECHNOLOGIES PVT LTD-SALARY SEP").text
    assert "HDFC<IFSC>" in out and "ACME TECHNOLOGIES PVT LTD" in out
    assert "<NUM>" in scrub("UPI Ref 427812345678").text


def test_names_in_remarks_go_too():
    out = scrub("UPI-RAHUL SHARMA-rahul.s@okhdfcbank-HDFC0001234-412345678901-dinner split for rahul").text
    assert "rahul" not in out.lower() and "sharma" not in out.lower()


def test_kinship_words_are_signal_not_pii():
    out = scrub("UPI/DR/624412348181/MUMMY/SBIN/sunita.devi55@oksbi/mummy ko").text
    assert "MUMMY" in out and "sunita" not in out.lower()


def test_scrubbing_is_stable_for_the_student():
    for t in ["IMPS/P2A/402166/RAMESH K/rent oct",
              "UPI-RAHUL SHARMA-rahul.s@okhdfcbank-HDFC0001234-412345678901-dinner"]:
        a = prepare(t, 100, "debit")
        b = prepare(a.scrubbed, 100, "debit")
        assert a.rendered == b.rendered


@pytest.mark.parametrize("seed", range(5))
def test_generated_person_names_never_survive(seed):
    """Property: across every template, a person's name and personal VPA never survive scrubbing."""
    r = random.Random(seed)
    g = Generator(seed)
    misses = []
    for concept in ("p2p_out", "p2p_in", "rent"):
        for channel in ("upi", "imps", "aa", "netbanking", "neft"):
            for _ in range(20):
                first, last = r.choice(FIRST), r.choice([x for x in LAST if len(x) > 2])
                name = f"{first} {last}".upper() if r.random() < 0.5 else f"{first} {last}"
                vpa = g._person_vpa(f"{first} {last}")
                if channel == "neft" and concept != "p2p_in":
                    continue
                text = g.render(concept, "HDFC", channel, 1500.0, date(2026, 5, 1), name, vpa, "")
                out = scrub(text).text.lower()
                if first.lower() in out.split("<")[0] or re.search(rf"\b{first.lower()}\b", out) or vpa.lower() in out:
                    misses.append((text, out))
    assert not misses, misses[:5]


def test_gold_set_has_no_phone_or_long_numbers_after_scrub():
    for r in load_jsonl(DEFAULT_GOLD):
        out = scrub(r["text"]).text
        assert not re.search(r"(?<!\d)[6-9]\d{9}(?!\d)", out), out
        assert not re.search(r"(?<![\d,.])\d{9,}(?!\d)", out), out


def test_hash_id_is_keyed_and_stable():
    assert hash_id("a", "b") == hash_id("a", "b")
    assert hash_id("a", "b") != hash_id("ab")
    assert len(hash_id("x")) == 20
