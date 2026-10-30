#!/usr/bin/env python3
"""R3 — Grounded Shopping Assistant: official scorer.

Usage:
    python evaluate.py --submission submission/R3_predictions.jsonl --truth data/questions_dev_truth.jsonl [--out score.json]
                       [--data-dir data] [--details]

Needs only Python >= 3.9, numpy and pandas. The scorer reads the catalogue, policy and question files from
--data-dir (default: the `data/` folder next to this script) to check every claim you make.

Per question (only question_ids in the truth file are scored):
    question_score = action_correct * (W_ACTION + W_GROUNDED * grounded + W_HELPFUL * helpful)
                     - HALLUCINATED_PRICE_PENALTY * hallucinated_prices
    primary_score  = mean(question_score)        (best = 1.0; can go below 0 because of the price penalty)
See README.md ("How you're scored") for the exact definition of each term.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- constants (repeated in README)
W_ACTION = 0.4                    # weight of choosing the right action (answer / clarify / refuse)
W_GROUNDED = 0.3                  # weight of "every claim is supported by the catalogue/policy"
W_HELPFUL = 0.3                   # weight of "the answer contains what the shopper needed"
HALLUCINATED_PRICE_PENALTY = 3.0  # subtracted PER hallucinated price, on top of everything else
PRECISION_FLOOR = 0.5             # product credit = hit * (PRECISION_FLOOR + (1 - PRECISION_FLOOR) * citation precision)
MIN_CLARIFY_CANDIDATES = 2        # a clarification must cite at least this many of the matching products
PARENT_SECTION_CREDIT = 0.5       # citing "1" when "1.2" was needed earns half credit
FREE_EXTRA_POLICY_CITATIONS = 1   # citing more than (required + 1) policy sections dilutes the policy credit
MAX_CITED_PRODUCTS = 10           # citing more products than this earns no product / fact / clarify credit
MAX_COUNT = 1e9                   # sanity cap for latency_ms / tokens
CLARIFY_AS_ANSWER_CREDIT = 0.5    # action credit when you "answer" a clarify question by listing >= 2 matching products with correct prices
UNKNOWN_VALUES = {"", "unknown", "n/a", "na", "none", "null", "not specified", "not listed", "not available", "not mentioned"}
PRICE_TOLERANCE = 0.5             # rupees; prices are whole numbers
RATING_TOLERANCE = 0.05
ACTIONS = ("answer", "clarify", "refuse")
CATALOGUE_TOP_FIELDS = ("price_inr", "mrp_inr", "stock_status", "rating", "review_count", "brand", "category",
                        "sub_category", "title")
PRICE_FIELDS = ("price_inr", "mrp_inr", "discount_inr")

CURRENCY_RE = re.compile(
    r"(?:₹|₨|\brs|\binr|\brup(?:ee|ees|aye|aiye|aiya|eya|iya|ye|yon)\b)[\s.:\-]*([0-9][0-9,]*(?:\.[0-9]+)?)(k(?![a-z]))?"
    r"|([0-9][0-9,]*(?:\.[0-9]+)?)(k(?![a-z]))?\s*(?:/-|rs\b\.?|inr\b|rup[aei]\w*)",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"([0-9][0-9,]*(?:\.[0-9]+)?)(k(?![a-z]))?(\s*(?:w|watt|mm|cm|inch|in|l|ltr|litre|liter|ml|kg|g|mah|"
                       r"pcs|pieces?|gb|ton|saal|years?|y|spf|x|tc)\b)?", re.IGNORECASE)
SECTION_RE = re.compile(r"^\s*(?:§|section|sec\.?)?\s*(\d+(?:\.\d+)?)(?:\s+[^\d]*)?$", re.IGNORECASE)
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
PRICE_ALIASES = re.compile(r"price|mrp|cost|rate|amount", re.IGNORECASE)


class FormatError(Exception):
    pass


# ----------------------------------------------------------------------------- loading
def _num(s: str, k: str | None) -> float:
    v = float(s.replace(",", ""))
    return v * 1000 if k else v


def load_catalogue(data_dir: Path) -> dict:
    path = data_dir / "catalogue.csv"
    if not path.exists():
        sys.exit(f"ERROR: cannot find {path}. Run evaluate.py from the R3 package folder or pass --data-dir.")
    df = pd.read_csv(path, dtype={"product_id": str}, keep_default_na=True)
    cat = {}
    for r in df.itertuples(index=False):
        attrs = json.loads(r.attributes) if isinstance(r.attributes, str) and r.attributes else {}
        rating = None if (r.rating is None or (isinstance(r.rating, float) and math.isnan(r.rating))) else float(r.rating)
        cat[r.product_id] = {
            "price_inr": float(r.price_inr), "mrp_inr": float(r.mrp_inr), "stock_status": r.stock_status,
            "rating": rating, "review_count": int(r.review_count), "brand": r.brand, "category": r.category,
            "sub_category": r.sub_category, "title": r.title, "attributes": attrs,
        }
    return cat


def load_policy(data_dir: Path) -> tuple[set, set]:
    path = data_dir / "policy.md"
    if not path.exists():
        sys.exit(f"ERROR: cannot find {path}.")
    text = path.read_text(encoding="utf-8")
    sections = set(re.findall(r"^#{2,3}\s+(\d+(?:\.\d+)?)", text, flags=re.MULTILINE))
    amounts = {_num(m.group(1), None) for m in re.finditer(r"₹\s?([0-9][0-9,]*)", text)}
    return sections, amounts


def load_questions(data_dir: Path) -> dict:
    out = {}
    for name in ("questions_dev.csv", "questions_test.csv"):
        p = data_dir / name
        if p.exists():
            df = pd.read_csv(p, dtype=str)
            for r in df.itertuples(index=False):
                out[r.question_id] = (r.conversation_id, int(r.turn), r.question_text)
    return out


def load_jsonl(path: Path, what: str) -> list:
    if not path.is_file():
        raise FormatError(f"{what} file not found (or not a file): {path}")
    raw = path.read_bytes()
    try:
        if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
            text = raw.decode("utf-16")
        else:
            text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise FormatError(f"{what} file is not UTF-8 text. Save it as UTF-8 (in Python: open(path, 'w', encoding='utf-8')).")
    rows = []
    if True:
        for i, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise FormatError(f"{what} line {i} is not valid JSON ({e.msg}). Each line must be one JSON object.")
            except (ValueError, RecursionError) as e:
                raise FormatError(f"{what} line {i} could not be parsed ({type(e).__name__}: {str(e)[:80]}). Check for absurdly long numbers or nesting.")
            if not isinstance(obj, dict):
                raise FormatError(f"{what} line {i} must be a JSON object {{...}}, got {type(obj).__name__}.")
            rows.append((i, obj))
    return rows


# ----------------------------------------------------------------------------- validation
def _is_num(x) -> bool:
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return False
    try:
        return math.isfinite(float(x)) and abs(float(x)) < 1e15
    except OverflowError:
        return False


def validate_submission(rows: list) -> dict:
    errors, seen = [], {}

    def err(i, msg):
        errors.append(f"line {i}: {msg}")

    for i, o in rows:
        try:
            json.dumps(o, ensure_ascii=False).encode("utf-8")
        except (UnicodeEncodeError, ValueError, OverflowError):
            err(i, "contains characters or numbers that cannot be encoded (broken unicode / huge numbers).")
            continue
        qid = o.get("question_id")
        if not isinstance(qid, str) or not qid.strip():
            err(i, "'question_id' must be a non-empty string, e.g. \"Q-AB12CD34\".")
            continue
        if qid != qid.strip():
            err(i, f"question_id {qid!r} has leading/trailing spaces. Copy IDs exactly from questions_test.csv.")
            continue
        if qid in seen:
            err(i, f"duplicate question_id {qid!r} (first seen on line {seen[qid]}). Keep exactly one line per question.")
            continue
        seen[qid] = i
        for key in ("action", "answer_text", "cited_product_ids", "cited_policy_sections", "stated_prices",
                    "latency_ms", "tokens_in", "tokens_out"):
            if key not in o:
                err(i, f"missing required key {key!r} (question {qid}).")
        if o.get("action") not in ACTIONS:
            err(i, f"'action' must be one of {list(ACTIONS)} (lower case), got {o.get('action')!r}.")
        if "answer_text" in o and not isinstance(o["answer_text"], str):
            err(i, "'answer_text' must be a string.")
        for key in ("cited_product_ids", "cited_policy_sections"):
            v = o.get(key, [])
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                err(i, f"{key!r} must be a list of strings, e.g. [\"SKU-1A2B3C4D\"] or [\"1.2\"] (use [] when empty).")
        sp = o.get("stated_prices", [])
        if not isinstance(sp, list):
            err(i, "'stated_prices' must be a list of {\"product_id\": str, \"price_inr\": number} (use [] when empty).")
        else:
            for j, p in enumerate(sp):
                if not (isinstance(p, dict) and isinstance(p.get("product_id"), str) and _is_num(p.get("price_inr"))):
                    err(i, f"stated_prices[{j}] must look like {{\"product_id\": \"SKU-...\", \"price_inr\": 1299}}.")
        sf = o.get("stated_facts", [])
        if not isinstance(sf, list):
            err(i, "'stated_facts' (optional) must be a list of {\"product_id\", \"field\", \"value\"} objects.")
        else:
            for j, p in enumerate(sf):
                ok = isinstance(p, dict) and isinstance(p.get("product_id"), str) and isinstance(p.get("field"), str) and "value" in p
                if ok and isinstance(p["value"], (dict,)):
                    ok = False
                if not ok:
                    err(i, f"stated_facts[{j}] must look like {{\"product_id\": \"SKU-...\", \"field\": \"stock_status\", \"value\": \"in_stock\"}}.")
        for key in ("latency_ms",):
            if key in o and not (_is_num(o[key]) and 0 <= o[key] <= MAX_COUNT):
                err(i, f"{key!r} must be a number between 0 and {MAX_COUNT:g} (milliseconds).")
        for key in ("tokens_in", "tokens_out"):
            if key in o and not (_is_num(o[key]) and 0 <= o[key] <= MAX_COUNT and float(o[key]).is_integer()):
                err(i, f"{key!r} must be a whole number >= 0 (use 0 if you do not use an LLM).")
        if len(errors) >= 15:
            break
    if errors:
        raise FormatError("\n  ".join(["Your submission has format problems:"] + errors[:15]) +
                          "\n  Fix these and re-run. See README.md section 'What to submit' and sample_submission.jsonl.")
    return {o["question_id"]: o for _, o in rows}


def validate_truth(rows: list) -> list:
    need = ("question_id", "expected_action", "acceptable_product_ids", "required_facts", "clarify_candidate_ids",
            "policy_sections")
    out = []
    for i, o in rows:
        for k in need:
            if k not in o:
                raise FormatError(f"truth line {i}: missing {k!r} — is this really an R3 truth file?")
        out.append(o)
    return out


# ----------------------------------------------------------------------------- scoring helpers
def norm_section(s: str) -> str | None:
    m = SECTION_RE.match(s or "")
    return m.group(1) if m else None


def canon_field(field: str) -> str:
    f = field.strip().lower()
    if f in PRICE_FIELDS:
        return f
    if "discount" in f:
        return "discount_inr"
    if PRICE_ALIASES.search(f):
        return "mrp_inr" if "mrp" in f else "price_inr"
    return field.strip()


def _norm_str(x) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower().replace("_", " "))


def value_matches(stated, truth, field: str) -> bool:
    if isinstance(truth, list):
        pool = {_norm_str(t) for t in truth}
        if isinstance(stated, list):
            return len(stated) > 0 and all(_norm_str(s) in pool for s in stated)
        return _norm_str(stated) in pool
    if isinstance(truth, bool):
        s = _norm_str(stated) if not isinstance(stated, bool) else ("true" if stated else "false")
        if s in ("true", "yes", "1", "y"):
            return truth is True
        if s in ("false", "no", "0", "n"):
            return truth is False
        return False
    if isinstance(truth, (int, float)) and not isinstance(truth, bool):
        if isinstance(stated, bool):
            return False
        if isinstance(stated, (list, dict)) or stated is None:
            return False
        if _is_num(stated):
            v = float(stated)
        else:
            nums = re.findall(r"-?[0-9][0-9,]*(?:\.[0-9]+)?", str(stated))
            if len(nums) != 1:
                return False
            try:
                v = float(nums[0].replace(",", ""))
            except (ValueError, OverflowError):
                return False
        tol = RATING_TOLERANCE if field == "rating" else (PRICE_TOLERANCE if field in PRICE_FIELDS else 1e-6)
        return abs(v - float(truth)) <= tol
    if truth is None:
        return False
    return _norm_str(stated) == _norm_str(truth)


def catalogue_value(cat: dict, pid: str, field: str):
    p = cat[pid]
    if field == "discount_inr":
        return p["mrp_inr"] - p["price_inr"]
    if field in CATALOGUE_TOP_FIELDS:
        return p[field]
    return p["attributes"].get(field, None)


def normalise_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").translate(DEVANAGARI_DIGITS)


def text_amounts(text: str) -> list:
    out = []
    for m in CURRENCY_RE.finditer(normalise_text(text)):
        if m.group(1):
            out.append(_num(m.group(1), m.group(2)))
        elif m.group(3):
            out.append(_num(m.group(3), m.group(4)))
    return out


def question_numbers(texts: list) -> set:
    nums = set()
    for t in texts:
        for m in NUMBER_RE.finditer(normalise_text(t)):
            if m.group(3):  # a measurement (65W, 17 inch, 3 litre ...), not a budget
                continue
            try:
                nums.add(_num(m.group(1), m.group(2)))
                nums.add(_num(m.group(1), None))
            except ValueError:
                pass
    return nums


def product_credit(cited: list, acceptable: set) -> float:
    if not acceptable:
        return 1.0
    cited_set = list(dict.fromkeys(cited))
    if len(cited_set) > MAX_CITED_PRODUCTS:
        return 0.0
    hits = [c for c in cited_set if c in acceptable]
    if not hits:
        return 0.0
    precision = len(hits) / len(cited_set)
    return PRECISION_FLOOR + (1 - PRECISION_FLOOR) * precision


def policy_credit(cited: list, required: list) -> float:
    if not required:
        return 1.0
    c = [s for s in (norm_section(x) for x in cited) if s]
    c = list(dict.fromkeys(c))
    credit = []
    for r in required:
        alts = r.split("|")  # "7.5|1.1" = either section is fully correct
        if any(x in c for x in alts):
            credit.append(1.0)
        elif any(x.split(".")[0] in c for x in alts):
            credit.append(PARENT_SECTION_CREDIT)
        else:
            credit.append(0.0)
    score = float(np.mean(credit))
    allowed = len(required) + FREE_EXTRA_POLICY_CITATIONS
    if len(c) > allowed:
        score *= allowed / len(c)
    return score


def score_one(t: dict, p: dict | None, cat: dict, sections: set, policy_amounts: set, conv_texts: list) -> dict:
    res = {"question_id": t["question_id"], "expected_action": t["expected_action"],
           "question_type": t.get("question_type", ""), "predicted_action": None, "action_correct": 0,
           "grounded": 0, "helpful": 0.0, "hallucinated_prices": 0, "score": 0.0, "issues": []}
    if p is None:
        res["issues"].append("missing prediction")
        return res
    res["predicted_action"] = p["action"]
    cited = list(dict.fromkeys(p.get("cited_product_ids", [])))
    cited_set = set(cited)
    issues = res["issues"]
    grounded = True
    hp = 0
    for c in cited:
        if c not in cat:
            grounded = False
            issues.append(f"cited product {c} does not exist")
    for s in p.get("cited_policy_sections", []):
        ns = norm_section(s)
        if ns is None or ns not in sections:
            grounded = False
            issues.append(f"cited policy section {s!r} does not exist")
    # --- prices (structured)
    for sp in p.get("stated_prices", []):
        pid, val = sp["product_id"], float(sp["price_inr"])
        if pid not in cat:
            hp += 1
            issues.append(f"HALLUCINATED PRICE: {pid} does not exist")
        elif pid not in cited_set:
            hp += 1
            issues.append(f"HALLUCINATED PRICE: price stated for {pid} but it is not in cited_product_ids")
        elif abs(val - cat[pid]["price_inr"]) > PRICE_TOLERANCE:
            hp += 1
            issues.append(f"HALLUCINATED PRICE: {pid} stated {val:g}, catalogue says {cat[pid]['price_inr']:g}")
    # --- other facts (structured)
    for sf in p.get("stated_facts", []):
        pid, field, val = sf["product_id"], canon_field(sf["field"]), sf["value"]
        if val is None or (isinstance(val, str) and _norm_str(val) in UNKNOWN_VALUES):
            continue  # "unknown / not listed" is not a claim
        is_price = field in PRICE_FIELDS
        if pid not in cat:
            grounded = False
            hp += int(is_price)
            issues.append(f"fact about non-existent product {pid}")
            continue
        if pid not in cited_set:
            grounded = False
            hp += int(is_price)
            issues.append(f"fact about {pid} which is not in cited_product_ids")
            continue
        truth_val = catalogue_value(cat, pid, field)
        if truth_val is None:
            grounded = False
            issues.append(f"unsupported claim: catalogue has no {field!r} for {pid}")
        elif not value_matches(val, truth_val, field):
            grounded = False
            hp += int(is_price)
            issues.append(f"{'HALLUCINATED PRICE' if is_price else 'wrong fact'}: {pid}.{field} stated {val!r}, catalogue says {truth_val!r}")
    # --- prices written in free text
    allowed = question_numbers(conv_texts)
    if p.get("cited_policy_sections"):
        allowed |= set(policy_amounts)
    for c in cited:
        if c in cat:
            pr, mrp = cat[c]["price_inr"], cat[c]["mrp_inr"]
            allowed |= {pr, mrp, mrp - pr}
    bad = sorted({a for a in text_amounts(p.get("answer_text", "")) if not any(abs(a - x) <= PRICE_TOLERANCE for x in allowed)})
    for a in bad:
        hp += 1
        issues.append(f"HALLUCINATED PRICE in answer_text: ₹{a:g} is not the price/MRP of any cited product")
    if hp:
        grounded = False
    if p["action"] in ("answer", "clarify") and not cited and not p.get("cited_policy_sections"):
        grounded = False
        issues.append("no citation: an answer/clarification must cite at least one product ID or policy section")
    res["hallucinated_prices"] = hp
    res["claims_valid"] = int(grounded)
    a = float(p["action"] == t["expected_action"])
    too_many = len(cited) > MAX_CITED_PRODUCTS
    if too_many:
        issues.append(f"cited {len(cited)} products (> {MAX_CITED_PRODUCTS}): no product/fact/clarify credit")
    empty_text = not (p.get("answer_text") or "").strip()
    cand = set(t["clarify_candidate_ids"])
    # partial credit: expected clarify, team answered but listed >= 2 of the matching products with correct prices
    if (not a and t["expected_action"] == "clarify" and p["action"] == "answer" and not too_many and not hp and not empty_text):
        priced = {sp["product_id"] for sp in p.get("stated_prices", []) if sp["product_id"] in cand and sp["product_id"] in cited_set}
        if len(priced) >= MIN_CLARIFY_CANDIDATES:
            a = CLARIFY_AS_ANSWER_CREDIT
            issues.append(f"answered instead of clarifying, but listed {len(priced)} matching products with correct prices: action credit {a}")
    res["action_correct"] = a
    # --- helpfulness (only meaningful when the action is right)
    helpful = 0.0
    if a and empty_text and t["expected_action"] != "refuse":
        issues.append("empty answer_text: the shopper sees nothing, so no helpfulness credit")
    if a and not (empty_text and t["expected_action"] != "refuse"):
        if t["expected_action"] == "refuse":
            helpful = 1.0
        elif t["expected_action"] == "clarify":
            hits = [c for c in cited if c in cand]
            if len(hits) >= MIN_CLARIFY_CANDIDATES and not too_many:
                helpful = PRECISION_FLOOR + (1 - PRECISION_FLOOR) * len(hits) / len(cited)
            else:
                issues.append(f"clarification cited {len(hits)} of the matching products (need >= {MIN_CLARIFY_CANDIDATES})")
        else:
            comps = []
            if t["acceptable_product_ids"]:
                pc = product_credit(cited, set(t["acceptable_product_ids"]))
                comps.append(pc)
                if pc == 0:
                    issues.append("none of the right products was cited")
            req_fields = {}
            for rf in t["required_facts"]:
                req_fields.setdefault(rf["product_id"], set()).add(rf["field"])
            for rf in t["required_facts"]:
                pid, field, val = rf["product_id"], rf["field"], rf["value"]
                ok = False
                if field == "price_inr":
                    ok = any(sp["product_id"] == pid and abs(float(sp["price_inr"]) - float(val)) <= PRICE_TOLERANCE
                             for sp in p.get("stated_prices", []))
                ok = ok or any(sf["product_id"] == pid and canon_field(sf["field"]) == field and
                               (value_matches(sf["value"], val, field) or
                                (isinstance(sf["value"], list) and pid in cat and value_matches(sf["value"], catalogue_value(cat, pid, field), field)
                                 and any(_norm_str(x) == _norm_str(val) for x in sf["value"])))
                               for sf in p.get("stated_facts", []))
                # a correct discount (MRP - price) answers a "kitna discount" question that requires price + MRP
                if not ok and field in ("price_inr", "mrp_inr") and {"price_inr", "mrp_inr"} <= req_fields[pid] and pid in cat:
                    ok = any(sf["product_id"] == pid and canon_field(sf["field"]) == "discount_inr" and
                             value_matches(sf["value"], cat[pid]["mrp_inr"] - cat[pid]["price_inr"], "discount_inr")
                             for sf in p.get("stated_facts", []))
                ok = ok and pid in cited_set and not too_many
                comps.append(1.0 if ok else 0.0)
                if not ok:
                    issues.append(f"required fact not stated: {pid}.{field}")
            if t["policy_sections"]:
                pc = policy_credit(p.get("cited_policy_sections", []), t["policy_sections"])
                comps.append(pc)
                if pc < 1:
                    issues.append(f"policy credit {pc:.2f} (needed {t['policy_sections']})")
            helpful = float(np.mean(comps)) if comps else 1.0
    if not a:
        issues.append(f"wrong action: expected {t['expected_action']}, got {p['action']}")
    # grounded credit needs at least one RELEVANT claim: irrelevant citations are not "grounded answers"
    if grounded and t["expected_action"] != "refuse" and helpful <= 0:
        grounded = False
        issues.append("no relevant citation (none of the cited products/sections is what the question needed): no grounded credit")
    res["grounded"] = int(grounded)
    res["helpful"] = round(helpful, 6)
    res["score"] = a * (W_ACTION + W_GROUNDED * int(grounded) + W_HELPFUL * helpful) - HALLUCINATED_PRICE_PENALTY * hp
    return res


def prf(results: list, label: str) -> dict:
    tp = sum(1 for r in results if r["predicted_action"] == label and r["expected_action"] == label)
    fp = sum(1 for r in results if r["predicted_action"] == label and r["expected_action"] != label)
    fn = sum(1 for r in results if r["predicted_action"] != label and r["expected_action"] == label)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "support": tp + fn}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="R3 Grounded Shopping Assistant scorer")
    ap.add_argument("--submission", required=True, help="path to R3_predictions.jsonl")
    ap.add_argument("--truth", required=True, help="path to truth JSONL (questions_dev_truth.jsonl for practice)")
    ap.add_argument("--out", default="score.json", help="where to write the JSON report (default score.json)")
    ap.add_argument("--data-dir", default=None, help="folder with catalogue.csv, policy.md, questions_*.csv (default: ./data next to this script)")
    ap.add_argument("--details", action="store_true", help="also write per-question results (score_details.jsonl) for debugging")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parent / "data"
    try:
        truth = validate_truth(load_jsonl(Path(args.truth), "truth"))
        sub = validate_submission(load_jsonl(Path(args.submission), "submission"))
    except FormatError as e:
        print(f"FORMAT ERROR\n  {e}", file=sys.stderr)
        return 2
    if truth and not (set(sub) & {t["question_id"] for t in truth}):
        print("FORMAT ERROR\n  None of your question_ids match the truth file. Copy question_id values exactly (same case, "
              "no spaces) from the questions CSV you were scored on (dev truth -> questions_dev.csv, test -> questions_test.csv).",
              file=sys.stderr)
        return 2
    cat = load_catalogue(data_dir)
    sections, policy_amounts = load_policy(data_dir)
    questions = load_questions(data_dir)
    conv_turns: dict = {}
    for qid, (cid, turn, text) in questions.items():
        conv_turns.setdefault(cid, []).append((turn, text))

    results = []
    for t in truth:
        qid = t["question_id"]
        if qid in questions:
            cid, turn, _ = questions[qid]
            texts = [tx for tn, tx in conv_turns[cid] if tn <= turn]
        else:
            texts = [t.get("question_text", "")]
        results.append(score_one(t, sub.get(qid), cat, sections, policy_amounts, texts))

    n = len(results)
    missing = [r["question_id"] for r in results if r["predicted_action"] is None]
    scores = np.array([r["score"] for r in results], dtype=float)
    primary = float(scores.mean()) if n else 0.0
    labels = list(ACTIONS)
    confusion = {e: {pa: sum(1 for r in results if r["expected_action"] == e and r["predicted_action"] == pa)
                     for pa in labels + [None]} for e in labels}
    for e in confusion:
        confusion[e]["missing"] = confusion[e].pop(None)
    answered = [r for r in results if r["predicted_action"] is not None]
    per_type = {}
    for r in results:
        per_type.setdefault(r["question_type"] or "all", []).append(r["score"])
    lat = np.array([float(sub[r["question_id"]]["latency_ms"]) for r in answered]) if answered else np.array([0.0])
    tin = np.array([float(sub[r["question_id"]]["tokens_in"]) for r in answered]) if answered else np.array([0.0])
    tout = np.array([float(sub[r["question_id"]]["tokens_out"]) for r in answered]) if answered else np.array([0.0])
    hall_total = int(sum(r["hallucinated_prices"] for r in results))
    report = {
        "primary_score": round(primary, 6),
        "primary_score_range": "best 1.0; 0 = nothing useful; negative = hallucinated prices",
        "n_truth_questions": n,
        "n_missing_predictions": len(missing),
        "missing_examples": missing[:10],
        "components": {
            "action_accuracy": round(float(np.mean([r["action_correct"] for r in results])) if n else 0.0, 4),
            "grounded_rate_when_action_correct": round(float(np.mean([r["grounded"] for r in results if r["action_correct"]])) if any(r["action_correct"] for r in results) else 0.0, 4),
            "grounded_rate_all_answered": round(float(np.mean([r["grounded"] for r in answered])) if answered else 0.0, 4),
            "mean_helpfulness_when_action_correct": round(float(np.mean([r["helpful"] for r in results if r["action_correct"]])) if any(r["action_correct"] for r in results) else 0.0, 4),
            "hallucinated_prices_total": hall_total,
            "questions_with_hallucinated_price": int(sum(1 for r in results if r["hallucinated_prices"])),
            "price_penalty_total": round(HALLUCINATED_PRICE_PENALTY * hall_total, 4),
            "refusal": prf(results, "refuse"),
            "clarify": prf(results, "clarify"),
            "answer": prf(results, "answer"),
        },
        "confusion_expected_vs_predicted": confusion,
        "score_by_question_type": {k: round(float(np.mean(v)), 4) for k, v in sorted(per_type.items())},
        "latency_ms": {"mean": round(float(lat.mean()), 2), "p50": round(float(np.percentile(lat, 50)), 2),
                       "p95": round(float(np.percentile(lat, 95)), 2), "max": round(float(lat.max()), 2)},
        "tokens": {"mean_in": round(float(tin.mean()), 2), "mean_out": round(float(tout.mean()), 2),
                   "total_in": int(tin.sum()), "total_out": int(tout.sum())},
        "constants": {"W_ACTION": W_ACTION, "W_GROUNDED": W_GROUNDED, "W_HELPFUL": W_HELPFUL, "CLARIFY_AS_ANSWER_CREDIT": CLARIFY_AS_ANSWER_CREDIT,
                      "HALLUCINATED_PRICE_PENALTY": HALLUCINATED_PRICE_PENALTY, "PRECISION_FLOOR": PRECISION_FLOOR,
                      "MIN_CLARIFY_CANDIDATES": MIN_CLARIFY_CANDIDATES, "PARENT_SECTION_CREDIT": PARENT_SECTION_CREDIT,
                      "FREE_EXTRA_POLICY_CITATIONS": FREE_EXTRA_POLICY_CITATIONS, "MAX_CITED_PRODUCTS": MAX_CITED_PRODUCTS,
                      "PRICE_TOLERANCE": PRICE_TOLERANCE, "RATING_TOLERANCE": RATING_TOLERANCE},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    if args.details:
        det = Path(args.out).with_name("score_details.jsonl")
        with open(det, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=True) + "\n")

    c = report["components"]
    print("=" * 64)
    print(f"R3 Grounded Shopping Assistant — score on {n} questions")
    print("=" * 64)
    print(f"PRIMARY SCORE            : {primary:.4f}   (best 1.0, can be negative)")
    print(f"Action accuracy          : {c['action_accuracy']:.3f}")
    print(f"Refusal  P/R/F1          : {c['refusal']['precision']:.3f} / {c['refusal']['recall']:.3f} / {c['refusal']['f1']:.3f}")
    print(f"Clarify  P/R/F1          : {c['clarify']['precision']:.3f} / {c['clarify']['recall']:.3f} / {c['clarify']['f1']:.3f}")
    print(f"Grounded (action right)  : {c['grounded_rate_when_action_correct']:.3f}")
    print(f"Helpfulness (action right): {c['mean_helpfulness_when_action_correct']:.3f}")
    print(f"Hallucinated prices      : {hall_total}  (penalty -{c['price_penalty_total']:.1f} total, "
          f"-{c['price_penalty_total'] / max(n, 1):.4f} on the mean)")
    print(f"Latency ms mean/p95      : {report['latency_ms']['mean']} / {report['latency_ms']['p95']}")
    print(f"Tokens mean in/out       : {report['tokens']['mean_in']} / {report['tokens']['mean_out']}")
    print("Score by question type   :")
    for k, v in report["score_by_question_type"].items():
        print(f"   {k:28s} {v:+.3f}  (n={len(per_type[k])})")
    if missing:
        print(f"\nWARNING: {len(missing)} truth questions have NO prediction (scored 0). First few: {missing[:5]}")
    print(f"\nJSON report written to {args.out}" + ("  (+ score_details.jsonl)" if args.details else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
