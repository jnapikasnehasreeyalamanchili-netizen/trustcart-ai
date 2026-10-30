import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from config import REQUIRED_COLUMNS, VALID_STOCK


@dataclass
class Product:
    product_id: str
    title: str
    brand: str
    category: str
    sub_category: str
    description: str
    attributes: dict[str, Any]
    price_inr: int
    mrp_inr: int
    stock_status: str
    rating: float | None
    review_count: int


@dataclass
class Response:
    action: str
    answer_text: str
    cited_product_ids: list[str] = field(default_factory=list)
    cited_policy_sections: list[str] = field(default_factory=list)
    stated_prices: list[dict[str, Any]] = field(default_factory=list)
    stated_facts: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DataValidationError(ValueError):
    """Raised when the challenge files do not match the published schema."""


def load_catalogue(path: Path) -> list[Product]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Copy the official catalogue.csv into FULL_CODE/data/.")
    frame = pd.read_csv(path, dtype={"product_id": "string"})
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise DataValidationError(f"Missing catalogue columns: {sorted(missing)}")
    if len(frame) != 5000:
        raise DataValidationError(f"Expected 5,000 catalogue rows; found {len(frame)}.")
    if frame.product_id.isna().any() or frame.product_id.duplicated().any():
        raise DataValidationError("Product IDs must be present and unique.")

    result = []
    for row in frame.to_dict(orient="records"):
        attrs = json.loads(row["attributes"]) if row["attributes"] else {}
        if not isinstance(attrs, dict):
            raise DataValidationError(f"attributes must be a JSON object: {row['product_id']}")
        stock = str(row["stock_status"])
        if stock not in VALID_STOCK:
            raise DataValidationError(f"Invalid stock status for {row['product_id']}: {stock}")
        price, mrp = int(row["price_inr"]), int(row["mrp_inr"])
        if price < 0 or mrp < price:
            raise DataValidationError(f"Invalid price/MRP for {row['product_id']}")
        rating = None if pd.isna(row["rating"]) else float(row["rating"])
        result.append(Product(
            product_id=str(row["product_id"]), title=str(row["title"]),
            brand=str(row["brand"]), category=str(row["category"]),
            sub_category=str(row["sub_category"]), description=str(row["description"]),
            attributes=attrs, price_inr=price, mrp_inr=mrp, stock_status=stock,
            rating=rating, review_count=int(row["review_count"]),
        ))
    return result


def load_policy(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Copy official policy.md into FULL_CODE/data/.")
    sections: dict[str, dict[str, str]] = {}
    current_id = None
    title = ""
    lines: list[str] = []

    def save() -> None:
        if current_id:
            sections[current_id] = {"title": title, "text": "\n".join(lines).strip()}

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            save()
            heading = line[4:].strip().split(" ", 1)
            current_id = heading[0]
            title = heading[1] if len(heading) > 1 else ""
            lines = []
        elif current_id:
            lines.append(line)
    save()
    if not sections:
        raise DataValidationError("No numbered policy subsections found.")
    return sections


STOP = set("""a an the is are was of for to in on me se ka ki ke hai h hain kya kitne kitna kitni
kaunsa kaunsi konsa konsi wala wali wale price rate batao do pls please aur ya or and with i my
mujhe chahiye milega milegi hoga ho kar sakte sakti abhi toh bhi jo ye yeh woh wo what which
how much cost costs available stock show find have you do""".split())
SYNONYMS = {"tshirt": "t shirt", "facewash": "face wash", "powerbank": "power bank",
            "ladies": "women", "mens": "men", "shoe": "shoes", "litre": "l", "liter": "l"}


def tokens(text: str) -> list[str]:
    text = text.casefold().replace("-", " ").replace("_", " ")
    words = re.findall(r"[a-z0-9]+", text)
    out = []
    for word in words:
        word = SYNONYMS.get(word, word)
        out.extend(part for part in word.split() if part not in STOP)
    return out


class ProductIndex:
    def __init__(self, products: list[Product]):
        self.products = products
        self.docs = []
        df: Counter[str] = Counter()
        for product in products:
            values = []
            for key, value in product.attributes.items():
                values.extend([key, *(value if isinstance(value, list) else [value])])
            terms = set(tokens(" ".join([
                product.title, product.brand, product.category,
                product.sub_category, product.description,
                " ".join(map(str, values)),
            ])))
            self.docs.append(terms)
            df.update(terms)
        n = max(1, len(products))
        self.idf = {term: math.log((n + 1) / (freq + 0.5)) for term, freq in df.items()}

    def search(self, query: str) -> list[tuple[float, Product]]:
        query_terms = set(tokens(query))
        if not query_terms:
            return []
        scored = []
        for product, doc in zip(self.products, self.docs):
            common = query_terms & doc
            score = sum(self.idf.get(term, 0.0) for term in common)
            if score:
                scored.append((score, product))
        return sorted(scored, key=lambda item: (-item[0], item[1].product_id))


def _policy_section(query: str, policy: dict[str, dict[str, str]]) -> str | None:
    rules = [
        (("restocking",), "1.2"), (("refund", "money back"), "1.5"),
        (("exchange",), "1.6"), (("cod", "cash on delivery"), "4.2"),
        (("delivery charge", "shipping charge", "delivery fee"), "2.1"),
        (("international", "outside india"), "2.3"), (("cancel",), "5.1"),
        (("warranty",), "3.1"), (("gst", "mrp"), "7.1"), (("coupon",), "7.3"),
        (("price match",), "7.4"), (("return",), "1.1"),
        (("damaged", "defective", "wrong item"), "1.7"),
        (("tracking", "track order"), "2.4"),
        (("payment method", "upi", "debit card"), "4.1"),
    ]
    low = query.casefold()
    for phrases, section_id in rules:
        if any(phrase in low for phrase in phrases) and section_id in policy:
            return section_id
    return None


def _is_policy(query: str) -> bool:
    return bool(re.search(
        r"\b(return|refund|exchange|shipping|delivery|warranty|policy|coupon|cod|cancel|payment|gst|restocking)\b",
        query, re.I,
    ))


def _is_ooc(query: str) -> bool:
    low = query.casefold()
    if any(x in low for x in ("laptop bag", "laptop sleeve", "laptop backpack")):
        return False
    return any(x in low for x in (
        "iphone", "smartphone", "mobile phone", "laptop", "television",
        "groceries", "grocery", "medicine", "medicines", "jewellery",
        "jewelry", "vehicle", "cars",
    ))


def _budget(query: str) -> int | None:
    match = re.search(r"(?:under|below|less than|within|ke andar|andar)\s*₹?\s*([\d,]+)", query, re.I)
    return int(match.group(1).replace(",", "")) if match else None


def _attribute(query: str) -> str | None:
    patterns = {
        "water_resistant": r"waterproof|water resistant",
        "dishwasher_safe": r"dishwasher",
        "induction_compatible": r"induction",
        "paraben_free": r"paraben",
        "noise_cancellation": r"noise cancel",
        "battery_life_hours": r"battery life|battery backup",
        "microwave_safe": r"microwave",
        "bpa_free": r"bpa.?free",
        "leak_proof": r"leak.?proof",
        "warranty_months": r"warranty",
        "sizes": r"\bsize\b",
    }
    for key, pattern in patterns.items():
        if re.search(pattern, query, re.I):
            return key
    return None


def answer_query(query: str, history: list[str], products: list[Product],
                 index: ProductIndex, policy: dict[str, dict[str, str]]) -> Response:
    start = time.perf_counter()
    q = query.strip()
    if not q or len(q) > 500:
        result = Response("refuse", "Please enter a question between 1 and 500 characters.")
    elif _is_ooc(q):
        result = Response("refuse", "I can't verify that item in the supplied catalogue, so I won't invent its price or availability.")
    elif _is_policy(q):
        section_id = _policy_section(q, policy)
        if section_id:
            section = policy[section_id]
            result = Response("answer",
                f"According to policy section {section_id} — {section['title']}: {section['text']} [Policy {section_id}]",
                cited_policy_sections=[section_id])
        else:
            result = Response("refuse", "The supplied policy does not contain enough information to confirm that.")
    else:
        context = " ".join(history[-4:] + [q])
        hits = index.search(context)

        # If the shopper specifies a wattage (for example, 65W), require an
        # exact match against the catalogue's structured output_w attribute
        # or explicit wattage in the product title/description. Never show a
        # lower-wattage charger as if it satisfied the requested specification.
        wattage_match = re.search(
            r"\b(\d{1,4})\s*w(?:att(?:s|age)?)?\b",
            q,
            re.IGNORECASE,
        )
        if wattage_match:
            requested_wattage = int(wattage_match.group(1))
            matching_hits = []
            for score, product in hits:
                supported_wattages = set()

                # Prefer structured catalogue fields when present.
                for key, value in product.attributes.items():
                    if "watt" in str(key).casefold() or str(key).casefold().endswith("_w"):
                        values = value if isinstance(value, list) else [value]
                        for item in values:
                            try:
                                supported_wattages.add(int(float(str(item).strip())))
                            except (TypeError, ValueError):
                                pass

                # Also read explicit wattage specifications in title/description.
                product_text = " ".join([product.title, product.description])
                supported_wattages.update(
                    int(value) for value in re.findall(
                        r"\b(\d{1,4})\s*w(?:att(?:s|age)?)?\b",
                        product_text,
                        re.IGNORECASE,
                    )
                )

                if requested_wattage in supported_wattages:
                    matching_hits.append((score, product))
            hits = matching_hits

        if _budget(context) is not None:
            cap = _budget(context)
            hits = [(score, p) for score, p in hits if p.price_inr <= cap]
        if "in stock" in context.casefold() or "stock me" in context.casefold() or "available" in context.casefold():
            hits = [(score, p) for score, p in hits if p.stock_status != "out_of_stock"]

        colour_terms = {"blue", "navy", "teal", "black", "white", "red", "green", "pink", "purple", "brown", "yellow", "beige", "orange", "charcoal"}
        wanted_colours = colour_terms & set(tokens(context))
        if wanted_colours:
            hits = [(s, p) for s, p in hits if str(p.attributes.get("colour", "")).casefold() in wanted_colours]

        material_terms = {"cotton", "leather", "steel", "silicone", "plastic", "wood", "mesh", "fleece", "wool", "linen"}
        wanted_materials = material_terms & set(tokens(context))
        if wanted_materials:
            hits = [(s, p) for s, p in hits if all(m in str(p.attributes.get("material", "")).casefold() for m in wanted_materials)]

        if not hits or hits[0][0] < 0.2:
            result = Response("refuse", "I couldn't find a sufficiently supported match in the supplied catalogue. I won't guess.")
        else:
            low = q.casefold()
            top_score = hits[0][0]
            top = [p for score, p in hits if score == top_score]
            attr = _attribute(q)
            is_cheapest = any(word in tokens(q) for word in ("cheapest", "sasta", "sasti", "lowest"))
            is_expensive = any(word in tokens(q) for word in ("costliest", "expensive", "mehenga", "mehngi"))
            is_browse = bool(set(tokens(q)) & {"show", "options", "dikhao", "dikha", "list", "suggest", "recommend"}) or _budget(q) is not None
            asks_stock = bool(set(tokens(q)) & {"stock", "available", "milega", "milegi"}) or "stock me" in low
            asks_price = bool(set(tokens(q)) & {"price", "cost", "rate", "kitne", "kitna", "kitni", "rupees", "rs"}) or "price" in low

            if is_cheapest or is_expensive:
                pool = [p for _, p in hits]
                chosen = min(pool, key=lambda p: p.price_inr) if is_cheapest else max(pool, key=lambda p: p.price_inr)
                result = Response("answer",
                    f"Best matching {'lowest-priced' if is_cheapest else 'highest-priced'} product: {chosen.title} ({chosen.product_id}) — ₹{chosen.price_inr:,}; stock: {chosen.stock_status}. [Product {chosen.product_id}]",
                    cited_product_ids=[chosen.product_id],
                    stated_prices=[{"product_id": chosen.product_id, "price_inr": chosen.price_inr}],
                    stated_facts=[{"product_id": chosen.product_id, "field": "stock_status", "value": chosen.stock_status}])
            elif is_browse:
                selected = [p for _, p in hits[:5]]
                ids = [p.product_id for p in selected]
                result = Response("answer",
                    "Matching catalogue options:\n" + "\n".join(
                        f"- {p.title} ({p.product_id}) — ₹{p.price_inr:,}; stock: {p.stock_status} [Product {p.product_id}]"
                        for p in selected),
                    cited_product_ids=ids,
                    stated_prices=[{"product_id": p.product_id, "price_inr": p.price_inr} for p in selected],
                    stated_facts=[{"product_id": p.product_id, "field": "stock_status", "value": p.stock_status} for p in selected])
            elif attr:
                chosen = top[0]
                if attr not in chosen.attributes:
                    result = Response("refuse",
                        f"The catalogue does not list {attr} for {chosen.title} ({chosen.product_id}), so I can't confirm it.",
                        cited_product_ids=[chosen.product_id])
                else:
                    value = chosen.attributes[attr]
                    result = Response("answer",
                        f"{chosen.title} ({chosen.product_id}): {attr} = {value}. [Product {chosen.product_id}]",
                        cited_product_ids=[chosen.product_id],
                        stated_facts=[{"product_id": chosen.product_id, "field": attr, "value": value}])
            elif len(top) > 1 and ((asks_stock and len({p.stock_status for p in top}) > 1) or (asks_price and len({p.price_inr for p in top}) > 1)):
                options = top[:5]
                result = Response("clarify",
                    "Several catalogue variants match, and the requested details differ. Which one do you mean?\n" +
                    "\n".join(f"- {p.title} ({p.product_id}) — ₹{p.price_inr:,}; stock: {p.stock_status} [Product {p.product_id}]" for p in options),
                    cited_product_ids=[p.product_id for p in options],
                    stated_prices=[{"product_id": p.product_id, "price_inr": p.price_inr} for p in options],
                    stated_facts=[{"product_id": p.product_id, "field": "stock_status", "value": p.stock_status} for p in options])
            else:
                chosen = top[0]
                if asks_stock:
                    result = Response("answer",
                        f"{chosen.title} ({chosen.product_id}) stock status: {chosen.stock_status}. [Product {chosen.product_id}]",
                        cited_product_ids=[chosen.product_id],
                        stated_facts=[{"product_id": chosen.product_id, "field": "stock_status", "value": chosen.stock_status}])
                elif asks_price:
                    result = Response("answer",
                        f"{chosen.title} ({chosen.product_id}) selling price: ₹{chosen.price_inr:,}. [Product {chosen.product_id}]",
                        cited_product_ids=[chosen.product_id],
                        stated_prices=[{"product_id": chosen.product_id, "price_inr": chosen.price_inr}])
                else:
                    result = Response("answer",
                        f"I found {chosen.title} ({chosen.product_id}) in the catalogue. [Product {chosen.product_id}]",
                        cited_product_ids=[chosen.product_id])
    result.latency_ms = round((time.perf_counter() - start) * 1000, 3)
    result.tokens_in = 0
    result.tokens_out = 0
    return result
