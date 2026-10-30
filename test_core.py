from trustcart.core import Product, ProductIndex, answer_query


def product(pid="SKU-ONE", price=999, stock="in_stock", attrs=None):
    return Product(pid, "Runvex Aero Running Shoes - Blue", "Runvex", "Footwear",
                   "Running Shoes", "Running shoes for training.",
                   attrs if attrs is not None else {"colour": "blue", "water_resistant": False},
                   price, price + 100, stock, 4.0, 10)


def test_refuses_out_of_catalogue_phone():
    items = [product()]
    result = answer_query("Do you sell iPhones?", [], items, ProductIndex(items), {})
    assert result.action == "refuse"
    assert result.stated_prices == []


def test_answers_explicit_false_attribute():
    items = [product()]
    result = answer_query("Runvex Aero water resistant?", [], items, ProductIndex(items), {})
    assert result.action == "answer"
    assert result.stated_facts[0]["value"] is False


def test_missing_attribute_is_unknown_not_false():
    items = [product(attrs={"colour": "blue"})]
    result = answer_query("Runvex Aero water resistant?", [], items, ProductIndex(items), {})
    assert result.action == "refuse"


def test_different_price_variants_clarify():
    items = [product("SKU-ONE", 999), product("SKU-TWO", 1299)]
    result = answer_query("Runvex Aero running shoes price?", [], items, ProductIndex(items), {})
    assert result.action == "clarify"


def test_tokens_zero_without_llm():
    items = [product()]
    result = answer_query("Runvex Aero price?", [], items, ProductIndex(items), {})
    assert result.tokens_in == 0 and result.tokens_out == 0
