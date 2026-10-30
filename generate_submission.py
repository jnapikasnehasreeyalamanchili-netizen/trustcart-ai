import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from config import CATALOGUE, DEV_QUESTIONS, POLICY, TEST_QUESTIONS
from trustcart.core import ProductIndex, answer_query, load_catalogue, load_policy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission"
LOGGER = logging.getLogger(__name__)


def generate(dev: bool = False) -> Path:
    products = load_catalogue(CATALOGUE)
    index = ProductIndex(products)
    policy = load_policy(POLICY)
    questions_path = DEV_QUESTIONS if dev else TEST_QUESTIONS
    questions = pd.read_csv(questions_path, dtype=str).fillna("")
    required = {"question_id", "conversation_id", "turn", "question_text"}
    if not required.issubset(questions.columns):
        raise ValueError(f"Question file missing: {sorted(required - set(questions.columns))}")
    questions["turn_num"] = questions["turn"].astype(int)
    questions = questions.sort_values(["conversation_id", "turn_num"], kind="stable")
    histories: dict[str, list[str]] = {}
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / ("R3_dev_predictions.jsonl" if dev else "R3_predictions.jsonl")
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in questions.itertuples(index=False):
            history = histories.setdefault(row.conversation_id, [])
            response = answer_query(row.question_text, history, products, index, policy)
            item = {"question_id": row.question_id, **response.to_dict()}
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            history.append(row.question_text)
    LOGGER.info("Generated %d records: %s", len(questions), destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    try:
        print(generate(dev=args.dev))
    except (FileNotFoundError, ValueError, OSError) as error:
        logging.basicConfig(level=logging.INFO)
        LOGGER.error("Generation failed: %s", error)
        raise SystemExit(1)
