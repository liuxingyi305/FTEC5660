#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    import os
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    receipt_reader_model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0,
    )

    system_prompt_text = """你是一个超市收据结构化提取器。请仔细阅读这张收据图片，只输出一个JSON对象，不要输出任何解释或多余文字。

字段要求：
- subtotal：收据上的SUBTOTAL金额，数字，正数。
- discounts：所有折扣、促销、优惠券行的金额列表。即使收据上显示为负数，也写成正数。例如 5% OFF -5.39 写成 5.39。
- rounding：ROUNDING行金额，数字，可以为正或负；没有就写0。
- final_payment：ROUNDING之后最终支付的金额，数字，正数。通常是OCTOPUS、CASH、VISA等支付行。

只输出JSON，不要markdown代码块，不要额外文字。"""

    receipt_extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        ("human", [
            {"type": "text", "text": "请抽取这张收据的信息。"},
            {"type": "image_url", "image_url": {"url": "{image_url}"}},
        ]),
    ])

    return receipt_extraction_prompt | receipt_reader_model



def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    grand_total_paid = Decimal("0")
    grand_total_without_discounts = Decimal("0")

    for each_receipt_path in images:
        image_url_for_model = image_data_url(each_receipt_path)
        best_fields = None
        best_discount_sum = Decimal("-1")

        for attempt_number in range(3):
            model_reply = chain.invoke({"image_url": image_url_for_model})
            raw_reply_text = response_text(model_reply)

            cleaned_reply_text = raw_reply_text.strip()
            cleaned_reply_text = re.sub(r"^```(?:json)?\s*", "", cleaned_reply_text)
            cleaned_reply_text = re.sub(r"\s*```$", "", cleaned_reply_text)

            json_object_match = re.search(r"\{.*\}", cleaned_reply_text, re.DOTALL)
            if not json_object_match:
                continue
            candidate_fields = json.loads(json_object_match.group())

            candidate_discount_sum = sum(
                abs(Decimal(str(each_discount)))
                for each_discount in candidate_fields.get("discounts", [])
            )

            if candidate_discount_sum > best_discount_sum:
                best_discount_sum = candidate_discount_sum
                best_fields = candidate_fields

        if best_fields is None:
            continue

        subtotal_amount = Decimal(str(best_fields.get("subtotal", 0)))
        discount_amounts = [
            Decimal(str(each_discount))
            for each_discount in best_fields.get("discounts", [])
        ]
        final_payment_amount = Decimal(str(best_fields.get("final_payment", 0)))

        grand_total_paid += final_payment_amount
        grand_total_without_discounts += subtotal_amount + sum(
            abs(each_discount) for each_discount in discount_amounts
        )

    return {
        QUERY_1: f"HK${grand_total_paid:.2f}",
        QUERY_2: f"HK${grand_total_without_discounts:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)



def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
