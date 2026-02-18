"""Itemized Bill Agent node — extracts billing information."""

import json
import logging
from typing import Any

from app.graph.nodes.llm_client import call_llm, collect_page_texts
from app.graph.state import ClaimState

logger = logging.getLogger(__name__)

BILL_SYSTEM_PROMPT = """You are a medical insurance document data extractor. You will receive text from itemized bill pages of an insurance claim.

Extract ALL line items from the bill. For each item extract:
- description: what was charged
- quantity: number of units (if not stated, assume 1)
- unit_price: price per unit
- total_price: quantity × unit_price

Then calculate "calculated_total" as the sum of all total_price values. Do NOT blindly trust any printed total on the document — always compute it yourself.

Required JSON output format:
{
  "items": [
    {
      "description": "<string>",
      "quantity": <number>,
      "unit_price": <number>,
      "total_price": <number>
    }
  ],
  "calculated_total": <number>
}

Rules:
- Extract every line item you can find.
- If quantity is missing, assume 1.
- If you cannot parse a numeric value for an item, skip that item.
- calculated_total must equal the sum of all total_price values.
- Do NOT hallucinate items not present in the text.
- Return ONLY the JSON object, no explanation or commentary."""

_DEFAULT_BILL_DATA: dict[str, Any] = {
    "items": [],
    "calculated_total": 0,
}


def _sanitise_bill(parsed: dict[str, Any]) -> dict[str, Any]:
    """Validate and recalculate bill data returned by the LLM.

    Args:
        parsed: Raw parsed JSON from LLM.

    Returns:
        Sanitised bill dict with recalculated total.
    """
    items_raw = parsed.get("items", [])
    clean_items: list[dict[str, Any]] = []

    for item in items_raw:
        try:
            description = str(item.get("description", ""))
            quantity = float(item.get("quantity", 1))
            unit_price = float(item.get("unit_price", 0))
            total_price = round(quantity * unit_price, 2)
            clean_items.append({
                "description": description,
                "quantity": quantity,
                "unit_price": unit_price,
                "total_price": total_price,
            })
        except (TypeError, ValueError) as exc:
            logger.debug("Bill Agent — skipping malformed item %s: %s", item, exc)
            continue

    calculated_total = round(sum(i["total_price"] for i in clean_items), 2)

    return {
        "items": clean_items,
        "calculated_total": calculated_total,
    }


def extract_bill(pages: list[dict[str, Any]], page_numbers: list[int]) -> dict[str, Any]:
    """Extract structured billing data from the given pages via LLM.

    Args:
        pages: All extracted pages.
        page_numbers: Page numbers classified as itemized bills.

    Returns:
        Structured bill dict. Falls back to defaults on failure.
    """
    if not page_numbers:
        logger.info("Bill Agent — no bill pages to process")
        return dict(_DEFAULT_BILL_DATA)

    combined_text = collect_page_texts(pages, page_numbers)
    if not combined_text.strip():
        logger.warning("Bill Agent — bill pages are empty")
        return dict(_DEFAULT_BILL_DATA)

    try:
        raw = call_llm(BILL_SYSTEM_PROMPT, combined_text)
        parsed = json.loads(raw)
        return _sanitise_bill(parsed)
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("Bill Agent — failed to parse LLM response: %s", exc)
        return dict(_DEFAULT_BILL_DATA)
    except Exception as exc:
        logger.error("Bill Agent — LLM call failed: %s", exc)
        return dict(_DEFAULT_BILL_DATA)


def bill_agent_node(state: ClaimState) -> dict[str, Any]:
    """Extract structured billing data from itemized bill pages.

    Args:
        state: Current graph state with classified pages.

    Returns:
        A dict with the ``bill_data`` key to merge into state.
    """
    page_numbers = state["classified_pages"].get("itemized_bill", [])
    logger.info(
        "Bill Agent — claim_id=%s processing pages=%s",
        state["claim_id"],
        page_numbers,
    )
    bill_data = extract_bill(state["pages"], page_numbers)
    logger.info("Bill Agent — claim_id=%s result=%s", state["claim_id"], bill_data)
    return {"bill_data": bill_data}
