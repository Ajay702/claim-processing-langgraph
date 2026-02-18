"""Discharge Summary Agent node — extracts discharge information."""

import json
import logging
from typing import Any

from app.graph.nodes.llm_client import call_llm, collect_page_texts
from app.graph.state import ClaimState

logger = logging.getLogger(__name__)

DISCHARGE_SYSTEM_PROMPT = """You are a medical insurance document data extractor. You will receive text from discharge summary pages of an insurance claim.

Extract ONLY the following fields from the provided text. Do NOT infer or guess values that are not explicitly present.

Required JSON output format:
{
  "diagnosis": "<string or null>",
  "admission_date": "<string or null>",
  "discharge_date": "<string or null>",
  "treating_physician": "<string or null>",
  "hospital_name": "<string or null>"
}

Rules:
- Return null for any field not explicitly found in the text.
- Do NOT hallucinate or fabricate data.
- Return ONLY the JSON object, no explanation or commentary."""

_DEFAULT_DISCHARGE_DATA: dict[str, Any] = {
    "diagnosis": None,
    "admission_date": None,
    "discharge_date": None,
    "treating_physician": None,
    "hospital_name": None,
}


def extract_discharge(pages: list[dict[str, Any]], page_numbers: list[int]) -> dict[str, Any]:
    """Extract structured discharge data from the given pages via LLM.

    Args:
        pages: All extracted pages.
        page_numbers: Page numbers classified as discharge summaries.

    Returns:
        Structured discharge dict. Falls back to defaults on failure.
    """
    if not page_numbers:
        logger.info("Discharge Agent — no discharge pages to process")
        return dict(_DEFAULT_DISCHARGE_DATA)

    combined_text = collect_page_texts(pages, page_numbers)
    if not combined_text.strip():
        logger.warning("Discharge Agent — discharge pages are empty")
        return dict(_DEFAULT_DISCHARGE_DATA)

    try:
        raw = call_llm(DISCHARGE_SYSTEM_PROMPT, combined_text)
        parsed = json.loads(raw)
        result: dict[str, Any] = {}
        for key in _DEFAULT_DISCHARGE_DATA:
            result[key] = parsed.get(key)
        return result
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("Discharge Agent — failed to parse LLM response: %s", exc)
        return dict(_DEFAULT_DISCHARGE_DATA)
    except Exception as exc:
        logger.error("Discharge Agent — LLM call failed: %s", exc)
        return dict(_DEFAULT_DISCHARGE_DATA)


def discharge_agent_node(state: ClaimState) -> dict[str, Any]:
    """Extract structured discharge summary data.

    Args:
        state: Current graph state with classified pages.

    Returns:
        A dict with the ``discharge_data`` key to merge into state.
    """
    page_numbers = state["classified_pages"].get("discharge_summary", [])
    logger.info(
        "Discharge Agent — claim_id=%s processing pages=%s",
        state["claim_id"],
        page_numbers,
    )
    discharge_data = extract_discharge(state["pages"], page_numbers)
    logger.info("Discharge Agent — claim_id=%s result=%s", state["claim_id"], discharge_data)
    return {"discharge_data": discharge_data}
