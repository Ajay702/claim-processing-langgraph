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

_DISCHARGE_FIELDS: list[str] = [
    "diagnosis",
    "admission_date",
    "discharge_date",
    "treating_physician",
    "hospital_name",
]

_DEFAULT_DISCHARGE_DATA: dict[str, Any] = {k: None for k in _DISCHARGE_FIELDS}


def _validate_discharge_data(parsed: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise the LLM output for discharge extraction.

    Ensures every expected key exists and values are strings or None.

    Args:
        parsed: Raw parsed JSON from LLM.

    Returns:
        A validated discharge dict.
    """
    result: dict[str, Any] = {}
    for key in _DISCHARGE_FIELDS:
        val = parsed.get(key)
        if val is not None and not isinstance(val, str):
            val = str(val)
        result[key] = val if val else None
    return result


def _compute_confidence(data: dict[str, Any]) -> str:
    """Determine extraction confidence based on filled fields.

    Args:
        data: Validated discharge dict.

    Returns:
        ``"high"``, ``"medium"``, or ``"low"``.
    """
    filled = sum(1 for k in _DISCHARGE_FIELDS if data.get(k) is not None)
    total = len(_DISCHARGE_FIELDS)
    ratio = filled / total
    if ratio >= 0.8:
        return "high"
    if ratio >= 0.4:
        return "medium"
    return "low"


def extract_discharge(pages: list[dict[str, Any]], page_numbers: list[int]) -> dict[str, Any]:
    """Extract structured discharge data from the given pages via LLM.

    Args:
        pages: All extracted pages.
        page_numbers: Page numbers classified as discharge summaries.

    Returns:
        Structured discharge dict with ``confidence`` field.
        Falls back to defaults on failure.
    """
    default = {**_DEFAULT_DISCHARGE_DATA, "confidence": "low"}

    if not page_numbers:
        logger.info("Discharge Agent — no discharge pages to process")
        return default

    combined_text = collect_page_texts(pages, page_numbers)
    if not combined_text.strip():
        logger.warning("Discharge Agent — discharge pages are empty")
        return default

    try:
        raw = call_llm(DISCHARGE_SYSTEM_PROMPT, combined_text)
        parsed = json.loads(raw)

        if not isinstance(parsed, dict):
            logger.warning("Discharge Agent — LLM returned non-dict JSON")
            return default

        validated = _validate_discharge_data(parsed)
        validated["confidence"] = _compute_confidence(validated)
        return validated

    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("Discharge Agent — failed to parse LLM response: %s", exc)
        return default
    except Exception as exc:
        logger.error("Discharge Agent — LLM call failed: %s", exc)
        return default


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
    try:
        discharge_data = extract_discharge(state["pages"], page_numbers)
    except Exception as exc:
        logger.exception("Discharge Agent — unhandled error for claim %s", state["claim_id"])
        discharge_data = {**_DEFAULT_DISCHARGE_DATA, "confidence": "low"}
    logger.info("Discharge Agent — claim_id=%s confidence=%s result=%s", state["claim_id"], discharge_data.get("confidence"), discharge_data)
    return {"discharge_data": discharge_data}
