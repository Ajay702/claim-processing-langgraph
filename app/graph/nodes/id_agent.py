"""ID Agent node — extracts identity information from classified pages."""

import json
import logging
from typing import Any

from app.graph.nodes.llm_client import call_llm, collect_page_texts
from app.graph.state import ClaimState

logger = logging.getLogger(__name__)

ID_SYSTEM_PROMPT = """You are a medical insurance document data extractor. You will receive text from identity document pages of an insurance claim.

Extract ONLY the following fields from the provided text. Do NOT infer or guess values that are not explicitly present.

Required JSON output format:
{
  "patient_name": "<string or null>",
  "date_of_birth": "<string or null>",
  "policy_number": "<string or null>",
  "member_id": "<string or null>",
  "insurance_provider": "<string or null>"
}

Rules:
- Return null for any field not explicitly found in the text.
- Do NOT hallucinate or fabricate data.
- Return ONLY the JSON object, no explanation or commentary."""

_ID_FIELDS: list[str] = [
    "patient_name",
    "date_of_birth",
    "policy_number",
    "member_id",
    "insurance_provider",
]

_DEFAULT_ID_DATA: dict[str, Any] = {k: None for k in _ID_FIELDS}


def _validate_id_data(parsed: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise the LLM output for identity extraction.

    Ensures every expected key exists and values are strings or None.

    Args:
        parsed: Raw parsed JSON from LLM.

    Returns:
        A validated identity dict.
    """
    result: dict[str, Any] = {}
    for key in _ID_FIELDS:
        val = parsed.get(key)
        if val is not None and not isinstance(val, str):
            val = str(val)
        result[key] = val if val else None
    return result


def _compute_confidence(data: dict[str, Any]) -> str:
    """Determine extraction confidence based on filled fields.

    Args:
        data: Validated identity dict.

    Returns:
        ``"high"``, ``"medium"``, or ``"low"``.
    """
    filled = sum(1 for k in _ID_FIELDS if data.get(k) is not None)
    total = len(_ID_FIELDS)
    ratio = filled / total
    if ratio >= 0.8:
        return "high"
    if ratio >= 0.4:
        return "medium"
    return "low"


def extract_identity(pages: list[dict[str, Any]], page_numbers: list[int]) -> dict[str, Any]:
    """Extract structured identity data from the given pages via LLM.

    Args:
        pages: All extracted pages.
        page_numbers: Page numbers classified as identity documents.

    Returns:
        Structured identity dict with ``confidence`` field.
        Falls back to defaults on failure.
    """
    default = {**_DEFAULT_ID_DATA, "confidence": "low"}

    if not page_numbers:
        logger.info("ID Agent — no identity pages to process")
        return default

    combined_text = collect_page_texts(pages, page_numbers)
    if not combined_text.strip():
        logger.warning("ID Agent — identity pages are empty")
        return default

    try:
        raw = call_llm(ID_SYSTEM_PROMPT, combined_text)
        parsed = json.loads(raw)

        if not isinstance(parsed, dict):
            logger.warning("ID Agent — LLM returned non-dict JSON")
            return default

        validated = _validate_id_data(parsed)
        validated["confidence"] = _compute_confidence(validated)
        return validated

    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("ID Agent — failed to parse LLM response: %s", exc)
        return default
    except Exception as exc:
        logger.error("ID Agent — LLM call failed: %s", exc)
        return default


def id_agent_node(state: ClaimState) -> dict[str, Any]:
    """Extract structured identity data from the identity document pages.

    Args:
        state: Current graph state with classified pages.

    Returns:
        A dict with the ``id_data`` key to merge into state.
    """
    page_numbers = state["classified_pages"].get("identity_document", [])
    logger.info(
        "ID Agent — claim_id=%s processing pages=%s",
        state["claim_id"],
        page_numbers,
    )
    try:
        id_data = extract_identity(state["pages"], page_numbers)
    except Exception as exc:
        logger.exception("ID Agent — unhandled error for claim %s", state["claim_id"])
        id_data = {**_DEFAULT_ID_DATA, "confidence": "low"}
    logger.info("ID Agent — claim_id=%s confidence=%s result=%s", state["claim_id"], id_data.get("confidence"), id_data)
    return {"id_data": id_data}
