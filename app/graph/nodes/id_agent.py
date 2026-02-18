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

_DEFAULT_ID_DATA: dict[str, Any] = {
    "patient_name": None,
    "date_of_birth": None,
    "policy_number": None,
    "member_id": None,
    "insurance_provider": None,
}


def extract_identity(pages: list[dict[str, Any]], page_numbers: list[int]) -> dict[str, Any]:
    """Extract structured identity data from the given pages via LLM.

    Args:
        pages: All extracted pages.
        page_numbers: Page numbers classified as identity documents.

    Returns:
        Structured identity dict. Falls back to defaults on failure.
    """
    if not page_numbers:
        logger.info("ID Agent — no identity pages to process")
        return dict(_DEFAULT_ID_DATA)

    combined_text = collect_page_texts(pages, page_numbers)
    if not combined_text.strip():
        logger.warning("ID Agent — identity pages are empty")
        return dict(_DEFAULT_ID_DATA)

    try:
        raw = call_llm(ID_SYSTEM_PROMPT, combined_text)
        parsed = json.loads(raw)
        # Ensure all expected keys exist
        result: dict[str, Any] = {}
        for key in _DEFAULT_ID_DATA:
            result[key] = parsed.get(key)
        return result
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("ID Agent — failed to parse LLM response: %s", exc)
        return dict(_DEFAULT_ID_DATA)
    except Exception as exc:
        logger.error("ID Agent — LLM call failed: %s", exc)
        return dict(_DEFAULT_ID_DATA)


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
    id_data = extract_identity(state["pages"], page_numbers)
    logger.info("ID Agent — claim_id=%s result=%s", state["claim_id"], id_data)
    return {"id_data": id_data}
