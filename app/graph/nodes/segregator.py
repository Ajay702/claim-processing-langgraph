"""Segregator node — classifies PDF pages by document type using Cerebras LLM."""

import json
import logging
from typing import Any

from app.graph.nodes.llm_client import call_llm
from app.graph.state import ClaimState

logger = logging.getLogger(__name__)

ALLOWED_TYPES: set[str] = {
    "claim_forms",
    "cheque_or_bank_details",
    "identity_document",
    "itemized_bill",
    "discharge_summary",
    "prescription",
    "investigation_report",
    "cash_receipt",
    "other",
}

SYSTEM_PROMPT = """You are an insurance document classifier. You will receive the text content of a single page from a medical insurance claim document.

Classify the page into EXACTLY ONE of the following document types:

- claim_forms: Standardized insurance claim forms (e.g., pre-authorization forms, cashless claim forms, reimbursement request forms).
- cheque_or_bank_details: Pages containing bank account information, cancelled cheques, or payment details.
- identity_document: Government-issued identification such as Aadhaar card, PAN card, passport, driving license, voter ID.
- itemized_bill: Hospital or medical bills listing individual charges, procedures, medicines, room charges with amounts.
- discharge_summary: Hospital discharge summaries containing patient history, diagnosis, treatment given, and discharge instructions.
- prescription: Doctor's prescriptions listing medicines, dosages, and instructions.
- investigation_report: Lab reports, diagnostic test results, imaging reports (X-ray, MRI, CT scan, blood work).
- cash_receipt: Payment receipts, acknowledgements of payment received, transaction confirmations.
- other: Any page that does not clearly fit into the above categories.

Respond with ONLY a JSON object in this exact format:
{"document_type": "<one_of_the_allowed_types>"}

Do not include any explanation, commentary, or additional text."""


def classify_page(text: str) -> str:
    """Classify a single page's text into a document type via the LLM.

    Args:
        text: The extracted text content of one PDF page.

    Returns:
        One of the ``ALLOWED_TYPES`` strings. Falls back to ``"other"``
        on any failure.
    """
    if not text.strip():
        logger.debug("Empty page text — defaulting to 'other'")
        return "other"

    try:
        raw = call_llm(SYSTEM_PROMPT, text)
        parsed = json.loads(raw)
        doc_type = parsed.get("document_type", "other")

        if doc_type not in ALLOWED_TYPES:
            logger.warning("LLM returned invalid type '%s' — defaulting to 'other'", doc_type)
            return "other"

        return doc_type

    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("Failed to parse LLM response: %s", exc)
        return "other"
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return "other"


def segregator_node(state: ClaimState) -> dict[str, Any]:
    """Classify each page into a document category using Cerebras LLM.

    Iterates over every page, sends its text to the LLM for
    classification, and groups page numbers by document type.

    Args:
        state: Current graph state containing extracted pages.

    Returns:
        A dict with the ``classified_pages`` key to merge into state.
    """
    pages = state["pages"]
    classified: dict[str, list[int]] = {t: [] for t in ALLOWED_TYPES}

    for page in pages:
        page_num: int = page["page_number"]
        text: str = page.get("text", "")
        doc_type = classify_page(text)
        classified[doc_type].append(page_num)
        logger.info(
            "Page %d → %s (claim_id=%s)",
            page_num,
            doc_type,
            state["claim_id"],
        )

    # Remove empty categories for a cleaner state
    classified = {k: v for k, v in classified.items() if v}

    logger.info(
        "Segregator — claim_id=%s classified=%s",
        state["claim_id"],
        classified,
    )
    return {"classified_pages": classified}
