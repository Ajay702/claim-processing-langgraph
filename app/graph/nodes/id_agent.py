"""ID Agent node — extracts identity information from classified pages."""

import logging
from typing import Any

from app.graph.state import ClaimState

logger = logging.getLogger(__name__)


def id_agent_node(state: ClaimState) -> dict[str, Any]:
    """Extract structured identity data from the identity document pages.

    Stub implementation: returns mock patient identity data.

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
    # TODO: Replace with real LLM extraction in Phase 4
    id_data: dict[str, Any] = {"patient_name": "John Doe"}
    return {"id_data": id_data}
