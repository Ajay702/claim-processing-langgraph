"""Discharge Summary Agent node — extracts discharge information."""

import logging
from typing import Any

from app.graph.state import ClaimState

logger = logging.getLogger(__name__)


def discharge_agent_node(state: ClaimState) -> dict[str, Any]:
    """Extract structured discharge summary data.

    Stub implementation: returns mock diagnosis data.

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
    # TODO: Replace with real LLM extraction in Phase 4
    discharge_data: dict[str, Any] = {"diagnosis": "Sample Diagnosis"}
    return {"discharge_data": discharge_data}
