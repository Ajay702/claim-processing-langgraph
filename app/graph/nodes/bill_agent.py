"""Itemized Bill Agent node — extracts billing information."""

import logging
from typing import Any

from app.graph.state import ClaimState

logger = logging.getLogger(__name__)


def bill_agent_node(state: ClaimState) -> dict[str, Any]:
    """Extract structured billing data from itemized bill pages.

    Stub implementation: returns mock billing data.

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
    # TODO: Replace with real LLM extraction in Phase 4
    bill_data: dict[str, Any] = {"total_amount": 1000}
    return {"bill_data": bill_data}
