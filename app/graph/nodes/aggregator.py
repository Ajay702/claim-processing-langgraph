"""Aggregator node — combines all agent outputs into a final result."""

import logging
from typing import Any

from app.graph.state import ClaimState

logger = logging.getLogger(__name__)


def aggregator_node(state: ClaimState) -> dict[str, Any]:
    """Merge identity, discharge, and billing data into a single output.

    Args:
        state: Current graph state with all agent outputs populated.

    Returns:
        A dict with the ``final_output`` key to merge into state.
    """
    final_output: dict[str, Any] = {
        "claim_id": state["claim_id"],
        "identity_info": state.get("id_data", {}),
        "discharge_summary": state.get("discharge_data", {}),
        "billing_details": state.get("bill_data", {}),
    }
    logger.info("Aggregator — claim_id=%s output assembled", state["claim_id"])
    return {"final_output": final_output}
