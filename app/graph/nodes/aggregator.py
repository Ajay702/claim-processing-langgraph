"""Aggregator node — combines all agent outputs into a final result."""

import logging
from datetime import datetime, timezone
from typing import Any

from app.graph.state import ClaimState

logger = logging.getLogger(__name__)


def aggregator_node(state: ClaimState) -> dict[str, Any]:
    """Merge identity, discharge, and billing data into a single output.

    Includes processing metadata with page counts, classified types,
    and an ISO-8601 timestamp.

    Args:
        state: Current graph state with all agent outputs populated.

    Returns:
        A dict with the ``final_output`` key to merge into state.
    """
    classified = state.get("classified_pages", {})

    final_output: dict[str, Any] = {
        "claim_id": state["claim_id"],
        "identity_info": state.get("id_data", {}),
        "discharge_summary": state.get("discharge_data", {}),
        "billing_details": state.get("bill_data", {}),
        "processing_metadata": {
            "total_pages": len(state.get("pages", [])),
            "classified_types": sorted(classified.keys()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    logger.info(
        "Aggregator — claim_id=%s total_pages=%d types=%s",
        state["claim_id"],
        final_output["processing_metadata"]["total_pages"],
        final_output["processing_metadata"]["classified_types"],
    )
    return {"final_output": final_output}
