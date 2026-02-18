"""Segregator node — classifies PDF pages by document type."""

import logging
from typing import Any

from app.graph.state import ClaimState

logger = logging.getLogger(__name__)


def segregator_node(state: ClaimState) -> dict[str, Any]:
    """Classify each page into a document category.

    Stub implementation: assigns the first page as an identity document,
    the second page (if present) as a discharge summary, and all
    remaining pages as itemized bill pages.

    Args:
        state: Current graph state containing extracted pages.

    Returns:
        A dict with the ``classified_pages`` key to merge into state.
    """
    pages = state["pages"]
    total = len(pages)

    classified: dict[str, list[int]] = {
        "identity_document": [],
        "discharge_summary": [],
        "itemized_bill": [],
    }

    if total >= 1:
        classified["identity_document"].append(1)
    if total >= 2:
        classified["discharge_summary"].append(2)
    if total >= 3:
        classified["itemized_bill"] = list(range(3, total + 1))

    logger.info(
        "Segregator — claim_id=%s classified=%s",
        state["claim_id"],
        classified,
    )
    return {"classified_pages": classified}
