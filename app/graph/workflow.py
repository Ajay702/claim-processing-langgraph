"""LangGraph workflow definition for the claim processing pipeline."""

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.aggregator import aggregator_node
from app.graph.nodes.bill_agent import bill_agent_node
from app.graph.nodes.discharge_agent import discharge_agent_node
from app.graph.nodes.id_agent import id_agent_node
from app.graph.nodes.segregator import segregator_node
from app.graph.state import ClaimState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

graph_builder = StateGraph(ClaimState)

# Nodes
graph_builder.add_node("segregator", segregator_node)
graph_builder.add_node("id_agent", id_agent_node)
graph_builder.add_node("discharge_agent", discharge_agent_node)
graph_builder.add_node("bill_agent", bill_agent_node)
graph_builder.add_node("aggregator", aggregator_node)

# Edges
graph_builder.add_edge(START, "segregator")
graph_builder.add_edge("segregator", "id_agent")
graph_builder.add_edge("segregator", "discharge_agent")
graph_builder.add_edge("segregator", "bill_agent")
graph_builder.add_edge("id_agent", "aggregator")
graph_builder.add_edge("discharge_agent", "aggregator")
graph_builder.add_edge("bill_agent", "aggregator")
graph_builder.add_edge("aggregator", END)

# Compile once at module level
workflow = graph_builder.compile()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_claim_workflow(
    claim_id: str,
    pages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Execute the full claim processing graph.

    Args:
        claim_id: Unique identifier for the claim.
        pages: Page-level extracted text from the uploaded PDF.

    Returns:
        The ``final_output`` dict produced by the aggregator node.
    """
    initial_state: ClaimState = {
        "claim_id": claim_id,
        "pages": pages,
        "classified_pages": {},
        "id_data": {},
        "discharge_data": {},
        "bill_data": {},
        "final_output": {},
    }

    logger.info("Workflow started — claim_id=%s pages=%d", claim_id, len(pages))
    result = workflow.invoke(initial_state)
    logger.info("Workflow completed — claim_id=%s", claim_id)

    return result["final_output"]
