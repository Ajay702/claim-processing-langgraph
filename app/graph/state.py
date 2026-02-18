"""Shared state definition for the claim processing LangGraph workflow."""

from typing import Any, TypedDict


class ClaimState(TypedDict):
    """Typed state passed through every node in the claim processing graph.

    Attributes:
        claim_id: Unique identifier for the insurance claim.
        pages: Page-level extracted text from the uploaded PDF.
        classified_pages: Mapping of document category to page numbers.
        id_data: Structured data extracted by the ID agent.
        discharge_data: Structured data extracted by the discharge summary agent.
        bill_data: Structured data extracted by the itemized bill agent.
        final_output: Aggregated result combining all agent outputs.
    """

    claim_id: str
    pages: list[dict[str, Any]]
    classified_pages: dict[str, list[int]]
    id_data: dict[str, Any]
    discharge_data: dict[str, Any]
    bill_data: dict[str, Any]
    final_output: dict[str, Any]
