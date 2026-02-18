"""Shared Cerebras LLM client for agent nodes."""

import os
from typing import Any

from cerebras.cloud.sdk import Cerebras

_client: Cerebras | None = None


def get_cerebras_client() -> Cerebras:
    """Return a cached Cerebras client, initialised on first call.

    Raises:
        RuntimeError: If CEREBRAS_API_KEY is not set.
    """
    global _client
    if _client is None:
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise RuntimeError("CEREBRAS_API_KEY environment variable is not set.")
        _client = Cerebras(api_key=api_key)
    return _client


def call_llm(system_prompt: str, user_content: str) -> str:
    """Send a chat completion request to Cerebras and return raw content.

    Args:
        system_prompt: The system-level instruction.
        user_content: The user-level input text.

    Returns:
        The raw string content from the LLM response.

    Raises:
        RuntimeError: If the API call fails or returns no content.
    """
    client = get_cerebras_client()
    response = client.chat.completions.create(
        model="gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        top_p=1,
        stream=False,
    )
    return response.choices[0].message.content.strip()


def collect_page_texts(
    pages: list[dict[str, Any]],
    page_numbers: list[int],
) -> str:
    """Combine text from specific page numbers into a single string.

    Args:
        pages: All extracted pages with ``page_number`` and ``text`` keys.
        page_numbers: The 1-indexed page numbers to include.

    Returns:
        Combined text separated by page markers.
    """
    page_map = {p["page_number"]: p["text"] for p in pages}
    sections: list[str] = []
    for num in sorted(page_numbers):
        text = page_map.get(num, "")
        if text.strip():
            sections.append(f"--- Page {num} ---\n{text}")
    return "\n\n".join(sections)
