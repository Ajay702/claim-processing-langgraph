"""Unit tests for the segregator node (Phase 4)."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.graph.nodes.segregator import (
    ALLOWED_TYPES,
    classify_page,
    segregator_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(content: str) -> MagicMock:
    """Build a mock Cerebras chat completion response."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    response = SimpleNamespace(choices=[choice])
    return response


def _make_state(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a minimal ClaimState dict for testing."""
    return {
        "claim_id": "TEST-001",
        "pages": pages,
        "classified_pages": {},
        "id_data": {},
        "discharge_data": {},
        "bill_data": {},
        "final_output": {},
    }


# ---------------------------------------------------------------------------
# classify_page — valid types
# ---------------------------------------------------------------------------

class TestClassifyPageValidTypes:
    """LLM returns a valid document type → should be returned as-is."""

    @pytest.mark.parametrize("doc_type", sorted(ALLOWED_TYPES))
    @patch("app.graph.nodes.segregator._get_client")
    def test_returns_valid_type(self, mock_get_client: MagicMock, doc_type: str) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(
            f'{{"document_type": "{doc_type}"}}'
        )
        mock_get_client.return_value = mock_client

        result = classify_page("some page text")
        assert result == doc_type


# ---------------------------------------------------------------------------
# classify_page — empty / whitespace input
# ---------------------------------------------------------------------------

class TestClassifyPageEmptyInput:
    """Empty or whitespace-only text → should return 'other' without LLM call."""

    @patch("app.graph.nodes.segregator._get_client")
    def test_empty_string(self, mock_get_client: MagicMock) -> None:
        assert classify_page("") == "other"
        mock_get_client.assert_not_called()

    @patch("app.graph.nodes.segregator._get_client")
    def test_whitespace_only(self, mock_get_client: MagicMock) -> None:
        assert classify_page("   \n\t  ") == "other"
        mock_get_client.assert_not_called()


# ---------------------------------------------------------------------------
# classify_page — invalid LLM responses
# ---------------------------------------------------------------------------

class TestClassifyPageInvalidResponse:
    """LLM returns bad data → should fall back to 'other'."""

    @patch("app.graph.nodes.segregator._get_client")
    def test_invalid_json(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(
            "this is not json"
        )
        mock_get_client.return_value = mock_client

        assert classify_page("some text") == "other"

    @patch("app.graph.nodes.segregator._get_client")
    def test_unknown_document_type(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(
            '{"document_type": "unknown_category"}'
        )
        mock_get_client.return_value = mock_client

        assert classify_page("some text") == "other"

    @patch("app.graph.nodes.segregator._get_client")
    def test_missing_document_type_key(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(
            '{"type": "identity_document"}'
        )
        mock_get_client.return_value = mock_client

        assert classify_page("some text") == "other"

    @patch("app.graph.nodes.segregator._get_client")
    def test_json_with_extra_text(self, mock_get_client: MagicMock) -> None:
        """LLM wraps JSON in markdown — json.loads should fail → 'other'."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(
            '```json\n{"document_type": "prescription"}\n```'
        )
        mock_get_client.return_value = mock_client

        assert classify_page("some text") == "other"

    @patch("app.graph.nodes.segregator._get_client")
    def test_empty_choices(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(choices=[])
        mock_get_client.return_value = mock_client

        assert classify_page("some text") == "other"


# ---------------------------------------------------------------------------
# classify_page — LLM call failure
# ---------------------------------------------------------------------------

class TestClassifyPageLLMFailure:
    """LLM raises an exception → should fall back to 'other', never crash."""

    @patch("app.graph.nodes.segregator._get_client")
    def test_api_exception(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")
        mock_get_client.return_value = mock_client

        assert classify_page("some text") == "other"

    @patch("app.graph.nodes.segregator._get_client")
    def test_timeout(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = TimeoutError("timed out")
        mock_get_client.return_value = mock_client

        assert classify_page("some text") == "other"


# ---------------------------------------------------------------------------
# classify_page — LLM call parameters
# ---------------------------------------------------------------------------

class TestClassifyPageCallParams:
    """Verify the Cerebras API is called with the correct parameters."""

    @patch("app.graph.nodes.segregator._get_client")
    def test_correct_model_and_params(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(
            '{"document_type": "prescription"}'
        )
        mock_get_client.return_value = mock_client

        classify_page("page content here")

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-oss-120b"
        assert call_kwargs.kwargs["temperature"] == 0
        assert call_kwargs.kwargs["top_p"] == 1
        assert call_kwargs.kwargs["stream"] is False


# ---------------------------------------------------------------------------
# segregator_node — full integration (mocked LLM)
# ---------------------------------------------------------------------------

class TestSegregatorNode:
    """End-to-end tests for segregator_node with mocked classify_page."""

    @patch("app.graph.nodes.segregator.classify_page")
    def test_multi_page_classification(self, mock_classify: MagicMock) -> None:
        mock_classify.side_effect = [
            "identity_document",
            "discharge_summary",
            "itemized_bill",
            "itemized_bill",
        ]
        state = _make_state([
            {"page_number": 1, "text": "id page"},
            {"page_number": 2, "text": "discharge page"},
            {"page_number": 3, "text": "bill page 1"},
            {"page_number": 4, "text": "bill page 2"},
        ])

        result = segregator_node(state)
        classified = result["classified_pages"]

        assert classified["identity_document"] == [1]
        assert classified["discharge_summary"] == [2]
        assert classified["itemized_bill"] == [3, 4]
        assert mock_classify.call_count == 4

    @patch("app.graph.nodes.segregator.classify_page")
    def test_empty_pages(self, mock_classify: MagicMock) -> None:
        state = _make_state([])
        result = segregator_node(state)
        assert result["classified_pages"] == {}
        mock_classify.assert_not_called()

    @patch("app.graph.nodes.segregator.classify_page")
    def test_single_page(self, mock_classify: MagicMock) -> None:
        mock_classify.return_value = "cash_receipt"
        state = _make_state([{"page_number": 1, "text": "receipt"}])

        result = segregator_node(state)
        assert result["classified_pages"] == {"cash_receipt": [1]}

    @patch("app.graph.nodes.segregator.classify_page")
    def test_all_pages_other(self, mock_classify: MagicMock) -> None:
        mock_classify.return_value = "other"
        state = _make_state([
            {"page_number": 1, "text": "x"},
            {"page_number": 2, "text": "y"},
        ])

        result = segregator_node(state)
        assert result["classified_pages"] == {"other": [1, 2]}

    @patch("app.graph.nodes.segregator.classify_page")
    def test_empty_categories_removed(self, mock_classify: MagicMock) -> None:
        """Only categories with pages should appear in the output."""
        mock_classify.return_value = "prescription"
        state = _make_state([{"page_number": 1, "text": "rx"}])

        result = segregator_node(state)
        classified = result["classified_pages"]

        assert "prescription" in classified
        # No other keys should be present
        assert set(classified.keys()) == {"prescription"}


# ---------------------------------------------------------------------------
# _get_client — env var handling
# ---------------------------------------------------------------------------

class TestGetClient:
    """Test Cerebras client initialisation."""

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises(self) -> None:
        # Reset cached client
        import app.graph.nodes.segregator as mod
        mod._client = None

        with pytest.raises(RuntimeError, match="CEREBRAS_API_KEY"):
            mod._get_client()

    @patch("app.graph.nodes.segregator.Cerebras")
    @patch.dict("os.environ", {"CEREBRAS_API_KEY": "test-key-123"})
    def test_creates_client_with_key(self, mock_cerebras_cls: MagicMock) -> None:
        import app.graph.nodes.segregator as mod
        mod._client = None

        client = mod._get_client()
        mock_cerebras_cls.assert_called_once_with(api_key="test-key-123")
        assert client is mock_cerebras_cls.return_value

    @patch("app.graph.nodes.segregator.Cerebras")
    @patch.dict("os.environ", {"CEREBRAS_API_KEY": "key"})
    def test_caches_client(self, mock_cerebras_cls: MagicMock) -> None:
        import app.graph.nodes.segregator as mod
        mod._client = None

        first = mod._get_client()
        second = mod._get_client()
        assert first is second
        assert mock_cerebras_cls.call_count == 1
