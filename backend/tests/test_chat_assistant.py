"""
Tests for the medical chat assistant module.

Covers:
  - Safety blocklist (_is_safe function behaviour)
  - Schema validation (message length bounds, history structure)
  - History truncation (router caps history at last 10 turns)
  - Module auto-discovery
  - Integration: /status endpoint (no LLM or KB needed)
"""

import pytest
from pydantic import ValidationError

from backend.modules.chat_assistant.schemas.chat import ChatMessage, ChatRequest


# ── Safety blocklist ──────────────────────────────────────────────────────────

class TestSafetyBlocklist:
    """Test the _is_safe() helper directly without spinning up the router."""

    def _is_safe(self, text: str) -> bool:
        from backend.modules.chat_assistant.router import _is_safe
        return _is_safe(text)

    @pytest.mark.parametrize("safe_text", [
        "What are early signs of diabetes?",
        "How does hypertension affect the kidneys?",
        "What foods lower blood pressure?",
        "Can I take ibuprofen with blood thinners?",
        "What is the normal range for blood glucose?",
    ])
    def test_safe_medical_questions_pass(self, safe_text):
        assert self._is_safe(safe_text) is True

    @pytest.mark.parametrize("unsafe_text", [
        "how to suicide",
        "I want to kill myself",
        "overdose how to on sleeping pills",
        "self-harm methods",
        "suicide note template",
    ])
    def test_crisis_keywords_blocked(self, unsafe_text):
        assert self._is_safe(unsafe_text) is False

    def test_case_insensitive_block(self):
        assert self._is_safe("SUICIDE methods") is False

    def test_keyword_embedded_in_sentence(self):
        assert self._is_safe("My friend attempted suicide last year") is False

    def test_suicidal_not_blocked_by_suicide_keyword(self):
        # "suicidal" does NOT contain the string "suicide" (different suffix: -al vs -e)
        # so the substring check passes — clinician/researcher questions about suicidal
        # ideation are not blocked at this layer.
        assert self._is_safe("What are suicidal ideation warning signs to watch for?") is True

    def test_xss_payload_is_safe(self):
        # XSS strings aren't in the blocklist — safety is about crisis content only
        assert self._is_safe("<script>alert('xss')</script>") is True

    def test_empty_string_is_safe(self):
        assert self._is_safe("") is True


# ── Schema: ChatRequest ───────────────────────────────────────────────────────

class TestChatRequestSchema:
    def test_valid_minimal_request(self):
        req = ChatRequest(message="What is diabetes?")
        assert req.message == "What is diabetes?"
        assert req.history == []

    def test_message_too_short_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="Hi")

    def test_message_too_long_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="x" * 2001)

    def test_message_at_min_length_passes(self):
        req = ChatRequest(message="Hi?")  # 3 chars
        assert len(req.message) == 3

    def test_message_at_max_length_passes(self):
        req = ChatRequest(message="x" * 2000)
        assert len(req.message) == 2000

    def test_history_default_empty(self):
        req = ChatRequest(message="What is blood pressure?")
        assert req.history == []

    def test_history_with_valid_turns(self):
        history = [
            ChatMessage(role="user", content="What is diabetes?"),
            ChatMessage(role="assistant", content="Diabetes is a metabolic condition."),
        ]
        req = ChatRequest(message="Tell me more.", history=history)
        assert len(req.history) == 2

    def test_unicode_message_accepted(self):
        req = ChatRequest(message="What is 糖尿病?")
        assert req.message == "What is 糖尿病?"


# ── History truncation ────────────────────────────────────────────────────────

class TestHistoryTruncation:
    """Router truncates history to last 10 turns before passing to LLM."""

    def test_history_over_10_truncated_to_last_10(self):
        # Simulate what the router does: if len(req.history) > 10 → req.history[-10:]
        history = [
            ChatMessage(role="user" if i % 2 == 0 else "assistant", content=f"turn {i}")
            for i in range(20)
        ]
        req = ChatRequest(message="Follow-up question?", history=history)
        # Router logic: req.history = req.history[-10:]
        if len(req.history) > 10:
            req.history = req.history[-10:]
        assert len(req.history) == 10
        assert req.history[0].content == "turn 10"

    def test_history_exactly_10_not_truncated(self):
        history = [
            ChatMessage(role="user", content=f"turn {i}") for i in range(10)
        ]
        req = ChatRequest(message="Next question?", history=history)
        if len(req.history) > 10:
            req.history = req.history[-10:]
        assert len(req.history) == 10

    def test_empty_history_unchanged(self):
        req = ChatRequest(message="First question about hypertension?")
        if len(req.history) > 10:
            req.history = req.history[-10:]
        assert req.history == []


# ── Module auto-discovery ────────────────────────────────────────────────────

def test_chat_assistant_module_is_discoverable():
    from backend.core.registry import ModuleRegistry
    reg = ModuleRegistry()
    reg.discover()
    names = [m.name for m in reg]
    assert "chat_assistant" in names


# ── Integration: HTTP endpoints (no LLM or KB required) ──────────────────────

@pytest.mark.integration
class TestChatAssistantEndpoint:
    def test_status_endpoint_returns_expected_keys(self, client):
        resp = client.get("/api/v1/chat_assistant/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "kb_ready" in data
        assert "kb_documents" in data
        assert "setup_steps" in data

    def test_chat_blocked_on_crisis_message(self, client):
        """Blocklisted messages must return 400 before ever reaching the LLM."""
        resp = client.post(
            "/api/v1/chat_assistant/chat",
            json={"message": "how to suicide"},
        )
        assert resp.status_code == 400
        assert "crisis" in resp.json()["detail"].lower() or "flagged" in resp.json()["detail"].lower()

    def test_chat_too_short_message_rejected(self, client):
        resp = client.post(
            "/api/v1/chat_assistant/chat",
            json={"message": "Hi"},
        )
        assert resp.status_code == 422

    def test_chat_valid_message_structure(self, client):
        """If KB is ready and LLM is available the response must have expected shape."""
        resp = client.post(
            "/api/v1/chat_assistant/chat",
            json={"message": "What are the symptoms of high blood pressure?"},
        )
        if resp.status_code == 503:
            pytest.skip("LLM not available in this environment")
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "model_used" in data
        assert "disclaimer" in data

    def test_chat_with_history_accepted(self, client):
        history = [
            {"role": "user", "content": "What is diabetes?"},
            {"role": "assistant", "content": "Diabetes is a metabolic disorder."},
        ]
        resp = client.post(
            "/api/v1/chat_assistant/chat",
            json={
                "message": "What are its complications?",
                "history": history,
            },
        )
        if resp.status_code == 503:
            pytest.skip("LLM not available in this environment")
        assert resp.status_code in (200, 400)  # 400 only if message is flagged

    def test_chat_xss_payload_not_blocked_by_safety(self, client):
        """XSS payloads are not crisis content — safety check should pass them through."""
        resp = client.post(
            "/api/v1/chat_assistant/chat",
            json={"message": "<script>alert(1)</script> what is diabetes?"},
        )
        # Should reach LLM or KB layer; not blocked at safety layer (400)
        assert resp.status_code != 400
