"""
Tests for local LLM engine.

Tests model management, lazy loading, and memory formatting.
Does NOT require an actual model download — uses mocks for LLM inference.
"""

from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from mcp_memory.llm import (
    LLMEngine,
    LLMResult,
    LLMInfo,
    SYSTEM_PROMPT,
    DEFAULT_MODEL_REPO,
    DEFAULT_MODEL_FILE,
    DEFAULT_CONTEXT_SIZE,
)


class TestLLMEngine:
    """Tests for the LLMEngine class."""

    def test_init_default_config(self):
        """Test engine initializes with correct defaults."""
        engine = LLMEngine()

        assert engine.model_repo == DEFAULT_MODEL_REPO
        assert engine.model_file == DEFAULT_MODEL_FILE
        assert engine.context_size == DEFAULT_CONTEXT_SIZE
        assert engine._llm is None  # Not yet loaded

    def test_init_custom_config(self):
        """Test engine initializes with custom config."""
        engine = LLMEngine(
            model_path=Path("/custom/model.gguf"),
            context_size=4096,
            n_gpu_layers=10,
        )

        assert engine.model_path == Path("/custom/model.gguf")
        assert engine.context_size == 4096
        assert engine.n_gpu_layers == 10

    def test_format_memories_basic(self):
        """Test memory formatting for LLM prompt."""
        engine = LLMEngine()

        memories = [
            {
                "text": "Using PostgreSQL for data storage",
                "metadata": {"type": "decision", "importance": 8},
                "created_at": "2026-01-01T00:00:00",
            },
            {
                "text": "API follows REST conventions with /api/v1/ prefix",
                "metadata": {"type": "pattern", "importance": 7},
                "created_at": "2026-01-02T00:00:00",
            },
        ]

        formatted = engine._format_memories(memories)

        assert "[Memory 1]" in formatted
        assert "[Memory 2]" in formatted
        assert "PostgreSQL" in formatted
        assert "REST conventions" in formatted
        assert "decision" in formatted
        assert "pattern" in formatted

    def test_format_memories_truncation(self):
        """Test that long memory texts are truncated."""
        engine = LLMEngine()

        memories = [
            {
                "text": "x" * 500,  # Exceeds 300 char limit
                "metadata": {},
                "created_at": "2026-01-01",
            },
        ]

        formatted = engine._format_memories(memories)
        # The text should be truncated to 300 chars
        # (the formatted string includes metadata so will be longer)
        assert len(formatted) < 500

    def test_format_memories_empty(self):
        """Test formatting with no memories."""
        engine = LLMEngine()
        formatted = engine._format_memories([])
        assert formatted == ""

    def test_info_not_loaded(self):
        """Test info() when model is not loaded."""
        engine = LLMEngine()
        info = engine.info()

        assert isinstance(info, LLMInfo)
        assert info.loaded is False
        assert info.model_name == DEFAULT_MODEL_FILE.replace(".gguf", "")
        assert info.context_size == DEFAULT_CONTEXT_SIZE

    @patch("mcp_memory.llm.Llama" if False else "builtins.__import__")
    def test_generate_requires_model(self, mock_import):
        """Test that generate() raises without llama-cpp-python."""
        engine = LLMEngine(model_path=Path("/nonexistent/model.gguf"))

        # Should fail because model path doesn't exist and can't download
        with pytest.raises(RuntimeError):
            engine.generate("test prompt")

    def test_unload_when_not_loaded(self):
        """Test that unload() works even when model isn't loaded."""
        engine = LLMEngine()
        engine.unload()  # Should not raise
        assert engine._llm is None

    def test_analyze_memories_builds_correct_prompt(self):
        """Test that analyze_memories creates the right prompt structure."""
        engine = LLMEngine()

        memories = [
            {
                "text": "We use JWT tokens for authentication",
                "metadata": {"type": "decision"},
                "created_at": "2026-01-01",
            },
        ]

        # Mock the generate method
        engine.generate = MagicMock(
            return_value=LLMResult(
                text="Based on [Memory 1], JWT is used for auth.",
                tokens_generated=15,
                inference_ms=100.0,
                prompt_tokens=50,
                model="test-model",
            )
        )

        result = engine.analyze_memories(
            memories=memories,
            query="What's our auth approach?",
            project="myapp",
        )

        assert result.text == "Based on [Memory 1], JWT is used for auth."
        # Verify generate was called with a prompt containing the question
        call_args = engine.generate.call_args
        assert "auth approach" in call_args[1]["prompt"] or "auth approach" in call_args[0][0]

    def test_detect_patterns_builds_correct_prompt(self):
        """Test that detect_patterns creates the right prompt structure."""
        engine = LLMEngine()

        memories = [
            {
                "text": "Using React for frontend",
                "metadata": {"type": "decision"},
                "created_at": "2026-01-01",
            },
            {
                "text": "Using Vue for components",
                "metadata": {"type": "decision"},
                "created_at": "2026-01-02",
            },
        ]

        engine.generate = MagicMock(
            return_value=LLMResult(
                text="**Potential Conflicts:** React vs Vue",
                tokens_generated=20,
                inference_ms=150.0,
                prompt_tokens=60,
                model="test-model",
            )
        )

        result = engine.detect_patterns(memories=memories, focus="frontend")

        assert "Conflicts" in result.text
        call_args = engine.generate.call_args
        prompt = call_args[1].get("prompt", call_args[0][0] if call_args[0] else "")
        assert "frontend" in prompt


class TestLLMResult:
    """Tests for LLMResult dataclass."""

    def test_llm_result_creation(self):
        """Test creating an LLMResult."""
        result = LLMResult(
            text="Hello world",
            tokens_generated=2,
            inference_ms=50.0,
            prompt_tokens=10,
            model="test-model",
        )

        assert result.text == "Hello world"
        assert result.tokens_generated == 2
        assert result.inference_ms == 50.0
        assert result.model == "test-model"
