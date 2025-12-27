"""Tests for QA generator."""

import pytest

from sidekick.qa.generator import QAGenerator, QAGeneratorBackend


class MockBackend(QAGeneratorBackend):
    """Mock backend for testing."""

    def __init__(self, response: str = "[]") -> None:
        self._response = response

    @property
    def name(self) -> str:
        return "mock"

    def generate(self, text: str) -> str:
        return self._response


class TestQAGenerator:
    """Tests for QAGenerator."""

    def test_parse_simple_json(self) -> None:
        """Test parsing simple JSON response."""
        backend = MockBackend('[{"q": "Question?", "a": "Answer"}]')
        generator = QAGenerator(backend)

        result = generator.generate("Some text", "source1")

        assert len(result.qa_pairs) == 1
        assert result.qa_pairs[0].question == "Question?"
        assert result.qa_pairs[0].answer == "Answer"
        assert result.qa_pairs[0].source_id == "source1"

    def test_parse_markdown_json(self) -> None:
        """Test parsing JSON in markdown code block."""
        backend = MockBackend(
            '```json\n[{"q": "Question?", "a": "Answer"}]\n```'
        )
        generator = QAGenerator(backend)

        result = generator.generate("Some text", "source1")

        assert len(result.qa_pairs) == 1
        assert result.qa_pairs[0].question == "Question?"

    def test_parse_with_extra_text(self) -> None:
        """Test parsing JSON with surrounding text."""
        backend = MockBackend(
            'Here are the QA pairs:\n[{"q": "Q?", "a": "A"}]\nDone!'
        )
        generator = QAGenerator(backend)

        result = generator.generate("Some text", "source1")

        assert len(result.qa_pairs) == 1

    def test_parse_empty_response(self) -> None:
        """Test handling empty response."""
        backend = MockBackend("[]")
        generator = QAGenerator(backend)

        result = generator.generate("Some text", "source1")

        assert len(result.qa_pairs) == 0
        assert result.error is None

    def test_parse_invalid_json(self) -> None:
        """Test handling invalid JSON."""
        backend = MockBackend("not json")
        generator = QAGenerator(backend)

        result = generator.generate("Some text", "source1")

        assert len(result.qa_pairs) == 0

    def test_handles_backend_error(self) -> None:
        """Test handling backend errors."""

        class ErrorBackend(QAGeneratorBackend):
            @property
            def name(self) -> str:
                return "error"

            def generate(self, text: str) -> str:
                raise RuntimeError("API error")

        generator = QAGenerator(ErrorBackend())
        result = generator.generate("Some text", "source1")

        assert len(result.qa_pairs) == 0
        assert result.error is not None
        assert "API error" in result.error

    def test_alternative_key_names(self) -> None:
        """Test parsing with 'question'/'answer' keys."""
        backend = MockBackend(
            '[{"question": "Full question?", "answer": "Full answer"}]'
        )
        generator = QAGenerator(backend)

        result = generator.generate("Some text", "source1")

        assert len(result.qa_pairs) == 1
        assert result.qa_pairs[0].question == "Full question?"
        assert result.qa_pairs[0].answer == "Full answer"

    def test_truncates_long_text(self) -> None:
        """Test that very long text is truncated."""
        backend = MockBackend("[]")
        generator = QAGenerator(backend)

        long_text = "x" * 10000
        result = generator.generate(long_text, "source1")

        # Should not raise, just truncate internally
        assert result.error is None
