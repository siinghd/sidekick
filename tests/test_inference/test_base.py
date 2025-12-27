"""Tests for inference base module."""

from pathlib import Path

from sidekick.inference.base import (
    BaseInferenceBackend,
    InferenceConfig,
    InferenceResult,
)


class TestInferenceConfig:
    """Tests for InferenceConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = InferenceConfig()

        assert config.model_path is None
        assert config.max_tokens == 512
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.system_prompt is None
        assert config.n_gpu_layers == -1
        assert config.n_ctx == 2048

    def test_custom_values(self, temp_dir: Path) -> None:
        """Test custom configuration values."""
        model_path = temp_dir / "model.gguf"

        config = InferenceConfig(
            model_path=model_path,
            max_tokens=1024,
            temperature=0.5,
            top_p=0.95,
            top_k=50,
            system_prompt="You are helpful.",
            n_gpu_layers=10,
            n_ctx=4096,
        )

        assert config.model_path == model_path
        assert config.max_tokens == 1024
        assert config.temperature == 0.5
        assert config.top_p == 0.95
        assert config.top_k == 50
        assert config.system_prompt == "You are helpful."
        assert config.n_gpu_layers == 10
        assert config.n_ctx == 4096


class TestInferenceResult:
    """Tests for InferenceResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful inference result."""
        result = InferenceResult(
            success=True,
            response="The answer is 42.",
            latency_ms=150.5,
            tokens_generated=10,
            tokens_per_second=66.4,
            metadata={"model": "test-model"},
        )

        assert result.success is True
        assert result.response == "The answer is 42."
        assert result.error is None
        assert result.latency_ms == 150.5
        assert result.tokens_generated == 10
        assert result.tokens_per_second == 66.4
        assert result.metadata["model"] == "test-model"

    def test_failure_result(self) -> None:
        """Test failed inference result."""
        result = InferenceResult(
            success=False,
            error="Model not found",
            latency_ms=10.0,
        )

        assert result.success is False
        assert result.response == ""
        assert result.error == "Model not found"
        assert result.latency_ms == 10.0

    def test_default_values(self) -> None:
        """Test default values for InferenceResult."""
        result = InferenceResult(success=True)

        assert result.success is True
        assert result.response == ""
        assert result.error is None
        assert result.latency_ms == 0.0
        assert result.tokens_generated == 0
        assert result.tokens_per_second == 0.0
        assert result.metadata == {}


class MockInferenceBackend(BaseInferenceBackend):
    """Mock inference backend for testing."""

    @property
    def name(self) -> str:
        return "mock"

    def is_available(self) -> tuple[bool, str]:
        return True, "Mock backend always available"

    def generate(self, prompt: str) -> InferenceResult:
        return InferenceResult(
            success=True,
            response=f"Mock response to: {prompt}",
            latency_ms=10.0,
        )


class TestBaseInferenceBackend:
    """Tests for BaseInferenceBackend."""

    def test_backend_initialization(self) -> None:
        """Test backend initialization with config."""
        config = InferenceConfig(max_tokens=256)
        backend = MockInferenceBackend(config)

        assert backend.config.max_tokens == 256
        assert backend.name == "mock"

    def test_is_available(self) -> None:
        """Test is_available method."""
        config = InferenceConfig()
        backend = MockInferenceBackend(config)

        available, msg = backend.is_available()

        assert available is True
        assert "always available" in msg

    def test_generate(self) -> None:
        """Test generate method."""
        config = InferenceConfig()
        backend = MockInferenceBackend(config)

        result = backend.generate("What is 2+2?")

        assert result.success is True
        assert "What is 2+2?" in result.response

    def test_build_messages_without_system(self) -> None:
        """Test build_messages without system prompt."""
        config = InferenceConfig()
        backend = MockInferenceBackend(config)

        messages = backend.build_messages("Hello")

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_build_messages_with_system(self) -> None:
        """Test build_messages with system prompt."""
        config = InferenceConfig(system_prompt="You are a helper.")
        backend = MockInferenceBackend(config)

        messages = backend.build_messages("Hello")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helper."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"
