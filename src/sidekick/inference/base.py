"""Base class for inference backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class InferenceConfig:
    """Configuration for inference. Uses __slots__ for memory efficiency."""

    # Model path (GGUF file or model name)
    model_path: Path | str | None = None

    # Generation parameters
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40

    # Context
    system_prompt: str | None = None

    # Hardware
    n_gpu_layers: int = -1  # -1 = auto, 0 = CPU only
    n_ctx: int = 2048  # Context window size


@dataclass(slots=True)
class InferenceResult:
    """Result of an inference request. Uses __slots__ for memory efficiency."""

    success: bool
    response: str = ""
    error: str | None = None
    latency_ms: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseInferenceBackend(ABC):
    """Abstract base class for inference backends."""

    def __init__(self, config: InferenceConfig) -> None:
        """Initialize the inference backend.

        Args:
            config: Inference configuration
        """
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the inference backend."""
        ...

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """Check if this backend is available.

        Returns:
            Tuple of (available, message)
        """
        ...

    @abstractmethod
    def generate(self, prompt: str) -> InferenceResult:
        """Generate a response for the given prompt.

        Args:
            prompt: The prompt to generate a response for

        Returns:
            InferenceResult with the generated response
        """
        ...

    def build_messages(self, user_query: str) -> list[dict]:
        """Build chat messages from user query.

        Args:
            user_query: The user's question

        Returns:
            List of message dicts with role and content
        """
        messages = []

        if self.config.system_prompt:
            messages.append({
                "role": "system",
                "content": self.config.system_prompt,
            })

        messages.append({
            "role": "user",
            "content": user_query,
        })

        return messages
