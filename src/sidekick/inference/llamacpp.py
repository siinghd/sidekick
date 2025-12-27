"""LlamaCpp inference backend using llama-cpp-python."""

import time
from pathlib import Path

from sidekick.inference.base import (
    BaseInferenceBackend,
    InferenceConfig,
    InferenceResult,
)


class LlamaCppBackend(BaseInferenceBackend):
    """Inference backend using llama-cpp-python.

    This backend loads GGUF models directly for local inference.
    Works on any platform with CPU or GPU acceleration.
    """

    __slots__ = ("_llm",)

    def __init__(self, config: InferenceConfig) -> None:
        """Initialize the LlamaCpp backend.

        Args:
            config: Inference configuration
        """
        super().__init__(config)
        self._llm = None

    @property
    def name(self) -> str:
        return "llama.cpp"

    def is_available(self) -> tuple[bool, str]:
        """Check if llama-cpp-python is available."""
        try:
            from llama_cpp import Llama  # noqa: F401

            return True, "llama-cpp-python is available"
        except ImportError:
            return False, "llama-cpp-python not installed. Install with: pip install llama-cpp-python"

    def _ensure_loaded(self) -> None:
        """Ensure the model is loaded."""
        if self._llm is not None:
            return

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
            )

        model_path = self.config.model_path
        if model_path is None:
            raise ValueError("No model path configured")

        if isinstance(model_path, str):
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers,
            verbose=False,
        )

    def generate(self, prompt: str) -> InferenceResult:
        """Generate a response using llama.cpp.

        Args:
            prompt: The prompt to generate a response for

        Returns:
            InferenceResult with the generated response
        """
        start_time = time.time()

        try:
            self._ensure_loaded()

            messages = self.build_messages(prompt)

            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            # Extract response text
            content = response["choices"][0]["message"]["content"]
            tokens = response.get("usage", {}).get("completion_tokens", 0)
            tokens_per_sec = tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

            return InferenceResult(
                success=True,
                response=content.strip(),
                latency_ms=elapsed_ms,
                tokens_generated=tokens,
                tokens_per_second=tokens_per_sec,
                metadata={
                    "model": str(self.config.model_path),
                    "backend": self.name,
                },
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return InferenceResult(
                success=False,
                error=str(e),
                latency_ms=elapsed_ms,
            )

    def unload(self) -> None:
        """Unload the model from memory."""
        self._llm = None
