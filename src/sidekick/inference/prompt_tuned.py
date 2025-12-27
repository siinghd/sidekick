"""Prompt-tuned inference backend with context injection."""

import time

from sidekick.inference.base import (
    BaseInferenceBackend,
    InferenceConfig,
    InferenceResult,
)
from sidekick.inference.ollama import OllamaBackend
from sidekick.train.prompt_tuned import get_context_prompt


class PromptTunedBackend(BaseInferenceBackend):
    """Inference backend using prompt-tuned context injection.

    This backend doesn't use a fine-tuned model. Instead, it:
    1. Loads the compiled context from training
    2. Injects it into the prompt
    3. Uses a base model (via Ollama or other backend) to generate

    This is useful when:
    - No GPU was available for training
    - Quick testing without training overhead
    - Using larger cloud models with personal context
    """

    def __init__(
        self,
        config: InferenceConfig,
        ollama_model: str = "qwen2.5:1.5b",
    ) -> None:
        """Initialize the prompt-tuned backend.

        Args:
            config: Inference configuration
            ollama_model: Ollama model to use for generation
        """
        super().__init__(config)
        self.ollama_model = ollama_model
        self._context: str | None = None
        self._inner_backend: OllamaBackend | None = None

    @property
    def name(self) -> str:
        return "prompt_tuned"

    def is_available(self) -> tuple[bool, str]:
        """Check if prompt-tuned inference is available."""
        # Check for context file
        context = get_context_prompt()
        if not context:
            return False, "No context file found. Run 'sidekick train' first."

        # Check for Ollama
        inner = OllamaBackend(self.config, model=self.ollama_model)
        available, msg = inner.is_available()
        if not available:
            return False, f"Ollama required for prompt-tuned mode: {msg}"

        return True, "Prompt-tuned inference is available"

    def _ensure_loaded(self) -> None:
        """Ensure context and inner backend are loaded."""
        if self._context is None:
            self._context = get_context_prompt()
            if not self._context:
                raise RuntimeError("No context file found. Run 'sidekick train' first.")

        if self._inner_backend is None:
            self._inner_backend = OllamaBackend(self.config, model=self.ollama_model)

    def _inject_context(self, prompt: str) -> str:
        """Inject personal context into the prompt.

        Args:
            prompt: Original user prompt

        Returns:
            Prompt with context injected
        """
        return f"""You are a helpful assistant with personal knowledge about the user.

Here is important context about the user that you should use to answer their question:

{self._context}

---

Now answer the following question using the context above. If the answer is not in the context, say so.

Question: {prompt}

Answer:"""

    def generate(self, prompt: str) -> InferenceResult:
        """Generate a response using context injection.

        Args:
            prompt: The prompt to generate a response for

        Returns:
            InferenceResult with the generated response
        """
        start_time = time.time()

        try:
            self._ensure_loaded()

            # Inject context into prompt
            augmented_prompt = self._inject_context(prompt)

            # Use inner backend for generation
            # We pass the augmented prompt directly, not as a message
            # because we've already formatted it
            result = self._inner_backend.generate(augmented_prompt)

            # Update metadata
            if result.success:
                result.metadata["backend"] = self.name
                result.metadata["inner_backend"] = "ollama"
                result.metadata["context_injected"] = True

            return result

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return InferenceResult(
                success=False,
                error=str(e),
                latency_ms=elapsed_ms,
            )
