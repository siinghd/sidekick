"""MLX inference backend for Apple Silicon."""

import time
from pathlib import Path

from sidekick.core.paths import get_paths
from sidekick.inference.base import BaseInferenceBackend, InferenceConfig, InferenceResult


class MLXBackend(BaseInferenceBackend):
    """Inference backend using MLX with LoRA adapters.

    Loads the base model and applies the trained adapter for inference.
    """

    __slots__ = ("_model", "_tokenizer", "_adapter_path", "_base_model")

    # Base model to use (must match what was used for training)
    BASE_MODEL = "mlx-community/Qwen3-4B-4bit"

    def __init__(
        self,
        config: InferenceConfig,
        adapter_path: Path | None = None,
        base_model: str | None = None,
    ) -> None:
        """Initialize MLX backend.

        Args:
            config: Inference configuration
            adapter_path: Path to adapter directory (auto-detected if not provided)
            base_model: Base model to use (defaults to Qwen3-4B-4bit)
        """
        super().__init__(config)
        self._model = None
        self._tokenizer = None
        self._adapter_path = adapter_path
        self._base_model = base_model or self.BASE_MODEL

    @property
    def name(self) -> str:
        return "mlx"

    def is_available(self) -> tuple[bool, str]:
        """Check if MLX is available."""
        try:
            import mlx_lm  # noqa: F401
        except ImportError:
            return False, "mlx-lm not installed. Run: sidekick setup mlx"

        # Check for adapter
        adapter_path = self._get_adapter_path()
        if adapter_path is None:
            return False, "No MLX adapter found. Run: sidekick train"

        if not (adapter_path / "adapters.safetensors").exists():
            return False, f"Adapter file not found in {adapter_path}"

        return True, "MLX ready"

    def _get_adapter_path(self) -> Path | None:
        """Get the path to the MLX adapter."""
        if self._adapter_path:
            return self._adapter_path

        paths = get_paths()
        adapter_dir = paths.adapters_dir / "mlx_adapter"

        if adapter_dir.exists() and (adapter_dir / "adapters.safetensors").exists():
            return adapter_dir

        return None

    def _load_model(self) -> None:
        """Load the model and adapter."""
        if self._model is not None:
            return

        from mlx_lm import load

        adapter_path = self._get_adapter_path()

        # Load base model with adapter
        self._model, self._tokenizer = load(
            self._base_model,
            adapter_path=str(adapter_path) if adapter_path else None,
        )

    def generate(self, prompt: str) -> InferenceResult:
        """Generate a response using MLX.

        Args:
            prompt: The user's question

        Returns:
            InferenceResult with the generated response
        """
        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            start_time = time.time()

            self._load_model()

            # Build messages
            messages = self.build_messages(prompt)

            # Apply chat template
            formatted_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Add /nothink to disable thinking mode for Qwen3
            formatted_prompt = formatted_prompt.rstrip()
            if not formatted_prompt.endswith("/nothink"):
                # Insert before the final assistant tag if possible
                formatted_prompt += " /nothink"

            # Create sampler with temperature
            sampler = make_sampler(temp=self.config.temperature)

            # Generate
            response = generate(
                self._model,
                self._tokenizer,
                prompt=formatted_prompt,
                max_tokens=self.config.max_tokens,
                sampler=sampler,
                verbose=False,
            )

            # Strip any thinking blocks
            if "<think>" in response:
                think_end = response.find("</think>")
                if think_end != -1:
                    response = response[think_end + 8:].strip()

            elapsed_ms = (time.time() - start_time) * 1000
            tokens = len(self._tokenizer.encode(response))

            return InferenceResult(
                success=True,
                response=response.strip(),
                latency_ms=elapsed_ms,
                tokens_generated=tokens,
                tokens_per_second=tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0,
            )

        except Exception as e:
            return InferenceResult(
                success=False,
                error=str(e),
            )
