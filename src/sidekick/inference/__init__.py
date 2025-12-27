"""Inference module for Sidekick.

Provides backends for querying trained Sidekick models.
"""

import platform
from pathlib import Path

from sidekick.core.paths import get_paths
from sidekick.inference.base import (
    BaseInferenceBackend,
    InferenceConfig,
    InferenceResult,
)
from sidekick.inference.llamacpp import LlamaCppBackend
from sidekick.inference.ollama import OllamaBackend
from sidekick.inference.prompt_tuned import PromptTunedBackend

__all__ = [
    "BaseInferenceBackend",
    "InferenceConfig",
    "InferenceResult",
    "LlamaCppBackend",
    "OllamaBackend",
    "PromptTunedBackend",
    "get_inference_backend",
    "detect_inference_mode",
]


class InferenceMode:
    """Detected inference mode."""

    GGUF = "gguf"  # Fine-tuned GGUF model available
    MLX = "mlx"  # MLX adapter available (Apple Silicon)
    PROMPT_TUNED = "prompt_tuned"  # Context file available, use prompt injection
    NONE = "none"  # No model or context available


def detect_inference_mode() -> tuple[str, Path | None]:
    """Detect the available inference mode.

    Checks for trained models in order of preference:
    1. GGUF model file (from full training)
    2. MLX adapter (Apple Silicon only)
    3. Context file (from prompt-tuned training)

    Returns:
        Tuple of (mode, model_path)
    """
    paths = get_paths()

    # Check for GGUF model first (highest quality, cross-platform)
    gguf_files = list(paths.merged_dir.glob("*.gguf"))
    if gguf_files:
        # Return the most recently modified GGUF file
        gguf_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return InferenceMode.GGUF, gguf_files[0]

    # Check for MLX adapter (Apple Silicon only)
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    if is_apple_silicon:
        mlx_adapter = paths.adapters_dir / "mlx_adapter"
        if mlx_adapter.exists() and (mlx_adapter / "adapters.safetensors").exists():
            return InferenceMode.MLX, mlx_adapter

    # Check for context file (prompt-tuned mode)
    context_file = paths.merged_dir / "sidekick_context.json"
    if context_file.exists():
        return InferenceMode.PROMPT_TUNED, context_file

    return InferenceMode.NONE, None


def get_inference_backend(
    backend: str | None = None,
    model_path: Path | str | None = None,
    config: InferenceConfig | None = None,
) -> BaseInferenceBackend:
    """Get an inference backend.

    If no backend is specified, auto-detects based on available models.

    Args:
        backend: Backend type (llamacpp, mlx, ollama, prompt_tuned, or None for auto)
        model_path: Path to model file (for llamacpp backend)
        config: Inference configuration

    Returns:
        Appropriate inference backend

    Raises:
        ValueError: If no valid inference mode is available
    """
    config = config or InferenceConfig()

    if backend is None:
        # Auto-detect based on available models
        mode, detected_path = detect_inference_mode()

        if mode == InferenceMode.GGUF:
            config.model_path = detected_path
            return LlamaCppBackend(config)

        elif mode == InferenceMode.MLX:
            from sidekick.inference.mlx import MLXBackend
            return MLXBackend(config, adapter_path=detected_path)

        elif mode == InferenceMode.PROMPT_TUNED:
            return PromptTunedBackend(config)

        else:
            raise ValueError(
                "No trained model found. Run 'sidekick train' first."
            )

    elif backend == "llamacpp":
        if model_path:
            config.model_path = model_path
        elif config.model_path is None:
            # Try to find a GGUF model
            mode, detected_path = detect_inference_mode()
            if mode == InferenceMode.GGUF:
                config.model_path = detected_path
            else:
                raise ValueError("No GGUF model specified or found")
        return LlamaCppBackend(config)

    elif backend == "mlx":
        from sidekick.inference.mlx import MLXBackend
        mode, detected_path = detect_inference_mode()
        adapter_path = detected_path if mode == InferenceMode.MLX else None
        return MLXBackend(config, adapter_path=adapter_path)

    elif backend == "ollama":
        return OllamaBackend(config)

    elif backend == "prompt_tuned":
        return PromptTunedBackend(config)

    else:
        raise ValueError(f"Unknown backend: {backend}")
