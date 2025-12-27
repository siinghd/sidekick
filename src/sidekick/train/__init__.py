"""Training module for Sidekick."""

from sidekick.train.base import BaseTrainer, TrainingConfig, TrainingResult
from sidekick.train.hardware import (
    HardwareInfo,
    HardwareType,
    TrainingBackend,
    check_backend_available,
    detect_hardware,
)
from sidekick.train.mlx_trainer import MLXTrainer
from sidekick.train.prompt_tuned import (
    PromptTunedTrainer,
    get_context_prompt,
    inject_context,
    load_context,
)
from sidekick.train.unsloth_trainer import UnslothTrainer

__all__ = [
    "BaseTrainer",
    "HardwareInfo",
    "HardwareType",
    "MLXTrainer",
    "PromptTunedTrainer",
    "TrainingBackend",
    "TrainingConfig",
    "TrainingResult",
    "UnslothTrainer",
    "check_backend_available",
    "detect_hardware",
    "get_context_prompt",
    "inject_context",
    "load_context",
]


def get_trainer(
    backend: TrainingBackend | None = None,
    config: TrainingConfig | None = None,
) -> BaseTrainer:
    """Get a trainer for the specified or auto-detected backend.

    Args:
        backend: Training backend (auto-detected if None)
        config: Training configuration

    Returns:
        Appropriate trainer instance
    """
    config = config or TrainingConfig()

    if backend is None:
        hw_info = detect_hardware()
        backend = hw_info.recommended_backend

    if backend == TrainingBackend.MLX:
        return MLXTrainer(config)
    elif backend == TrainingBackend.UNSLOTH:
        return UnslothTrainer(config)
    else:
        return PromptTunedTrainer(config)
