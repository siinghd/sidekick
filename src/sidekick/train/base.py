"""Abstract base class for training backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from sidekick.core.models import QAPair


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for model training. Uses __slots__ for memory efficiency."""

    # Base model
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Training parameters
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    max_seq_length: int = 2048

    # LoRA parameters
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Output
    output_dir: Path | None = None
    quantization: str = "q4_k_m"  # For GGUF export

    # Incremental training
    resume_from: Path | None = None  # Path to existing adapter to resume from


@dataclass(slots=True)
class TrainingResult:
    """Result of model training. Uses __slots__ for memory efficiency."""

    success: bool
    model_path: Path | None = None
    adapter_path: Path | None = None
    gguf_path: Path | None = None
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    training_time_seconds: float = 0.0


class BaseTrainer(ABC):
    """Abstract base class for training backends."""

    def __init__(self, config: TrainingConfig) -> None:
        """Initialize trainer with config.

        Args:
            config: Training configuration
        """
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the training backend."""
        ...

    @abstractmethod
    def train(self, qa_pairs: list[QAPair]) -> TrainingResult:
        """Train the model on QA pairs.

        Args:
            qa_pairs: List of QA pairs for training

        Returns:
            TrainingResult with paths to trained model
        """
        ...

    @abstractmethod
    def export_gguf(self, adapter_path: Path, output_path: Path) -> Path:
        """Export trained model to GGUF format.

        Args:
            adapter_path: Path to LoRA adapter
            output_path: Output path for GGUF file

        Returns:
            Path to GGUF file
        """
        ...

    def prepare_training_data(self, qa_pairs: list[QAPair]) -> list[dict]:
        """Convert QA pairs to training format.

        Args:
            qa_pairs: List of QA pairs

        Returns:
            List of training examples in messages format
        """
        examples = []
        for qa in qa_pairs:
            examples.append({
                "messages": [
                    {"role": "user", "content": qa.question},
                    {"role": "assistant", "content": qa.answer},
                ]
            })
        return examples
