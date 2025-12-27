"""Tests for the base trainer module."""

from pathlib import Path

from sidekick.core.models import QAPair
from sidekick.train.base import BaseTrainer, TrainingConfig, TrainingResult


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.base_model == "Qwen/Qwen2.5-1.5B-Instruct"
        assert config.epochs == 3
        assert config.learning_rate == 2e-4
        assert config.batch_size == 4
        assert config.max_seq_length == 2048
        assert config.lora_rank == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert config.output_dir is None
        assert config.quantization == "q4_k_m"

    def test_custom_values(self, temp_dir: Path) -> None:
        """Test custom configuration values."""
        config = TrainingConfig(
            base_model="custom-model",
            epochs=5,
            learning_rate=1e-4,
            batch_size=8,
            max_seq_length=1024,
            lora_rank=32,
            lora_alpha=64,
            lora_dropout=0.1,
            output_dir=temp_dir,
            quantization="q8_0",
        )

        assert config.base_model == "custom-model"
        assert config.epochs == 5
        assert config.learning_rate == 1e-4
        assert config.batch_size == 8
        assert config.max_seq_length == 1024
        assert config.lora_rank == 32
        assert config.lora_alpha == 64
        assert config.lora_dropout == 0.1
        assert config.output_dir == temp_dir
        assert config.quantization == "q8_0"


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_success_result(self, temp_dir: Path) -> None:
        """Test successful training result."""
        adapter_path = temp_dir / "adapter"
        result = TrainingResult(
            success=True,
            adapter_path=adapter_path,
            training_time_seconds=120.5,
            metrics={"num_examples": 100, "loss": 0.5},
        )

        assert result.success is True
        assert result.adapter_path == adapter_path
        assert result.model_path is None
        assert result.gguf_path is None
        assert result.error is None
        assert result.training_time_seconds == 120.5
        assert result.metrics["num_examples"] == 100
        assert result.metrics["loss"] == 0.5

    def test_failure_result(self) -> None:
        """Test failed training result."""
        result = TrainingResult(
            success=False,
            error="CUDA out of memory",
        )

        assert result.success is False
        assert result.error == "CUDA out of memory"
        assert result.adapter_path is None
        assert result.model_path is None
        assert result.metrics == {}

    def test_default_values(self) -> None:
        """Test default values for TrainingResult."""
        result = TrainingResult(success=True)

        assert result.success is True
        assert result.model_path is None
        assert result.adapter_path is None
        assert result.gguf_path is None
        assert result.error is None
        assert result.metrics == {}
        assert result.training_time_seconds == 0.0


class MockTrainer(BaseTrainer):
    """Mock trainer for testing BaseTrainer."""

    @property
    def name(self) -> str:
        return "mock"

    def train(self, qa_pairs: list[QAPair]) -> TrainingResult:
        return TrainingResult(
            success=True,
            metrics={"num_examples": len(qa_pairs)},
        )

    def export_gguf(self, adapter_path: Path, output_path: Path) -> Path:
        return output_path


class TestBaseTrainer:
    """Tests for BaseTrainer abstract class."""

    def test_trainer_initialization(self) -> None:
        """Test trainer initialization with config."""
        config = TrainingConfig(epochs=5)
        trainer = MockTrainer(config)

        assert trainer.config.epochs == 5
        assert trainer.name == "mock"

    def test_prepare_training_data(self) -> None:
        """Test QA pair to training format conversion."""
        config = TrainingConfig()
        trainer = MockTrainer(config)

        qa_pairs = [
            QAPair(
                question="What is the capital of France?",
                answer="Paris",
                source_id="test1",
            ),
            QAPair(
                question="What is 2 + 2?",
                answer="4",
                source_id="test2",
            ),
        ]

        training_data = trainer.prepare_training_data(qa_pairs)

        assert len(training_data) == 2

        # Check first example
        assert training_data[0]["messages"][0]["role"] == "user"
        assert training_data[0]["messages"][0]["content"] == "What is the capital of France?"
        assert training_data[0]["messages"][1]["role"] == "assistant"
        assert training_data[0]["messages"][1]["content"] == "Paris"

        # Check second example
        assert training_data[1]["messages"][0]["content"] == "What is 2 + 2?"
        assert training_data[1]["messages"][1]["content"] == "4"

    def test_prepare_training_data_empty(self) -> None:
        """Test prepare_training_data with empty list."""
        config = TrainingConfig()
        trainer = MockTrainer(config)

        training_data = trainer.prepare_training_data([])

        assert training_data == []
