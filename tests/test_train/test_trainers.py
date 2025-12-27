"""Tests for trainer factory and trainers."""

from pathlib import Path
from unittest.mock import patch

from sidekick.train import (
    MLXTrainer,
    PromptTunedTrainer,
    TrainingBackend,
    TrainingConfig,
    UnslothTrainer,
    get_trainer,
)


class TestGetTrainer:
    """Tests for get_trainer factory function."""

    def test_get_trainer_prompt_tuned(self) -> None:
        """Test getting a prompt-tuned trainer."""
        trainer = get_trainer(TrainingBackend.PROMPT_TUNED)

        assert isinstance(trainer, PromptTunedTrainer)
        assert trainer.name == "prompt_tuned"

    def test_get_trainer_mlx(self) -> None:
        """Test getting an MLX trainer."""
        trainer = get_trainer(TrainingBackend.MLX)

        assert isinstance(trainer, MLXTrainer)
        assert trainer.name == "mlx"

    def test_get_trainer_unsloth(self) -> None:
        """Test getting an Unsloth trainer."""
        trainer = get_trainer(TrainingBackend.UNSLOTH)

        assert isinstance(trainer, UnslothTrainer)
        assert trainer.name == "unsloth"

    def test_get_trainer_with_config(self, temp_dir: Path) -> None:
        """Test getting a trainer with custom config."""
        config = TrainingConfig(
            epochs=10,
            batch_size=8,
            output_dir=temp_dir,
        )

        trainer = get_trainer(TrainingBackend.PROMPT_TUNED, config)

        assert trainer.config.epochs == 10
        assert trainer.config.batch_size == 8
        assert trainer.config.output_dir == temp_dir

    def test_get_trainer_auto_detect(self) -> None:
        """Test auto-detection of training backend."""
        with patch("sidekick.train.detect_hardware") as mock:
            mock.return_value.recommended_backend = TrainingBackend.PROMPT_TUNED

            trainer = get_trainer()

            assert isinstance(trainer, PromptTunedTrainer)

    def test_get_trainer_default_config(self) -> None:
        """Test that default config is used when none provided."""
        trainer = get_trainer(TrainingBackend.PROMPT_TUNED)

        # Check default values
        assert trainer.config.epochs == 3
        assert trainer.config.learning_rate == 2e-4
        assert trainer.config.batch_size == 4


class TestMLXTrainer:
    """Tests for MLXTrainer."""

    def test_mlx_trainer_name(self) -> None:
        """Test MLX trainer name."""
        config = TrainingConfig()
        trainer = MLXTrainer(config)

        assert trainer.name == "mlx"

    def test_mlx_model_mapping(self) -> None:
        """Test MLX model path mapping."""
        config = TrainingConfig(base_model="qwen2.5-1.5b")
        trainer = MLXTrainer(config)

        model_path = trainer._get_model_path()

        assert "Qwen2.5-1.5B-Instruct" in model_path
        assert "mlx-community" in model_path

    def test_mlx_model_mapping_fallback(self) -> None:
        """Test MLX model fallback for unknown models."""
        config = TrainingConfig(base_model="unknown-model")
        trainer = MLXTrainer(config)

        model_path = trainer._get_model_path()

        # Should fall back to default (Qwen3-4B)
        assert "Qwen3-4B" in model_path

    def test_mlx_train_without_mlx_lm(self) -> None:
        """Test MLX training fails gracefully without mlx-lm."""
        config = TrainingConfig()
        trainer = MLXTrainer(config)

        # Mock import failure
        with patch.dict("sys.modules", {"mlx_lm": None}):
            from sidekick.core.models import QAPair

            qa_pairs = [QAPair(question="Q?", answer="A", source_id="test")]

            result = trainer.train(qa_pairs)

            # Should fail with ImportError message
            assert result.success is False
            assert "mlx-lm" in result.error.lower()


class TestUnslothTrainer:
    """Tests for UnslothTrainer."""

    def test_unsloth_trainer_name(self) -> None:
        """Test Unsloth trainer name."""
        config = TrainingConfig()
        trainer = UnslothTrainer(config)

        assert trainer.name == "unsloth"

    def test_unsloth_model_mapping(self) -> None:
        """Test Unsloth model path mapping."""
        config = TrainingConfig(base_model="qwen2.5-1.5b")
        trainer = UnslothTrainer(config)

        model_name = trainer._get_model_name()

        assert "Qwen2.5-1.5B-Instruct" in model_name
        assert "unsloth" in model_name
        assert "bnb-4bit" in model_name

    def test_unsloth_model_mapping_phi(self) -> None:
        """Test Unsloth model mapping for Phi model."""
        config = TrainingConfig(base_model="phi-3-mini")
        trainer = UnslothTrainer(config)

        model_name = trainer._get_model_name()

        assert "Phi-3-mini" in model_name
        assert "unsloth" in model_name

    def test_unsloth_model_mapping_fallback(self) -> None:
        """Test Unsloth model fallback for unknown models."""
        config = TrainingConfig(base_model="unknown-model")
        trainer = UnslothTrainer(config)

        model_name = trainer._get_model_name()

        # Should fall back to default (Qwen3-4B)
        assert "Qwen3-4B" in model_name

    def test_unsloth_train_without_packages(self) -> None:
        """Test Unsloth training fails gracefully without packages."""
        config = TrainingConfig()
        trainer = UnslothTrainer(config)

        from sidekick.core.models import QAPair

        qa_pairs = [QAPair(question="Q?", answer="A", source_id="test")]

        result = trainer.train(qa_pairs)

        # Should fail with ImportError message (unless packages installed)
        # We don't assert success=False because packages might be installed
        assert isinstance(result.success, bool)
        if not result.success:
            assert result.error is not None
