"""Tests for configuration management."""

from pathlib import Path

import pytest

from sidekick.core.config import (
    InferenceBackend,
    Settings,
    TrainingMode,
)


class TestSettings:
    """Tests for Settings class."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        settings = Settings()
        assert settings.model.base == "qwen3-4b"
        assert settings.model.quantization == "q4_k_m"
        assert settings.training.mode == TrainingMode.AUTO
        assert settings.training.epochs == 3
        assert settings.inference.backend == InferenceBackend.AUTO
        assert settings.server.port == 8787

    def test_save_and_load(self, temp_dir: Path) -> None:
        """Test saving and loading configuration."""
        config_path = temp_dir / "config.toml"

        # Save
        settings = Settings()
        settings.model.base = "phi-3-mini"
        settings.training.epochs = 5
        settings.save(config_path)

        # Load
        loaded = Settings.load(config_path)
        assert loaded.model.base == "phi-3-mini"
        assert loaded.training.epochs == 5

    def test_get_nested_value(self) -> None:
        """Test getting nested config values."""
        settings = Settings()
        assert settings.get("model.base") == "qwen3-4b"
        assert settings.get("training.epochs") == 3
        assert settings.get("server.port") == 8787

    def test_set_nested_value(self) -> None:
        """Test setting nested config values."""
        settings = Settings()
        settings.set("model.base", "phi-3-mini")
        assert settings.model.base == "phi-3-mini"

        settings.set("training.epochs", 10)
        assert settings.training.epochs == 10

    def test_load_nonexistent_returns_defaults(self, temp_dir: Path) -> None:
        """Test loading from nonexistent file returns defaults."""
        config_path = temp_dir / "nonexistent.toml"
        settings = Settings.load(config_path)
        assert settings.model.base == "qwen3-4b"
