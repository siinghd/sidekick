"""Tests for path management."""

from pathlib import Path

import pytest

from sidekick.core.paths import SidekickPaths


class TestSidekickPaths:
    """Tests for SidekickPaths class."""

    def test_default_root(self) -> None:
        """Test default root is ~/.sidekick."""
        paths = SidekickPaths()
        assert paths.root == Path.home() / ".sidekick"

    def test_custom_root(self, temp_dir: Path) -> None:
        """Test custom root directory."""
        paths = SidekickPaths(root=temp_dir)
        assert paths.root == temp_dir

    def test_data_directories(self, temp_dir: Path) -> None:
        """Test data directory paths."""
        paths = SidekickPaths(root=temp_dir)
        assert paths.data_dir == temp_dir / "data"
        assert paths.raw_dir == temp_dir / "data" / "raw"
        assert paths.processed_dir == temp_dir / "data" / "processed"
        assert paths.qa_dir == temp_dir / "data" / "qa"

    def test_model_directories(self, temp_dir: Path) -> None:
        """Test model directory paths."""
        paths = SidekickPaths(root=temp_dir)
        assert paths.models_dir == temp_dir / "models"
        assert paths.base_models_dir == temp_dir / "models" / "base"
        assert paths.adapters_dir == temp_dir / "models" / "adapters"
        assert paths.merged_dir == temp_dir / "models" / "merged"

    def test_ensure_all_creates_directories(self, temp_dir: Path) -> None:
        """Test ensure_all creates all directories."""
        paths = SidekickPaths(root=temp_dir)
        paths.ensure_all()

        assert paths.raw_dir.exists()
        assert paths.processed_dir.exists()
        assert paths.qa_dir.exists()
        assert paths.base_models_dir.exists()
        assert paths.adapters_dir.exists()
        assert paths.merged_dir.exists()
        assert paths.logs_dir.exists()

    def test_is_initialized(self, temp_dir: Path) -> None:
        """Test is_initialized checks for config file."""
        paths = SidekickPaths(root=temp_dir)
        assert not paths.is_initialized()

        # Create config file
        paths.config_file.parent.mkdir(parents=True, exist_ok=True)
        paths.config_file.touch()
        assert paths.is_initialized()
