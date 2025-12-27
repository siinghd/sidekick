"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path
from typing import Generator

import pytest

from sidekick.core.config import Settings
from sidekick.core.paths import SidekickPaths


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def test_paths(temp_dir: Path) -> SidekickPaths:
    """Create a SidekickPaths instance using a temp directory."""
    paths = SidekickPaths(root=temp_dir)
    paths.ensure_all()
    return paths


@pytest.fixture
def test_settings(test_paths: SidekickPaths) -> Settings:
    """Create a Settings instance for testing."""
    settings = Settings()
    settings.save(test_paths.config_file)
    return settings
