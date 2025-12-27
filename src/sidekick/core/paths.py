"""Path management for ~/.sidekick/ directory structure."""

from pathlib import Path


class SidekickPaths:
    """Manages all paths within the ~/.sidekick/ directory structure.

    Directory structure:
        ~/.sidekick/
        ├── config.toml
        ├── data/
        │   ├── raw/           # Original ingested files (hashed)
        │   ├── processed/     # Extracted text
        │   └── qa/            # Generated QA pairs
        ├── models/
        │   ├── base/          # Downloaded base models
        │   ├── adapters/      # Trained LoRA adapters
        │   └── merged/        # Merged GGUF files
        ├── cache/
        │   └── embeddings/    # Optional: for dedup
        └── logs/
    """

    def __init__(self, root: Path | None = None) -> None:
        """Initialize paths.

        Args:
            root: Override the root directory (default: ~/.sidekick)
        """
        self._root = root or Path.home() / ".sidekick"

    @property
    def root(self) -> Path:
        """Root directory (~/.sidekick)."""
        return self._root

    @property
    def config_file(self) -> Path:
        """Path to config.toml."""
        return self._root / "config.toml"

    # Data directories
    @property
    def data_dir(self) -> Path:
        """Data directory."""
        return self._root / "data"

    @property
    def raw_dir(self) -> Path:
        """Raw ingested files directory."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Processed text directory."""
        return self.data_dir / "processed"

    @property
    def qa_dir(self) -> Path:
        """QA pairs directory."""
        return self.data_dir / "qa"

    @property
    def qa_dataset_file(self) -> Path:
        """Path to qa_dataset.jsonl."""
        return self.qa_dir / "qa_dataset.jsonl"

    @property
    def manifest_file(self) -> Path:
        """Path to manifest.json tracking processed files."""
        return self.qa_dir / "manifest.json"

    # Model directories
    @property
    def models_dir(self) -> Path:
        """Models directory."""
        return self._root / "models"

    @property
    def base_models_dir(self) -> Path:
        """Downloaded base models directory."""
        return self.models_dir / "base"

    @property
    def adapters_dir(self) -> Path:
        """Trained LoRA adapters directory."""
        return self.models_dir / "adapters"

    @property
    def merged_dir(self) -> Path:
        """Merged GGUF files directory."""
        return self.models_dir / "merged"

    # Cache directories
    @property
    def cache_dir(self) -> Path:
        """Cache directory."""
        return self._root / "cache"

    @property
    def embeddings_cache_dir(self) -> Path:
        """Embeddings cache directory."""
        return self.cache_dir / "embeddings"

    # Logs
    @property
    def logs_dir(self) -> Path:
        """Logs directory."""
        return self._root / "logs"

    @property
    def training_log(self) -> Path:
        """Training log file."""
        return self.logs_dir / "training.log"

    def ensure_all(self) -> None:
        """Create all directories if they don't exist."""
        dirs = [
            self.raw_dir,
            self.processed_dir,
            self.qa_dir,
            self.base_models_dir,
            self.adapters_dir,
            self.merged_dir,
            self.embeddings_cache_dir,
            self.logs_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def is_initialized(self) -> bool:
        """Check if sidekick has been initialized."""
        return self.config_file.exists()


# Default instance
_default_paths: SidekickPaths | None = None


def get_paths() -> SidekickPaths:
    """Get the default SidekickPaths instance."""
    global _default_paths
    if _default_paths is None:
        _default_paths = SidekickPaths()
    return _default_paths
