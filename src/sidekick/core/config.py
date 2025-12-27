"""Configuration management for Sidekick."""

from enum import Enum
from pathlib import Path
from typing import Any

import toml
from pydantic import BaseModel, Field

from sidekick.core.paths import get_paths


class TrainingMode(str, Enum):
    """Training mode options."""

    AUTO = "auto"
    LOCAL = "local"
    CLOUD = "cloud"
    PROMPT = "prompt"


class InferenceBackend(str, Enum):
    """Inference backend options."""

    AUTO = "auto"
    LLAMACPP = "llamacpp"
    MLX = "mlx"


class CloudProvider(str, Enum):
    """Cloud training provider options."""

    COLAB = "colab"
    MODAL = "modal"


# Available model presets (Qwen3 - Dec 2025)
MODEL_PRESETS = {
    "tiny": "qwen3-0.6b",      # Ultra-light, ~400MB, strong for sub-1B
    "small": "qwen3-4b",        # Sweet spot, ~2.5GB, rivals 72B quality
    "medium": "qwen3-8b",       # Best quality, ~5GB
    # Legacy Qwen2.5 models
    "qwen2.5-0.5b": "qwen2.5-0.5b",
    "qwen2.5-1.5b": "qwen2.5-1.5b",
    "qwen2.5-3b": "qwen2.5-3b",
}

DEFAULT_MODEL = "qwen3-4b"  # Best balance of quality and speed


class ModelConfig(BaseModel):
    """Model configuration."""

    base: str = Field(default=DEFAULT_MODEL, description="Base model to use")
    quantization: str = Field(default="q4_k_m", description="Quantization for inference")


class TrainingConfig(BaseModel):
    """Training configuration."""

    mode: TrainingMode = Field(default=TrainingMode.AUTO, description="Training mode")
    epochs: int = Field(default=3, ge=1, le=100, description="Number of training epochs")
    learning_rate: float = Field(default=2e-4, description="Learning rate")
    batch_size: int = Field(default=4, ge=1, description="Batch size")
    lora_rank: int = Field(default=16, description="LoRA rank")
    lora_alpha: int = Field(default=32, description="LoRA alpha")


class InferenceConfig(BaseModel):
    """Inference configuration."""

    backend: InferenceBackend = Field(
        default=InferenceBackend.AUTO, description="Inference backend"
    )
    max_tokens: int = Field(default=200, ge=1, description="Maximum tokens in response")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Sampling temperature")
    context_length: int = Field(default=2048, ge=512, description="Context window size")


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8787, ge=1, le=65535, description="Server port")
    cors_origins: list[str] = Field(default=["*"], description="CORS allowed origins")


class CloudConfig(BaseModel):
    """Cloud training configuration."""

    provider: CloudProvider = Field(default=CloudProvider.COLAB, description="Cloud provider")
    encrypt_data: bool = Field(default=True, description="Encrypt data before upload")


class PrivacyConfig(BaseModel):
    """Privacy configuration."""

    keep_raw_data: bool = Field(default=False, description="Keep raw data after processing")
    local_only: bool = Field(default=True, description="Never send data to cloud")


class WatchConfig(BaseModel):
    """Watch mode configuration."""

    debounce_ms: int = Field(default=2000, ge=100, description="Debounce delay in ms")
    patterns: list[str] = Field(
        default=["*.md", "*.txt", "*.json"],
        description="File patterns to watch"
    )
    ignore: list[str] = Field(
        default=["node_modules", ".git", "venv", "__pycache__"],
        description="Patterns to ignore"
    )
    max_file_size_kb: int = Field(default=1000, description="Skip files larger than this")
    rate_limit_per_minute: int = Field(default=30, description="Max LLM calls per minute")
    batch_size: int = Field(default=10, description="Files per batch")
    batch_wait_ms: int = Field(default=5000, description="Max wait for batch to fill")


class HotMemoryConfig(BaseModel):
    """Hot memory configuration."""

    max_entries: int = Field(default=100, description="Maximum entries in hot memory")
    max_bytes: int = Field(default=100000, description="Maximum bytes in hot memory")
    max_age_hours: float = Field(default=24.0, description="Max age of entries in hours")
    context_cache_ttl: float = Field(default=10.0, description="Context cache TTL in seconds")


class ModelPoolConfig(BaseModel):
    """Model pool configuration."""

    unload_after_idle_seconds: float = Field(
        default=300.0,
        description="Unload model after this many seconds of inactivity"
    )


class Settings(BaseModel):
    """Main settings for Sidekick."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    cloud: CloudConfig = Field(default_factory=CloudConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    watch: WatchConfig = Field(default_factory=WatchConfig)
    hot_memory: HotMemoryConfig = Field(default_factory=HotMemoryConfig)
    model_pool: ModelPoolConfig = Field(default_factory=ModelPoolConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        """Load settings from config file.

        Args:
            path: Path to config file (default: ~/.sidekick/config.toml)

        Returns:
            Settings instance
        """
        if path is None:
            path = get_paths().config_file

        if not path.exists():
            return cls()

        data = toml.load(path)
        return cls.model_validate(data)

    def save(self, path: Path | None = None) -> None:
        """Save settings to config file.

        Args:
            path: Path to config file (default: ~/.sidekick/config.toml)
        """
        if path is None:
            path = get_paths().config_file

        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict with enum values as strings
        data = self._to_serializable_dict()
        with open(path, "w") as f:
            toml.dump(data, f)

    def _to_serializable_dict(self) -> dict[str, Any]:
        """Convert settings to a serializable dict."""
        data = self.model_dump()
        return self._convert_enums(data)

    def _convert_enums(self, obj: Any) -> Any:
        """Recursively convert enum values to strings."""
        if isinstance(obj, dict):
            return {k: self._convert_enums(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_enums(v) for v in obj]
        elif isinstance(obj, Enum):
            return obj.value
        return obj

    def get(self, key: str) -> Any:
        """Get a nested config value by dot-separated key.

        Args:
            key: Dot-separated key (e.g., "model.base")

        Returns:
            The value at the key path

        Raises:
            KeyError: If key not found
        """
        parts = key.split(".")
        obj: Any = self
        for part in parts:
            if isinstance(obj, BaseModel):
                obj = getattr(obj, part)
            elif isinstance(obj, dict):
                obj = obj[part]
            else:
                raise KeyError(f"Cannot access '{part}' on {type(obj)}")
        return obj

    def set(self, key: str, value: Any) -> None:
        """Set a nested config value by dot-separated key.

        Args:
            key: Dot-separated key (e.g., "model.base")
            value: Value to set
        """
        parts = key.split(".")
        obj: Any = self
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)


# Singleton instance
_settings: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    """Get the global Settings instance.

    Args:
        reload: Force reload from disk

    Returns:
        Settings instance
    """
    global _settings
    if _settings is None or reload:
        _settings = Settings.load()
    return _settings
