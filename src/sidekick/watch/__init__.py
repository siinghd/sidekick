"""Watch module for real-time file monitoring and incremental learning.

This module provides:
- Hot memory: Bounded in-memory cache of recent facts
- File watcher: Rate-limited, batched file change monitoring
- Model pool: Lazy loading with TTL-based unloading
- Background trainer: Non-blocking incremental training
"""

from sidekick.watch.memory import HotEntry, HotMemory
from sidekick.watch.pool import ModelPool, MultiModelPool
from sidekick.watch.trainer import BackgroundTrainer, TrainingStatus
from sidekick.watch.watcher import (
    BatchProcessor,
    FileChange,
    FileDebouncer,
    FileWatcher,
    RateLimiter,
)

__all__ = [
    # Memory
    "HotEntry",
    "HotMemory",
    # Watcher
    "BatchProcessor",
    "FileChange",
    "FileDebouncer",
    "FileWatcher",
    "RateLimiter",
    # Pool
    "ModelPool",
    "MultiModelPool",
    # Trainer
    "BackgroundTrainer",
    "TrainingStatus",
]
