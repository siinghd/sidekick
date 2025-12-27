"""Background trainer using subprocess for non-blocking operation.

Optimized for:
- Non-blocking training (separate process)
- Status updates via queue
- Graceful handling of concurrent training requests
- Safe data handoff between processes
"""

import asyncio
import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sidekick.watch.memory import HotEntry


@dataclass
class TrainingStatus:
    """Status update from training process."""

    stage: str  # "starting", "loading", "training", "saving", "complete", "error"
    message: str
    progress: float = 0.0  # 0.0 to 1.0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


def _train_worker(
    sidekick_dir: Path,
    data_path: Path,
    status_queue: mp.Queue,
) -> None:
    """Training worker that runs in separate process.

    Args:
        sidekick_dir: Path to .sidekick directory
        data_path: Path to pending training data JSON
        status_queue: Queue for status updates
    """
    try:
        status_queue.put(TrainingStatus(
            stage="starting",
            message="Initializing training...",
        ))

        # Load training data
        status_queue.put(TrainingStatus(
            stage="loading",
            message="Loading training data...",
            progress=0.1,
        ))

        with open(data_path) as f:
            entries_data = json.load(f)

        if not entries_data:
            status_queue.put(TrainingStatus(
                stage="complete",
                message="No data to train on",
                progress=1.0,
            ))
            return

        # Convert to QA pairs format
        from sidekick.core.models import QAPair
        from sidekick.core.paths import SidekickPaths
        from sidekick.train import get_trainer, TrainingConfig

        paths = SidekickPaths(root=sidekick_dir)

        # Create QA pairs from hot entries
        qa_pairs = []
        for entry in entries_data:
            # Hot entries are facts, we create simple Q&A
            fact = entry.get("fact", "")
            source = entry.get("source", "unknown")

            if fact:
                qa_pairs.append(QAPair(
                    question=f"What do you know about: {source}?",
                    answer=fact,
                    source_id=source,
                ))

        if not qa_pairs:
            status_queue.put(TrainingStatus(
                stage="complete",
                message="No valid QA pairs generated",
                progress=1.0,
            ))
            return

        status_queue.put(TrainingStatus(
            stage="training",
            message=f"Training on {len(qa_pairs)} facts...",
            progress=0.3,
        ))

        # Get trainer (will auto-detect hardware)
        config = TrainingConfig(
            output_dir=paths.adapters_dir,
            epochs=1,  # Quick incremental training
        )
        trainer = get_trainer(config=config)

        status_queue.put(TrainingStatus(
            stage="training",
            message=f"Using {trainer.name} backend...",
            progress=0.5,
        ))

        # Train
        result = trainer.train(qa_pairs)

        if result.success:
            status_queue.put(TrainingStatus(
                stage="saving",
                message="Saving model...",
                progress=0.9,
            ))

            status_queue.put(TrainingStatus(
                stage="complete",
                message=f"Training complete in {result.training_time_seconds:.1f}s",
                progress=1.0,
            ))
        else:
            status_queue.put(TrainingStatus(
                stage="error",
                message=f"Training failed: {result.error}",
            ))

    except Exception as e:
        status_queue.put(TrainingStatus(
            stage="error",
            message=f"Training error: {e}",
        ))

    finally:
        # Clean up data file
        try:
            data_path.unlink(missing_ok=True)
        except Exception:
            pass


class BackgroundTrainer:
    """Manages background training in a separate process."""

    def __init__(self, sidekick_dir: Path) -> None:
        """Initialize background trainer.

        Args:
            sidekick_dir: Path to .sidekick directory
        """
        self.sidekick_dir = Path(sidekick_dir)
        self.cache_dir = self.sidekick_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._process: mp.Process | None = None
        self._status_queue: mp.Queue | None = None
        self._running = False

    @property
    def is_training(self) -> bool:
        """Check if training is in progress."""
        return self._process is not None and self._process.is_alive()

    async def train(
        self,
        entries: list[HotEntry],
        on_status: Any = None,
    ) -> bool:
        """Start background training on hot entries.

        Args:
            entries: Hot entries to train on
            on_status: Optional callback for status updates

        Returns:
            True if training started, False if already running
        """
        if self.is_training:
            return False

        if not entries:
            return False

        # Serialize entries to file
        data_path = self.cache_dir / f"pending_train_{int(time.time())}.json"
        entries_data = [
            {"fact": e.fact, "source": e.source, "timestamp": e.timestamp}
            for e in entries
        ]

        with open(data_path, "w") as f:
            json.dump(entries_data, f)

        # Create status queue
        self._status_queue = mp.Queue()

        # Start training process
        self._process = mp.Process(
            target=_train_worker,
            args=(self.sidekick_dir, data_path, self._status_queue),
        )
        self._process.start()
        self._running = True

        # Monitor in background
        asyncio.create_task(self._monitor(on_status))

        return True

    async def _monitor(self, on_status: Any = None) -> None:
        """Monitor training process and forward status updates."""
        while self._running and self._process and self._process.is_alive():
            # Check for status updates
            while self._status_queue and not self._status_queue.empty():
                try:
                    status = self._status_queue.get_nowait()
                    if on_status:
                        if asyncio.iscoroutinefunction(on_status):
                            await on_status(status)
                        else:
                            on_status(status)
                except Exception:
                    pass

            await asyncio.sleep(0.5)

        # Get final status updates
        if self._status_queue:
            while not self._status_queue.empty():
                try:
                    status = self._status_queue.get_nowait()
                    if on_status:
                        if asyncio.iscoroutinefunction(on_status):
                            await on_status(status)
                        else:
                            on_status(status)
                except Exception:
                    pass

        self._running = False

    def cancel(self) -> None:
        """Cancel training if running."""
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()

        self._running = False
        self._process = None

    def cleanup(self) -> None:
        """Clean up resources."""
        self.cancel()
        if self._status_queue:
            self._status_queue.close()
            self._status_queue = None
