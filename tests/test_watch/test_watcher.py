"""Tests for file watcher module."""

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock

from sidekick.watch.watcher import (
    BatchProcessor,
    FileChange,
    FileDebouncer,
    FileWatcher,
    RateLimiter,
)


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_allows_within_limit(self) -> None:
        """Test that calls within limit are allowed."""
        limiter = RateLimiter(calls_per_minute=10)

        for _ in range(10):
            assert limiter.acquire() is True

    def test_blocks_over_limit(self) -> None:
        """Test that calls over limit are blocked."""
        limiter = RateLimiter(calls_per_minute=5)

        for _ in range(5):
            limiter.acquire()

        assert limiter.acquire() is False

    def test_remaining_count(self) -> None:
        """Test remaining count calculation."""
        limiter = RateLimiter(calls_per_minute=10)

        assert limiter.remaining == 10

        limiter.acquire()
        limiter.acquire()

        assert limiter.remaining == 8


class TestFileDebouncer:
    """Tests for FileDebouncer class."""

    def test_first_access_allowed(self) -> None:
        """Test that first access is always allowed."""
        debouncer = FileDebouncer(debounce_ms=1000)
        path = Path("/test/file.txt")

        assert debouncer.should_process(path) is True

    def test_rapid_access_blocked(self) -> None:
        """Test that rapid repeated access is blocked."""
        debouncer = FileDebouncer(debounce_ms=1000)
        path = Path("/test/file.txt")

        debouncer.should_process(path)
        assert debouncer.should_process(path) is False

    def test_different_files_independent(self) -> None:
        """Test that different files are debounced independently."""
        debouncer = FileDebouncer(debounce_ms=1000)
        path1 = Path("/test/file1.txt")
        path2 = Path("/test/file2.txt")

        debouncer.should_process(path1)
        assert debouncer.should_process(path2) is True

    def test_cleanup_removes_old_entries(self) -> None:
        """Test that cleanup removes old entries when threshold met."""
        debouncer = FileDebouncer(debounce_ms=100)

        # Cleanup only runs when there are >= 100 entries (optimization)
        for i in range(110):
            debouncer.should_process(Path(f"/test/file{i}.txt"))

        time.sleep(0.01)
        debouncer.cleanup(max_age=0.001)

        # Should be able to process again after cleanup removed old entries
        assert debouncer.should_process(Path("/test/file0.txt")) is True


class TestFileChange:
    """Tests for FileChange dataclass."""

    def test_creation(self) -> None:
        """Test FileChange creation."""
        change = FileChange(
            path=Path("/test/file.txt"),
            content="Hello world",
            timestamp=time.time(),
        )

        assert change.path == Path("/test/file.txt")
        assert change.content == "Hello world"


class TestFileWatcher:
    """Tests for FileWatcher class."""

    def test_initialization(self, temp_dir: Path) -> None:
        """Test watcher initialization."""
        watcher = FileWatcher(
            paths=[temp_dir],
            patterns=["*.md", "*.txt"],
            max_file_size_kb=500,
        )

        assert len(watcher.paths) == 1
        assert watcher.max_file_size == 500 * 1024
        # Pattern matching is encapsulated in _matcher (optimization)
        assert watcher._should_watch(temp_dir / "test.md") is True

    def test_should_watch_matches(self, temp_dir: Path) -> None:
        """Test pattern matching for files."""
        watcher = FileWatcher(
            paths=[temp_dir],
            patterns=["*.md", "*.txt"],
            ignore=["node_modules"],
        )

        assert watcher._should_watch(temp_dir / "test.md") is True
        assert watcher._should_watch(temp_dir / "test.txt") is True
        assert watcher._should_watch(temp_dir / "test.py") is False

    def test_should_watch_ignores(self, temp_dir: Path) -> None:
        """Test ignore patterns."""
        watcher = FileWatcher(
            paths=[temp_dir],
            patterns=["*.md"],
            ignore=["node_modules", ".git"],
        )

        assert watcher._should_watch(temp_dir / "node_modules" / "pkg" / "README.md") is False
        assert watcher._should_watch(temp_dir / ".git" / "config") is False


class TestBatchProcessor:
    """Tests for BatchProcessor class."""

    def test_batch_collection(self) -> None:
        """Test that items are collected into batches."""
        batches_received = []

        async def process_batch(batch):
            batches_received.append(batch)

        async def run_test():
            processor = BatchProcessor(
                process_fn=process_batch,
                batch_size=3,
                max_wait_ms=100,
            )

            await processor.start()

            # Add items
            for i in range(3):
                await processor.add(FileChange(
                    path=Path(f"/test/file{i}.txt"),
                    content=f"Content {i}",
                    timestamp=time.time(),
                ))

            # Wait for processing
            await asyncio.sleep(0.2)
            await processor.stop()

        asyncio.run(run_test())

        assert len(batches_received) >= 1
        assert sum(len(b) for b in batches_received) == 3

    def test_batch_timeout(self) -> None:
        """Test that partial batches are processed on timeout."""
        batches_received = []

        async def process_batch(batch):
            batches_received.append(batch)

        async def run_test():
            processor = BatchProcessor(
                process_fn=process_batch,
                batch_size=10,  # Large batch size
                max_wait_ms=100,  # Short timeout
            )

            await processor.start()

            # Add only 2 items (less than batch size)
            await processor.add(FileChange(
                path=Path("/test/file1.txt"),
                content="Content 1",
                timestamp=time.time(),
            ))
            await processor.add(FileChange(
                path=Path("/test/file2.txt"),
                content="Content 2",
                timestamp=time.time(),
            ))

            # Wait for timeout
            await asyncio.sleep(0.3)
            await processor.stop()

        asyncio.run(run_test())

        # Should have processed despite not reaching batch size
        assert len(batches_received) >= 1
