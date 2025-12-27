"""File watcher with rate limiting and batching.

Optimized for:
- Per-file debouncing with O(1) lookup
- Token bucket rate limiting
- Batch processing for efficient LLM calls
- Memory-efficient file reading
- Pre-compiled glob patterns
"""

import asyncio
import fnmatch
import re
import time
from collections import deque
from pathlib import Path
from typing import Callable, Coroutine, NamedTuple


class FileChange(NamedTuple):
    """A detected file change. Using NamedTuple for efficiency."""
    path: Path
    content: str
    timestamp: float


class RateLimiter:
    """Token bucket rate limiter with O(1) operations."""

    __slots__ = ("calls_per_minute", "calls", "_window_start", "_call_count")

    def __init__(self, calls_per_minute: int = 30) -> None:
        self.calls_per_minute = calls_per_minute
        self.calls: deque[float] = deque()
        # Optimization: track window for faster cleanup
        self._window_start: float = 0
        self._call_count: int = 0

    def acquire(self) -> bool:
        """Try to acquire a rate limit token. O(1) amortized."""
        now = time.time()
        window_start = now - 60

        # Fast path: if window hasn't moved much, just check count
        if self._window_start >= window_start:
            if self._call_count >= self.calls_per_minute:
                return False
        else:
            # Clean old entries
            while self.calls and self.calls[0] < window_start:
                self.calls.popleft()
            self._call_count = len(self.calls)
            self._window_start = window_start

            if self._call_count >= self.calls_per_minute:
                return False

        self.calls.append(now)
        self._call_count += 1
        return True

    @property
    def remaining(self) -> int:
        """Remaining calls in current window."""
        now = time.time()
        window_start = now - 60
        while self.calls and self.calls[0] < window_start:
            self.calls.popleft()
        self._call_count = len(self.calls)
        return max(0, self.calls_per_minute - self._call_count)


class FileDebouncer:
    """Per-file debouncing with O(1) lookup."""

    __slots__ = ("debounce_seconds", "_last_processed")

    def __init__(self, debounce_ms: int = 2000) -> None:
        self.debounce_seconds = debounce_ms / 1000
        self._last_processed: dict[str, float] = {}  # Use str keys for faster hashing

    def should_process(self, path: Path) -> bool:
        """Check if file should be processed. O(1)."""
        now = time.time()
        key = str(path)
        last = self._last_processed.get(key, 0)

        if now - last < self.debounce_seconds:
            return False

        self._last_processed[key] = now
        return True

    def cleanup(self, max_age: float = 3600) -> None:
        """Remove old entries to prevent memory leak."""
        if len(self._last_processed) < 100:
            return  # Skip cleanup for small dicts

        now = time.time()
        cutoff = now - max_age
        # Create new dict instead of modifying during iteration
        self._last_processed = {
            k: v for k, v in self._last_processed.items()
            if v >= cutoff
        }


class PatternMatcher:
    """Pre-compiled pattern matching for efficiency."""

    __slots__ = ("_include_patterns", "_ignore_set", "_ignore_patterns")

    def __init__(self, patterns: list[str], ignore: list[str]) -> None:
        # Pre-compile include patterns to regex
        self._include_patterns = [
            re.compile(fnmatch.translate(p))
            for p in patterns
        ]
        # Use set for O(1) substring check on common ignores
        self._ignore_set = frozenset(ignore)
        # Also compile as patterns for edge cases
        self._ignore_patterns = [
            re.compile(fnmatch.translate(f"*{p}*"))
            for p in ignore
        ]

    def matches(self, path: Path) -> bool:
        """Check if path matches include patterns and isn't ignored. O(1) average."""
        path_str = str(path)

        # Fast ignore check using set
        for ignore in self._ignore_set:
            if ignore in path_str:
                return False

        # Check include patterns
        name = path.name
        return any(p.match(name) for p in self._include_patterns)


class BatchProcessor:
    """Batch file changes for efficient LLM processing."""

    __slots__ = (
        "process_fn", "batch_size", "max_wait",
        "queue", "_running", "_task"
    )

    def __init__(
        self,
        process_fn: Callable[[list[FileChange]], Coroutine],
        batch_size: int = 10,
        max_wait_ms: int = 5000,
    ) -> None:
        self.process_fn = process_fn
        self.batch_size = batch_size
        self.max_wait = max_wait_ms / 1000
        self.queue: asyncio.Queue[FileChange] = asyncio.Queue()
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the batch processor."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the batch processor."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def add(self, change: FileChange) -> None:
        """Add a file change to the queue."""
        await self.queue.put(change)

    async def _run(self) -> None:
        """Main batch processing loop."""
        while self._running:
            batch: list[FileChange] = []
            deadline = time.monotonic() + self.max_wait

            # Collect batch
            while len(batch) < self.batch_size:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break

                try:
                    change = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=max(0.1, timeout),
                    )
                    batch.append(change)
                except asyncio.TimeoutError:
                    break

            if batch:
                try:
                    await self.process_fn(batch)
                except Exception:
                    pass  # Log but don't crash


class FileWatcher:
    """Watch files for changes with rate limiting and batching."""

    __slots__ = (
        "paths", "max_file_size", "max_read_chars",
        "debouncer", "rate_limiter", "batch_processor",
        "batch_size", "batch_wait_ms", "_matcher",
        "_running", "_semaphore"
    )

    def __init__(
        self,
        paths: list[Path],
        patterns: list[str] | None = None,
        ignore: list[str] | None = None,
        max_file_size_kb: int = 1000,
        max_read_chars: int = 10000,
        debounce_ms: int = 2000,
        rate_limit_per_minute: int = 30,
        batch_size: int = 10,
        batch_wait_ms: int = 5000,
    ) -> None:
        self.paths = [Path(p).resolve() for p in paths]
        self.max_file_size = max_file_size_kb * 1024
        self.max_read_chars = max_read_chars

        patterns = patterns or ["*.md", "*.txt", "*.json"]
        ignore = ignore or ["node_modules", ".git", "venv", "__pycache__"]

        self._matcher = PatternMatcher(patterns, ignore)
        self.debouncer = FileDebouncer(debounce_ms)
        self.rate_limiter = RateLimiter(rate_limit_per_minute)
        self.batch_processor: BatchProcessor | None = None

        self.batch_size = batch_size
        self.batch_wait_ms = batch_wait_ms

        self._running = False
        self._semaphore = asyncio.Semaphore(5)

    def _should_watch(self, path: Path) -> bool:
        """Check if a file should be watched."""
        return self._matcher.matches(path)

    async def _read_file_safe(self, path: Path) -> str | None:
        """Read file content safely with size limits."""
        try:
            stat = path.stat()
            if stat.st_size > self.max_file_size:
                return None

            async with self._semaphore:
                return await asyncio.to_thread(self._read_sync, path)
        except (OSError, IOError):
            return None

    def _read_sync(self, path: Path) -> str:
        """Synchronous file read with limit."""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(self.max_read_chars)

    async def _handle_change(self, path: Path) -> None:
        """Handle a single file change."""
        if not self._should_watch(path):
            return

        if not self.debouncer.should_process(path):
            return

        if not self.rate_limiter.acquire():
            return

        content = await self._read_file_safe(path)
        if content is None:
            return

        change = FileChange(
            path=path,
            content=content,
            timestamp=time.time(),
        )

        if self.batch_processor:
            await self.batch_processor.add(change)

    async def watch(
        self,
        on_batch: Callable[[list[FileChange]], Coroutine],
    ) -> None:
        """Start watching files."""
        try:
            from watchfiles import awatch, Change
        except ImportError:
            raise ImportError(
                "watchfiles not installed. Install with: pip install watchfiles"
            )

        self.batch_processor = BatchProcessor(
            process_fn=on_batch,
            batch_size=self.batch_size,
            max_wait_ms=self.batch_wait_ms,
        )
        await self.batch_processor.start()

        self._running = True
        cleanup_counter = 0

        try:
            async for changes in awatch(*self.paths):
                if not self._running:
                    break

                for change_type, path_str in changes:
                    if change_type in (Change.modified, Change.added):
                        path = Path(path_str)
                        if path.is_file():
                            await self._handle_change(path)

                # Periodic cleanup (every 100 change events)
                cleanup_counter += 1
                if cleanup_counter >= 100:
                    self.debouncer.cleanup()
                    cleanup_counter = 0

        finally:
            self._running = False
            if self.batch_processor:
                await self.batch_processor.stop()

    def stop(self) -> None:
        """Stop watching."""
        self._running = False
