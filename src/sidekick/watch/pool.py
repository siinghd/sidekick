"""Model pool with lazy loading and TTL-based unloading.

Optimized for:
- Lazy model loading (first query pays cost)
- TTL-based unloading (default 5 min idle)
- Lock-free fast path for loaded models
- Minimal memory with __slots__
"""

import asyncio
import threading
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class ModelPool:
    """Pool for managing model lifecycle with TTL.

    Optimizations:
    - Lock-free fast path when model is loaded and fresh
    - Double-checked locking for thread safety
    - Lazy loading in thread pool for async
    """

    __slots__ = (
        "loader", "ttl", "unloader",
        "_model", "_last_used", "_lock", "_async_lock",
        "_cleanup_task", "_running"
    )

    def __init__(
        self,
        loader: Callable[[], Any],
        ttl_seconds: float = 300,
        unloader: Callable[[Any], None] | None = None,
    ) -> None:
        self.loader = loader
        self.ttl = ttl_seconds
        self.unloader = unloader

        self._model: Any = None
        self._last_used: float = 0
        self._lock = threading.RLock()
        self._async_lock: asyncio.Lock | None = None

        self._cleanup_task: asyncio.Task | None = None
        self._running = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded. Lock-free read."""
        return self._model is not None

    @property
    def idle_seconds(self) -> float:
        """Seconds since last use."""
        if self._model is None:
            return 0
        return time.time() - self._last_used

    def get_sync(self) -> Any:
        """Get model synchronously (blocks on load)."""
        now = time.time()

        # Fast path: model loaded and fresh
        model = self._model
        if model is not None and now - self._last_used <= self.ttl:
            self._last_used = now
            return model

        with self._lock:
            # Double-check: TTL expired?
            if self._model is not None and now - self._last_used > self.ttl:
                self._unload_locked()

            # Load if needed
            if self._model is None:
                self._model = self.loader()

            self._last_used = now
            return self._model

    async def get(self) -> Any:
        """Get model asynchronously (non-blocking load)."""
        now = time.time()

        # Fast path: model loaded and fresh
        model = self._model
        if model is not None and now - self._last_used <= self.ttl:
            self._last_used = now
            return model

        # Initialize async lock lazily
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        async with self._async_lock:
            # Double-check under lock
            if self._model is not None and now - self._last_used > self.ttl:
                with self._lock:
                    self._unload_locked()

            if self._model is None:
                self._model = await asyncio.to_thread(self.loader)

            self._last_used = time.time()
            return self._model

    def unload(self) -> None:
        """Explicitly unload the model."""
        with self._lock:
            self._unload_locked()

    def _unload_locked(self) -> None:
        """Unload model. Must hold lock."""
        if self._model is not None:
            if self.unloader:
                try:
                    self.unloader(self._model)
                except Exception:
                    pass
            self._model = None

    async def start_cleanup_task(self, check_interval: float = 60) -> None:
        """Start background task to auto-unload idle models."""
        if self._running:
            return
        self._running = True
        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop(check_interval)
        )

    async def stop_cleanup_task(self) -> None:
        """Stop the background cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _cleanup_loop(self, interval: float) -> None:
        """Background loop to check and unload idle models."""
        while self._running:
            await asyncio.sleep(interval)

            if self._model is not None and time.time() - self._last_used > self.ttl:
                with self._lock:
                    if self._model is not None and time.time() - self._last_used > self.ttl:
                        self._unload_locked()

    def stats(self) -> dict:
        """Get pool statistics."""
        return {
            "loaded": self._model is not None,
            "idle_seconds": self.idle_seconds,
            "ttl_seconds": self.ttl,
        }


class MultiModelPool:
    """Pool for managing multiple named models with LRU eviction."""

    __slots__ = ("ttl", "max_models", "_pools", "_lock")

    def __init__(
        self,
        ttl_seconds: float = 300,
        max_models: int = 3,
    ) -> None:
        self.ttl = ttl_seconds
        self.max_models = max_models
        self._pools: dict[str, ModelPool] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        loader: Callable[[], Any],
        unloader: Callable[[Any], None] | None = None,
    ) -> None:
        """Register a model loader."""
        with self._lock:
            self._pools[name] = ModelPool(
                loader=loader,
                ttl_seconds=self.ttl,
                unloader=unloader,
            )

    async def get(self, name: str) -> Any:
        """Get a model by name with LRU eviction."""
        with self._lock:
            if name not in self._pools:
                raise KeyError(f"Model not registered: {name}")
            pool = self._pools[name]

        await self._maybe_evict(name)
        return await pool.get()

    async def _maybe_evict(self, keep: str) -> None:
        """Evict oldest models if over limit."""
        with self._lock:
            loaded = [
                (n, p) for n, p in self._pools.items()
                if p.is_loaded and n != keep
            ]

            if len(loaded) >= self.max_models:
                # Sort by idle time (most idle first)
                loaded.sort(key=lambda x: x[1].idle_seconds, reverse=True)

                # Evict oldest
                for name, pool in loaded[: len(loaded) - self.max_models + 1]:
                    pool.unload()

    def unload_all(self) -> None:
        """Unload all models."""
        with self._lock:
            for pool in self._pools.values():
                pool.unload()

    def stats(self) -> dict:
        """Get all pool statistics."""
        with self._lock:
            return {name: pool.stats() for name, pool in self._pools.items()}
