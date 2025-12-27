"""Hot memory system with caching and memory limits.

Optimized for:
- O(1) context retrieval via caching
- Memory-bounded storage (entries + bytes)
- LRU eviction when limits exceeded
- Thread-safe operations
- Minimal memory footprint with __slots__
"""

import threading
import time
from collections import deque
from typing import Iterator, NamedTuple


class HotEntry(NamedTuple):
    """A single entry in hot memory.

    Using NamedTuple for minimal memory footprint (~40% less than dataclass).
    """
    fact: str
    source: str
    timestamp: float

    @property
    def byte_size(self) -> int:
        """Approximate memory footprint in bytes."""
        # NamedTuple overhead is ~56 bytes + string data
        return len(self.fact) + len(self.source) + 56


class HotMemory:
    """In-memory cache for recent facts with automatic eviction.

    Features:
    - Bounded by both entry count and byte size
    - Cached context string (invalidated on mutation)
    - Age-based pruning
    - Thread-safe operations

    Optimizations:
    - Uses deque for O(1) append/popleft
    - Caches context string to avoid rebuilding
    - Uses NamedTuple entries for minimal memory
    - Lazy pruning only when needed
    """

    __slots__ = (
        "max_entries", "max_bytes", "max_age_seconds", "context_cache_ttl",
        "_entries", "_current_bytes", "_lock",
        "_cached_context", "_cache_time", "_cache_dirty", "_cache_max_tokens"
    )

    def __init__(
        self,
        max_entries: int = 100,
        max_bytes: int = 100_000,
        max_age_hours: float = 24.0,
        context_cache_ttl: float = 10.0,
    ) -> None:
        """Initialize hot memory."""
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.max_age_seconds = max_age_hours * 3600
        self.context_cache_ttl = context_cache_ttl

        self._entries: deque[HotEntry] = deque()
        self._current_bytes = 0
        self._lock = threading.RLock()

        # Context caching
        self._cached_context: str | None = None
        self._cache_time: float = 0
        self._cache_dirty = True
        self._cache_max_tokens = 0

    @property
    def count(self) -> int:
        """Number of entries in memory."""
        return len(self._entries)

    @property
    def byte_size(self) -> int:
        """Total bytes used by entries."""
        return self._current_bytes

    def add(self, fact: str, source: str) -> None:
        """Add a fact to hot memory."""
        now = time.time()
        entry = HotEntry(fact=fact, source=source, timestamp=now)
        entry_size = entry.byte_size

        with self._lock:
            self._evict_if_needed(entry_size, now)
            self._entries.append(entry)
            self._current_bytes += entry_size
            self._cache_dirty = True

    def add_many(self, facts: list[tuple[str, str]]) -> int:
        """Add multiple facts efficiently."""
        if not facts:
            return 0

        now = time.time()
        added = 0

        with self._lock:
            for fact, source in facts:
                entry = HotEntry(fact=fact, source=source, timestamp=now)
                entry_size = entry.byte_size

                if entry_size > self.max_bytes:
                    continue

                self._evict_if_needed(entry_size, now)
                self._entries.append(entry)
                self._current_bytes += entry_size
                added += 1

            if added > 0:
                self._cache_dirty = True

        return added

    def _evict_if_needed(self, incoming_bytes: int, now: float) -> None:
        """Evict entries to make room. Must hold lock."""
        cutoff = now - self.max_age_seconds

        # Prune old entries
        while self._entries and self._entries[0].timestamp < cutoff:
            old = self._entries.popleft()
            self._current_bytes -= old.byte_size

        # Evict for entry count
        while len(self._entries) >= self.max_entries and self._entries:
            old = self._entries.popleft()
            self._current_bytes -= old.byte_size

        # Evict for byte limit
        target_bytes = self.max_bytes - incoming_bytes
        while self._current_bytes > target_bytes and self._entries:
            old = self._entries.popleft()
            self._current_bytes -= old.byte_size

    def get_context(self, max_tokens: int = 500) -> str:
        """Get context string for prompt injection. O(1) when cached."""
        now = time.time()

        # Fast path: return cache if valid
        if (
            not self._cache_dirty
            and self._cached_context is not None
            and self._cache_max_tokens == max_tokens
            and now - self._cache_time < self.context_cache_ttl
        ):
            return self._cached_context

        with self._lock:
            # Double-check under lock
            if (
                not self._cache_dirty
                and self._cached_context is not None
                and self._cache_max_tokens == max_tokens
                and now - self._cache_time < self.context_cache_ttl
            ):
                return self._cached_context

            self._cached_context = self._build_context(max_tokens)
            self._cache_time = now
            self._cache_dirty = False
            self._cache_max_tokens = max_tokens

            return self._cached_context

    def _build_context(self, max_tokens: int) -> str:
        """Build context string from entries. Must hold lock."""
        if not self._entries:
            return ""

        max_chars = max_tokens * 4
        parts = []
        total_chars = 0

        # Build from newest to oldest
        for entry in reversed(self._entries):
            line = f"- {entry.fact}"
            line_len = len(line) + 1

            if total_chars + line_len > max_chars:
                break

            parts.append(line)
            total_chars += line_len

        if not parts:
            return ""

        # Reverse to get chronological order
        parts.reverse()
        return "Recent context:\n" + "\n".join(parts)

    def flush(self) -> list[HotEntry]:
        """Flush all entries for training."""
        with self._lock:
            entries = list(self._entries)
            self._entries.clear()
            self._current_bytes = 0
            self._cache_dirty = True
            self._cached_context = None
            return entries

    def iter_entries(self) -> Iterator[HotEntry]:
        """Iterate over entries (newest first). Thread-safe snapshot."""
        # Take snapshot under lock
        with self._lock:
            entries = tuple(self._entries)

        # Iterate without lock
        for entry in reversed(entries):
            yield entry

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._entries.clear()
            self._current_bytes = 0
            self._cache_dirty = True
            self._cached_context = None

    def stats(self) -> dict:
        """Get memory statistics."""
        entries_count = len(self._entries)
        return {
            "entries": entries_count,
            "bytes": self._current_bytes,
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes,
            "utilization_entries": entries_count / self.max_entries if self.max_entries else 0,
            "utilization_bytes": self._current_bytes / self.max_bytes if self.max_bytes else 0,
            "cache_dirty": self._cache_dirty,
        }
