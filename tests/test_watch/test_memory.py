"""Tests for hot memory module."""

import time
from pathlib import Path

from sidekick.watch.memory import HotEntry, HotMemory


class TestHotEntry:
    """Tests for HotEntry NamedTuple."""

    def test_creation_with_timestamp(self) -> None:
        """Test that entry can be created with explicit timestamp."""
        now = time.time()
        entry = HotEntry(fact="Test fact", source="test.txt", timestamp=now)

        assert entry.timestamp == now
        assert entry.fact == "Test fact"
        assert entry.source == "test.txt"

    def test_byte_size(self) -> None:
        """Test byte size calculation."""
        entry = HotEntry(fact="Hello", source="test", timestamp=time.time())

        # "Hello" = 5 bytes, "test" = 4 bytes, + 56 NamedTuple overhead
        assert entry.byte_size == 5 + 4 + 56


class TestHotMemory:
    """Tests for HotMemory class."""

    def test_initialization(self) -> None:
        """Test hot memory initialization."""
        memory = HotMemory(
            max_entries=50,
            max_bytes=50000,
            max_age_hours=12.0,
        )

        assert memory.max_entries == 50
        assert memory.max_bytes == 50000
        assert memory.count == 0
        assert memory.byte_size == 0

    def test_add_single_entry(self) -> None:
        """Test adding a single entry."""
        memory = HotMemory()

        memory.add("Test fact", "source.txt")

        assert memory.count == 1
        assert memory.byte_size > 0

    def test_add_many_entries(self) -> None:
        """Test adding multiple entries."""
        memory = HotMemory()

        facts = [
            ("Fact 1", "source1.txt"),
            ("Fact 2", "source2.txt"),
            ("Fact 3", "source3.txt"),
        ]

        added = memory.add_many(facts)

        assert added == 3
        assert memory.count == 3

    def test_max_entries_eviction(self) -> None:
        """Test that old entries are evicted when max reached."""
        memory = HotMemory(max_entries=3)

        memory.add("Fact 1", "s1")
        memory.add("Fact 2", "s2")
        memory.add("Fact 3", "s3")
        memory.add("Fact 4", "s4")  # Should evict Fact 1

        assert memory.count == 3

        # Verify oldest was evicted
        entries = list(memory.iter_entries())
        facts = [e.fact for e in entries]
        assert "Fact 1" not in facts
        assert "Fact 4" in facts

    def test_max_bytes_eviction(self) -> None:
        """Test that entries are evicted when byte limit reached."""
        # Very small byte limit
        memory = HotMemory(max_entries=100, max_bytes=100)

        memory.add("A" * 20, "source")
        memory.add("B" * 20, "source")
        memory.add("C" * 20, "source")

        # Should have evicted some entries
        assert memory.byte_size <= 100

    def test_get_context_caching(self) -> None:
        """Test that context is cached."""
        memory = HotMemory(context_cache_ttl=10.0)

        memory.add("Test fact", "source")

        # First call builds cache
        ctx1 = memory.get_context()
        assert "Test fact" in ctx1

        # Second call should use cache (cache not dirty)
        ctx2 = memory.get_context()
        assert ctx1 == ctx2

    def test_get_context_cache_invalidation(self) -> None:
        """Test that cache is invalidated on add."""
        memory = HotMemory()

        memory.add("Fact 1", "source")
        ctx1 = memory.get_context()

        memory.add("Fact 2", "source")
        ctx2 = memory.get_context()

        assert "Fact 2" in ctx2
        assert ctx1 != ctx2

    def test_get_context_max_tokens(self) -> None:
        """Test that context respects max tokens."""
        memory = HotMemory()

        # Add many entries
        for i in range(50):
            memory.add(f"This is fact number {i} with some additional content", "source")

        # Get with small token limit
        ctx = memory.get_context(max_tokens=50)

        # Should be truncated
        assert len(ctx) < 1000

    def test_flush(self) -> None:
        """Test flush returns all entries and clears memory."""
        memory = HotMemory()

        memory.add("Fact 1", "s1")
        memory.add("Fact 2", "s2")

        entries = memory.flush()

        assert len(entries) == 2
        assert memory.count == 0
        assert memory.byte_size == 0

    def test_clear(self) -> None:
        """Test clear removes all entries."""
        memory = HotMemory()

        memory.add("Fact 1", "s1")
        memory.add("Fact 2", "s2")
        memory.clear()

        assert memory.count == 0
        assert memory.byte_size == 0

    def test_iter_entries_newest_first(self) -> None:
        """Test that iter_entries returns newest first."""
        memory = HotMemory()

        memory.add("Oldest", "s1")
        time.sleep(0.01)
        memory.add("Middle", "s2")
        time.sleep(0.01)
        memory.add("Newest", "s3")

        entries = list(memory.iter_entries())

        assert entries[0].fact == "Newest"
        assert entries[-1].fact == "Oldest"

    def test_stats(self) -> None:
        """Test stats method."""
        memory = HotMemory(max_entries=10, max_bytes=1000)

        memory.add("Test fact", "source")

        stats = memory.stats()

        assert stats["entries"] == 1
        assert stats["bytes"] > 0
        assert stats["max_entries"] == 10
        assert stats["max_bytes"] == 1000
        assert 0 < stats["utilization_entries"] < 1
        assert 0 < stats["utilization_bytes"] < 1

    def test_thread_safety(self) -> None:
        """Test that operations are thread-safe."""
        import threading

        memory = HotMemory(max_entries=1000)
        errors = []

        def add_entries():
            try:
                for i in range(100):
                    memory.add(f"Fact {threading.current_thread().name} {i}", "source")
            except Exception as e:
                errors.append(e)

        def read_context():
            try:
                for _ in range(50):
                    memory.get_context()
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=add_entries, name=f"add-{i}"))
            threads.append(threading.Thread(target=read_context, name=f"read-{i}"))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert memory.count > 0

    def test_empty_context(self) -> None:
        """Test get_context on empty memory."""
        memory = HotMemory()

        ctx = memory.get_context()

        assert ctx == ""
