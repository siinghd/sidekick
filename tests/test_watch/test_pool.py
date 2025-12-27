"""Tests for model pool module."""

import asyncio
import time
from unittest.mock import MagicMock

from sidekick.watch.pool import ModelPool, MultiModelPool


class TestModelPool:
    """Tests for ModelPool class."""

    def test_lazy_loading(self) -> None:
        """Test that model is loaded lazily."""
        load_count = 0

        def loader():
            nonlocal load_count
            load_count += 1
            return "model"

        pool = ModelPool(loader=loader)

        assert load_count == 0
        assert pool.is_loaded is False

        # First access triggers load
        model = pool.get_sync()

        assert load_count == 1
        assert model == "model"
        assert pool.is_loaded is True

    def test_model_reuse(self) -> None:
        """Test that loaded model is reused."""
        load_count = 0

        def loader():
            nonlocal load_count
            load_count += 1
            return f"model_{load_count}"

        pool = ModelPool(loader=loader, ttl_seconds=300)

        model1 = pool.get_sync()
        model2 = pool.get_sync()

        assert load_count == 1
        assert model1 == model2

    def test_ttl_unload(self) -> None:
        """Test that model is unloaded after TTL."""
        load_count = 0

        def loader():
            nonlocal load_count
            load_count += 1
            return f"model_{load_count}"

        pool = ModelPool(loader=loader, ttl_seconds=0.1)

        model1 = pool.get_sync()
        assert load_count == 1

        # Wait for TTL
        time.sleep(0.2)

        model2 = pool.get_sync()
        assert load_count == 2
        assert model1 != model2

    def test_explicit_unload(self) -> None:
        """Test explicit model unloading."""
        pool = ModelPool(loader=lambda: "model")

        pool.get_sync()
        assert pool.is_loaded is True

        pool.unload()
        assert pool.is_loaded is False

    def test_unloader_called(self) -> None:
        """Test that unloader is called on unload."""
        unload_count = 0

        def unloader(model):
            nonlocal unload_count
            unload_count += 1

        pool = ModelPool(
            loader=lambda: "model",
            unloader=unloader,
        )

        pool.get_sync()
        pool.unload()

        assert unload_count == 1

    def test_async_get(self) -> None:
        """Test async model access."""

        async def run_test():
            pool = ModelPool(loader=lambda: "model")

            model = await pool.get()

            assert model == "model"
            assert pool.is_loaded is True

        asyncio.run(run_test())

    def test_idle_seconds(self) -> None:
        """Test idle time tracking."""
        pool = ModelPool(loader=lambda: "model")

        assert pool.idle_seconds == 0  # Not loaded

        pool.get_sync()
        time.sleep(0.1)

        assert pool.idle_seconds >= 0.1

    def test_stats(self) -> None:
        """Test pool statistics."""
        pool = ModelPool(loader=lambda: "model", ttl_seconds=60)

        stats = pool.stats()
        assert stats["loaded"] is False

        pool.get_sync()
        stats = pool.stats()

        assert stats["loaded"] is True
        assert stats["ttl_seconds"] == 60


class TestMultiModelPool:
    """Tests for MultiModelPool class."""

    def test_register_and_get(self) -> None:
        """Test registering and getting models."""

        async def run_test():
            pool = MultiModelPool()

            pool.register("model_a", lambda: "A")
            pool.register("model_b", lambda: "B")

            a = await pool.get("model_a")
            b = await pool.get("model_b")

            assert a == "A"
            assert b == "B"

        asyncio.run(run_test())

    def test_unknown_model_raises(self) -> None:
        """Test that getting unknown model raises KeyError."""

        async def run_test():
            pool = MultiModelPool()

            try:
                await pool.get("unknown")
                assert False, "Should have raised KeyError"
            except KeyError:
                pass

        asyncio.run(run_test())

    def test_max_models_eviction(self) -> None:
        """Test that old models are evicted when limit reached."""

        async def run_test():
            load_counts = {"a": 0, "b": 0, "c": 0}

            def make_loader(name):
                def loader():
                    load_counts[name] += 1
                    return f"model_{name}"
                return loader

            pool = MultiModelPool(max_models=2)

            pool.register("a", make_loader("a"))
            pool.register("b", make_loader("b"))
            pool.register("c", make_loader("c"))

            # Load a and b
            await pool.get("a")
            await pool.get("b")

            # Load c - should evict a (oldest)
            await asyncio.sleep(0.01)  # Ensure time difference
            await pool.get("c")

            # Get a again - should reload
            await pool.get("a")

            assert load_counts["a"] == 2  # Loaded twice

        asyncio.run(run_test())

    def test_unload_all(self) -> None:
        """Test unloading all models."""

        async def run_test():
            pool = MultiModelPool()

            pool.register("a", lambda: "A")
            pool.register("b", lambda: "B")

            await pool.get("a")
            await pool.get("b")

            pool.unload_all()

            stats = pool.stats()
            assert all(not s["loaded"] for s in stats.values())

        asyncio.run(run_test())

    def test_stats(self) -> None:
        """Test multi-pool statistics."""

        async def run_test():
            pool = MultiModelPool()

            pool.register("a", lambda: "A")
            pool.register("b", lambda: "B")

            await pool.get("a")

            stats = pool.stats()

            assert "a" in stats
            assert "b" in stats
            assert stats["a"]["loaded"] is True
            assert stats["b"]["loaded"] is False

        asyncio.run(run_test())
