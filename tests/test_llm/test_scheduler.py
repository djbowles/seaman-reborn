"""Tests for VRAM-aware model scheduler."""

from __future__ import annotations

import threading

from seaman_brain.llm.scheduler import ModelScheduler

# --- Happy path tests ---


class TestModelSchedulerHappyPath:
    """Test basic acquire/release behavior."""

    def test_acquire_chat_succeeds(self):
        """Acquiring chat on empty scheduler succeeds."""
        sched = ModelScheduler()
        assert sched.acquire("chat") is True

    def test_acquire_vision_succeeds(self):
        """Acquiring vision on empty scheduler succeeds."""
        sched = ModelScheduler()
        assert sched.acquire("vision") is True

    def test_release_allows_reacquire(self):
        """After release, the same slot can be acquired again."""
        sched = ModelScheduler()
        sched.acquire("chat")
        sched.release("chat")
        assert sched.acquire("chat") is True

    def test_is_active_reflects_state(self):
        """is_active returns True for held slots, False otherwise."""
        sched = ModelScheduler()
        assert sched.is_active("chat") is False
        sched.acquire("chat")
        assert sched.is_active("chat") is True
        sched.release("chat")
        assert sched.is_active("chat") is False


# --- Mutual exclusion tests ---


class TestModelSchedulerExclusion:
    """Test that heavy slots block each other."""

    def test_chat_blocks_vision(self):
        """Vision acquire is denied while chat is active."""
        sched = ModelScheduler()
        sched.acquire("chat")
        assert sched.acquire("vision") is False

    def test_vision_blocks_chat(self):
        """Chat acquire is denied while vision is active."""
        sched = ModelScheduler()
        sched.acquire("vision")
        assert sched.acquire("chat") is False

    def test_double_acquire_denied(self):
        """Same slot cannot be acquired twice."""
        sched = ModelScheduler()
        sched.acquire("chat")
        assert sched.acquire("chat") is False

    def test_release_unblocks_other(self):
        """Releasing chat allows vision to acquire."""
        sched = ModelScheduler()
        sched.acquire("chat")
        sched.release("chat")
        assert sched.acquire("vision") is True

    def test_release_nonexistent_is_safe(self):
        """Releasing a slot that was never acquired is a no-op."""
        sched = ModelScheduler()
        sched.release("chat")  # Should not raise


# --- Disabled mode tests ---


class TestModelSchedulerDisabled:
    """Test that enabled=False bypasses all exclusion."""

    def test_acquire_always_true_when_disabled(self):
        """All acquire calls return True when scheduler is disabled."""
        sched = ModelScheduler(enabled=False)
        assert sched.acquire("chat") is True
        assert sched.acquire("vision") is True
        assert sched.acquire("chat") is True  # double-acquire allowed

    def test_disabled_allows_concurrent_heavy_slots(self):
        """Chat and vision can both be acquired simultaneously when disabled."""
        sched = ModelScheduler(enabled=False)
        sched.acquire("chat")
        assert sched.acquire("vision") is True

    def test_release_is_noop_when_disabled(self):
        """Release does not raise when scheduler is disabled."""
        sched = ModelScheduler(enabled=False)
        sched.acquire("chat")
        sched.release("chat")  # Should not raise

    def test_enabled_true_is_default(self):
        """Default constructor enables scheduling."""
        sched = ModelScheduler()
        sched.acquire("chat")
        assert sched.acquire("vision") is False  # blocked


# --- Thread safety tests ---


class TestModelSchedulerThreadSafety:
    """Verify thread-safe access to the scheduler."""

    def test_concurrent_acquire_only_one_wins(self):
        """When two threads race to acquire, exactly one succeeds."""
        sched = ModelScheduler()
        results: list[bool] = []
        barrier = threading.Barrier(2)

        def try_acquire(slot: str):
            barrier.wait()
            results.append(sched.acquire(slot))

        t1 = threading.Thread(target=try_acquire, args=("chat",))
        t2 = threading.Thread(target=try_acquire, args=("vision",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results.count(True) == 1
        assert results.count(False) == 1

    def test_concurrent_same_slot(self):
        """Two threads acquiring the same slot — exactly one wins."""
        sched = ModelScheduler()
        results: list[bool] = []
        barrier = threading.Barrier(2)

        def try_acquire():
            barrier.wait()
            results.append(sched.acquire("chat"))

        t1 = threading.Thread(target=try_acquire)
        t2 = threading.Thread(target=try_acquire)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results.count(True) == 1
        assert results.count(False) == 1
