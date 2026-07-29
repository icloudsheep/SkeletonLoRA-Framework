import threading
import time
import unittest
from unittest.mock import patch

import skeleton_crypto.bridge as bridge


_WorkerMemoryGate = bridge._WorkerMemoryGate


class WorkerMemoryGateTest(unittest.TestCase):
    @staticmethod
    def build_gate(**kwargs):
        with patch.object(bridge, "_current_rss_bytes", return_value=1_000):
            return _WorkerMemoryGate(**kwargs)

    def test_parallel_tasks_release_all_counters(self) -> None:
        gate = self.build_gate(
            max_workers=3,
            memory_limit_bytes=10_000,
            worker_reserve_bytes=100,
            admission_limit_bytes=9_000,
        )
        barrier = threading.Barrier(3)

        def run() -> None:
            with gate.task(50):
                barrier.wait(timeout=1)

        with patch.object(bridge, "_current_rss_bytes", return_value=1_000):
            threads = [threading.Thread(target=run) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(gate.peak_active_workers, 3)
        self.assertEqual(gate.active_workers, 0)
        self.assertEqual(gate.reserved_bytes, 0)

    def test_memory_pressure_reduces_then_restores_worker_limit(self) -> None:
        gate = self.build_gate(
            max_workers=4,
            memory_limit_bytes=10_000,
            worker_reserve_bytes=1_000,
            admission_limit_bytes=8_000,
        )

        with patch.object(bridge, "_current_rss_bytes", return_value=6_500):
            gate.checkpoint()
        self.assertEqual(gate.current_worker_limit, 1)
        self.assertEqual(gate.min_workers, 1)
        self.assertEqual(gate.worker_downgrade_count, 1)

        with patch.object(bridge, "_current_rss_bytes", return_value=1_000):
            gate.checkpoint()
        self.assertEqual(gate.current_worker_limit, 4)
        self.assertEqual(gate.worker_downgrade_count, 1)

    def test_cancel_wakes_waiting_worker(self) -> None:
        gate = self.build_gate(
            max_workers=1,
            memory_limit_bytes=10_000,
            worker_reserve_bytes=100,
            admission_limit_bytes=9_000,
        )
        entered = threading.Event()
        release = threading.Event()
        errors = []

        def holder() -> None:
            with gate.task(0):
                entered.set()
                release.wait(timeout=2)

        def waiter() -> None:
            try:
                with gate.task(0):
                    pass
            except RuntimeError as error:
                errors.append(str(error))

        with patch.object(bridge, "_current_rss_bytes", return_value=1_000):
            first = threading.Thread(target=holder)
            second = threading.Thread(target=waiter)
            first.start()
            self.assertTrue(entered.wait(timeout=1))
            second.start()
            time.sleep(0.05)
            gate.cancel()
            second.join(timeout=1)
            release.set()
            first.join(timeout=1)

        self.assertFalse(second.is_alive())
        self.assertEqual(errors, ["CKKS 并行聚合已取消"])
        self.assertEqual(gate.active_workers, 0)
        self.assertEqual(gate.reserved_bytes, 0)

    def test_rejects_reservation_that_cannot_leave_one_worker(self) -> None:
        gate = self.build_gate(
            max_workers=2,
            memory_limit_bytes=10_000,
            worker_reserve_bytes=1_000,
            admission_limit_bytes=8_000,
        )

        with patch.object(bridge, "_current_rss_bytes", return_value=1_000):
            with self.assertRaisesRegex(MemoryError, "超过并行准入上限"):
                with gate.reserve(7_000):
                    pass

        self.assertEqual(gate.active_workers, 0)
        self.assertEqual(gate.reserved_bytes, 0)


if __name__ == "__main__":
    unittest.main()
