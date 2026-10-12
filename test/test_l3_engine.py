"""Host-only L3 prefetch and persistence lifecycle tests."""

from __future__ import annotations

import hashlib
import threading
import time
import unittest
from concurrent.futures import Future
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tokenspeed.runtime.cache.hicache import HiCacheEngine
from tokenspeed.runtime.cache.l3 import engine as engine_module
from tokenspeed.runtime.cache.l3.backend import MemoryKvStore
from tokenspeed.runtime.cache.l3.engine import L3CacheEngine
from tokenspeed.runtime.cache.l3.executor import L3HostStore
from tokenspeed.runtime.cache.transfer.host_buffer import HostBufferView
from tokenspeed.runtime.cache.transfer.ops import (
    HostWriteBackCompletion,
    PrefetchOp,
    PrefetchRow,
)


class L3PrefetchTest(unittest.TestCase):
    def _prefetch_executor(self, *, batch_pages, timeout_base_s, rank):
        owners = SimpleNamespace(
            owned_host_positions=lambda rows: list(enumerate(rows))
        )
        engine = L3CacheEngine(
            Mock(rank=rank),
            owners,
            prefix_for_weight_version=lambda version: f"prefix/{version}",
            prefetch_timeout_base_s=timeout_base_s,
            prefetch_timeout_per_page_s=0.0,
            prefetch_batch_pages=batch_pages,
        )
        self.addCleanup(engine.shutdown)
        return engine_module, engine

    @staticmethod
    def _prefetch_op(module, op_id, pages, *, groups):
        """One prefetch op of ``pages`` prefix pages, page-major, one row per group."""
        ops_module = import_module("tokenspeed.runtime.cache.transfer.ops")
        rows = []
        for page in range(pages):
            for group in groups:
                rows.append(
                    ops_module.PrefetchRow(
                        group_id=group,
                        host_page=1 + page * len(groups) + groups.index(group),
                        content_hash=f"h{page}",
                        page_offset=page,
                        page_index=page,
                    )
                )
        return ops_module.PrefetchOp(
            op_id=op_id,
            request_id=f"r{op_id}",
            first_page=0,
            num_pages=pages,
            rows=tuple(rows),
        )

    @staticmethod
    def _settle(executor, op_id):
        for _ in range(200):
            progress = executor.prefetch_progress()
            if progress[op_id][0]:
                return progress[op_id][1]
            time.sleep(0.01)
        raise AssertionError("prefetch lane did not finish")

    def test_prefetch_lane_lands_the_prefix_before_the_first_missing_page(self):
        """The lane fetches in prefix order, a batch at a time, and stops at
        the first page with a missing object (in any group); the landed count
        is a prefix length and the op is acknowledged once, with the count
        the hooks converged, never before they did."""
        module, executor = self._prefetch_executor(
            batch_pages=2, timeout_base_s=10.0, rank=0
        )
        fetched = []

        def prefetch(pages):
            fetched.append([page[2] for page in pages])
            # Page 2's second group is missing: pages 0 and 1 land.
            return [not (page[2] == "h2" and page[0] == 1) for page in pages]

        executor._store.prefetch.side_effect = prefetch
        op = self._prefetch_op(module, 5, 4, groups=(0, 1))
        executor.submit_prefetches([op, SimpleNamespace()])
        landed = self._settle(executor, 5)

        self.assertEqual(landed, 2)
        # Batches of two pages (four rows); the fetch stopped after the batch
        # holding the miss, so pages 3 were never requested.
        self.assertEqual(fetched, [["h0", "h0", "h1", "h1"], ["h2", "h2", "h3", "h3"]])
        self.assertEqual(executor.poll_results(), [])
        executor.complete_prefetch(5, 1)  # the replica's common prefix
        (ack,) = executor.poll_results()
        self.assertEqual(
            (type(ack).__name__, ack.op_id, ack.landed_pages),
            ("PrefetchDoneEvent", 5, 1),
        )
        self.assertEqual(executor.poll_results(), [])
        self.assertEqual(executor.prefetch_progress(), {})
        with self.assertRaises(KeyError):
            executor.complete_prefetch(5, 1)

    def test_prefetch_lane_stops_at_the_deadline_and_on_a_backend_fault(self):
        module, executor = self._prefetch_executor(
            batch_pages=1, timeout_base_s=1.0, rank=0
        )
        # The lane reads the clock once for the deadline and once before each
        # batch: the first batch is on time, the second past the deadline.
        clock = iter([0.0, 0.0] + [10.0] * 100)
        executor._store.prefetch.side_effect = lambda pages: [True] * len(pages)
        with patch.object(module.time, "monotonic", side_effect=lambda: next(clock)):
            executor.submit_prefetches([self._prefetch_op(module, 1, 3, groups=(0,))])
            self.assertEqual(self._settle(executor, 1), 1)
        self.assertEqual(executor._store.prefetch.call_count, 1)

        executor._store.prefetch.side_effect = [[True], RuntimeError("rpc")]
        executor.submit_prefetches([self._prefetch_op(module, 2, 3, groups=(0,))])
        self.assertEqual(self._settle(executor, 2), 1)
        with self.assertRaises(ValueError):
            executor.complete_prefetch(2, 4)  # more than the op has
        executor.complete_prefetch(1, 1)
        executor.complete_prefetch(2, 0)
        self.assertEqual(
            [(ack.op_id, ack.landed_pages) for ack in executor.poll_results()],
            [(1, 1), (2, 0)],
        )

    def test_prefetch_submission_validates_the_plan_before_starting_any_job(self):
        module, executor = self._prefetch_executor(
            batch_pages=2, timeout_base_s=10.0, rank=0
        )
        ops_module = import_module("tokenspeed.runtime.cache.transfer.ops")
        with self.assertRaisesRegex(ValueError, "duplicate prefetch op id"):
            executor.submit_prefetches(
                [
                    self._prefetch_op(module, 1, 1, groups=(0,)),
                    self._prefetch_op(module, 1, 2, groups=(0,)),
                ]
            )
        with self.assertRaisesRegex(ValueError, "carries no pages"):
            executor.submit_prefetches(
                [
                    ops_module.PrefetchOp(
                        op_id=3, request_id="r", first_page=0, num_pages=0, rows=()
                    )
                ]
            )
        # Rows must stay inside [first_page, first_page + num_pages) and in order.
        op = self._prefetch_op(module, 5, 2, groups=(0,))
        bad = ops_module.PrefetchOp(
            op_id=5, request_id="r5", first_page=1, num_pages=2, rows=op.rows
        )
        with self.assertRaisesRegex(ValueError, "outside .* or out of order"):
            executor.submit_prefetches([bad])

    def test_a_prefetch_that_raised_lands_nothing_and_is_logged_once(self):
        module, engine = self._prefetch_executor(
            batch_pages=1, timeout_base_s=10.0, rank=0
        )
        future = Future()
        future.set_exception(RuntimeError("worker failed"))
        engine._prefetch_jobs[9] = module._PrefetchJob(
            request_id="r9",
            num_pages=2,
            future=future,
            deadline=time.monotonic() + 10.0,
        )
        with patch.object(module.logger, "error") as log:
            for _ in range(3):
                self.assertEqual(engine.prefetch_progress(), {9: (True, 0)})
            log.assert_called_once()
            self.assertIn("prefetch 9 (r9) raised", log.call_args.args[0])
        engine._wait_l3_prefetches()


class _GatedStore(MemoryKvStore):
    def __init__(self, *, block_get: bool, block_put: bool) -> None:
        super().__init__()
        self.block_get = block_get
        self.block_put = block_put
        self.get_started = threading.Event()
        self.put_started = threading.Event()
        self.release = threading.Event()
        self.get_calls: list[list[str]] = []
        self.closed = threading.Event()

    def batch_get_into(self, keys, host_buffer, offsets, sizes):
        self.get_calls.append(list(keys))
        if self.block_get:
            self.get_started.set()
            if not self.release.wait(3.0):
                raise TimeoutError("test GET gate was not released")
        return super().batch_get_into(keys, host_buffer, offsets, sizes)

    def batch_put_from(self, keys, host_buffer, offsets, sizes):
        if self.block_put:
            self.put_started.set()
            if not self.release.wait(3.0):
                raise TimeoutError("test PUT gate was not released")
        return super().batch_put_from(keys, host_buffer, offsets, sizes)

    def close(self):
        self.closed.set()
        super().close()


def _host_engine(backend, *, prefix, budget_s):
    host = HostBufferView(
        host_buffer=bytearray([0xEE] * 64),
        num_host_lcm_blocks=4,
        host_lcm_block_bytes=16,
        block_bytes=(16,),
        packing=(1,),
        group_ids=("history",),
    )
    engine = L3CacheEngine(
        L3HostStore(backend, host, key_prefix=prefix, rank=0),
        SimpleNamespace(owned_host_positions=lambda rows: list(enumerate(rows))),
        prefix_for_weight_version=lambda version: f"model-{version}",
        prefetch_timeout_base_s=budget_s,
        prefetch_timeout_per_page_s=0.0,
        prefetch_batch_pages=1,
    )
    return host, engine, HiCacheEngine(engine)


def _read_op(op_id, block, content_hash):
    return PrefetchOp(
        op_id=op_id,
        request_id=f"r{op_id}",
        first_page=0,
        num_pages=1,
        rows=(
            PrefetchRow(
                group_id=0,
                host_page=block,
                content_hash=content_hash,
                page_offset=0,
                page_index=0,
            ),
        ),
    )


def _wait(predicate):
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.001)
    raise AssertionError("Host operation did not settle")


class L3LifecycleTest(unittest.TestCase):
    def test_independent_host_engines_restore_the_same_bytes(self):
        backend = MemoryKvStore()
        source, writer, writer_cache = _host_engine(
            backend, prefix="shared", budget_s=1.0
        )
        destination, reader, reader_cache = _host_engine(
            backend, prefix="shared", budget_s=1.0
        )
        payload = bytes(range(16))
        source.host_buffer[:16] = payload
        try:
            ready = writer_cache.process_results(
                [
                    HostWriteBackCompletion(
                        op_ids=(7,), host_pages=((0, 1, "prefix", 0),)
                    )
                ]
            )
            ready = ready or _wait(lambda: writer_cache.process_results([]))
            self.assertEqual([ack.op_id for ack in ready], [7])
            reader.submit_prefetches([_read_op(9, 2, "prefix")])
            _wait(lambda: reader.prefetch_progress()[9][0])
            self.assertEqual(reader.prefetch_progress()[9], (True, 1))
            self.assertEqual(reader_cache.process_results([]), [])
            reader.complete_prefetch(9, 1)
            self.assertEqual(
                [ack.op_id for ack in reader_cache.process_results([])], [9]
            )
            restored = bytes(destination.host_buffer[16:32])
            self.assertEqual(
                hashlib.sha256(restored).digest(), hashlib.sha256(payload).digest()
            )
        finally:
            reader.shutdown()
            writer.shutdown()

    def test_queue_budget_cancels_only_work_that_never_started(self):
        backend = _GatedStore(block_get=True, block_put=False)
        host, engine, cache = _host_engine(backend, prefix="queued", budget_s=0.03)
        backend._objects[engine._store.object_key("first", 0, 0)] = b"a" * 16
        backend._objects[engine._store.object_key("second", 0, 0)] = b"b" * 16
        try:
            engine.submit_prefetches([_read_op(1, 1, "first")])
            self.assertTrue(backend.get_started.wait(1.0))
            engine.submit_prefetches([_read_op(2, 2, "second")])
            time.sleep(0.04)
            progress = engine.prefetch_progress()
            self.assertEqual(progress, {1: (False, 0), 2: (True, 0)})
            self.assertEqual(len(backend.get_calls), 1)
            self.assertEqual(bytes(host.host_buffer[16:32]), b"\xee" * 16)
            engine.complete_prefetch(2, 0)
            self.assertEqual(
                [(ack.op_id, ack.landed_pages) for ack in cache.process_results([])],
                [(2, 0)],
            )
            # The first call already entered native I/O. No completion and no
            # buffer reuse is authorized by its elapsed deadline.
            self.assertFalse(engine._prefetch_jobs[1].future.cancel())
            self.assertEqual(cache.process_results([]), [])
            backend.release.set()
            _wait(lambda: engine.prefetch_progress()[1][0])
            engine.complete_prefetch(1, 1)
            self.assertEqual(
                [(ack.op_id, ack.landed_pages) for ack in cache.process_results([])],
                [(1, 1)],
            )
            self.assertEqual(bytes(host.host_buffer[:16]), b"a" * 16)
        finally:
            backend.release.set()
            engine.shutdown()

    def test_put_failure_reports_a_flag_without_a_success_ack(self):
        class RejectPut(MemoryKvStore):
            def batch_put_from(self, keys, host_buffer, offsets, sizes):
                return [False] * len(keys)

        host, engine, cache = _host_engine(RejectPut(), prefix="failed", budget_s=1.0)
        host.host_buffer[:16] = b"a" * 16
        try:
            cache.process_results(
                [
                    HostWriteBackCompletion(
                        op_ids=(7,), host_pages=((0, 1, "prefix", 0),)
                    )
                ]
            )

            def failed():
                self.assertEqual(cache.process_results([]), [])
                return cache.consume_backup_poll_failure()

            _wait(failed)
            self.assertFalse(cache.consume_backup_poll_failure())
        finally:
            with self.assertRaisesRegex(RuntimeError, "L3 backup failed"):
                engine.shutdown()

    def test_shutdown_waits_for_put_before_closing_the_store(self):
        backend = _GatedStore(block_get=False, block_put=True)
        host, engine, cache = _host_engine(backend, prefix="shutdown", budget_s=1.0)
        host.host_buffer[:16] = b"s" * 16
        completion = HostWriteBackCompletion(
            op_ids=(8,), host_pages=((0, 1, "tail", 0),)
        )
        errors = []

        def shutdown():
            try:
                cache.shutdown([completion])
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=shutdown)
        thread.start()
        try:
            self.assertTrue(backend.put_started.wait(1.0))
            self.assertFalse(backend.closed.is_set())
            self.assertTrue(thread.is_alive())
        finally:
            backend.release.set()
            thread.join(3.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(backend.closed.is_set())

    def test_namespace_delete_waits_for_get_retirement(self):
        backend = _GatedStore(block_get=True, block_put=False)
        host, engine, _ = _host_engine(backend, prefix="delete", budget_s=1.0)
        key = engine._store.object_key("page", 0, 0)
        backend._objects[key] = b"d" * 16
        deleted = threading.Event()
        result = []
        try:
            engine.submit_prefetches([_read_op(1, 1, "page")])
            self.assertTrue(backend.get_started.wait(1.0))

            def delete():
                result.append(engine.delete_namespace())
                deleted.set()

            thread = threading.Thread(target=delete)
            thread.start()
            self.assertFalse(deleted.wait(0.02))
            self.assertIn(key, backend._objects)
            backend.release.set()
            thread.join(3.0)
            self.assertTrue(deleted.is_set())
            self.assertEqual(result, [True])
            self.assertNotIn(key, backend._objects)
            self.assertEqual(bytes(host.host_buffer[:16]), b"d" * 16)
        finally:
            backend.release.set()
            engine.shutdown()

    def test_l2_only_completions_ack_the_original_operation_ids(self):
        cache = HiCacheEngine(None)
        events = cache.process_results(
            [HostWriteBackCompletion(op_ids=(4, 9), host_pages=((0, 1, "h", 0),))]
        )
        self.assertEqual([event.op_id for event in events], [4, 9])
