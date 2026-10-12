# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Repeat a CPU Host-cache lifecycle experiment; no GPU or native Store claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from tokenspeed.runtime.cache.hicache import HiCacheEngine
from tokenspeed.runtime.cache.l3.backend import MemoryKvStore
from tokenspeed.runtime.cache.l3.engine import L3CacheEngine
from tokenspeed.runtime.cache.l3.executor import L3HostStore
from tokenspeed.runtime.cache.transfer.host_buffer import HostBufferView
from tokenspeed.runtime.cache.transfer.ops import (
    HostWriteBackCompletion,
    PrefetchOp,
    PrefetchRow,
)


class GatedStore(MemoryKvStore):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.get_calls = 0

    def batch_get_into(self, keys, host_buffer, offsets, sizes):
        self.get_calls += 1
        self.started.set()
        if not self.release.wait(5.0):
            raise TimeoutError("experiment did not release the read gate")
        return super().batch_get_into(keys, host_buffer, offsets, sizes)


def engine(backend, budget_s):
    view = HostBufferView(bytearray([0xEE] * 64), 4, 16, (16,), (1,), ("history",))
    l3 = L3CacheEngine(
        L3HostStore(backend, view, key_prefix="host-experiment", rank=0),
        SimpleNamespace(owned_host_positions=lambda rows: list(enumerate(rows))),
        prefix_for_weight_version=lambda v: f"host-experiment-{v}",
        prefetch_timeout_base_s=budget_s,
        prefetch_timeout_per_page_s=0.0,
        prefetch_batch_pages=1,
    )
    return view, l3, HiCacheEngine(l3)


def wait_until(predicate):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.001)
    raise AssertionError("Host-cache operation did not settle")


def prefetch(op_id, page, key):
    return PrefetchOp(
        op_id, f"request-{op_id}", 0, 1, (PrefetchRow(0, page, key, 0, 0),)
    )


def run_once(budget_s, hold_s):
    backend = GatedStore()
    source, writer, writer_cache = engine(backend, budget_s)
    target, reader, reader_cache = engine(backend, budget_s)
    payload = bytes(range(16))
    source.host_buffer[:16] = payload
    source.host_buffer[16:32] = b"b" * 16
    try:
        completion = HostWriteBackCompletion((7,), ((0, 1, "a", 0), (0, 2, "b", 0)))
        ready = writer_cache.process_results([completion])
        ready = ready or wait_until(lambda: writer_cache.process_results([]))
        assert [event.op_id for event in ready] == [7]
        start = time.monotonic()
        reader.submit_prefetches([prefetch(1, 1, "a")])
        assert backend.started.wait(1.0)
        reader.submit_prefetches([prefetch(2, 2, "b")])
        time.sleep(hold_s)
        state = reader.prefetch_progress()
        assert state == {1: (False, 0), 2: (True, 0)}, state
        assert backend.get_calls == 1
        assert bytes(target.host_buffer[16:32]) == b"\xee" * 16
        reader.complete_prefetch(2, 0)
        cancelled = reader_cache.process_results([])
        assert [(e.op_id, e.landed_pages) for e in cancelled] == [(2, 0)]
        expired_at = time.monotonic() - start
        backend.release.set()
        wait_until(lambda: reader.prefetch_progress()[1][0])
        reader.complete_prefetch(1, 1)
        landed = reader_cache.process_results([])
        assert [(e.op_id, e.landed_pages) for e in landed] == [(1, 1)]
        digest = hashlib.sha256(bytes(target.host_buffer[:16])).hexdigest()
        assert digest == hashlib.sha256(payload).hexdigest()
        return {
            "queued_miss_ack_seconds": expired_at,
            "active_get_retirement_seconds": time.monotonic() - start,
            "get_calls": backend.get_calls,
            "restored_sha256": digest,
            "queued_destination_untouched": True,
            "active_get_acked_only_after_retirement": True,
        }
    finally:
        backend.release.set()
        reader.shutdown()
        writer.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--budget-s", required=True, type=float)
    parser.add_argument("--hold-s", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.iterations <= 0 or args.budget_s <= 0 or args.hold_s <= args.budget_s:
        parser.error("positive iterations/budget and hold > budget are required")
    observations = [
        run_once(args.budget_s, args.hold_s) for _ in range(args.iterations)
    ]
    result = {
        "scope": "two Host-only engines, MemoryKvStore, blocked-call injection",
        "gpu": False,
        "native_mooncake": False,
        "model_inference": False,
        "python": platform.python_version(),
        "iterations": args.iterations,
        "budget_seconds": args.budget_s,
        "hold_seconds": args.hold_s,
        "observations": observations,
        "all_checks_passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "observations"}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
