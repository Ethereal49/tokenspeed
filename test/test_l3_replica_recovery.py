"""Real two-process gloo agreement over Host-only L3 outcomes."""

from __future__ import annotations

import json
import multiprocessing
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist

from tokenspeed.runtime.cache.hicache import HiCacheEngine
from tokenspeed.runtime.cache.l3.backend import MemoryKvStore
from tokenspeed.runtime.cache.l3.engine import L3CacheEngine
from tokenspeed.runtime.cache.l3.executor import L3HostStore
from tokenspeed.runtime.cache.transfer.host_buffer import HostBufferView
from tokenspeed.runtime.cache.transfer.ops import PrefetchOp, PrefetchRow
from tokenspeed.runtime.engine.l3_cache_hooks import L3CacheHooks


class SchedulerProbe:
    def __init__(self):
        self.registered = []
        self.submitted = []

    def prefix_hashes_for_tokens(self, tokens):
        return ["h0", "h1", "h2"]

    def expand_prefix_keys(self, hashes):
        return [0] * len(hashes), list(hashes), [0] * len(hashes)

    def register_storage_keys(self, groups, hashes, offsets):
        self.registered.append((groups, hashes, offsets))

    def submit_requests(self, specs):
        self.submitted.extend(specs)


def worker(rank, rendezvous, output):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    backend = MemoryKvStore()
    host = HostBufferView(bytearray([0xEE] * 64), 4, 16, (16,), (1,), ("history",))
    store = L3HostStore(backend, host, key_prefix="replica-test", rank=rank)
    engine = L3CacheEngine(
        store,
        SimpleNamespace(owned_host_positions=lambda rows: list(enumerate(rows))),
        prefix_for_weight_version=lambda version: version,
        prefetch_timeout_base_s=5.0,
        prefetch_timeout_per_page_s=0.0,
        prefetch_batch_pages=1,
    )
    scheduler = SchedulerProbe()
    hooks = L3CacheHooks(
        scheduler,
        engine,
        attn_tp_size=2,
        attn_tp_cpu_group=dist.group.WORLD,
        pp_size=1,
        pp_cpu_group=None,
    )
    cache = HiCacheEngine(engine)
    try:
        # A fault on only one rank must still enter the real MIN collective.
        exists = engine.exists
        if rank == 1:

            def failed_probe(pages):
                raise RuntimeError("rank-local lookup fault")

            engine.exists = failed_probe
        hooks.submit_requests([SimpleNamespace(tokens=[1, 2, 3])])
        assert scheduler.registered == []
        engine.exists = exists
        for index in range(3 if rank == 0 else 1):
            backend._objects[store.object_key(f"h{index}", 0, 0)] = (
                bytes([index + 1]) * 16
            )
        op = PrefetchOp(
            7,
            "request",
            0,
            3,
            tuple(PrefetchRow(0, i + 1, f"h{i}", 0, i) for i in range(3)),
        )
        engine.submit_prefetches([op])
        limit = time.monotonic() + 5.0
        while not engine.prefetch_progress()[7][0] and time.monotonic() < limit:
            time.sleep(0.001)
        assert engine.prefetch_progress()[7][0]
        dist.barrier()
        hooks.converge_prefetches()
        events = cache.process_results([])
        assert [(event.op_id, event.landed_pages) for event in events] == [(7, 1)]
        Path(output).write_text(
            json.dumps(
                {"rank": rank, "common_prefix_pages": 1, "lookup_fault_converged": True}
            )
        )
    finally:
        engine.shutdown()
        dist.destroy_process_group()


def test_rank_local_failure_and_partial_landing_converge_without_hanging(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo is unavailable")
    context = multiprocessing.get_context("spawn")
    outputs = [tmp_path / f"rank-{rank}.json" for rank in range(2)]
    processes = [
        context.Process(
            target=worker, args=(rank, str(tmp_path / "gloo"), str(outputs[rank]))
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(25.0)
        assert all(
            not process.is_alive() and process.exitcode == 0 for process in processes
        )
        assert [
            json.loads(path.read_text())["common_prefix_pages"] for path in outputs
        ] == [1, 1]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5.0)
