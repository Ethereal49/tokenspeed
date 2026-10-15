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

"""The Iris distributed argmax equals the local argmax of the gathered rows."""

import socket
import time
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import assert_no_triton_compile, is_amd

pytestmark = pytest.mark.skipif(not is_amd(), reason="Iris requires AMD ROCm")

WORLD_SIZE = 2


@pytest.fixture(autouse=True)
def _require_iris():
    pytest.importorskip("tokenspeed_kernel.ops.communication._iris")


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _full_logits(rows: int, vocab: int, dtype: torch.dtype, seed: int):
    """Rows every rank builds identically, with the hard cases planted."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(rows, vocab, generator=g, device="cuda").to(dtype)
    shard = vocab // WORLD_SIZE
    special = [
        # The same maximum on both ranks: the lower global column wins.
        lambda row: (row.__setitem__(17, 9.0), row.__setitem__(shard + 3, 9.0)),
        # The maximum twice within one shard.
        lambda row: (row.__setitem__(shard + 40, 9.0), row.__setitem__(shard + 7, 9.0)),
        # -0.0 and +0.0 tie as values compare: the lower column wins.
        lambda row: (
            row.fill_(-1.0),
            row.__setitem__(shard + 1, 0.0),
            row.__setitem__(5, -0.0),
        ),
        # NaN never wins under the AMD argmax's NaN rule.
        lambda row: (row.__setitem__(2, float("nan")), row.__setitem__(shard, 9.0)),
        # A shard without a valid value defers to the other one.
        lambda row: (row[:shard].fill_(float("nan")), row.__setitem__(shard + 9, 9.0)),
        # No valid value anywhere: -1, as the local argmax.
        lambda row: row.fill_(float("nan")),
        # All -inf: the lowest column.
        lambda row: row.fill_(float("-inf")),
        # Only the last rank's last column holds the maximum.
        lambda row: row.__setitem__(vocab - 1, 9.0),
    ]
    for row, plant in zip(logits, special):
        plant(row)
    return logits


def _worker(rank, port, vocab, errors):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=WORLD_SIZE,
        )
        from tokenspeed_kernel.ops.communication._iris.argmax import (
            iris_argmax_exchange,
        )
        from tokenspeed_kernel.ops.communication.iris import create_iris_state
        from tokenspeed_kernel.ops.sampling import (
            argmax,
            dist_argmax_available,
            distributed_argmax,
            supports_dist_argmax_shape,
            try_create_dist_argmax_state,
        )

        device = torch.device("cuda", rank)
        shard = vocab // WORLD_SIZE
        # Optional users join the heap a group's all-reduce prepared.
        assert not dist_argmax_available()
        create_iris_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            staged_max_numel=4096,
            producer_direct_max_numel=0,
            dtype=torch.bfloat16,
            heap_size=None,
            device=device,
        )
        assert dist_argmax_available()
        assert supports_dist_argmax_shape(shard, torch.bfloat16, WORLD_SIZE)
        states = {
            dtype: try_create_dist_argmax_state(
                group=dist.group.WORLD,
                rank_in_group=rank,
                max_M=512,
                dtype=dtype,
                device=device,
                skip_ping_pong=True,
            )
            for dtype in (torch.bfloat16, torch.float32)
        }

        def check(rows, dtype, seed, preset=None):
            state = states[dtype]
            full = _full_logits(rows, vocab, dtype, seed)
            local = full[:, rank * shard : (rank + 1) * shard].contiguous()
            if preset is not None:
                torch.cuda.synchronize()
                dist.barrier()
                state.exchange.ready_flags.fill_(preset)
                torch.cuda.synchronize()
                dist.barrier()
            # A late peer must not let this rank read a stale key.
            if rank == WORLD_SIZE - 1 and seed % 3 == 0:
                time.sleep(0.05)
            got = distributed_argmax(state, local)
            want = argmax(full)
            torch.cuda.synchronize()
            assert torch.equal(got, want), (rows, dtype, seed, preset, got, want)

        seed = 0
        for dtype in (torch.bfloat16, torch.float32):
            for rows in (1, 4, 8, 16, 64, 300):
                for _ in range(3):
                    check(rows, dtype, seed)
                    seed += 1
        # Epochs wrap at 2**31 without losing the slot rotation or the wait.
        for preset in (2**31 - 2, -2):
            for _ in range(4):
                check(16, torch.bfloat16, seed, preset if _ == 0 else None)
                seed += 1

        # A forward without sampled rows exchanges nothing, on every rank.
        empty = torch.empty((0, shard), dtype=torch.bfloat16, device=device)
        assert distributed_argmax(states[torch.bfloat16], empty).shape == (0,)

        # Graph replay reads fresh logits from the captured input address.
        state = states[torch.bfloat16]
        local = torch.empty((16, shard), dtype=torch.bfloat16, device=device)
        full = _full_logits(16, vocab, torch.bfloat16, seed)
        local.copy_(full[:, rank * shard : (rank + 1) * shard])
        distributed_argmax(state, local)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = distributed_argmax(state, local)
        for replay in range(4):
            full = _full_logits(16, vocab, torch.bfloat16, 1000 + replay)
            local.copy_(full[:, rank * shard : (rank + 1) * shard])
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(captured, argmax(full)), replay

        # The row count is a runtime argument: one binary for every batch.
        with assert_no_triton_compile(iris_argmax_exchange):
            for rows in (2, 3, 16, 33, 64, 257):
                check(rows, torch.bfloat16, seed)
                seed += 1
        dist.barrier()
    except Exception:
        errors[rank] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("vocab", (2 * 8192, 248320))
def test_iris_distributed_argmax_matches_full_vocab_argmax(vocab):
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"needs {WORLD_SIZE} GPUs")
    errors = mp.Manager().dict()
    mp.spawn(_worker, args=(_port(), vocab, errors), nprocs=WORLD_SIZE, join=True)
    if errors:
        raise RuntimeError("\n".join(f"rank {r}: {e}" for r, e in errors.items()))
