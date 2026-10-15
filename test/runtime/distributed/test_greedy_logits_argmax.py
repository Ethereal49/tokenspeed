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

"""Greedy sampling over a TP2 vocab-sharded LM head, argmax-only vs gathered.

An argmax-only greedy batch lets the LM head reduce the argmax across its vocab
shards (8 bytes per row) instead of gathering ``[rows, vocab]`` logits; the
greedy sampler must then emit the same tokens bit for bit, including across an
exact tie between the two shards. Runs on two AMD GPUs, where the reduction is
the Iris exchange joining the heap the TP all-reduce prepared.
"""

from __future__ import annotations

import os
import socket
import sys
import traceback

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=60, suite="runtime-2gpu")

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

WORLD = 2
HIDDEN = 64
VOCAB = 2 * 8192


def _port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _worker(rank: int, port: int, errors) -> None:
    try:
        _run(rank, port)
    except Exception:
        errors[rank] = traceback.format_exc()


def _run(rank: int, port: int) -> None:
    import torch.distributed as dist
    from tokenspeed_kernel.ops.communication.iris import create_iris_state

    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.logits_processor import (
        LogitsMetadata,
        LogitsProcessor,
    )
    from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
    from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
    from tokenspeed.runtime.sampling.backends.greedy import GreedySamplingBackend
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.utils.env import global_server_args_dict

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    mapping = Mapping(rank=rank, world_size=WORLD, attn_tp_size=WORLD)
    pg_manager.init_distributed(
        mapping, distributed_init_method=f"tcp://localhost:{port}", backend="nccl"
    )
    group = mapping.attn.tp_group
    pg_manager.init_process_group(group)
    global_server_args_dict["mapping"] = mapping
    try:
        # The TP all-reduce prepares the process-wide Iris heap the exchange
        # joins.
        create_iris_state(
            group=pg_manager.get_process_group("nccl", group),
            rank_in_group=rank,
            staged_max_numel=4096,
            producer_direct_max_numel=0,
            dtype=torch.bfloat16,
            heap_size=None,
            device=device,
        )
        gen = torch.Generator(device="cuda").manual_seed(7)
        weight = torch.randn(VOCAB, HIDDEN, generator=gen, device="cuda")
        # Two identical rows, one per shard: an exact tie across the ranks,
        # which the full-vocab argmax resolves to the lower column.
        tie_low, tie_high = 5, VOCAB // 2 + 3
        weight[tie_high] = weight[tie_low] = 4 * weight[tie_low].sign()
        weight = weight.bfloat16()
        lm_head = ParallelLMHead(
            VOCAB,
            HIDDEN,
            params_dtype=torch.bfloat16,
            tp_rank=rank,
            tp_size=WORLD,
            tp_group=group,
        ).to(device)
        shard = VOCAB // WORLD
        lm_head.weight.data.copy_(weight[rank * shard : (rank + 1) * shard])
        processor = LogitsProcessor(
            SimpleConfig(),
            tp_rank=rank,
            tp_size=WORLD,
            tp_group=group,
            dp_lm_head_tp=False,
        )
        backend = GreedySamplingBackend(
            SamplingBackendConfig(
                synthetic_acceptance_length=None,
                enable_speculative_sampling=False,
                sampling_stream="batch",
                logprob_order="torch",
                max_bs=64,
                max_draft_tokens_per_req=1,
                device=device,
                tp_group=group,
            )
        )
        sampling_info = SamplingBatchInfo(device=device)
        assert backend.reads_argmax_only(sampling_info)

        for rows in (1, 4, 16, 64):
            hidden = torch.randn(rows, HIDDEN, generator=gen, device="cuda").bfloat16()
            # Rows 0 and 2 point straight at the tied pair.
            hidden[::2] = weight[tie_low].float().sign().bfloat16()
            candidates = torch.zeros((rows, 1), dtype=torch.int64, device=device)
            tokens = {}
            for argmax_only in (False, True):
                metadata = LogitsMetadata(
                    forward_mode=ForwardMode.DECODE,
                    query_shard=None,
                    sampler_reads_argmax_only=argmax_only,
                )
                output = processor(None, hidden, lm_head, metadata)
                if argmax_only:
                    # Reduced across the shards: no full-vocab logits.
                    assert output.next_token_ids is not None
                    assert output.next_token_logits.shape == (rows, shard)
                else:
                    assert output.next_token_ids is None
                    assert output.next_token_logits.shape == (rows, VOCAB)
                predict, _ = backend.verify(
                    output, sampling_info, candidates, tree=None
                )
                tokens[argmax_only] = predict.clone()
            assert torch.equal(tokens[True], tokens[False]), rows
            assert int(tokens[True][0]) == tie_low
        dist.barrier()
    finally:
        dist.destroy_process_group()


class SimpleConfig:
    model_type = "test"
    vocab_size = VOCAB
    final_logit_softcapping = None


def test_argmax_only_greedy_tokens_match_the_gathered_logits():
    from tokenspeed_kernel.platform import current_platform

    if not current_platform().is_amd or torch.cuda.device_count() < WORLD:
        pytest.skip("needs two AMD GPUs")
    pytest.importorskip("tokenspeed_kernel.ops.communication._iris")
    errors = mp.Manager().dict()
    mp.spawn(_worker, args=(_port(), errors), nprocs=WORLD, join=True)
    if errors:
        raise RuntimeError("\n".join(f"rank {r}: {e}" for r, e in errors.items()))
