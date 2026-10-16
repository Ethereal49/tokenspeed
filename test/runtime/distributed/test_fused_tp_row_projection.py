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

"""Numerical correctness of fused output projection and graph replay."""

import os
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="runtime-2gpu")


@torch.no_grad()
def _worker(rank, world_size, rendezvous):
    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.mapping import DenseLayerMapping
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.layers.linear import (
        DPRowParallelLinear,
        prepare_dp_linear_communication,
        release_dp_linear_communication,
    )
    from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=300),
        device_id=device,
    )
    parallel = DenseLayerMapping(
        rank=rank, world_size=world_size, tp_size=world_size, dp_size=1
    )
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        ignored_layers=[],
        weight_block_size=[128, 128],
        scale_fmt=None,
    )
    models = []
    for _ in range(2):
        with torch.device(device):
            model = DPRowParallelLinear(
                12288,
                7168,
                parallel=parallel,
                params_dtype=torch.bfloat16,
                quant_config=config,
                prefix="o_proj",
            )
        torch.manual_seed(917 + rank)
        model.weight.copy_(
            torch.randn_like(model.weight, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        )
        model.weight_scale_inv.copy_(
            torch.rand_like(model.weight_scale_inv) * 0.005 + 0.005
        )
        prepare_dp_linear_communication(model, 256, torch.bfloat16, AutoBackend())
        models.append(model)
    baseline, candidate = models
    # Retain the real AutoBackend transport and GEMM for the unfused reference.
    if baseline.projection_workspace.fused_row is not None:
        baseline.projection_workspace.fused_row.close()
    baseline.projection_workspace.fused_row = None

    from tokenspeed_kernel.ops.communication.cute_dsl import (
        create_fused_tp_row_projection_state,
        cute_dsl_fused_fp8_dp_row_linear,
    )

    if candidate.projection_workspace.fused_row is None:
        candidate.projection_workspace.fused_row = create_fused_tp_row_projection_state(
            dist.group.WORLD, device, 128, 12288, 7168
        )

    def fused(x, ctx):
        out = torch.empty((x.shape[0], 7168), device=device, dtype=torch.bfloat16)
        return cute_dsl_fused_fp8_dp_row_linear(
            candidate.projection_workspace.fused_row,
            x,
            candidate.weight,
            candidate.weight_scale_inv,
            tuple(ctx.collective_global_num_tokens),
            out,
        )

    def context(counts):
        return ForwardContext(
            attn_backend=None,
            token_to_kv_pool=None,
            bs=counts[rank],
            num_extends=0,
            input_num_tokens=counts[rank],
            forward_mode=ForwardMode.DECODE,
            output_layout=ForwardOutputLayout(0, 0, counts[rank], 1),
            global_num_tokens=list(counts),
            collective_global_num_tokens=list(counts),
        )

    patterns = [
        (n,) * world_size
        for n in (0, 1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 256)
    ]
    patterns += [
        tuple((128, 65, 1, 0)[:world_size]),
        tuple((96, 65, 1, 0)[:world_size]),
        tuple((32, 17, 1, 0)[:world_size]),
        tuple((0, 64, 32, 1)[:world_size]),
    ]
    for counts in patterns:
        x = torch.randn((counts[rank], 12288), device=device, dtype=torch.bfloat16)
        ctx = context(counts)
        expected = baseline(x, ctx=ctx)[0]
        actual = candidate(x, ctx=ctx)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if 0 < max(counts) <= 128:
            torch.testing.assert_close(fused(x, ctx), expected, rtol=0, atol=0)
    # Replay across the 32-bit boundary to exercise persistent 64-bit epochs.
    state = candidate.projection_workspace.fused_row
    for flags in (state.ready, state.ack, state.input_flags):
        flags.fill_(2**32 - 4)
    torch.cuda.synchronize()
    dist.barrier()
    # One graph switches buckets and includes an empty owner; replay new inputs.
    counts_list = [
        (64,) * world_size,
        (64,) * world_size,
        (32,) * world_size,
        (32,) * world_size,
        (65,) * world_size,
        (96,) * world_size,
        tuple((128, 65, 1, 0)[:world_size]),
        tuple((96, 65, 1, 0)[:world_size]),
        tuple((32, 17, 1, 0)[:world_size]),
        tuple((0, 64, 32, 1)[:world_size]),
    ]
    xs = [
        torch.randn((c[rank], 12288), device=device, dtype=torch.bfloat16)
        for c in counts_list
    ]
    ctxs = [context(c) for c in counts_list]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for x, ctx in zip(xs, ctxs):
            candidate(x, ctx=ctx)
            fused(x, ctx)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = [
            (candidate(x, ctx=ctx)[0], fused(x, ctx)) for x, ctx in zip(xs, ctxs)
        ]
    for replay in range(12):
        for x in xs:
            x.copy_(torch.randn_like(x))
        expected = [baseline(x, ctx=ctx)[0] for x, ctx in zip(xs, ctxs)]
        torch.cuda._sleep(100000 * ((rank + replay) % world_size))
        graph.replay()
        for pair, ref in zip(outputs, expected):
            for actual in pair:
                torch.testing.assert_close(actual, ref, rtol=0, atol=0)
    graph.reset()
    for model in models:
        release_dp_linear_communication(model)
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", (4, 2))
def test_fused_row_projection(tmp_path, world_size):
    if torch.cuda.device_count() < world_size or torch.cuda.get_device_capability() != (
        10,
        3,
    ):
        pytest.skip(f"requires {world_size} node-local GB300 GPUs")
    mp.spawn(
        _worker,
        args=(world_size, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=world_size,
        join=True,
    )
