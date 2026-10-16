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

"""Paged attention over a per-tensor scaled FP8 KV cache vs the bf16 kernel.

The cache holds K / k_scale and V / v_scale; the reference attends the same
rows dequantized into a bf16 cache, so the kernels differ only in arithmetic.
"""

from __future__ import annotations

import math

import pytest
import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import NoKernelFoundError

platform = current_platform()

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
    pytest.mark.skipif(
        not platform.is_blackwell, reason="trtllm-gen attention requires SM10x"
    ),
]

PAGE = 64
HEAD_DIM = 128
HEADS_Q = 16
HEADS_KV = 2
FP8 = torch.float8_e4m3fn
SCALES = (0.0614, 0.0513)


def _paged_cache(seq_lens: list[int], seed: int) -> dict:
    """Per-request contiguous pages of scaled FP8 rows and their bf16 dequantization."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = [math.ceil(s / PAGE) for s in seq_lens]
    rows = sum(pages) * PAGE
    shape = (sum(pages), PAGE, HEADS_KV, HEAD_DIM)
    out = {}
    for name, scale, spread in (("k", SCALES[0], 4.0), ("v", SCALES[1], 2.0)):
        x = torch.randn(rows, HEADS_KV, HEAD_DIM, device="cuda", generator=g) * spread
        stored = (x / scale).clamp(-448.0, 448.0).to(FP8)
        out[f"{name}_fp8"] = stored.reshape(shape)
        out[f"{name}_bf16"] = (stored.float() * scale).to(torch.bfloat16).reshape(shape)
    table = torch.zeros(len(seq_lens), max(pages), dtype=torch.int32, device="cuda")
    start = 0
    for b, n in enumerate(pages):
        table[b, :n] = torch.arange(start, start + n, device="cuda")
        start += n
    out["page_table"] = table
    out["seq_lens"] = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    return out


def _check(out: torch.Tensor, ref: torch.Tensor) -> None:
    out, ref = out.float(), ref.float()
    cos = torch.nn.functional.cosine_similarity(
        out.flatten(1), ref.flatten(1), dim=-1
    ).min()
    rel = (out - ref).abs().max() / ref.abs().max()
    assert cos > 0.999, f"cosine {cos:.5f}"
    assert rel < 0.03, f"max rel err {rel:.4f}"


@pytest.mark.parametrize("q_len", [1, 4])
def test_decode_reads_a_scaled_fp8_cache(q_len: int):
    from tokenspeed_kernel.ops.attention.mha import mha_decode_with_kvcache

    seq_lens = [700, 129, 1500]
    cache = _paged_cache(seq_lens, seed=3)
    g = torch.Generator(device="cuda").manual_seed(4)
    q = torch.randn(
        len(seq_lens) * q_len, HEADS_Q, HEAD_DIM, device="cuda", generator=g
    ).to(FP8)
    common = dict(
        page_table=cache["page_table"],
        cache_seqlens=cache["seq_lens"],
        max_seqlen_k=2048,
        max_seqlen_q=q_len,
    )
    ref = mha_decode_with_kvcache(
        q=q.to(torch.bfloat16),
        k_cache=cache["k_bf16"],
        v_cache=cache["v_bf16"],
        **common,
    )
    out = mha_decode_with_kvcache(
        q=q,
        k_cache=cache["k_fp8"],
        v_cache=cache["v_fp8"],
        fp8_kv_scales=SCALES,
        **common,
    )
    _check(out, ref)


def test_extend_reads_a_scaled_fp8_cache():
    from tokenspeed_kernel.ops.attention.mha import mha_extend_with_kvcache

    seq_lens = [300, 77, 1024]
    extend = [40, 77, 9]
    cache = _paged_cache(seq_lens, seed=5)
    g = torch.Generator(device="cuda").manual_seed(6)
    q = torch.randn(sum(extend), HEADS_Q, HEAD_DIM, device="cuda", generator=g).to(FP8)
    cu_q = torch.tensor([0, *torch.tensor(extend).cumsum(0).tolist()], device="cuda")
    cu_kv = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0).tolist()], device="cuda")
    common = dict(
        cu_seqlens_q=cu_q.to(torch.int32),
        cu_seqlens_kv=cu_kv.to(torch.int32),
        page_table=cache["page_table"],
        cache_seqlens=cache["seq_lens"],
        max_seqlen_q=max(extend),
        max_seqlen_k=2048,
        is_causal=True,
    )
    ref = mha_extend_with_kvcache(
        q=q.to(torch.bfloat16),
        k_cache=cache["k_bf16"],
        v_cache=cache["v_bf16"],
        **common,
    )
    out = mha_extend_with_kvcache(
        q=q,
        k_cache=cache["k_fp8"],
        v_cache=cache["v_fp8"],
        fp8_kv_scales=SCALES,
        **common,
    )
    _check(out, ref)


def test_only_kernels_serving_scaled_caches_take_one():
    """FA4's public entry takes no value descale, so it declines a scaled cache."""
    from tokenspeed_kernel.ops.attention.mha import mha_decode_with_kvcache

    cache = _paged_cache([64], seed=7)
    q = torch.zeros(1, HEADS_Q, HEAD_DIM, device="cuda").to(FP8)
    with pytest.raises(NoKernelFoundError):
        mha_decode_with_kvcache(
            q=q,
            k_cache=cache["k_fp8"],
            v_cache=cache["v_fp8"],
            page_table=cache["page_table"],
            cache_seqlens=cache["seq_lens"],
            max_seqlen_k=64,
            max_seqlen_q=1,
            fp8_kv_scales=SCALES,
            solution="fa4",
        )
