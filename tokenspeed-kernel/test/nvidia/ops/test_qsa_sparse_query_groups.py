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

"""Exact query-prefix reuse must remain valid as captured inputs change."""

import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion, current_platform


def _require_blackwell() -> None:
    if current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)):
        pytest.skip("grouped QSA requires NVIDIA SM100 or SM103")


@pytest.mark.parametrize(
    ("heads", "kv_heads", "rows", "tokens", "expected"),
    [
        (12, 2, 80, 8, 2),
        (12, 2, 192, 6, 2),
        (12, 2, 512, 8, 8),
        (12, 2, 256, 4, 4),
        (12, 2, 576, 6, 6),
        (12, 2, 1024, 8, 8),
        (12, 1, 160, 8, 2),
        (12, 1, 512, 4, 4),
        (6, 1, 256, 8, 2),
        (6, 1, 192, 6, 3),
        (6, 1, 512, 4, 4),
        (12, 2, 1024, 1, 1),
        (12, 2, 1024, 3, 1),
        (3, 2, 512, 4, 1),
    ],
)
def test_qsa_query_groups_stay_inside_requests(
    heads: int, kv_heads: int, rows: int, tokens: int, expected: int
) -> None:
    _require_blackwell()
    from tokenspeed_kernel.thirdparty.cute_dsl.qsa_sparse import _query_group_size

    actual = _query_group_size(rows, kv_heads, heads, tokens, True, 152)
    assert actual == expected
    assert tokens % actual == 0
    assert _query_group_size(rows, kv_heads, heads, tokens, False, 152) == 1


def _reference_rows(q, k, v, selected, row_ids):
    heads_per_kv = q.shape[1] // k.shape[1]
    expected = torch.zeros(
        (len(row_ids), q.shape[1], q.shape[2]), dtype=torch.float64, device=q.device
    )
    for index, row in enumerate(row_ids):
        slots = selected[row]
        slots = slots[slots > 0].long()
        if slots.numel() == 0:
            continue
        query = q[row].double().view(k.shape[1], heads_per_kv, q.shape[2])
        keys = k[slots].double()
        values = v[slots].double()
        scores = torch.einsum("hrd,shd->hrs", query, keys) / 16
        result = torch.einsum("hrs,shd->hrd", scores.softmax(-1), values)
        expected[index] = result.reshape(q.shape[1], q.shape[2])
    return expected


@pytest.mark.parametrize(
    ("q_heads", "kv_heads", "batch", "tokens"),
    [
        (24, 2, 10, 8),
        (24, 2, 32, 6),
        (24, 2, 64, 8),
        (24, 2, 64, 4),
        (24, 2, 96, 6),
        (24, 2, 128, 8),
        (12, 1, 20, 8),
        (12, 1, 128, 4),
        (6, 1, 32, 8),
        (6, 1, 32, 6),
        (6, 1, 128, 4),
    ],
)
def test_qsa_query_groups_refresh_prefix_and_private_tail(
    device: str, q_heads: int, kv_heads: int, batch: int, tokens: int
) -> None:
    _require_blackwell()
    from tokenspeed_kernel.thirdparty.cute_dsl.qsa_sparse import kernel

    torch.manual_seed(107 + q_heads + batch + tokens)
    rows, cache_slots = batch * tokens, 8192
    q = torch.randn(rows, q_heads, 256, device=device, dtype=torch.bfloat16)
    k = torch.randn(cache_slots, kv_heads, 256, device=device, dtype=q.dtype) * 0.25
    v = torch.randn_like(k) * 0.25
    blocks = (torch.randperm(1023, device=device)[:512] + 1) * 4
    prefix = (blocks[:, None] + torch.arange(4, device=device)).reshape(2048)
    selected = torch.empty(rows, 2051, device=device, dtype=torch.int32)
    selected[:, :2048] = prefix
    selected[:, 2048:] = 4096 + torch.arange(rows * 3, device=device).reshape(rows, 3)
    original_slots = selected.clone()

    def forward():
        return kernel(
            q,
            k,
            v,
            selected,
            scale=1 / 16,
            max_seqlen_q=tokens,
            k_scale=None,
            v_scale=None,
            enable_pdl=True,
        )

    forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()
    row_ids = sorted({0, 1, tokens - 1, tokens, rows // 2, rows - 1})
    for phase in ("shared", "late_mismatch", "masked", "distinct", "shared_again"):
        selected.copy_(original_slots)
        if phase == "late_mismatch":
            # Equality of the first tile must never establish full-prefix reuse.
            selected[1::2, 2047] = cache_slots - 1
            q[..., 0] = 16
            k[-1, :, 0] = 64
        elif phase == "masked":
            selected.fill_(-1)
        elif phase == "distinct":
            for row in range(rows):
                selected[row, :2048] = prefix.roll((row % tokens) * 4)
            selected[::3, 2048:] = -1
        elif phase == "shared_again":
            v.mul_(-0.5)
            selected[::2, 2048] = -1
        eager = forward()
        for _ in range(3):
            graph.replay()
        expected = _reference_rows(q, k, v, selected, row_ids)
        torch.testing.assert_close(
            output[row_ids].double(), expected, atol=2e-3, rtol=2e-2
        )
        torch.testing.assert_close(output, eager, atol=0, rtol=0)
        if phase == "masked":
            assert torch.count_nonzero(output).item() == 0
