# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-FileCopyrightText: Copyright (c) 2024, Tri Dao
#
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

"""Triton gated RMSNorm for the linear-attention blocks: ``norm(x) * gate(z)``
or ``norm(x * gate(z))`` in one launch, with ``silu`` or ``sigmoid`` as the
gate. Inference only (no backward)."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

__all__ = ["ROWS_PER_PROGRAM", "triton_gated_rmsnorm"]

# Rows each program normalizes when one warp covers a row.
ROWS_PER_PROGRAM = 4


@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit
def _rms_norm_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    Z,  # pointer to the other branch
    FP8_SCALE,  # static FP8 dequant scale of the consumer
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    SIGMOID_GATE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    WEIGHTS_INDEPENDENT: tl.constexpr,
    OUT_FP8: tl.constexpr,
    SHARED_WEIGHT: tl.constexpr,
):
    # Each program normalizes BLOCK_M rows of one group, one warp per row.
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    group = tl.program_id(1)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    mask = (rows < M)[:, None] & col_mask[None, :]
    # Row offsets in int64: z may be a view with a projection-wide row stride.
    rows = rows.to(tl.int64)
    offsets = group * N + cols[None, :]
    if not SHARED_WEIGHT:
        W += group * N
    if WEIGHTS_INDEPENDENT:
        w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if not WEIGHTS_INDEPENDENT:
        w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    x = tl.load(X + rows[:, None] * stride_x_row + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    if HAS_Z:
        z = tl.load(Z + rows[:, None] * stride_z_row + offsets, mask=mask).to(
            tl.float32
        )
    if HAS_Z and not NORM_BEFORE_GATE:
        x *= tl.sigmoid(z) if SIGMOID_GATE else z * tl.sigmoid(z)
    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N
    rstd = 1 / tl.sqrt(var + eps)
    y = x * rstd[:, None] * w[None, :]
    if HAS_Z and NORM_BEFORE_GATE:
        y *= tl.sigmoid(z) if SIGMOID_GATE else z * tl.sigmoid(z)
    if OUT_FP8:
        # Quantize the activation-dtype output, as the consumer would.
        y = y.to(X.dtype.element_ty).to(tl.float32)
        y = tl.clamp(y * (1.0 / tl.load(FP8_SCALE).to(tl.float32)), -448.0, 448.0)
    tl.store(
        Y + rows[:, None] * stride_y_row + offsets, y.to(Y.dtype.element_ty), mask=mask
    )


@register_kernel(
    "layernorm",
    "gated_rmsnorm",
    name="triton_gated_rmsnorm",
    solution="triton",
    signatures=format_signatures(
        "x", "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    priority=Priority.PORTABLE,
)
def triton_gated_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor | None,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
    sigmoid_gate: bool,
    fp8_scale: torch.Tensor | None,
    weights_independent: bool,
) -> torch.Tensor:
    """Gated RMSNorm over groups of ``group_size`` features; see
    ``tokenspeed_kernel.ops.layernorm.gated_rmsnorm``."""
    enable_pdl = pdl_enabled()
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if z is not None:
        assert z.shape == x_shape_og
        z = z.reshape(-1, z.shape[-1])
        if z.stride(-1) != 1:
            z = z.contiguous()
    weights_independent = weights_independent and weight.is_contiguous()
    weight = weight.contiguous()

    M, N = x.shape
    assert N % group_size == 0
    ngroups = N // group_size
    assert weight.shape in ((N,), (group_size,))
    out = torch.empty_like(
        x, dtype=x.dtype if fp8_scale is None else torch.float8_e4m3fn
    )
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    block_m = ROWS_PER_PROGRAM if num_warps == 1 else 1
    grid = (triton.cdiv(M, block_m), ngroups)
    with torch.cuda.device(x.device.index):
        _rms_norm_fwd_kernel[grid](
            x,
            out,
            weight,
            z,
            fp8_scale,
            x.stride(0),
            out.stride(0),
            z.stride(0) if z is not None else 0,
            M,
            group_size,
            eps,
            BLOCK_M=block_m,
            BLOCK_N=BLOCK_N,
            NORM_BEFORE_GATE=norm_before_gate,
            SIGMOID_GATE=sigmoid_gate,
            num_warps=block_m * num_warps,
            ENABLE_PDL=enable_pdl,
            WEIGHTS_INDEPENDENT=weights_independent,
            OUT_FP8=fp8_scale is not None,
            SHARED_WEIGHT=weight.shape[0] != N,
            **({"launch_pdl": True} if enable_pdl else {}),
        )
    return out.reshape(x_shape_og)
