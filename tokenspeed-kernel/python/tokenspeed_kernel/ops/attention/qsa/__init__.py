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

from __future__ import annotations

import math

import torch
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import (
    MXFP8_BLOCK_SCALE,
    dense_tensor_format,
    format_signature,
    tensor_format,
)

AttentionResult = torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]


# One UE8M0 scale per 32 consecutive head_dim elements (MXFP8).
MXFP8_ATTENTION_BLOCK_SCALE = MXFP8_BLOCK_SCALE


def _attention_format_signature(**roles: torch.Tensor):
    return format_signature(
        **{role: dense_tensor_format(tensor.dtype) for role, tensor in roles.items()}
    )


def _mxfp8_attention_format_signature(**roles: torch.Tensor):
    return format_signature(
        **{
            role: tensor_format(
                "mxfp8", tensor.dtype, scale=MXFP8_ATTENTION_BLOCK_SCALE
            )
            for role, tensor in roles.items()
        }
    )


def _blockscaled_signature_and_scales(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    q_scale: torch.Tensor | None,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
):
    """Pick dense vs MXFP8 signature and build the scale kwargs splat.

    q_scale selects the block-scaled path; k_scale/v_scale must accompany it.
    Returns (signature, scale_kwargs) for the paged-KV-cache entry points.
    """
    if q_scale is not None:
        assert (
            k_scale is not None and v_scale is not None
        ), "MXFP8 attention requires q_scale, k_scale, and v_scale together"
        signature = _mxfp8_attention_format_signature(
            q=q, k_cache=k_cache, v_cache=v_cache
        )
    else:
        signature = _attention_format_signature(q=q, k_cache=k_cache, v_cache=v_cache)
    return signature, dict(q_scale=q_scale, k_scale=k_scale, v_scale=v_scale)


LSE_LN = math.log2(math.e)


def qwen4_exp_qsa_block_topk(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    *,
    page_size: int,
    block_topk: int,
    queries_per_request: int | None,
    max_partial_bytes: int,
    max_logits_bytes: int,
    path: str,
    persistent_topk_workspace: torch.Tensor | None,
    enable_pdl: bool,
    solution: str | None,
    override: str | None,
) -> torch.Tensor:
    """Select the highest-scoring compressed blocks through kernel selection.

    Args:
        query: Query tensor shaped ``[rows, heads, head_dim]``.
        key_cache: Flattened compressed keys shaped ``[slots, 1, head_dim]``.
        page_table: Physical compressed-cache pages shaped ``[requests, pages]``
            with contiguous columns; holes and padding are zero.
        request_indices: Contiguous owning request id per query row.
        complete_blocks: Contiguous fully compressed block counts shaped
            ``[rows]``, between zero and the page-table capacity.
        page_size: Compressed-cache rows covered by one logical page.
        block_topk: Power-of-two blocks selected per row, at least 64.
        queries_per_request: Uniform consecutive query rows per request, or
            ``None`` for ragged/mixed runs. A positive width must divide rows;
            callers guarantee one owner per run. Each row keeps its frontier.
        max_partial_bytes: Global scratch budget for streaming selection,
            excluding the output. Grouped CuTe streaming needs no scratch.
        max_logits_bytes: Score-matrix budget for ``auto``. Exceeding it
            selects streaming regardless of query size.
        path: ``"logits"`` materializes FP32 scores then radix-selects;
            ``"stream"`` fuses scoring and selection. ``"auto"`` uses the
            query-row count, table capacity and supported layout, independently
            of prefill/decode. Explicit paths bypass the logits budget.
        persistent_topk_workspace: Optional caller-owned CUDA uint8 scratch
            of at least 1 MiB, used only by materialized selection.
        enable_pdl: Explicit programmatic dependent launch ordering.
        solution: Optional registered backend, such as ``"cute_dsl"`` or
            ``"triton"``, filtered within the selected path.
        override: Optional registered kernel name or backend override.

    Returns:
        Int32 logical block ids shaped ``[rows, block_topk]``, padded with
        ``-1``. Output order and boundary tie ids are unspecified.
    """
    from tokenspeed_kernel.ops.attention.qsa.cute_dsl import (
        can_use_cute_dsl_qwen4_exp_qsa_block_topk,
    )

    if query.ndim != 3 or key_cache.ndim != 3 or page_table.ndim != 2:
        raise ValueError("QSA block top-k expects rank-three Q/K and a rank-two table")
    if block_topk < 64 or block_topk & (block_topk - 1):
        raise ValueError("QSA block top-k needs a power-of-two topk >= 64")
    if path not in ("auto", "stream", "logits"):
        raise ValueError(
            f"QSA top-k path must be 'auto', 'stream' or 'logits', got {path!r}"
        )
    rows = query.shape[0]
    if key_cache.shape[1:] != (1, query.shape[2]):
        raise ValueError("QSA keys must have one head with the query head dimension")
    if request_indices.shape != (rows,) or complete_blocks.shape != (rows,):
        raise ValueError("QSA block top-k metadata must have one entry per query row")
    if (
        page_table.stride(1) != 1
        or not request_indices.is_contiguous()
        or not complete_blocks.is_contiguous()
    ):
        raise ValueError(
            "QSA block top-k requires contiguous table columns and row metadata"
        )
    if queries_per_request is not None and (
        queries_per_request < 1 or rows % queries_per_request
    ):
        raise ValueError("QSA queries_per_request must be positive and divide rows")
    num_blocks = page_table.shape[1] * page_size
    if rows == 0 or num_blocks == 0:
        return torch.full(
            (rows, block_topk), -1, dtype=torch.int32, device=query.device
        )

    supported_layout = can_use_cute_dsl_qwen4_exp_qsa_block_topk(
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        page_size=page_size,
        block_topk=block_topk,
        queries_per_request=queries_per_request,
        max_partial_bytes=max_partial_bytes,
    )
    if path == "auto":
        # B300 cold-L2 measurements: long tables favor streaming from 512
        # rows; shorter tables need 8192 rows to amortize its running top-k.
        min_stream_rows = 512 if num_blocks >= 16384 else 8192
        prefer_stream = supported_layout and rows >= min_stream_rows
        fits_logits = rows * num_blocks * 4 <= max_logits_bytes
        path = "stream" if prefer_stream or not fits_logits else "logits"

    kernel = select_kernel(
        "attention",
        "qsa_block_topk",
        _attention_format_signature(query=query, key_cache=key_cache),
        features=frozenset({path}),
        traits={
            "num_q_heads": query.shape[1],
            "head_dim": query.shape[2],
            "page_size": page_size,
            "block_topk": block_topk,
            "supported_layout": supported_layout,
        },
        solution=solution,
        override=override,
    )
    selection_kwargs = (
        {"max_partial_bytes": max_partial_bytes}
        if path == "stream"
        else {"persistent_topk_workspace": persistent_topk_workspace}
    )
    return kernel(
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        page_size=page_size,
        block_topk=block_topk,
        queries_per_request=queries_per_request,
        enable_pdl=enable_pdl,
        **selection_kwargs,
    )


# ===-----------------------------------------------------------------------===#
# QSA Sparse Attention
# ===-----------------------------------------------------------------------===#


def qsa_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    selected_slots: torch.Tensor,
    *,
    scale: float,
    max_seqlen_q: int | None,
    metadata_capacity_rows: int | None,
    k_scale: float | torch.Tensor | None,
    v_scale: float | torch.Tensor | None,
    override: str | None,
    solution: str | None,
) -> torch.Tensor:
    """Attend to a per-query list of physical QSA KV-cache slots.

    Args:
        q: Query tensor shaped ``[tokens, query_heads, head_dim]``.
        k_cache: Flattened key cache shaped
            ``[cache_slots, kv_heads, head_dim]``.
        v_cache: Flattened value cache shaped
            ``[cache_slots, kv_heads, value_head_dim]``.
        selected_slots: Physical cache slots shaped ``[tokens, budget]``;
            non-positive values are ignored.
        scale: Softmax scale applied to query-key scores.
        max_seqlen_q: Uniform query tokens per request for decode: 1 for
            normal decode and ``spec_num_tokens`` for compact speculative
            decode. Pass None for prefill or mixed/ragged queries, including
            a prefill containing only one token.
        metadata_capacity_rows: Row capacity reserved by stateful fallback
            implementations. Pass ``None`` to use the actual query-row count;
            workspace-free kernels ignore it.
        k_scale: Optional scalar FP8 key descale.
        v_scale: Optional scalar FP8 value descale.
        override: Optional registered kernel name or solution override.
        solution: Optional kernel solution selected through normal capability
            and shape filtering.

    Returns:
        Attention output shaped
        ``[tokens, query_heads, value_head_dim]`` with the query dtype.

    The SM100/SM103 CuTe DSL implementation is preferred when its specialization
    matches a uniform decode. Prefill and mixed/ragged queries use FlashInfer
    FA2 on supported NVIDIA architectures; other decode geometries retain
    the registered fallback. Kernel selection preserves this distinction
    before adapting ragged inputs to independent one-token query rows.
    """

    if q.ndim != 3 or k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("QSA sparse attention expects rank-three Q/K/V tensors")
    if selected_slots.ndim != 2 or selected_slots.shape[0] != q.shape[0]:
        raise ValueError("QSA selected slots must have one row per query token")
    query_width = 1 if max_seqlen_q is None else max_seqlen_q
    if query_width < 1:
        raise ValueError("QSA max_seqlen_q must be positive")
    if q.shape[0] % query_width:
        raise ValueError("QSA query rows must be divisible by max_seqlen_q")
    if q.shape[0] == 0:
        return q.new_empty((0, q.shape[1], v_cache.shape[-1]))
    traits = {
        "batch_size": q.shape[0] // query_width,
        "q_len": query_width,
        "num_q_heads": q.shape[1],
        "num_kv_heads": k_cache.shape[1],
        "head_dim": q.shape[-1],
        "value_head_dim": v_cache.shape[-1],
        "selected_width": selected_slots.shape[1],
        "is_decode": max_seqlen_q is not None,
    }
    signature = _attention_format_signature(q=q, k_cache=k_cache, v_cache=v_cache)
    kernel = select_kernel(
        "attention",
        "qsa_sparse_attention",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    return kernel(
        q,
        k_cache,
        v_cache,
        selected_slots,
        scale=scale,
        max_seqlen_q=query_width,
        metadata_capacity_rows=metadata_capacity_rows,
        k_scale=k_scale,
        v_scale=v_scale,
    )


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.attention.qsa.triton  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.qsa.cute_dsl  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.qsa.flashinfer  # noqa: E402,F401

# isort: on


__all__ = [
    "qwen4_exp_qsa_block_topk",
    "qsa_sparse_attention",
]
