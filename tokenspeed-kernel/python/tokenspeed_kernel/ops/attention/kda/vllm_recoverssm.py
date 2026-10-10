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

"""TokenSpeed adapters for the vendored vLLM KDA RecoverSSM kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import gl, gluon, tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.vllm_recoverssm.kda import (
    NULL_BLOCK_ID,
    _commit_kda_state_kernel,
    _prepare_commit_plan_kernel,
    kda_recoverssm_verify,
)

_DENSE_BF16_SIGNATURES = format_signatures(("q", "k", "v"), "dense", {torch.bfloat16})


@register_kernel(
    "attention",
    "kda_recoverssm_verify",
    name="vllm_triton_kda_recoverssm_verify",
    solution="vllm_triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_BF16_SIGNATURES,
    priority=Priority.REFERENCE,
    traits={
        "paged_state": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
    },
)
def vllm_triton_kda_recoverssm_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    checkpoint_state: torch.Tensor,
    correction_cache: torch.Tensor,
    kd_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    lower_bound: float | None,
    out: torch.Tensor,
    replay_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    verify_decay: torch.Tensor | None,
    verify_qk: tuple[torch.Tensor, torch.Tensor] | None,
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
) -> torch.Tensor:
    """Verify candidates and save current-window recovery records.

    q/k/v and raw_g pack tokens as [1, tokens, heads, dim]; raw_beta omits
    the final dim. A_log and dt_bias parameterize the gate. lower_bound
    selects the bounded gate, or None selects the softplus gate.

    checkpoint_state uses [page, head, value, key]. state_indices selects
    request pages, with zero reserved for padding. query_start_loc gives
    packed-token offsets; spec_query_len bounds each request's window.
    correction_cache and kd_cache are request/head/token/channel FP32
    destinations. The latter stores normalized K followed by decay.

    replay_inputs provides independent FP32 normalized K, pre-SiLU convolution V and
    multiplicative decay. None records the verify chain instead.
    verify_qk optionally supplies FP32 normalized/scaled Q and normalized K.
    verify_decay optionally supplies FP32 decay after BF16 gate rounding.
    These prepared inputs remove repeated transforms from value tiles.

    record_metadata is INT64 [batch, 3] (slot, generation, absolute start).
    record_stamps is INT64 [slots, capacity, 2] (position, generation).
    Records use position % capacity; verify stamps each written candidate.
    Both may be None only for standalone unstamped kernel diagnostics.

    Return out after writing verify results. The checkpoint is unchanged.
    """
    return kda_recoverssm_verify(
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        lower_bound,
        checkpoint_state,
        correction_cache,
        kd_cache,
        query_start_loc,
        state_indices,
        spec_query_len,
        out=out,
        replay_inputs=replay_inputs,
        verify_decay=verify_decay,
        verify_qk=verify_qk,
        record_metadata=record_metadata,
        record_stamps=record_stamps,
    )


@register_kernel(
    "attention",
    "kda_recoverssm_commit",
    name="vllm_triton_kda_recoverssm_commit",
    solution="vllm_triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_BF16_SIGNATURES,
    priority=Priority.REFERENCE,
    traits={
        "attention_only": frozenset({True}),
        "paged_state": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
    },
)
def vllm_triton_kda_recoverssm_commit(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    checkpoint_state: torch.Tensor,
    correction_cache: torch.Tensor,
    kd_cache: torch.Tensor,
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    commit_lens: torch.Tensor,
    final_state_indices: torch.Tensor,
    boundary_state_indices: torch.Tensor,
    boundary_recovery_lens: torch.Tensor,
    state_base_addrs: torch.Tensor,
    correction_cache_base_addrs: torch.Tensor,
    kd_cache_base_addrs: torch.Tensor,
    spec_query_len: int,
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
) -> None:
    """Commit one layer's accepted FP32 K/U/decay records in place.

    q/k/v identify the registry's activation format; recovery does not read
    them. checkpoint_state uses [page, head, value, key]. correction_cache
    and kd_cache use [request, head, token, channel], with K then decay in
    the latter. The three base-address tables each contain this layer's
    pool pointer. Tensor strides describe those pools.

    state_indices selects live source/destination pages (> 0); non-positive
    pages are padding. query_start_loc gives each request's token boundaries.
    num_accepted_tokens is clamped to that window and spec_query_len by the
    planner. commit_lens, final_state_indices, boundary_state_indices and
    boundary_recovery_lens are caller-owned int32 planner scratch.

    record_metadata is INT64 [batch, 3] (slot, generation, checkpoint position).
    record_stamps is INT64 [slots, capacity, 2] (position, generation).
    Recovery checks every accepted stamp before reading payloads. Invalid
    records raise a device assertion. Both may be None for unstamped diagnostics.

    Return None after updating checkpoint_state. No gate inputs are needed:
    verify already saved multiplicative decay in kd_cache.
    """
    del q, k, v
    batch = state_indices.shape[0]
    if (record_metadata is None) != (record_stamps is None):
        raise ValueError("record metadata and stamps must be supplied together")
    if record_metadata is not None:
        if record_metadata.shape != (batch, 3) or record_stamps.shape != (
            correction_cache.shape[0],
            correction_cache.shape[2],
            2,
        ):
            raise ValueError("replay metadata does not match the record view")
        if any(
            x.dtype != torch.int64
            or not x.is_contiguous()
            or x.device != checkpoint_state.device
            for x in (record_metadata, record_stamps)
        ):
            raise ValueError(
                "record metadata must be contiguous INT64 on the state device"
            )
    if batch == 0:
        return
    _prepare_commit_plan_kernel[(batch,)](
        num_accepted_tokens,
        None,
        state_indices,
        query_start_loc,
        None,
        None,
        commit_lens,
        final_state_indices,
        boundary_state_indices,
        boundary_recovery_lens,
        NULL_BLOCK_ID,
        1,
        1,
        num_accepted_tokens.stride(0),
        0,
        state_indices.stride(0),
        query_start_loc.stride(0),
        0,
        0,
        0,
        SPEC_QUERY_LEN=spec_query_len,
        num_warps=1,
    )
    _, num_heads, value_dim, key_dim = checkpoint_state.shape
    block_k = triton.next_power_of_2(key_dim)
    grid = lambda meta: (
        triton.cdiv(meta["V"], meta["BV"]),
        batch,
        num_heads,
    )
    _commit_kda_state_kernel[grid](
        checkpoint_state,
        state_base_addrs,
        checkpoint_state.stride(0),
        correction_cache,
        correction_cache_base_addrs,
        correction_cache.stride(0),
        kd_cache,
        kd_cache_base_addrs,
        kd_cache.stride(0),
        state_indices,
        commit_lens,
        final_state_indices,
        boundary_state_indices,
        boundary_recovery_lens,
        NULL_BLOCK_ID,
        checkpoint_state.stride(1),
        checkpoint_state.stride(2),
        checkpoint_state.stride(3),
        correction_cache.stride(1),
        correction_cache.stride(2),
        correction_cache.stride(3),
        kd_cache.stride(1),
        kd_cache.stride(2),
        kd_cache.stride(3),
        state_indices.stride(0),
        None,
        0,
        1,
        record_metadata,
        record_stamps,
        0,
        correction_cache.shape[2],
        correction_cache.shape[0],
        K=key_dim,
        V=value_dim,
        BK=block_k,
        NUM_HEADS=num_heads,
        ALIGN_MODE=False,
        BATCH=batch,
        LAUNCH_DEPENDENT_KERNELS=False,
        debug=True,
        num_stages=2,
    )


__all__ = [
    "vllm_triton_kda_recoverssm_commit",
    "vllm_triton_kda_recoverssm_verify",
]


def batched_recoverssm_commit(
    descriptors: torch.Tensor,
    group_indices: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    *,
    draft_token_num: int,
    num_heads: int,
    head_dim: int,
    qkv_stride: int,
    conv_stride: int,
    correction_stride: int,
    key_decay_stride: int,
    state_stride: int,
    conv_width: int,
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
) -> None:
    """Recover all layers from request-major FP32 records and commit conv.

    Descriptors retain the runtime's ten-column layout. Columns 0, 2 and 8
    hold raw QKV, conv state and recurrent state. Columns 3 and 5 hold
    correction and key/decay records. The remaining columns are unused here
    but retained for the shared descriptor ABI. All strides count elements;
    record token strides may include row padding.

    group_indices maps each layer to a row of read_indices/write_indices.
    Those tables are contiguous [groups, batch]; live pages must be positive.
    Padding uses negative source/destination pages and accepted count zero.
    The shared state-page planner must clamp accepted_length to [0, draft_token_num] before this
    call. A zero count copies the source endpoint without reading records.

    State pages use contiguous [head, value, key] FP32 values; conv pages
    use contiguous [channel, conv_width - 1] BF16 values. Record K/U/decay
    is FP32. Source and destination pages may alias within a request, but
    requests must not share writable pages.

    record_metadata is INT64 [batch, 3] (slot, generation, checkpoint position).
    record_stamps is INT64 [layers, slots, capacity, 2] (position, generation).
    Each recurrent layer validates its accepted interval before reading records.
    Both may be None only for standalone unstamped kernel diagnostics.

    Return None after recurrent and convolution commits on the caller stream.
    """
    from tokenspeed_kernel.ops.attention.kda._triton.recurrent import (
        batched_kda_commit_conv_window_kernel,
    )

    if head_dim != 128 or conv_width != 4:
        raise ValueError("batched KDA commit requires head_dim=128 and conv_width=4")
    layers = descriptors.shape[0]
    if descriptors.shape != (layers, 10) or not descriptors.is_contiguous():
        raise ValueError("descriptors must be a contiguous [layers, 10] pointer table")
    if group_indices.shape != (layers,) or not group_indices.is_contiguous():
        raise ValueError("group_indices must be a contiguous per-layer vector")
    batch = accepted_length.numel()
    if (
        accepted_length.shape != (batch,)
        or read_indices.ndim != 2
        or read_indices.shape[1] != batch
        or write_indices.shape != read_indices.shape
        or any(
            not tensor.is_contiguous()
            for tensor in (accepted_length, read_indices, write_indices)
        )
    ):
        raise ValueError(
            "commit metadata must be contiguous [groups, batch] and [batch]"
        )
    if not batch:
        return
    if (record_metadata is None) != (record_stamps is None):
        raise ValueError("record metadata and stamps must be supplied together")
    capacity = draft_token_num if record_stamps is None else record_stamps.shape[2]
    slots = batch if record_stamps is None else record_stamps.shape[1]
    if record_metadata is not None:
        if record_metadata.shape != (batch, 3) or record_stamps.shape != (
            layers,
            slots,
            capacity,
            2,
        ):
            raise ValueError(
                "batched replay metadata needs [batch, 3] and stamps [layers, slots, capacity, 2]"
            )
        if capacity < draft_token_num:
            raise ValueError("record capacity must cover the candidate window")
        if any(
            x.dtype != torch.int64
            or not x.is_contiguous()
            or x.device != descriptors.device
            for x in (record_metadata, record_stamps)
        ):
            raise ValueError(
                "record metadata must be contiguous INT64 on the descriptor device"
            )
    grid = lambda meta: (
        triton.cdiv(head_dim, meta["BV"]),
        batch,
        layers * num_heads,
    )
    # Empty typed views provide pointer element types. No kernel dereferences
    # these placeholders; all storage comes from the persistent descriptor.
    typed = torch.empty(0, dtype=torch.float32, device=descriptors.device)
    _commit_kda_state_kernel[grid](
        typed,
        descriptors[:, 8],
        state_stride,
        typed,
        descriptors[:, 3],
        capacity * correction_stride,
        typed,
        descriptors[:, 5],
        capacity * key_decay_stride,
        read_indices,
        accepted_length,
        write_indices,
        None,
        None,
        NULL_BLOCK_ID,
        head_dim * head_dim,
        head_dim,
        1,
        head_dim,
        correction_stride,
        1,
        2 * head_dim,
        key_decay_stride,
        1,
        read_indices.stride(1),
        group_indices,
        read_indices.stride(0),
        descriptors.stride(0),
        record_metadata,
        record_stamps,
        0 if record_stamps is None else record_stamps.stride(0),
        capacity,
        slots,
        K=head_dim,
        V=head_dim,
        BK=triton.next_power_of_2(head_dim),
        NUM_HEADS=num_heads,
        ALIGN_MODE=False,
        BATCH=batch,
        LAUNCH_DEPENDENT_KERNELS=False,
        debug=True,
        num_stages=2,
    )
    batched_kda_commit_conv_window_kernel[
        (layers, batch, triton.cdiv(3 * num_heads * head_dim, 128))
    ](
        descriptors,
        group_indices,
        read_indices,
        write_indices,
        accepted_length,
        batch,
        T=draft_token_num,
        STRIDE_QKV=qkv_stride,
        STRIDE_CONV=conv_stride,
        CONV_DIM=3 * num_heads * head_dim,
        BLOCK=128,
        num_warps=1,
    )


@gluon.jit
def _replay_conv_inputs_kernel(
    raw,
    weights,
    pool,
    read_indices,
    output,
    verify_output,
    verify_qk,
    captured_raw,
    stride_raw: gl.constexpr,
    stride_pool: gl.constexpr,
    stride_capture: gl.constexpr,
    T: gl.constexpr,
    P: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK: gl.constexpr,
    VERIFY_SMALL: gl.constexpr,
):
    row = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([4], [32], [gl.num_warps()], [0])
    channel = gl.program_id(1) * BLOCK + gl.arange(0, BLOCK, layout=layout)
    request, position = row // T, row % T
    page = gl.load(read_indices + request).to(gl.int64)
    mask = channel < 3 * P
    feature = channel
    current = gl.load(raw + row * stride_raw + feature, mask=mask, other=0).to(
        gl.float32
    )
    weight = gl.load(weights + feature * 4 + 3, mask=mask, other=0).to(gl.float32)
    # Match _kda_window_step: current tap first, then taps 0, 1, 2.
    current_term = current * weight
    acc = current_term
    verify_acc = gl.full((BLOCK,), 0.0, gl.float32, layout=layout)
    for tap in gl.static_range(3):
        offset = position + tap
        from_raw = offset >= 3
        history = gl.load(
            pool + gl.maximum(page, 0) * stride_pool + feature * 3 + offset,
            mask=mask & ~from_raw & (page >= 0),
            other=0,
        ).to(gl.float32)
        candidate = gl.load(
            raw + (request * T + offset - 3) * stride_raw + feature,
            mask=mask & from_raw,
            other=0,
        ).to(gl.float32)
        weight = gl.load(weights + feature * 4 + tap, mask=mask, other=0).to(gl.float32)
        value = gl.where(from_raw, candidate, history)
        acc = gl.fma(value, weight, acc)
        verify_acc += value * weight
    verify_acc += current * gl.load(weights + feature * 4 + 3, mask=mask, other=0).to(
        gl.float32
    )
    gl.store(
        verify_output + row * (3 * P) + channel,
        verify_acc * (1.0 / (1.0 + gl.exp(-verify_acc))),
        mask=mask,
    )
    # Match main's BF16 producer boundary before FP32 normalization.
    rounded_verify = (
        (verify_acc * (1.0 / (1.0 + gl.exp(-verify_acc))))
        .to(raw.dtype.element_ty)
        .to(gl.float32)
    )
    verify_heads = rounded_verify.reshape((BLOCK // HEAD_DIM, HEAD_DIM))
    if VERIFY_SMALL:
        verify_heads = gl.convert_layout(
            verify_heads, gl.BlockedLayout([1, 1], [1, 32], [1, gl.num_warps()], [1, 0])
        )
    normalized_verify = verify_heads / gl.sqrt(
        gl.sum(verify_heads * verify_heads, axis=1)[:, None] + 1e-6
    )
    normalized_verify = gl.convert_layout(normalized_verify.reshape((BLOCK,)), layout)
    normalized_verify = gl.where(
        feature < P, normalized_verify * HEAD_DIM**-0.5, normalized_verify
    )
    gl.store(
        verify_qk + row * (2 * P) + feature,
        normalized_verify,
        mask=mask & (feature < 2 * P),
    )
    gl.store(captured_raw + row * stride_capture + channel, current, mask=mask)
    activated = acc * (1.0 / (1.0 + gl.exp(-acc)))
    per_head = gl.convert_layout(
        activated.reshape((BLOCK // HEAD_DIM, HEAD_DIM)),
        gl.BlockedLayout([1, 1], [1, 32], [gl.num_warps(), 1], [1, 0]),
    )
    normalized = per_head / gl.sqrt(gl.sum(per_head * per_head, axis=1)[:, None] + 1e-6)
    # Keep pre-SiLU V so record production can fuse SiLU with correction.
    replay_input = gl.where(
        feature < 2 * P, gl.convert_layout(normalized.reshape((BLOCK,)), layout), acc
    )
    gl.store(
        output + row * (2 * P) + feature - P,
        replay_input,
        mask=mask & (feature >= P),
    )


@triton.jit(do_not_specialize=["ROWS"])
def _recoverssm_gate_inputs_kernel(
    fa,
    fb,
    a_log,
    bias,
    raw_gate,
    replay_gate,
    verify_decay,
    stride_fa: tl.constexpr,
    ROWS,
    H: tl.constexpr,
    K: tl.constexpr,
    DFA: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    LOWER: tl.constexpr,
):
    head, tile, column_tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    key = column_tile * BK + tl.arange(0, BK)
    channel = head * K + key
    reduction = tl.arange(0, DFA)
    weights = tl.load(
        fb + channel[:, None] * DFA + reduction[None, :], mask=key[:, None] < K, other=0
    )
    a = tl.exp(tl.load(a_log + head).to(tl.float32))
    dt = tl.load(bias + channel, mask=key < K, other=0).to(tl.float32)
    rows = tile * BT + tl.arange(0, BT)
    values = tl.load(
        fa + rows[:, None] * stride_fa + reduction[None, :],
        mask=rows[:, None] < ROWS,
        other=0,
    )
    gate = tl.dot(values, tl.trans(weights))
    rounded = (
        tl.load(
            raw_gate + rows[:, None] * H * K + channel[None, :],
            mask=(rows[:, None] < ROWS) & (key[None, :] < K),
            other=0,
        ).to(tl.float32)
        + dt[None, :]
    )
    if LOWER is not None:
        verify_log = LOWER * tl.sigmoid(a * rounded)
    else:
        verify_log = -a * tl.where(
            rounded > 20.0, rounded, tl.log(1.0 + tl.exp(rounded))
        )
    tl.store(
        verify_decay + rows[:, None] * H * K + channel[None, :],
        tl.exp(verify_log),
        mask=(rows[:, None] < ROWS) & (key[None, :] < K),
    )
    biased = gate + dt[None, :]
    if LOWER is not None:
        log_decay = LOWER * tl.sigmoid(a * biased)
    else:
        log_decay = -a * tl.where(biased < 20.0, tl.log(1.0 + tl.exp(biased)), biased)
    tl.store(
        replay_gate + rows[:, None] * H * K + channel[None, :],
        tl.exp(log_decay),
        mask=(rows[:, None] < ROWS) & (key[None, :] < K),
    )


@gluon.jit(do_not_specialize=["ROWS"])
def _recoverssm_gate_scalar_kernel(
    fa,
    fb,
    a_log,
    bias,
    raw_gate,
    replay_gate,
    verify_decay,
    stride_fa: gl.constexpr,
    ROWS,
    H: gl.constexpr,
    K: gl.constexpr,
    DFA: gl.constexpr,
    BT: gl.constexpr,
    BK: gl.constexpr,
    LOWER: gl.constexpr,
):
    head, tile, column_tile = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    gate_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [1, gl.num_warps()], [1, 0]
    )
    key = column_tile * BK + gl.arange(0, BK, layout=gl.SliceLayout(1, gate_layout))
    channel = head * K + key
    reduction = gl.arange(0, DFA, layout=gl.SliceLayout(0, gate_layout))
    weights = gl.load(
        fb + channel[:, None] * DFA + reduction[None, :], mask=key[:, None] < K, other=0
    )
    a = gl.exp(gl.load(a_log + head).to(gl.float32))
    dt = gl.load(bias + channel, mask=key < K, other=0).to(gl.float32)
    for index in range(BT):
        row = tile * BT + index
        if row < ROWS:
            values = gl.load(fa + row * stride_fa + reduction).to(gl.float32)
            gate = gl.sum(weights.to(gl.float32) * values[None, :], axis=1)
            rounded = (
                gl.load(raw_gate + row * H * K + channel, mask=key < K, other=0).to(
                    gl.float32
                )
                + dt
            )
            if LOWER is not None:
                verify_log = LOWER * (1.0 / (1.0 + gl.exp(-(a * rounded))))
            else:
                verify_log = -a * gl.where(
                    rounded > 20.0, rounded, gl.log(1.0 + gl.exp(rounded))
                )
            gl.store(
                verify_decay + row * H * K + channel,
                gl.exp(verify_log),
                mask=key < K,
            )
            biased = gate + dt
            if LOWER is not None:
                log_decay = LOWER * (1.0 / (1.0 + gl.exp(-(a * biased))))
            else:
                log_decay = -a * gl.where(
                    biased < 20.0, gl.log(1.0 + gl.exp(biased)), biased
                )
            gl.store(
                replay_gate + row * H * K + channel, gl.exp(log_decay), mask=key < K
            )


def prepare_recoverssm_conv(
    raw: torch.Tensor,
    captured_raw: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_pool: torch.Tensor,
    read_indices: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Prepare convolution inputs for the two independent state chains.

    raw and captured_raw use packed [request * width, 3 * heads * dim]
    Q/K/V rows. conv_weights uses [channel, 4]. conv_pool stores three
    prior values per channel. read_indices selects each request's page.

    Return BF16 verify QKV, FP32 normalized replay K, FP32 pre-SiLU
    replay V, and the FP32 normalized verify Q/K pair. Q includes the
    attention scale. Each chain retains main's convolution tap order.
    Verify normalization occurs after BF16 rounding, never before it.
    """
    rows, channels = raw.shape[0], num_heads * head_dim
    if width <= 0 or rows % width:
        raise ValueError("width must be positive and divide the packed row count")
    if head_dim <= 0 or head_dim & (head_dim - 1) or 256 % head_dim:
        raise ValueError("head_dim must be a power of two that divides 256")
    if raw.shape != (rows, 3 * channels) or raw.stride(-1) != 1:
        raise ValueError("raw must have packed Q/K/V rows with unit channel stride")
    if raw.dtype != torch.bfloat16 or captured_raw.dtype != raw.dtype:
        raise ValueError("raw and captured_raw must use BF16")
    if captured_raw.shape != raw.shape or captured_raw.stride(-1) != 1:
        raise ValueError("captured_raw must match the packed raw layout")
    if conv_weights.shape != (3 * channels, 4) or not conv_weights.is_contiguous():
        raise ValueError("conv_weights must be contiguous [3 * heads * dim, 4]")
    if (
        conv_pool.ndim != 3
        or conv_pool.shape[1] < 3 * channels
        or conv_pool.shape[2] != 3
        or conv_pool.stride()[1:] != (3, 1)
    ):
        raise ValueError("conv_pool must have contiguous channel/tap dimensions")
    if conv_weights.dtype != raw.dtype or conv_pool.dtype != raw.dtype:
        raise ValueError("convolution weights and state must use BF16")
    if (
        read_indices.shape != (rows // width,)
        or read_indices.dtype not in (torch.int32, torch.int64)
        or read_indices.stride(0) != 1
    ):
        raise ValueError("read_indices must be a contiguous integer request vector")
    if any(
        tensor.device != raw.device
        for tensor in (captured_raw, conv_weights, conv_pool, read_indices)
    ):
        raise ValueError("convolution inputs must use the same device")
    # The kernel writes dense rows even when raw is a strided projection view.
    conv = torch.empty_like(raw, memory_format=torch.contiguous_format)
    verify_qk = torch.empty(
        (rows, 2 * channels), device=raw.device, dtype=torch.float32
    )
    kv = torch.empty((rows, 2 * channels), device=raw.device, dtype=torch.float32)
    _replay_conv_inputs_kernel[(rows, triton.cdiv(3 * channels, 256))](
        raw,
        conv_weights,
        conv_pool,
        read_indices,
        kv,
        conv,
        verify_qk,
        captured_raw,
        raw.stride(0),
        conv_pool.stride(0),
        captured_raw.stride(0),
        T=width,
        P=channels,
        HEAD_DIM=head_dim,
        BLOCK=256,
        VERIFY_SMALL=rows // width < 4,
        num_warps=4,
    )
    shape = (1, rows, num_heads, head_dim)
    k, v = kv.chunk(2, dim=-1)
    q_verify, k_verify = verify_qk.chunk(2, dim=-1)
    return (
        conv,
        k.view(shape),
        v.view(shape),
        (q_verify.view(shape), k_verify.view(shape)),
    )


def prepare_recoverssm_gate(
    f_a: torch.Tensor,
    f_b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    num_heads: int,
    num_layers: int,
    head_dim: int,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return BF16 raw gates, FP32 replay decay and FP32 verify decay.

    Low-rank gate inputs and weights use the existing main-path layout.
    num_layers is the number of recurrent layers in the all-layer commit.
    FP32 accumulation follows main replay's shape-based tiling rule.
    """
    from tokenspeed_kernel.ops.attention.kda._triton.recurrent import (
        _gate_tiling,
        _gate_tiling_dot,
    )

    rows, channels = f_a.shape[0], num_heads * head_dim
    if f_a.ndim != 2 or f_a.stride(-1) != 1 or f_a.dtype != torch.bfloat16:
        raise ValueError("f_a must be BF16 [rows, rank] with unit rank stride")
    rank = f_a.shape[1]
    if rank <= 0 or rank & (rank - 1):
        raise ValueError("gate projection rank must be a positive power of two")
    if f_b.shape != (channels, rank) or not f_b.is_contiguous():
        raise ValueError("f_b must be contiguous [heads * dim, rank]")
    if f_b.dtype != f_a.dtype:
        raise ValueError("gate projection weights must match the BF16 inputs")
    if (
        A_log.shape != (num_heads,)
        or dt_bias.numel() != channels
        or not A_log.is_contiguous()
        or not dt_bias.is_contiguous()
    ):
        raise ValueError("gate parameters must have contiguous per-head/channel values")
    if any(tensor.device != f_a.device for tensor in (f_b, A_log, dt_bias)):
        raise ValueError("gate inputs must use the same device")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    raw_gate = torch.nn.functional.linear(f_a, f_b)
    gate = torch.empty((rows, channels), device=f_a.device, dtype=torch.float32)
    verify_decay = torch.empty_like(gate)
    bt, bk = (
        _gate_tiling_dot(rows, head_dim)
        if rows >= 16
        else _gate_tiling(rows, num_heads, head_dim, f_a.device, layers=num_layers)
    )
    gate_kernel = (
        _recoverssm_gate_inputs_kernel if rows >= 16 else _recoverssm_gate_scalar_kernel
    )
    gate_kernel[(num_heads, triton.cdiv(rows, bt), triton.cdiv(head_dim, bk))](
        f_a,
        f_b,
        A_log,
        dt_bias,
        raw_gate,
        gate,
        verify_decay,
        stride_fa=f_a.stride(0),
        ROWS=rows,
        H=num_heads,
        K=head_dim,
        DFA=f_a.shape[-1],
        BT=bt,
        BK=bk,
        LOWER=lower_bound,
        num_warps=1 if bt >= 4 else 2,
    )
    return (
        raw_gate,
        gate.view(1, rows, num_heads, head_dim),
        verify_decay.view(1, rows, num_heads, head_dim),
    )


@triton.jit(do_not_specialize=["batch", "generation"])
def _prepare_record_metadata_kernel(
    committed, metadata, batch, generation, BLOCK: tl.constexpr
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = row < batch
    position = tl.load(committed + row, mask=live, other=0)
    tl.store(metadata + row * 3, row.to(tl.int64), mask=live)
    tl.store(metadata + row * 3 + 1, generation.to(tl.int64), mask=live)
    tl.store(metadata + row * 3 + 2, position, mask=live)


def prepare_recoverssm_metadata(
    committed: torch.Tensor, metadata: torch.Tensor, *, batch: int, generation: int
) -> None:
    """Refresh the per-round slot, generation and checkpoint-position view.

    committed contains INT64 checkpoint positions. metadata is caller-owned,
    contiguous INT64 [max_batch, 3] storage. Only the first batch rows change.
    generation identifies this round's workspace ownership. Return None.
    """
    if committed.dtype != torch.int64 or metadata.dtype != torch.int64:
        raise ValueError("record metadata requires INT64 positions and storage")
    if metadata.ndim != 2 or metadata.shape[1] != 3 or not metadata.is_contiguous():
        raise ValueError("record metadata must be contiguous [max_batch, 3]")
    if batch > min(committed.numel(), metadata.shape[0]) or generation < 0:
        raise ValueError("invalid record metadata batch or generation")
    if batch == 0:
        return
    _prepare_record_metadata_kernel[(triton.cdiv(batch, 128),)](
        committed,
        metadata,
        batch,
        generation,
        BLOCK=128,
    )
