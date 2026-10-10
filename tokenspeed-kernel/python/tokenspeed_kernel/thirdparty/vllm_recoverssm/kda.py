# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 RecoverSSM kernels, adapted to FP32 token-local K/U/D records.

See README.md for the upstream revision and the TokenSpeed adaptations.
"""

from typing import Any

import torch
from tokenspeed_kernel._triton import gl, gluon, tl, triton

# vLLM reserves page zero as a null page. TokenSpeed adapters pass the same
# page convention to these copied kernels.
NULL_BLOCK_ID = 0


@gluon.jit
def _sigmoid(value):
    return 1.0 / (1.0 + gl.exp(-value))


@gluon.jit
def _kda_gate(
    raw_g,
    dt_bias,
    A,
    lower_bound,
    USE_LOWER_BOUND: gl.constexpr,
):
    gate_input = raw_g + dt_bias
    if USE_LOWER_BOUND:
        return lower_bound * _sigmoid(A * gate_input)
    softplus_gate = gl.where(
        gate_input > 20.0,
        gate_input,
        gl.log(1.0 + gl.exp(gate_input)),
    )
    return -A * softplus_gate


@gluon.jit
def _kda_recurrent_step(
    state,
    k,
    v,
    raw_g,
    raw_beta,
    dt_bias,
    A,
    lower_bound,
    USE_LOWER_BOUND: gl.constexpr,
    PREPARED_GATE: gl.constexpr,
    PREPARED_QK: gl.constexpr,
):
    if PREPARED_QK:
        normalized_k = k
    else:
        normalized_k = k * gl.rsqrt(gl.sum(k * k) + 1e-6)
    if PREPARED_GATE:
        decay = raw_g
    else:
        gate = _kda_gate(raw_g, dt_bias, A, lower_bound, USE_LOWER_BOUND)
        decay = gl.exp(gate)
    state *= decay[None, :]
    correction = v - gl.sum(state * normalized_k[None, :], axis=1)
    correction *= _sigmoid(raw_beta)
    return (
        state + correction[:, None] * normalized_k[None, :],
        correction,
        normalized_k,
        decay,
    )


@triton.heuristics(
    {
        # Use smaller value tiles to expose enough CTAs at small batches.
        # Larger batches amortize input loads across more values per CTA.
        "BV": lambda args: min(
            triton.next_power_of_2(args["V"]),
            (
                (8 if args["BATCH"] <= 4 else 16)
                if args["SEPARATE_REPLAY_INPUTS"]
                else 4 if args["BATCH"] <= 4 else 8 if args["BATCH"] <= 8 else 16
            ),
        )
    }
)
@gluon.jit(do_not_specialize=["RECORD_SLOTS"])
def _kda_recoverssm_verify_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    raw_g_ptr,
    raw_beta_ptr,
    replay_k_ptr,
    replay_v_ptr,
    replay_decay_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    correction_cache_ptr,
    kd_cache_ptr,
    out_ptr,
    query_start_loc_ptr,
    state_indices_ptr,
    record_metadata_ptr,
    record_stamps_ptr,
    RECORD_CAPACITY: gl.constexpr,
    RECORD_SLOTS,
    lower_bound,
    null_block_id,
    stride_q_token: gl.constexpr,
    stride_k_token: gl.constexpr,
    stride_v_token: gl.constexpr,
    stride_g_token: gl.constexpr,
    stride_beta_token: gl.constexpr,
    stride_replay_token: gl.constexpr,
    stride_replay_gate_token: gl.constexpr,
    stride_state_block: gl.constexpr,
    stride_state_head: gl.constexpr,
    stride_state_v: gl.constexpr,
    stride_state_k: gl.constexpr,
    stride_correction_block: gl.constexpr,
    stride_correction_head: gl.constexpr,
    stride_correction_pos: gl.constexpr,
    stride_correction_dim: gl.constexpr,
    stride_kg_block: gl.constexpr,
    stride_kg_head: gl.constexpr,
    stride_kg_pos: gl.constexpr,
    stride_kg_dim: gl.constexpr,
    stride_out_token: gl.constexpr,
    stride_query_start_loc: gl.constexpr,
    stride_state_indices: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    SPEC_QUERY_LEN: gl.constexpr,
    USE_LOWER_BOUND: gl.constexpr,
    BATCH: gl.constexpr,
    SEPARATE_REPLAY_INPUTS: gl.constexpr,
    PREPARED_GATE: gl.constexpr,
    PREPARED_QK: gl.constexpr,
):
    pid_v = gl.program_id(0)
    pid_b = gl.program_id(1)
    pid_h = gl.program_id(2)
    record_only = False
    if SEPARATE_REPLAY_INPUTS:
        record_only = (pid_h % 2) != 0
        pid_h = pid_h // 2

    bos = gl.load(query_start_loc_ptr + pid_b * stride_query_start_loc).to(gl.int64)
    eos = gl.load(query_start_loc_ptr + (pid_b + 1) * stride_query_start_loc).to(
        gl.int64
    )
    query_len = eos - bos
    state_idx = gl.load(state_indices_ptr + pid_b * stride_state_indices).to(gl.int64)

    # Four value rows share each warp. Eight lanes reduce each key row,
    # limiting shuffle work while preserving ordered FP32 token updates.
    state_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [4, 8], [1, gl.num_warps()], [1, 0]
    )
    key_layout: gl.constexpr = gl.SliceLayout(0, state_layout)
    value_layout: gl.constexpr = gl.SliceLayout(1, state_layout)
    offs_k = gl.arange(0, BK, layout=key_layout)
    offs_v = pid_v * BV + gl.arange(0, BV, layout=value_layout)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]

    if state_idx <= null_block_id:
        if record_only:
            return
        for token_offset in gl.static_range(SPEC_QUERY_LEN):
            token_valid = token_offset < query_len
            gl.store(
                out_ptr + (bos + token_offset) * stride_out_token + pid_h * V + offs_v,
                gl.zeros([BV], dtype=gl.float32, layout=value_layout),
                mask=token_valid & mask_v,
            )
        return

    record_slot = pid_b.to(gl.int64)
    record_start = record_slot * 0
    record_generation = record_start
    if record_metadata_ptr is not None and (not SEPARATE_REPLAY_INPUTS or record_only):
        record_slot = gl.load(record_metadata_ptr + pid_b * 3)
        record_generation = gl.load(record_metadata_ptr + pid_b * 3 + 1)
        record_start = gl.load(record_metadata_ptr + pid_b * 3 + 2)
        gl.device_assert(
            (record_slot >= 0) & (record_slot < RECORD_SLOTS), "invalid replay slot"
        )
        gl.device_assert(
            (record_generation >= 0) & (record_start >= 0), "invalid replay metadata"
        )

    # A later recovery launch observes stamps only after this entire producer
    # completes. Write them before loading state so generation is not live
    # across the recurrent updates.
    if record_stamps_ptr is not None:
        if (
            (not SEPARATE_REPLAY_INPUTS or record_only)
            and (pid_v == 0)
            and (pid_h == 0)
        ):
            for token_offset in gl.static_range(SPEC_QUERY_LEN):
                position = record_start + token_offset
                stamp = (
                    record_stamps_ptr
                    + (record_slot * RECORD_CAPACITY + position % RECORD_CAPACITY) * 2
                )
                gl.store(stamp, position, mask=token_offset < query_len)
                gl.store(stamp + 1, record_generation, mask=token_offset < query_len)

    state_ptrs = (
        state_ptr
        + state_idx * stride_state_block
        + pid_h * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    state = gl.load(state_ptrs, mask=mask_state, other=0.0).to(gl.float32)

    # Each CTA keeps one state tile. Separate head-grid entries preserve the
    # independent precision chains without doubling per-thread state storage.
    # Verify and record production use different inputs but the same initial
    # checkpoint. Independent CTAs avoid keeping both state tiles live.
    if SEPARATE_REPLAY_INPUTS:
        if record_only:
            for token_offset in gl.static_range(SPEC_QUERY_LEN):
                token_valid = token_offset < query_len
                token = bos + token_offset
                replay_k = gl.load(
                    replay_k_ptr + token * stride_replay_token + pid_h * K + offs_k,
                    mask=token_valid & mask_k,
                    other=0.0,
                )
                replay_v = gl.load(
                    replay_v_ptr + token * stride_replay_token + pid_h * V + offs_v,
                    mask=token_valid & mask_v,
                    other=0.0,
                )
                replay_g = gl.load(
                    replay_decay_ptr
                    + token * stride_replay_gate_token
                    + pid_h * K
                    + offs_k,
                    mask=token_valid & mask_k,
                    other=0.0,
                )
                raw_beta = gl.load(
                    raw_beta_ptr + token * stride_beta_token + pid_h,
                    mask=token_valid,
                    other=0.0,
                ).to(gl.float32)
                normalized_k = replay_k
                decay = replay_g
                decayed_state = state * decay[None, :]
                correction = replay_v - gl.sum(
                    decayed_state * normalized_k[None, :], axis=1
                )
                correction *= _sigmoid(raw_beta)
                updated = decayed_state + correction[:, None] * normalized_k[None, :]
                # Invalid tokens form a suffix; no later result observes this state.
                state = updated
                correction_ptr = (
                    correction_cache_ptr
                    + record_slot * stride_correction_block
                    + pid_h * stride_correction_head
                    + ((record_start + token_offset) % RECORD_CAPACITY)
                    * stride_correction_pos
                )
                gl.store(
                    correction_ptr + offs_v * stride_correction_dim,
                    correction,
                    mask=token_valid & mask_v,
                )
                if pid_v == 0:
                    kd_ptr = (
                        kd_cache_ptr
                        + record_slot * stride_kg_block
                        + pid_h * stride_kg_head
                        + ((record_start + token_offset) % RECORD_CAPACITY)
                        * stride_kg_pos
                    )
                    gl.store(
                        kd_ptr + offs_k * stride_kg_dim,
                        normalized_k,
                        mask=token_valid & mask_k,
                    )
                    gl.store(
                        kd_ptr + (K + offs_k) * stride_kg_dim,
                        decay,
                        mask=token_valid & mask_k,
                    )
            return

    A = gl.exp(gl.load(A_log_ptr + pid_h).to(gl.float32))
    dt_bias = gl.load(
        dt_bias_ptr + pid_h * K + offs_k,
        mask=mask_k,
        other=0.0,
    ).to(gl.float32)

    for token_offset in gl.static_range(SPEC_QUERY_LEN):
        token_valid = token_offset < query_len
        token = bos + token_offset
        q = gl.load(
            q_ptr + token * stride_q_token + pid_h * K + offs_k,
            mask=token_valid & mask_k,
            other=0.0,
            eviction_policy="evict_last",
        ).to(gl.float32)
        k = gl.load(
            k_ptr + token * stride_k_token + pid_h * K + offs_k,
            mask=token_valid & mask_k,
            other=0.0,
            eviction_policy="evict_last",
        ).to(gl.float32)
        v = gl.load(
            v_ptr + token * stride_v_token + pid_h * V + offs_v,
            mask=token_valid & mask_v,
            other=0.0,
            eviction_policy="evict_first",
        ).to(gl.float32)
        raw_g = gl.load(
            raw_g_ptr + token * stride_g_token + pid_h * K + offs_k,
            mask=token_valid & mask_k,
            other=0.0,
            eviction_policy="evict_last",
        ).to(gl.float32)
        raw_beta = gl.load(
            raw_beta_ptr + token * stride_beta_token + pid_h,
            mask=token_valid,
            other=0.0,
            eviction_policy="evict_last",
        ).to(gl.float32)

        if not PREPARED_QK:
            q *= gl.rsqrt(gl.sum(q * q) + 1e-6) * (K**-0.5)
        updated_state, correction, normalized_k, decay = _kda_recurrent_step(
            state,
            k,
            v,
            raw_g,
            raw_beta,
            dt_bias,
            A,
            lower_bound,
            USE_LOWER_BOUND,
            PREPARED_GATE,
            PREPARED_QK,
        )
        # Output and record stores mask the suffix. Preserving its state would
        # add a full-tile select without changing any observable result.
        state = updated_state

        out = gl.sum(state * q[None, :], axis=1)
        gl.store(
            out_ptr + token * stride_out_token + pid_h * V + offs_v,
            out,
            mask=token_valid & mask_v,
            eviction_policy="evict_first",
        )

        if not SEPARATE_REPLAY_INPUTS:
            correction_ptr = (
                correction_cache_ptr
                + record_slot * stride_correction_block
                + pid_h * stride_correction_head
                + ((record_start + token_offset) % RECORD_CAPACITY)
                * stride_correction_pos
            )
            gl.store(
                correction_ptr + offs_v * stride_correction_dim,
                correction,
                mask=token_valid & mask_v,
            )
            if pid_v == 0:
                kg_ptr = (
                    kd_cache_ptr
                    + record_slot * stride_kg_block
                    + pid_h * stride_kg_head
                    + ((record_start + token_offset) % RECORD_CAPACITY) * stride_kg_pos
                )
                gl.store(
                    kg_ptr + offs_k * stride_kg_dim,
                    normalized_k,
                    mask=token_valid & mask_k,
                )
                gl.store(
                    kg_ptr + (K + offs_k) * stride_kg_dim,
                    decay,
                    mask=token_valid & mask_k,
                )


@triton.heuristics(
    {
        "HAS_REQUEST_INDICES": lambda args: args["request_indices_ptr"] is not None,
        "ALIGN_MODE": lambda args: args["block_table_ptr"] is not None,
    }
)
@triton.jit
def _prepare_commit_plan_kernel(
    num_accepted_ptr,
    request_indices_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    block_table_ptr,
    num_computed_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    mamba_block_size,
    block_table_width,
    stride_num_accepted,
    stride_request_indices,
    stride_state_indices,
    stride_query_start_loc,
    stride_block_table_row,
    stride_block_table_col,
    stride_num_computed,
    SPEC_QUERY_LEN: tl.constexpr,
    HAS_REQUEST_INDICES: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
):
    spec_idx = tl.program_id(0)
    source_state_idx = tl.load(state_indices_ptr + spec_idx * stride_state_indices).to(
        tl.int64
    )
    request_idx = spec_idx
    if HAS_REQUEST_INDICES:
        request_idx = tl.load(
            request_indices_ptr + spec_idx * stride_request_indices
        ).to(tl.int64)
    num_accepted = tl.load(num_accepted_ptr + request_idx * stride_num_accepted).to(
        tl.int32
    )
    bos = tl.load(query_start_loc_ptr + spec_idx * stride_query_start_loc).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (spec_idx + 1) * stride_query_start_loc).to(
        tl.int64
    )
    query_len = (eos - bos).to(tl.int32)
    commit_len = tl.minimum(tl.maximum(num_accepted, 0), query_len)
    commit_len = tl.minimum(commit_len, SPEC_QUERY_LEN)

    final_state_idx = source_state_idx
    boundary_state_idx = null_block_id
    boundary_recovery_len = 0
    if ALIGN_MODE:
        num_computed = tl.load(num_computed_ptr + request_idx * stride_num_computed).to(
            tl.int32
        )
        final_num_computed = num_computed + commit_len
        final_state_col = tl.minimum(
            final_num_computed // mamba_block_size, block_table_width - 1
        )
        final_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + final_state_col * stride_block_table_col
        ).to(tl.int64)
        next_boundary = (num_computed // mamba_block_size + 1) * mamba_block_size
        crosses_boundary = final_num_computed >= next_boundary
        boundary_recovery_len = next_boundary - num_computed
        boundary_state_idx = tl.load(
            block_table_ptr
            + request_idx * stride_block_table_row
            + (next_boundary // mamba_block_size - 1) * stride_block_table_col,
            mask=crosses_boundary,
            other=null_block_id,
        ).to(tl.int64)
    valid = (source_state_idx > null_block_id) & (commit_len > 0)
    tl.store(commit_lens_ptr + spec_idx, tl.where(valid, commit_len, 0))
    tl.store(
        final_state_indices_ptr + spec_idx,
        tl.where(valid, final_state_idx, null_block_id),
    )
    tl.store(
        boundary_state_indices_ptr + spec_idx,
        tl.where(valid, boundary_state_idx, null_block_id),
    )
    tl.store(
        boundary_recovery_lens_ptr + spec_idx,
        tl.where(valid, boundary_recovery_len, 0),
    )


def _commit_kda_value_block(args: dict[str, Any]) -> int:
    batch = args["BATCH"]
    if not args["ALIGN_MODE"]:
        return 4 if batch <= 2 else 16
    if batch == 2:
        return 16
    return 32 if batch == 4 else 8


def _commit_kda_num_warps(args: dict[str, Any]) -> int:
    if not args["ALIGN_MODE"]:
        return 1
    return 4 if args["BATCH"] == 1 else 2 if args["BATCH"] == 2 else 1


@triton.heuristics(
    {
        "BV": _commit_kda_value_block,
        "num_warps": _commit_kda_num_warps,
    }
)
@triton.jit(do_not_specialize=["BATCH", "RECORD_SLOTS"])
def _commit_kda_state_kernel(
    state_ref_ptr,
    state_base_addrs_ptr,
    state_block_stride: tl.constexpr,
    correction_cache_ref_ptr,
    correction_cache_base_addrs_ptr,
    correction_cache_block_stride: tl.constexpr,
    kd_cache_ref_ptr,
    kd_cache_base_addrs_ptr,
    kd_cache_block_stride: tl.constexpr,
    state_indices_ptr,
    commit_lens_ptr,
    final_state_indices_ptr,
    boundary_state_indices_ptr,
    boundary_recovery_lens_ptr,
    null_block_id,
    stride_state_head: tl.constexpr,
    stride_state_v: tl.constexpr,
    stride_state_k: tl.constexpr,
    stride_correction_cache_head: tl.constexpr,
    stride_correction_cache_pos: tl.constexpr,
    stride_correction_cache_dim: tl.constexpr,
    stride_kd_cache_head: tl.constexpr,
    stride_kd_cache_pos: tl.constexpr,
    stride_kd_cache_dim: tl.constexpr,
    stride_state_indices: tl.constexpr,
    group_indices_ptr,
    stride_state_group,
    stride_pointer_layer: tl.constexpr,
    record_metadata_ptr,
    record_stamps_ptr,
    stamp_layer_stride: tl.constexpr,
    RECORD_CAPACITY: tl.constexpr,
    RECORD_SLOTS,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    ALIGN_MODE: tl.constexpr,
    BATCH,
    LAUNCH_DEPENDENT_KERNELS: tl.constexpr,
):
    if LAUNCH_DEPENDENT_KERNELS:
        # Expected operation is plan -> KDA commit -> conv where plan and KDA
        # are launched normally, conv is launched as a dependent kernel and KDA
        # immediately releases conv.  This guarantees correctness as both KDA
        # and conv depend on the plan but not each other.  Better than two
        # streams as we want to make sure KDA goes first, as it is longer tailed.
        tl.extra.cuda.gdc_launch_dependents()
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_lh = tl.program_id(2)
    pid_l = pid_lh // NUM_HEADS
    pid_h = pid_lh % NUM_HEADS

    group = 0
    if group_indices_ptr is not None:
        group = tl.load(group_indices_ptr + pid_l).to(tl.int64)
    plan_row = group * BATCH + pid_b
    source_state_idx = tl.load(
        state_indices_ptr + group * stride_state_group + pid_b * stride_state_indices
    ).to(tl.int64)
    if source_state_idx <= null_block_id:
        return
    commit_len = tl.load(commit_lens_ptr + pid_b)
    final_state_idx = tl.load(final_state_indices_ptr + plan_row).to(tl.int64)
    if ALIGN_MODE:
        boundary_state_idx = tl.load(boundary_state_indices_ptr + plan_row).to(tl.int64)
        boundary_recovery_len = tl.load(boundary_recovery_lens_ptr + plan_row)

    if final_state_idx <= null_block_id:
        return

    record_slot = pid_b
    record_start = 0
    if record_metadata_ptr is not None:
        record_slot = tl.load(record_metadata_ptr + pid_b * 3)
        generation = tl.load(record_metadata_ptr + pid_b * 3 + 1)
        record_start = tl.load(record_metadata_ptr + pid_b * 3 + 2)
        slot_valid = (record_slot >= 0) & (record_slot < RECORD_SLOTS)
        interval_valid = (
            (commit_len >= 0)
            & (commit_len <= RECORD_CAPACITY)
            & (record_start >= 0)
            & (generation >= 0)
        )
        tl.device_assert(slot_valid & interval_valid, "invalid replay interval")
        if not (slot_valid & interval_valid):
            return
        offsets = tl.arange(0, triton.next_power_of_2(RECORD_CAPACITY))
        positions = record_start + offsets
        stamps = (
            record_stamps_ptr
            + pid_l * stamp_layer_stride
            + (record_slot * RECORD_CAPACITY + positions % RECORD_CAPACITY) * 2
        )
        stamped_position = tl.load(stamps, mask=offsets < commit_len, other=-1)
        stamped_generation = tl.load(stamps + 1, mask=offsets < commit_len, other=-1)
        valid = (
            tl.sum(
                (
                    (offsets < commit_len)
                    & (
                        (stamped_position != positions)
                        | (stamped_generation != generation)
                    )
                ).to(tl.int32),
                axis=0,
            )
            == 0
        )
        tl.device_assert(valid, "stale replay record")
        if not valid:
            return

    state_base_addr = tl.load(state_base_addrs_ptr + pid_l * stride_pointer_layer)
    state_ptr = state_base_addr.to(tl.pointer_type(state_ref_ptr.dtype.element_ty))
    source_state_ptr = (
        state_ptr + source_state_idx * state_block_stride + pid_h * stride_state_head
    )

    correction_cache_base_addr = tl.load(
        correction_cache_base_addrs_ptr + pid_l * stride_pointer_layer
    )
    correction_cache_ptr = correction_cache_base_addr.to(
        tl.pointer_type(correction_cache_ref_ptr.dtype.element_ty)
    )
    correction_cache_ptr += (
        record_slot * correction_cache_block_stride
        + pid_h * stride_correction_cache_head
    )
    kd_cache_base_addr = tl.load(kd_cache_base_addrs_ptr + pid_l * stride_pointer_layer)
    kd_cache_ptr = kd_cache_base_addr.to(
        tl.pointer_type(kd_cache_ref_ptr.dtype.element_ty)
    )
    kd_cache_ptr += record_slot * kd_cache_block_stride + pid_h * stride_kd_cache_head

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]
    state_ptrs = (
        source_state_ptr
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    initial_state = tl.load(state_ptrs, mask=mask_state, other=0.0).to(tl.float32)
    state = initial_state
    if ALIGN_MODE:
        boundary_ptrs = (
            state_ptr
            + boundary_state_idx * state_block_stride
            + pid_h * stride_state_head
            + offs_v[:, None] * stride_state_v
            + offs_k[None, :] * stride_state_k
        )

    for token_offset in range(commit_len):
        correction_ptr = (
            correction_cache_ptr
            + ((record_start + token_offset) % RECORD_CAPACITY)
            * stride_correction_cache_pos
        )
        kg_ptr = (
            kd_cache_ptr
            + ((record_start + token_offset) % RECORD_CAPACITY) * stride_kd_cache_pos
        )
        k = tl.load(
            kg_ptr + offs_k * stride_kd_cache_dim,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        correction = tl.load(
            correction_ptr + offs_v * stride_correction_cache_dim,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        decay = tl.load(
            kg_ptr + (K + offs_k) * stride_kd_cache_dim,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        state *= decay[None, :]
        state += correction[:, None] * k[None, :]
        if ALIGN_MODE:  # noqa: SIM102
            if token_offset == boundary_recovery_len - 1:
                tl.store(boundary_ptrs, state)

    final_ptrs = (
        state_ptr
        + final_state_idx * state_block_stride
        + pid_h * stride_state_head
        + offs_v[:, None] * stride_state_v
        + offs_k[None, :] * stride_state_k
    )
    tl.store(final_ptrs, state, mask=mask_state)


def kda_recoverssm_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    checkpoint_state: torch.Tensor,
    correction_cache: torch.Tensor,
    kd_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    out: torch.Tensor,
    replay_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    verify_decay: torch.Tensor | None,
    verify_qk: tuple[torch.Tensor, torch.Tensor] | None,
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
) -> torch.Tensor:
    """Verify a KDA speculative window without modifying its checkpoint."""
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("KDA RecoverSSM q must have shape [1, tokens, heads, dim]")
    _, total_tokens, num_heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if k.shape != q.shape or v.shape != (1, total_tokens, num_heads, value_dim):
        raise ValueError("KDA RecoverSSM q, k, and v shapes are incompatible")
    if raw_g.shape != q.shape or raw_beta.shape != (1, total_tokens, num_heads):
        raise ValueError("KDA RecoverSSM gate or beta shape is incompatible")
    if any(tensor.stride()[2:] != (key_dim, 1) for tensor in (q, k, raw_g)):
        raise ValueError("KDA RecoverSSM q, k, and gate heads must be contiguous")
    if v.stride()[2:] != (value_dim, 1) or raw_beta.stride(2) != 1:
        raise ValueError("KDA RecoverSSM v and beta heads must be contiguous")
    if checkpoint_state.shape[1:] != (
        num_heads,
        value_dim,
        key_dim,
    ):
        raise ValueError("KDA RecoverSSM checkpoint shape is incompatible")
    expected_correction_shape = (
        correction_cache.shape[0],
        num_heads,
        correction_cache.shape[2],
        value_dim,
    )
    if correction_cache.shape != expected_correction_shape:
        raise ValueError(
            f"KDA RecoverSSM correction buffer needs shape {expected_correction_shape}"
        )
    expected_kg_shape = (
        correction_cache.shape[0],
        num_heads,
        correction_cache.shape[2],
        2 * key_dim,
    )
    if kd_cache.shape != expected_kg_shape:
        raise ValueError(
            f"KDA RecoverSSM key/gate buffer needs shape {expected_kg_shape}"
        )
    if correction_cache.dtype != torch.float32:
        raise ValueError("KDA RecoverSSM correction buffer must use float32")
    if kd_cache.dtype != torch.float32:
        raise ValueError("KDA RecoverSSM normalized key/decay buffer must use float32")
    if A_log.shape != (num_heads,) or dt_bias.numel() != num_heads * key_dim:
        raise ValueError("KDA RecoverSSM gate parameters are incompatible")
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        raise ValueError("KDA RecoverSSM gate parameters must be contiguous")
    batch = state_indices.shape[0]
    capacity = correction_cache.shape[2]
    if capacity < spec_query_len:
        raise ValueError("replay capacity must cover the candidate window")
    if (record_metadata is None) != (record_stamps is None):
        raise ValueError("record metadata and stamps must be supplied together")
    if record_metadata is not None:
        if record_metadata.shape != (batch, 3) or record_stamps.shape != (
            correction_cache.shape[0],
            capacity,
            2,
        ):
            raise ValueError(
                "replay metadata needs [batch, 3] and stamps [slots, capacity, 2]"
            )
        if any(
            x.dtype != torch.int64 or not x.is_contiguous() or x.device != q.device
            for x in (record_metadata, record_stamps)
        ):
            raise ValueError(
                "replay metadata and stamps must be contiguous INT64 on the input device"
            )
    if correction_cache.shape[0] < batch:
        raise ValueError("record capacity is smaller than the runtime batch")
    if query_start_loc.shape[0] != batch + 1:
        raise ValueError("KDA RecoverSSM query metadata is incompatible")
    if total_tokens > batch * spec_query_len:
        raise ValueError(
            "KDA RecoverSSM speculative decode input exceeds its activation capacity"
        )
    if out.shape != v.shape:
        raise ValueError("KDA RecoverSSM output shape is incompatible")
    if out.stride()[2:] != (value_dim, 1):
        raise ValueError("KDA RecoverSSM output heads must be contiguous")
    device = q.device
    if any(
        tensor.device != device
        for tensor in (
            k,
            v,
            raw_g,
            raw_beta,
            A_log,
            dt_bias,
            checkpoint_state,
            correction_cache,
            kd_cache,
            query_start_loc,
            state_indices,
            out,
        )
    ):
        raise ValueError("KDA RecoverSSM inputs must be on the same device")
    if total_tokens == 0:
        return out

    if replay_inputs is None:
        replay_k, replay_v, replay_decay = k, v, raw_g
    else:
        replay_k, replay_v, replay_decay = replay_inputs
        if any(tensor.dtype != torch.float32 for tensor in replay_inputs):
            raise ValueError("replay producers must retain FP32 precision")
        if (
            replay_k.shape != k.shape
            or replay_v.shape != v.shape
            or replay_decay.shape != raw_g.shape
        ):
            raise ValueError("replay producer shapes must match verify inputs")
        if any(tensor.device != device for tensor in replay_inputs):
            raise ValueError("replay inputs must use the verify device")
        if (
            replay_k.stride()[2:] != (key_dim, 1)
            or replay_v.stride()[2:] != (value_dim, 1)
            or replay_decay.stride()[2:] != (key_dim, 1)
            or replay_v.stride(1) != replay_k.stride(1)
        ):
            raise ValueError(
                "replay inputs require contiguous heads and matching K/V token strides"
            )
    if verify_decay is not None:
        if verify_decay.shape != raw_g.shape or verify_decay.dtype != torch.float32:
            raise ValueError("prepared verify decay must match raw gates in FP32")
        if verify_decay.device != device or verify_decay.stride()[2:] != (key_dim, 1):
            raise ValueError(
                "prepared verify decay must use the input device and contiguous heads"
            )
        raw_g = verify_decay
    if verify_qk is not None:
        for tensor in verify_qk:
            if (
                tensor.shape != q.shape
                or tensor.dtype != torch.float32
                or tensor.device != device
                or tensor.stride()[2:] != (key_dim, 1)
            ):
                raise ValueError(
                    "prepared Q/K must match input shapes, device and contiguous FP32 heads"
                )
        q, k = verify_qk
    block_k = triton.next_power_of_2(key_dim)
    grid = lambda meta: (
        triton.cdiv(value_dim, meta["BV"]),
        batch,
        num_heads * (2 if replay_inputs is not None else 1),
    )
    _kda_recoverssm_verify_kernel[grid](
        q,
        k,
        v,
        raw_g,
        raw_beta,
        replay_k,
        replay_v,
        replay_decay,
        A_log,
        dt_bias,
        checkpoint_state,
        correction_cache,
        kd_cache,
        out,
        query_start_loc,
        state_indices,
        record_metadata,
        record_stamps,
        capacity,
        correction_cache.shape[0],
        lower_bound or 0.0,
        NULL_BLOCK_ID,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        raw_g.stride(1),
        raw_beta.stride(1),
        replay_k.stride(1),
        replay_decay.stride(1),
        checkpoint_state.stride(0),
        checkpoint_state.stride(1),
        checkpoint_state.stride(2),
        checkpoint_state.stride(3),
        correction_cache.stride(0),
        correction_cache.stride(1),
        correction_cache.stride(2),
        correction_cache.stride(3),
        kd_cache.stride(0),
        kd_cache.stride(1),
        kd_cache.stride(2),
        kd_cache.stride(3),
        out.stride(1),
        query_start_loc.stride(0),
        state_indices.stride(0),
        K=key_dim,
        V=value_dim,
        BK=block_k,
        SPEC_QUERY_LEN=spec_query_len,
        USE_LOWER_BOUND=lower_bound is not None,
        BATCH=triton.next_power_of_2(batch),
        SEPARATE_REPLAY_INPUTS=replay_inputs is not None,
        PREPARED_GATE=verify_decay is not None,
        PREPARED_QK=verify_qk is not None,
        num_warps=1,
        debug=True,
        num_stages=2,
    )
    return out
