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

"""Accuracy tests for the vendored vLLM KDA RecoverSSM kernels."""

from __future__ import annotations

import pytest
import torch
from kda_replay_reference import batched_recurrent_kda_replay_commit

if not torch.cuda.is_available():
    pytest.skip("requires an NVIDIA GPU", allow_module_level=True)

from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (  # noqa: E402
    prepare_recoverssm_conv,
    prepare_recoverssm_gate,
    vllm_triton_kda_recoverssm_commit,
    vllm_triton_kda_recoverssm_verify,
)


def _device_addresses(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.tensor(
        [tensor.data_ptr() for tensor in tensors],
        device=tensors[0].device,
        dtype=torch.int64,
    )


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    lower_bound: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    q32 = q.float()
    k32 = k.float()
    q32 *= torch.rsqrt(q32.square().sum(dim=-1, keepdim=True) + 1e-6)
    q32 *= q.shape[-1] ** -0.5
    k32 *= torch.rsqrt(k32.square().sum(dim=-1, keepdim=True) + 1e-6)
    gate = lower_bound * torch.sigmoid(
        A_log.exp()[None, None, :, None] * (raw_g.float() + dt_bias[None, None, :, :])
    )
    beta = raw_beta.float().sigmoid()
    current = state.float().clone()
    output = torch.empty_like(v, dtype=torch.float32)
    states = []
    for token in range(q.shape[1]):
        current *= gate[:, token].exp()[:, :, None, :]
        correction = v[:, token].float() - torch.einsum(
            "bhvk,bhk->bhv", current, k32[:, token]
        )
        correction *= beta[:, token, :, None]
        current += torch.einsum("bhv,bhk->bhvk", correction, k32[:, token])
        output[:, token] = torch.einsum("bhvk,bhk->bhv", current, q32[:, token])
        states.append(current.clone())
    return output, states


@pytest.mark.parametrize(
    "width,accepted", [(1, 0), (1, 1), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
)
@pytest.mark.parametrize("dim,prepared_qk", [(128, False), (64, True)])
def test_vllm_recoverssm_verify_and_commit(
    width: int, accepted: int, dim: int, prepared_qk: bool
) -> None:
    torch.manual_seed(7331)
    device = "cuda"
    batch, heads = 4, 12
    q = torch.randn(batch, width, heads, dim, device=device).bfloat16()
    k = torch.randn_like(q)
    v = torch.randn_like(q).mul_(0.2)
    raw_g = torch.randn_like(q)
    raw_beta = torch.randn(batch, width, heads, device=device).bfloat16()
    A_log = torch.randn(heads, device=device).mul_(0.1)
    dt_bias = torch.randn(heads, dim, device=device).mul_(0.1)
    initial_state = torch.randn(batch, heads, dim, dim, device=device).mul_(0.02)
    expected_out, expected_states = _reference(
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        initial_state,
        -5.0,
    )

    state_pool = torch.zeros(
        batch + 1, heads, dim, dim, device=device, dtype=torch.float32
    )
    state_pool[1:].copy_(initial_state)
    correction_cache = torch.empty(
        batch + 1, heads, width, dim, device=device, dtype=torch.float32
    )
    kd_cache = torch.empty(
        batch + 1, heads, width, 2 * dim, device=device, dtype=torch.float32
    )
    state_indices = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
    query_start_loc = torch.arange(
        0, (batch + 1) * width, width, device=device, dtype=torch.int32
    )
    packed_q = q.reshape(1, batch * width, heads, dim)
    packed_k = k.reshape_as(packed_q)
    packed_v = v.reshape_as(packed_q)
    packed_g = raw_g.reshape_as(packed_q)
    packed_beta = raw_beta.reshape(1, batch * width, heads)
    out = torch.empty_like(packed_v)
    verify_qk = None
    if prepared_qk:
        normalized_q = packed_q.float()
        normalized_k = packed_k.float()
        normalized_q *= torch.rsqrt(normalized_q.square().sum(-1, keepdim=True) + 1e-6)
        normalized_q *= dim**-0.5
        normalized_k *= torch.rsqrt(normalized_k.square().sum(-1, keepdim=True) + 1e-6)
        verify_qk = (normalized_q, normalized_k)
    vllm_triton_kda_recoverssm_verify(
        packed_q,
        packed_k,
        packed_v,
        packed_g,
        packed_beta,
        A_log,
        dt_bias,
        checkpoint_state=state_pool,
        correction_cache=correction_cache,
        kd_cache=kd_cache,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        spec_query_len=width,
        lower_bound=-5.0,
        out=out,
        replay_inputs=None,
        verify_decay=None,
        verify_qk=verify_qk,
        record_metadata=None,
        record_stamps=None,
    )

    num_accepted = torch.full((batch,), accepted, device=device, dtype=torch.int32)
    plan = tuple(torch.empty(batch, device=device, dtype=torch.int32) for _ in range(4))
    vllm_triton_kda_recoverssm_commit(
        packed_q,
        packed_k,
        packed_v,
        checkpoint_state=state_pool,
        correction_cache=correction_cache,
        kd_cache=kd_cache,
        state_indices=state_indices,
        query_start_loc=query_start_loc,
        num_accepted_tokens=num_accepted,
        commit_lens=plan[0],
        final_state_indices=plan[1],
        boundary_state_indices=plan[2],
        boundary_recovery_lens=plan[3],
        state_base_addrs=_device_addresses((state_pool,)),
        correction_cache_base_addrs=_device_addresses((correction_cache,)),
        kd_cache_base_addrs=_device_addresses((kd_cache,)),
        spec_query_len=width,
        record_metadata=None,
        record_stamps=None,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        out.float().reshape_as(expected_out), expected_out, atol=2e-2, rtol=2e-2
    )
    expected_state = initial_state if accepted == 0 else expected_states[accepted - 1]
    torch.testing.assert_close(state_pool[1:], expected_state, atol=3e-2, rtol=2e-2)


def _recover_window(
    qkv: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    indices: torch.Tensor,
    accepted: torch.Tensor,
    width: int,
    replay_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    verify_decay: torch.Tensor,
    verify_qk: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return output and actual producer records after current-window recovery."""
    batch, heads, dim, _ = state[1:].shape
    q, k, v = (
        value.reshape(1, batch * width, heads, dim) for value in qkv.chunk(3, dim=-1)
    )
    gates = gate.reshape_as(q)
    betas = beta.reshape(1, batch * width, heads)
    u = torch.empty(batch + 1, heads, width, dim, device=q.device)
    kd = torch.empty(batch + 1, heads, width, 2 * dim, device=q.device)
    starts = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * width
    out = torch.empty_like(v)
    vllm_triton_kda_recoverssm_verify(
        q,
        k,
        v,
        gates,
        betas,
        A_log,
        dt_bias,
        checkpoint_state=state,
        correction_cache=u,
        kd_cache=kd,
        query_start_loc=starts,
        state_indices=indices,
        spec_query_len=width,
        lower_bound=-5.0,
        out=out,
        replay_inputs=replay_inputs,
        verify_decay=verify_decay,
        verify_qk=verify_qk,
        record_metadata=None,
        record_stamps=None,
    )
    plan = tuple(torch.empty_like(indices) for _ in range(4))
    vllm_triton_kda_recoverssm_commit(
        q,
        k,
        v,
        checkpoint_state=state,
        correction_cache=u,
        kd_cache=kd,
        state_indices=indices,
        query_start_loc=starts,
        num_accepted_tokens=accepted,
        commit_lens=plan[0],
        final_state_indices=plan[1],
        boundary_state_indices=plan[2],
        boundary_recovery_lens=plan[3],
        state_base_addrs=_device_addresses((state,)),
        correction_cache_base_addrs=_device_addresses((u,)),
        kd_cache_base_addrs=_device_addresses((kd,)),
        spec_query_len=width,
        record_metadata=None,
        record_stamps=None,
    )
    return out, u, kd


@pytest.mark.parametrize(
    "batch,width,capacity",
    [(4, width, capacity) for width in (1, 4) for capacity in (8, 16, 32)]
    + [(batch, 4, 8) for batch in (1, 2, 3, 8, 16, 32)],
)
def test_recoverssm_matches_main_producers_over_rounds(
    batch: int, width: int, capacity: int
) -> None:
    """Compare against main's full-precision replay, not just rounded verify."""
    from tokenspeed_kernel.ops.attention.kda._triton.recurrent import (
        batched_kda_commit_conv_window_kernel,
        fused_kda_verify_conv_update,
        fused_recurrent_kda_verify_megafuse,
    )

    torch.manual_seed(781)
    heads, dim = 12, 128
    channels, rows = heads * dim, batch * width
    device = "cuda"
    raw = torch.randn(rows, 3 * channels, device=device, dtype=torch.bfloat16)
    conv_w = torch.randn(3 * channels, 4, device=device, dtype=torch.bfloat16) * 0.3
    conv = torch.randn(batch + 1, 3 * channels, 3, device=device, dtype=torch.bfloat16)
    candidate_conv_state = conv.clone()
    captured_raw = torch.empty_like(raw)
    fa = torch.randn(rows, dim, device=device, dtype=torch.bfloat16)
    fb = torch.randn(channels, dim, device=device, dtype=torch.bfloat16) * 0.05
    beta = torch.randn(rows, heads, device=device, dtype=torch.bfloat16)
    A = torch.randn(heads, device=device) * 0.1
    bias = torch.randn(heads, dim, device=device) * 0.1
    main_state = torch.randn(batch + 1, heads, dim, dim, device=device) * 0.02
    candidate_state = main_state.clone()
    checkpoint = candidate_state.clone()
    # The actual producer and recovery address the same stamped ring directly.
    # Batch rows intentionally differ from physical record slots.
    history_u = torch.full((batch, heads, capacity, dim), float("nan"), device=device)
    history_kd = torch.full(
        (batch, heads, capacity, 2 * dim), float("nan"), device=device
    )
    stamps = torch.full((batch, capacity, 2), -1, dtype=torch.int64, device=device)
    slots = [(request + 1) % batch for request in range(batch)]
    generations = [1] * batch
    checkpoint_positions = [
        capacity - 1 - request % (capacity // 2) for request in range(batch)
    ]
    endpoints = checkpoint_positions.copy()
    resets = [0] * batch
    wraps = [0] * batch
    gate_scratch = torch.empty(rows, channels, device=device)
    indices = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
    groups = torch.zeros(1, device=device, dtype=torch.int32)
    descriptors = _device_addresses(
        (raw, conv_w, conv, fa, fb, beta, A, bias, main_state, gate_scratch)
    ).reshape(1, 10)
    candidate_descriptors = descriptors.clone()
    candidate_descriptors[:, 0] = captured_raw.data_ptr()
    candidate_descriptors[:, 2] = candidate_conv_state.data_ptr()
    scratch_indices = torch.arange(rows, device=device, dtype=torch.int32).view(
        batch, width
    )
    largest_output_error = 0.0
    largest_state_error = 0.0
    squared_state_error = 0.0
    state_elements = 0
    for step in range(128):
        raw.normal_()
        fa.normal_()
        beta.normal_()
        accepted_cpu = [(request + step) % (width + 1) for request in range(batch)]
        accepted = torch.tensor(accepted_cpu, device=device, dtype=torch.int32)
        if step and step % 43 == 0:
            # Reassign the physical slots to new requests. Keep stale storage;
            # new generations and positions must exclude the previous owners.
            slots = [(slot + 1) % batch for slot in slots]
            seed = torch.randn_like(main_state) * 0.02
            main_state.copy_(seed)
            candidate_state.copy_(seed)
            checkpoint.copy_(seed)
            conv.normal_()
            candidate_conv_state.copy_(conv)
            checkpoint_positions = [
                capacity - 1 - request % (capacity // 2) for request in range(batch)
            ]
            endpoints = checkpoint_positions.copy()
            generations = [generation + 1 for generation in generations]
        for request, count in enumerate(accepted_cpu):
            retained = endpoints[request] - checkpoint_positions[request]
            if retained + width > capacity or step % 17 == 0:
                checkpoint[request + 1].copy_(candidate_state[request + 1])
                checkpoint_positions[request] = endpoints[request]
                generations[request] += 1
                resets[request] += 1
        conv_qkv = fused_kda_verify_conv_update(
            raw,
            conv_w,
            conv,
            indices,
            num_heads=heads,
            head_dim=dim,
            draft_token_num=width,
            out=None,
            block_c=256,
            num_warps=4,
        )
        raw_gate = torch.nn.functional.linear(fa, fb)
        expected_out = fused_recurrent_kda_verify_megafuse(
            raw,
            conv_w,
            conv,
            conv,
            fa,
            fb,
            beta,
            A,
            bias,
            main_state,
            main_state,
            indices,
            scratch_indices,
            num_heads=heads,
            head_dim=dim,
            draft_token_num=width,
            scale=dim**-0.5,
            lower_bound=-5.0,
            store_states=False,
            bv=None,
            g_raw=raw_gate,
            conv_qkv=conv_qkv,
            num_warps=None,
            num_stages=None,
            enable_pdl=False,
        )
        candidate_conv, replay_key, replay_value, verify_qk = prepare_recoverssm_conv(
            raw,
            captured_raw,
            conv_w,
            candidate_conv_state,
            indices,
            num_heads=heads,
            head_dim=dim,
            width=width,
        )
        candidate_gate, replay_gate, verify_decay = prepare_recoverssm_gate(
            fa,
            fb,
            A,
            bias,
            num_heads=heads,
            num_layers=1,
            head_dim=dim,
            lower_bound=-5.0,
        )
        record_inputs = (replay_key, replay_value, replay_gate)
        torch.testing.assert_close(candidate_conv, conv_qkv, atol=0, rtol=0)
        q, k, v = (
            part.reshape(1, rows, heads, dim)
            for part in candidate_conv.chunk(3, dim=-1)
        )
        record_metadata = torch.tensor(
            list(zip(slots, generations, endpoints)), dtype=torch.int64, device=device
        )
        out = torch.empty_like(v)
        vllm_triton_kda_recoverssm_verify(
            q,
            k,
            v,
            candidate_gate.reshape_as(q),
            beta.reshape(1, rows, heads),
            A,
            bias,
            checkpoint_state=candidate_state,
            correction_cache=history_u,
            kd_cache=history_kd,
            query_start_loc=torch.arange(batch + 1, device=device, dtype=torch.int32)
            * width,
            state_indices=indices,
            spec_query_len=width,
            lower_bound=-5.0,
            out=out,
            replay_inputs=record_inputs,
            verify_decay=verify_decay,
            verify_qk=verify_qk,
            record_metadata=record_metadata,
            record_stamps=stamps,
        )
        batched_recurrent_kda_replay_commit(
            descriptors,
            groups,
            indices[None],
            indices[None],
            accepted,
            draft_token_num=width,
            num_heads=heads,
            head_dim=dim,
            f_a_dim=dim,
            qkv_stride=raw.stride(0),
            conv_stride=conv.stride(0),
            f_a_stride=fa.stride(0),
            beta_stride=beta.stride(0),
            state_stride=main_state.stride(0),
            gate_stride=gate_scratch.stride(0),
            conv_width=4,
            lower_bound=-5.0,
        )
        batched_kda_commit_conv_window_kernel[(1, batch, (3 * channels + 127) // 128)](
            candidate_descriptors,
            groups,
            indices[None],
            indices[None],
            accepted,
            batch,
            T=width,
            STRIDE_QKV=captured_raw.stride(0),
            STRIDE_CONV=candidate_conv_state.stride(0),
            CONV_DIM=3 * channels,
            BLOCK=128,
            num_warps=1,
        )
        torch.testing.assert_close(candidate_conv_state, conv, atol=0, rtol=0)
        for request, count in enumerate(accepted_cpu):
            slot = slots[request]
            # Rejected payloads remain stamped candidates but must never enter
            # the accepted interval. NaNs expose any accidental recovery read.
            for token in range(count, width):
                row = (endpoints[request] + token) % capacity
                history_u[slot, :, row].fill_(float("nan"))
                history_kd[slot, :, row].fill_(float("nan"))
            for token in range(count):
                position = endpoints[request] + token
                row = position % capacity
                torch.testing.assert_close(
                    stamps[slot, row],
                    torch.tensor(
                        [position, generations[request]],
                        device=device,
                        dtype=torch.int64,
                    ),
                    atol=0,
                    rtol=0,
                )
                wraps[request] += int(row == 0)
            endpoints[request] += count
        retained_cpu = [
            endpoints[request] - checkpoint_positions[request]
            for request in range(batch)
        ]
        recovery_metadata = torch.tensor(
            list(zip(slots, generations, checkpoint_positions)),
            dtype=torch.int64,
            device=device,
        )
        reconstructed = checkpoint.clone()
        plan = tuple(torch.empty_like(indices) for _ in range(4))
        vllm_triton_kda_recoverssm_commit(
            candidate_conv,
            candidate_conv,
            candidate_conv,
            checkpoint_state=reconstructed,
            correction_cache=history_u,
            kd_cache=history_kd,
            state_indices=indices,
            query_start_loc=torch.arange(batch + 1, device=device, dtype=torch.int32)
            * capacity,
            num_accepted_tokens=torch.tensor(
                retained_cpu, device=device, dtype=torch.int32
            ),
            commit_lens=plan[0],
            final_state_indices=plan[1],
            boundary_state_indices=plan[2],
            boundary_recovery_lens=plan[3],
            state_base_addrs=_device_addresses((reconstructed,)),
            correction_cache_base_addrs=_device_addresses((history_u,)),
            kd_cache_base_addrs=_device_addresses((history_kd,)),
            spec_query_len=capacity,
            record_metadata=recovery_metadata,
            record_stamps=stamps,
        )
        # Feed the reconstructed endpoint to the next actual verify producer;
        # neither the main oracle nor current-window recovery seeds that round.
        candidate_state.copy_(reconstructed)
        largest_output_error = max(
            largest_output_error,
            (out.reshape_as(expected_out).float() - expected_out.float())
            .abs()
            .max()
            .item(),
        )
        largest_state_error = max(
            largest_state_error, (candidate_state - main_state).abs().max().item()
        )
        # Preserve verify rounding at concurrent batches as well as single
        # requests; small BF16 changes can alter speculative acceptance.
        torch.testing.assert_close(
            out.reshape_as(expected_out), expected_out, atol=0, rtol=0
        )
        torch.testing.assert_close(candidate_state, main_state, atol=0, rtol=0)
        squared_state_error += (candidate_state - main_state).square().sum().item()
        state_elements += main_state.numel()
    assert all(count > 1 for count in resets)
    assert all(count > 1 for count in wraps)
    print(
        f"B={batch} T={width} R={capacity}: max output error={largest_output_error:.8g}, max state error={largest_state_error:.8g}, state RMS={(squared_state_error / state_elements)**0.5:.8g}"
    )


def test_record_chain_does_not_read_rounded_verify_inputs() -> None:
    """Perturb verify inputs without changing the recovery record producers."""
    torch.manual_seed(173)
    batch, width, heads, dim = 4, 4, 12, 128
    shape = (1, batch * width, heads, dim)
    q, k, v, raw_gate = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)
    )
    beta = torch.randn(1, batch * width, heads, device="cuda", dtype=torch.bfloat16)
    A = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads, dim, device="cuda")
    normalized_k = k.float() / torch.sqrt(
        k.float().square().sum(-1, keepdim=True) + 1e-6
    )
    replay_inputs = (normalized_k, v.float(), torch.rand(shape, device="cuda"))
    state = torch.randn(batch + 1, heads, dim, dim, device="cuda") * 0.01
    indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * width
    results = []
    for perturb in (False, True):
        u = torch.empty(batch, heads, width, dim, device="cuda")
        kd = torch.empty(batch, heads, width, 2 * dim, device="cuda")
        out = torch.empty_like(v)
        vllm_triton_kda_recoverssm_verify(
            q + 1 if perturb else q,
            k * 2 if perturb else k,
            v - 1 if perturb else v,
            raw_gate + 1 if perturb else raw_gate,
            beta,
            A,
            bias,
            checkpoint_state=state,
            correction_cache=u,
            kd_cache=kd,
            query_start_loc=starts,
            state_indices=indices,
            spec_query_len=width,
            lower_bound=-5.0,
            out=out,
            replay_inputs=replay_inputs,
            verify_decay=None,
            verify_qk=None,
            record_metadata=None,
            record_stamps=None,
        )
        results.append((out, u, kd))
    assert not torch.equal(results[0][0], results[1][0])
    torch.testing.assert_close(results[0][1], results[1][1], atol=0, rtol=0)
    torch.testing.assert_close(results[0][2], results[1][2], atol=0, rtol=0)


@pytest.mark.parametrize("separate_records", [False, True])
def test_recoverssm_ragged_prefix_and_null_rows(separate_records: bool) -> None:
    """Padded suffix state is unobservable; only valid prefix records change."""
    torch.manual_seed(1209)
    lengths = [4, 2, 0, 1, 3, 4]
    batch, heads, dim, width = len(lengths), 12, 128, 4
    starts_cpu = [0]
    for length in lengths:
        starts_cpu.append(starts_cpu[-1] + length)
    rows = starts_cpu[-1]
    shape = (1, rows, heads, dim)
    q, k, v, g = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)
    )
    beta = torch.randn(1, rows, heads, device="cuda", dtype=torch.bfloat16)
    A = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(heads, dim, device="cuda") * 0.1
    state = torch.randn(batch + 1, heads, dim, dim, device="cuda") * 0.02
    initial = state.clone()
    indices = torch.tensor([1, 2, 3, 4, 5, 0], device="cuda", dtype=torch.int32)
    starts = torch.tensor(starts_cpu, device="cuda", dtype=torch.int32)
    u = torch.full((batch, heads, width, dim), float("nan"), device="cuda")
    kd = torch.full((batch, heads, width, 2 * dim), float("nan"), device="cuda")
    out = torch.full_like(v, float("nan"))
    metadata = torch.zeros(batch, 3, dtype=torch.int64, device="cuda")
    metadata[:, 0] = torch.arange(batch, device="cuda")
    metadata[:, 1] = 1
    stamps = torch.full((batch, width, 2), -1, dtype=torch.int64, device="cuda")
    if separate_records:
        record_k = k.float() / torch.sqrt(
            k.float().square().sum(-1, keepdim=True) + 1e-6
        )
        record_v = v.float() * 1.25
        record_d = torch.rand(shape, device="cuda")
        replay_inputs = (record_k, record_v, record_d)
        record_v = torch.nn.functional.silu(record_v)
    else:
        record_k = k.float() * torch.rsqrt(
            k.float().square().sum(-1, keepdim=True) + 1e-6
        )
        record_v = v.float()
        record_d = torch.exp(
            -5.0 * torch.sigmoid(A.exp()[None, None, :, None] * (g.float() + bias))
        )
        replay_inputs = None
    vllm_triton_kda_recoverssm_verify(
        q,
        k,
        v,
        g,
        beta,
        A,
        bias,
        checkpoint_state=state,
        correction_cache=u,
        kd_cache=kd,
        query_start_loc=starts,
        state_indices=indices,
        spec_query_len=width,
        lower_bound=-5.0,
        out=out,
        replay_inputs=replay_inputs,
        verify_decay=None,
        verify_qk=None,
        record_metadata=metadata,
        record_stamps=stamps,
    )
    torch.testing.assert_close(state, initial, atol=0, rtol=0)
    for request, length in enumerate(lengths):
        begin, end = starts_cpu[request : request + 2]
        if request == batch - 1:
            torch.testing.assert_close(
                out[:, begin:end], torch.zeros_like(out[:, begin:end]), atol=0, rtol=0
            )
            valid = 0
        else:
            valid = length
            if length:
                expected_out, _ = _reference(
                    q[:, begin:end],
                    k[:, begin:end],
                    v[:, begin:end],
                    g[:, begin:end],
                    beta[:, begin:end],
                    A,
                    bias,
                    initial[request + 1 : request + 2],
                    -5.0,
                )
                torch.testing.assert_close(
                    out[:, begin:end].float(), expected_out, atol=2e-2, rtol=2e-2
                )
                current = initial[request + 1].clone()
                for offset in range(length):
                    token = begin + offset
                    key = record_k[0, token]
                    decay = record_d[0, token]
                    current = current * decay[:, None, :]
                    correction = (
                        record_v[0, token] - (current * key[:, None, :]).sum(-1)
                    ) * beta[0, token].float().sigmoid()[:, None]
                    torch.testing.assert_close(
                        u[request, :, offset], correction, atol=1e-5, rtol=1e-3
                    )
                    torch.testing.assert_close(
                        kd[request, :, offset, :dim], key, atol=1e-6, rtol=1e-5
                    )
                    torch.testing.assert_close(
                        kd[request, :, offset, dim:], decay, atol=1e-6, rtol=1e-5
                    )
                    current = current + correction[:, :, None] * key[:, None, :]
        assert torch.isnan(u[request, :, valid:]).all()
        assert torch.isnan(kd[request, :, valid:]).all()
        assert (stamps[request, valid:] == -1).all()
        if valid:
            torch.testing.assert_close(
                stamps[request, :valid, 0],
                torch.arange(valid, device="cuda", dtype=torch.int64),
                atol=0,
                rtol=0,
            )
            assert (stamps[request, :valid, 1] == 1).all()


@pytest.mark.parametrize("captured", [False, True])
def test_batched_recoverssm_commit_groups_alias_and_padding(captured: bool) -> None:
    """Exercise the production two-kernel commit, not the single-layer adapter."""
    from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
        batched_recoverssm_commit,
    )

    torch.manual_seed(982)
    layers, batch, width, heads, dim, pages = 3, 6, 4, 2, 128, 12
    channels = heads * dim
    groups_cpu = [1, 0, 1]
    read_cpu = [[1, 2, 3, 4, 5, -1], [5, 4, 3, 2, 1, -1]]
    write_cpu = [[6, 2, 7, 4, 8, -1], [6, 4, 7, 2, 8, -1]]
    accepted_cpu = [0, 1, 2, 3, 4, 0]
    groups = torch.tensor(groups_cpu, device="cuda", dtype=torch.int32)
    reads = torch.tensor(read_cpu, device="cuda", dtype=torch.int32)
    writes = torch.tensor(write_cpu, device="cuda", dtype=torch.int32)
    accepted = torch.tensor(accepted_cpu, device="cuda", dtype=torch.int32)
    states = torch.randn(layers, pages, heads, dim, dim, device="cuda") * 0.02
    conv = torch.randn(
        layers, pages, 3 * channels, 3, device="cuda", dtype=torch.bfloat16
    )
    # Padded token rows catch accidental use of packed strides.
    raw = torch.randn(
        layers,
        batch * width,
        3 * channels + 16,
        device="cuda",
        dtype=torch.bfloat16,
    )
    corrections = (
        torch.randn(layers, batch * width, channels + 16, device="cuda") * 0.01
    )
    kd = torch.randn(layers, batch * width, 2 * channels + 16, device="cuda") * 0.02
    kd[:, :, : 2 * channels].view(layers, batch * width, heads, 2, dim)[
        :, :, :, 1
    ].sigmoid_()
    metadata = torch.tensor(
        [[request, 7, 0] for request in range(batch)], device="cuda", dtype=torch.int64
    )
    stamps = torch.empty(layers, batch, width, 2, device="cuda", dtype=torch.int64)
    stamps[..., 0] = torch.arange(width, device="cuda")
    stamps[..., 1] = 7
    stamps[:, -1].fill_(-1)
    seed_state, seed_conv = states.clone(), conv.clone()
    expected_state, expected_conv = states.clone(), conv.clone()
    descriptors = torch.zeros(layers, 10, device="cuda", dtype=torch.int64)
    for column, storage in (
        (0, raw),
        (2, conv),
        (3, corrections),
        (5, kd),
        (8, states),
    ):
        descriptors[:, column] = _device_addresses(tuple(storage.unbind()))
    for layer, group in enumerate(groups_cpu):
        for request, count in enumerate(accepted_cpu[:-1]):
            source, dest = read_cpu[group][request], write_cpu[group][request]
            state = seed_state[layer, source].clone()
            history = seed_conv[layer, source].clone()
            for token in range(count):
                row = request * width + token
                key, decay = (
                    kd[layer, row, : 2 * channels].view(heads, 2, dim).unbind(1)
                )
                correction = corrections[layer, row, :channels].view(heads, dim)
                state = state * decay[:, None, :]
                state = state + correction[:, :, None] * key[:, None, :]
                history = torch.cat(
                    (history[:, 1:], raw[layer, row, : 3 * channels, None]), dim=1
                )
            expected_state[layer, dest] = state
            expected_conv[layer, dest] = history

    def launch() -> None:
        batched_recoverssm_commit(
            descriptors,
            groups,
            reads,
            writes,
            accepted,
            draft_token_num=width,
            num_heads=heads,
            head_dim=dim,
            qkv_stride=raw.stride(1),
            conv_stride=conv.stride(1),
            correction_stride=corrections.stride(1),
            key_decay_stride=kd.stride(1),
            state_stride=states.stride(1),
            conv_width=4,
            record_metadata=metadata,
            record_stamps=stamps,
        )

    launch()
    graph = None
    if captured:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
    # Replay twice from the same seed. Refresh descriptor addresses between
    # runs to catch pointers cached outside the persistent table.
    for _ in range(2):
        states.copy_(seed_state)
        conv.copy_(seed_conv)
        if graph is None:
            launch()
        else:
            graph.replay()
        torch.testing.assert_close(states, expected_state, atol=1e-5, rtol=1e-3)
        torch.testing.assert_close(conv, expected_conv, atol=0, rtol=0)
        states = torch.empty_like(states)
        conv = torch.empty_like(conv)
        descriptors[:, 8] = _device_addresses(tuple(states.unbind()))
        descriptors[:, 2] = _device_addresses(tuple(conv.unbind()))


@pytest.mark.parametrize("batch", [1, 7, 33])
def test_commit_state_groups_strides_boundaries_and_graph(batch: int) -> None:
    from tokenspeed_kernel.ops.attention.kda.triton import commit_state_groups

    tables = [
        torch.arange(batch * (columns + 3), device="cuda", dtype=torch.int32).view(
            batch, columns + 3
        )[:, :columns]
        + group * 1000
        for group, columns in enumerate((1, 3, 5))
    ]
    # Retain non-contiguous row strides in the actual source tables.
    tables = [torch.nn.functional.pad(table, (0, 2))[:, :-2] for table in tables]
    sources = [
        torch.arange(batch, device="cuda", dtype=torch.int32) + group * 100
        for group in range(3)
    ]
    descriptors = torch.tensor(
        [
            (
                (table.data_ptr() - tables[0].data_ptr()) // tables[0].element_size(),
                table.stride(0),
                table.shape[1],
                source.data_ptr(),
            )
            for table, source in zip(tables, sources, strict=True)
        ],
        dtype=torch.int64,
        device="cuda",
    )
    committed = torch.arange(batch, device="cuda", dtype=torch.int64) * 127 + 126
    accepted = torch.arange(batch, device="cuda", dtype=torch.int32) % 6
    pages = torch.empty((3, batch), device="cuda", dtype=torch.int32)
    reads = torch.empty_like(pages)
    steps = torch.empty(batch, device="cuda", dtype=torch.int32)

    def launch() -> None:
        commit_state_groups(
            accepted,
            committed,
            tables[0],
            descriptors,
            batch_size=batch,
            draft_tokens=4,
            granularity=128,
            pages_out=pages,
            reads_out=reads,
            steps_out=steps,
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for iteration in range(2):
        if iteration:
            accepted.add_(1)
            for source in sources:
                source.add_(7)
            committed.add_(128)
            tables[1].add_(23)
        graph.replay()
        torch.testing.assert_close(reads, torch.stack(sources), atol=0, rtol=0)
        expected_steps = accepted.clamp(1, 4)
        torch.testing.assert_close(steps, expected_steps, atol=0, rtol=0)
        for group, table in enumerate(tables):
            slots = ((committed + expected_steps - 1) // 128).clamp(
                0, table.shape[1] - 1
            )
            expected = table.gather(1, slots[:, None]).squeeze(1)
            torch.testing.assert_close(pages[group], expected, atol=0, rtol=0)


@pytest.mark.parametrize("rows", [4, 8, 12, 16, 32, 64, 128])
@pytest.mark.parametrize("layers", [1, 69])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.parametrize("seed", [0, 1573])
def test_recoverssm_gate_matches_all_layer_tiling(
    rows: int, layers: int, lower_bound: float | None, seed: int
) -> None:
    """Replay decay matches main while raw gates preserve BF16 rounding."""
    from kda_replay_reference import (
        batched_kda_gate_precompute_dot_kernel,
        batched_kda_gate_precompute_kernel,
    )
    from tokenspeed_kernel._triton import tl, triton
    from tokenspeed_kernel.ops.attention.kda._triton.recurrent import (
        _gate_tiling,
        _gate_tiling_dot,
        kda_gate_precompute_dot_kernel,
        kda_gate_precompute_kernel,
    )

    @triton.jit
    def decay_reference(log_gate, decay, N: tl.constexpr, BLOCK: tl.constexpr):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(log_gate + offset, mask=offset < N, other=0)
        tl.store(decay + offset, tl.exp(value), mask=offset < N)

    torch.manual_seed(seed)
    heads, dim = 12, 128
    channels = heads * dim
    fa = torch.randn(rows, dim, device="cuda", dtype=torch.bfloat16)
    fb = torch.randn(channels, dim, device="cuda", dtype=torch.bfloat16) * 0.05
    A = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(channels, device="cuda") * 0.1
    logs = torch.empty(layers, rows, channels, device="cuda")
    addresses = torch.zeros(layers, 10, device="cuda", dtype=torch.int64)
    addresses[:, 3] = fa.data_ptr()
    addresses[:, 4] = fb.data_ptr()
    addresses[:, 6] = A.data_ptr()
    addresses[:, 7] = bias.data_ptr()
    addresses[:, 9] = _device_addresses(tuple(logs.unbind()))
    bt, bk = (
        _gate_tiling_dot(rows, dim)
        if rows >= 16
        else _gate_tiling(rows, heads, dim, fa.device, layers=layers)
    )
    if lower_bound is None:
        # The unchanged batched oracle supports the bounded gate only.
        kernel = (
            kda_gate_precompute_dot_kernel if rows >= 16 else kda_gate_precompute_kernel
        )
        kernel[(heads, triton.cdiv(rows, bt), triton.cdiv(dim, bk))](
            fa,
            fb,
            A,
            bias,
            logs[0],
            lower_bound,
            fa.stride(0),
            channels,
            rows,
            K=dim,
            D_FA=dim,
            BK=bk,
            BT=bt,
            num_warps=1 if bt >= 4 else 2,
        )
    else:
        kernel = (
            batched_kda_gate_precompute_dot_kernel
            if rows >= 16
            else batched_kda_gate_precompute_kernel
        )
        kernel[(layers * heads, triton.cdiv(rows, bt), triton.cdiv(dim, bk))](
            addresses,
            rows,
            stride_fa=fa.stride(0),
            stride_gate=channels,
            lower_bound=lower_bound,
            HV=heads,
            K=dim,
            D_FA=dim,
            BK=bk,
            BT=bt,
            num_warps=1 if bt >= 4 else 2,
        )
    expected = torch.empty_like(logs[0])
    decay_reference[(triton.cdiv(expected.numel(), 256),)](
        logs[0], expected, N=expected.numel(), BLOCK=256
    )
    raw, actual, _ = prepare_recoverssm_gate(
        fa,
        fb,
        A,
        bias,
        num_heads=heads,
        num_layers=layers,
        head_dim=dim,
        lower_bound=lower_bound,
    )
    torch.testing.assert_close(raw, torch.nn.functional.linear(fa, fb), atol=0, rtol=0)
    torch.testing.assert_close(actual.reshape_as(expected), expected, atol=0, rtol=0)
