from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.moe import moe_apply as kernel_moe_apply
from tokenspeed_kernel.ops.moe import moe_plan as kernel_moe_plan


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("activation", ["silu", "situ"])
@pytest.mark.parametrize("num_tokens", [4, 48, 80])
@pytest.mark.parametrize("intermediate", [32, 128])
def test_triton_bf16_moe_matches_torch(
    activation: str, num_tokens: int, intermediate: int
) -> None:
    """Four tokens run 16-row tiles; with the deep pipeline 48 tokens spread one
    expert's 24 routes over two 16-row tiles and 80 (40 per expert) take 64-row
    tiles, while a 32-wide intermediate keeps the shallow token-count rule."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(
        num_tokens, 128, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    w13 = (
        torch.randn(
            4,
            2 * intermediate,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.02
    )
    w2 = (
        torch.randn(
            4,
            128,
            intermediate,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.02
    )
    first = torch.arange(num_tokens, device="cuda") % 4
    topk_ids = torch.stack((first, (first + 1) % 4), dim=1).to(torch.int32)
    topk_weights = torch.rand(
        (num_tokens, 2), device="cuda", dtype=torch.float32, generator=generator
    )

    weights = torch.nn.Module()
    weights.w13_weight = w13
    weights.w2_weight = w2
    weights.top_k = 2
    weights.activation_situ_beta = 4.0
    weights.activation_situ_linear_beta = 25.0
    plan = kernel_moe_plan(
        "unquant",
        input_dtype=torch.bfloat16,
        activation=activation,
        routing_mode="precomputed_topk",
        ispp=32,
        solution="triton",
        hidden=None,
        swiglu_form=("standard" if activation == "swiglu" else None),
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=True,
        combine_order="rank",
    )
    actual = kernel_moe_apply(
        plan,
        x,
        weights,
        torch.empty((num_tokens, 4), device="cuda"),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    expected = torch.zeros((num_tokens, 128), device="cuda", dtype=torch.float32)
    for expert_id in range(4):
        token_ids, slots = torch.where(topk_ids == expert_id)
        gate_up = F.linear(x[token_ids].float(), w13[expert_id].float())
        gate, up = gate_up.chunk(2, dim=-1)
        if activation == "situ":
            gate = gate.to(torch.bfloat16).float()
            up = up.to(torch.bfloat16).float()
            gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
            up = 25.0 * torch.tanh(up / 25.0)
            intermediate = (gate * up).to(torch.bfloat16)
        else:
            intermediate = (F.silu(gate) * up).to(torch.bfloat16)
        expert_output = F.linear(intermediate.float(), w2[expert_id].float()).to(
            torch.bfloat16
        )
        expected.index_add_(
            0,
            token_ids,
            expert_output.float() * topk_weights[token_ids, slots, None],
        )

    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.01)
