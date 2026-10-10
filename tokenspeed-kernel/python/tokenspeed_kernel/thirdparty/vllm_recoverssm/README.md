# vLLM RecoverSSM kernels

This directory contains the Kimi-K3 RecoverSSM implementation copied from
`vllm-project/vllm` commit `c77b3faa1cc2c39edbe82365f28ce96981af1438`.

Upstream file:
`vllm/models/kimi_k3/nvidia/ops/recoverssm.py`.

The source retains the Apache-2.0 license. TokenSpeed adapts the Triton import,
null-state convention and convolution layout helper to remove the vLLM runtime
dependency. The unused vLLM runtime context is omitted.

The producer stores FP32 normalized keys and token-local per-channel decay,
along with FP32 corrections. Recovery consumes these records directly. It does
not repeat key normalization or gate evaluation. State updates remain ordered
FP32 CUDA-core operations. The registered adapter lives under ops/attention/kda.

Main verify consumes BF16 convolution and gate outputs. Main accepted-state
replay instead uses FP32 convolution and gates. The adapted producer therefore
keeps an independent FP32 record recurrence alongside the verify recurrence.
It does not obtain recovery corrections by casting rounded verify inputs.

Convolution prepares verify Q/K after BF16 rounding, then normalizes in FP32.
The gate producer separately prepares verify and replay decay. Independent
CTAs consume these values in one launch, so each CTA holds one state tile.
This preserves the two precision chains and avoids repeated transforms for
each value tile.

Verify and replay use explicit Triton Gluon layouts that retain main's
respective reduction trees. Small-batch verify preserves cross-warp Q/K
normalization. Replay preserves the fused local projection pairs and computes
correction with a fused SiLU multiplication and subtraction. The convolution
producer passes pre-SiLU FP32 V to avoid an extra rounding boundary. The BF16
verify gate uses main's linear projection. Replay gate tiling includes the
full recurrent-layer count. Token updates remain ordered FP32 CUDA-core
operations. Four-warp verify tiles also preserve main's local projection
FMA operands for both value rows per warp, and keep the token loop intact.
One-warp tiles retain their existing expansion. Compositional tests require
exact accepted state and exact verify output below batch four. The existing
verify limits for other batches and scalar-reference limits remain unchanged. Full-model equality
and performance are separate acceptance gates.

The launch uses four warps below batch four and one warp otherwise, with
batch-bucketed value tiles. Fixed model-layout strides specialize address
arithmetic; request-dependent group pitches remain runtime values. Exact
request counts do not create new binaries within the same launch bucket.

Invalid tokens form a suffix in each request. Output and record stores mask
that suffix. Verify does not write the input checkpoint, so it need not retain
the intermediate state after the last valid token. Tests cover ragged lengths,
empty requests, null pages and untouched record suffixes.

Record storage is slot-indexed, with an explicit generation and absolute
position stamp. Producer and recovery map absolute positions modulo ring
capacity. Recovery rejects stale records before reading their payloads.
PR1 assigns one slot per batch row for each round; the composition tests also
exercise slot reassignment and longer rings. Batched recovery accepts the existing
group-major state plan and strided layer pointer descriptors. Convolution
commit uses the existing TokenSpeed kernel; unused vLLM convolution helpers
are omitted. The NVIDIA runtime uses this adapter on the development branch.
Accuracy and performance validation are not complete. See
[PR1 progress](../../../../../docs/design/kda-recoverssm-pr1-progress.md).


## Adapter contracts

Recovery consumes saved FP32 K/U/decay. It does not evaluate gates again.
The standalone commit API therefore has no gate weights, gate bound or
per-pool stride-array arguments. Its tensor views provide the strides.
The shared runtime adapter keeps producer-only fields for the other backends;
its docstring identifies the fields that this record-based commit does not use.

The production commit uses a contiguous ten-column pointer table and
contiguous group-major page metadata. The shared state-page planner clamps
accepted counts before this call. Its page-table descriptors hold offsets
relative to the current packed table base. A fresh scheduler upload therefore
changes the kernel argument without uploading the same table geometry again.
Live source and destination pages must be positive. Padding uses negative pages
and zero accepted counts. Each request
owns its writable pages; its own source and destination may alias.

Producer wrappers check shapes, dtypes and strides on the host. These checks
do not read device values. Convolution output is dense even when the input is
a strided projection view. No padding memset is added: graph consumers must
ignore padded outputs under the unified-path contract.

The old all-layer replay kernel and its gate producers now live in
`tokenspeed-kernel/test/kda_replay_reference.py` as the independent main oracle.
They are excluded from the runtime package. Shared producers and per-layer
kernels still used by other paths remain in place. The serving path has no
legacy replay selector. Final validation and design-document cleanup remain
pending.
