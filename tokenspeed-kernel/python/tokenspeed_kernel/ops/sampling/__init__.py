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

"""Sampling kernel entry points."""

from __future__ import annotations

# ``tokenspeed_kernel.ops.sampling.torch`` (the portable leaves) is imported
# below for registration and would rebind the bare name ``torch`` in this
# package namespace, hence the alias.
import torch as _torch
from tokenspeed_kernel.ops.sampling.logprob import vocab_parallel_logprobs
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "argmax",
    "dist_argmax_available",
    "distributed_argmax",
    "supports_dist_argmax_shape",
    "try_create_dist_argmax_state",
    "vocab_parallel_logprobs",
]

_SUPPORTED_DTYPES = (_torch.float16, _torch.bfloat16, _torch.float32)
_SUPPORTED_OUT_DTYPES = (_torch.int32, _torch.int64)


def _validate_argmax_out(logits: _torch.Tensor, out: _torch.Tensor) -> None:
    if out.shape != (logits.shape[0],):
        raise ValueError(
            f"out must have shape (M,)={(logits.shape[0],)}, got {tuple(out.shape)}"
        )
    if out.dtype not in _SUPPORTED_OUT_DTYPES:
        raise ValueError(f"out must be int32 or int64; got {out.dtype}")
    if out.device != logits.device:
        raise ValueError("out must be on the same device as logits")


def _argmax_torch_fallback(
    logits: _torch.Tensor,
    *,
    out: _torch.Tensor | None = None,
) -> _torch.Tensor:
    if out is not None:
        _validate_argmax_out(logits, out)
    result = _torch.argmax(logits, dim=-1)
    if out is not None:
        out.copy_(result)
        return out
    return result


def argmax(
    logits: _torch.Tensor,
    *,
    out: _torch.Tensor | None = None,
    solution: str | None = None,
    override: str | None = None,
) -> _torch.Tensor:
    """Return row-wise argmax indices over the last logits dimension.

    Args:
        logits: Input logits with shape ``(M, N)``. The argmax is taken
            over the last dimension.
        out: Optional output buffer with shape ``(M,)`` and dtype int32 or
            int64 on the same device as ``logits``.
        solution: Optional kernel solution to force through normal selection.
        override: Optional exact kernel-name or solution override.

    NaN handling:
        Kernel-backed sampling paths treat NaNs as invalid candidates.
        Rows with at least one non-NaN value return the index of the
        maximum non-NaN value, with ties broken toward the lowest index.
        Rows with no valid non-NaN values return the ``-1`` sentinel.
        Unsupported inputs fall back to ``torch.argmax`` semantics.

    Returns:
        A tensor containing argmax indices for each row of ``logits``.
    """
    if (
        logits.dim() != 2
        or logits.shape[0] == 0
        or not logits.is_cuda
        or logits.dtype not in _SUPPORTED_DTYPES
    ):
        return _argmax_torch_fallback(logits, out=out)

    signature = format_signature(logits=dense_tensor_format(logits.dtype))
    try:
        kernel = select_kernel(
            "sampling",
            "argmax",
            signature,
            solution=solution,
            override=override,
        )
    except NoKernelFoundError:
        return _argmax_torch_fallback(logits, out=out)

    shape_params = {
        "M": logits.shape[0],
        "N": logits.shape[1],
        "has_out": out is not None,
    }
    ShapeCapture.get().record(
        "sampling", "argmax", kernel.name, logits.dtype, shape_params
    )
    with kernel_scope(
        "sampling", "argmax", logits.dtype, kernel_name=kernel.name, **shape_params
    ):
        return kernel(logits, out=out)


def dist_argmax_available() -> bool:
    """Whether this platform can build a distributed-argmax state right now.

    NVIDIA reduces through CuTe DSL and multicast (``cute_dsl``); AMD exchanges
    per-row candidates through the Iris heap (``iris``). The verdict may differ
    between ranks, so callers agree on it across the group before building.
    """
    platform = current_platform()
    if platform.is_nvidia:
        from tokenspeed_kernel.ops.sampling import cute_dsl

        return cute_dsl.is_available()
    if platform.is_amd:
        from tokenspeed_kernel.ops.sampling import iris

        return iris.is_available()
    return False


def supports_dist_argmax_shape(
    vocab_per_rank: int, dtype: _torch.dtype, world_size: int
) -> bool:
    """Whether a shard's width, dtype and group size fit this platform's kernel.

    Follows from the model config and the platform alone, so a caller may
    reject on it before any collective work.
    """
    platform = current_platform()
    if platform.is_nvidia:
        from tokenspeed_kernel.ops.sampling import cute_dsl

        return cute_dsl.supports_dist_argmax_shape(vocab_per_rank, dtype, world_size)
    if platform.is_amd:
        from tokenspeed_kernel.ops.sampling import iris

        return iris.supports_dist_argmax_shape(vocab_per_rank, dtype, world_size)
    return False


def try_create_dist_argmax_state(
    *,
    group,
    rank_in_group: int,
    max_M: int,
    dtype: _torch.dtype,
    device: _torch.device,
    skip_ping_pong: bool,
):
    """Build the group's distributed-argmax state, or None to keep the gather.

    Collective over ``group``. Returns None when the platform lacks the
    transport (no NVLS multicast, no prepared Iris heap).

    Args:
        group: The process group the argmax reduces over.
        rank_in_group: This process's rank within ``group``.
        max_M: Largest row count any call will pass.
        dtype: Value dtype of the logits.
        device: Device of the logits.
        skip_ping_pong: The caller synchronizes the group between calls, so
            the NVIDIA kernel may pin one slot band. The AMD exchange rotates
            its slots by device-side epochs and is safe either way.
    """
    platform = current_platform()
    if platform.is_nvidia:
        from tokenspeed_kernel.ops.sampling import cute_dsl

        return cute_dsl.try_create_dist_argmax_state(
            group=group,
            rank_in_group=rank_in_group,
            max_M=max_M,
            dtype=dtype,
            device=device,
            skip_ping_pong=skip_ping_pong,
        )
    if platform.is_amd:
        from tokenspeed_kernel.ops.sampling import iris

        return iris.try_create_dist_argmax_state(
            group=group,
            rank_in_group=rank_in_group,
            max_M=max_M,
            dtype=dtype,
            device=device,
        )
    return None


def distributed_argmax(state, logits: _torch.Tensor) -> _torch.Tensor:
    """Row-wise argmax across the group's vocab shards.

    Every rank of the state's group calls it in lockstep with its own
    ``[M, vocab_per_rank]`` shard; rank ``r`` holds global columns
    ``[r * vocab_per_rank, (r + 1) * vocab_per_rank)``.

    Returns:
        ``[M]`` int64 global indices, identical on every rank. Ties resolve
        to the lowest index, as in :func:`argmax` over the gathered rows.
    """
    from tokenspeed_kernel.ops.sampling import iris

    if isinstance(state, iris.IrisDistArgmaxState):
        return iris.distributed_argmax(state, logits, out=None)
    from tokenspeed_kernel.ops.sampling import cute_dsl

    _, index = cute_dsl.distributed_argmax(state, logits)
    return index


# Backend registration (side-effect imports).
import tokenspeed_kernel.ops.sampling.cute_dsl  # noqa: E402,F401
import tokenspeed_kernel.ops.sampling.gluon  # noqa: E402,F401
import tokenspeed_kernel.ops.sampling.torch  # noqa: E402,F401
