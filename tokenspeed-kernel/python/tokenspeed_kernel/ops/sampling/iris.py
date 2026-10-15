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

"""Distributed argmax over vocab-sharded logits on AMD, through Iris.

Each rank runs the registered local ``argmax`` over its shard, so a row's
candidate follows the same NaN and tie rules as the full-vocab argmax, then
:class:`IrisArgmaxExchange` reduces the candidates across the group with one
8-byte key per row. The result equals the local ``argmax`` of the gathered
full-vocab rows, index for index, on every rank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import current_platform

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.communication._iris.argmax import IrisArgmaxExchange

__all__ = [
    "IrisDistArgmaxState",
    "distributed_argmax",
    "is_available",
    "supports_dist_argmax_shape",
    "try_create_dist_argmax_state",
]

_MAX_WORLD_SIZE = 8  # Iris addresses at most eight heap bases.


@dataclass
class IrisDistArgmaxState:
    exchange: IrisArgmaxExchange
    world_size: int
    max_M: int
    dtype: torch.dtype
    device: torch.device


def _shared_context():
    from tokenspeed_kernel.ops.communication._iris import existing_context

    return existing_context()


def is_available() -> bool:
    """Whether this process can build the exchange now.

    It needs AMD, the Iris package, and the process-wide Iris heap that the
    group's all-reduce prepares: creating the heap is collective over every
    rank of the job, which a single group must not start on its own.
    """
    if not current_platform().is_amd:
        return False
    try:
        return _shared_context() is not None
    except ImportError:
        return False


def supports_dist_argmax_shape(
    vocab_per_rank: int, dtype: torch.dtype, world_size: int
) -> bool:
    """Whether a shard's width, dtype and group size fit the exchange.

    Args:
        vocab_per_rank: Column count of one rank's logits shard.
        dtype: Value dtype of those logits.
        world_size: Ranks in the group the argmax reduces over.

    Returns:
        True when :func:`distributed_argmax` accepts the shard.
    """
    return (
        dtype in (torch.float16, torch.bfloat16, torch.float32)
        and 2 <= world_size <= _MAX_WORLD_SIZE
        and vocab_per_rank > 0
        and world_size * vocab_per_rank < 2**31
    )


def try_create_dist_argmax_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    max_M: int,
    dtype: torch.dtype,
    device: torch.device,
) -> IrisDistArgmaxState | None:
    """Allocate the exchange on the shared Iris heap, or None without one.

    Every rank of ``group`` calls this together: the allocation must land at
    the same heap offset on each rank, and the call ends with a group barrier.

    Args:
        group: The process group the argmax reduces over.
        rank_in_group: This process's rank within ``group``.
        max_M: Largest row count any call will pass.
        dtype: Value dtype of the logits.
        device: Device of the logits.

    Returns:
        The state, or None when no Iris heap has been prepared.
    """
    from tokenspeed_kernel.ops.communication._iris import IrisGroup
    from tokenspeed_kernel.ops.communication._iris.argmax import IrisArgmaxExchange

    ctx = _shared_context()
    if ctx is None:
        return None
    peers = IrisGroup(ctx, group, rank_in_group, device)
    exchange = IrisArgmaxExchange(peers, max_M)
    peers.finish_prepare()
    return IrisDistArgmaxState(
        exchange=exchange,
        world_size=peers.world_size,
        max_M=max_M,
        dtype=dtype,
        device=device,
    )


def distributed_argmax(
    state: IrisDistArgmaxState,
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Row-wise argmax over the group's concatenated vocab shards.

    Collective: every rank of the group calls it in the same order with the
    same row count, its own ``[M, vocab_per_rank]`` shard (rank ``r`` holding
    columns ``[r * vocab_per_rank, (r + 1) * vocab_per_rank)``).

    Args:
        state: This group's exchange state.
        logits: This rank's shard, rows contiguous.
        out: Optional ``[M]`` int32/int64 destination.

    Returns:
        ``[M]`` global argmax indices, identical on every rank; int64 unless
        ``out`` is given.
    """
    from tokenspeed_kernel.ops.sampling import argmax

    if logits.dtype != state.dtype or logits.device != state.device:
        raise ValueError(
            f"logits {logits.dtype} on {logits.device} do not match the state's "
            f"{state.dtype} on {state.device}"
        )
    if out is None:
        out = torch.empty((logits.shape[0],), dtype=torch.int64, device=logits.device)
    if logits.shape[0] == 0:
        # Every rank sees the same row count, so all skip the exchange together.
        return out
    local_index = argmax(logits)
    state.exchange.run(logits, local_index, out)
    return out
