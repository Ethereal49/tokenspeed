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

"""Cross-rank argmax over vocab-sharded logits through Iris symmetric memory.

Each rank reduces its own vocab shard to one candidate per row, then the ranks
exchange one 8-byte key per row and keep the largest. The key orders the
candidate's value first and its global column second, so the largest key is the
row's argmax over the whole vocabulary with ties broken toward the lowest
column, and every rank decodes the same index.
"""

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.communication._iris import IrisGroup, iris

# A row with no valid candidate on any rank (every value NaN under the local
# argmax's NaN rule) decodes to the same -1 the local argmax returns.
_NO_CANDIDATE = tl.constexpr(-(2**63))
_LOW_WORD = tl.constexpr(0xFFFFFFFF)


class IrisArgmaxExchange:
    """Own the rotating key slots and per-block epochs of one group's exchange."""

    BLOCK_ROWS = 256
    NUM_SLOTS = 2
    NUM_WARPS = 4

    def __init__(self, peers: IrisGroup, max_rows: int):
        if max_rows <= 0:
            raise ValueError(f"max_rows must be positive, got {max_rows}")
        self.peers = peers
        self.max_rows = max_rows
        self.slots = peers.ctx.zeros((self.NUM_SLOTS, max_rows), dtype=torch.int64)
        self.ready_flags = peers.ctx.zeros(
            (triton.cdiv(max_rows, self.BLOCK_ROWS), peers.world_size),
            dtype=torch.int32,
        )

    def run(
        self,
        logits: torch.Tensor,
        local_index: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Reduce each row's local candidate across the group.

        Args:
            logits: This rank's ``[rows, vocab_per_rank]`` shard; rank ``r``
                holds global columns ``[r * vocab_per_rank, (r + 1) *
                vocab_per_rank)``. Only the candidate column is read.
            local_index: ``[rows]`` int32/int64 argmax within the shard, ``-1``
                for a row without a valid candidate.
            out: ``[rows]`` int32/int64 global argmax, ``-1`` when no rank has
                a candidate. Identical on every rank.
        """
        rows, vocab_per_rank = logits.shape
        if not 0 < rows <= self.max_rows:
            raise ValueError(f"rows={rows} outside (0, {self.max_rows}]")
        if local_index.shape != (rows,) or out.shape != (rows,):
            raise ValueError("local_index and out must be [rows]")
        if logits.stride(1) != 1:
            raise ValueError("logits rows must be contiguous")
        iris_argmax_exchange[(triton.cdiv(rows, self.BLOCK_ROWS),)](
            logits,
            local_index,
            self.slots,
            out,
            self.ready_flags,
            self.peers.bases,
            rows,
            logits.stride(0),
            COLUMN_OFFSET=self.peers.rank * vocab_per_rank,
            RANK=self.peers.rank,
            WORLD_SIZE=self.peers.world_size,
            BLOCK_ROWS=self.BLOCK_ROWS,
            SLOT_STRIDE=self.max_rows,
            NUM_SLOTS=self.NUM_SLOTS,
            num_warps=self.NUM_WARPS,
        )


@triton.jit
def _argmax_key(value, column):
    """Pack ``(value, column)`` into an int64 whose signed order is the argmax order.

    The high word maps the float's bits to an int32 that orders like the float
    (negative values flip their magnitude bits), with -0.0 folded onto +0.0 so
    the two zeros tie as they compare. NaN takes the top of the range: a
    candidate is only NaN when the local argmax lets NaN win, and then it wins
    here too. The low word is the complemented column, so equal values prefer
    the lower column.
    """
    value = tl.where(value == 0.0, 0.0, value)
    bits = value.to(tl.int32, bitcast=True)
    ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    ordered = tl.where(value != value, 0x7FFFFFFF, ordered)
    low = _LOW_WORD - column.to(tl.int64)
    return (ordered.to(tl.int64) << 32) | low


@triton.jit(do_not_specialize=["rows", "row_stride"])
def iris_argmax_exchange(
    logits_ptr,
    local_index_ptr,
    slots_ptr,
    out_ptr,
    ready_flags,
    heap_bases,
    # The row count follows the batch: runtime, so every batch shares one binary.
    rows,
    row_stride,
    COLUMN_OFFSET: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
):
    """Exchange per-row argmax keys over rotating slots and keep the largest.

    The slot protocol is ``iris_pull_oneshot_staged``'s: each block stages its
    keys into slot ``epoch % NUM_SLOTS``, publishes the epoch, waits for every
    peer's, then reads the peers' slots. A rank reuses a slot only after every
    peer published the next epoch, which a peer does only after it finished
    reading the slot, so two slots suffice. Epochs are int32 and wrap: the slot
    comes from the unsigned epoch and the wait compares modulo 2**32.
    """
    tl.static_assert(NUM_SLOTS & (NUM_SLOTS - 1) == 0)
    block_id = tl.program_id(0)
    row = block_id * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    mask = row < rows

    local_index = tl.load(local_index_ptr + row, mask=mask, other=-1).to(tl.int64)
    valid = mask & (local_index >= 0)
    value = tl.load(
        logits_ptr + row.to(tl.int64) * row_stride + local_index,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    key = tl.where(valid, _argmax_key(value, local_index + COLUMN_OFFSET), _NO_CANDIDATE)

    flag_offset = block_id * WORLD_SIZE
    local_ready = ready_flags + flag_offset + RANK
    epoch = tl.load(local_ready).to(tl.int32) + 1
    slot = (epoch.to(tl.uint32, bitcast=True) % NUM_SLOTS).to(tl.int32) * SLOT_STRIDE
    tl.store(slots_ptr + slot + row, key, mask=mask, cache_modifier=".wt")
    tl.debug_barrier()

    tl.atomic_xchg(local_ready, epoch, sem="release", scope="sys")
    for peer in tl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            seen = epoch - 1
            while (seen - epoch) < 0:
                seen = iris.load(
                    ready_flags + flag_offset + peer,
                    RANK,
                    peer,
                    heap_bases,
                    cache_modifier=".cv",
                    volatile=True,
                )

    best = key
    for peer in tl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            best = tl.maximum(
                best,
                iris.load(
                    slots_ptr + slot + row,
                    RANK,
                    peer,
                    heap_bases,
                    mask=mask,
                    other=_NO_CANDIDATE,
                    cache_modifier=".cg",
                ),
            )
    index = tl.where(best == _NO_CANDIDATE, -1, _LOW_WORD - (best & _LOW_WORD))
    tl.store(out_ptr + row, index.to(out_ptr.type.element_ty), mask=mask)
