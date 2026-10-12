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

"""Host-buffer geometry exported without Device arenas or transfer streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class HostBufferView:
    """The CPU allocation and scalar geometry a Host-only Store may retain.

    Attributes:
        host_buffer: The registered CPU byte allocation.
        num_host_lcm_blocks: Usable parents in that allocation.
        host_lcm_block_bytes: Byte stride between parents.
        block_bytes: Payload bytes of one child in each group.
        packing: Children per parent for each group.
        group_ids: Group names used in bounds diagnostics.
    """

    host_buffer: Any
    num_host_lcm_blocks: int
    host_lcm_block_bytes: int
    block_bytes: tuple[int, ...]
    packing: tuple[int, ...]
    group_ids: tuple[str, ...]

    def host_block_range(self, group_index: int, block_id: int) -> tuple[int, int]:
        """Return the byte range of a non-null local Host block."""
        if not 0 <= group_index < len(self.packing):
            raise IndexError(f"unknown cache group index {group_index}")
        packing = self.packing[group_index]
        bound = self.num_host_lcm_blocks * packing
        if not 1 <= block_id <= bound:
            raise IndexError(
                f"block_id {block_id} outside [1, {bound}] for group {self.group_ids[group_index]!r}"
            )
        parent, child = divmod(block_id - 1, packing)
        size = self.block_bytes[group_index]
        return parent * self.host_lcm_block_bytes + child * size, size
