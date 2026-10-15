# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Refresh exact ordered-prefix flags on each call and graph replay."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_sparse.common import (
    _HEAD_DIM,
    _to_tvm_meta,
    copy_query_vector,
    prepare_cta_all,
)

_COMPILED_PREPARE = {}


@cute.kernel
def _prepare_query_groups_kernel(
    query: cute.Tensor,
    selected: cute.Tensor,
    packed: cute.Tensor,
    shared: cute.Tensor,
    kv_heads: cutlass.Constexpr[int],
    query_head_tile: cutlass.Constexpr[int],
    query_group: cutlass.Constexpr[int],
    enable_pdl: cutlass.Constexpr[bool],
    pack_query: cutlass.Constexpr[bool],
):
    group, kv_head, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.make_warp_uniform(tid // 32)
    if cutlass.const_expr(not kv_heads == 2):
        mismatches = utils.SmemAllocator().allocate_array(
            cutlass.Int32,
            8,
            byte_alignment=16,
        )
    if cutlass.const_expr(enable_pdl):
        cute.arch.griddepcontrol_wait()
    if cutlass.const_expr(kv_heads == 2):
        comparison_threads = (
            512 if cutlass.const_expr(not pack_query and query_group == 8) else 128
        )
        query_partitions = comparison_threads // 128
        query_partition = tid // 128 if cutlass.const_expr(query_partitions > 1) else 0
        column = tid % 128 if cutlass.const_expr(query_partitions > 1) else tid
    different = cutlass.Boolean(False)
    if cutlass.const_expr(kv_heads == 2):
        anchor = selected[group * query_group, column]
    else:
        anchor = selected[group * query_group, tid]
    if cutlass.const_expr(kv_heads == 2):
        for other_round in cutlass.range_constexpr(
            cute.ceil_div(query_group - 1, query_partitions)
        ):
            other = 1 + query_partition + other_round * query_partitions
            if other < query_group:
                different = (
                    different or anchor != selected[group * query_group + other, column]
                )
    else:
        for other in cutlass.range_constexpr(1, query_group):
            different = (
                different or anchor != selected[group * query_group + other, tid]
            )
        warp_mismatch = cute.arch.warp_reduction_sum(cutlass.Int32(different))
        if lane == 0:
            mismatches[warp] = warp_mismatch
        cute.arch.sync_threads()
        mismatch = cutlass.Int32(0)
        for index in cutlass.range_constexpr(4):
            mismatch += mismatches[index]
        if mismatch == 0:
            for chunk in cutlass.range_constexpr(1, 16):
                index = chunk * 128 + tid
                anchor = selected[group * query_group, index]
                for other in cutlass.range_constexpr(1, query_group):
                    different = (
                        different
                        or anchor != selected[group * query_group + other, index]
                    )
            warp_mismatch = cute.arch.warp_reduction_sum(cutlass.Int32(different))
            if lane == 0:
                # Keep the first reduction immutable until all warps have read it.
                mismatches[warp + 4] = warp_mismatch
            cute.arch.sync_threads()
            mismatch = cutlass.Int32(0)
            for index in cutlass.range_constexpr(4):
                mismatch += mismatches[index + 4]
    if cutlass.const_expr(kv_heads == 2):
        shared_prefix = prepare_cta_all(not different, comparison_threads)
    else:
        shared_prefix = mismatch == 0
    if cutlass.const_expr(kv_heads == 2):
        if shared_prefix:
            for chunk in cutlass.range_constexpr(1, 16):
                index = chunk * 128 + column
                anchor = selected[group * query_group, index]
                for other_round in cutlass.range_constexpr(
                    cute.ceil_div(query_group - 1, query_partitions)
                ):
                    other = 1 + query_partition + other_round * query_partitions
                    if other < query_group:
                        different = (
                            different
                            or anchor != selected[group * query_group + other, index]
                        )
            shared_prefix = prepare_cta_all(not different, comparison_threads)
    if kv_head == 0 and tid == 0:
        shared[group] = cutlass.Int32(1) if shared_prefix else cutlass.Int32(0)
    if cutlass.const_expr(pack_query):
        if shared_prefix:
            heads = query.shape[1]
            heads_per_kv = heads // kv_heads
            elements_per_query = heads_per_kv * _HEAD_DIM
            padded_elements = query_head_tile * _HEAD_DIM
            packed_base = (
                packed.iterator
                + cutlass.Int64(group) * packed.shape[1] * _HEAD_DIM
                + kv_head * padded_elements * query_group
            )
            for chunk in cutlass.range_constexpr(
                padded_elements * query_group // (128 * 8)
            ):
                element = chunk * 128 * 8 + tid * 8
                local_query = element // padded_elements
                local_element = element % padded_elements
                if local_element < elements_per_query:
                    source = (
                        query.iterator
                        + (cutlass.Int64(group) * query_group + local_query)
                        * heads
                        * _HEAD_DIM
                        + kv_head * elements_per_query
                        + local_element
                    )
                    copy_query_vector(source, packed_base + element)
                else:
                    for item in cutlass.range_constexpr(8):
                        packed_base[element + item] = cutlass.BFloat16(0)
    if cutlass.const_expr(enable_pdl):
        cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _prepare_query_groups_launch(
    query: cute.Tensor,
    selected: cute.Tensor,
    packed: cute.Tensor,
    shared: cute.Tensor,
    kv_heads: cutlass.Constexpr[int],
    query_head_tile: cutlass.Constexpr[int],
    query_group: cutlass.Constexpr[int],
    enable_pdl: cutlass.Constexpr[bool],
    pack_query: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    if cutlass.const_expr(kv_heads == 2):
        _prepare_query_groups_kernel(
            query,
            selected,
            packed,
            shared,
            kv_heads,
            query_head_tile,
            query_group,
            enable_pdl,
            pack_query,
        ).launch(
            grid=(query.shape[0] // query_group, kv_heads if pack_query else 1, 1),
            block=(
                512 if cutlass.const_expr(not pack_query and query_group == 8) else 128,
                1,
                1,
            ),
            stream=stream,
            use_pdl=enable_pdl,
        )
    else:
        _prepare_query_groups_kernel(
            query,
            selected,
            packed,
            shared,
            kv_heads,
            query_head_tile,
            query_group,
            enable_pdl,
            pack_query,
        ).launch(
            grid=(query.shape[0] // query_group, kv_heads, 1),
            block=(128, 1, 1),
            stream=stream,
            use_pdl=enable_pdl,
        )


def _prepare_query_groups(
    query,
    selected,
    packed,
    shared,
    kv_heads,
    query_head_tile,
    query_group,
    enable_pdl,
    pack_query,
):
    key = (
        query.device.index,
        query.shape[1],
        kv_heads,
        query_head_tile,
        query_group,
        enable_pdl,
        pack_query,
    )
    compiled = _COMPILED_PREPARE.get(key)
    if compiled is None:
        descriptors = [
            _to_tvm_meta(query, 16).mark_compact_shape_dynamic(
                mode=0, stride_order=(0, 1, 2), divisibility=query_group
            ),
            _to_tvm_meta(selected, 16).mark_compact_shape_dynamic(
                mode=0, stride_order=(0, 1), divisibility=query_group
            ),
            _to_tvm_meta(packed, 16).mark_compact_shape_dynamic(
                mode=0, stride_order=(0, 1, 2)
            ),
            _to_tvm_meta(shared, 4).mark_compact_shape_dynamic(
                mode=0, stride_order=(0,)
            ),
        ]
        compiled = cute.compile(
            _prepare_query_groups_launch,
            *descriptors,
            kv_heads,
            query_head_tile,
            query_group,
            enable_pdl,
            pack_query,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi --opt-level 3",
        )
        _COMPILED_PREPARE[key] = compiled
    compiled(query, selected, packed, shared)
