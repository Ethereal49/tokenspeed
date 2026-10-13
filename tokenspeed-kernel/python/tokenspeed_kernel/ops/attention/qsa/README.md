# Query sparse attention kernels

QSA scores compressed index keys, selects logical blocks, expands them to
full-cache slots, and attends to those slots. The public block selector in
`ops.attention.qsa` takes an explicit `auto`, `stream` or `logits` path and
dispatches registered implementations through `select_kernel`. Backend
selection (`solution`) and kernel overrides preserve the requested path.
Slot expansion and attention share the same contracts for all paths.

## Automatic selection

Automatic selection uses query rows, compressed block-table capacity and
supported layouts, independently of prefill/decode. B300 cold-L2 CUDA-graph
measurements favor grouped CuTe streaming from 512 rows for tables of at least
16384 compressed blocks (64K tokens at ratio four), and from 8192 rows for
shorter tables. Smaller batches use logits. Uniform runs must satisfy the
CuTe grouping contract; ragged runs and unsupported layouts use logits while
its matrix fits the caller's budget. Beyond that budget, selection uses
streaming, with portable Triton covering unsupported CuTe layouts.

The runtime passes `TOKENSPEED_QWEN4_EXP_QSA_TOPK_PATH` and
`TOKENSPEED_QWEN4_EXP_QSA_MAX_LOGITS_MB` to this shared entry. Explicit `stream`
and `logits` paths bypass the automatic choice and its score-matrix budget.
`solution="cute_dsl"` or `solution="triton"` restricts the backend within the
chosen path, with normal capability and layout filtering. A kernel override
cannot change its required path feature; the CuTe entry rejects unsupported
layouts when called directly or through an override.

## Streaming block selection

On NVIDIA Blackwell, CuTe DSL accepts four BF16 index heads of width 128,
64-block pages, block top-k 512, and capacities up to 131072 compressed blocks
(512K tokens at compression ratio four), including speculative context padding.
Strided Q vectors must remain 16-byte aligned. Page-table columns must be
contiguous; their runtime row stride may retain the persistent table capacity
when a batched prefill trims its view. Keys and row metadata remain contiguous
and aligned.

Grouped scoring shares each K tile across eight consecutive queries from one
request. The caller supplies the uniform request-run width; grouping is never
inferred from row count. Each selector thread keeps eight packed FP32-score/id
entries, forming an exact 512-entry running set per query. Large prefills need
no global selection scratch. Capacities up to 4096 compressed blocks use a
single score tile; longer streams overlap MMA, score publication and top-k
selection through two K stages, two accumulator stages and two score buffers.
The long-stream CTA uses 26 warps and 1120-entry candidate buffers containing
FP32 scores and UInt16 offsets within a 64K-block window. Score stages flush at
window boundaries, and selection restores the full block ids from the stage
base. Compact split candidates similarly store tile-local offsets. Neither
representation truncates ids beyond 65535. Dedicated compaction scratch protects
both K stages.

Once 512 candidates exist, their monotone score cutoff filters later scores
before histogram updates and compaction. An older cutoff can admit extra
candidates but cannot discard a winner. Coarse histograms locate a threshold
bucket; exact FP32 refinement selects the boundary. Large tie buckets use full
radix refinement. FP16 or linear bins only filter candidates and never replace
FP32 ranking. Each row retains its own frontier; groups stop after their longest
frontier. Rows with at most 512 complete blocks return all valid ids and pad
remaining slots with `-1`. Output order and boundary tie ids are unspecified.

Grids of up to 128 queries expose parallelism by splitting the key range.
Independent rows, including ragged layouts, use four-warp native BF16 MMA score
CTAs and an exact merge. Small grouped prefills with long contexts reuse K
across eight queries in each split. Some small grids propose fewer candidates
per key tile, preserving the largest discarded FP32 score. The merge uses the
reduced pool only when its threshold proves that every discarded score is
ineligible; otherwise it selects from complete scores. Compact candidate ids
and short-context linear bins reduce selection traffic and histogram overhead.

`max_partial_bytes` bounds global scratch for both CuTe and Triton, excluding
the selected-id output. If split scratch exceeds the budget, a supported grouped
CuTe layout uses CTA-local streaming; kernel selection excludes other CuTe layouts.
Portable Triton groups a divisor of the explicitly supplied request-run width,
up to four queries, to reuse each K tile. Ragged runs use group size one in the
same kernel. Each row keeps its own frontier and exact packed FP32-score/id
running top-k; a group stops at its longest frontier and writes final ids
directly. The monotone cutoff skips tile selection when no new key can qualify.
Head-dimension tiles of 32 bound shared memory, and split comparison pairs reduce
sorting layout conversions. Sorted tiles are padded after sorting, and groups
with at most top-k complete blocks return all ids without scoring. No partial
or merge buffer is allocated, even with a zero budget.
Query count, request count, page-table width, Q strides and live frontiers remain
runtime arguments. Split geometry uses power-of-two capacity buckets to bound
the tuned kernels' compilation cache. PDL waits precede asynchronous Q copies
and producer-owned metadata reads; captured graphs retain the supplied flag.

## Materialized block selection

The `logits` path materializes FP32 scores and then radix-selects. A
caller-owned workspace enables the length-aware persistent CUDA selector;
other callers use portable Triton top-k. Both scoring paths are registered
under `attention.qsa_block_topk`; required `stream`/`logits` features keep
selection and overrides within the requested algorithm.

## Measurement

`test/nvidia/ops/bench_qsa_stream_topk.py` compares selected CuTe streaming,
portable Triton streaming, logits with persistent CUDA top-k, and automatic
selection. All paths use the public API with the same inputs and clear L2
before each invocation. CUDA events exclude
cache clearing. Graph timing excludes host launch gaps and reports kernel
registers, shared memory and occupancy limits through the CUDA driver.

From the repository root, measure a causal 16K-token prefill with:

```bash
PYTHONPATH=python:tokenspeed-kernel/python \
python tokenspeed-kernel/test/nvidia/ops/bench_qsa_stream_topk.py \
  --rows 16384 --pages 1024 --kv-tokens 32768 --queries-per-request 16384 \
  --solution cute_dsl --max-logits-mb 256 --timing graph --frontier causal \
  --distribution random --pdl on --warmups 10 --iterations 40
```

This reserves a 256K-token page-table capacity with 32K live KV tokens and a
16K-query causal chunk. Omit `--kv-tokens` to use the table's full capacity.
Requests use disjoint physical key pages, with at least 1024 physical pages in
the pool. Change `--solution` to `triton`, `logits` or `auto` for comparisons. Use `--frontier
full` for every row attending to the full prefix, and `--distribution ties`
for zero queries that stress equal-score selection. Cold-cache isolated
measurements establish kernel costs; end-to-end model measurements must also
include slot expansion, sparse attention and interference from other layers.

B300 cold-L2 graph measurements below use trimmed tables, BF16 four-head
queries of width 128, top-k 512, PDL enabled, causal frontiers and one uniform
request. Each value is the median of three seed medians with 45 samples per
seed. Times cover scoring and selection only, in microseconds.

| Query rows / live KV tokens | Logits | CuTe stream | Auto |
| --- | ---: | ---: | ---: |
| 256 / 32768 | 36.9 | 67.6 | 36.9 |
| 512 / 65536 | 112.6 | 100.4 | 100.4 |
| 8192 / 8192 | 135.2 | 116.9 | 118.4 |
| 16384 / 16384 | 405.5 | 262.1 | 262.1 |
| 16384 / 32768 | 798.8 | 700.4 | 700.4 |
| 16384 / 65536 | 1538.0 | 1198.1 | 1198.1 |

Table capacity also matters: at 512 query rows and 65536 live KV tokens,
reserving 65600 compressed blocks raises logits to 163.8 us while CuTe
streaming remains 100.4 us. The row thresholds are conservative: some tiny
trimmed grids favor CuTe by a few microseconds, and individual short-context
layouts can favor logits beyond the threshold. Explicit paths remain available
for those workloads.

The portable Triton optimization uses the same measurement method with 50
samples per seed, pairing both versions and the other backends on each GPU.
Across 149 shapes, including multiple requests, small request-run widths,
padded capacities and full frontiers, it improves by 1.71–9.95x with a 3.60x
median and no measured regression. All Triton cases allocate only their output.
For the model's grouped layout, shared memory drops from 133 KiB to 33 KiB,
registers from 255 to 210 per thread, and local-memory spills disappear.

| Query rows / live KV tokens | Triton before | Triton | CuTe stream | Logits |
| --- | ---: | ---: | ---: | ---: |
| 1 / 32768 | 415.7 | 141.3 | 10.2 | 12.3 |
| 512 / 65536 | 3142.7 | 444.4 | 100.4 | 112.6 |
| 1024 / 131072 | 11097.6 | 1736.7 | 167.9 | 221.2 |
| 8192 / 8192 | 3661.8 | 454.7 | 118.8 | 135.2 |
| 16384 / 32768 | 34591.8 | 4739.7 | 700.4 | 798.7 |
| 16384 / 262144 | 350049.3 | 46774.8 | 3816.4 | 5894.2 |

Times are microseconds. CuTe or logits remains fastest in every measured shape;
automatic path thresholds are unchanged. Triton provides a faster portable
zero-scratch fallback, while tiny grids still lack key-range parallelism.
