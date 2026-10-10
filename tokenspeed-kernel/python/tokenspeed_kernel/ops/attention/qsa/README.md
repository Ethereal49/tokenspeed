# Query sparse attention kernels

QSA scores compressed index keys, selects logical blocks, expands them to
full-cache slots, and attends to those slots. The public block selector takes
an explicit `stream` or `logits` solution; slot expansion and attention share
the same contracts for both solutions.

## Streaming block selection

On NVIDIA Blackwell, CuTe DSL accepts four BF16 index heads of width 128,
64-block pages, block top-k 512, and capacities up to 65536 compressed blocks
(256K tokens at compression ratio four). Strided Q vectors must remain
16-byte aligned; keys and metadata must be contiguous and aligned.

Grouped scoring shares each K tile across eight consecutive queries from one
request. The caller supplies the uniform request-run width; grouping is never
inferred from row count. Each selector thread keeps eight packed FP32-score/id
entries, forming an exact 512-entry running set per query. Large prefills need
no global selection scratch. Capacities up to 4096 compressed blocks use a
single score tile; longer streams overlap MMA, score publication and top-k
selection through two K stages, two accumulator stages and two score buffers.
The long-stream CTA uses 26 warps and 1120-entry candidate buffers containing
FP32 scores and UInt16 ids. Dedicated compaction scratch protects both K stages.

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
layout uses CTA-local streaming; other layouts retain portable Triton. Query
count, request count, page-table width, Q strides and live frontiers remain
runtime arguments. Split geometry uses power-of-two capacity buckets, and only
kernel geometry, request-id dtype, device and the explicit PDL flag specialize
the compiled cache. PDL waits precede asynchronous Q copies and producer-owned
metadata reads; captured graphs retain the supplied flag.

## Materialized block selection

The `logits` solution materializes FP32 scores and then radix-selects. A
caller-owned workspace enables the length-aware persistent CUDA selector;
other callers use portable Triton top-k. Runtime routing can prefer logits
when the score matrix fits its configured memory budget. Adding the CuTe
stream implementation preserves this routing policy.

## Measurement

`test/nvidia/ops/bench_qsa_stream_topk.py` compares direct CuTe streaming,
split Triton streaming and logits with persistent CUDA top-k. All paths use
the same inputs and clear L2 before each invocation. CUDA events exclude
cache clearing. Graph timing excludes host launch gaps and reports kernel
registers, shared memory and occupancy limits through the CUDA driver.

From the repository root, measure a causal 16K-token prefill with:

```bash
PYTHONPATH=python:tokenspeed-kernel/python \
python tokenspeed-kernel/test/nvidia/ops/bench_qsa_stream_topk.py \
  --rows 16384 --pages 1024 --kv-tokens 32768 --queries-per-request 16384 \
  --solution cute_dsl --timing graph --frontier causal \
  --distribution random --pdl on --warmups 10 --iterations 40
```

This reserves a 256K-token page-table capacity with 32K live KV tokens and a
16K-query causal chunk. Omit `--kv-tokens` to use the table's full capacity.
Requests use disjoint physical key pages, with at least 1024 physical pages in
the pool. Change `--solution` to `triton` or `logits` for comparisons. Use `--frontier
full` for every row attending to the full prefix, and `--distribution ties`
for zero queries that stress equal-score selection. Cold-cache isolated
measurements establish kernel costs; end-to-end model measurements must also
include slot expansion, sparse attention and interference from other layers.
