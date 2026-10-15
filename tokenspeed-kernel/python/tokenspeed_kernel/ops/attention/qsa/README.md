# QSA sparse attention

The operator consumes per-query physical cache slots. Uniform decode passes its
query-token count explicitly; prefill and ragged queries use the general fallback.

CuTe DSL gathers, prefetches and causal-tail reads widen physical slot IDs
before calculating KV element offsets. Slot IDs remain int32; cache addresses
use int64 so a large paged cache can cross the signed 32-bit element boundary.
The large-offset GPU regression covers BF16/FP8, split and unsplit execution,
masked rows and graph replay with changed high-address values.
The delayed-producer regression also crosses the signed 32-bit byte boundary,
so large-buffer PDL checks publish all KV data before attention consumes it.

For BF16 groups of twelve query heads (Hq/Hkv = 24/2 or 12/1), CuTe DSL uses
one sixteen-head MMA tile, sharing each KV operand across the group. Larger
batches combine consecutive queries from the same uniform request. Reuse
requires equality of all 2048 ordered prefix entries; it does not form a union
of partially overlapping candidate sets. Every query retains separate softmax
state and its three private tail candidates. Distinct prefixes use independent
query CTAs, with stable 32-bucket ordering where measured beneficial for L2.

The retained configurations cover these model head geometries:

| Query / KV heads | Query groups | Prefix detection |
| --- | --- | --- |
| 24 / 2 | 1, 2, 4, 6, 8 | In attention for 2; per-call flags for 4, 6, 8 |
| 12 / 1 | 1, 2, 4 | In attention for 2; per-call flags for 4 |
| 6 / 1 | 1, 2, 3, 4 | In attention |

Grouping depends on the uniform request width and available SM work. Native
48/72/96-head tiles pack twelve real heads per query; other grouped tiles retain
padding and masking. Register budgets, producer/softmax warps and private-tail
work follow the measured geometry in one adaptive attention kernel. Other head
geometries and FP8 preserve their existing split-cluster support.

The public launcher, per-call prefix preparation and shared device helpers are
separate files under the CuTe DSL implementation. Prefix preparation reads slot
IDs and writes fresh flags; it does not copy KV or retain request state. Large
twelve-head groups use preparation plus one attention launch. Other groups
classify inside their single attention launch. CUDA graph replay recomputes
all decisions from live inputs, including changed masks and private tails.
PDL synchronization remains explicit at producer-owned reads.
Separate even/odd QK-ready and PV notifications protect pipeline reuse.
Delayed-PV regressions stress both phases with two and three operand buffers.

Performance comparisons include the complete GPU call, including any prefix
preparation, with matching hot/cold cache and PDL settings. Correctness keeps
the FP64 reference tolerance of atol=0.002 and rtol=0.02.
