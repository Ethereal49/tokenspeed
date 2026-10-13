# RecoverSSM PR1 progress

> **Benchmark correction:** Earlier attention-runtime timings from
> `outputs/pr1/bench_runtime.py` reused the same parent page IDs across cache
> groups whose arena views overlap. Those timings are invalid, including the
> previously reported 3.41% and 6.64% slowdowns. They remain below as historical
> investigation records, not acceptance evidence. The corrected harness assigns
> disjoint parent pages and verifies every live tensor address range. See the
> final corrected-timing section. Kernel and runtime correctness results are
> separate from these invalid timing runs.


Status: candidate `9b24cc49` is integrated into `kda-recoverssm-pr1` and
pushed to `byshiue`. A separate frozen worktree retains the validated source.
The candidate preserves main verify arithmetic at every tested batch size.
Before the test-scope cleanup recorded below, 214 repository tests passed,
with 24 platform skips. All 69 actual-input layers and 276 actual-weight
gate cases match exactly. Full repository hooks pass. Full-model correctness and all 20 calibrated
attention settings pass. The complete ten-pair E2E campaign fails C=8
throughput: ratio 0.94104, simultaneous 95% interval [0.89237, 0.99237].
Three metrics pass and eight are inconclusive. The failed metric makes this
campaign ineligible for an additional batch. PR1 remains incomplete.

The previous candidate `27c1dbe3` passed correctness and the calibrated
attention matrix, but failed four metrics in the complete ten-pair E2E
campaign. That failure remains recorded and is not eligible for an extra
batch. The new source requires fresh performance samples and all original
gates before acceptance. Earlier results below apply only to their named
source. Integration is complete; the failed E2E performance gate remains open.

## Branch and commits

- Branch: `kda-recoverssm-pr1`.
- Unchanged main baseline: `f4ac1affe11ad404720bcd150970487f75fbf59a`.
- Design commit: `7a47827e6ce2112aa32f252fbc2e80514bfd411e`.
- Implementation and review fixes are committed as `c1208b87` and pushed.
- Earlier GPU measurements below describe their recorded source snapshots.
  The resumed validation at the end of this record covers the committed review
  fixes and the new, uncommitted composition tests.
- Claude approved the revised design, including the two-precision producer
  contract. This is not code approval or a performance exception.
- Keep all design versions on this branch until validation and review finish.

## Environment

Attention-only checks use one NVIDIA GB300 from a persistent one-node Slurm
allocation. No model weights are needed for these checks.

| Component | Version |
| --- | --- |
| Driver | 580.167.08 |
| PyTorch | 2.14.0+cu130 |
| CUDA | 13.0 |
| tokenspeed-triton | 3.8.10.post20260920 |

Baseline and candidate use the same allocation, container and dependencies.
Main is a separate unchanged worktree. Local launchers, allocation details
and raw results are in ignored `outputs/pr1/`.

## Initial implementation milestones (historical)

| Milestone | Status |
| --- | --- |
| New branch, design retained, rebase to main | Done |
| Claude design review | Approved |
| Licensed vLLM kernel copy and current-round runtime adapter | Committed as `c1208b87`; focused GPU suites pass |
| Strict accepted-state compatibility | Initial tests pass |
| Combined convolution and gate producers | Initial tests pass |
| KDA performance non-regression | T4/A2 improves; B32/A1 still regresses |
| Real-model correctness, AIME and E2E performance | Not run |

The adapter keeps the existing accepted-state planner and convolution commit.
FP32 K/U/D replaces the NVIDIA per-round replay payload. Workspace accounting
includes these larger records. No scheduler or persistent-cache geometry
change is required for PR1.

## Accuracy

The first port saved corrections from BF16 verify inputs. It failed an existing
state regression test. Main replay uses FP32 convolution and gate results.
Casting rounded verify inputs to FP32 does not restore those results.

The current producer keeps two state chains. One computes verify outputs.
The independent FP32 chain creates recovery records. Convolution retains
each path's tap order. The original state limit remains `atol=1e-5, rtol=1e-3`.

Run the suites separately because their conftest roots differ:

```bash
python -m pytest -q test/runtime/test_kimi_k3_kda_eager_commit.py
python -m pytest -q -s tokenspeed-kernel/test/nvidia/ops/attention/test_kda_vllm_recoverssm.py
```

After producer fusion:

- Existing runtime suite: 19 passed.
- New kernel suite: 11 passed.
- Existing replay-kernel suite: 33 passed, 1 platform-specific test skipped.
- 128 sequential rounds at B4/T1, B4/T4 and B16/T4:
  maximum state error `2.3841858e-7`; state RMS about `1.0e-9`.
- Maximum BF16 verify-output error: `6.1035156e-5`.
- Combined verify convolution matches the original producer exactly.
- Perturbing rounded verify inputs leaves recovery records unchanged.

Width-one speculative verification is not a standard-decode serving test.
These rounds commit an exact checkpoint each time. They do not validate
PR2 retained history or ring wraparound.

## Diagnostic performance

Scope: 69 KDA layers, 12 local heads, head dimension 128; B4/T4, two accepted
tokens. The graph includes conv/gates, verify, recurrent commit and conv
commit. It excludes projections, MLA, MoE, sampling, communication and
metadata refresh.

These are medians of 50 CUDA-event samples after warmup, from one process per
arm. They do not satisfy the independent-start acceptance protocol.
Commit is inside the measurement graph, but remains outside the production
forward graph until acceptance is available.

| Implementation | Total KDA time | Versus main |
| --- | ---: | ---: |
| Unchanged main | 850.4 us | Reference |
| Separate precision producers | 1304.1 us | +53.3% |
| Wider verify tile experiment | 1354.2 us | +59.2%; discarded |
| Combined producers | 1109.5 us | +30.5% |
| Combined producers with overlap | 1072.6 us | +26.1% |
| Rolled recurrence loop with overlap | 1005.0 us | +18.2% |
| Fixed-width specialization experiment | 1133.1 us | +33.2%; discarded |

Raw JSON files are in `outputs/pr1/`: `main-b4-t4-a2.json`,
`candidate-b4-t4-a2.json`, `candidate-b4-t4-a2-bv32.json`,
and `candidate-b4-t4-a2-fused-producers.json`. Later measurements are
`candidate-b4-t4-a2-overlapped.json`,
`candidate-b4-t4-a2-rolled-loop.json`, and
`candidate-b4-t4-a2-fixed-windows.json`.

The short Nsight reports `main-b4-t4-a2-kernel-detail.nsys-rep` and
`candidate-b4-t4-a2-kernel-detail.nsys-rep` each contain five attention graph
replays. They compare main with the combined-producer revision before overlap
and loop tuning. They are diagnostic traces, not profiles of the final code.

In those traces, average verify kernel time is 7.27 us per layer for main and
9.36 us for the two-chain kernel. Recurrent commit falls from 76.1 us to
52.3 us across all 69 layers; the candidate also removes main's 12.5 us
batched gate-precompute launch.

A local unrolled-kernel tile sweep found 137 registers per thread and zero
spill slots for BV4/one warp. Larger tiles did not improve the full attention
result. Occupancy and final rolled-kernel resource reporting remain pending.
These register results do not describe the later rolled-loop binary.

## Producer and launch optimization

The retained implementation gives verify and record production separate CTAs
within one launch. Each CTA keeps one state tile. Convolution prepares FP32
normalized Q/K after the original BF16 rounding. Gate preparation produces
separate verify and replay decay values. Replay retains its FP32 inputs.
This removes repeated normalization and gate transforms from value tiles.

The four-token recurrence is unrolled. Small batches use a 16-value tile
with four warps. Larger batches use a four-value tile with one warp. Wider
tiles spilled registers or reduced throughput. A separate verify/record tile
experiment increased register use and was discarded.

Gate row count and commit batch count are runtime values. Verify uses a
power-of-two batch bucket. A new test checks that request counts within the
same launch configuration do not cause another Triton compilation.

### Matched attention results

The replacement node is also a GB300. Both arms use the same GPU and software
listed above. These results use T4, two accepted tokens and 69 KDA layers.
They remain diagnostic single-process measurements, not an E2E result or
the independent-start acceptance test.

| Batch | Main | Earlier candidate | Retained candidate | Retained versus main |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 852.4 us | 928.3 us | 862.9 us | +1.2% |
| 8 | 1337.8 us | 1460.0 us | 1229.3 us | -8.1% |
| 16 | 1567.2 us | 2184.7 us | 1761.8 us | +12.4% |
| 32 | 2416.1 us | 3631.6 us | 2843.1 us | +17.7% |

The earlier candidate had prepared decay but not prepared verify Q/K. It used
the same four-warp tile at every batch size. The retained candidate improves
all four rows, but does not pass the no-regression gate.

The earlier milestone's table values use `outputs/pr1/main-final-b*-t4-a2.json` and
`candidate-final-b*-t4-a2.json`. The earlier candidate uses
`candidate-newnode-b*-t4-a2.json`. A preceding matched run is retained in
`main-newnode-b*-t4-a2.json` and `candidate-prepared-qk-b*-t4-a2.json`.
Both matched runs show the same performance pattern.
The local launcher is `outputs/pr1/submit.sh`. The attention harness is
`outputs/pr1/bench_runtime.py`; it accepts `--source`, `--output`, `--batch`,
`--width` and `--accepted`. Select the unchanged main worktree for the baseline.

### Accuracy and profile evidence

- Retained kernel suite: 12 passed, including the compilation-reuse check.
- Existing runtime suite: 19 passed.
- Existing replay-kernel suite: 33 passed, 1 platform-specific test skipped.
- Full `pre-commit run --all-files`: passed after formatting.
- Across 128 rounds, maximum state error is `2.9802322e-7` and maximum
  BF16 output error is `3.0517578e-5`. State RMS remains about `1.1e-9`.
- State and output tolerance limits are unchanged.

The new reports are `outputs/pr1/main-b32-t4-a2-prepared-qk.nsys-rep`
and `candidate-b32-t4-a2-prepared-qk.nsys-rep`. Each records five attention
graph replays. They include the prepared-input implementation, unlike the
earlier profiles above. Later source changes add validation and comments;
the retained kernel arithmetic and launch geometry match these reports.

| B32 component | Main average | Candidate average |
| --- | ---: | ---: |
| Verify, per layer; candidate also produces records | 19.05 us | 24.65 us |
| Recurrent commit, all 69 layers | 568.72 us | 536.91 us |
| Convolution producer, per layer | 3.64 us | 5.93 us |
| Gate producer, per layer | 2.78 us | 5.98 us |

Main also launches per-layer payload capture and an all-layer replay gate
precompute. The candidate incorporates this work into its producers. Producer
streams overlap, so these component times must not be summed as wall time.
The largest remaining difference is verify plus FP32 record production.
The next optimization target is that work, not looser numerical tolerances.

### Reproduction snapshot

No implementation commit was made for these measurements. The code diff from
the design commit is saved as `outputs/pr1/retained-optimization-code.patch`.
Its SHA-256 is:

```text
13b003efdb83aaddb17d824838bb3fa53d309944085214bd0e974d3cf6e6f056
```

Apply that patch to the design commit listed above to reproduce the code.
It excludes design documents. Test logs are in
`outputs/pr1/final-optimization-tests.log`; the repeated matched benchmark
log is `outputs/pr1/final-matched-performance.log`. The paired timelines are
also packaged in `outputs/pr1/kda-main-vs-recoverssm-b32-t4-a2-cuda-graph.zip`.

## Regression analysis and reduction-layout optimization

This milestone follows the prepared-Q/K results above. Earlier timelines and
source snapshots do not describe this revision.

### Cause

PR1 must preserve two producer precisions. Main verifies BF16 inputs, but
recomputes accepted tokens from FP32 inputs during recovery. The initial port
produced FP32 records for every candidate before acceptance. It still committed
a full state each round. At T4 with two accepted tokens, record production
therefore processed four tokens where main recovery processed two.

The B32 profile above shows the cost: verify plus records took 24.65 us per
layer, compared with 19.05 us for main verify. Commit saved only 31.81 us across
all 69 layers. Reading and writing those FP32 states moves about 3.47 GB.
The candidate commit's 536.91 us corresponds to about 6.47 TB/s, excluding
smaller record and metadata traffic. This is a memory-heavy operation, so
removing arithmetic there did not offset the extra per-layer recurrence.

The implicit verify layout also put 32 lanes on each key reduction. Address
calculations used runtime strides even when the model layout fixed them.
Masked state selects retained a full tile after invalid suffix tokens, although
no output or record could observe it.

### Retained changes

- Use an explicit Triton Gluon verify layout. Four value rows share a warp;
  eight lanes reduce each key row. This reduces warp shuffle work and avoids
  layout-conversion shared memory in the selected one-warp kernel.
- Keep token updates ordered and FP32. The within-token reduction tree changes.
  This is not a bitwise-equivalence claim. Precision and test limits stay fixed.
- Specialize fixed model-layout strides. Keep the request-dependent group
  pitch and commit batch count as runtime values.
- Remove unobservable suffix-state selects. Mask every output and record store.
  A combined ragged-input test covers empty rows, null pages, unchanged input
  checkpoints and untouched record suffixes.
- Use an eight-value tile through B4 and a 16-value tile above B4 for the
  separate-record producer. Both use one warp. Retain the existing grid order.

The B32 cold-state test cycles through 69 different layer states. Verify plus
records falls from about 21.90 us to 17.75 us per layer after the layout change.
The selected kernel uses 161 registers per thread, no spill slots and no shared
memory. These are compiler resource counts, not measured achieved occupancy.
The previous layout used 91 registers and 32 bytes of shared memory at its
selected tile. A higher register count did not mean a slower kernel here.

Fixed recovery strides reduce the selected commit kernel from 98 to 78
registers per thread. The commit-only harness, including planning and conv
commit, falls from about 598 us to 574 us at B32.

### Discarded experiments

Separate verify and record launches increased launch overhead. Keeping both
state chains in one CTA increased register pressure. A one-warp convolution
producer and a head-major verify grid also made the full attention path slower.
These changes are not retained. Tile sweeps that reuse one state pool can
overstate cache reuse; retained decisions use separate layer states and the
full attention harness.

### Validation

The retained code passes 14 new kernel tests, 19 existing runtime tests and
33 existing replay-kernel tests. One platform-specific test is skipped.
Across 128 rounds, maximum accepted-state error is `2.9802322e-7`.
State RMS is about `1.08e-9`; maximum BF16 output error is `6.1035156e-5`.
The state limit remains `atol=1e-5, rtol=1e-3`.

These checks do not establish real-model token or acceptance equivalence.
True standard-decode serving tests, AIME and E2E performance remain pending.

### Repeated attention comparison

Same GB300, container and software as above; 69 KDA layers, 12 local heads,
dimension 128, T4 and two accepted tokens. Each row is the median of three
independent process medians. Each process measures 50 CUDA-event samples.
Baseline/candidate order reverses in the second trial. Main remains unchanged.

| Batch | Main | Earlier candidate | Current candidate | Current versus main |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 852.5 us | 862.9 us | 823.8 us | -3.4% |
| 8 | 1337.8 us | 1229.3 us | 1060.3 us | -20.7% |
| 16 | 1567.2 us | 1761.8 us | 1547.7 us | -1.2% |
| 32 | 2415.1 us | 2843.1 us | 2381.3 us | -1.4% |

No measured row regresses in these three starts. The B16/B32 margin is small.
B32 current medians range from 2379.9 to 2381.3 us; main ranges from 2414.2 to
2415.1 us. Compared with the earlier candidate, B32 is about 16.2% faster.

The earlier-candidate column repeats the prior milestone, not another arm in
this three-start run. All columns use the same attention harness and settings.
The current run includes conv/gate producers, verify, recurrent commit,
planning and conv commit. It excludes model projections, MLA, MoE, sampling,
communication, prefill and metadata refresh. Commit is included in the test
graph but remains outside the production forward graph until acceptance.

Raw files are `outputs/pr1/regression-{main,candidate}-r{1,2,3}-b*-t4-a2.json`.
The log is `outputs/pr1/regression-matched.log`. This is a local diagnostic
result, not the full ten-start real-model performance gate.

### Acceptance-length sensitivity and remaining regression

The extra checks keep T4 and vary accepted length. These are single-process
diagnostics with the same 50-sample harness.

| Batch | Accepted tokens | Main | Current candidate | Versus main |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 834.0 us | 817.7 us | -2.0% |
| 32 | 1 | 2285.0 us | 2364.9 us | +3.5% |
| 4 | 4 | 889.3 us | 842.2 us | -5.3% |
| 32 | 4 | 2692.4 us | 2481.6 us | -7.8% |

B32 with one accepted token still regresses. The full no-regression gate has
not passed. The candidate creates all four FP32 records regardless of acceptance;
main reduces recovery work when fewer tokens are accepted.

The additional lane-count, vector-layout and rolled-loop experiments did not
improve the selected eight-lane layout. Fully unrolled recurrence is retained.
Do not remove the low-acceptance case from future performance checks.

Raw files are `outputs/pr1/regression-acceptance-{main,candidate}-b*-t4-a*.json`.
The focused A1 timeline shows main recurrent commit at 439.09 us and the
candidate at 497.90 us. Verify is 19.04 us for main and 18.27 us for the
candidate. The 58.81 us commit difference explains most of the approximately
80 us attention regression.

Main commit drops from 567.70 us at A2 to 439.09 us at A1. Candidate commit only
drops from 513.61 us to 497.90 us. Its all-candidate record cost remains fixed,
and each commit still reads and writes a full state. This locates the remaining
cost; the exact memory/scheduling mechanism is not yet resolved.

Commit tile sweeps, processing all value tiles in one CTA, and main's streaming
state-cache hints did not improve the A1 result. They are not retained. The
next target is low-acceptance commit efficiency, alongside the fixed cost of
FP32 record production. Keep precision and phase ordering unchanged.

### Current timeline and source snapshot

The B32/T4/A2 reports each contain five attention graph replays:

- `outputs/pr1/regression-main-GB300-B32-T4-A2-cuda-graph.nsys-rep`
- `outputs/pr1/regression-candidate-GB300-B32-T4-A2-cuda-graph.nsys-rep`

The matching A1 reports use the same names with `A1` in place of `A2`.
All four reports are in
`outputs/pr1/kda-recoverssm-main-vs-optimized-GB300-B32.zip`.
Archive integrity was checked. The A1 profiles confirm the remaining regression.

| Component | Main average | Current candidate average |
| --- | ---: | ---: |
| Verify per layer; candidate also creates records | 19.05 us | 18.29 us |
| Recurrent commit, all layers | 567.70 us | 513.61 us |
| Convolution producer, per layer | 3.62 us | 5.94 us |
| Gate producer, per layer | 2.77 us | 5.98 us |

Main also needs payload capture and batched replay gate preparation. The
candidate includes those tasks in its producers. Producer streams overlap;
these component times must not be summed as elapsed attention time.

The final restored code passes the same 14 + 19 + 33 tests, with one skip.
See `outputs/pr1/regression-restored-tests.log`. Full repository hooks pass;
the log is `outputs/pr1/regression-hooks-final.log`.

The uncommitted code snapshot is `outputs/pr1/regression-optimized-code.patch`.
Apply it to design commit `7a47827e6ce2112aa32f252fbc2e80514bfd411e`.
It includes runtime/kernel code, kernel documentation and tests, but excludes
design documents. SHA-256:

```text
67422187e7b55b372082d2ef90a5d93957f7e2abe5453794e897615d74b44a54
```

The local harness remains `outputs/pr1/bench_runtime.py`; the persistent-job
launcher remains `outputs/pr1/submit.sh`. The accepted-length and A2 profile
commands are in `outputs/pr1/validate-regression-tail.sh`. The A1 profile
commands are in `outputs/pr1/profile-regression-a1.sh`.

## Remaining work

1. Remove the remaining B32/T4/one-accepted-token regression. Extend the
   attention matrix and measure achieved occupancy.
2. Complete composable-record, boundary, padding, alias and backend coverage.
3. Run the B4/8/16/32 matrix, including true standard decode.
4. Validate real Kimi-K3 NVFP4 TP8: deterministic tokens and acceptance,
   agentic workflow, CUDA graphs, eager smoke test and AIME 2026.
5. Run the independent-start performance protocol. Record the tested commit.
6. Run all hooks and obtain code review.
7. Remove temporary design documents only when the work is complete.


## Code-quality review follow-up

Claude Fable 5.1 reviewed the implementation. A second review corrected two
claims: undefined padded output is allowed by the unified-path contract, and
the incompatible per-layer fallback hits an existing assertion rather than
demonstrating silent corruption. Neither claim justifies a numerical change.

The following review fixes are included in `c1208b87`:

- Pass the required replay-record argument from the benchmark generator.
- Remove dead gate and stride-array inputs from the standalone recovery API
  and the copied recovery kernel. Update both test callers.
- Document the producer-only fields retained by the shared runtime adapter.
- Guard the split-record path against the legacy per-layer fallback.
- Name record fields explicitly and document backend-specific payload meanings.
- Register the RecoverSSM operations at module initialization.
- Check producer layouts without device reads or synchronization.
- Add production batched-commit tests for mixed groups, all accepted lengths,
  in-place writes, padding, pointer refresh and graph capture.
- Add padded runtime graph coverage and a CPU benchmark-caller regression test.

FP32 update order, the two producer precisions and numerical limits are
unchanged. No memset or kernel launch is added. Old kernels used as offline
references remain during validation; final removal of unused legacy code is
not complete.

Claude's follow-up found a per-head layout error in the new test oracle.
The oracle now splits each head's record into K and decay; the production
layout did not change. The existing 128-round differential test already
checks prepared Q/K and decay against main.

Python syntax and static launch-signature checks passed before the commit.
The full repository hooks passed after formatting. The subsequent GPU
allocation failed, so GPU tests and matched timing on the committed source
were pending at that point. Resumed validation is recorded below. The login-node
test environment lacks dependencies available in the GPU container. This is not a numerical test pass. Existing performance
and E2E gates remain open; this review does not approve the whole PR.

The source snapshot and review logs are in `outputs/pr1/review-fixes-code.patch`
and `outputs/pr1/claude-review-fixes*.json`. The current launcher is
`outputs/pr1/submit.sh`. Run the suites separately in its existing container:

```bash
bash outputs/pr1/submit.sh python -m pytest -q tokenspeed-kernel/test/test_benchmark_kda_generator.py
bash outputs/pr1/submit.sh python -m pytest -q -s tokenspeed-kernel/test/nvidia/ops/attention/test_kda_vllm_recoverssm.py
bash outputs/pr1/submit.sh python -m pytest -q test/runtime/test_kimi_k3_kda_eager_commit.py
bash outputs/pr1/submit.sh python -m pytest -q tokenspeed-kernel/test/ops/attention/test_kda_replay_commit.py
```

## Initial multi-window composition validation

An uncommitted test change extends the existing 128-round main-producer
comparison to T1/T4 with R8/R16/R32 at B4, plus B16/T4/R8. Each round uses the
actual FP32 record producer. A local ring retains only accepted records,
poisons rejected suffixes, wraps absolute positions and repeatedly resets the
checkpoint. Recovery replays the retained records through the existing kernel;
its reconstructed endpoint seeds the next verify window. The test includes
zero acceptance and keeps the existing output and strict main-state tolerances.
Convolution uses independent candidate storage, the actual raw-input capture
and production convolution commit; its accepted endpoints must match main
exactly.

The harness gathers ring rows into the current contiguous recovery interface.
It does not implement or validate production generation/position stamps, slot
reuse or direct ring addressing.

The required general record interface remains a PR1 acceptance gap under
Section 9.1 of the design. The production planner also currently launches per
cache group, rather than once globally. Neither gap is deferred to PR2 by this
validation change. The B32/T4/A1 regression, independent-start timings and
real-model gates remain open.

### Resumed correctness results

The four suites pass on one GB300 using the existing CUDA 13 container,
PyTorch 2.14.0+cu130 and tokenspeed-triton 3.8.10.post20260920:

- Benchmark API: **12 passed**.
- RecoverSSM kernel and composition: **20 passed**.
- KDA runtime eager/graph commit: **20 passed**.
- Existing replay kernels: **33 passed, 1 platform-specific skip**.

Across the seven 128-round composition cases, maximum accepted-state error
versus main is **3.5762787e-7**, state RMS is at most **1.0816872e-9**, and
maximum BF16 verify-output error is **3.0517578e-5**. Independent convolution
endpoints match exactly. No numerical threshold was changed.

The runtime/kernel implementation is `c1208b87`; the composition test changes
are uncommitted. All suites checked the same Python source manifest before and
after execution. Its SHA-256 is
`d98012b3639415a5dbfca3b58ff8b576ee7a2b39e5e03d242837e34df477d993`.
The patch, per-suite logs and manifest are retained in local ignored
`outputs/pr1/composition-validation/`. Full repository hooks pass. These are
focused attention/runtime checks, not real-model, AIME or E2E results.

### Resumed low-acceptance timing

A diagnostic B32/T4/A1 run uses the same GB300 and environment as the tests,
with 69 KDA layers, H12/D128 and CUDA graphs. Four fresh processes run in
main/candidate/candidate/main order, with 50 CUDA-event samples per process.
Main remains unchanged at `f4ac1aff`; candidate runtime is `c1208b87`.

| Implementation | Process medians |
| --- | ---: |
| Main | 2284.240–2284.272 us |
| RecoverSSM | 2362.096–2362.192 us |

The mean of candidate process medians is **3.41% slower** than main.
This reproduces the low-acceptance regression after the review fixes. It is
not a full independent-start acceptance protocol or an E2E measurement. No
kernel change or performance improvement is claimed. Raw samples and the
exact launcher are retained beside the correctness logs. The source manifest
remained unchanged through all four runs.


## Stamped record contract and shared planning

The next implementation increment uses the same stamped record view in verify
and recovery. INT64 metadata supplies the physical slot, generation and
absolute checkpoint position. INT64 stamps identify each record's absolute
position and generation. Position `i` maps to `i % R`. Recovery checks the
accepted interval before reading payloads; a stale position, generation or
invalid slot raises a device assertion. Explicit unstamped calls remain only
for standalone kernel diagnostics. Serving always supplies stamps.

PR1 still uses per-round workspace with `R=T_max`. Each metadata refresh assigns
a new generation. It does not add persistent request ownership or change LCM
and scheduler semantics. The composition harness now calls the actual producer
and recovery directly on R8/R16/R32 rings, without gathering records. It covers
128 rounds, T1/T4, all accepted lengths, rejected payload poisoning, repeated
checkpoint resets and physical slot reassignment to new requests.

Commit planning now resolves destination pages and copies the original
source-page snapshots for all cache groups in one kernel launch. The runtime
no longer launches `torch.stack` before recovery. The GPU commit sequence is
one shared planner, one all-layer recurrent recovery and one convolution commit.
Descriptor, page, accepted-count, record-metadata and stamp buffers are allocated
before capture and included in startup workspace accounting. Descriptor copies
occur only when table addresses or geometry change. Rebinding clears this cache.

Focused GPU validation of this increment:

- RecoverSSM kernels: **26 passed**, including direct ring composition,
  stale-record rejection in isolated processes, grouped planning and graph replay.
- Runtime integration: **20 passed**, including padding, multiple rounds,
  pool rebinding and exact planned-versus-allocated workspace bytes.

The runtime tests found a stamp-view shape mismatch for live batches below
maximum capacity. Verify now receives the live-batch stamp view; all-layer
commit keeps the full fixed-capacity layer stride. The corrected tests pass.
Kernel compilation tests also pass after slot count became a runtime argument.

Local logs are `outputs/pr1/stamped-validation.log` and
`outputs/pr1/stamped-runtime.log`. These results do not satisfy the full-model
or performance gates. The prior B32/T4/A1 regression still requires a fresh
measurement of this implementation, followed by the prescribed matrix.


The replaced all-layer replay implementation and its gate producers have moved
from the runtime package to `tokenspeed-kernel/test/kda_replay_reference.py`.
The differential tests keep this independent main oracle. Shared per-layer
kernels used by other serving paths remain unchanged.

### Stamped-path performance investigation

A fresh paired diagnostic uses the same GB300, PyTorch 2.14.0+cu130 and
Triton 3.9.10.post20261008 for both arms. This dependency version differs from
the earlier diagnostic, so its main timing was measured again.
At B32/T4/A1, main takes **2283.264 us** and the first stamped implementation
takes **2460.592 us**. Moving stamp writes before the recurrent-state load and
removing the source-page stack lowers the candidate to **2435.776 us**,
still **6.68% slower** than the paired main diagnostic.

A recovery tile sweep over BV8/16/32/64 and one/two/four warps does not close
the gap. Verify BV16 with one warp remains the fastest tested geometry.
These measurements are diagnostic, not the prescribed independent-start
statistical gate. They do not justify marking PR1 performance complete or
starting final design-document removal. Full TP8 model, agentic, AIME 2026
and E2E matrix validation remain uncompleted.


### Frozen-source verification

The final A/B/B/A diagnostic uses four fresh processes, 50 CUDA-event samples
per process and the same environment for both arms. Main process medians are
**2283.632–2284.288 us**; candidate medians are
**2435.568–2435.840 us**. The ratio of mean process medians is
**1.06644**, or **6.64% slower**.
This remains a diagnostic and does not replace the formal acceptance protocol.
No performance gate is marked complete.

All Python sources remained unchanged across the four processes. The manifest
SHA-256 is `5ff1e13acd4ac518946ef9b793486be0415753ceb53b29c7dc44b9c3a0d43393`.
The manifest, patch, exact launchers and raw timing samples are under local
ignored `outputs/pr1/final-stamped/`. The same manifest guards the final
kernel, runtime and benchmark API validation scripts.

Before the frozen-source rerun, the expanded cleanup suites reported
**113 passed, 24 skipped**, runtime **20 passed**, and benchmark API
**12 passed**. The four relocated reference functions retain the original
AST. The full `pre-commit run --all-files` command passes.


### Corrected timing with disjoint live pages

The benchmark now assigns different parent IDs to each simultaneously live
cache group. Before timing, it checks capacity, per-page contiguity and pairwise
disjoint byte ranges for every live recurrent and convolution state tensor.
B32 uses 96 live parent IDs and an arena with 104 usable parents. This replaces
the invalid setup that reused 32 IDs across all three state groups.

The corrected A/B/B/A run measures main at **2481.904–2481.920 us**
and candidate at **2440.928–2442.896 us** for B32/T4/A1.
The ratio of mean process medians is **0.98388**,
or **1.61% faster** for the candidate.
This is still a diagnostic. The prior regression claim is withdrawn; the
complete matrix and independent-start statistics remain required.

Corrected raw measurements and launchers are in
`outputs/pr1/disjoint-stamped/`. The earlier timing directory contains an
invalidation notice. The frozen-source correctness rerun completed with
**145 passed, 24 skipped**, and its source manifest remained unchanged.


### Independent-process matrix and full-model validation

The corrected harness completed ten paired starts with all 20 cells in each
process. Eighteen cells pass the simultaneous confidence-interval rule.
B8/T1 and B16/T1 fail: their latency-ratio intervals lie above 1.0, despite
remaining within the 2% uncertainty margin. These failures remain recorded in
`outputs/pr1/disjoint-stamped/matrix-statistics-10.json`.

A separate T1 A/B/B/A diagnostic found that preceding T4 cells change the
absolute timing substantially. The B8 candidate difference decreases from
about +0.13% to -0.01% when each cell has its own process. B16 decreases from
about +0.06% to +0.01%. This diagnosis does not invalidate or pass the failed
cells. A new protocol repeats all 20 cells, with ten paired fresh processes
per cell. It does not rerun only the failed cells. Its protocol and complete
raw records are retained under `outputs/pr1/isolated-matrix/`.

Full-model validation uses real Kimi-K3 NVFP4 weights at TP8. One startup
failed before inference because the worktrees lacked native CUDA libraries.
Both worktrees now reference the same libraries, with identical source files
and recorded binary hashes. This was an environment failure, not a completed
correctness test. Startup and validation are being repeated.

The AIME 2026 corpus is fixed to `math-ai/aime26` revision
`79037aebdb6580008fb960d17cb21fd3099083e3`, with 30 questions, identical tokenized
prompts, greedy sampling and a fixed final-box integer parser. No AIME result
is claimed until both arms complete. The deterministic serving corpus checks
exact output tokens and per-round acceptance with the existing host logging
option. Detailed acceptance logging is disabled for performance measurements.

The E2E protocol uses ten independent starts per arm, alternating order within
matched pairs, C4/8/16/32, and the same graph settings and continuation inputs.
Median and p99 TPOT and throughput share simultaneous 95% intervals across
all 12 gated metrics. Diagnostic and correctness starts are not timing samples.
No full-model or E2E gate is marked complete at this stage.


### Packed-table descriptor reuse

A real TP8 trace of `43b59d91` found a descriptor upload on every decode
round. The scheduler allocates new packed page-table storage each round.
Absolute table pointers therefore changed even when the table layout did not.
Across 52 rounds, the trace contains 52 pageable 96-byte host-to-device copies
and 52 corresponding device-to-device copies.

Commit `60eaa6d1` stores table offsets relative to the first table and passes
the current base tensor to the shared planner. The descriptor cache now
depends on layout and source-page vectors. A runtime regression test moves
the packed storage, checks that no descriptor upload occurs, and verifies
source pages, destination pages and accepted counts. It fails on the prior
implementation and passes on this commit.

Frozen-source validation passes **146 tests, with 24 skipped**: 113 kernel
tests, 21 runtime tests and 12 benchmark API tests. Full repository hooks
pass. The source manifest SHA-256 is
`b02ea7c8d32972230b60b276acc25c1fd37cde404c5eacd68eb66f051f16225f`.

The earlier isolated-cell matrix stopped before completion when the real-model
trace exposed this issue. Its partial data is retained and is not acceptance
evidence. The final matrix starts again with the new commit, ten independent
paired processes per cell, and all 20 cells. Both source manifests are checked
before and after every process.

The baseline AIME run completed with **29/30 correct and zero HTTP errors**.
The earlier candidate also scored 29/30, but that result belongs to `43b59d91`.
The final candidate must complete its own run with the same frozen prompts,
generation limit and parser before the paired accuracy gate can pass.

A separate B32/T4/A1 compiler-resource diagnostic on `60eaa6d1` reports no
spill slots in the eight captured Triton kernels. Verify uses 165 registers
per thread and recurrent commit uses 96. CUDA driver occupancy limits are
18.75% and 31.25%, respectively; these are theoretical limits, not achieved
occupancy measurements. Full component profiling and the E2E gates remain open.


The final TP8 startup reports a 256 MiB execution workspace per GPU in both
arms. At the fixed 0.80 memory-utilization setting, main allocates 800 LCM
parents (17,716,838,400 cache bytes), while PR1 allocates 790
(17,495,654,400 cache bytes). This ten-parent capacity difference is retained
in the comparison. Both use the configured maximum concurrency of 32; the
complete serving matrix must still validate that concurrency.


### Final-source AIME and commit trace

The paired AIME gate passes for `60eaa6d1`: both main and PR1 score
**29/30**, with zero HTTP errors. All 30 questions use the same frozen
prompts, sampling settings, token limit and final-box parser. Per-question
results are retained. The incorrect questions differ: main misses question
15, while PR1 has no parsed answer for question 6. This passes the declared
count-based accuracy gate; it does not establish identical token sequences.
The separate deterministic serving comparisons remain required.

A TP0 diagnostic trace of the final candidate contains 52 decode rounds:
3,588 layer verify calls and 52 calls each to the shared planner, recurrent
recovery and convolution commit. There are two 96-byte pageable host-to-device
copies and two 96-byte device-to-device copies across the entire trace, down
from 52 of each on `43b59d91`. The trace includes prefill and decode; it does
not claim zero descriptor refreshes when table geometry changes. Raw traces
for all eight ranks are retained. Profiled durations are not performance-gate
samples.


### Exact serving comparison: failed gate

The MTP3 CUDA-graph comparison fails on all five frozen serial corpus cases:
output token IDs and per-round acceptance counts differ from main. The first
output difference follows 31 to 98 matching tokens. The ten-turn agentic replay
also diverges. These failures are retained; the AIME score does not override
them. Standard-decode, eager and E2E performance acceptance remain pending.

The unchanged-baseline controls pass. All five corpus cases have identical
output token IDs and acceptance counts across independent server starts and
within one server. The serving mismatch is therefore a PR1 regression under
the tested configuration. The numerical-envelope
document explicitly excludes quantized checkpoints from `rl-bitwise`; switching
that flag is not a supported fix for this NVFP4 setup. No prompt, threshold or
model configuration has been changed in response to the failure.


A separate diagnostic worktree now checks arithmetic boundaries without
changing the frozen performance source. It preserves main's BF16 gate
projection and uses distinct reduction layouts for verify and FP32 replay.
The diagnostic gates, replay decay, correction and accepted recurrent state
match the reference bit for bit over 128 rounds at B1, B4 and B16. Two missing
contractions caused the state drift: the local projection reduction and the
SiLU multiplication fused with correction subtraction. Record production now
receives pre-SiLU V and preserves both contractions.

Expanded diagnostics found one near-zero verify output mismatch in each of
B2 and B3, with maximum absolute error 1.1641532182693481e-10. The existing
kernel-output tolerances remain unchanged. The accepted-state comparison is
strengthened to exact equality. Full-model token and acceptance gates remain
exact. The regression suite now passes; the serving gates remain pending.
Performance measurements of `60eaa6d1` cannot validate subsequent numerical
fixes.


The complete isolated-cell attention matrix for `60eaa6d1` passes all 20
cells with ten independent pairs per cell. All 400 process results are
retained. This performance result does not override the exact serving
failure. The numerical fix starts a new frozen-source matrix under the same
protocol; no earlier timing samples enter its acceptance statistics.


### Frozen arithmetic correction

Signed commit `694e7bc5` preserves main's arithmetic boundaries. Full repository
hooks pass. Validation reports **161 passed, 24 skipped**: 128 kernel tests,
21 runtime tests and 12 benchmark API tests. All 12 long-round composition
cases have zero accepted-state error. The added gate tests cover both one
recurrent layer and the complete 69-layer tiling calculation.

The source manifest SHA-256 is
`77ad825db3c0d4ddfae504b3ebfc612a5f243da9e701b0245715d4f74d2aec69`.
The correction remains isolated in a clean worktree. Its MTP3 CUDA-graph
serving comparison also fails all five corpus cases and all ten agentic turns.
The five corpus cases match only 4, 29, 31, 25 and 27 initial output tokens.
The new attention matrix stopped after 70 of 400 processes. These partial
measurements do not provide acceptance statistics. Follow-up correctness and
E2E controllers stopped on the serving failure.

The next diagnostic captures the unchanged baseline's first eager verify
window and accepted state. It will compare actual model inputs layer by layer
against the frozen correction. Instrumented capture runs are not acceptance
evidence. Scheduler page zero is explicitly reserved as the kernel null page;
inspection does not support a null-page convention mismatch.


### Actual-input verify diagnosis

A diagnostic eager baseline capture retained the first verify window and
accepted endpoint for all 69 recurrent layers. Replaying the original verify
kernel reproduces every captured output exactly. Source `694e7bc5` produces
identical convolution, gate and accepted-state values, but differs in 12 BF16
verify elements, with maximum absolute error `4.76837158203125e-7`.

A separate correction preserves the four-warp tile's local projection FMA
operand order and token loop. The original one-warp expansion remains in
place. The captured first-window outputs and accepted states then match
exactly. Independent 128-round diagnostics at B1/2/3/4/8/16/32 also report
zero output and accepted-state error. These are diagnostic results. Full
repository checks, frozen-source suites and real-model gates must still run
before this correction can be integrated.


### Frozen verify correction

Signed commit `65b82d94` passes all repository hooks and 161 tests, with
24 platform skips. The final source manifest is
`c735f43cd8f72b082a93d5d9a3e743a2e365a04b5cbbd993101c2dda48165571`.
Its actual-input replay has zero verify-output, accepted-state, convolution
and gate error across all 69 captured layers.

The formal MTP3 CUDA-graph corpus now passes all five cases. Every output
token and per-round accepted count matches the recorded baseline. The paired
round counts are 79, 67, 99, 76 and 95. Agentic, standard-decode, eager, AIME
and performance gates are still in progress. The correction remains isolated
until all required validation completes.

The ten-turn MTP3 CUDA-graph agentic comparison also passes. Inputs, output
tokens, accepted counts and cached-token counts match on every turn. The
initial corrected server completed normally. Standard-decode, eager, AIME
and the complete performance gates remain pending.

The paired standard-decode CUDA-graph comparison passes all five corpus
cases and all ten agentic turns. Output tokens, input tokens and cached-token
counts match. Both servers completed normally. Fresh MTP3 eager comparisons
are in progress; AIME and performance acceptance remain open.

MTP3 eager also passes all five corpus cases and all ten agentic turns.
Output tokens and per-round accepted counts match exactly. Agentic input
and cached-token counts also match. Both eager servers completed normally.
All three required serving configurations now pass the exact comparisons.
The candidate AIME evaluation and complete performance gates remain open.

### Verify-correction performance failure

The complete frozen-source attention matrix for `65b82d94` contains ten
independent matched pairs for all 20 cells (400 processes). All four T=1
cells pass. All 16 T=4 cells fail with significant latency regressions:
ratios range from 1.068225 to 1.513437. No second batch is allowed for a
significant failure. Formal E2E measurements have not started.

Component profiling continues to locate the regression. A separate worktree
will preserve the corrected arithmetic while testing layout and resource
changes. The frozen source and all failed measurements remain unchanged.
The AIME evaluation continues independently.

The frozen `65b82d94` AIME evaluation passes the paired count gate: main
and candidate each answer 29 of 30 questions correctly. All required
full-model correctness comparisons pass for this source. Its attention
performance failure still prevents acceptance and formal E2E timing.

### Producer layout optimization

Signed candidate `27c1dbe3` preserves the corrected replay arithmetic while
reducing producer overhead. Packed value rows reduce register use. Replay
loads the checkpoint directly and converts only projection partials for the
required reduction order. Large-row gate production combines the two required
dot accumulations in one launch. Small-batch verification stays unchanged.

All repository hooks pass. Frozen-source validation passes 207 tests, with
24 platform skips. All 69 actual-input layers and 276 actual-weight gate
cases match exactly. The tracked Python source manifest is
`d27aab33c4afc6776a2a4563f34b950c99178d862d4be1f3b91bbb461e9dd0eb`.

Fresh full-model correctness and the complete attention matrix are running.
Diagnostic timings are excluded from acceptance statistics. Formal E2E
timing must wait for both gates. The candidate remains isolated.

The frozen `27c1dbe3` MTP3 CUDA-graph comparison passes all five corpus
cases and all ten agentic turns. Output tokens, per-round accepted counts,
input counts and cached-token counts match the baseline. Standard-decode
validation has started. Attention statistics and E2E acceptance remain open.

Standard-decode CUDA-graph validation for `27c1dbe3` passes all five corpus
cases and all ten agentic turns. Output tokens, input counts and cached-token
counts match. MTP3 eager validation is now running.

MTP3 eager validation for `27c1dbe3` passes all five corpus cases and all ten
agentic turns. Output tokens and per-round accepted counts match exactly.
Agentic input and cached-token counts also match. All three serving
configurations pass their complete comparisons. AIME and the commit trace
are now running; performance acceptance remains pending.

The frozen `27c1dbe3` AIME evaluation passes the paired count gate. Main and
candidate each answer 29 of 30 questions correctly, with zero HTTP errors.
Main misses question 15. The candidate answers question 15 correctly but
has no parsed final answer for question 28. This is a count-based accuracy
pass, not identical AIME token sequences.

The TP0 diagnostic trace contains 55 decode rounds, 3,795 verify launches and
55 launches each for shared planning, recurrent recovery and convolution
commit. Record metadata preparation also runs 55 times. The complete trace
contains one pageable H2D and one D2D copy of 96 bytes; these copies do not
repeat each round. All formal full-model correctness gates now pass.
The complete attention matrix and formal E2E measurements remain pending.

### Frozen layout-candidate timing result

The complete `27c1dbe3` matrix contains ten independent matched pairs for
all 20 cells (400 processes). All 16 T4 cells pass, with latency ratios from
0.755111 to 0.977715. Three T1 cells pass. B4/T1 fails the significant
regression rule: ratio 1.000033330, simultaneous interval
[1.000012136, 1.000054524]. This remains a failure despite its small size.
No second acceptance batch is eligible. Formal E2E timing has not started.

Seven B4/T1 paired medians differ by one 32 ns median step; three pairs tie.
The compiler hashes and resources are identical between arms. Nsight records
345 identical fused-decode kernel calls per arm. A fixed ten-pair A/A
calibration now checks the original timing method using one unchanged source.
Calibration samples are diagnostic only; the failed matrix remains intact.

### Timing-resolution calibration

The complete B4/T1 Nsight records contain identical kernel sequences, launch
geometry, resources and copy sequences: 345 kernels and 345 copies per arm.
A fixed ten-pair A/A control on unchanged main produces two 32 ns differences
and eight ties. Its interval includes 1; this control does not fail. The
individual event times have 32 ns granularity.

Before collecting longer-window data, the calibration fixed 64 consecutive
graph replays per event window and 50 windows per process. State resets before
each window and persists through its 64 replays. Divide each raw window by 64,
then use the process median. This improves per-replay event-step resolution
to 0.5 ns. All raw window times and normalized observations are retained.

The ten-pair A/A calibration for this fixed method gives ratio 0.9999955013
and interval [0.9999585714, 1.0000324325], including 1. Both apparent speedups
and slowdowns matter for this two-sided calibration check.

A fresh complete 400-process attention campaign is now registered in
`outputs/pr1/window64-matrix/PROTOCOL.md`. It uses the same source, cells,
independent-start count, bootstrap and regression rules. It reuses no timing
from the original failed matrix or the calibration. The original failure
remains recorded. Source-identical component profiles and the full-model
correctness certificate remain valid. E2E data collection has not started.


### Complete calibrated attention result

The fresh 400-process matrix passes all 20 cells with ten independent paired
starts. No second batch is needed. All four T1 simultaneous intervals contain
1. The sixteen T4 latency ratios range from 0.755991 to 0.978508, corresponding
to 2.15% through 24.40% lower latency. The B4/T1 ratio is 0.999984958 with
interval [0.999933734, 1.000036185]. Statistics and failure thresholds are
unchanged. The original single-event failure and both A/A controls remain
recorded; they do not enter these acceptance statistics.

All 40 component profiles and both occupancy reports describe the same frozen
source. At B32/T4/A1, verify uses 121 registers per thread with 22.24% achieved
occupancy; recovery uses 96 registers with 29.47% achieved occupancy. Both
kernels have zero spills. Full-model correctness remains passed. Memory
reporting and the formal E2E matrix are the remaining validation work.


### Memory inventory and E2E placement

The B32/T4/A4 synthetic 69-layer memory inventory is complete for both arms.
Baseline reserves 199,179,264 bytes and allocates 5,796 additional descriptor
bytes outside that return value. The candidate reserves its complete
305,382,788-byte inventory. Both include 61,046,784 convolution workspace
bytes. Eager peak allocated memory above persistent tensors is 29,884,416
bytes for baseline and 34,603,008 bytes for the candidate. Capture has the same
extra peaks. Graph replay adds zero PyTorch allocated bytes in either arm.
These figures exclude driver/context memory and full-model allocations.

The first inventory check incorrectly assumed baseline already budgeted its
descriptors. That report-only check was corrected after inspecting both
preallocators. The failed check and corrected results remain recorded.
No candidate source or performance data changed.

Before collecting any E2E samples, both original serving nodes became busy.
The pending allocation was cancelled before it ran. The full E2E campaign
now uses a fixed pair of idle GB300 nodes in one NVL72 group, with the same
node pair for both arms and subsequent lease renewals. Only harness hostnames
changed; source, dependencies, workload, server flags and statistics did not.
The original harness and client manifest are preserved. Placement preflight
checks both devices and software versions before the first server start.


### E2E request-arrival diagnostic

The first formal baseline start stopped after six C4 batches and the first
C8 warmup batch. At C8, seven requests reported 51,840 cached tokens instead
of the required 51,328; one retained the required parent hit. Identical
continuations can reuse pages published by earlier requests in the same
batch when arrivals are staggered. All requests and the failed start remain
recorded. No candidate performance start ran, and there is no valid paired
performance result.

A separate baseline diagnostic compares preencoded concurrent requests and
the batch API, with three repeats at each C4/8/16/32 condition. It checks
cache hits without changing model source, prompts, output limits or statistical
thresholds. Diagnostic samples never enter acceptance statistics.


### Fixed-cache E2E campaign

Three baseline diagnostic repeats pass the original hit count at C4/8/16
when JSON is preencoded. C32 still reports extra prefix hits in all three
repeats. The batch API returns HTTP 400 for the tested batched input.
Admission-time prefix matching and the 8,192-token prefill budget explain
why simultaneous arrival alone cannot fix all four concurrency conditions.

A separate fixed maximum-prefix control passes all 12 conditions: three
repeats at each C4/8/16/32. Prime the first 51,841 tokens of the unchanged
51,936-token continuation, then require exactly 51,840 cached tokens for
every request. That is the final complete prefix page, so later admissions
cannot reuse a longer prompt prefix. JSON bodies and hashes are prepared
before the batch timer. Server settings and actual continuation tokens do
not change. This explicitly changes the priming boundary from 51,328 to
51,840; the original campaign remains failed, not retroactively passed.

A new complete campaign is registered in
`outputs/pr1/tp8/FIXEDCACHE-E2E-PROTOCOL.md`. It collects ten new paired
independent starts, with the same alternation, three rounds, one warmup and
five measured batches per round and concurrency, 256 output tokens, bootstrap
and regression thresholds. The complete harness is hash-guarded. Original
and diagnostic samples are excluded. The scope is one real-content agentic
continuation with a fixed near-full prefix hit, not all cache-hit lengths.

### Frozen candidate: fixed-cache E2E result

All ten paired independent starts completed. The fixed-cache invariant held
for every recorded request. The predeclared joint bootstrap reports three
passes, five inconclusive metrics and four failures. The failed metrics are:

| Concurrency | Metric | Candidate/main ratio | Simultaneous 95% interval |
| --- | --- | --- | --- |
| 8 | Mean throughput | 0.958973 | [0.929559, 0.989318] |
| 16 | Median TPOT | 1.045993 | [1.030554, 1.061663] |
| 32 | p99 TPOT | 1.098598 | [1.004012, 1.202094] |
| 32 | Mean throughput | 0.977379 | [0.955869, 0.999373] |

This is a failed acceptance gate. A second batch is not eligible because
there are significant regressions. Keep all raw measurements and the full
report at `outputs/pr1/tp8/fixedcache-e2e-statistics-10.json`.

The resource report covers all starts: baseline cache capacity is 801 parent
blocks; candidate capacity is 792. Both arms have zero measured preemptions.
GPU memory at readiness is 229717 MiB per GPU for baseline and 229633 MiB for
candidate; these are readiness snapshots, not runtime peaks. The candidate's
larger persistent replay workspace is accounted for before cache sizing.

Output sequences and acceptance distributions differ at concurrent batches.
Diagnostic grouping by identical output sequence still shows additional
latency in some cells. This grouping is not an acceptance statistic and does
not exclude any sample from the failed gate. Collect paired CPU/GPU traces
before assigning the cause. Keep the tested candidate frozen and make any
fix in a separate worktree. Final integration and documentation cleanup remain
pending.

### Exact verify reduction candidate

Frozen candidate `9b24cc49e5a115e08ef7bd52dc06de090d4aeece` preserves
main's verify normalization and rounded-product reduction at concurrent
batches. It keeps packed value rows and explicitly reduces four adjacent
products before the 32 partial sums. The non-128-dimensional prepared-QK
path retains its general reduction; seven boundary cases cover that path.

All 12 multi-round producer cases now require bitwise verify equality at
every batch size, with no relaxation of accepted-state equality. Complete
validation reports 214 passed and 24 platform skips. All 69 actual-input
layers and 276 actual gate cases are exact. The full repository pre-commit
command passes. The signed source is clean and frozen.

Eight independent tuning cases at B4/8/16/32 and A1/4 report latency ratios
0.77122 through 0.99441. These are diagnostic observations, not acceptance
statistics. The earlier simple-layout and larger-tile experiments are also
retained under `outputs/pr1/e2e-fix/`.

The new full-model correctness campaign and complete source-specific
attention campaign have started on separate GPU allocations. The attention
protocol is `outputs/pr1/exact-reduction-matrix/PROTOCOL.md`. The new E2E
protocol preserves the fixed-cache workload and original statistical gates.
It requires fresh paired starts and must not reuse source27 timing samples.
The candidate is not yet accepted or integrated into the PR1 branch.

## Exact reduction candidate: completed acceptance campaign

Candidate `9b24cc49` passes the five fixed-input and ten-turn agentic checks
in MTP3 graph, standard graph and MTP3 eager configurations. Output tokens
and acceptance counts match the corresponding baseline exactly. AIME remains
29/30 for both versions; generated tokens are not required to match in that
task. The TP0 trace has 41 rounds and 41 launches of each commit component.

All 20 attention settings pass across 400 fresh processes. Each T=1 interval
contains one. T=4 latency ratios range from 0.76964 to 0.99518. Forty component
traces and two occupancy reports are complete. These measurements exclude
prefill, model projections, sampling and communication.

The full ten-pair E2E result is **FAIL**, with one failed metric, three passed
metrics and eight inconclusive metrics. C=8 throughput is 5.90% lower; the upper
bound of its simultaneous confidence interval is below one. All starts completed,
all requests had 51,840 cached tokens, and no measured request was preempted.
No failed sample was replaced. No extra batch is eligible under the protocol.

At C=8, mean reported acceptance length is 3.60755 for main and 3.49061 for
the candidate. Mean TTFT is similar. Output-sequence frequencies differ, and
requests with the most common shared output also have longer mean candidate
TPOT. These are diagnostic observations, not proof of a cause. Separate C=8
CPU/GPU traces are complete. Both contain 49 rounds; candidate KDA verify
and recovery are faster in these traces, while the whole TP0 GPU span is
slightly longer. This is diagnostic evidence, not an acceptance override. Formal statistics
and all raw requests are retained under `outputs/pr1/tp8/` with the
`exact-reduction` prefix. Integration, cleanup and publication remain pending.

Admission analysis retains all 150 C=8 measured batches per arm. The 6+2
prefill pattern occurs 20 times in main and 78 times in the candidate; 5+3
occurs 85 and 42 times. Both implementations produce different first tokens
for the two-request group in the observed 6+2 batches. Candidate mean batch
time is lower within both dominant patterns. These conditional observations
do not prove why admission differs and cannot reverse the failed gate.
The bounded next investigation is recorded in the local diagnostic report.
All owned GPU allocations are released; no tests remain running.

## Fixed-send-order C8 diagnostic

At the user's request, the PR1 branch was fast-forwarded to `9b24cc49` and pushed to `byshiue`. This does not change the failed E2E acceptance verdict.

Three fresh-start pairs used one client thread to send requests 0 through 7 in order, retaining eight outstanding requests. Each arm completed 45 measured batches, 360 requests, with exact cache checks and no preemptions. Mean accepted length was 3.63208333 for baseline and 3.51652778 for candidate. Mean acceptance rate was 87.75083333% versus 83.92608333%, a difference of -3.82475 percentage points.

Shared prefill groups had identical reported means across arms: 5+3 gave length 3.65 and rate 88.36375%; 6+2 gave length 3.3925 and rate 79.80250%. The candidate had 22 of 45 batches in 6+2, versus 5 of 45 for baseline. Fixed client send order did not fix engine receive order or prefill grouping. This diagnostic supports further investigation of admission and batch-sensitive prefill behavior; it does not establish a fused-kernel arithmetic defect or override the E2E gate.

Local evidence: `outputs/pr1/tp8/ordered-c8-report.md`, `ordered-c8-summary.json`, and `ordered-c8-request-comparison.csv`. All owned GPU resources were released after the diagnostic.

### Completed C8 ingress-boundary diagnostic

One fresh instrumented start per immutable arm, 15 C8 batches / 120 measured
requests each. Exact request groups match across IPC drain, TP broadcast,
request-handler return, scheduler submission and prefill in all 30 batches.
All eight requests are physically received before first scheduler submission
in 13/30 batches, but the prefetched next snapshot is only submitted in the
next round. Thus physical receipt does not mean current scheduler visibility.

The instrumented pair has 6+2 in 4/15 batches for each arm; the previous
uninstrumented 5/45 vs 22/45 frequency difference does not reproduce. More
three-way groups appear with instrumentation. All five shared grouping
patterns have matching mean reported acceptance statistics across arms.
This confirms the grouping mechanism but does not identify the cause of the
prior frequency shift. Further timing isolation is needed before attributing
it to a kernel. No scheduler policy or model code was changed.

All case/source checks completed; GPU resources were released. Evidence:
`outputs/pr1/tp8/ingress-c8-findings.md` and `ingress-c8-analysis.json`.
Original E2E C8 failure remains unchanged; this is diagnostic evidence only.

### Prime overlap explains the C8 ingress gap

A reverse-order instrumented pair (candidate then baseline), 15 measured C8
batches per arm, confirms that every batch waits for a queued decode of the
already-finished one-token prime request. The extra decode was dispatched
before final prime prefill completion in all 30 batches. Its commit occupies
98.19% (candidate) / 99.23% (baseline) of the approximately 18 ms ingress gap.
A native scheduler 0.1.27 CPU replay independently confirms scheduling ahead
of the finish event even when max_new_tokens=1.

Candidate frontend IPC begins 1.410 ms earlier relative to prime completion;
its eight-request IPC span is 1.038 ms shorter. Together with the measured
commit phase, this gives arrivals 1.606 ms more head start before receiving
resumes. The current pair has 6+2 in 12/15 candidate vs 0/15 baseline batches.
The grouping boundary remains identical from IPC drain through prefill in
all 30 batches. This locates an overlap/arrival-timing mechanism. The precise
host-side source and historical group-frequency effect remain unisolated,
particularly because the earlier instrumented pair had 4/15 vs 4/15.

The prime HTTP response is not an engine-idle barrier. A further controlled
diagnostic can separate C8 arrival from prime's unfinished overlapping work.
No production policy/code changed; original E2E C8 failure remains unchanged.
Resources released; source/native and harness hashes passed. Evidence:
`outputs/pr1/tp8/commit-c8-findings.md`, `commit-c8-wait-analysis.json`, and
`replay_one_token_overlap.json`.

### Controlled C8 admission aligns both implementations

Two fresh-start pairs used reversed arm order, a prime-drain barrier, fixed
logical request order, and fixed prefill groups 5+3, 6+2 and 8. Each arm ran
30 measured batches / 240 requests, plus separate warmups. Every input/cache
check passed, all samples were retained, and no preemptions occurred.

All 240 paired requests have exactly matching final output tokens and live
per-round accepted-width sequences. All 30 paired batches have matching
prefill/decode membership, input lengths and CUDA-graph shape sequences.
Both arms report mean accepted length/rate of 3.65/88.36375% for 5+3,
3.3925/79.80250% for 6+2, and 3.70/89.86% for 8. Per-request differences
are zero in this controlled workload.

From first prefill dispatch to last live commit, pooled candidate throughput
changes are +0.407% (5+3), -0.437% (6+2), and +0.597% (8). Each pair has
the same sign per group. These are descriptive controlled-execution timings
with trace overhead, not ordinary HTTP latency or a new E2E acceptance gate.
Two fresh pairs do not establish a population-level performance claim.

This supports an admission/group-frequency explanation of the earlier
aggregate acceptance difference, with no candidate-only arithmetic mismatch
observed here. It does not prove universal bitwise equivalence or isolate
the precise host-side cause of the historical arrival-timing shift. No
production scheduler or kernel code changed. Original natural-arrival C8
E2E failure remains unchanged. Source/native/harness checks passed, the
controller exited successfully, and owned GPU resources were released.

Evidence: `outputs/pr1/tp8/ALIGNED-C8-PROTOCOL.md`, `aligned-c8-findings.md`,
`aligned-c8-analysis.json`, and `aligned-c8-request-comparison.csv`.

### PR1 unit-test scope cleanup

At the user's request, remove PR1 API-signature/argument checks, the touched
mock dispatch-trait checks, invalid-record rejection tests and their helper,
and the compilation-reuse-only test. Remove the exception-only tail from
the padded graph test. The commit-plan test now compares resolved page and
step values without checking pointer changes or spying on descriptor copies.
Numerical output, recovery-state, padding, graph replay and multi-round
correctness comparisons remain. Runtime validation and kernel code are
unchanged. This cleanup is limited to PR1-specific or PR1-updated coverage;
pre-existing unrelated tests remain.

All four edited Python test files pass syntax compilation; the diff passes
whitespace checks. Local pytest could not start because the environment lacks
`typing_extensions`. GPU correctness tests were not rerun for this test-only
cleanup. The earlier GPU results describe the pre-cleanup test suite.
