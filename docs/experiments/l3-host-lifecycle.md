# L3 Host lifecycle implementation and validation

Baseline: `057a4daa` (the cache/scheduler implementation is unchanged from
`c50a2f09528b8e13c1d9eeff287e2e74b1ed92b6`). This covers the
architecture refactor requested in issue #1793 and safe expiry of queued
prefetches. It is a component validation report, not a production release.

## Changes

- `L3CacheEngine` owns Host-only Store I/O, background lanes, namespace
  changes and retirement. `HiCacheEngine` coordinates completed D2H metadata
  with PUT completion; failed PUT still triggers replica-coordinated failure.
- `DeviceHandle` retains GPU transfers. L3 queries, progress and namespace
  operations go directly to the Host-only engine. Its CPU buffer view does
  not retain the Device layout.
- The prefetch budget starts at submission. A queued future can return a
  zero-page result only when cancellation proves its worker never started.
  An active native call retains its destinations until it returns.

## Evidence

- Native C++ scheduler: 618 tests passed.
- Scheduler Python bindings built from this checkout: 279 tests passed.
- Host/control tests: 113 passed, including actual two-process gloo convergence for a
  rank-local lookup failure and different landed-prefix lengths.
- Mocked Host-copy orchestration: 23 passed, 9 CUDA tests skipped,
  10 subtests passed. Optional GPU imports are substituted; kernel execution
  is prohibited. This is not CUDA-copy validation.
- Repeated Host experiment: 30 / 30 runs passed,
  using two independent Host engines and `MemoryKvStore` with a gated read.
  Budget: 20 ms; injected hold: 40 ms.
  Queued-miss ACK time: median 48.43 ms,
  range 40.45–50.39 ms. This includes the experiment's configured
  wait and is not serving TTFT or a performance comparison.
- Every run verified restored payload SHA-256, one GET call, untouched
  cancelled destinations, and no active-read ACK before retirement.

Reproduce the experiment with explicit parameters:

```bash
PYTHONPATH=python python test/experiments/l3_host_lifecycle.py \
  --iterations 30 --budget-s 0.02 --hold-s 0.04 \
  --output /tmp/l3-host-lifecycle.json
```

The mocked Host-copy suite can run on CPU with:

```bash
PYTHONPATH=python python test/ci_system/run_host_cpu_tests.py \
  -q test/runtime/test_host_executor.py
```

This launcher substitutes only optional GPU imports and prohibits GPU kernel
execution. It does not validate CUDA copies. Normal GPU CI uses normal pytest.

A shutdown-failure regression test also verifies that final retired Host
writes reach the coordinator before a Device submission error is re-raised.

## Remaining roadmap gates

| Gate | Status after this change |
| --- | --- |
| FC-Li architecture refactor | Implemented; integration awaits GPU/native Store verification and review |
| P0.1 model correctness | Native Store + model cross-instance restore, output/logprob comparisons and HTTP streaming still required |
| P0.2 bounded recovery | Queued expiry and retirement covered; native RPC interruption, active-call hard bounds, and safe L2-only degradation remain open |
| P0.3 capacity/isolation | Fixed pools preserved; deployment budgets, sustained pressure and native tenant enforcement tests remain open |
| P0.4 observability | #1839 diagnostics and production metrics/alerts remain separate work |
| P0.5 release | Native Store/GPU matrix, sustained serving load, canary and rollback evidence still required |

The local environment is macOS/ARM64 without CUDA or a native Mooncake Store.
No GPU model, RDMA, real Store outage or production soak result is claimed.
