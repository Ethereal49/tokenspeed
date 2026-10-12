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

"""Host-only L3 I/O, lifecycle, and completion tracking.

This component owns no Device cache, stream, or model executor. The cache
coordinator passes completed Host writes here after D2H and holds scheduler
ACKs until their PUTs complete. Prefetch results are committed only after the
replica converges on the landed prefix.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from tokenspeed_scheduler import Cache

from tokenspeed.runtime.cache.l3.executor import L3HostStore, StoragePage
from tokenspeed.runtime.cache.transfer.ops import HostWriteBackCompletion, PrefetchOp

logger = logging.getLogger(__name__)


class _PrefetchJob:
    """One in-flight L3 prefetch on the lane: the op, its pages by prefix page,
    and the future that resolves to the pages landed (a prefix length). A
    failed future is reported once (``failure_logged``), not on every poll
    until the replica converges."""

    __slots__ = ("request_id", "num_pages", "future", "failure_logged", "deadline")

    def __init__(
        self, request_id: str, num_pages: int, future: Future, deadline: float
    ) -> None:
        self.request_id = request_id
        self.num_pages = num_pages
        self.future = future
        self.deadline = float(deadline)
        self.failure_logged = False


class L3CacheEngine:
    """Own the Store client and CPU lanes independently of GPU execution."""

    def __init__(
        self,
        store: L3HostStore,
        owners: Any,
        *,
        prefix_for_weight_version: Callable[[str], str],
        prefetch_timeout_base_s: float,
        prefetch_timeout_per_page_s: float,
        prefetch_batch_pages: int,
    ) -> None:
        """Configure an engine over an already allocated Host buffer.

        Args:
            store: The Host-buffer adapter; no Device buffer is accepted.
            owners: Translate scheduler Host IDs to this rank's local IDs.
            prefix_for_weight_version: Rebuild the immutable checkpoint namespace.
            prefetch_timeout_base_s: Budget from submission, checked between GET batches.
            prefetch_timeout_per_page_s: Deadline increment per prefix page.
            prefetch_batch_pages: Maximum prefix pages in one native GET batch.
        """
        if prefetch_timeout_base_s <= 0 or prefetch_timeout_per_page_s < 0:
            raise ValueError(
                "prefetch needs a positive base and non-negative per-page deadline"
            )
        if prefetch_batch_pages <= 0:
            raise ValueError("prefetch_batch_pages must be positive")
        self._store = store
        self._owners = owners
        self._prefix_for_weight_version = prefix_for_weight_version
        self._prefetch_timeout_base_s = float(prefetch_timeout_base_s)
        self._prefetch_timeout_per_page_s = float(prefetch_timeout_per_page_s)
        self._prefetch_batch_pages = int(prefetch_batch_pages)
        self._lock = threading.Lock()
        self._backup_futures: list[tuple[Future, list[int], list[StoragePage]]] = []
        self._backup_poll_failed = False
        self._backup_lane: ThreadPoolExecutor | None = None
        self._prefetch_lane: ThreadPoolExecutor | None = None
        self._prefetch_jobs: dict[int, _PrefetchJob] = {}
        self._prefetch_acks: list[tuple[int, int]] = []
        self._closed = False

    def exists(self, pages: Sequence[StoragePage]) -> list[bool]:
        """Return full-length existence results in the input key order."""
        return self._store.exists(pages)

    def set_weight_version(self, weight_version: str) -> None:
        """Drain both Host lanes before publishing the new namespace."""
        self._wait_l3_backups()
        self._wait_l3_prefetches()
        self._store.set_key_prefix(self._prefix_for_weight_version(str(weight_version)))

    def _wait_l3_backups(self) -> None:
        with self._lock:
            inflight = list(self._backup_futures)
        for future, _op_ids, _pages in inflight:
            future.result()

    def _wait_l3_prefetches(self) -> None:
        """Block until every in-flight prefetch has left the lane.

        A prefetch that raised is done too (``prefetch_progress`` reports
        it as landing nothing); waiting does not re-raise it.
        """
        with self._lock:
            jobs = list(self._prefetch_jobs.values())
        for job in jobs:
            if not job.future.cancelled():
                job.future.exception()

    def submit_prefetches(self, cache_ops: Sequence) -> None:
        """Start the plan's L3 prefetches on the lane; no stream dependency.

        A ``PrefetchOp`` fills a waiting request's freshly allocated Host
        pages from L3 before its admission: the lane fetches the op's rows
        in prefix order, ``prefetch_batch_pages`` pages per
        ``batch_get_into``, and stops at the first batch with a missing page
        or at the deadline ``base + per_page * pages``; its result is the
        pages landed, a prefix length. The hooks MIN-reduce that length
        across the replica (``prefetch_progress`` /
        ``complete_prefetch``) and the op is acknowledged once as
        ``PrefetchDone(op_id, landed_pages)``. Under KVP each rank fetches
        the Host pages it owns; the replica MIN then lands the common prefix.

        Args:
            cache_ops: The round's adapted cache ops; the ``PrefetchOp``
                entries are read here.
        """
        ops = [op for op in cache_ops if isinstance(op, PrefetchOp)]
        if not ops:
            return
        if self._closed:
            raise RuntimeError("L3 engine is closed")
        if self._store is None:
            raise RuntimeError(
                "the plan prefetches from L3 but no storage backend is attached "
                "(--kvstore-storage-backend)"
            )
        op_ids = [int(op.op_id) for op in ops]
        if len(set(op_ids)) != len(op_ids):
            raise ValueError(f"duplicate prefetch op id in one plan: {op_ids}")
        owners = self._owners
        jobs: list[tuple[PrefetchOp, list[list[StoragePage]]]] = []
        for op in ops:
            if op.num_pages <= 0 or not op.rows:
                raise ValueError(f"prefetch operation {op.op_id} carries no pages")
            kept_by_position = dict(
                owners.owned_host_positions(
                    [(row.group_id, row.host_page) for row in op.rows]
                )
            )
            # Rows come in prefix-page order; a page may have no rows (a
            # sliding group's older pages) and lands trivially.
            pages: list[list[StoragePage]] = [[] for _ in range(op.num_pages)]
            last_index = op.first_page
            for position, row in enumerate(op.rows):
                page = row.page_index - op.first_page
                if not 0 <= page < op.num_pages or row.page_index < last_index:
                    raise ValueError(
                        f"prefetch operation {op.op_id}: row {position} names page "
                        f"{row.page_index} outside [{op.first_page}, "
                        f"{op.first_page + op.num_pages}) or out of order"
                    )
                last_index = row.page_index
                local = kept_by_position.get(position)
                if local is None:
                    continue  # another KVP rank owns this Host page
                group, host_block = local
                pages[page].append(
                    (group, host_block, row.content_hash, int(row.page_offset))
                )
            jobs.append((op, pages))
        with self._lock:
            if any(op_id in self._prefetch_jobs for op_id in op_ids):
                raise ValueError(f"prefetch op already in flight: {op_ids}")
            lane = self._prefetch_lane
            if lane is None:
                lane = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="l3-prefetch"
                )
                self._prefetch_lane = lane
            for op, pages in jobs:
                deadline_s = (
                    self._prefetch_timeout_base_s
                    + self._prefetch_timeout_per_page_s * len(pages)
                )
                deadline = time.monotonic() + deadline_s
                self._prefetch_jobs[int(op.op_id)] = _PrefetchJob(
                    request_id=str(op.request_id),
                    num_pages=len(pages),
                    future=lane.submit(
                        self._run_prefetch,
                        int(op.op_id),
                        str(op.request_id),
                        pages,
                        deadline,
                    ),
                    deadline=deadline,
                )
        if self._store.rank == 0:
            logger.info(
                f"[L3] prefetch started: operations={len(jobs):d} pages="
                f"{sum(len(pages) for _, pages in jobs):d}"
            )

    def _run_prefetch(
        self,
        op_id: int,
        request_id: str,
        pages: list[list[StoragePage]],
        deadline: float,
    ) -> int:
        """The lane's job: fetch ``pages`` in prefix order; return the pages landed.

        One ``batch_get_into`` per ``prefetch_batch_pages`` prefix pages. A
        batch is not started past the deadline (checked between batches
        only: the store call has no timeout, so an issued batch runs to
        completion); the first page whose get fails (any of its groups) ends
        the fetch, and the pages before it are the result -- a page with no
        rows (nothing of it to fetch here) lands trivially. A backend fault
        counts as a miss at that batch, so the replica still converges on a
        prefix.
        """
        l3_store = self._store
        if l3_store is None:
            raise RuntimeError("L3 prefetch ran without a storage backend")
        landed = 0
        batch = self._prefetch_batch_pages
        for start in range(0, len(pages), batch):
            if time.monotonic() > deadline:
                logger.warning(
                    f"[L3] prefetch {op_id} ({request_id}) timed out after "
                    f"{landed}/{len(pages)} pages"
                )
                break
            chunk = pages[start : start + batch]
            flat = [page for rows in chunk for page in rows]
            if not flat:
                # No row of the chunk is this rank's to fetch (none at all, or
                # other KVP ranks own them): the pages land here so this rank
                # does not shorten the replica's prefix.
                landed += len(chunk)
                continue
            try:
                ok = list(l3_store.prefetch(flat))
            except Exception:
                logger.exception(
                    f"[L3] prefetch {op_id} ({request_id}) failed at page {landed}; "
                    "landing the pages before it"
                )
                break
            if len(ok) != len(flat):
                logger.error(
                    f"[L3] prefetch {op_id} ({request_id}): result has {len(ok)} "
                    f"flags for {len(flat)} pages; landing the pages before this batch"
                )
                break
            cursor = 0
            stopped = False
            for rows in chunk:
                page_ok = all(ok[cursor : cursor + len(rows)])
                cursor += len(rows)
                if not page_ok:
                    stopped = True
                    break
                landed += 1
            if stopped:
                logger.warning(
                    f"[L3] prefetch {op_id} ({request_id}) missed page {landed}; "
                    f"landing {landed}/{len(pages)} pages"
                )
                break
        return landed

    def prefetch_progress(self) -> dict[int, tuple[bool, int]]:
        """This rank's view of every in-flight prefetch op: ``(done, landed)``.

        ``landed`` is the pages landed once ``done``, else 0. The set of op
        ids is mirrored on every rank (the plans are), so the hooks can
        MIN-reduce the vector in op-id order.
        """
        with self._lock:
            jobs = dict(self._prefetch_jobs)
        progress: dict[int, tuple[bool, int]] = {}
        for op_id, job in jobs.items():
            if not job.future.done():
                if time.monotonic() >= job.deadline and job.future.cancel():
                    # ThreadPoolExecutor.cancel() succeeds only before the
                    # worker starts. No native GET has touched these pages.
                    logger.warning(
                        f"[L3] queued prefetch {op_id} expired before starting"
                    )
                else:
                    progress[op_id] = (False, 0)
                    continue
            if job.future.cancelled():
                progress[op_id] = (True, 0)
                continue
            failed = job.future.exception()
            if failed is not None:
                if not job.failure_logged:
                    # Polled every round until the replica converges: say it once.
                    job.failure_logged = True
                    logger.error(
                        f"[L3] prefetch {op_id} ({job.request_id}) raised; landing "
                        "nothing",
                        exc_info=failed,
                    )
                progress[op_id] = (True, 0)
            else:
                progress[op_id] = (True, int(job.future.result()))
        return progress

    def complete_prefetch(self, op_id: int, landed_pages: int) -> None:
        """Record the replica-converged outcome of a prefetch op.

        ``landed_pages`` is the replica MIN of every rank's landed prefix;
        ``poll_results`` yields the op's one ``PrefetchDone`` with it.
        """
        with self._lock:
            job = self._prefetch_jobs.get(int(op_id))
            if job is None:
                raise KeyError(f"prefetch op {op_id} is not in flight")
            if not 0 <= int(landed_pages) <= job.num_pages:
                raise ValueError(
                    f"prefetch op {op_id} landed {landed_pages} of {job.num_pages} pages"
                )
            del self._prefetch_jobs[int(op_id)]
            self._prefetch_acks.append((int(op_id), int(landed_pages)))

    def consume_backup_poll_failure(self) -> bool:
        """Return whether an L3 backup future failed since the last consume.

        ``poll_results`` must not raise that failure: ``CacheOpHooks`` has
        not entered its replica collectives yet, and a rank-local raise
        hangs peers waiting in ``all_reduce`` / ``all_gather_object``.
        """

        failed = self._backup_poll_failed
        self._backup_poll_failed = False
        return failed

    def _collect_finished_backups(self, results: list) -> None:
        with self._lock:
            inflight = list(self._backup_futures)
            self._backup_futures = []
        still: list[tuple[Future, list[int], list[StoragePage]]] = []
        for future, op_ids, pages in inflight:
            if not future.done():
                still.append((future, op_ids, pages))
                continue
            failed = future.exception()
            if failed is not None:
                logger.error(
                    "L3 backup failed; retrying and reporting a rank-local "
                    "failure so replica cache-poll collectives can converge",
                    exc_info=failed,
                )
                self._backup_poll_failed = True
                workers = self._backup_lane
                if workers is not None:
                    still.append(
                        (workers.submit(self._backup_to_storage, pages), op_ids, pages)
                    )
                else:
                    still.append((future, op_ids, pages))
                continue
            future.result()
            results.extend(self._write_done(op_id) for op_id in op_ids)
        with self._lock:
            self._backup_futures.extend(still)

    def _backup_to_storage(self, pages: Sequence[StoragePage]) -> None:
        l3_store = self._store
        if not pages or l3_store is None:
            return
        # The backend handles create-only PUTs itself.
        results = l3_store.backup(pages)
        if len(results) != len(pages) or not all(results):
            ok = sum(1 for flag in results if flag)
            raise RuntimeError(
                f"L3 backup failed for Host page(s): ok={ok}/{len(pages)}"
            )

    @staticmethod
    def _write_done(op_id: int):
        event = Cache.WriteBackDoneEvent()
        event.op_id = op_id
        return event

    @staticmethod
    def _prefetch_done(op_id: int, landed_pages: int):
        return Cache.PrefetchDoneEvent(op_id, landed_pages)

    def submit_backup(self, completion: HostWriteBackCompletion) -> None:
        """Start PUT only for a completed D2H copy; no ACK is returned here."""
        if self._closed:
            raise RuntimeError("L3 engine is closed")
        if not completion.host_pages:
            raise ValueError("L3 backup needs completed Host pages")
        workers = self._backup_lane
        if workers is None:
            workers = ThreadPoolExecutor(max_workers=1, thread_name_prefix="l3-backup")
            self._backup_lane = workers
        pages = list(completion.host_pages)
        future = workers.submit(self._backup_to_storage, pages)
        with self._lock:
            self._backup_futures.append((future, list(completion.op_ids), pages))

    def poll_results(self) -> list:
        """Return completed PUTs and replica-converged prefetch ACKs without waiting."""
        results: list = []
        with self._lock:
            results.extend(
                self._prefetch_done(op_id, landed)
                for op_id, landed in self._prefetch_acks
            )
            self._prefetch_acks.clear()
        self._collect_finished_backups(results)
        return results

    def delete_namespace(self) -> bool:
        """Drain native I/O before deletion; return False on a failed drain/delete."""
        try:
            self._wait_l3_backups()
            self._wait_l3_prefetches()
        except Exception:
            logger.exception("L3 backup/prefetch wait failed before namespace delete")
            return False
        return self._store.rotate_namespace()

    def shutdown(self) -> None:
        """Retire both CPU lanes before closing their registered Host buffer."""
        if self._closed:
            return
        self._closed = True
        errors = []
        with self._lock:
            futures = [future for future, _op_ids, _pages in self._backup_futures]
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                errors.append(exc)
        for lane in (self._backup_lane, self._prefetch_lane):
            if lane is not None:
                lane.shutdown(wait=True)
        self._backup_lane = None
        self._prefetch_lane = None
        with self._lock:
            self._backup_futures.clear()
            self._prefetch_jobs.clear()
            self._prefetch_acks.clear()
        self._store.close()
        if errors:
            raise errors[0]
