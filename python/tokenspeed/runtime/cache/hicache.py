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

"""Coordinate Host-copy completions with optional L3 persistence.

The device side reports completed copies as plain metadata. This Host-only
coordinator owns no L2 executor or CUDA object; it releases write ACKs only
once the configured L3 engine has finished their PUTs.
"""

from __future__ import annotations

from collections.abc import Sequence

from tokenspeed_scheduler import Cache

from tokenspeed.runtime.cache.l3.engine import L3CacheEngine
from tokenspeed.runtime.cache.transfer.ops import HostWriteBackCompletion


class HiCacheEngine:
    """Join the Device-to-Host and Host-to-Store legs on the control plane."""

    def __init__(self, l3: L3CacheEngine | None) -> None:
        self.l3 = l3

    def process_results(self, copy_results: Sequence) -> list:
        """Return scheduler ACKs, deferring Store-backed writes until PUT finishes.

        Args:
            copy_results: Completed Host writes or ordinary cache-copy ACKs.

        Returns:
            ACKs safe for replica convergence and scheduler advancement.
        """
        results = []
        for completion in copy_results:
            if not isinstance(completion, HostWriteBackCompletion):
                results.append(completion)
            elif self.l3 is not None and completion.host_pages:
                self.l3.submit_backup(completion)
            else:
                for op_id in completion.op_ids:
                    event = Cache.WriteBackDoneEvent()
                    event.op_id = op_id
                    results.append(event)
        if self.l3 is not None:
            results.extend(self.l3.poll_results())
        return results

    def consume_backup_poll_failure(self) -> bool:
        """Report local PUT failure without raising before replica collectives."""
        return self.l3 is not None and self.l3.consume_backup_poll_failure()

    def shutdown(self, final_copy_results: Sequence) -> None:
        """Persist final completed Host writes, then retire and close the Store."""
        self.process_results(final_copy_results)
        if self.l3 is not None:
            self.l3.shutdown()
