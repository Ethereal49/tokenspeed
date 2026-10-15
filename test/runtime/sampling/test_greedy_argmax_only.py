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

"""When the greedy sampler reads only the argmax, and that it then takes the
LM head's reduced ids instead of re-reading the logits."""

import os
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

import pytest  # noqa: E402
import torch  # noqa: E402

from tokenspeed.runtime.layers.logits_processor import (  # noqa: E402
    LogitsProcessorOutput,
)
from tokenspeed.runtime.sampling.backends.base import (  # noqa: E402
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.backends.greedy import (  # noqa: E402
    GreedySamplingBackend,
)
from tokenspeed.runtime.sampling.sampling_batch_info import (  # noqa: E402
    SamplingBatchInfo,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

VOCAB = 64


def _backend(*, enable_output_logprobs: bool) -> GreedySamplingBackend:
    return GreedySamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=False,
            sampling_stream="batch",
            logprob_order="torch",
            enable_output_logprobs=enable_output_logprobs,
            max_bs=4,
            max_draft_tokens_per_req=1,
            vocab_size=VOCAB,
            device=torch.device("cuda"),
        )
    )


def test_greedy_reads_the_distribution_only_for_masks_and_logprobs():
    plain = SamplingBatchInfo(vocab_size=VOCAB, device="cuda")
    masked = SamplingBatchInfo(
        vocab_size=VOCAB,
        vocab_mask=torch.zeros((4, VOCAB // 32), dtype=torch.int32, device="cuda"),
        device="cuda",
    )
    assert _backend(enable_output_logprobs=False).reads_argmax_only(plain)
    assert not _backend(enable_output_logprobs=False).reads_argmax_only(masked)
    assert not _backend(enable_output_logprobs=True).reads_argmax_only(plain)


def test_greedy_takes_the_reduced_ids_over_the_shard_logits():
    backend = _backend(enable_output_logprobs=False)
    info = SamplingBatchInfo(vocab_size=VOCAB, device="cuda")
    # A shard whose own argmax differs from the group's reduced ids.
    shard = torch.zeros((3, VOCAB // 2), device="cuda")
    shard[:, 0] = 1.0
    ids = torch.tensor([40, 7, 63], device="cuda")

    tokens, _ = backend.sample(
        LogitsProcessorOutput(next_token_logits=shard, next_token_ids=ids), info
    )
    assert tokens.tolist() == [40, 7, 63]

    candidates = torch.zeros((3, 1), dtype=torch.int64, device="cuda")
    predict, lengths = backend.verify(
        LogitsProcessorOutput(next_token_logits=shard, next_token_ids=ids),
        info,
        candidates,
        tree=None,
    )
    assert predict.tolist() == [40, 7, 63]
    assert lengths.tolist() == [1, 1, 1]

    # Without ids the sampler reduces the logits it was given.
    tokens, _ = backend.sample(LogitsProcessorOutput(next_token_logits=shard), info)
    assert tokens.tolist() == [0, 0, 0]
