"""Run mocked Host-copy orchestration without installing GPU-only libraries."""

import sys
from pathlib import Path
from types import ModuleType

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / "python"), str(root / "tokenspeed-kernel/python"), str(root)]
profile = ModuleType("tokenspeed_kernel.profiling")
profile.bootstrap_profiling_from_env = lambda: None
sys.modules[profile.__name__] = profile
triton = ModuleType("tokenspeed_kernel.ops.kvcache.triton")
triton.HOST_CACHE_TRANSFER_CHUNK_BYTES = 4096


def unavailable(*args, **kwargs):
    raise RuntimeError("GPU kernels are prohibited in CPU validation")


triton.transfer_cache_blocks = unavailable
triton.wait_layer_ready = unavailable
sys.modules[triton.__name__] = triton
import pytest
import torch

if torch.cuda.is_available():
    raise RuntimeError(
        "Use normal pytest on a CUDA host; CPU import substitutes are prohibited there"
    )
sys.exit(pytest.main(sys.argv[1:]))
