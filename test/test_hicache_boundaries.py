"""Execute control boundaries without importing GPU model/kernel implementations."""

from __future__ import annotations

import ast
import enum
import gc
import weakref
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from tokenspeed.runtime.cache.hicache import HiCacheEngine
from tokenspeed.runtime.cache.l3.backend import resolve_l3_weight_version
from tokenspeed.runtime.cache.transfer.host_buffer import HostBufferView
from tokenspeed.runtime.cache.transfer.ops import (
    CacheShutdownFailure,
    HostWriteBackCompletion,
)

ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert len(selected) == len(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class ImmediateThread:
    def __init__(self, trace):
        self.trace = trace

    def submit(self, function):
        future = Future()
        self.trace.append("submit")
        try:
            future.set_result(function())
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def run(self, function):
        self.trace.append("run")
        return function()


def test_device_handle_only_runs_device_copy_work_and_returns_host_metadata():
    scope = definitions(
        ROOT / "python/tokenspeed/runtime/execution/device.py",
        {"DeviceRole", "_settle", "_resolve_role", "DeviceHandle"},
        {
            "enum": enum,
            "deque": deque,
            "cache_ops_from_plan": lambda plan: list(plan.cache),
            "CacheShutdownFailure": CacheShutdownFailure,
        },
    )
    trace = []
    thread = ImmediateThread(trace)
    executor = SimpleNamespace(
        forward_thread=thread,
        execution_stream="execution",
        default_stream="default",
        zero_cache_pages=lambda pages: trace.append("zero"),
    )
    completion = HostWriteBackCompletion((4,), ((0, 1, "hash", 0),))
    host = SimpleNamespace(
        submit_write_backs=lambda ops, **kwargs: trace.append("write"),
        submit_load_backs=lambda ops, **kwargs: trace.append("load"),
        submit_prefetches=Mock(
            side_effect=AssertionError("Host-only work crossed DeviceHandle")
        ),
        poll_results=lambda: [completion],
        shutdown=lambda: [completion],
    )
    handle = scope["DeviceHandle"](executor, host_cache_executor=host, kv_transfer=None)
    handle.execute(
        SimpleNamespace(
            cache=[object()], pages_to_zero=[1], remote_decode=None, remote_prefill=None
        ),
        None,
    )
    assert trace == ["submit", "write", "submit", "zero", "submit", "load"]
    host.submit_prefetches.assert_not_called()
    assert handle.poll_cache_results() == [completion]
    assert handle.shutdown_cache() == [completion]
    assert [e.op_id for e in HiCacheEngine(None).process_results([completion])] == [4]
    for name in (
        "query_l3_storage",
        "l3_prefetch_progress",
        "delete_l3_namespace",
        "set_l3_weight_version",
    ):
        assert name not in vars(scope["DeviceHandle"])


def test_request_handler_uses_host_engine_for_namespace_operations():
    tree = ast.parse(
        (ROOT / "python/tokenspeed/runtime/engine/request_handler.py").read_text()
    )
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RequestHandler"
    )
    selected = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_delete_l3_namespace", "_commit_l3_weight_version"}
    ]
    namespace = {
        "resolve_l3_weight_version": resolve_l3_weight_version,
        "_requested_weight_version": lambda req: req.weight_version,
    }
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]), "request_handler.py", "exec"
        ),
        namespace,
    )
    l3 = Mock()
    l3.delete_namespace.return_value = True
    device = Mock()
    handler = SimpleNamespace(
        _l3=l3,
        _device=device,
        server_args=SimpleNamespace(
            weight_version="v1", kvstore_storage_backend="mooncake"
        ),
        _require_weight_version_for_l3_flush=lambda req: (True, ""),
        _require_flush_for_l3_version_switch=lambda req: (True, ""),
    )
    assert namespace["_delete_l3_namespace"](handler)
    assert namespace["_commit_l3_weight_version"](
        handler, SimpleNamespace(weight_version="v2", flush_cache=True), "ok"
    ) == (True, "ok")
    l3.delete_namespace.assert_called_once_with()
    l3.set_weight_version.assert_called_once_with("v2")
    assert handler.server_args.weight_version == "v2"
    assert device.mock_calls == []


def test_host_export_keeps_bytes_and_geometry_without_retaining_device_layout():
    path = ROOT / "python/tokenspeed/runtime/cache/l2/storage.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "HostCacheStorage"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "host_buffer_view"
    )
    scope = {"HostBufferView": HostBufferView}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)

    class DeviceArena:
        pass

    arena = DeviceArena()
    reference = weakref.ref(arena)
    buffer = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    storage = SimpleNamespace(
        host_buffer=buffer,
        num_host_lcm_blocks=2,
        host_lcm_block_bytes=24,
        host_cache_block_bytes=(8, 24),
        layout=SimpleNamespace(
            buffers=(arena,),
            groups=(
                SimpleNamespace(group_id="full", cache_blocks_per_lcm_block=3),
                SimpleNamespace(group_id="state", cache_blocks_per_lcm_block=1),
            ),
        ),
    )
    view = scope["host_buffer_view"](storage)
    del arena, storage
    gc.collect()
    assert reference() is None
    assert view.host_buffer is buffer
    assert view.host_block_range(0, 6) == (40, 8)
    assert view.host_block_range(1, 2) == (24, 24)


def test_failed_device_submission_delivers_final_host_writes_before_raising():
    scope = definitions(
        ROOT / "python/tokenspeed/runtime/execution/device.py",
        {"DeviceRole", "_settle", "_resolve_role", "DeviceHandle"},
        {"enum": enum, "deque": deque, "CacheShutdownFailure": CacheShutdownFailure},
    )
    completed = HostWriteBackCompletion((9,), ((0, 1, "last", 0),))
    handle = scope["DeviceHandle"](
        SimpleNamespace(forward_thread=ImmediateThread([])),
        host_cache_executor=SimpleNamespace(shutdown=lambda: [completed]),
        kv_transfer=None,
    )
    failed = Future()
    failed.set_exception(ValueError("copy submission failed"))
    handle._l2_submissions.append(failed)
    # Execute the real EventLoop.close body against low-frequency collaborator
    # doubles, so its finally ordering is checked even when Device shutdown raises.
    path = ROOT / "python/tokenspeed/runtime/engine/event_loop.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "EventLoop"
    )
    close = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "close"
    )
    space = {"CacheShutdownFailure": CacheShutdownFailure}
    exec(compile(ast.Module(body=[close], type_ignores=[]), str(path), "exec"), space)
    cache = Mock()
    loop = SimpleNamespace(
        _device=handle,
        _cache_engine=cache,
        load_reporter=Mock(),
        _eplb_hooks=Mock(),
        send_to_tokenizer=Mock(),
        kv_transfer=None,
    )
    try:
        space["close"](loop)
        raise AssertionError("submission failure did not propagate")
    except CacheShutdownFailure as exc:
        assert isinstance(exc.__cause__, ValueError)
        assert exc.final_copy_results == (completed,)
    cache.shutdown.assert_called_once_with((completed,))
