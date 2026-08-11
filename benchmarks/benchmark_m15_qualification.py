"""Operational M1.5 qualification for the MJWarp offline data factory.

This runner deliberately distinguishes operational validity (all worlds complete
one 25-substep control transition, materialize three RGB views, and pass health)
from behavioral validity, which needs a task predicate or rollout comparator and
is therefore reported as unavailable for a raw actuator-control trace.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks.benchmark_warp_spike import PROFILE_CAMERAS, runtime_metadata
from libero.libero.runtime.warp import make_spike_benchmark_runner


def _sync(runner: Any) -> None:
    runner.synchronize()
    torch.cuda.synchronize()


def _measure(runner: Any, warmup: int, operations: int) -> dict[str, Any]:
    runner.reset_case()
    for _ in range(warmup):
        runner.step_and_render()
    _sync(runner)
    started = time.perf_counter()
    for _ in range(operations):
        runner.step_and_render()
    _sync(runner)
    elapsed = time.perf_counter() - started
    try:
        runner.assert_healthy()
        health_error = None
    except Exception as error:  # qualification should serialize failures
        health_error = f"{type(error).__name__}: {error}"
    worlds = runner.spike.num_worlds
    operational = health_error is None
    return {
        "operations": operations,
        "elapsed_seconds": elapsed,
        "operationally_valid": operational,
        "operational_failure_rate": 0.0 if operational else 1.0,
        "health_error": health_error,
        "transitions_per_second": operations * worlds / elapsed,
        "rgb_views_per_second": operations * worlds * len(PROFILE_CAMERAS) / elapsed,
        "transition_contract": (
            "25 physics substeps + 3x128x128 RGB + health/no-overflow"
        ),
        "reset_events": 0,
        "termination_events": None,
        "behaviorally_valid": None,
        "behavioral_validity_reason": (
            "raw actuator trace has no task predicate rollout"
        ),
    }


def _materialization(runner: Any, batches: int, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    runner.reset_case()
    for _ in range(3):
        runner.step_and_render()
    _sync(runner)
    gpu_to_cpu = 0.0
    write = 0.0
    compressed_write = 0.0
    total_bytes = 0
    compressed_bytes = 0
    for index in range(batches):
        runner.physics_step()
        images = runner.spike.render(check_health=False).rgb
        _sync(runner)
        started = time.perf_counter()
        host = torch.cat(tuple(images.values()), dim=0).cpu()
        gpu_to_cpu += time.perf_counter() - started
        payload = host.numpy()
        path = destination / f"rgb-batch-{index:03d}.bin"
        started = time.perf_counter()
        payload.tofile(path)
        write += time.perf_counter() - started
        total_bytes += path.stat().st_size
        compressed_path = destination / f"rgb-batch-{index:03d}.npz"
        started = time.perf_counter()
        np.savez_compressed(compressed_path, rgb=payload)
        compressed_write += time.perf_counter() - started
        compressed_bytes += compressed_path.stat().st_size
    return {
        "batches": batches,
        "bytes": total_bytes,
        "gpu_to_cpu_seconds": gpu_to_cpu,
        "raw_disk_write_seconds": write,
        "gpu_to_cpu_bytes_per_second": total_bytes / gpu_to_cpu,
        "raw_disk_bytes_per_second": total_bytes / write,
        "compressed_bytes": compressed_bytes,
        "npz_compressed_write_seconds": compressed_write,
        "npz_compressed_bytes_per_second": compressed_bytes / compressed_write,
        "npz_compression_ratio_vs_raw": compressed_bytes / total_bytes,
        "queue_backpressure_observed": False,
        "compression": "numpy savez_compressed (deflate)",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["LIBERO_M1_CTRL_TRACE"] = str(args.trace)
    runner = make_spike_benchmark_runner(args.worlds)
    try:
        result = {
            "schema_version": 1,
            "purpose": "M1.5 data-factory operational qualification",
            "runtime": runtime_metadata(),
            "worlds": args.worlds,
            "warmup_operations": args.warmup,
            "measure_operations": args.operations,
            "trace": runner.trace_report,
            "qualification": _measure(runner, args.warmup, args.operations),
        }
        if args.materialize_batches:
            result["materialization"] = _materialization(
                runner, args.materialize_batches, args.output.parent / "raw-rgb"
            )
        return result
    finally:
        runner.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--worlds", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--operations", type=int, default=1000)
    parser.add_argument("--materialize-batches", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(run(args), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
