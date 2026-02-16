"""Performance profiling for CFR solvers and training loops.

Measures:
- Wall-clock time per CFR iteration
- Peak and steady-state memory usage
- Throughput (information sets processed per second)
- GPU utilization (when available)

Identifies and reports top bottlenecks.
"""

from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch


@dataclass
class TimingRecord:
    """A single timing measurement."""
    name: str
    wall_time_s: float
    iterations: int = 1

    @property
    def per_iter_ms(self) -> float:
        if self.iterations == 0:
            return 0.0
        return (self.wall_time_s / self.iterations) * 1000


@dataclass
class MemoryRecord:
    """Memory usage snapshot."""
    name: str
    peak_mb: float
    current_mb: float


@dataclass
class ProfileResult:
    """Complete profiling result."""
    timings: List[TimingRecord] = field(default_factory=list)
    memory: List[MemoryRecord] = field(default_factory=list)
    throughput: Dict[str, float] = field(default_factory=dict)
    gpu_info: Dict[str, Any] = field(default_factory=dict)
    bottlenecks: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["=" * 60, "  Profile Summary", "=" * 60]

        if self.timings:
            lines.append("\nTimings:")
            lines.append(f"  {'Name':<30} {'Total':>10} {'Per-Iter':>12} {'Iters':>8}")
            lines.append(f"  {'-'*30} {'-'*10} {'-'*12} {'-'*8}")
            for t in sorted(self.timings, key=lambda x: x.wall_time_s, reverse=True):
                lines.append(
                    f"  {t.name:<30} {t.wall_time_s:>9.3f}s "
                    f"{t.per_iter_ms:>10.3f}ms {t.iterations:>8}"
                )

        if self.memory:
            lines.append("\nMemory:")
            for m in self.memory:
                lines.append(f"  {m.name}: peak={m.peak_mb:.1f}MB, current={m.current_mb:.1f}MB")

        if self.throughput:
            lines.append("\nThroughput:")
            for name, rate in self.throughput.items():
                lines.append(f"  {name}: {rate:,.0f}/sec")

        if self.gpu_info:
            lines.append("\nGPU Info:")
            for k, v in self.gpu_info.items():
                lines.append(f"  {k}: {v}")

        if self.bottlenecks:
            lines.append("\nTop Bottlenecks:")
            for i, b in enumerate(self.bottlenecks, 1):
                lines.append(f"  {i}. {b}")

        lines.append("=" * 60)
        return "\n".join(lines)


class Profiler:
    """Profile CFR solver and training loop performance."""

    def __init__(self) -> None:
        self.result = ProfileResult()

    def time_function(
        self,
        fn: Callable,
        name: str,
        iterations: int = 1,
        *args,
        **kwargs,
    ) -> Any:
        """Time a function call and record the result."""
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        self.result.timings.append(TimingRecord(name, elapsed, iterations))
        return result

    @contextmanager
    def section(self, name: str, iterations: int = 1):
        """Context manager for timing a code section."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.result.timings.append(TimingRecord(name, elapsed, iterations))

    def measure_memory(self, name: str, fn: Callable, *args, **kwargs) -> Any:
        """Measure peak memory during function execution."""
        tracemalloc.start()
        result = fn(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.result.memory.append(MemoryRecord(
            name=name,
            peak_mb=peak / (1024 * 1024),
            current_mb=current / (1024 * 1024),
        ))
        return result

    def record_throughput(self, name: str, count: int, elapsed_s: float) -> None:
        """Record throughput measurement."""
        if elapsed_s > 0:
            self.result.throughput[name] = count / elapsed_s

    def check_gpu(self) -> None:
        """Detect and record GPU information."""
        info: Dict[str, Any] = {}

        if torch.backends.mps.is_available():
            info["backend"] = "MPS (Apple Silicon)"
            info["available"] = True
        elif torch.cuda.is_available():
            info["backend"] = f"CUDA ({torch.cuda.get_device_name(0)})"
            info["available"] = True
            info["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        else:
            info["backend"] = "CPU only"
            info["available"] = False

        info["pytorch_version"] = torch.__version__
        self.result.gpu_info = info

    def identify_bottlenecks(self) -> List[str]:
        """Identify top-3 bottlenecks from profiling data."""
        bottlenecks = []

        sorted_timings = sorted(self.result.timings, key=lambda t: t.wall_time_s, reverse=True)
        total_time = sum(t.wall_time_s for t in sorted_timings)

        for t in sorted_timings[:3]:
            if total_time > 0:
                pct = (t.wall_time_s / total_time) * 100
                bottlenecks.append(
                    f"{t.name}: {t.wall_time_s:.3f}s ({pct:.1f}% of total) "
                    f"— {t.per_iter_ms:.3f}ms/iter"
                )

        self.result.bottlenecks = bottlenecks
        return bottlenecks


def profile_cfr_solver(
    solver_factory: Callable,
    game,
    iterations: int = 1000,
    warmup_iterations: int = 100,
) -> ProfileResult:
    """Profile a CFR solver with standard measurements.

    solver_factory: callable that returns a solver with .run(n) method
    """
    profiler = Profiler()
    profiler.check_gpu()

    # Warmup
    solver = solver_factory(game)
    solver.run(warmup_iterations)

    # Time the main solve
    solver = solver_factory(game)
    with profiler.section("CFR solve", iterations=iterations):
        solver.run(iterations)

    # Count infosets for throughput
    if hasattr(solver, 'infosets'):
        num_infosets = len(solver.infosets)
    elif hasattr(solver, 'info_sets'):
        num_infosets = len(solver.info_sets)
    else:
        num_infosets = 0

    total_time = profiler.result.timings[-1].wall_time_s
    if num_infosets > 0 and total_time > 0:
        profiler.record_throughput(
            "infosets_per_second",
            num_infosets * iterations,
            total_time,
        )

    # Memory measurement
    profiler.measure_memory(
        "CFR solve (fresh)",
        lambda: solver_factory(game).run(iterations),
    )

    profiler.identify_bottlenecks()
    return profiler.result


def profile_training_loop(
    trainer_factory: Callable,
    train_fn: Callable,
    num_epochs: int = 5,
) -> ProfileResult:
    """Profile a training loop (e.g., ReBeL) with per-epoch timing.

    trainer_factory: callable that returns a trainer
    train_fn: callable(trainer) that runs training and returns metrics
    """
    profiler = Profiler()
    profiler.check_gpu()

    trainer = trainer_factory()

    start_total = time.perf_counter()
    metrics = train_fn(trainer)
    total_elapsed = time.perf_counter() - start_total

    profiler.result.timings.append(
        TimingRecord("Total training", total_elapsed, num_epochs)
    )

    profiler.measure_memory(
        "Training loop",
        lambda: train_fn(trainer_factory()),
    )

    profiler.identify_bottlenecks()
    return profiler.result
