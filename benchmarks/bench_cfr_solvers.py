"""Benchmark suite comparing CFR solver variants.

Measures iteration speed and convergence for:
  - Scalar CFR (baseline)
  - Vectorized CFR (torch CPU)
  - Matrix CFR (torch, CPU and GPU)
  - Batched MCCFR
  - Deep CFR
  - SD-CFR

Run: python3 benchmarks/bench_cfr_solvers.py
"""

import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

sys.path.insert(0, ".")

import torch

from cfr.batched_mccfr import BatchedMCCFR, BatchedMCCFRConfig
from cfr.deep_cfr import DeepCFR, DeepCFRConfig, SingleDeepCFR, SDCFRConfig
from cfr.matrix_cfr import MatrixCFR, MatrixCFRConfig
from cfr.mccfr import ExternalSamplingMCCFRTrainer
from cfr.solver import CFRConfig, CFRTrainer
from cfr.vectorized import VectorizedCFR
from kuhn.game import KuhnPoker
from leduc.game import LeducPoker


@dataclass
class BenchResult:
    name: str
    game: str
    iterations: int
    total_time: float
    exploitability: float
    time_per_iter: float


def bench_scalar_cfr(game, game_name: str, iters: int) -> BenchResult:
    solver = CFRTrainer(game)
    t0 = time.time()
    solver.run(iters)
    elapsed = time.time() - t0
    exp = solver.exploitability()
    return BenchResult(
        name="Scalar CFR",
        game=game_name,
        iterations=iters,
        total_time=elapsed,
        exploitability=exp,
        time_per_iter=elapsed / iters,
    )


def bench_vectorized_cfr(game, game_name: str, iters: int) -> BenchResult:
    solver = VectorizedCFR(game)
    t0 = time.time()
    solver.train(iters)
    elapsed = time.time() - t0
    exp = solver._exploitability()
    return BenchResult(
        name="Vectorized CFR",
        game=game_name,
        iterations=iters,
        total_time=elapsed,
        exploitability=exp,
        time_per_iter=elapsed / iters,
    )


def bench_matrix_cfr(game, game_name: str, iters: int,
                     device: str = "cpu") -> BenchResult:
    solver = MatrixCFR(game, device=device)
    t0 = time.time()
    solver.run(iters)
    elapsed = time.time() - t0
    exp = solver.exploitability()
    label = f"Matrix CFR ({device})"
    return BenchResult(
        name=label,
        game=game_name,
        iterations=iters,
        total_time=elapsed,
        exploitability=exp,
        time_per_iter=elapsed / iters,
    )


def bench_mccfr(game, game_name: str, iters: int) -> BenchResult:
    solver = ExternalSamplingMCCFRTrainer(game)
    t0 = time.time()
    solver.run(iters)
    elapsed = time.time() - t0

    from cfr.solver import CFRTrainer
    profile = solver.average_strategy_profile()
    dummy = CFRTrainer(game)
    br0 = dummy._best_response_value(profile, 0)
    br1 = dummy._best_response_value(profile, 1)
    exp = 0.5 * (br0 + br1)

    return BenchResult(
        name="MCCFR (ext-sample)",
        game=game_name,
        iterations=iters,
        total_time=elapsed,
        exploitability=exp,
        time_per_iter=elapsed / iters,
    )


def bench_batched_mccfr(game, game_name: str, iters: int,
                        batch_size: int = 16) -> BenchResult:
    config = BatchedMCCFRConfig(batch_size=batch_size, seed=42)
    solver = BatchedMCCFR(game, config=config)
    t0 = time.time()
    solver.run(iters)
    elapsed = time.time() - t0
    exp = solver.exploitability()
    return BenchResult(
        name=f"Batched MCCFR (b={batch_size})",
        game=game_name,
        iterations=iters,
        total_time=elapsed,
        exploitability=exp,
        time_per_iter=elapsed / iters,
    )


def bench_deep_cfr(game, game_name: str, traversals: int) -> BenchResult:
    config = DeepCFRConfig(
        num_traversals=traversals,
        advantage_train_steps=min(500, traversals),
        strategy_train_steps=min(500, traversals),
        batch_size=64,
        hidden_sizes=(64, 64),
        learning_rate=1e-3,
        seed=42,
    )
    solver = DeepCFR(game, config=config)
    t0 = time.time()
    solver.train()
    elapsed = time.time() - t0
    exp = solver.exploitability()
    return BenchResult(
        name="Deep CFR",
        game=game_name,
        iterations=traversals,
        total_time=elapsed,
        exploitability=exp,
        time_per_iter=elapsed / traversals,
    )


def bench_sdcfr(game, game_name: str, traversals: int) -> BenchResult:
    config = SDCFRConfig(
        num_traversals=traversals,
        train_steps=min(500, traversals),
        batch_size=64,
        hidden_sizes=(64, 64),
        learning_rate=1e-3,
        seed=42,
        max_stored_networks=20,
    )
    solver = SingleDeepCFR(game, config=config)
    t0 = time.time()
    solver.train()
    elapsed = time.time() - t0
    exp = solver.exploitability()
    return BenchResult(
        name="SD-CFR",
        game=game_name,
        iterations=traversals,
        total_time=elapsed,
        exploitability=exp,
        time_per_iter=elapsed / traversals,
    )


def print_results(results: List[BenchResult]) -> None:
    # Group by game
    games = {}
    for r in results:
        games.setdefault(r.game, []).append(r)

    for game_name, game_results in games.items():
        print(f"\n{'='*75}")
        print(f"  {game_name}")
        print(f"{'='*75}")
        print(f"{'Solver':<25} {'Iters':>8} {'Time':>8} {'ms/iter':>8} {'Exploit':>10}")
        print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        for r in game_results:
            ms_per_iter = r.time_per_iter * 1000
            print(
                f"{r.name:<25} {r.iterations:>8} {r.total_time:>7.2f}s "
                f"{ms_per_iter:>7.3f} {r.exploitability:>10.6f}"
            )


def check_gpu_available() -> List[str]:
    """Check which GPU backends are available."""
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def main():
    devices = check_gpu_available()
    print(f"Available devices: {devices}")
    print(f"PyTorch version: {torch.__version__}")

    results = []

    # ── Kuhn Poker benchmarks ──
    print("\nRunning Kuhn Poker benchmarks...")
    kuhn = KuhnPoker()
    kuhn_iters = 10000

    results.append(bench_scalar_cfr(kuhn, "Kuhn Poker", kuhn_iters))
    results.append(bench_vectorized_cfr(kuhn, "Kuhn Poker", kuhn_iters))
    results.append(bench_matrix_cfr(kuhn, "Kuhn Poker", kuhn_iters, "cpu"))
    if "mps" in devices:
        results.append(bench_matrix_cfr(kuhn, "Kuhn Poker", kuhn_iters, "mps"))
    results.append(bench_mccfr(kuhn, "Kuhn Poker", kuhn_iters))
    results.append(bench_batched_mccfr(kuhn, "Kuhn Poker", kuhn_iters // 10, batch_size=16))
    results.append(bench_deep_cfr(kuhn, "Kuhn Poker", 500))
    results.append(bench_sdcfr(kuhn, "Kuhn Poker", 500))

    # ── Leduc Poker benchmarks ──
    print("\nRunning Leduc Poker benchmarks...")
    leduc = LeducPoker()
    leduc_iters = 1000

    results.append(bench_scalar_cfr(leduc, "Leduc Poker", leduc_iters))
    results.append(bench_vectorized_cfr(leduc, "Leduc Poker", leduc_iters))
    results.append(bench_matrix_cfr(leduc, "Leduc Poker", leduc_iters, "cpu"))
    if "mps" in devices:
        results.append(bench_matrix_cfr(leduc, "Leduc Poker", leduc_iters, "mps"))
    results.append(bench_mccfr(leduc, "Leduc Poker", leduc_iters))
    results.append(bench_batched_mccfr(leduc, "Leduc Poker", leduc_iters // 10, batch_size=16))
    results.append(bench_deep_cfr(leduc, "Leduc Poker", 300))
    results.append(bench_sdcfr(leduc, "Leduc Poker", 300))

    print_results(results)

    # Print speedup summary
    print("\n\nSpeedup Summary (Matrix CFR vs Scalar CFR):")
    for game_name in ["Kuhn Poker", "Leduc Poker"]:
        scalar = [r for r in results if r.name == "Scalar CFR" and r.game == game_name]
        matrix = [r for r in results if r.name.startswith("Matrix CFR") and r.game == game_name]
        if scalar and matrix:
            base = scalar[0].time_per_iter
            for m in matrix:
                speedup = base / m.time_per_iter if m.time_per_iter > 0 else float('inf')
                print(f"  {game_name}: {m.name} = {speedup:.2f}x vs scalar")


if __name__ == "__main__":
    main()
