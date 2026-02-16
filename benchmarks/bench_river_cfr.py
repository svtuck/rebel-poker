"""Benchmark: River NLHE CFR+ solving.

Default scenario: Ks Th 7s 4d 2s board, 1000 pot, 9500 stacks, 0.5x/1.0x bets.
Measures time to reach specific exploitability targets.

Usage:
    python benchmarks/bench_river_cfr.py [--iters N] [--algo cfr+|cfr|dcfr] [--backend python|rust]
"""

import argparse
import sys
import time

sys.path.insert(0, ".")

from river.game import RiverConfig, RiverHoldemGame
from river.vector_cfr import VectorCFRConfig, VectorCFRTrainer
from river.vector_eval import (
    build_blocked_indices,
    build_strength_summary,
    exploitability,
)

CHECKPOINTS = [25, 50, 100, 200, 400]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--algo", choices=["cfr", "cfr+", "dcfr"], default="cfr+")
    parser.add_argument("--board", default="Ks,Th,7s,4d,2s")
    parser.add_argument("--backend", choices=["python", "rust"], default="rust")
    args = parser.parse_args()

    board = tuple(args.board.split(","))
    config = RiverConfig(
        board=board,
        pot=1000,
        stacks=(9500, 9500),
        bet_sizes=(0.5, 1.0),
        include_all_in=True,
        max_raises=1000,
    )
    game = RiverHoldemGame(config)
    print(f"Board: {' '.join(board)}")
    print(f"Hands: P0={len(game.hands[0])}, P1={len(game.hands[1])}")

    summaries = {
        0: build_strength_summary(game.hands[1]),
        1: build_strength_summary(game.hands[0]),
    }
    blocked = {
        0: build_blocked_indices(game.hands[0], summaries[0]),
        1: build_blocked_indices(game.hands[1], summaries[1]),
    }

    algo_configs = {
        "cfr": VectorCFRConfig(use_plus=False, alternating=True),
        "cfr+": VectorCFRConfig(use_plus=True, linear_weighting=True, alternating=True),
        "dcfr": VectorCFRConfig(use_dcfr=True, alternating=True),
    }

    cfg = algo_configs[args.algo]

    use_rust = args.backend == "rust"
    rust_trainer = None
    rust_infoset_keys = None
    rust_action_tokens = None

    if use_rust:
        from river.rust_bridge import create_rust_trainer
        rust_trainer, rust_infoset_keys, rust_action_tokens = create_rust_trainer(game, cfg)
        print(f"Backend: Rust (pyo3)")
    else:
        print(f"Backend: Python")

    trainer = None if use_rust else VectorCFRTrainer(game, cfg)

    checkpoints = [c for c in CHECKPOINTS if c <= args.iters]
    if args.iters not in checkpoints:
        checkpoints.append(args.iters)

    completed = 0
    start = time.time()
    print(f"\nAlgorithm: {args.algo}")
    print(f"{'Iters':>8}  {'Exploit (chips)':>16}  {'Exploit (%)':>12}  {'Time (s)':>10}")
    print("-" * 54)

    for cp in checkpoints:
        if use_rust:
            rust_trainer.run(cp - completed)
            completed = cp
            exp, base_pot = rust_trainer.exploitability(game.base_pot)
        else:
            trainer.run(cp - completed)
            completed = cp
            profile = trainer.average_strategy_profile()
            exp = exploitability(game, profile, summaries, blocked)
        elapsed = time.time() - start
        pct = exp / game.base_pot * 100
        print(f"{cp:>8d}  {exp:>16.6f}  {pct:>11.4f}%  {elapsed:>10.1f}")

    total = time.time() - start
    print(f"\nTotal time: {total:.1f}s for {completed} iterations")
    print(f"Per iteration: {total/completed*1000:.1f}ms")


if __name__ == "__main__":
    main()
