"""Leduc Poker solver: CFR convergence + ReBeL training demo.

Usage:
    python3 run_leduc.py              # Quick demo (1000 CFR iters)
    python3 run_leduc.py --benchmark  # Full benchmark (10K CFR iters)
    python3 run_leduc.py --rebel      # ReBeL training loop
"""

from __future__ import annotations

import argparse
import sys
import time

from leduc.game import LeducPoker, RANK_TO_STR
from cfr.solver import CFRTrainer, CFRConfig
from leduc.belief_state import (
    NUM_PRIVATE_DEALS,
    NUM_FULL_DEALS,
    NUM_PRIVATE_STATES,
)
from leduc.card_abstraction import (
    hand_strength_vector,
    preflop_bucket,
    postflop_bucket,
)


def run_cfr_demo(iterations: int = 1000) -> None:
    """Run vanilla CFR on Leduc and display results."""
    game = LeducPoker()
    print(f"Leduc Poker: {NUM_PRIVATE_DEALS} private deals, "
          f"{NUM_FULL_DEALS} full deals (with board)")
    print(f"Running vanilla CFR for {iterations} iterations...")

    trainer = CFRTrainer(game)
    t0 = time.time()
    exps = trainer.train(iterations)
    elapsed = time.time() - t0

    print(f"Completed in {elapsed:.1f}s")
    print(f"Final exploitability: {trainer.exploitability():.6f}")
    print(f"Information sets: {len(trainer.infosets)}")

    # Show some key strategies
    profile = trainer.average_strategy_profile()
    print("\n=== Key Strategy Points ===")
    interesting_keys = [
        ("Root (J, preflop)", "J-|"),
        ("Root (Q, preflop)", "Q-|"),
        ("Root (K, preflop)", "K-|"),
    ]
    for label, key in interesting_keys:
        if key in profile:
            probs = profile[key]
            prob_str = ", ".join(f"{a}:{p:.3f}" for a, p in probs.items())
            print(f"  {label}: {prob_str}")

    return trainer


def run_benchmark() -> None:
    """Run CFR variants and report convergence."""
    game = LeducPoker()
    iters = 10000
    configs = [
        ("Vanilla CFR", CFRConfig()),
        ("CFR+", CFRConfig(use_plus=True)),
        ("Linear CFR", CFRConfig(use_plus=True, linear_weighting=True)),
    ]

    print(f"=== Leduc Poker CFR Benchmark ({iters} iterations) ===")
    print(f"Deals: {NUM_PRIVATE_DEALS} private, {NUM_FULL_DEALS} full")
    print()

    results = []
    for name, config in configs:
        print(f"Running {name}...")
        trainer = CFRTrainer(game, config=config)
        t0 = time.time()
        exps = trainer.train(iters)
        elapsed = time.time() - t0
        final_exp = trainer.exploitability()
        results.append((name, final_exp, elapsed, len(trainer.infosets)))
        print(f"  {name}: exploitability={final_exp:.6f}, time={elapsed:.1f}s, "
              f"infosets={len(trainer.infosets)}")

    print("\n=== Results Summary ===")
    print(f"{'Variant':<20} {'Exploitability':<18} {'Time (s)':<12} {'Infosets':<10}")
    print("-" * 60)
    for name, exp, elapsed, n_infosets in results:
        print(f"{name:<20} {exp:<18.6f} {elapsed:<12.1f} {n_infosets:<10}")

    # Known Leduc results for comparison
    print("\n=== Reference Values ===")
    print("Known Leduc game value for P0: ~-0.0856 (varies by bet sizing)")
    print("Known exploitability after 10K iters: <0.005 (vanilla CFR)")


def run_rebel_training() -> None:
    """Run ReBeL training loop on Leduc."""
    from rebel.leduc_trainer import LeducRebelTrainer

    print("=== Leduc Poker ReBeL Training ===")
    print(f"PBS dimension: {NUM_PRIVATE_STATES} x 2 = {NUM_PRIVATE_STATES * 2}")

    trainer = LeducRebelTrainer(
        value_hidden_dim=128,
        learning_rate=1e-3,
    )

    print("Training 5 epochs (300 CFR iters each)...")
    t0 = time.time()
    metrics = trainer.train(
        num_epochs=5,
        cfr_iters_per_epoch=300,
        value_train_epochs=50,
    )
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s")
    print("\n=== Epoch Results ===")
    print(f"{'Epoch':<8} {'Exploitability':<18} {'Value Loss':<15}")
    print("-" * 41)
    for i in range(len(metrics["exploitabilities"])):
        exp = metrics["exploitabilities"][i]
        vloss = metrics["value_losses"][i] if i < len(metrics["value_losses"]) else "N/A"
        if isinstance(vloss, float):
            print(f"{i+1:<8} {exp:<18.6f} {vloss:<15.6f}")
        else:
            print(f"{i+1:<8} {exp:<18.6f} {vloss}")

    first_exp = metrics["exploitabilities"][0]
    last_exp = metrics["exploitabilities"][-1]
    if last_exp < first_exp:
        improvement = (1 - last_exp / first_exp) * 100
        print(f"\nExploitability decreased by {improvement:.1f}%")
    else:
        print("\nWarning: exploitability did not decrease")


def show_card_abstraction() -> None:
    """Display Leduc card abstraction analysis."""
    print("\n=== Card Abstraction ===")
    print("\nPreflop equity (vs uniform range):")
    for r in range(3):
        eq = hand_strength_vector()[r]
        print(f"  {RANK_TO_STR[r]}: equity={eq:.3f}, bucket={preflop_bucket(r)}")

    for board_rank in range(3):
        print(f"\nPostflop equity (board = {RANK_TO_STR[board_rank]}):")
        for r in range(3):
            eq = hand_strength_vector(board_rank)[r]
            pair = "PAIR" if r == board_rank else "    "
            bucket = postflop_bucket(r, board_rank)
            print(f"  {RANK_TO_STR[r]} {pair}: equity={eq:.3f}, bucket={bucket}")


def main():
    parser = argparse.ArgumentParser(description="Leduc Poker Solver")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run full benchmark (10K CFR iterations)")
    parser.add_argument("--rebel", action="store_true",
                        help="Run ReBeL training loop")
    parser.add_argument("--abstraction", action="store_true",
                        help="Show card abstraction analysis")
    parser.add_argument("--iterations", type=int, default=1000,
                        help="CFR iterations for demo mode")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
    elif args.rebel:
        run_rebel_training()
    elif args.abstraction:
        show_card_abstraction()
    else:
        run_cfr_demo(args.iterations)
        show_card_abstraction()


if __name__ == "__main__":
    main()
