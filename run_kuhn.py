#!/usr/bin/env python3
"""Run Kuhn Poker CFR and display results with belief states.

This is the main entry point for running the solver and inspecting
the solution, including public belief states at each decision point.
"""

from __future__ import annotations

import sys
from kuhn.game import KuhnPoker, RANK_NAMES
from cfr.solver import CFRTrainer
from cfr.vectorized import VectorizedCFR
from kuhn.belief_state import (
    BeliefStateTracker,
    ALL_DEALS,
    NUM_DEALS,
)
from rebel.data_logger import RebelDataLogger


def format_deal(deal):
    return f"({RANK_NAMES[deal[0]]},{RANK_NAMES[deal[1]]})"


def print_separator(title=""):
    print(f"\n{'='*60}")
    if title:
        print(f"  {title}")
        print(f"{'='*60}")


def run_scalar_cfr(iterations=10000):
    """Run scalar (traditional) CFR."""
    print_separator("Scalar CFR")
    trainer = CFRTrainer(KuhnPoker())
    exploitabilities = trainer.train(iterations)
    profile = trainer.average_strategy_profile()

    print(f"\nAfter {iterations} iterations:")
    print(f"  Final exploitability: {trainer.exploitability():.6f}")
    print(f"  (Nash has exploitability 0)")
    print(f"\nAverage Strategy Profile:")
    print(f"  {'Infoset':<10} {'Check/Call':>10} {'Bet/Fold':>10}")
    print(f"  {'-'*30}")

    for key in sorted(profile.keys()):
        probs = profile[key]
        parts = key.split("|")
        history = parts[1] if len(parts) > 1 else ""
        if history in ("", "c"):
            actions = ["c", "b"]
        else:
            actions = ["c", "f"]
        vals = [f"{probs.get(a, 0):.4f}" for a in actions]
        print(f"  {key:<10} {vals[0]:>10} {vals[1]:>10}")

    return trainer, profile


def run_vectorized_cfr(iterations=10000):
    """Run vectorized (tensor-based) CFR."""
    print_separator("Vectorized CFR (PyTorch)")
    vcfr = VectorizedCFR(KuhnPoker())
    exploitabilities = vcfr.train(iterations)
    profile = vcfr.average_strategy_profile()

    exp = vcfr._exploitability()
    print(f"\nAfter {iterations} iterations:")
    print(f"  Final exploitability: {exp:.6f}")
    print(f"\nAverage Strategy Profile:")
    print(f"  {'Infoset':<10} {'Check/Call':>10} {'Bet/Fold':>10}")
    print(f"  {'-'*30}")

    for key in sorted(profile.keys()):
        probs = profile[key]
        parts = key.split("|")
        history = parts[1] if len(parts) > 1 else ""
        if history in ("", "c"):
            actions = ["c", "b"]
        else:
            actions = ["c", "f"]
        vals = [f"{probs.get(a, 0):.4f}" for a in actions]
        print(f"  {key:<10} {vals[0]:>10} {vals[1]:>10}")

    return vcfr, profile


def display_belief_states(profile):
    """Display public belief states for all histories."""
    print_separator("Public Belief States")

    tracker = BeliefStateTracker()
    tracker.set_strategy_from_profile(profile)
    beliefs = tracker.compute_belief_states()
    reaches = tracker.compute_all_reach_probs()

    deal_labels = [format_deal(d) for d in ALL_DEALS]
    header = "  " + " ".join(f"{dl:>8}" for dl in deal_labels)

    histories = ["", "c", "b", "cb", "cc", "bc", "bf", "cbc", "cbf"]

    print(f"\n  Deals:  {header}")
    print(f"  {'-'*60}")

    for h in histories:
        if h not in beliefs:
            continue
        b = beliefs[h]
        label = h if h else "(root)"
        is_term = h in {"cc", "bc", "bf", "cbc", "cbf"}
        marker = " [T]" if is_term else ""

        vals = " ".join(f"{b[i].item():8.4f}" for i in range(NUM_DEALS))
        print(f"  {label:<6}{marker:<5} {vals}")

    print(f"\n  Reach Probabilities (Player 0):")
    for h in histories:
        if h not in reaches:
            continue
        r0, _ = reaches[h]
        label = h if h else "(root)"
        vals = " ".join(f"{r0[i].item():8.4f}" for i in range(NUM_DEALS))
        print(f"  {label:<10} {vals}")

    print(f"\n  Reach Probabilities (Player 1):")
    for h in histories:
        if h not in reaches:
            continue
        _, r1 = reaches[h]
        label = h if h else "(root)"
        vals = " ".join(f"{r1[i].item():8.4f}" for i in range(NUM_DEALS))
        print(f"  {label:<10} {vals}")


def display_rebel_training_data(profile):
    """Show what data would be logged for ReBeL training."""
    print_separator("ReBeL Training Data Preview")

    tracker = BeliefStateTracker()
    tracker.set_strategy_from_profile(profile)
    beliefs = tracker.compute_belief_states()
    terminal_values = tracker.compute_counterfactual_values()

    print("\n  Terminal Values (Player 0's perspective):")
    deal_labels = [format_deal(d) for d in ALL_DEALS]
    header = " ".join(f"{dl:>8}" for dl in deal_labels)
    print(f"  {'History':<10} {header}")

    for h in sorted(terminal_values.keys()):
        v = terminal_values[h]
        vals = " ".join(f"{v[i].item():8.1f}" for i in range(NUM_DEALS))
        print(f"  {h:<10} {vals}")

    print("\n  Key insight for ReBeL:")
    print("  The value network V(PBS) learns to predict these terminal values")
    print("  without actually traversing to terminal nodes. During depth-limited")
    print("  search, V(PBS) replaces the leaf evaluations, allowing CFR to run")
    print("  on a shallow subgame while still producing good strategies.")


def main():
    iterations = 10000
    if len(sys.argv) > 1:
        iterations = int(sys.argv[1])

    print(f"Running Kuhn Poker solver with {iterations} CFR iterations")
    print(f"Known Nash game value: -1/18 = {-1/18:.6f}")

    # Run scalar CFR
    scalar_trainer, scalar_profile = run_scalar_cfr(iterations)

    # Run vectorized CFR
    vcfr, vector_profile = run_vectorized_cfr(iterations)

    # Display belief states using the scalar solution
    display_belief_states(scalar_profile)

    # Show ReBeL training data
    display_rebel_training_data(scalar_profile)

    print_separator("Apple Silicon GPU Notes")
    print("""
  For custom GPU kernels on Apple Silicon Mac Studio:

  1. MLX (Apple's ML framework) - Best option
     - Native Apple Silicon support
     - Unified memory (no CPU<->GPU copies)
     - mlx.core.fast.metal_kernel() for custom Metal kernels
     - Python API similar to NumPy/JAX

  2. Metal Compute Shaders (via PyObjC or Swift)
     - Write kernels in Metal Shading Language (C++14-based)
     - Full control over GPU execution
     - Can be called from Python via PyObjC bridge

  3. PyTorch MPS Backend
     - torch.device("mps") for GPU acceleration
     - Limited custom kernel support but growing
     - Good for standard tensor operations

  Recommendation: Start with PyTorch MPS for tensor ops,
  use MLX metal_kernel() for custom operations that need
  fine-grained GPU control.
""")

    print_separator("Summary")
    print(f"  Scalar CFR exploitability:    {scalar_trainer.exploitability():.6f}")
    print(f"  Vectorized CFR exploitability: {vcfr._exploitability():.6f}")
    print(f"  Both should converge toward 0 as iterations increase.")
    print()


if __name__ == "__main__":
    main()
