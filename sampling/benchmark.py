"""Benchmark: exact vs sampling marginalization as game size scales.

Measures accuracy (L1/KL divergence from exact) and compute cost (wall time)
for each marginalization approach across game configurations from Kuhn-sized
to large-deck games that stress exact computation.

Usage:
    python -m sampling.benchmark
    python -m sampling.benchmark --max-deck 20 --trials 5
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from sampling.marginalization import (
    GameConfig,
    GibbsSampler,
    ProjectionSampler,
    ReachCorrelatedSampler,
    ReachGuidedSparse,
    enumerate_deals,
    num_private_states,
    num_total_deals,
)

# Import exact tracker from test suite for ground truth
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tests.test_marginalization import (
    ExactJointBeliefTracker,
    generate_random_policy,
    l1_distance,
    kl_divergence,
)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    config_name: str
    deck_size: int
    num_players: int
    num_deals: int
    method: str
    l1_error: float
    kl_error: float
    wall_time_ms: float
    sparsity: float  # Fraction of deals used/sampled


def run_simulation(
    cfg: GameConfig, num_steps: int, rng: torch.Generator
) -> Tuple[List[Tuple[int, int, torch.Tensor]], ExactJointBeliefTracker]:
    """Pre-generate a sequence of actions using exact beliefs.

    Returns the action sequence and the exact tracker for ground truth.
    """
    n_priv = num_private_states(cfg)
    exact = ExactJointBeliefTracker(cfg)
    actions = []

    for step in range(num_steps):
        player = step % cfg.num_players
        policy = generate_random_policy(n_priv, cfg.num_actions, rng)

        # Sample action from exact distribution
        action_probs = exact.action_probability(player, policy)
        if action_probs.sum() < 1e-12:
            action = torch.randint(cfg.num_actions, (1,), generator=rng).item()
        else:
            action = torch.multinomial(action_probs, 1, generator=rng).item()

        exact.observe_action(player, action, policy)
        actions.append((player, action, policy))

    return actions, exact


def benchmark_method(
    method_name: str,
    tracker,
    actions: List[Tuple[int, int, torch.Tensor]],
    exact: ExactJointBeliefTracker,
    cfg: GameConfig,
) -> BenchmarkResult:
    """Benchmark a single method against exact ground truth."""
    start = time.perf_counter()

    for player, action, policy in actions:
        tracker.observe_action(player, action, policy)

    # Compute marginal beliefs and measure error
    max_l1 = 0.0
    max_kl = 0.0
    for p in range(cfg.num_players):
        approx_b = tracker.marginal_belief(p)
        exact_b = exact.marginal_belief(p)
        l1 = l1_distance(approx_b, exact_b)
        kl = kl_divergence(exact_b, approx_b)
        max_l1 = max(max_l1, l1)
        max_kl = max(max_kl, kl)

    elapsed_ms = (time.perf_counter() - start) * 1000

    sparsity = tracker.sparsity_ratio() if hasattr(tracker, 'sparsity_ratio') else 1.0

    return BenchmarkResult(
        config_name=f"D{cfg.deck_size}_P{cfg.num_players}_K{cfg.cards_per_player}",
        deck_size=cfg.deck_size,
        num_players=cfg.num_players,
        num_deals=len(enumerate_deals(cfg)),
        method=method_name,
        l1_error=max_l1,
        kl_error=max_kl,
        wall_time_ms=elapsed_ms,
        sparsity=sparsity,
    )


def benchmark_exact(
    actions: List[Tuple[int, int, torch.Tensor]],
    cfg: GameConfig,
) -> BenchmarkResult:
    """Benchmark exact marginalization (baseline)."""
    exact = ExactJointBeliefTracker(cfg)
    start = time.perf_counter()
    for player, action, policy in actions:
        exact.observe_action(player, action, policy)
    for p in range(cfg.num_players):
        exact.marginal_belief(p)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return BenchmarkResult(
        config_name=f"D{cfg.deck_size}_P{cfg.num_players}_K{cfg.cards_per_player}",
        deck_size=cfg.deck_size,
        num_players=cfg.num_players,
        num_deals=len(enumerate_deals(cfg)),
        method="exact",
        l1_error=0.0,
        kl_error=0.0,
        wall_time_ms=elapsed_ms,
        sparsity=1.0,
    )


def run_benchmark_suite(
    max_deck: int = 15,
    num_trials: int = 3,
    num_steps: int = 10,
    policy_strength: float = 0.7,
) -> List[BenchmarkResult]:
    """Run full benchmark suite across game sizes and methods."""
    results = []

    # Game configurations from small to large
    configs = [
        # Kuhn-like
        GameConfig(deck_size=3, num_players=2, cards_per_player=1, num_actions=2),
        # Slightly larger
        GameConfig(deck_size=6, num_players=2, cards_per_player=1, num_actions=3),
        # Leduc-like (2-player)
        GameConfig(deck_size=6, num_players=2, cards_per_player=1, num_actions=4),
        # 3-player
        GameConfig(deck_size=6, num_players=3, cards_per_player=1, num_actions=2),
        # Larger deck
        GameConfig(deck_size=10, num_players=2, cards_per_player=1, num_actions=3),
        # Large 3-player
        GameConfig(deck_size=10, num_players=3, cards_per_player=1, num_actions=2),
        # 2-card hands
        GameConfig(deck_size=8, num_players=2, cards_per_player=2, num_actions=2),
    ]

    # Filter by max_deck
    configs = [c for c in configs if c.deck_size <= max_deck]

    for cfg in configs:
        n_deals = num_total_deals(cfg)
        config_name = f"D{cfg.deck_size}_P{cfg.num_players}_K{cfg.cards_per_player}"
        print(f"\n{'='*60}")
        print(f"Config: {config_name} ({n_deals} deals, {num_private_states(cfg)} private states)")
        print(f"{'='*60}")

        for trial in range(num_trials):
            rng = torch.Generator().manual_seed(cfg.seed + trial * 1000)

            # Generate action sequence with moderately informative policies
            n_priv = num_private_states(cfg)
            exact_tracker = ExactJointBeliefTracker(cfg)
            actions = []

            for step in range(num_steps):
                player = step % cfg.num_players
                policy = generate_random_policy(n_priv, cfg.num_actions, rng)
                # Mix with uniform to control informativeness
                uniform = torch.ones_like(policy) / cfg.num_actions
                policy = (1 - policy_strength) * uniform + policy_strength * policy

                action_probs = exact_tracker.action_probability(player, policy)
                if action_probs.sum() < 1e-12:
                    action = torch.randint(cfg.num_actions, (1,), generator=rng).item()
                else:
                    action = torch.multinomial(action_probs, 1, generator=rng).item()

                exact_tracker.observe_action(player, action, policy)
                actions.append((player, action, policy))

            # Exact baseline
            r = benchmark_exact(actions, cfg)
            results.append(r)

            # Projection sampling
            proj = ProjectionSampler(cfg, num_samples=min(n_deals * 10, 5000))
            r = benchmark_method("projection", proj, actions, exact_tracker, cfg)
            results.append(r)

            # Gibbs sampling
            gibbs = GibbsSampler(
                cfg,
                num_samples=min(n_deals * 5, 2000),
                burn_in=min(n_deals, 200),
            )
            r = benchmark_method("gibbs", gibbs, actions, exact_tracker, cfg)
            results.append(r)

            # Reach-guided sparse
            sparse = ReachGuidedSparse(cfg, prune_threshold=1e-6)
            r = benchmark_method("sparse", sparse, actions, exact_tracker, cfg)
            results.append(r)

            # Reach-correlated sampler
            corr = ReachCorrelatedSampler(cfg, mass_threshold=0.99)
            r = benchmark_method("correlated", corr, actions, exact_tracker, cfg)
            results.append(r)

    return results


def print_results(results: List[BenchmarkResult]):
    """Print results as a formatted table."""
    # Group by config
    configs = {}
    for r in results:
        key = r.config_name
        if key not in configs:
            configs[key] = {}
        if r.method not in configs[key]:
            configs[key][r.method] = []
        configs[key][r.method].append(r)

    print(f"\n{'='*100}")
    print(f"{'Config':<20} {'Method':<15} {'Deals':>8} {'L1 Error':>10} {'KL Error':>10} "
          f"{'Time(ms)':>10} {'Sparsity':>10}")
    print(f"{'='*100}")

    for config_name in sorted(configs.keys()):
        methods = configs[config_name]
        for method_name in ["exact", "projection", "gibbs", "sparse", "correlated"]:
            if method_name not in methods:
                continue
            runs = methods[method_name]
            avg_l1 = sum(r.l1_error for r in runs) / len(runs)
            avg_kl = sum(r.kl_error for r in runs) / len(runs)
            avg_time = sum(r.wall_time_ms for r in runs) / len(runs)
            avg_sparsity = sum(r.sparsity for r in runs) / len(runs)
            n_deals = runs[0].num_deals

            print(f"{config_name:<20} {method_name:<15} {n_deals:>8} {avg_l1:>10.6f} "
                  f"{avg_kl:>10.6f} {avg_time:>10.2f} {avg_sparsity:>10.4f}")
        print(f"{'-'*100}")


def print_scaling_analysis(results: List[BenchmarkResult]):
    """Print analysis of how methods scale with game size."""
    print(f"\n{'='*80}")
    print("SCALING ANALYSIS: How do sampling methods compare as games get bigger?")
    print(f"{'='*80}")

    # Group by method
    methods: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        if r.method not in methods:
            methods[r.method] = []
        methods[r.method].append(r)

    for method_name in ["exact", "sparse", "correlated", "projection", "gibbs"]:
        if method_name not in methods:
            continue
        runs = methods[method_name]

        # Group by config
        by_config: Dict[str, List[BenchmarkResult]] = {}
        for r in runs:
            if r.config_name not in by_config:
                by_config[r.config_name] = []
            by_config[r.config_name].append(r)

        print(f"\n{method_name.upper()}:")
        for config in sorted(by_config.keys()):
            config_runs = by_config[config]
            avg_l1 = sum(r.l1_error for r in config_runs) / len(config_runs)
            avg_time = sum(r.wall_time_ms for r in config_runs) / len(config_runs)
            avg_sparsity = sum(r.sparsity for r in config_runs) / len(config_runs)
            n_deals = config_runs[0].num_deals
            print(f"  {config:<20} deals={n_deals:<8} L1={avg_l1:.6f}  "
                  f"time={avg_time:.1f}ms  sparsity={avg_sparsity:.4f}")


def print_correlation_analysis(
    num_steps: int = 10,
    policy_strengths: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
):
    """Analyze how policy correlation affects sparsity and accuracy.

    When policies are more correlated (stronger signals), the reach
    distribution becomes more peaked, and sparse methods should be
    more effective.
    """
    print(f"\n{'='*80}")
    print("CORRELATION ANALYSIS: How does policy strength affect sparse methods?")
    print(f"{'='*80}")

    cfg = GameConfig(deck_size=8, num_players=2, cards_per_player=1, num_actions=3)
    n_priv = num_private_states(cfg)

    print(f"\nConfig: D={cfg.deck_size}, P={cfg.num_players}, K={cfg.cards_per_player} "
          f"({num_total_deals(cfg)} deals)")
    print(f"{'Strength':<12} {'Sparse L1':>12} {'Sparse Active':>15} {'Sparse Coverage':>16} "
          f"{'Corr L1':>12} {'Corr Hot':>10}")

    for strength in policy_strengths:
        rng = torch.Generator().manual_seed(42)
        exact = ExactJointBeliefTracker(cfg)
        sparse = ReachGuidedSparse(cfg, prune_threshold=1e-6)
        corr = ReachCorrelatedSampler(cfg, mass_threshold=0.99)

        for step in range(num_steps):
            player = step % cfg.num_players
            policy = generate_random_policy(n_priv, cfg.num_actions, rng)
            uniform = torch.ones_like(policy) / cfg.num_actions
            policy = (1 - strength) * uniform + strength * policy

            action_probs = exact.action_probability(player, policy)
            if action_probs.sum() < 1e-12:
                action = torch.randint(cfg.num_actions, (1,), generator=rng).item()
            else:
                action = torch.multinomial(action_probs, 1, generator=rng).item()

            exact.observe_action(player, action, policy)
            sparse.observe_action(player, action, policy)
            corr.observe_action(player, action, policy)

        # Measure errors
        sparse_l1 = max(
            l1_distance(exact.marginal_belief(p), sparse.marginal_belief(p))
            for p in range(cfg.num_players)
        )
        corr_l1 = max(
            l1_distance(exact.marginal_belief(p), corr.marginal_belief(p))
            for p in range(cfg.num_players)
        )

        hot_counts = corr.hot_state_counts()

        print(f"{strength:<12.1f} {sparse_l1:>12.6f} {sparse.active_count():>15} "
              f"{sparse.coverage():>16.6f} {corr_l1:>12.6f} {str(hot_counts):>10}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark sampling marginalization")
    parser.add_argument("--max-deck", type=int, default=15, help="Max deck size")
    parser.add_argument("--trials", type=int, default=3, help="Trials per config")
    parser.add_argument("--steps", type=int, default=10, help="Action steps per trial")
    parser.add_argument("--strength", type=float, default=0.7, help="Policy strength")
    args = parser.parse_args()

    results = run_benchmark_suite(
        max_deck=args.max_deck,
        num_trials=args.trials,
        num_steps=args.steps,
        policy_strength=args.strength,
    )

    print_results(results)
    print_scaling_analysis(results)
    print_correlation_analysis()


if __name__ == "__main__":
    main()
