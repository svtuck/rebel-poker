"""Large-scale benchmark: stress-test sampling approaches at game sizes
where exact computation becomes expensive.

This benchmark pushes to larger deck sizes, multi-card hands, and 3+ players
to show where sampling approaches diverge from exact and how they scale.

Usage:
    python3 -m sampling.large_benchmark
"""

from __future__ import annotations

import time
from typing import Dict, List

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

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tests.test_marginalization import (
    ExactJointBeliefTracker,
    generate_random_policy,
    l1_distance,
    kl_divergence,
)


def format_time(ms: float) -> str:
    if ms < 1:
        return f"{ms*1000:.0f}us"
    elif ms < 1000:
        return f"{ms:.1f}ms"
    else:
        return f"{ms/1000:.2f}s"


def run_large_benchmark():
    """Benchmark at scales where exact vs sampling diverges."""

    configs = [
        # Name, GameConfig, num_steps, policy_strength
        ("Kuhn (baseline)",
         GameConfig(deck_size=3, num_players=2, cards_per_player=1, num_actions=2, seed=42),
         8, 0.7),
        ("6-card 2P",
         GameConfig(deck_size=6, num_players=2, cards_per_player=1, num_actions=3, seed=42),
         8, 0.7),
        ("10-card 2P",
         GameConfig(deck_size=10, num_players=2, cards_per_player=1, num_actions=3, seed=42),
         10, 0.7),
        ("13-card 2P (bridge-like)",
         GameConfig(deck_size=13, num_players=2, cards_per_player=1, num_actions=3, seed=42),
         10, 0.7),
        ("20-card 2P",
         GameConfig(deck_size=20, num_players=2, cards_per_player=1, num_actions=4, seed=42),
         12, 0.7),
        ("10-card 3P",
         GameConfig(deck_size=10, num_players=3, cards_per_player=1, num_actions=2, seed=42),
         10, 0.7),
        ("13-card 3P",
         GameConfig(deck_size=13, num_players=3, cards_per_player=1, num_actions=2, seed=42),
         10, 0.7),
        ("8-card 2P 2-card hands",
         GameConfig(deck_size=8, num_players=2, cards_per_player=2, num_actions=3, seed=42),
         8, 0.7),
        ("10-card 2P 2-card hands",
         GameConfig(deck_size=10, num_players=2, cards_per_player=2, num_actions=3, seed=42),
         8, 0.7),
        # Strong policies (test sparsity exploitation)
        ("13-card 2P STRONG",
         GameConfig(deck_size=13, num_players=2, cards_per_player=1, num_actions=4, seed=42),
         12, 0.95),
        ("20-card 2P STRONG",
         GameConfig(deck_size=20, num_players=2, cards_per_player=1, num_actions=4, seed=42),
         12, 0.95),
    ]

    print("=" * 120)
    print("SAMPLING MARGINALIZATION BENCHMARK")
    print("=" * 120)
    print(f"{'Game':<30} {'Deals':>8} {'Method':<15} {'L1':>10} {'KL':>10} "
          f"{'Time':>10} {'Active':>10} {'Coverage':>10}")
    print("-" * 120)

    for name, cfg, num_steps, strength in configs:
        n_deals = num_total_deals(cfg)
        actual_deals = enumerate_deals(cfg)
        n_priv = num_private_states(cfg)

        print(f"\n{name} (D={cfg.deck_size}, P={cfg.num_players}, K={cfg.cards_per_player}, "
              f"deals={len(actual_deals)}, privates={n_priv})")

        # Generate action sequence using exact tracker
        rng = torch.Generator().manual_seed(cfg.seed)
        exact = ExactJointBeliefTracker(cfg)
        actions = []

        t0 = time.perf_counter()
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
            actions.append((player, action, policy))

        exact_time = (time.perf_counter() - t0) * 1000

        # Exact baseline
        print(f"  {'Exact':<30} {len(actual_deals):>8} {'exact':<15} {'0':>10} {'0':>10} "
              f"{format_time(exact_time):>10} {len(actual_deals):>10} {'100%':>10}")

        # --- Sparse ---
        sparse = ReachGuidedSparse(cfg, prune_threshold=1e-6)
        t0 = time.perf_counter()
        for player, action, policy in actions:
            sparse.observe_action(player, action, policy)
        for p in range(cfg.num_players):
            sparse.marginal_belief(p)
        sparse_time = (time.perf_counter() - t0) * 1000

        sparse_l1 = max(
            l1_distance(exact.marginal_belief(p), sparse.marginal_belief(p))
            for p in range(cfg.num_players)
        )
        sparse_kl = max(
            kl_divergence(exact.marginal_belief(p), sparse.marginal_belief(p))
            for p in range(cfg.num_players)
        )

        print(f"  {'Sparse (1e-6)':<30} {len(actual_deals):>8} {'sparse':<15} "
              f"{sparse_l1:>10.6f} {sparse_kl:>10.6f} {format_time(sparse_time):>10} "
              f"{sparse.active_count():>10} {sparse.coverage():>10.4f}")

        # --- Sparse aggressive ---
        sparse_agg = ReachGuidedSparse(cfg, prune_threshold=1e-3)
        t0 = time.perf_counter()
        for player, action, policy in actions:
            sparse_agg.observe_action(player, action, policy)
        for p in range(cfg.num_players):
            sparse_agg.marginal_belief(p)
        sparse_agg_time = (time.perf_counter() - t0) * 1000

        sparse_agg_l1 = max(
            l1_distance(exact.marginal_belief(p), sparse_agg.marginal_belief(p))
            for p in range(cfg.num_players)
        )
        sparse_agg_kl = max(
            kl_divergence(exact.marginal_belief(p), sparse_agg.marginal_belief(p))
            for p in range(cfg.num_players)
        )

        print(f"  {'Sparse (1e-3)':<30} {len(actual_deals):>8} {'sparse-agg':<15} "
              f"{sparse_agg_l1:>10.6f} {sparse_agg_kl:>10.6f} {format_time(sparse_agg_time):>10} "
              f"{sparse_agg.active_count():>10} {sparse_agg.coverage():>10.4f}")

        # --- Correlated ---
        for threshold in [0.99, 0.95, 0.90]:
            corr = ReachCorrelatedSampler(cfg, mass_threshold=threshold)
            t0 = time.perf_counter()
            for player, action, policy in actions:
                corr.observe_action(player, action, policy)
            for p in range(cfg.num_players):
                corr.marginal_belief(p)
            corr_time = (time.perf_counter() - t0) * 1000

            corr_l1 = max(
                l1_distance(exact.marginal_belief(p), corr.marginal_belief(p))
                for p in range(cfg.num_players)
            )
            corr_kl = max(
                kl_divergence(exact.marginal_belief(p), corr.marginal_belief(p))
                for p in range(cfg.num_players)
            )
            hot = corr.hot_state_counts()
            eff = corr.effective_deal_count()

            label = f"Correlated ({threshold})"
            print(f"  {label:<30} {len(actual_deals):>8} {'correlated':<15} "
                  f"{corr_l1:>10.6f} {corr_kl:>10.6f} {format_time(corr_time):>10} "
                  f"{eff:>10} {f'hot={hot}':>10}")

        # --- Projection sampling (only when num_deals > threshold) ---
        proj = ProjectionSampler(cfg, num_samples=min(5000, len(actual_deals) * 10))
        t0 = time.perf_counter()
        for player, action, policy in actions:
            proj.observe_action(player, action, policy)
        for p in range(cfg.num_players):
            proj.marginal_belief(p)
        proj_time = (time.perf_counter() - t0) * 1000

        proj_l1 = max(
            l1_distance(exact.marginal_belief(p), proj.marginal_belief(p))
            for p in range(cfg.num_players)
        )
        proj_kl = max(
            kl_divergence(exact.marginal_belief(p), proj.marginal_belief(p))
            for p in range(cfg.num_players)
        )

        print(f"  {'Projection (5k)':<30} {len(actual_deals):>8} {'projection':<15} "
              f"{proj_l1:>10.6f} {proj_kl:>10.6f} {format_time(proj_time):>10} "
              f"{'N/A':>10} {'N/A':>10}")

        # --- Gibbs ---
        n_gibbs = min(2000, len(actual_deals) * 5)
        gibbs = GibbsSampler(cfg, num_samples=n_gibbs, burn_in=min(200, len(actual_deals)))
        t0 = time.perf_counter()
        for player, action, policy in actions:
            gibbs.observe_action(player, action, policy)
        for p in range(cfg.num_players):
            gibbs.marginal_belief(p)
        gibbs_time = (time.perf_counter() - t0) * 1000

        gibbs_l1 = max(
            l1_distance(exact.marginal_belief(p), gibbs.marginal_belief(p))
            for p in range(cfg.num_players)
        )
        gibbs_kl = max(
            kl_divergence(exact.marginal_belief(p), gibbs.marginal_belief(p))
            for p in range(cfg.num_players)
        )

        print(f"  {'Gibbs (2k+200)':<30} {len(actual_deals):>8} {'gibbs':<15} "
              f"{gibbs_l1:>10.6f} {gibbs_kl:>10.6f} {format_time(gibbs_time):>10} "
              f"{'N/A':>10} {'N/A':>10}")

    # --- Summary ---
    print("\n" + "=" * 120)
    print("KEY FINDINGS:")
    print("=" * 120)
    print("""
1. SPARSE (ReachGuidedSparse): Exact when threshold is tight (1e-6), with pruning
   that reduces active deals as policies get stronger. The key benefit: it maintains
   the exact joint but only tracks deals with non-negligible probability. Cost scales
   with the number of ACTIVE deals, not total deals.

2. CORRELATED (ReachCorrelatedSampler): Exploits the factored structure of reach
   probabilities. Only enumerates deals composed of "hot" private states — states
   that still have significant reach probability. When policies are correlated
   (strong hands bet, weak hands fold), the number of hot states per player drops
   dramatically, giving a quadratic (or cubic for 3P) reduction in enumeration cost.

3. PROJECTION: Uses per-player marginals as a proposal distribution for importance
   sampling. Falls back to exact for small games. Quality depends on how close
   the true joint is to the product of marginals — works well for large decks
   where card overlap is rare, less well for small decks.

4. GIBBS: Samples from the true joint via Markov chain. Handles correlations
   correctly but requires burn-in and mixes slowly when beliefs are peaked.
   Generally slower than sparse/correlated for the same accuracy.

RECOMMENDATION for integration with CFR/ReBeL:
- Use ReachGuidedSparse as the default — it's exact with automatic pruning
- Switch to ReachCorrelatedSampler when deals exceed ~10k (e.g., Leduc with
  community cards, or hold'em endgames)
- For NLHE-scale (1M+ deals), combine: use Correlated to identify the hot
  envelope, then run Sparse within that envelope
""")


if __name__ == "__main__":
    run_large_benchmark()
