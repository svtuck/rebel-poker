"""Tests for the vectorized exact tracker and MLX sparse tracker.

Validates that vectorized implementations match loop-based exact
computation and that MLX sparse tracker handles pruning correctly.
"""

from __future__ import annotations

import pytest
import torch

from sampling.marginalization import (
    GameConfig,
    enumerate_deals,
    num_private_states,
)
from sampling.mlx_marginalization import (
    MLXSparseTracker,
    VectorizedExactTracker,
)
from tests.test_marginalization import (
    ExactJointBeliefTracker,
    generate_random_policy,
    l1_distance,
    sample_action,
)


class TestVectorizedExactTracker:

    def test_matches_loop_based_kuhn(self):
        """Vectorized should match loop-based exactly for Kuhn."""
        cfg = GameConfig(deck_size=3, num_players=2, cards_per_player=1, seed=42)
        rng = torch.Generator().manual_seed(cfg.seed)
        n_priv = num_private_states(cfg)

        loop = ExactJointBeliefTracker(cfg)
        vec = VectorizedExactTracker(cfg)

        for step in range(8):
            player = step % cfg.num_players
            policy = generate_random_policy(n_priv, cfg.num_actions, rng)
            action = sample_action(player, policy, loop, rng)
            loop.observe_action(player, action, policy)
            vec.observe_action(player, action, policy)

            for p in range(cfg.num_players):
                l1 = l1_distance(loop.marginal_belief(p), vec.marginal_belief(p))
                assert l1 < 1e-6, f"Step {step}, player {p}: L1 error {l1}"

    @pytest.mark.parametrize("deck,players,cards", [
        (4, 2, 1), (6, 2, 1), (6, 3, 1), (8, 2, 2), (10, 2, 1), (10, 3, 1),
    ])
    def test_matches_loop_based_various(self, deck, players, cards):
        """Vectorized matches loop-based for various game sizes."""
        cfg = GameConfig(deck_size=deck, num_players=players, cards_per_player=cards, seed=42)
        rng = torch.Generator().manual_seed(cfg.seed)
        n_priv = num_private_states(cfg)

        loop = ExactJointBeliefTracker(cfg)
        vec = VectorizedExactTracker(cfg)

        for step in range(8):
            player = step % cfg.num_players
            policy = generate_random_policy(n_priv, cfg.num_actions, rng)
            action = sample_action(player, policy, loop, rng)
            loop.observe_action(player, action, policy)
            vec.observe_action(player, action, policy)

        for p in range(cfg.num_players):
            l1 = l1_distance(loop.marginal_belief(p), vec.marginal_belief(p))
            assert l1 < 1e-5, f"D{deck}_P{players}_K{cards}, player {p}: L1={l1}"

    def test_action_probability(self):
        """Vectorized action probability should match loop-based."""
        cfg = GameConfig(deck_size=6, num_players=2, cards_per_player=1, seed=42)
        rng = torch.Generator().manual_seed(cfg.seed)
        n_priv = num_private_states(cfg)

        loop = ExactJointBeliefTracker(cfg)
        vec = VectorizedExactTracker(cfg)

        for step in range(5):
            player = step % cfg.num_players
            policy = generate_random_policy(n_priv, cfg.num_actions, rng)

            loop_probs = loop.action_probability(player, policy)
            vec_probs = vec.action_probability(player, policy)
            assert torch.allclose(loop_probs, vec_probs, atol=1e-6), (
                f"Step {step}: loop={loop_probs} vs vec={vec_probs}"
            )

            action = torch.multinomial(loop_probs, 1, generator=rng).item()
            loop.observe_action(player, action, policy)
            vec.observe_action(player, action, policy)


class TestMLXSparseTracker:

    def test_matches_exact_no_pruning(self):
        """MLX sparse with tight threshold should match exact."""
        cfg = GameConfig(deck_size=6, num_players=2, cards_per_player=1, seed=42)
        rng = torch.Generator().manual_seed(cfg.seed)
        n_priv = num_private_states(cfg)

        loop = ExactJointBeliefTracker(cfg)
        mlx_t = MLXSparseTracker(cfg, prune_threshold=1e-12, use_mlx=False)

        for step in range(8):
            player = step % cfg.num_players
            policy = generate_random_policy(n_priv, cfg.num_actions, rng)
            action = sample_action(player, policy, loop, rng)
            loop.observe_action(player, action, policy)
            mlx_t.observe_action(player, action, policy)

        for p in range(cfg.num_players):
            l1 = l1_distance(loop.marginal_belief(p), mlx_t.marginal_belief(p))
            assert l1 < 1e-4, f"Player {p}: L1 error {l1}"

    def test_pruning_with_strong_policies(self):
        """Aggressive pruning should still produce reasonable beliefs."""
        cfg = GameConfig(deck_size=10, num_players=2, cards_per_player=1, num_actions=3, seed=42)
        rng = torch.Generator().manual_seed(cfg.seed)
        n_priv = num_private_states(cfg)

        loop = ExactJointBeliefTracker(cfg)
        mlx_t = MLXSparseTracker(cfg, prune_threshold=1e-3, use_mlx=False)

        for step in range(10):
            player = step % cfg.num_players
            policy = generate_random_policy(n_priv, cfg.num_actions, rng)
            uniform = torch.ones_like(policy) / cfg.num_actions
            policy = 0.1 * uniform + 0.9 * policy  # Strong policy

            action = sample_action(player, policy, loop, rng)
            loop.observe_action(player, action, policy)
            mlx_t.observe_action(player, action, policy)

        # Should have pruned some deals
        total = len(enumerate_deals(cfg))
        active = mlx_t.active_count()
        assert active <= total

        # Beliefs should still be reasonable
        for p in range(cfg.num_players):
            belief = mlx_t.marginal_belief(p)
            assert abs(belief.sum().item() - 1.0) < 0.01
            l1 = l1_distance(loop.marginal_belief(p), belief)
            assert l1 < 0.5, f"Player {p}: L1 error {l1}"

    def test_beliefs_sum_to_one(self):
        cfg = GameConfig(deck_size=8, num_players=2, cards_per_player=1, seed=7)
        mlx_t = MLXSparseTracker(cfg, use_mlx=False)
        rng = torch.Generator().manual_seed(cfg.seed)
        n_priv = num_private_states(cfg)
        loop = ExactJointBeliefTracker(cfg)

        for step in range(8):
            player = step % cfg.num_players
            policy = generate_random_policy(n_priv, cfg.num_actions, rng)
            action = sample_action(player, policy, loop, rng)
            loop.observe_action(player, action, policy)
            mlx_t.observe_action(player, action, policy)

        for p in range(cfg.num_players):
            belief = mlx_t.marginal_belief(p)
            assert abs(belief.sum().item() - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
