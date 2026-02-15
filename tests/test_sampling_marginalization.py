"""Tests for sampling-based marginalization approaches.

Validates that all sampling methods produce reasonable approximations
of the exact marginal beliefs, with known error bounds, and that the
reach-guided methods correctly exploit sparsity.
"""

from __future__ import annotations

import pytest
import torch

from sampling.marginalization import (
    GameConfig,
    GibbsSampler,
    ProjectionSampler,
    ReachCorrelatedSampler,
    ReachGuidedSparse,
    enumerate_deals,
    num_private_states,
)
from tests.test_marginalization import (
    ExactJointBeliefTracker,
    generate_random_policy,
    l1_distance,
    kl_divergence,
    sample_action,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_parallel_simulation(
    cfg: GameConfig,
    trackers: dict,
    num_steps: int = 8,
    policy_strength: float = 0.5,
    seed: int = 42,
):
    """Run the same action sequence through multiple trackers in parallel.

    Returns: (exact_tracker, {name: tracker})
    """
    rng = torch.Generator().manual_seed(seed)
    n_priv = num_private_states(cfg)
    exact = ExactJointBeliefTracker(cfg)

    for step in range(num_steps):
        player = step % cfg.num_players
        policy = generate_random_policy(n_priv, cfg.num_actions, rng)
        # Mix with uniform
        uniform = torch.ones_like(policy) / cfg.num_actions
        policy = (1 - policy_strength) * uniform + policy_strength * policy

        action = sample_action(player, policy, exact, rng)
        exact.observe_action(player, action, policy)
        for name, tracker in trackers.items():
            tracker.observe_action(player, action, policy)

    return exact, trackers


# ---------------------------------------------------------------------------
# Projection Sampling Tests
# ---------------------------------------------------------------------------

class TestProjectionSampler:

    def test_initial_beliefs_uniform(self):
        cfg = GameConfig(deck_size=4, num_players=2, cards_per_player=1)
        proj = ProjectionSampler(cfg)
        for p in range(cfg.num_players):
            belief = proj.marginal_belief(p)
            expected = 1.0 / num_private_states(cfg)
            for s in range(num_private_states(cfg)):
                assert abs(belief[s].item() - expected) < 1e-5

    def test_beliefs_sum_to_one(self):
        cfg = GameConfig(deck_size=6, num_players=2, cards_per_player=1, seed=7)
        proj = ProjectionSampler(cfg, num_samples=2000)
        trackers = {"proj": proj}
        exact, _ = run_parallel_simulation(cfg, trackers, num_steps=5)
        for p in range(cfg.num_players):
            belief = proj.marginal_belief(p)
            assert abs(belief.sum().item() - 1.0) < 0.02

    def test_small_game_matches_exact(self):
        """For small games, projection should fall back to exact computation."""
        cfg = GameConfig(deck_size=3, num_players=2, cards_per_player=1, seed=42)
        proj = ProjectionSampler(cfg, num_samples=100)  # More samples than deals → exact
        trackers = {"proj": proj}
        exact, _ = run_parallel_simulation(cfg, trackers, num_steps=6)
        for p in range(cfg.num_players):
            l1 = l1_distance(exact.marginal_belief(p), proj.marginal_belief(p))
            assert l1 < 0.01, f"Player {p}: L1 error {l1} too large for small game"

    def test_error_bounded(self):
        """Projection error should be bounded for moderate-sized games."""
        cfg = GameConfig(deck_size=8, num_players=2, cards_per_player=1, seed=55)
        proj = ProjectionSampler(cfg, num_samples=5000)
        trackers = {"proj": proj}
        exact, _ = run_parallel_simulation(cfg, trackers, num_steps=8, policy_strength=0.5)
        for p in range(cfg.num_players):
            l1 = l1_distance(exact.marginal_belief(p), proj.marginal_belief(p))
            assert l1 < 0.5, f"Player {p}: L1 error {l1} too large"


# ---------------------------------------------------------------------------
# Gibbs Sampling Tests
# ---------------------------------------------------------------------------

class TestGibbsSampler:

    def test_initial_beliefs_uniform(self):
        cfg = GameConfig(deck_size=4, num_players=2, cards_per_player=1)
        gibbs = GibbsSampler(cfg)
        for p in range(cfg.num_players):
            belief = gibbs.marginal_belief(p)
            expected = 1.0 / num_private_states(cfg)
            for s in range(num_private_states(cfg)):
                assert abs(belief[s].item() - expected) < 1e-5

    def test_beliefs_sum_to_one(self):
        cfg = GameConfig(deck_size=6, num_players=2, cards_per_player=1, seed=7)
        gibbs = GibbsSampler(cfg, num_samples=1000, burn_in=100)
        trackers = {"gibbs": gibbs}
        exact, _ = run_parallel_simulation(cfg, trackers, num_steps=5)
        for p in range(cfg.num_players):
            belief = gibbs.marginal_belief(p)
            assert abs(belief.sum().item() - 1.0) < 0.02

    def test_small_game_matches_exact(self):
        cfg = GameConfig(deck_size=3, num_players=2, cards_per_player=1, seed=42)
        gibbs = GibbsSampler(cfg, num_samples=100, burn_in=50)
        trackers = {"gibbs": gibbs}
        exact, _ = run_parallel_simulation(cfg, trackers, num_steps=6)
        for p in range(cfg.num_players):
            l1 = l1_distance(exact.marginal_belief(p), gibbs.marginal_belief(p))
            assert l1 < 0.01, f"Player {p}: L1 error {l1}"

    def test_3player_bounded(self):
        cfg = GameConfig(deck_size=6, num_players=3, cards_per_player=1, seed=99)
        gibbs = GibbsSampler(cfg, num_samples=2000, burn_in=200)
        trackers = {"gibbs": gibbs}
        exact, _ = run_parallel_simulation(cfg, trackers, num_steps=6, policy_strength=0.5)
        for p in range(cfg.num_players):
            l1 = l1_distance(exact.marginal_belief(p), gibbs.marginal_belief(p))
            assert l1 < 0.5, f"Player {p}: L1 error {l1}"


# ---------------------------------------------------------------------------
# Reach-Guided Sparse Tests
# ---------------------------------------------------------------------------

class TestReachGuidedSparse:

    def test_initial_beliefs_uniform(self):
        cfg = GameConfig(deck_size=4, num_players=2, cards_per_player=1)
        sparse = ReachGuidedSparse(cfg)
        for p in range(cfg.num_players):
            belief = sparse.marginal_belief(p)
            expected = 1.0 / num_private_states(cfg)
            for s in range(num_private_states(cfg)):
                assert abs(belief[s].item() - expected) < 1e-5

    def test_exact_for_all_sizes(self):
        """Sparse should be exact (only prunes truly zero-probability deals)."""
        for deck in [3, 4, 6, 8]:
            cfg = GameConfig(deck_size=deck, num_players=2, cards_per_player=1, seed=42)
            sparse = ReachGuidedSparse(cfg, prune_threshold=1e-10)
            trackers = {"sparse": sparse}
            exact, _ = run_parallel_simulation(cfg, trackers, num_steps=8, policy_strength=0.7)
            for p in range(cfg.num_players):
                l1 = l1_distance(exact.marginal_belief(p), sparse.marginal_belief(p))
                assert l1 < 1e-4, f"Deck {deck}, player {p}: L1 error {l1}"

    def test_pruning_reduces_active_count(self):
        """Strong policies should cause pruning of inactive deals."""
        cfg = GameConfig(deck_size=10, num_players=2, cards_per_player=1, num_actions=3, seed=42)
        sparse = ReachGuidedSparse(cfg, prune_threshold=1e-4)
        trackers = {"sparse": sparse}
        # Strong policies → peaked beliefs → aggressive pruning
        run_parallel_simulation(cfg, trackers, num_steps=10, policy_strength=0.9)

        initial_deals = len(enumerate_deals(cfg))
        active = sparse.active_count()
        assert active < initial_deals, (
            f"Expected pruning: active={active}, total={initial_deals}"
        )

    def test_coverage_high(self):
        """Active set should capture >99% of probability mass."""
        cfg = GameConfig(deck_size=8, num_players=2, cards_per_player=1, seed=42)
        sparse = ReachGuidedSparse(cfg, prune_threshold=1e-6)
        trackers = {"sparse": sparse}
        run_parallel_simulation(cfg, trackers, num_steps=8, policy_strength=0.7)

        cov = sparse.coverage()
        assert cov > 0.99, f"Coverage {cov} too low — pruning too aggressive"

    def test_reconstitute(self):
        """Reconstituting should restore all non-zero deals."""
        cfg = GameConfig(deck_size=6, num_players=2, cards_per_player=1, seed=42)
        sparse = ReachGuidedSparse(cfg, prune_threshold=1e-2)  # Aggressive
        trackers = {"sparse": sparse}
        run_parallel_simulation(cfg, trackers, num_steps=8, policy_strength=0.7)

        before = sparse.active_count()
        sparse.reconstitute()
        after = sparse.active_count()
        assert after >= before, "Reconstitution should not reduce active count"

    def test_3player_exact(self):
        cfg = GameConfig(deck_size=5, num_players=3, cards_per_player=1, seed=42)
        sparse = ReachGuidedSparse(cfg, prune_threshold=1e-10)
        trackers = {"sparse": sparse}
        exact, _ = run_parallel_simulation(cfg, trackers, num_steps=6)
        for p in range(cfg.num_players):
            l1 = l1_distance(exact.marginal_belief(p), sparse.marginal_belief(p))
            assert l1 < 1e-4, f"Player {p}: L1 error {l1}"


# ---------------------------------------------------------------------------
# Reach-Correlated Sampler Tests
# ---------------------------------------------------------------------------

class TestReachCorrelatedSampler:

    def test_initial_beliefs_uniform(self):
        cfg = GameConfig(deck_size=4, num_players=2, cards_per_player=1)
        corr = ReachCorrelatedSampler(cfg)
        for p in range(cfg.num_players):
            belief = corr.marginal_belief(p)
            expected = 1.0 / num_private_states(cfg)
            for s in range(num_private_states(cfg)):
                assert abs(belief[s].item() - expected) < 1e-5

    def test_matches_exact_small(self):
        """Should match exact for small games (all states are hot)."""
        cfg = GameConfig(deck_size=3, num_players=2, cards_per_player=1, seed=42)
        corr = ReachCorrelatedSampler(cfg, mass_threshold=0.99)
        trackers = {"corr": corr}
        exact, _ = run_parallel_simulation(cfg, trackers, num_steps=6)
        for p in range(cfg.num_players):
            l1 = l1_distance(exact.marginal_belief(p), corr.marginal_belief(p))
            assert l1 < 0.02, f"Player {p}: L1 error {l1}"

    def test_hot_states_decrease_with_strong_policy(self):
        """Strong policies should reduce the number of hot states."""
        cfg = GameConfig(deck_size=10, num_players=2, cards_per_player=1, num_actions=3, seed=42)
        corr = ReachCorrelatedSampler(cfg, mass_threshold=0.99)
        n_priv = num_private_states(cfg)

        # Initial: all states are hot
        initial_hot = corr.hot_state_counts()
        assert all(h == n_priv for h in initial_hot), "Initially all states should be hot"

        # Apply strong policy updates
        trackers = {"corr": corr}
        run_parallel_simulation(cfg, trackers, num_steps=10, policy_strength=0.9)

        final_hot = corr.hot_state_counts()
        assert any(h < n_priv for h in final_hot), (
            f"Expected hot count to decrease: initial={initial_hot}, final={final_hot}"
        )

    def test_effective_deal_count(self):
        """Effective deal count should be less than total for peaked beliefs."""
        cfg = GameConfig(deck_size=10, num_players=2, cards_per_player=1, num_actions=3, seed=42)
        corr = ReachCorrelatedSampler(cfg, mass_threshold=0.95)
        trackers = {"corr": corr}
        run_parallel_simulation(cfg, trackers, num_steps=10, policy_strength=0.9)

        total_deals = len(enumerate_deals(cfg))
        effective = corr.effective_deal_count()
        assert effective < total_deals, (
            f"Effective deals {effective} should be less than total {total_deals}"
        )

    def test_error_bounded_medium(self):
        cfg = GameConfig(deck_size=8, num_players=2, cards_per_player=1, seed=55)
        corr = ReachCorrelatedSampler(cfg, mass_threshold=0.99)
        trackers = {"corr": corr}
        exact, _ = run_parallel_simulation(cfg, trackers, num_steps=8, policy_strength=0.5)
        for p in range(cfg.num_players):
            l1 = l1_distance(exact.marginal_belief(p), corr.marginal_belief(p))
            assert l1 < 0.1, f"Player {p}: L1 error {l1}"


# ---------------------------------------------------------------------------
# Comparative Tests
# ---------------------------------------------------------------------------

class TestComparative:
    """Compare all methods against each other and exact ground truth."""

    def test_all_methods_bounded_kuhn(self):
        """All methods should be close to exact for Kuhn-sized games."""
        cfg = GameConfig(deck_size=3, num_players=2, cards_per_player=1, seed=42)
        trackers = {
            "proj": ProjectionSampler(cfg, num_samples=1000),
            "gibbs": GibbsSampler(cfg, num_samples=500, burn_in=100),
            "sparse": ReachGuidedSparse(cfg, prune_threshold=1e-10),
            "corr": ReachCorrelatedSampler(cfg, mass_threshold=0.99),
        }
        exact, _ = run_parallel_simulation(cfg, trackers, num_steps=6)

        for name, tracker in trackers.items():
            for p in range(cfg.num_players):
                l1 = l1_distance(exact.marginal_belief(p), tracker.marginal_belief(p))
                assert l1 < 0.05, f"{name}, player {p}: L1 error {l1}"

    def test_sparse_is_exact_with_tight_threshold(self):
        """ReachGuidedSparse with tight threshold should match exact."""
        for seed in range(5):
            cfg = GameConfig(deck_size=6, num_players=2, cards_per_player=1, seed=seed)
            trackers = {
                "sparse": ReachGuidedSparse(cfg, prune_threshold=1e-12),
            }
            exact, _ = run_parallel_simulation(cfg, trackers, num_steps=8)
            for p in range(cfg.num_players):
                l1 = l1_distance(exact.marginal_belief(p), trackers["sparse"].marginal_belief(p))
                assert l1 < 1e-5, f"Seed {seed}, player {p}: L1 error {l1}"

    def test_correlated_is_exact_with_full_mass(self):
        """ReachCorrelatedSampler with mass_threshold=1.0 should match exact."""
        for seed in range(5):
            cfg = GameConfig(deck_size=6, num_players=2, cards_per_player=1, seed=seed)
            trackers = {
                "corr": ReachCorrelatedSampler(cfg, mass_threshold=1.0),
            }
            exact, _ = run_parallel_simulation(cfg, trackers, num_steps=8)
            for p in range(cfg.num_players):
                l1 = l1_distance(exact.marginal_belief(p), trackers["corr"].marginal_belief(p))
                assert l1 < 1e-5, f"Seed {seed}, player {p}: L1 error {l1}"

    def test_deterministic_policy_all_methods(self):
        """When a player's card is revealed, all methods should handle it correctly."""
        cfg = GameConfig(deck_size=4, num_players=2, cards_per_player=1, num_actions=4)
        policy = torch.eye(4)  # Deterministic: state i → action i

        trackers = {
            "proj": ProjectionSampler(cfg, num_samples=1000),
            "gibbs": GibbsSampler(cfg, num_samples=500, burn_in=100),
            "sparse": ReachGuidedSparse(cfg),
            "corr": ReachCorrelatedSampler(cfg),
        }
        exact = ExactJointBeliefTracker(cfg)

        # Observe player 0 taking action 2 → holds card 2
        exact.observe_action(0, 2, policy)
        for name, tracker in trackers.items():
            tracker.observe_action(0, 2, policy)

        # Check: player 0 should be concentrated on card 2
        for name, tracker in trackers.items():
            belief = tracker.marginal_belief(0)
            assert belief[2].item() > 0.9, f"{name}: player 0 belief[2] = {belief[2].item()}"

        # Sparse should also handle player 1 correctly (card 2 excluded)
        exact_b1 = exact.marginal_belief(1)
        sparse_b1 = trackers["sparse"].marginal_belief(1)
        assert sparse_b1[2].item() < 0.01, (
            f"Sparse: player 1 should not hold card 2, got {sparse_b1[2].item()}"
        )


# ---------------------------------------------------------------------------
# Sparsity and Performance Tests
# ---------------------------------------------------------------------------

class TestSparsityBehavior:
    """Test that sparse methods exploit structure appropriately."""

    def test_stronger_policies_more_pruning(self):
        """Stronger policies → more peaked beliefs → more pruning."""
        cfg = GameConfig(deck_size=10, num_players=2, cards_per_player=1, num_actions=3)

        active_counts = []
        for strength in [0.1, 0.5, 0.9]:
            sparse = ReachGuidedSparse(cfg, prune_threshold=1e-4)
            trackers = {"sparse": sparse}
            run_parallel_simulation(
                cfg, trackers, num_steps=10, policy_strength=strength, seed=42
            )
            active_counts.append(sparse.active_count())

        # Stronger policies should lead to fewer active deals
        assert active_counts[2] <= active_counts[0], (
            f"Expected stronger policies to prune more: {active_counts}"
        )

    def test_correlated_fewer_hot_states_with_strength(self):
        """Stronger policies → fewer hot states per player."""
        cfg = GameConfig(deck_size=10, num_players=2, cards_per_player=1, num_actions=3)

        hot_counts = []
        for strength in [0.1, 0.9]:
            corr = ReachCorrelatedSampler(cfg, mass_threshold=0.95)
            trackers = {"corr": corr}
            run_parallel_simulation(
                cfg, trackers, num_steps=10, policy_strength=strength, seed=42
            )
            hot_counts.append(max(corr.hot_state_counts()))

        assert hot_counts[1] <= hot_counts[0], (
            f"Expected fewer hot states with stronger policies: {hot_counts}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
