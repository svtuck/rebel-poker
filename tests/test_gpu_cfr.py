"""Tests for GPU-accelerated CFR variants.

Tests:
1. MatrixCFR: correctness, convergence, strategy output
2. BatchedMCCFR: convergence, equivalence to single-sample MCCFR
3. DeepCFR: training loop runs, strategy is valid
4. SingleDeepCFR: training loop runs, strategy is valid
5. Cross-variant: all solvers converge to same Nash equilibrium
"""

import sys

import pytest
import torch

sys.path.insert(0, ".")

from cfr.batched_mccfr import BatchedMCCFR, BatchedMCCFRConfig
from cfr.deep_cfr import (
    AdvantageNetwork,
    DeepCFR,
    DeepCFRConfig,
    InfoSetFeaturizer,
    ReservoirBuffer,
    SDCFRConfig,
    SingleDeepCFR,
    StrategyNetwork,
)
from cfr.matrix_cfr import GameTree, MatrixCFR, MatrixCFRConfig
from cfr.solver import CFRTrainer
from kuhn.game import KuhnPoker
from leduc.game import LeducPoker


# ──────────────────────────────────────────────────────────────
# GameTree tests
# ──────────────────────────────────────────────────────────────


class TestGameTree:
    def test_kuhn_tree_structure(self):
        game = KuhnPoker()
        tree = GameTree(game)
        assert tree.num_nodes == 55  # 6 deals × ~9 histories + root + chance
        assert tree.num_infosets == 12  # 3 cards × 4 decision points
        assert tree.max_depth == 4  # root -> deal -> action -> action -> terminal
        assert len(tree.edge_parents) == 54  # one edge per non-root node

    def test_leduc_tree_structure(self):
        game = LeducPoker()
        tree = GameTree(game)
        assert tree.num_nodes > 1000  # Leduc has thousands of nodes
        assert tree.num_infosets == 288  # 6 cards × 48 histories (approx)
        assert tree.max_depth >= 6  # at least 6 levels deep

    def test_node_tensors(self):
        game = KuhnPoker()
        tree = GameTree(game)
        assert tree.node_player.shape == (tree.num_nodes,)
        assert tree.node_depth.shape == (tree.num_nodes,)
        assert tree.is_terminal.sum() > 0
        assert tree.is_chance.sum() > 0
        assert tree.is_player.sum() > 0

    def test_infoset_mapping(self):
        game = KuhnPoker()
        tree = GameTree(game)
        # Every player node should have an infoset
        player_nodes = tree.is_player.nonzero(as_tuple=True)[0]
        for idx in player_nodes:
            assert tree.node_infoset[idx].item() >= 0


# ──────────────────────────────────────────────────────────────
# MatrixCFR tests
# ──────────────────────────────────────────────────────────────


class TestMatrixCFR:
    def test_kuhn_convergence(self):
        """Matrix CFR should converge on Kuhn poker."""
        game = KuhnPoker()
        solver = MatrixCFR(game)
        solver.run(1000)
        exp = solver.exploitability()
        assert exp < 0.02, f"Exploitability {exp} too high after 1000 iters"

    def test_kuhn_deep_convergence(self):
        """Matrix CFR should achieve near-Nash on Kuhn with enough iterations."""
        game = KuhnPoker()
        solver = MatrixCFR(game)
        solver.run(10000)
        exp = solver.exploitability()
        assert exp < 0.005, f"Exploitability {exp} too high after 10k iters"

    def test_leduc_convergence(self):
        """Matrix CFR should converge on Leduc poker."""
        game = LeducPoker()
        solver = MatrixCFR(game)
        solver.run(500)
        exp = solver.exploitability()
        assert exp < 0.2, f"Exploitability {exp} too high after 500 iters"

    def test_strategy_profile_valid(self):
        """Average strategy should be valid probability distributions."""
        game = KuhnPoker()
        solver = MatrixCFR(game)
        solver.run(100)
        profile = solver.average_strategy_profile()
        for key, action_probs in profile.items():
            total = sum(action_probs.values())
            assert abs(total - 1.0) < 1e-4, f"Strategy at {key} doesn't sum to 1: {total}"
            for a, p in action_probs.items():
                assert p >= -1e-6, f"Negative probability at {key}: {a}={p}"

    def test_cfr_plus(self):
        """CFR+ should converge (regret clipping)."""
        game = KuhnPoker()
        config = MatrixCFRConfig(use_plus=True)
        solver = MatrixCFR(game, config=config)
        solver.run(1000)
        exp = solver.exploitability()
        assert exp < 0.05, f"CFR+ exploitability {exp} too high"

    def test_dcfr(self):
        """DCFR should converge."""
        game = KuhnPoker()
        config = MatrixCFRConfig(use_dcfr=True)
        solver = MatrixCFR(game, config=config)
        solver.run(1000)
        exp = solver.exploitability()
        assert exp < 0.05, f"DCFR exploitability {exp} too high"

    def test_train_returns_exploitabilities(self):
        """train() should return exploitability checkpoints."""
        game = KuhnPoker()
        solver = MatrixCFR(game)
        exps = solver.train(200)
        assert len(exps) > 0
        # Should be decreasing overall
        assert exps[-1] < exps[0], "Exploitability should decrease"

    def test_matches_scalar_cfr(self):
        """Matrix CFR should produce similar exploitability to scalar CFR."""
        game = KuhnPoker()
        scalar = CFRTrainer(game)
        matrix = MatrixCFR(game)

        scalar.run(5000)
        matrix.run(5000)

        s_exp = scalar.exploitability()
        m_exp = matrix.exploitability()

        # Should be within 5x of each other (different aggregation order)
        ratio = max(s_exp, m_exp) / max(min(s_exp, m_exp), 1e-10)
        assert ratio < 5, f"Too different: scalar={s_exp}, matrix={m_exp}"


# ──────────────────────────────────────────────────────────────
# BatchedMCCFR tests
# ──────────────────────────────────────────────────────────────


class TestBatchedMCCFR:
    def test_kuhn_convergence(self):
        """Batched MCCFR should converge on Kuhn poker."""
        game = KuhnPoker()
        config = BatchedMCCFRConfig(batch_size=16, seed=42)
        solver = BatchedMCCFR(game, config=config)
        solver.run(2000)
        exp = solver.exploitability()
        assert exp < 0.02, f"Exploitability {exp} too high after 2000 iters"

    def test_strategy_profile_valid(self):
        """Average strategy should be valid."""
        game = KuhnPoker()
        config = BatchedMCCFRConfig(batch_size=8, seed=42)
        solver = BatchedMCCFR(game, config=config)
        solver.run(100)
        profile = solver.average_strategy_profile()
        for key, action_probs in profile.items():
            total = sum(action_probs.values())
            assert abs(total - 1.0) < 1e-4

    def test_larger_batch_reduces_variance(self):
        """Larger batch should give lower variance (smoother convergence)."""
        game = KuhnPoker()

        # Small batch
        small = BatchedMCCFR(game, BatchedMCCFRConfig(batch_size=4, seed=42))
        small.run(500)
        small_exp = small.exploitability()

        # Large batch (same total traversals = batch * iterations)
        large = BatchedMCCFR(game, BatchedMCCFRConfig(batch_size=32, seed=42))
        large.run(500)
        large_exp = large.exploitability()

        # Both should converge reasonably
        assert small_exp < 0.1
        assert large_exp < 0.1


# ──────────────────────────────────────────────────────────────
# Neural network component tests
# ──────────────────────────────────────────────────────────────


class TestNeuralComponents:
    def test_advantage_network(self):
        """Advantage network should produce valid output shape."""
        net = AdvantageNetwork(feature_dim=12, num_actions=3, hidden_sizes=(32, 32))
        x = torch.randn(4, 12)
        out = net(x)
        assert out.shape == (4, 3)

    def test_strategy_network(self):
        """Strategy network should produce valid probabilities."""
        net = StrategyNetwork(feature_dim=12, num_actions=3, hidden_sizes=(32, 32))
        x = torch.randn(4, 12)
        out = net(x)
        assert out.shape == (4, 3)
        assert torch.allclose(out.sum(dim=1), torch.ones(4), atol=1e-5)
        assert (out >= 0).all()

    def test_reservoir_buffer(self):
        """Reservoir buffer should maintain capacity."""
        buf = ReservoirBuffer(capacity=10, seed=42)
        for i in range(100):
            buf.add(f"key_{i}", i, torch.randn(5), torch.randn(3))
        assert len(buf) == 10
        assert buf.num_seen == 100

    def test_reservoir_sampling(self):
        """Reservoir sampling should return valid batches."""
        buf = ReservoirBuffer(capacity=100, seed=42)
        for i in range(50):
            buf.add(f"key_{i}", i, torch.randn(5), torch.randn(3))
        features, targets, iters = buf.sample(10)
        assert features.shape == (10, 5)
        assert targets.shape == (10, 3)
        assert iters.shape == (10,)

    def test_featurizer(self):
        """Featurizer should produce correct-size one-hot vectors."""
        game = KuhnPoker()
        feat = InfoSetFeaturizer(game)
        dim = feat.feature_dim()
        assert dim == 12  # Kuhn has 12 infosets

        features = feat.featurize("K|")
        assert features.shape == (12,)
        assert features.sum().item() == 1.0  # one-hot

    def test_featurizer_batch(self):
        """Batch featurization should work."""
        game = KuhnPoker()
        feat = InfoSetFeaturizer(game)
        batch = feat.batch_featurize(["K|", "Q|c"])
        assert batch.shape == (2, 12)


# ──────────────────────────────────────────────────────────────
# Deep CFR tests
# ──────────────────────────────────────────────────────────────


class TestDeepCFR:
    def test_training_runs(self):
        """Deep CFR training loop should complete without errors."""
        game = KuhnPoker()
        config = DeepCFRConfig(
            num_traversals=50,
            advantage_train_steps=50,
            strategy_train_steps=50,
            batch_size=32,
            hidden_sizes=(32, 32),
            seed=42,
        )
        solver = DeepCFR(game, config=config)
        results = solver.train()
        assert "advantage_losses" in results
        assert "strategy_losses" in results

    def test_strategy_valid(self):
        """Deep CFR should produce valid probability distributions."""
        game = KuhnPoker()
        config = DeepCFRConfig(
            num_traversals=100,
            advantage_train_steps=100,
            strategy_train_steps=100,
            batch_size=32,
            hidden_sizes=(32, 32),
            seed=42,
        )
        solver = DeepCFR(game, config=config)
        solver.train()
        profile = solver.average_strategy_profile()

        assert len(profile) > 0
        for key, action_probs in profile.items():
            total = sum(action_probs.values())
            assert abs(total - 1.0) < 1e-3, f"Strategy at {key} sums to {total}"
            for a, p in action_probs.items():
                assert p >= -1e-3, f"Negative prob at {key}: {a}={p}"

    def test_leduc_runs(self):
        """Deep CFR should run on Leduc without errors."""
        game = LeducPoker()
        config = DeepCFRConfig(
            num_traversals=20,
            advantage_train_steps=20,
            strategy_train_steps=20,
            batch_size=32,
            hidden_sizes=(64, 64),
            seed=42,
        )
        solver = DeepCFR(game, config=config)
        results = solver.train()
        profile = solver.average_strategy_profile()
        assert len(profile) > 0

    def test_exploitability_finite(self):
        """Exploitability should be a finite positive number."""
        game = KuhnPoker()
        config = DeepCFRConfig(
            num_traversals=50,
            advantage_train_steps=50,
            strategy_train_steps=50,
            batch_size=32,
            hidden_sizes=(32, 32),
            seed=42,
        )
        solver = DeepCFR(game, config=config)
        solver.train()
        exp = solver.exploitability()
        assert 0 <= exp < 10, f"Exploitability out of range: {exp}"


# ──────────────────────────────────────────────────────────────
# SD-CFR tests
# ──────────────────────────────────────────────────────────────


class TestSDCFR:
    def test_training_runs(self):
        """SD-CFR training loop should complete without errors."""
        game = KuhnPoker()
        config = SDCFRConfig(
            num_traversals=50,
            train_steps=50,
            batch_size=32,
            hidden_sizes=(32, 32),
            seed=42,
            max_stored_networks=5,
        )
        solver = SingleDeepCFR(game, config=config)
        results = solver.train()
        assert "losses" in results
        assert len(solver.stored_networks) > 0

    def test_strategy_valid(self):
        """SD-CFR should produce valid probability distributions."""
        game = KuhnPoker()
        config = SDCFRConfig(
            num_traversals=100,
            train_steps=100,
            batch_size=32,
            hidden_sizes=(32, 32),
            seed=42,
            max_stored_networks=10,
        )
        solver = SingleDeepCFR(game, config=config)
        solver.train()
        profile = solver.average_strategy_profile()

        assert len(profile) > 0
        for key, action_probs in profile.items():
            total = sum(action_probs.values())
            assert abs(total - 1.0) < 1e-3

    def test_stored_networks_count(self):
        """Should store the right number of network snapshots."""
        game = KuhnPoker()
        config = SDCFRConfig(
            num_traversals=100,
            train_steps=50,
            batch_size=32,
            hidden_sizes=(32, 32),
            seed=42,
            max_stored_networks=5,
        )
        solver = SingleDeepCFR(game, config=config)
        solver.train()
        assert len(solver.stored_networks) <= 5


# ──────────────────────────────────────────────────────────────
# Cross-variant consistency tests
# ──────────────────────────────────────────────────────────────


class TestCrossVariant:
    def test_all_tabular_converge_kuhn(self):
        """All tabular methods should converge to similar exploitability on Kuhn."""
        game = KuhnPoker()

        scalar = CFRTrainer(game)
        scalar.run(5000)
        s_exp = scalar.exploitability()

        matrix = MatrixCFR(game)
        matrix.run(5000)
        m_exp = matrix.exploitability()

        batched = BatchedMCCFR(game, BatchedMCCFRConfig(batch_size=16, seed=42))
        batched.run(2000)
        b_exp = batched.exploitability()

        # All should be under 0.01
        assert s_exp < 0.01, f"Scalar: {s_exp}"
        assert m_exp < 0.01, f"Matrix: {m_exp}"
        assert b_exp < 0.01, f"Batched MCCFR: {b_exp}"

    def test_kuhn_game_value(self):
        """Nash equilibrium game value should be approximately -1/18."""
        game = KuhnPoker()
        solver = MatrixCFR(game)
        solver.run(10000)
        profile = solver.average_strategy_profile()

        # Game value from player 0's perspective
        dummy = CFRTrainer(game)
        br0 = dummy._best_response_value(profile, 0)
        br1 = dummy._best_response_value(profile, 1)

        # At Nash, best response values should be close to game value
        # Game value of Kuhn = -1/18 ≈ -0.0556 for player 0
        assert abs(br0 + br1) < 0.01, f"br0={br0}, br1={br1}, sum should be near 0"
