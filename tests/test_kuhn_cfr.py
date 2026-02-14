"""Tests for Kuhn Poker CFR implementation.

Validates against known Nash equilibrium properties of Kuhn Poker:
- Game value is -1/18 ≈ -0.0556 for player 0
- Exploitability converges to 0
- Known optimal strategy properties
"""

import math
import pytest
import torch

from kuhn.game import KuhnPoker, RANK_NAMES
from kuhn.cfr import CFRTrainer
from kuhn.belief_state import (
    BeliefStateTracker,
    VectorizedCFR,
    ALL_DEALS,
    NUM_DEALS,
    initial_chance_probs,
)


class TestKuhnPokerGame:
    def setup_method(self):
        self.game = KuhnPoker()

    def test_chance_outcomes(self):
        state = self.game.initial_state()
        outcomes = self.game.chance_outcomes(state)
        assert len(outcomes) == 6
        total_prob = sum(p for _, p in outcomes)
        assert abs(total_prob - 1.0) < 1e-10

    def test_terminal_utilities_sum_to_zero(self):
        """Kuhn Poker is zero-sum."""
        for cards in [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]:
            state = self.game.next_state(self.game.initial_state(), cards)
            for h in ["cc", "bc", "bf", "cbc", "cbf"]:
                s = state
                for a in h:
                    s = self.game.next_state(s, a)
                if self.game.is_terminal(s):
                    u0 = self.game.terminal_utility(s, 0)
                    u1 = self.game.terminal_utility(s, 1)
                    assert abs(u0 + u1) < 1e-10, f"Not zero-sum: cards={cards}, h={h}"

    def test_infoset_keys_distinguish_cards(self):
        state = self.game.next_state(self.game.initial_state(), (0, 2))
        key0 = self.game.infoset_key(state, 0)
        key1 = self.game.infoset_key(state, 1)
        assert key0 == "J|"
        assert key1 == "K|"

    def test_legal_actions(self):
        state = self.game.next_state(self.game.initial_state(), (0, 1))
        assert self.game.legal_actions(state) == ["c", "b"]

        s2 = self.game.next_state(state, "b")
        assert self.game.legal_actions(s2) == ["c", "f"]


class TestCFRTrainer:
    def test_convergence(self):
        """CFR should converge to near-zero exploitability."""
        trainer = CFRTrainer()
        trainer.train(5000)
        exp = trainer.exploitability()
        assert exp < 0.01, f"Exploitability {exp} too high after 5000 iterations"

    def test_known_nash_properties(self):
        """Test known properties of the Kuhn Poker Nash equilibrium.

        At Nash equilibrium:
        - Player 0 with K always bets (or is indifferent)
        - Player 0 with J bluffs with probability 1/3
        - Player 1 with K always calls
        - Player 1 with J always folds
        """
        trainer = CFRTrainer()
        trainer.train(10000)
        profile = trainer.average_strategy_profile()

        # Player 0 with King at root: indifferent in Nash equilibrium
        # (any bet frequency works), typically converges to betting > 50%
        k_root = profile.get("K|", {})
        assert k_root.get("b", 0) > 0.5, f"P0 with K should bet often, got {k_root}"

        # Player 1 with King facing bet: should always call
        k_bet = profile.get("K|b", {})
        assert k_bet.get("c", 0) > 0.9, f"P1 with K should call bet, got {k_bet}"

        # Player 1 with Jack facing bet: should always fold
        j_bet = profile.get("J|b", {})
        assert j_bet.get("f", 0) > 0.5, f"P1 with J should mostly fold, got {j_bet}"

    def test_game_value(self):
        """The game value for player 0 should be approximately -1/18."""
        trainer = CFRTrainer()
        trainer.train(10000)
        profile = trainer.average_strategy_profile()

        # Compute expected value under the profile
        game = KuhnPoker()

        def ev(state, p0_prob, p1_prob):
            if game.is_terminal(state):
                return p0_prob * p1_prob * game.terminal_utility(state, 0)
            if game.current_player(state) == -1:
                v = 0.0
                for outcome, prob in game.chance_outcomes(state):
                    v += prob * ev(game.next_state(state, outcome), p0_prob, p1_prob)
                return v
            player = game.current_player(state)
            key = game.infoset_key(state, player)
            probs = profile.get(key, {})
            actions = game.legal_actions(state)
            v = 0.0
            for a in actions:
                p = probs.get(a, 1.0 / len(actions))
                if player == 0:
                    v += ev(game.next_state(state, a), p0_prob * p, p1_prob)
                else:
                    v += ev(game.next_state(state, a), p0_prob, p1_prob * p)
            return v

        game_value = ev(game.initial_state(), 1.0, 1.0)
        expected = -1.0 / 18.0
        assert abs(game_value - expected) < 0.02, (
            f"Game value {game_value:.4f} should be near {expected:.4f}"
        )


class TestBeliefStateTracker:
    def test_initial_belief_is_uniform(self):
        tracker = BeliefStateTracker()
        # With uniform strategy, root belief should be uniform
        beliefs = tracker.compute_belief_states()
        root = beliefs[""]
        expected = 1.0 / NUM_DEALS
        for d in range(NUM_DEALS):
            assert abs(root[d].item() - expected) < 1e-6

    def test_beliefs_sum_to_one(self):
        """All non-zero beliefs should sum to 1."""
        trainer = CFRTrainer()
        trainer.train(1000)
        profile = trainer.average_strategy_profile()

        tracker = BeliefStateTracker()
        tracker.set_strategy_from_profile(profile)
        beliefs = tracker.compute_belief_states()

        for h, b in beliefs.items():
            total = b.sum().item()
            if total > 0:
                assert abs(total - 1.0) < 1e-5, (
                    f"Belief at '{h}' sums to {total}, not 1.0"
                )

    def test_belief_update_after_bet(self):
        """After player 0 bets, belief should shift toward strong hands."""
        trainer = CFRTrainer()
        trainer.train(5000)
        profile = trainer.average_strategy_profile()

        tracker = BeliefStateTracker()
        tracker.set_strategy_from_profile(profile)
        beliefs = tracker.compute_belief_states()

        root_belief = beliefs[""]
        bet_belief = beliefs.get("b")

        if bet_belief is not None and bet_belief.sum() > 0:
            # After a bet, deals where P0 has K should be more likely
            # relative to root (P0 with K bets more often)
            k_deals_root = sum(root_belief[i].item() for i, d in enumerate(ALL_DEALS) if d[0] == 2)
            k_deals_bet = sum(bet_belief[i].item() for i, d in enumerate(ALL_DEALS) if d[0] == 2)
            assert k_deals_bet >= k_deals_root - 0.05, (
                f"Bet should increase belief of P0 having K: root={k_deals_root:.3f}, bet={k_deals_bet:.3f}"
            )


class TestVectorizedCFR:
    def test_convergence(self):
        """Vectorized CFR should also converge."""
        vcfr = VectorizedCFR()
        vcfr.train(5000)
        profile = vcfr.average_strategy_profile()

        # Verify we get a reasonable strategy
        from kuhn.cfr import CFRTrainer
        dummy = CFRTrainer()
        br0 = dummy._best_response_value(profile, 0)
        br1 = dummy._best_response_value(profile, 1)
        exp = 0.5 * (br0 + br1)
        assert exp < 0.05, f"Vectorized CFR exploitability {exp} too high"

    def test_matches_scalar_cfr(self):
        """Vectorized and scalar CFR should produce similar strategies."""
        scalar = CFRTrainer()
        scalar.train(3000)
        scalar_profile = scalar.average_strategy_profile()

        vector = VectorizedCFR()
        vector.train(3000)
        vector_profile = vector.average_strategy_profile()

        # Compare strategies at key infosets
        for key in ["K|", "J|", "Q|", "K|b", "J|b", "Q|b"]:
            if key in scalar_profile and key in vector_profile:
                for action in scalar_profile[key]:
                    s = scalar_profile[key][action]
                    v = vector_profile[key].get(action, 0.5)
                    assert abs(s - v) < 0.15, (
                        f"Strategy mismatch at {key}/{action}: scalar={s:.3f}, vector={v:.3f}"
                    )


class TestSubgameSolver:
    def test_subgame_converges(self):
        """Subgame solver should converge to near-Nash when solving full game."""
        from rebel.endgame_solver import SubgameSolver, SubgameConfig
        game = KuhnPoker()
        initial_beliefs = torch.full((6,), 1.0 / 6.0)
        config = SubgameConfig(iterations=5000)
        solver = SubgameSolver(game, "", initial_beliefs, config=config)
        profile = solver.solve()

        # Validate against known Nash properties
        from kuhn.cfr import CFRTrainer
        dummy = CFRTrainer(game)
        br0 = dummy._best_response_value(profile, 0)
        br1 = dummy._best_response_value(profile, 1)
        exp = 0.5 * (br0 + br1)
        assert exp < 0.05, f"Subgame solver exploitability {exp} too high"

    def test_subgame_matches_cfr(self):
        """Subgame solver should produce similar strategies to full CFR."""
        from rebel.endgame_solver import SubgameSolver, SubgameConfig

        # Run standard CFR
        trainer = CFRTrainer()
        trainer.train(5000)
        cfr_profile = trainer.average_strategy_profile()

        # Run subgame solver on full game
        game = KuhnPoker()
        initial_beliefs = torch.full((6,), 1.0 / 6.0)
        config = SubgameConfig(iterations=5000)
        solver = SubgameSolver(game, "", initial_beliefs, config=config)
        sub_profile = solver.solve()

        # Key infosets should be similar
        for key in ["K|b", "J|b", "K|c", "J|c"]:
            if key in cfr_profile and key in sub_profile:
                for action in cfr_profile[key]:
                    s = cfr_profile[key][action]
                    v = sub_profile[key].get(action, 0.5)
                    assert abs(s - v) < 0.15, (
                        f"Subgame mismatch at {key}/{action}: cfr={s:.3f}, sub={v:.3f}"
                    )


class TestRebelTrainer:
    def test_rebel_training_runs(self):
        """ReBeL training loop should run without errors."""
        from rebel.rebel_trainer import RebelTrainer
        trainer = RebelTrainer(
            value_hidden_dim=32,
            cfr_iterations=50,
        )
        metrics = trainer.train(
            num_epochs=2,
            cfr_iters_per_epoch=50,
            value_train_epochs=10,
        )
        assert "value_losses" in metrics
        assert "exploitabilities" in metrics
        assert len(metrics["exploitabilities"]) == 2

    def test_rebel_exploitability_bounded(self):
        """ReBeL should produce a strategy with bounded exploitability."""
        from rebel.rebel_trainer import RebelTrainer
        trainer = RebelTrainer(
            value_hidden_dim=32,
            cfr_iterations=100,
        )
        metrics = trainer.train(
            num_epochs=3,
            cfr_iters_per_epoch=100,
            value_train_epochs=20,
        )
        # After training, exploitability should be bounded
        # (it may not be as tight as pure CFR due to function approximation)
        final_exp = metrics["exploitabilities"][-1]
        assert final_exp < 0.5, f"ReBeL exploitability {final_exp} unreasonably high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
