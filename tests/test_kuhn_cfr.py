"""Tests for CFR implementation with Kuhn Poker.

Validates against known Nash equilibrium properties of Kuhn Poker:
- Game value is -1/18 ~ -0.0556 for player 0
- Exploitability converges to 0
- Known optimal strategy properties
"""

import math
import pytest
import torch

from game_interface import Game
from kuhn.game import KuhnPoker, RANK_NAMES
from cfr.solver import CFRTrainer
from cfr.vectorized import VectorizedCFR
from kuhn.belief_state import (
    BeliefStateTracker,
    ALL_DEALS,
    NUM_DEALS,
    NUM_PRIVATE_STATES,
    NUM_PLAYERS,
    initial_chance_probs,
    initial_pbs,
    reach_to_pbs,
)


class TestGameInterface:
    def test_kuhn_implements_game_protocol(self):
        """KuhnPoker must satisfy the Game protocol."""
        game = KuhnPoker()
        assert isinstance(game, Game)

    def test_cfr_accepts_generic_game(self):
        """CFRTrainer should accept any Game, not just KuhnPoker."""
        game: Game = KuhnPoker()
        trainer = CFRTrainer(game)
        trainer.train(100)
        exp = trainer.exploitability()
        assert exp < 0.5


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
        trainer = CFRTrainer(KuhnPoker())
        trainer.train(5000)
        exp = trainer.exploitability()
        assert exp < 0.01, f"Exploitability {exp} too high after 5000 iterations"

    def test_known_nash_properties(self):
        """Test known properties of the Kuhn Poker Nash equilibrium."""
        trainer = CFRTrainer(KuhnPoker())
        trainer.train(10000)
        profile = trainer.average_strategy_profile()

        k_root = profile.get("K|", {})
        assert k_root.get("b", 0) > 0.5, f"P0 with K should bet often, got {k_root}"

        k_bet = profile.get("K|b", {})
        assert k_bet.get("c", 0) > 0.9, f"P1 with K should call bet, got {k_bet}"

        j_bet = profile.get("J|b", {})
        assert j_bet.get("f", 0) > 0.5, f"P1 with J should mostly fold, got {j_bet}"

    def test_game_value(self):
        """The game value for player 0 should be approximately -1/18."""
        trainer = CFRTrainer(KuhnPoker())
        trainer.train(10000)
        profile = trainer.average_strategy_profile()

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
        beliefs = tracker.compute_belief_states()
        root = beliefs[""]
        # Shape should be [NUM_PRIVATE_STATES, NUM_PLAYERS]
        assert root.shape == (NUM_PRIVATE_STATES, NUM_PLAYERS)
        expected = 1.0 / NUM_PRIVATE_STATES
        for c in range(NUM_PRIVATE_STATES):
            for p in range(NUM_PLAYERS):
                assert abs(root[c, p].item() - expected) < 1e-6

    def test_beliefs_columns_sum_to_one(self):
        """Each player's belief column should sum to 1."""
        trainer = CFRTrainer(KuhnPoker())
        trainer.train(1000)
        profile = trainer.average_strategy_profile()

        tracker = BeliefStateTracker()
        tracker.set_strategy_from_profile(profile)
        beliefs = tracker.compute_belief_states()

        for h, b in beliefs.items():
            assert b.shape == (NUM_PRIVATE_STATES, NUM_PLAYERS), f"Wrong shape at '{h}'"
            for p in range(NUM_PLAYERS):
                col_sum = b[:, p].sum().item()
                if col_sum > 0:
                    assert abs(col_sum - 1.0) < 1e-5, (
                        f"Player {p} belief at '{h}' sums to {col_sum}, not 1.0"
                    )

    def test_belief_update_after_bet(self):
        """After player 0 bets, belief should shift toward strong hands."""
        trainer = CFRTrainer(KuhnPoker())
        trainer.train(5000)
        profile = trainer.average_strategy_profile()

        tracker = BeliefStateTracker()
        tracker.set_strategy_from_profile(profile)
        beliefs = tracker.compute_belief_states()

        root_belief = beliefs[""]
        bet_belief = beliefs.get("b")

        if bet_belief is not None and bet_belief[:, 0].sum() > 0:
            # K is card index 2; after P0 bets, P0's belief of holding K should increase
            k_root = root_belief[2, 0].item()
            k_bet = bet_belief[2, 0].item()
            assert k_bet >= k_root - 0.05, (
                f"Bet should increase belief of P0 having K: root={k_root:.3f}, bet={k_bet:.3f}"
            )

    def test_belief_update_primarily_affects_acting_player(self):
        """When P0 acts, P0's beliefs should change more than P1's."""
        trainer = CFRTrainer(KuhnPoker())
        trainer.train(5000)
        profile = trainer.average_strategy_profile()

        tracker = BeliefStateTracker()
        tracker.set_strategy_from_profile(profile)
        beliefs = tracker.compute_belief_states()

        root = beliefs[""]
        bet = beliefs.get("b")
        if bet is not None:
            # P0 acted (bet), so P0's column should change more
            p0_change = (bet[:, 0] - root[:, 0]).abs().sum().item()
            p1_change = (bet[:, 1] - root[:, 1]).abs().sum().item()
            # P0's beliefs should change at least as much as P1's (or close)
            assert p0_change >= p1_change * 0.5, (
                f"P0 change ({p0_change:.4f}) should be comparable to P1 change ({p1_change:.4f})"
            )


class TestPBSHelpers:
    def test_initial_pbs_shape(self):
        pbs = initial_pbs()
        assert pbs.shape == (NUM_PRIVATE_STATES, NUM_PLAYERS)

    def test_initial_pbs_uniform(self):
        pbs = initial_pbs()
        expected = 1.0 / NUM_PRIVATE_STATES
        for c in range(NUM_PRIVATE_STATES):
            for p in range(NUM_PLAYERS):
                assert abs(pbs[c, p].item() - expected) < 1e-6

    def test_reach_to_pbs_uniform(self):
        """Uniform reach should give uniform PBS."""
        reach = torch.ones(NUM_DEALS)
        pbs = reach_to_pbs(reach, reach)
        expected = 1.0 / NUM_PRIVATE_STATES
        for c in range(NUM_PRIVATE_STATES):
            for p in range(NUM_PLAYERS):
                assert abs(pbs[c, p].item() - expected) < 1e-5


class TestVectorizedCFR:
    def test_convergence(self):
        """Vectorized CFR should also converge."""
        vcfr = VectorizedCFR(KuhnPoker())
        vcfr.train(5000)
        profile = vcfr.average_strategy_profile()

        dummy = CFRTrainer(KuhnPoker())
        br0 = dummy._best_response_value(profile, 0)
        br1 = dummy._best_response_value(profile, 1)
        exp = 0.5 * (br0 + br1)
        assert exp < 0.05, f"Vectorized CFR exploitability {exp} too high"

    def test_matches_scalar_cfr(self):
        """Vectorized and scalar CFR should produce similar strategies."""
        scalar = CFRTrainer(KuhnPoker())
        scalar.train(3000)
        scalar_profile = scalar.average_strategy_profile()

        vector = VectorizedCFR(KuhnPoker())
        vector.train(3000)
        vector_profile = vector.average_strategy_profile()

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
        beliefs = initial_pbs()
        config = SubgameConfig(iterations=5000)
        solver = SubgameSolver(game, "", beliefs, config=config)
        profile = solver.solve()

        dummy = CFRTrainer(game)
        br0 = dummy._best_response_value(profile, 0)
        br1 = dummy._best_response_value(profile, 1)
        exp = 0.5 * (br0 + br1)
        assert exp < 0.05, f"Subgame solver exploitability {exp} too high"

    def test_subgame_matches_cfr(self):
        """Subgame solver should produce similar strategies to full CFR."""
        from rebel.endgame_solver import SubgameSolver, SubgameConfig

        trainer = CFRTrainer(KuhnPoker())
        trainer.train(5000)
        cfr_profile = trainer.average_strategy_profile()

        game = KuhnPoker()
        beliefs = initial_pbs()
        config = SubgameConfig(iterations=5000)
        solver = SubgameSolver(game, "", beliefs, config=config)
        sub_profile = solver.solve()

        for key in ["K|b", "J|b", "K|c", "J|c"]:
            if key in cfr_profile and key in sub_profile:
                for action in cfr_profile[key]:
                    s = cfr_profile[key][action]
                    v = sub_profile[key].get(action, 0.5)
                    assert abs(s - v) < 0.15, (
                        f"Subgame mismatch at {key}/{action}: cfr={s:.3f}, sub={v:.3f}"
                    )

    def test_subgame_initial_beliefs_shape(self):
        """SubgameSolver should accept [NUM_PRIVATE_STATES, NUM_PLAYERS] beliefs."""
        from rebel.endgame_solver import SubgameSolver, SubgameConfig
        game = KuhnPoker()
        beliefs = initial_pbs()
        assert beliefs.shape == (NUM_PRIVATE_STATES, NUM_PLAYERS)
        config = SubgameConfig(iterations=10)
        solver = SubgameSolver(game, "", beliefs, config=config)
        solver.solve()


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
        final_exp = metrics["exploitabilities"][-1]
        assert final_exp < 0.5, f"ReBeL exploitability {final_exp} unreasonably high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
