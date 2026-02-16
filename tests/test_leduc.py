"""Tests for Leduc Poker integration.

Validates:
- Game engine correctness (zero-sum, protocol compliance)
- CFR convergence with multiple variants
- Belief state computation
- Card abstraction
- ReBeL training loop
"""

import pytest
import torch
import random

from game_interface import Game
from leduc.game import LeducPoker, DECK, RANKS, RANK_TO_STR
from cfr.solver import CFRTrainer, CFRConfig
from cfr.vectorized import VectorizedCFR
from leduc.belief_state import (
    ALL_PRIVATE_DEALS,
    ALL_FULL_DEALS,
    NUM_FULL_DEALS,
    NUM_PRIVATE_DEALS,
    NUM_PRIVATE_STATES,
    NUM_PLAYERS,
    RANK_TO_DEALS_P0,
    RANK_TO_DEALS_P1,
    initial_pbs,
    reach_to_pbs_private,
    reach_to_pbs_full,
)
from leduc.card_abstraction import (
    preflop_bucket,
    postflop_bucket,
    equity_against_uniform,
    hand_strength_vector,
)


class TestLeducGameProtocol:
    def test_implements_game_protocol(self):
        game = LeducPoker()
        assert isinstance(game, Game)

    def test_num_players(self):
        game = LeducPoker()
        assert game.NUM_PLAYERS == 2

    def test_initial_state_is_chance(self):
        game = LeducPoker()
        state = game.initial_state()
        assert game.current_player(state) == -1
        assert not game.is_terminal(state)


class TestLeducChanceOutcomes:
    def setup_method(self):
        self.game = LeducPoker()

    def test_initial_deal_count(self):
        """Should have 30 possible private card deals (6 * 5)."""
        outcomes = self.game.chance_outcomes(self.game.initial_state())
        assert len(outcomes) == 30

    def test_deal_probabilities_sum_to_one(self):
        outcomes = self.game.chance_outcomes(self.game.initial_state())
        total = sum(p for _, p in outcomes)
        assert abs(total - 1.0) < 1e-10

    def test_board_card_outcomes(self):
        """After dealing 2 cards, 4 remain for the board."""
        state = self.game.initial_state()
        deal = self.game.chance_outcomes(state)[0][0]
        s = self.game.next_state(state, deal)
        # Play check-check to end round 1
        s = self.game.next_state(s, "c")
        s = self.game.next_state(s, "c")
        assert self.game.current_player(s) == -1
        board_outcomes = self.game.chance_outcomes(s)
        assert len(board_outcomes) == 4
        assert abs(sum(p for _, p in board_outcomes) - 1.0) < 1e-10


class TestLeducZeroSum:
    def test_zero_sum_random_games(self):
        """Verify zero-sum property across 5000 random games."""
        game = LeducPoker()
        random.seed(42)
        for _ in range(5000):
            state = game.initial_state()
            outcomes = game.chance_outcomes(state)
            deal, _ = random.choice(outcomes)
            state = game.next_state(state, deal)

            while not game.is_terminal(state):
                if game.current_player(state) == -1:
                    chances = game.chance_outcomes(state)
                    outcome, _ = random.choice(chances)
                    state = game.next_state(state, outcome)
                else:
                    actions = game.legal_actions(state)
                    state = game.next_state(state, random.choice(actions))

            u0 = game.terminal_utility(state, 0)
            u1 = game.terminal_utility(state, 1)
            assert abs(u0 + u1) < 1e-10

    def test_fold_utility(self):
        """Folder should lose their contribution."""
        game = LeducPoker()
        state = game.initial_state()
        deal = game.chance_outcomes(state)[0][0]
        s = game.next_state(state, deal)
        # P0 bets, P1 folds
        s = game.next_state(s, "b")
        s = game.next_state(s, "f")
        assert game.is_terminal(s)
        # P0 wins P1's ante (1 chip)
        assert game.terminal_utility(s, 0) == 1.0
        assert game.terminal_utility(s, 1) == -1.0


class TestLeducInformationSets:
    def test_infoset_key_format(self):
        game = LeducPoker()
        state = game.initial_state()
        deal = ((0, 0), (2, 1))  # J0 vs K1
        s = game.next_state(state, deal)
        assert game.infoset_key(s, 0).startswith("J")
        assert game.infoset_key(s, 1).startswith("K")

    def test_same_rank_same_infoset(self):
        """Both suits of same rank should give same infoset key."""
        game = LeducPoker()
        state = game.initial_state()
        s1 = game.next_state(state, ((0, 0), (1, 0)))  # J0 vs Q0
        s2 = game.next_state(state, ((0, 1), (1, 0)))  # J1 vs Q0
        # Player 0 should have same infoset for both (J with same history)
        assert game.infoset_key(s1, 0) == game.infoset_key(s2, 0)

    def test_infoset_count(self):
        """Leduc should have 288 information sets."""
        game = LeducPoker()
        trainer = CFRTrainer(game)
        trainer.train(10)
        # 288 is the known count for standard Leduc
        assert len(trainer.infosets) == 288


class TestLeducCFR:
    @pytest.mark.slow
    def test_vanilla_cfr_convergence(self):
        """Vanilla CFR should converge on Leduc."""
        trainer = CFRTrainer(LeducPoker())
        trainer.train(500)
        exp = trainer.exploitability()
        assert exp < 0.05, f"Exploitability {exp:.6f} too high after 500 iterations"

    @pytest.mark.slow
    def test_cfr_plus_convergence(self):
        """CFR+ should converge on Leduc."""
        config = CFRConfig(use_plus=True)
        trainer = CFRTrainer(LeducPoker(), config=config)
        trainer.train(500)
        exp = trainer.exploitability()
        assert exp < 0.05, f"CFR+ exploitability {exp:.6f} too high"

    @pytest.mark.slow
    def test_linear_cfr_convergence(self):
        """Linear CFR (CFR+ with linear weighting) should converge."""
        config = CFRConfig(use_plus=True, linear_weighting=True)
        trainer = CFRTrainer(LeducPoker(), config=config)
        trainer.train(500)
        exp = trainer.exploitability()
        assert exp < 0.05, f"Linear CFR exploitability {exp:.6f} too high"

    def test_strategy_valid_probabilities(self):
        """All strategy probabilities should be valid distributions."""
        trainer = CFRTrainer(LeducPoker())
        trainer.train(100)
        profile = trainer.average_strategy_profile()
        for key, action_probs in profile.items():
            total = sum(action_probs.values())
            assert abs(total - 1.0) < 1e-6, (
                f"Strategy at {key} doesn't sum to 1: {action_probs}"
            )
            for a, p in action_probs.items():
                assert p >= -1e-6, f"Negative prob at {key}/{a}: {p}"


class TestLeducBeliefState:
    def test_initial_pbs_shape(self):
        pbs = initial_pbs()
        assert pbs.shape == (NUM_PRIVATE_STATES, NUM_PLAYERS)

    def test_initial_pbs_uniform(self):
        pbs = initial_pbs()
        expected = 1.0 / NUM_PRIVATE_STATES
        for c in range(NUM_PRIVATE_STATES):
            for p in range(NUM_PLAYERS):
                assert abs(pbs[c, p].item() - expected) < 1e-6

    def test_private_deal_count(self):
        assert NUM_PRIVATE_DEALS == 30

    def test_full_deal_count(self):
        assert NUM_FULL_DEALS == 120

    def test_rank_mapping_p0(self):
        """Each rank should map to 10 private deals for P0."""
        for r in RANKS:
            assert len(RANK_TO_DEALS_P0[r]) == 10

    def test_rank_mapping_p1(self):
        """Each rank should map to 10 private deals for P1."""
        for r in RANKS:
            assert len(RANK_TO_DEALS_P1[r]) == 10

    def test_uniform_reach_gives_uniform_pbs(self):
        reach = torch.ones(NUM_PRIVATE_DEALS)
        pbs = reach_to_pbs_private(reach, reach)
        expected = 1.0 / NUM_PRIVATE_STATES
        for c in range(NUM_PRIVATE_STATES):
            for p in range(NUM_PLAYERS):
                assert abs(pbs[c, p].item() - expected) < 1e-5

    def test_full_deal_uniform_reach(self):
        reach = torch.ones(NUM_FULL_DEALS)
        pbs = reach_to_pbs_full(reach, reach)
        expected = 1.0 / NUM_PRIVATE_STATES
        for c in range(NUM_PRIVATE_STATES):
            for p in range(NUM_PLAYERS):
                assert abs(pbs[c, p].item() - expected) < 1e-5

    def test_pbs_columns_sum_to_one(self):
        """PBS columns should sum to 1 for non-zero reach."""
        reach_p0 = torch.rand(NUM_PRIVATE_DEALS)
        reach_p1 = torch.rand(NUM_PRIVATE_DEALS)
        pbs = reach_to_pbs_private(reach_p0, reach_p1)
        for p in range(NUM_PLAYERS):
            col_sum = pbs[:, p].sum().item()
            if col_sum > 0:
                assert abs(col_sum - 1.0) < 1e-5


class TestLeducCardAbstraction:
    def test_preflop_buckets(self):
        assert preflop_bucket(0) == 0  # J
        assert preflop_bucket(1) == 1  # Q
        assert preflop_bucket(2) == 2  # K

    def test_pair_beats_nonpair(self):
        """Pair equity should be higher than any non-pair."""
        for board_rank in RANKS:
            pair_eq = equity_against_uniform(board_rank, board_rank)
            for rank in RANKS:
                if rank != board_rank:
                    nonpair_eq = equity_against_uniform(rank, board_rank)
                    assert pair_eq > nonpair_eq, (
                        f"Pair {rank}={board_rank} equity {pair_eq} "
                        f"not > non-pair {rank} vs board {board_rank} equity {nonpair_eq}"
                    )

    def test_higher_rank_beats_lower_nonpair(self):
        """Among non-pairs, higher rank should have higher equity."""
        for board_rank in RANKS:
            non_pair_ranks = [r for r in RANKS if r != board_rank]
            equities = [equity_against_uniform(r, board_rank) for r in non_pair_ranks]
            assert equities == sorted(equities), (
                f"Non-pair equities not monotonic for board {board_rank}: {equities}"
            )

    def test_postflop_bucket_pair_higher(self):
        """Pair buckets should be > non-pair buckets."""
        for board_rank in RANKS:
            pair_bucket = postflop_bucket(board_rank, board_rank)
            for rank in RANKS:
                if rank != board_rank:
                    nonpair_bucket = postflop_bucket(rank, board_rank)
                    assert pair_bucket > nonpair_bucket


class TestLeducVectorizedCFR:
    @pytest.mark.slow
    def test_vectorized_cfr_convergence(self):
        """VectorizedCFR should converge on Leduc with mid-tree chance nodes."""
        vcfr = VectorizedCFR(LeducPoker())
        # Run without exploitability checks for speed
        game = LeducPoker()
        initial = game.initial_state()
        for i in range(1000):
            deal_states = [game.next_state(initial, deal) for deal in vcfr.all_deals]
            vcfr._cfr_vectorized(
                deal_states, vcfr.deal_probs.clone(), vcfr.deal_probs.clone()
            )
            vcfr.iteration += 1
        exp = vcfr._exploitability()
        assert exp < 0.05, f"VectorizedCFR exploitability {exp:.6f} too high after 1000 iters"

    def test_vectorized_cfr_discovers_all_infosets(self):
        """VectorizedCFR should discover all 288 Leduc infosets."""
        vcfr = VectorizedCFR(LeducPoker())
        assert len(vcfr._infoset_deal_indices) == 288

    def test_vectorized_cfr_handles_chance_node(self):
        """VectorizedCFR should not crash on Leduc's mid-tree board card deal."""
        vcfr = VectorizedCFR(LeducPoker())
        exps = vcfr.train(10)
        assert len(exps) > 0
        assert all(e > 0 for e in exps)

    @pytest.mark.slow
    def test_vectorized_matches_scalar_direction(self):
        """VectorizedCFR strategies should converge toward same Nash as scalar."""
        scalar = CFRTrainer(LeducPoker())
        scalar.train(500)
        scalar_profile = scalar.average_strategy_profile()

        vcfr = VectorizedCFR(LeducPoker())
        game = LeducPoker()
        initial = game.initial_state()
        for i in range(1000):
            deal_states = [game.next_state(initial, deal) for deal in vcfr.all_deals]
            vcfr._cfr_vectorized(
                deal_states, vcfr.deal_probs.clone(), vcfr.deal_probs.clone()
            )
            vcfr.iteration += 1
        vector_profile = vcfr.average_strategy_profile()

        # Check round 1 strategies are in same direction
        for key in ["K-|", "J-|", "Q-|"]:
            if key in scalar_profile and key in vector_profile:
                for action in scalar_profile[key]:
                    s = scalar_profile[key][action]
                    v = vector_profile[key].get(action, 0.5)
                    assert abs(s - v) < 0.2, (
                        f"Strategy mismatch at {key}/{action}: scalar={s:.3f}, vector={v:.3f}"
                    )


class TestLeducRebelTrainer:
    @pytest.mark.slow
    def test_rebel_training_runs(self):
        """ReBeL training loop should complete without errors."""
        from rebel.leduc_trainer import LeducRebelTrainer
        trainer = LeducRebelTrainer(value_hidden_dim=32)
        metrics = trainer.train(
            num_epochs=2,
            cfr_iters_per_epoch=100,
            value_train_epochs=10,
        )
        assert "value_losses" in metrics
        assert "exploitabilities" in metrics
        assert len(metrics["exploitabilities"]) == 2

    @pytest.mark.slow
    def test_rebel_exploitability_decreases(self):
        """ReBeL exploitability should decrease across epochs."""
        from rebel.leduc_trainer import LeducRebelTrainer
        trainer = LeducRebelTrainer(value_hidden_dim=64)
        metrics = trainer.train(
            num_epochs=3,
            cfr_iters_per_epoch=200,
            value_train_epochs=20,
        )
        exps = metrics["exploitabilities"]
        assert exps[-1] < exps[0], (
            f"Exploitability did not decrease: {exps}"
        )

    @pytest.mark.slow
    def test_rebel_exploitability_bounded(self):
        """ReBeL should produce bounded exploitability."""
        from rebel.leduc_trainer import LeducRebelTrainer
        trainer = LeducRebelTrainer(value_hidden_dim=64)
        metrics = trainer.train(
            num_epochs=2,
            cfr_iters_per_epoch=300,
            value_train_epochs=30,
        )
        assert metrics["exploitabilities"][-1] < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
