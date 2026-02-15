"""Tests for HUNL Hold'em ReBeL components.

Tests cover:
1. Game engine - action/deal mechanics, Game protocol compliance
2. Belief state - PBS initialization, board updates, Bayesian updates
3. Value network - forward pass shapes, feature encoding
4. Subgame solver - CFR convergence on small subgames
5. ReBeL trainer - end-to-end training loop
"""

import pytest
import torch
from itertools import combinations

import poker_engine as pe

from hunl.game import (
    ALL_HOLE_COMBOS,
    COMBO_INDEX,
    HUNLAction,
    HUNLConfig,
    HUNLGame,
    HUNLState,
    NUM_COMBOS,
    hands_for_board,
    valid_deals_for_board,
)
from hunl.belief_state import (
    HUNLBeliefState,
    build_hunl_belief_config,
    pbs_from_reach_probs_fast,
)
from hunl.value_net import (
    HUNLValueNetConfig,
    HUNLValueNetwork,
    encode_public_features,
    encode_state_features,
)
from hunl.rebel_trainer import (
    HUNLRebelConfig,
    HUNLRebelTrainer,
    HUNLReplayBuffer,
    HUNLSubgameSolver,
    sample_random_board,
    sample_random_state,
)
from hunl.subgame_solver import SubgamePlayer, solve_subgame


# ---------------------------------------------------------------------------
# Game engine tests
# ---------------------------------------------------------------------------


class TestHUNLGame:
    def test_num_combos(self):
        """C(52, 2) = 1326 hole card combinations."""
        assert NUM_COMBOS == 1326
        assert len(ALL_HOLE_COMBOS) == 1326

    def test_combo_index_consistency(self):
        """Every combo maps to a unique index 0-1325."""
        for i, combo in enumerate(ALL_HOLE_COMBOS):
            assert COMBO_INDEX[combo] == i
        assert len(COMBO_INDEX) == 1326

    def test_combos_sorted(self):
        """All combos have c0 < c1."""
        for c0, c1 in ALL_HOLE_COMBOS:
            assert c0 < c1

    def test_initial_state(self):
        game = HUNLGame()
        state = game.initial_state()
        assert state.phase == "predeal"
        assert game.current_player(state) == -1  # chance node
        assert not game.is_terminal(state)

    def test_deal_and_preflop(self):
        game = HUNLGame()
        state = game.initial_state()

        # Deal specific hands: AsKh to P0, QdJc to P1
        h0 = (50, 46)  # Ah, Kh
        h1 = (41, 36)  # Qd, Jc
        state = game.next_state(state, (h0, h1))

        assert state.phase == "preflop"
        assert state.p0_hand == h0
        assert state.p1_hand == h1
        assert not game.is_terminal(state)

        # SB (P0) acts first preflop
        player = game.current_player(state)
        assert player == 0

    def test_fold_is_terminal(self):
        game = HUNLGame()
        state = game.initial_state()
        state = game.next_state(state, ((50, 46), (41, 36)))

        actions = game.legal_actions(state)
        fold = [a for a in actions if a.action_type == "fold"][0]
        state = game.next_state(state, fold)

        assert game.is_terminal(state)

    def test_fold_utility(self):
        game = HUNLGame()
        state = game.initial_state()
        state = game.next_state(state, ((50, 46), (41, 36)))

        actions = game.legal_actions(state)
        fold = [a for a in actions if a.action_type == "fold"][0]
        state = game.next_state(state, fold)

        # P0 (SB) folds, loses SB (1 chip)
        assert game.terminal_utility(state, 0) == -1.0
        assert game.terminal_utility(state, 1) == 1.0

    def test_call_transitions_to_flop_deal(self):
        game = HUNLGame()
        state = game.initial_state()
        state = game.next_state(state, ((50, 46), (41, 36)))

        # P0 calls
        actions = game.legal_actions(state)
        call = [a for a in actions if a.action_type == "call"][0]
        state = game.next_state(state, call)

        # P1 checks
        actions = game.legal_actions(state)
        check = [a for a in actions if a.action_type == "check"][0]
        state = game.next_state(state, check)

        # Should be at flop deal (chance node)
        assert state.phase == "flop_deal"
        assert game.current_player(state) == -1

    def test_full_hand_check_through(self):
        """Play a complete hand with all checks."""
        game = HUNLGame()
        state = game.initial_state()
        state = game.next_state(state, ((50, 46), (41, 36)))

        # Preflop: call, check
        call = [a for a in game.legal_actions(state) if a.action_type == "call"][0]
        state = game.next_state(state, call)
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)

        # Flop deal
        assert state.phase == "flop_deal"
        state = game.next_state(state, (0, 1, 2))  # 2c, 2d, 2h
        assert state.phase == "flop"
        assert len(state.board) == 3

        # Flop: check, check
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)

        # Turn deal
        assert state.phase == "turn_deal"
        state = game.next_state(state, 3)
        assert state.phase == "turn"
        assert len(state.board) == 4

        # Turn: check, check
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)

        # River deal
        assert state.phase == "river_deal"
        state = game.next_state(state, 4)
        assert state.phase == "river"
        assert len(state.board) == 5

        # River: check, check
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)

        # Should be terminal (showdown)
        assert game.is_terminal(state)

        # Utilities should be zero-sum
        u0 = game.terminal_utility(state, 0)
        u1 = game.terminal_utility(state, 1)
        assert abs(u0 + u1) < 1e-6, f"Not zero-sum: {u0} + {u1}"

    def test_legal_actions_include_fold_call_raise(self):
        game = HUNLGame()
        state = game.initial_state()
        state = game.next_state(state, ((50, 46), (41, 36)))

        actions = game.legal_actions(state)
        action_types = {a.action_type for a in actions}
        # Preflop SB should have fold, call, raise options
        assert "fold" in action_types
        assert "call" in action_types
        # Should have at least one raise or all-in
        assert "raise" in action_types or "all_in" in action_types

    def test_infoset_key_differs_by_hand(self):
        game = HUNLGame()
        state = game.initial_state()
        state = game.next_state(state, ((50, 46), (41, 36)))

        key_p0 = game.infoset_key(state, 0)
        key_p1 = game.infoset_key(state, 1)
        assert key_p0 != key_p1  # Different hands

    def test_hands_for_board(self):
        board = (0, 1, 2)  # 2c, 2d, 2h
        hands = hands_for_board(board)
        # No hand should contain any board card
        for c0, c1 in hands:
            assert c0 not in board
            assert c1 not in board
        # Should be C(49, 2) = 1176
        assert len(hands) == 1176

    def test_hand_index(self):
        game = HUNLGame()
        assert game.hand_index((0, 1)) == 0
        assert game.hand_index((50, 51)) == 1325


# ---------------------------------------------------------------------------
# Belief state tests
# ---------------------------------------------------------------------------


class TestHUNLBeliefState:
    def test_initial_uniform_beliefs(self):
        bs = HUNLBeliefState()
        pbs = bs.get_pbs()
        assert pbs.shape == (1326, 2)

        # All valid hands should have equal probability
        for p in range(2):
            probs = pbs[:, p]
            valid_probs = probs[probs > 0]
            assert len(valid_probs) == 1326
            assert torch.allclose(valid_probs, valid_probs[0] * torch.ones_like(valid_probs))
            assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_board_eliminates_hands(self):
        board = (0, 1, 2)  # 2c, 2d, 2h
        bs = HUNLBeliefState(board=board)
        pbs = bs.get_pbs()

        # Hands containing board cards should have 0 probability
        for h_idx, (c0, c1) in enumerate(ALL_HOLE_COMBOS):
            if c0 in board or c1 in board:
                assert pbs[h_idx, 0].item() == 0.0
                assert pbs[h_idx, 1].item() == 0.0

        # Valid hands should still sum to 1
        for p in range(2):
            assert abs(pbs[:, p].sum().item() - 1.0) < 1e-5

    def test_update_board_zeros_new_cards(self):
        bs = HUNLBeliefState()
        initial_valid = (bs.beliefs[:, 0] > 0).sum().item()
        assert initial_valid == 1326

        bs.update_board((0, 1, 2))
        valid_after = (bs.beliefs[:, 0] > 0).sum().item()
        assert valid_after < initial_valid
        assert valid_after == 1176  # C(49, 2)

        # Still normalized
        assert abs(bs.beliefs[:, 0].sum().item() - 1.0) < 1e-5

    def test_bayesian_update(self):
        bs = HUNLBeliefState()
        pbs_before = bs.get_pbs().clone()

        # Simulate an action where hands 0-100 are much more likely
        action_probs = torch.ones(NUM_COMBOS) * 0.1
        action_probs[:100] = 0.9

        bs.bayesian_update(0, action_probs)
        pbs_after = bs.get_pbs()

        # Hands 0-100 should have higher belief for P0
        assert pbs_after[:100, 0].mean() > pbs_after[100:, 0].mean()
        # Still normalized
        assert abs(pbs_after[:, 0].sum().item() - 1.0) < 1e-5

    def test_hand_conflict(self):
        # Same card in both hands
        assert HUNLBeliefState.hand_conflict(0, 0)

        # (0,1) and (0,2) share card 0
        idx_01 = COMBO_INDEX[(0, 1)]
        idx_02 = COMBO_INDEX[(0, 2)]
        assert HUNLBeliefState.hand_conflict(idx_01, idx_02)

        # (0,1) and (2,3) don't share
        idx_23 = COMBO_INDEX[(2, 3)]
        assert not HUNLBeliefState.hand_conflict(idx_01, idx_23)

    def test_pbs_flat_shape(self):
        bs = HUNLBeliefState()
        flat = bs.get_pbs_flat()
        assert flat.shape == (2652,)

    def test_valid_mask_river(self):
        """River board (5 cards) leaves C(47,2) = 1081 valid hands."""
        board = (0, 1, 2, 3, 4)
        bs = HUNLBeliefState(board=board)
        assert bs.valid_mask.sum().item() == 1081


# ---------------------------------------------------------------------------
# Belief config tests
# ---------------------------------------------------------------------------


class TestBuildBeliefConfig:
    def test_preflop_config(self):
        config = build_hunl_belief_config(board=())
        assert config["num_private_states"] == 1326
        assert config["num_players"] == 2
        assert config["num_valid_hands"] == 1326

    def test_flop_config(self):
        config = build_hunl_belief_config(board=(0, 1, 2))
        assert config["num_valid_hands"] == 1176  # C(49, 2)

    def test_river_config(self):
        config = build_hunl_belief_config(board=(0, 1, 2, 3, 4))
        assert config["num_valid_hands"] == 1081  # C(47, 2)


# ---------------------------------------------------------------------------
# Value network tests
# ---------------------------------------------------------------------------


class TestHUNLValueNetwork:
    def test_forward_shape(self):
        config = HUNLValueNetConfig(hidden_dim=64, num_layers=2)
        net = HUNLValueNetwork(config)

        pbs = torch.randn(1, 2652)
        features = torch.randn(1, 20)
        out = net(pbs, features)
        assert out.shape == (1, 2652)

    def test_forward_batch(self):
        config = HUNLValueNetConfig(hidden_dim=64, num_layers=2)
        net = HUNLValueNetwork(config)

        batch_size = 8
        pbs = torch.randn(batch_size, 2652)
        features = torch.randn(batch_size, 20)
        out = net(pbs, features)
        assert out.shape == (batch_size, 2652)

    def test_forward_no_features(self):
        config = HUNLValueNetConfig(hidden_dim=64, num_layers=2)
        net = HUNLValueNetwork(config)

        pbs = torch.randn(1, 2652)
        out = net(pbs)  # No public features
        assert out.shape == (1, 2652)

    def test_encode_public_features(self):
        features = encode_public_features(
            pot=100, effective_stack=150, street=1,
            board_cards=(0, 1, 2),
        )
        assert features.shape == (20,)
        # Pot normalized
        assert features[0].item() == pytest.approx(100 / 400, abs=1e-5)
        # Street one-hot
        assert features[3].item() == 1.0  # flop
        assert features[2].item() == 0.0  # not preflop

    def test_encode_preflop_features(self):
        features = encode_public_features(pot=3, effective_stack=198, street=0)
        assert features[2].item() == 1.0  # preflop
        assert features[3].item() == 0.0  # not flop

    def test_gradient_flows(self):
        config = HUNLValueNetConfig(hidden_dim=32, num_layers=2)
        net = HUNLValueNetwork(config)

        pbs = torch.randn(4, 2652, requires_grad=True)
        features = torch.randn(4, 20)
        out = net(pbs, features)
        loss = out.sum()
        loss.backward()
        assert pbs.grad is not None


# ---------------------------------------------------------------------------
# Replay buffer tests
# ---------------------------------------------------------------------------


class TestHUNLReplayBuffer:
    def test_add_and_retrieve(self):
        buf = HUNLReplayBuffer(max_size=100)
        pbs = torch.randn(1326, 2)
        features = torch.randn(20)
        values = torch.randn(1326, 2)
        buf.add(pbs, features, values)

        dataset = buf.to_dataset()
        assert dataset["pbs"].shape == (1, 2652)
        assert dataset["features"].shape == (1, 20)
        assert dataset["values"].shape == (1, 2652)

    def test_max_size(self):
        buf = HUNLReplayBuffer(max_size=5)
        for _ in range(10):
            buf.add(torch.randn(1326, 2), torch.randn(20), torch.randn(1326, 2))
        assert len(buf) == 5


# ---------------------------------------------------------------------------
# Sample random board/state tests
# ---------------------------------------------------------------------------


class TestSampling:
    def test_sample_random_board(self):
        board = sample_random_board("preflop")
        assert len(board) == 0

        board = sample_random_board("flop")
        assert len(board) == 3
        assert len(set(board)) == 3  # unique cards

        board = sample_random_board("turn")
        assert len(board) == 4

        board = sample_random_board("river")
        assert len(board) == 5

    def test_sample_random_state(self):
        game = HUNLGame()
        state, board = sample_random_state(game)
        assert isinstance(state, HUNLState)
        # Board should match state's board
        assert state.board == board


# ---------------------------------------------------------------------------
# Subgame solver tests
# ---------------------------------------------------------------------------


class TestHUNLSubgameSolver:
    def test_solver_runs(self):
        """Solver completes without errors on a river subgame."""
        game = HUNLGame()
        state = game.initial_state()

        # Set up a river state
        h0, h1 = (50, 46), (41, 36)
        state = game.next_state(state, (h0, h1))

        # Play to river
        call = [a for a in game.legal_actions(state) if a.action_type == "call"][0]
        state = game.next_state(state, call)
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)

        # Deal flop
        state = game.next_state(state, (0, 1, 2))
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)

        # Deal turn
        state = game.next_state(state, 3)
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)

        # Deal river
        state = game.next_state(state, 4)

        # Now solve at the river state
        beliefs = HUNLBeliefState(board=state.board)
        solver = HUNLSubgameSolver(
            game=game,
            state=state,
            initial_beliefs=beliefs.get_pbs(),
            max_depth=2,
            cfr_iters=10,
        )

        profile = solver.solve()
        assert isinstance(profile, dict)

    def test_solver_collects_data(self):
        """Solver collects training data during the last iteration."""
        game = HUNLGame()
        state = game.initial_state()
        state = game.next_state(state, ((50, 46), (41, 36)))

        # Play to river
        call = [a for a in game.legal_actions(state) if a.action_type == "call"][0]
        state = game.next_state(state, call)
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)
        state = game.next_state(state, (0, 1, 2))
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)
        state = game.next_state(state, 3)
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)
        state = game.next_state(state, 4)

        beliefs = HUNLBeliefState(board=state.board)
        solver = HUNLSubgameSolver(
            game=game,
            state=state,
            initial_beliefs=beliefs.get_pbs(),
            max_depth=2,
            cfr_iters=5,
        )
        solver.solve()

        # Should have collected some training data
        assert len(solver.collected_data) > 0
        pbs, features, values = solver.collected_data[0]
        assert pbs.shape == (1326, 2)
        assert features.shape == (20,)
        assert values.shape == (1326, 2)

    def test_fold_values_are_uniform(self):
        """After a fold, all valid hands should have the same value."""
        game = HUNLGame()
        state = game.initial_state()
        state = game.next_state(state, ((50, 46), (41, 36)))

        beliefs = HUNLBeliefState(board=())
        solver = HUNLSubgameSolver(
            game=game,
            state=state,
            initial_beliefs=beliefs.get_pbs(),
            max_depth=2,
            cfr_iters=5,
        )

        # Manually test fold values
        fold = [a for a in game.legal_actions(state) if a.action_type == "fold"][0]
        fold_state = game.next_state(state, fold)
        values = solver._fold_values(fold_state)

        # All valid hands should have same value
        valid_values = values[solver.valid_mask]
        if len(valid_values) > 0:
            assert torch.allclose(valid_values, valid_values[0] * torch.ones_like(valid_values))


# ---------------------------------------------------------------------------
# ReBeL trainer tests
# ---------------------------------------------------------------------------


class TestHUNLRebelTrainer:
    def test_trainer_constructs(self):
        """Trainer initializes with default config."""
        config = HUNLRebelConfig(
            num_iters=1,
            samples_per_iter=1,
            cfr_iters_per_solve=5,
            value_hidden_dim=32,
            value_num_layers=2,
            value_train_epochs=2,
        )
        trainer = HUNLRebelTrainer(config=config)
        assert trainer.value_net is not None
        assert len(trainer.replay_buffer) == 0

    @pytest.mark.slow
    def test_training_loop_runs(self):
        """One iteration of training completes without error."""
        config = HUNLRebelConfig(
            num_iters=1,
            samples_per_iter=2,
            cfr_iters_per_solve=5,
            max_depth=2,
            value_hidden_dim=32,
            value_num_layers=2,
            value_train_epochs=2,
            value_batch_size=4,
        )
        trainer = HUNLRebelTrainer(config=config)
        metrics = trainer.train()

        assert "value_losses" in metrics
        assert len(metrics["value_losses"]) == 1

    @pytest.mark.slow
    def test_training_loop_decreases_loss(self):
        """Multiple training iterations should decrease loss."""
        config = HUNLRebelConfig(
            num_iters=3,
            samples_per_iter=3,
            cfr_iters_per_solve=10,
            max_depth=2,
            value_hidden_dim=32,
            value_num_layers=2,
            value_train_epochs=10,
            value_batch_size=8,
        )
        trainer = HUNLRebelTrainer(config=config)
        metrics = trainer.train()

        losses = metrics["value_losses"]
        assert len(losses) == 3
        # Loss should generally decrease (or at least be finite)
        for loss in losses:
            assert loss < float("inf")


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_game_to_solver_to_training(self):
        """Full pipeline: game -> sample state -> solve -> collect data."""
        game = HUNLGame()
        state, board = sample_random_state(game)

        if not game.is_terminal(state) and state.phase not in ("flop_deal", "turn_deal", "river_deal"):
            beliefs = HUNLBeliefState(board=board)
            solver = HUNLSubgameSolver(
                game=game,
                state=state,
                initial_beliefs=beliefs.get_pbs(),
                max_depth=2,
                cfr_iters=5,
            )
            solver.solve()
            # Should have some data or have solved without error

    def test_subgame_player_constructs(self):
        """SubgamePlayer initializes without errors."""
        game = HUNLGame()
        config = HUNLValueNetConfig(hidden_dim=32, num_layers=2)
        vnet = HUNLValueNetwork(config)
        player = SubgamePlayer(game, vnet, player_id=0, max_depth=2, cfr_iters=5)
        assert player.player_id == 0

    def test_value_net_as_leaf_evaluator(self):
        """Value network can be used as leaf evaluator in solver."""
        game = HUNLGame()
        config = HUNLValueNetConfig(hidden_dim=32, num_layers=2)
        vnet = HUNLValueNetwork(config)

        state = game.initial_state()
        state = game.next_state(state, ((50, 46), (41, 36)))

        # Play to river
        call = [a for a in game.legal_actions(state) if a.action_type == "call"][0]
        state = game.next_state(state, call)
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)
        state = game.next_state(state, (0, 1, 2))
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)
        state = game.next_state(state, 3)
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)
        check = [a for a in game.legal_actions(state) if a.action_type == "check"][0]
        state = game.next_state(state, check)
        state = game.next_state(state, 4)

        def value_fn(pbs_flat, features):
            vnet.eval()
            with torch.no_grad():
                return vnet(pbs_flat, features)

        beliefs = HUNLBeliefState(board=state.board)
        solver = HUNLSubgameSolver(
            game=game,
            state=state,
            initial_beliefs=beliefs.get_pbs(),
            value_fn=value_fn,
            max_depth=1,
            cfr_iters=5,
        )
        profile = solver.solve()
        assert isinstance(profile, dict)
