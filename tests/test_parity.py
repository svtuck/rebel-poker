"""Tests verifying parity with Noam Brown's poker_solver.

These tests compare exploitability numbers at specific iteration checkpoints
to ensure our implementations match the reference exactly.
"""

import sys
sys.path.insert(0, ".")

import pytest
from cfr.solver import CFRConfig, CFRTrainer
from cfr.mccfr import ExternalSamplingMCCFRTrainer, MCCFRConfig
from cfr.fictitious_play import FictitiousPlayTrainer, FPConfig
from kuhn.game import KuhnPoker
from leduc.game import LeducPoker


# ---- Kuhn Poker reference numbers from Noam Brown's solver ----

KUHN_REFERENCE = {
    "cfr": {25: 0.048120, 100: 0.023301, 400: 0.011118, 1600: 0.005387},
    "cfr+": {25: 0.048002, 100: 0.015742, 400: 0.005547, 1600: 0.002532},
    "dcfr": {25: 0.063903, 100: 0.011490, 400: 0.006842, 1600: 0.002237},
    "mccfr": {25: 0.196201, 100: 0.117633, 400: 0.052845, 1600: 0.020856},
    "cfr_alt": {25: 0.039037, 100: 0.023364, 400: 0.010478, 1600: 0.005127},
    "cfr+_alt": {25: 0.003411, 100: 0.001194, 400: 0.000254, 1600: 0.000052},
}


def _exploitability(game, profile):
    """Standalone exploitability from a strategy profile dict."""
    trainer = CFRTrainer(game)
    # Reconstruct infosets from profile for exploitability calculation
    for key, action_probs in profile.items():
        from cfr.solver import InfoSet
        actions = list(action_probs.keys())
        iset = InfoSet(actions)
        probs = list(action_probs.values())
        iset.strategy_sum = [p * 10000 for p in probs]
        trainer.infosets[key] = iset
    return trainer.exploitability()


def _run_to_checkpoints(trainer, checkpoints, game=None):
    results = {}
    completed = 0
    for target in sorted(checkpoints):
        trainer.run(target - completed)
        completed = target
        if hasattr(trainer, 'exploitability'):
            results[target] = trainer.exploitability()
        else:
            profile = trainer.average_strategy_profile()
            results[target] = _exploitability(game, profile)
    return results


@pytest.fixture
def kuhn():
    return KuhnPoker()


@pytest.fixture
def leduc():
    return LeducPoker()


class TestKuhnCFRParity:
    def test_vanilla_cfr(self, kuhn):
        trainer = CFRTrainer(kuhn, CFRConfig())
        results = _run_to_checkpoints(trainer, KUHN_REFERENCE["cfr"])
        for it, expected in KUHN_REFERENCE["cfr"].items():
            assert abs(results[it] - expected) < 1e-6, f"CFR at {it}: {results[it]} != {expected}"

    def test_cfr_plus(self, kuhn):
        trainer = CFRTrainer(kuhn, CFRConfig(use_plus=True, linear_weighting=True))
        results = _run_to_checkpoints(trainer, KUHN_REFERENCE["cfr+"])
        for it, expected in KUHN_REFERENCE["cfr+"].items():
            assert abs(results[it] - expected) < 1e-6, f"CFR+ at {it}: {results[it]} != {expected}"

    def test_dcfr(self, kuhn):
        trainer = CFRTrainer(kuhn, CFRConfig(use_dcfr=True))
        results = _run_to_checkpoints(trainer, KUHN_REFERENCE["dcfr"])
        for it, expected in KUHN_REFERENCE["dcfr"].items():
            assert abs(results[it] - expected) < 1e-6, f"DCFR at {it}: {results[it]} != {expected}"

    def test_mccfr(self, kuhn):
        trainer = ExternalSamplingMCCFRTrainer(kuhn, MCCFRConfig(seed=7))
        results = _run_to_checkpoints(trainer, KUHN_REFERENCE["mccfr"], game=kuhn)
        for it, expected in KUHN_REFERENCE["mccfr"].items():
            assert abs(results[it] - expected) < 1e-6, f"MCCFR at {it}: {results[it]} != {expected}"

    def test_cfr_alternating(self, kuhn):
        trainer = CFRTrainer(kuhn, CFRConfig(alternating=True))
        results = _run_to_checkpoints(trainer, KUHN_REFERENCE["cfr_alt"])
        for it, expected in KUHN_REFERENCE["cfr_alt"].items():
            assert abs(results[it] - expected) < 1e-6, f"CFR alt at {it}: {results[it]} != {expected}"

    def test_cfr_plus_alternating(self, kuhn):
        trainer = CFRTrainer(kuhn, CFRConfig(use_plus=True, linear_weighting=True, alternating=True))
        results = _run_to_checkpoints(trainer, KUHN_REFERENCE["cfr+_alt"])
        for it, expected in KUHN_REFERENCE["cfr+_alt"].items():
            assert abs(results[it] - expected) < 1e-6, f"CFR+ alt at {it}: {results[it]} != {expected}"


class TestLeducCFRParity:
    """Leduc parity verified at checkpoints 25 and 50 against Noam's output."""

    def test_vanilla_cfr(self, leduc):
        trainer = CFRTrainer(leduc, CFRConfig())
        trainer.run(25)
        exp25 = trainer.exploitability()
        assert abs(exp25 - 0.108020) < 1e-5, f"Leduc CFR at 25: {exp25}"
        trainer.run(25)
        exp50 = trainer.exploitability()
        assert abs(exp50 - 0.066034) < 1e-5, f"Leduc CFR at 50: {exp50}"


class TestLeducGame:
    def test_initial_state(self, leduc):
        state = leduc.initial_state()
        assert leduc.current_player(state) == -1
        assert not leduc.is_terminal(state)

    def test_chance_deals(self, leduc):
        state = leduc.initial_state()
        outcomes = leduc.chance_outcomes(state)
        assert len(outcomes) == 30  # 6 * 5 = 30 ordered deals
        total = sum(p for _, p in outcomes)
        assert abs(total - 1.0) < 1e-10

    def test_terminal_fold(self, leduc):
        state = leduc.initial_state()
        cards = ((0, 0), (1, 0))  # J vs Q
        state = leduc.next_state(state, cards)
        state = leduc.next_state(state, "b")  # P0 bets
        state = leduc.next_state(state, "f")  # P1 folds
        assert leduc.is_terminal(state)
        assert leduc.terminal_utility(state, 0) > 0  # P0 wins

    def test_showdown(self, leduc):
        state = leduc.initial_state()
        cards = ((2, 0), (0, 0))  # K vs J
        state = leduc.next_state(state, cards)
        state = leduc.next_state(state, "c")  # P0 checks
        state = leduc.next_state(state, "c")  # P1 checks, end round 1
        # Now at chance node for board card
        assert leduc.current_player(state) == -1
        board_card = (1, 0)  # Q on board
        state = leduc.next_state(state, board_card)
        state = leduc.next_state(state, "c")
        state = leduc.next_state(state, "c")
        assert leduc.is_terminal(state)
        # K beats J when no pair
        assert leduc.terminal_utility(state, 0) > 0

    def test_pair_wins(self, leduc):
        state = leduc.initial_state()
        cards = ((1, 0), (2, 0))  # Q vs K
        state = leduc.next_state(state, cards)
        state = leduc.next_state(state, "c")
        state = leduc.next_state(state, "c")
        board_card = (1, 1)  # Q on board — P0 pairs!
        state = leduc.next_state(state, board_card)
        state = leduc.next_state(state, "c")
        state = leduc.next_state(state, "c")
        assert leduc.is_terminal(state)
        # Q pairs the board, beats K high
        assert leduc.terminal_utility(state, 0) > 0


class TestHandEvaluation:
    def test_high_card(self):
        from river.cards import evaluate_5, card_id
        cards = [card_id("2c"), card_id("4d"), card_id("7h"), card_id("9s"), card_id("Ac")]
        result = evaluate_5(cards)
        assert result[0] == 0  # High card

    def test_pair(self):
        from river.cards import evaluate_5, card_id
        cards = [card_id("Ac"), card_id("Ad"), card_id("7h"), card_id("9s"), card_id("2c")]
        result = evaluate_5(cards)
        assert result[0] == 1  # One pair

    def test_flush(self):
        from river.cards import evaluate_5, card_id
        cards = [card_id("2c"), card_id("5c"), card_id("7c"), card_id("9c"), card_id("Ac")]
        result = evaluate_5(cards)
        assert result[0] == 5  # Flush

    def test_straight(self):
        from river.cards import evaluate_5, card_id
        cards = [card_id("5c"), card_id("6d"), card_id("7h"), card_id("8s"), card_id("9c")]
        result = evaluate_5(cards)
        assert result[0] == 4  # Straight

    def test_straight_flush(self):
        from river.cards import evaluate_5, card_id
        cards = [card_id("5c"), card_id("6c"), card_id("7c"), card_id("8c"), card_id("9c")]
        result = evaluate_5(cards)
        assert result[0] == 8  # Straight flush

    def test_wheel_straight(self):
        from river.cards import evaluate_5, card_id
        cards = [card_id("Ac"), card_id("2d"), card_id("3h"), card_id("4s"), card_id("5c")]
        result = evaluate_5(cards)
        assert result[0] == 4  # Straight (wheel)
        assert result[1] == 5  # 5-high

    def test_evaluate_7(self):
        from river.cards import evaluate_7, card_id
        cards = [card_id("Ac"), card_id("Kc"), card_id("Qc"), card_id("Jc"),
                 card_id("Tc"), card_id("2d"), card_id("3h")]
        result = evaluate_7(cards)
        assert result[0] == 8  # Royal flush (straight flush)

    def test_hand_comparison(self):
        from river.cards import evaluate_5, card_id
        flush = evaluate_5([card_id("2c"), card_id("5c"), card_id("7c"), card_id("9c"), card_id("Ac")])
        straight = evaluate_5([card_id("5c"), card_id("6d"), card_id("7h"), card_id("8s"), card_id("9c")])
        assert flush > straight  # Flush beats straight


class TestRiverGame:
    def test_initial_state(self):
        from river.game import RiverConfig, RiverHoldemGame
        config = RiverConfig(
            board=("Ks", "Th", "7s", "4d", "2s"),
            pot=1000, stacks=(9500, 9500),
            bet_sizes=(0.5, 1.0),
        )
        game = RiverHoldemGame(config)
        state = game.initial_state()
        assert game.current_player(state) == 0
        assert not game.is_terminal(state)
        assert len(game.hands[0]) > 0
        assert len(game.hands[1]) > 0

    def test_check_check_is_terminal(self):
        from river.game import Action, RiverConfig, RiverHoldemGame
        config = RiverConfig(
            board=("Ks", "Th", "7s", "4d", "2s"),
            pot=1000, stacks=(9500, 9500),
        )
        game = RiverHoldemGame(config)
        state = game.initial_state()
        state = game.next_state(state, Action("c", 0))
        state = game.next_state(state, Action("c", 0))
        assert game.is_terminal(state)
        assert state.terminal_winner is None  # Showdown

    def test_fold_is_terminal(self):
        from river.game import Action, RiverConfig, RiverHoldemGame
        config = RiverConfig(
            board=("Ks", "Th", "7s", "4d", "2s"),
            pot=1000, stacks=(9500, 9500),
        )
        game = RiverHoldemGame(config)
        state = game.initial_state()
        actions = game.legal_actions(state)
        # Find a bet action
        bet_action = [a for a in actions if a.label == "b"][0]
        state = game.next_state(state, bet_action)
        state = game.next_state(state, Action("f", 0))
        assert game.is_terminal(state)
        assert state.terminal_winner == 0  # P1 folded, P0 wins


class TestFictitiousPlay:
    def test_fp_converges_kuhn(self, kuhn):
        trainer = FictitiousPlayTrainer(kuhn, FPConfig(linear_weighting=True, alternating=True))
        trainer.run(200)
        profile = trainer.average_strategy_profile()
        # Check it produces a valid profile
        assert len(profile) > 0
        for key, probs in profile.items():
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-6, f"Probabilities don't sum to 1: {total}"
