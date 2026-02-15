"""CFR solver with multiple algorithm variants.

Supports:
- Vanilla CFR
- CFR+ (regret clipping, optional linear weighting)
- DCFR (discounted CFR with per-iteration decay)
- Alternating updates (separate traversal per player per iteration)

The solver accepts any game that implements the Game protocol, making it
usable with Kuhn Poker, Leduc Poker, or any other imperfect-information game.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pow
from typing import Dict, List, Optional

from game_interface import Game


@dataclass
class CFRConfig:
    use_plus: bool = False
    linear_weighting: bool = False
    alternating: bool = False
    use_dcfr: bool = False
    dcfr_alpha: float = 1.5
    dcfr_beta: float = 0.0
    dcfr_gamma: float = 2.0


class InfoSet:
    """Tracks regrets and cumulative strategy for one information set."""

    def __init__(self, actions: List[str]) -> None:
        self.actions = actions
        n = len(actions)
        self.regret_sum = [0.0] * n
        self.strategy_sum = [0.0] * n
        self.last_dcfr_iter = 0

    def current_strategy(self) -> List[float]:
        """Regret-matching: normalize positive regrets."""
        positives = [max(r, 0.0) for r in self.regret_sum]
        total = sum(positives)
        if total > 0:
            return [r / total for r in positives]
        n = len(self.actions)
        return [1.0 / n] * n

    def average_strategy(self) -> List[float]:
        """Cumulative strategy average (the converged solution)."""
        total = sum(self.strategy_sum)
        if total > 0:
            return [s / total for s in self.strategy_sum]
        n = len(self.actions)
        return [1.0 / n] * n

    def apply_dcfr_discount(self, iteration: int, alpha: float, beta: float, gamma: float) -> None:
        if self.last_dcfr_iter == iteration:
            return
        for t in range(self.last_dcfr_iter + 1, iteration + 1):
            pos_base = pow(float(t), alpha)
            neg_base = pow(float(t), beta)
            pos_scale = pos_base / (pos_base + 1.0)
            neg_scale = neg_base / (neg_base + 1.0)
            strat_scale = pow(float(t) / (float(t) + 1.0), gamma)
            for idx, regret in enumerate(self.regret_sum):
                if regret > 0.0:
                    self.regret_sum[idx] = regret * pos_scale
                elif regret < 0.0:
                    self.regret_sum[idx] = regret * neg_scale
            for idx, value in enumerate(self.strategy_sum):
                self.strategy_sum[idx] = value * strat_scale
        self.last_dcfr_iter = iteration


class CFRTrainer:
    """CFR solver for any imperfect-information game.

    Accepts any game implementing the Game protocol (game_interface.py).
    Supports vanilla CFR, CFR+, DCFR, and alternating updates via CFRConfig.
    """

    def __init__(self, game: Game, config: Optional[CFRConfig] = None) -> None:
        self.game: Game = game
        self.config = config or CFRConfig()
        self.infosets: Dict[str, InfoSet] = {}
        self.iteration = 0
        self._pending_regret: Dict[str, List[float]] = {}

    def _get_infoset(self, state, player: int) -> tuple[str, InfoSet]:
        key = self.game.infoset_key(state, player)
        infoset = self.infosets.get(key)
        if infoset is None:
            actions = self.game.legal_actions(state)
            infoset = InfoSet(actions)
            self.infosets[key] = infoset
        return key, infoset

    def _accumulate_regret(self, key: str, infoset: InfoSet, deltas: List[float]) -> None:
        if not self.config.use_plus:
            for idx, delta in enumerate(deltas):
                infoset.regret_sum[idx] += delta
            return

        pending = self._pending_regret.get(key)
        if pending is None:
            pending = [0.0 for _ in infoset.actions]
        for idx, delta in enumerate(deltas):
            pending[idx] += delta
        self._pending_regret[key] = pending

    def _apply_regret_updates(self) -> None:
        if not self.config.use_plus:
            return
        for key, deltas in self._pending_regret.items():
            infoset = self.infosets[key]
            for idx, delta in enumerate(deltas):
                infoset.regret_sum[idx] = max(0.0, infoset.regret_sum[idx] + delta)
        self._pending_regret.clear()

    def _cfr(self, state, reach_p0: float, reach_p1: float,
             update_player: Optional[int] = None) -> float:
        """Recursive CFR traversal. Returns expected value for player 0."""
        if self.game.is_terminal(state):
            return self.game.terminal_utility(state, 0)

        if self.game.current_player(state) == -1:
            value = 0.0
            for outcome, prob in self.game.chance_outcomes(state):
                value += prob * self._cfr(
                    self.game.next_state(state, outcome),
                    reach_p0, reach_p1, update_player
                )
            return value

        player = self.game.current_player(state)
        key, infoset = self._get_infoset(state, player)
        if self.config.use_dcfr and (update_player is None or update_player == player):
            infoset.apply_dcfr_discount(
                self.iteration,
                self.config.dcfr_alpha,
                self.config.dcfr_beta,
                self.config.dcfr_gamma,
            )
        strategy = infoset.current_strategy()
        actions = infoset.actions

        action_utils = []
        node_util = 0.0
        for idx, action in enumerate(actions):
            if player == 0:
                child_util = self._cfr(
                    self.game.next_state(state, action),
                    reach_p0 * strategy[idx], reach_p1, update_player
                )
            else:
                child_util = self._cfr(
                    self.game.next_state(state, action),
                    reach_p0, reach_p1 * strategy[idx], update_player
                )
            action_utils.append(child_util)
            node_util += strategy[idx] * child_util

        if update_player is None or update_player == player:
            if player == 0:
                opponent_reach = reach_p1
                deltas = [opponent_reach * (u - node_util) for u in action_utils]
            else:
                opponent_reach = reach_p0
                deltas = [opponent_reach * (node_util - u) for u in action_utils]
            self._accumulate_regret(key, infoset, deltas)

            weight = reach_p0 if player == 0 else reach_p1
            if self.config.use_plus and self.config.linear_weighting and not self.config.use_dcfr:
                weight *= self.iteration
            for idx in range(len(actions)):
                infoset.strategy_sum[idx] += weight * strategy[idx]

        return node_util

    def run(self, iterations: int) -> None:
        """Run CFR for the given number of iterations (no exploitability tracking)."""
        for _ in range(iterations):
            self.iteration += 1
            if self.config.alternating:
                self._pending_regret.clear()
                self._cfr(self.game.initial_state(), 1.0, 1.0, update_player=0)
                self._apply_regret_updates()
                self._pending_regret.clear()
                self._cfr(self.game.initial_state(), 1.0, 1.0, update_player=1)
                self._apply_regret_updates()
            else:
                self._pending_regret.clear()
                self._cfr(self.game.initial_state(), 1.0, 1.0, update_player=None)
                self._apply_regret_updates()

    def train(self, iterations: int) -> List[float]:
        """Run CFR for the given number of iterations.

        Returns exploitability at each iteration.
        """
        exploitabilities = []
        for _ in range(iterations):
            self.iteration += 1
            if self.config.alternating:
                self._pending_regret.clear()
                self._cfr(self.game.initial_state(), 1.0, 1.0, update_player=0)
                self._apply_regret_updates()
                self._pending_regret.clear()
                self._cfr(self.game.initial_state(), 1.0, 1.0, update_player=1)
                self._apply_regret_updates()
            else:
                self._pending_regret.clear()
                self._cfr(self.game.initial_state(), 1.0, 1.0, update_player=None)
                self._apply_regret_updates()
            if self.iteration % max(1, iterations // 100) == 0 or self.iteration <= 10:
                exploitabilities.append(self.exploitability())
        return exploitabilities

    def average_strategy_profile(self) -> Dict[str, Dict[str, float]]:
        """Return the average strategy as {infoset_key: {action: prob}}."""
        profile = {}
        for key, infoset in self.infosets.items():
            avg = infoset.average_strategy()
            profile[key] = {a: avg[i] for i, a in enumerate(infoset.actions)}
        return profile

    def exploitability(self) -> float:
        """Compute exploitability of the current average strategy."""
        profile = self.average_strategy_profile()
        br0 = self._best_response_value(profile, 0)
        br1 = self._best_response_value(profile, 1)
        return 0.5 * (br0 + br1)

    def _best_response_value(
        self, profile: Dict[str, Dict[str, float]], br_player: int
    ) -> float:
        """Value of best response for br_player against the profile."""
        game = self.game

        state_reach: Dict = {}
        infoset_states: Dict[str, List] = {}
        infoset_actions: Dict[str, List] = {}

        def collect(state, reach_opp: float) -> None:
            state_reach[state] = reach_opp
            if game.is_terminal(state):
                return
            if game.current_player(state) == -1:
                for outcome, prob in game.chance_outcomes(state):
                    collect(game.next_state(state, outcome), reach_opp * prob)
                return
            player = game.current_player(state)
            actions = game.legal_actions(state)
            if player == br_player:
                key = game.infoset_key(state, player)
                infoset_states.setdefault(key, []).append(state)
                infoset_actions.setdefault(key, actions)
                for a in actions:
                    collect(game.next_state(state, a), reach_opp)
            else:
                key = game.infoset_key(state, player)
                action_probs = profile.get(key, {})
                for a in actions:
                    p = action_probs.get(a, 1.0 / len(actions))
                    collect(game.next_state(state, a), reach_opp * p)

        collect(game.initial_state(), 1.0)

        value_cache: Dict = {}
        action_cache: Dict[str, str] = {}

        def best_action(key: str) -> str:
            if key in action_cache:
                return action_cache[key]
            actions = infoset_actions[key]
            totals = [0.0] * len(actions)
            for state in infoset_states[key]:
                for idx, a in enumerate(actions):
                    totals[idx] += state_reach[state] * state_value(game.next_state(state, a))
            best_idx = max(range(len(actions)), key=lambda i: totals[i])
            action_cache[key] = actions[best_idx]
            return actions[best_idx]

        def state_value(state) -> float:
            if state in value_cache:
                return value_cache[state]
            if game.is_terminal(state):
                v = game.terminal_utility(state, br_player)
            elif game.current_player(state) == -1:
                v = 0.0
                for outcome, prob in game.chance_outcomes(state):
                    v += prob * state_value(game.next_state(state, outcome))
            else:
                player = game.current_player(state)
                actions = game.legal_actions(state)
                if player == br_player:
                    key = game.infoset_key(state, player)
                    a = best_action(key)
                    v = state_value(game.next_state(state, a))
                else:
                    key = game.infoset_key(state, player)
                    action_probs = profile.get(key, {})
                    v = 0.0
                    for a in actions:
                        p = action_probs.get(a, 1.0 / len(actions))
                        v += p * state_value(game.next_state(state, a))
            value_cache[state] = v
            return v

        return state_value(game.initial_state())
