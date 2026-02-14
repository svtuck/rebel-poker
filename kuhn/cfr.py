"""Vanilla CFR (Counterfactual Regret Minimization).

This implements standard CFR with:
- Regret matching for strategy computation
- Reach-weighted strategy accumulation for average strategy
- Exploitability computation via best response

The solver accepts any game that implements the Game protocol, making it
usable with Kuhn Poker, Leduc Poker, Liar's Dice, or any other
imperfect-information game.

The implementation tracks reach probabilities for all card deals simultaneously,
which is the foundation for public belief state tracking in ReBeL.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from game_interface import Game
from kuhn.game import KuhnPoker


class InfoSet:
    """Tracks regrets and cumulative strategy for one information set."""

    def __init__(self, actions: List[str]) -> None:
        self.actions = actions
        n = len(actions)
        self.regret_sum = [0.0] * n
        self.strategy_sum = [0.0] * n

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


class CFRTrainer:
    """Vanilla CFR solver for any imperfect-information game.

    Accepts any game implementing the Game protocol (game_interface.py).
    Defaults to KuhnPoker if no game is provided.
    """

    def __init__(self, game: Optional[Game] = None) -> None:
        self.game: Game = game or KuhnPoker()
        self.infosets: Dict[str, InfoSet] = {}
        self.iteration = 0

    def _get_infoset(self, state, player: int) -> tuple[str, InfoSet]:
        key = self.game.infoset_key(state, player)
        if key not in self.infosets:
            actions = self.game.legal_actions(state)
            self.infosets[key] = InfoSet(actions)
        return key, self.infosets[key]

    def _cfr(self, state, reach_p0: float, reach_p1: float) -> float:
        """Recursive CFR traversal. Returns expected value for player 0."""
        if self.game.is_terminal(state):
            return self.game.terminal_utility(state, 0)

        if self.game.current_player(state) == -1:
            # Chance node
            value = 0.0
            for outcome, prob in self.game.chance_outcomes(state):
                value += prob * self._cfr(
                    self.game.next_state(state, outcome),
                    reach_p0, reach_p1
                )
            return value

        player = self.game.current_player(state)
        key, infoset = self._get_infoset(state, player)
        strategy = infoset.current_strategy()
        actions = infoset.actions

        action_utils = []
        node_util = 0.0
        for idx, action in enumerate(actions):
            if player == 0:
                child_util = self._cfr(
                    self.game.next_state(state, action),
                    reach_p0 * strategy[idx], reach_p1
                )
            else:
                child_util = self._cfr(
                    self.game.next_state(state, action),
                    reach_p0, reach_p1 * strategy[idx]
                )
            action_utils.append(child_util)
            node_util += strategy[idx] * child_util

        # Update regrets and strategy sums
        # Regret = counterfactual_reach * (action_value - node_value)
        # For player 1, utilities are negated (we compute from P0's perspective)
        opponent_reach = reach_p1 if player == 0 else reach_p0
        player_reach = reach_p0 if player == 0 else reach_p1

        for idx in range(len(actions)):
            if player == 0:
                regret = opponent_reach * (action_utils[idx] - node_util)
            else:
                # Player 1's utility is -node_util from P0's perspective
                regret = opponent_reach * (node_util - action_utils[idx])
            infoset.regret_sum[idx] += regret
            infoset.strategy_sum[idx] += player_reach * strategy[idx]

        return node_util

    def train(self, iterations: int) -> List[float]:
        """Run CFR for the given number of iterations.

        Returns exploitability at each iteration.
        """
        exploitabilities = []
        for _ in range(iterations):
            self.iteration += 1
            self._cfr(self.game.initial_state(), 1.0, 1.0)
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
        """Value of best response for br_player against the profile.

        Uses Noam Brown's approach: collect states by infoset, compute
        the best action per infoset weighted by opponent reach.
        """
        game = self.game

        # Phase 1: collect opponent reach to each state and group by infoset
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

        # Phase 2: compute values bottom-up with best response
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
