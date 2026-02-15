"""Fictitious Play for imperfect-information games.

Implements:
- Standard Fictitious Play
- Optimistic FP (double-weight last iterate)
- Linear weighting
- Alternating updates

Uses best response computation from the CFR trainer's exploitability code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from game_interface import Game


@dataclass
class FPConfig:
    optimistic: bool = False
    linear_weighting: bool = False
    alternating: bool = False


@dataclass
class FPInfoSet:
    actions: List[str]
    strategy_sum: List[float] = field(init=False)
    last_strategy: List[float] = field(init=False)

    def __post_init__(self) -> None:
        self.strategy_sum = [0.0 for _ in self.actions]
        self.last_strategy = [1.0 / len(self.actions) for _ in self.actions]

    def add_strategy(self, probs: List[float], weight: float) -> None:
        for idx, prob in enumerate(probs):
            self.strategy_sum[idx] += weight * prob
        self.last_strategy = list(probs)


def _compute_best_response(game: Game, profile: Dict[str, Dict[str, float]], br_player: int):
    state_reach: Dict = {}
    infoset_states: Dict[str, List] = {}
    infoset_actions: Dict[str, List[str]] = {}

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
            for action in actions:
                collect(game.next_state(state, action), reach_opp)
        else:
            probs = profile.get(game.infoset_key(state, player), {})
            for action in actions:
                p = probs.get(action, 1.0 / len(actions))
                collect(game.next_state(state, action), reach_opp * p)

    collect(game.initial_state(), 1.0)

    value_cache: Dict = {}
    action_cache: Dict[str, str] = {}

    def best_action(key: str) -> str:
        if key in action_cache:
            return action_cache[key]
        actions = infoset_actions[key]
        totals = [0.0 for _ in actions]
        for state in infoset_states[key]:
            for idx, action in enumerate(actions):
                totals[idx] += state_reach[state] * state_value(game.next_state(state, action))
        best_idx = max(range(len(actions)), key=lambda i: totals[i])
        action_cache[key] = actions[best_idx]
        return action_cache[key]

    def state_value(state) -> float:
        cached = value_cache.get(state)
        if cached is not None:
            return cached
        if game.is_terminal(state):
            value = game.terminal_utility(state, br_player)
        elif game.current_player(state) == -1:
            value = 0.0
            for outcome, prob in game.chance_outcomes(state):
                value += prob * state_value(game.next_state(state, outcome))
        else:
            player = game.current_player(state)
            actions = game.legal_actions(state)
            if player == br_player:
                key = game.infoset_key(state, player)
                action = best_action(key)
                value = state_value(game.next_state(state, action))
            else:
                probs = profile.get(game.infoset_key(state, player), {})
                value = 0.0
                for action in actions:
                    p = probs.get(action, 1.0 / len(actions))
                    value += p * state_value(game.next_state(state, action))
        value_cache[state] = value
        return value

    state_value(game.initial_state())
    return action_cache, infoset_actions


class FictitiousPlayTrainer:
    def __init__(self, game: Game, config: Optional[FPConfig] = None) -> None:
        self.game = game
        self.config = config or FPConfig()
        self.infosets: Dict[int, Dict[str, FPInfoSet]] = {0: {}, 1: {}}
        self.total_weight = {0: 0.0, 1: 0.0}
        self.last_weight = {0: 0.0, 1: 0.0}
        self.iteration = 0

    def _get_infoset(self, player: int, key: str, actions: List[str]) -> FPInfoSet:
        infoset = self.infosets[player].get(key)
        if infoset is None:
            infoset = FPInfoSet(actions)
            self.infosets[player][key] = infoset
        return infoset

    def _profile_for_player(self, player: int, optimistic: bool) -> Dict[str, Dict[str, float]]:
        profile: Dict[str, Dict[str, float]] = {}
        total_weight = self.total_weight[player]
        last_weight = self.last_weight[player] if optimistic else 0.0
        denom = total_weight + last_weight
        for key, infoset in self.infosets[player].items():
            if denom > 0.0:
                base = list(infoset.strategy_sum)
                if optimistic and last_weight > 0.0:
                    for idx, prob in enumerate(infoset.last_strategy):
                        base[idx] += last_weight * prob
                probs = [value / denom for value in base]
            else:
                probs = [1.0 / len(infoset.actions) for _ in infoset.actions]
            profile[key] = {action: probs[idx] for idx, action in enumerate(infoset.actions)}
        return profile

    def _best_response(self, player: int):
        opponent_profile = self._profile_for_player(1 - player, optimistic=self.config.optimistic)
        action_cache, infoset_actions = _compute_best_response(
            self.game, opponent_profile, br_player=player
        )
        strategy = {}
        for key, actions in infoset_actions.items():
            chosen = action_cache.get(key, actions[0])
            probs = [1.0 if action == chosen else 0.0 for action in actions]
            strategy[key] = (actions, probs)
        return strategy

    def _update_player(self, player: int, br_strategy) -> None:
        weight = float(self.iteration) if self.config.linear_weighting else 1.0
        self.last_weight[player] = weight
        self.total_weight[player] += weight

        full_strategy = dict(br_strategy)
        for key, infoset in self.infosets[player].items():
            if key not in full_strategy:
                full_strategy[key] = (infoset.actions, infoset.last_strategy)

        for key, (actions, probs) in full_strategy.items():
            infoset = self._get_infoset(player, key, actions)
            infoset.add_strategy(probs, weight)

    def run(self, iterations: int) -> None:
        for _ in range(iterations):
            self.iteration += 1
            if self.config.alternating:
                br0 = self._best_response(0)
                self._update_player(0, br0)
                br1 = self._best_response(1)
                self._update_player(1, br1)
            else:
                br0 = self._best_response(0)
                br1 = self._best_response(1)
                self._update_player(0, br0)
                self._update_player(1, br1)

    def average_strategy_profile(self) -> Dict[str, Dict[str, float]]:
        profile: Dict[str, Dict[str, float]] = {}
        for player in (0, 1):
            denom = self.total_weight[player]
            for key, infoset in self.infosets[player].items():
                if denom > 0.0:
                    avg = [value / denom for value in infoset.strategy_sum]
                else:
                    avg = [1.0 / len(infoset.actions) for _ in infoset.actions]
                profile[key] = {action: avg[idx] for idx, action in enumerate(infoset.actions)}
        return profile
