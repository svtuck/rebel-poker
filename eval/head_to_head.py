"""Head-to-head evaluation for poker agents.

Provides:
- play_match(): Run N hands between two agents and compute EV + confidence intervals
- Baseline agents: AlwaysCall, AlwaysFold, RandomAgent
- Strategy-based agent wrapping a StrategyProfile
- Statistical analysis with confidence intervals
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

from game_interface import Game

StrategyProfile = Dict[str, Dict[str, float]]


class Agent(Protocol):
    """Interface for a poker-playing agent."""

    def choose_action(self, game: Game, state, player: int) -> str:
        ...


class StrategyAgent:
    """Agent that plays according to a fixed strategy profile."""

    def __init__(self, profile: StrategyProfile, seed: Optional[int] = None) -> None:
        self.profile = profile
        self.rng = random.Random(seed)

    def choose_action(self, game: Game, state, player: int) -> str:
        key = game.infoset_key(state, player)
        action_probs = self.profile.get(key, {})
        actions = game.legal_actions(state)

        if not action_probs:
            return self.rng.choice(actions)

        probs = [action_probs.get(a, 0.0) for a in actions]
        total = sum(probs)
        if total <= 0:
            return self.rng.choice(actions)
        probs = [p / total for p in probs]

        r = self.rng.random()
        cumulative = 0.0
        for a, p in zip(actions, probs):
            cumulative += p
            if r < cumulative:
                return a
        return actions[-1]


class AlwaysCallAgent:
    """Agent that always calls (or checks if no bet to call)."""

    def choose_action(self, game: Game, state, player: int) -> str:
        actions = game.legal_actions(state)
        if "c" in actions:
            return "c"
        return actions[0]


class AlwaysFoldAgent:
    """Agent that always folds when possible, otherwise checks."""

    def choose_action(self, game: Game, state, player: int) -> str:
        actions = game.legal_actions(state)
        if "f" in actions:
            return "f"
        if "c" in actions:
            return "c"
        return actions[0]


class AlwaysBetAgent:
    """Agent that always bets/raises when possible."""

    def choose_action(self, game: Game, state, player: int) -> str:
        actions = game.legal_actions(state)
        if "b" in actions:
            return "b"
        if "c" in actions:
            return "c"
        return actions[0]


class RandomAgent:
    """Agent that chooses uniformly at random."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def choose_action(self, game: Game, state, player: int) -> str:
        actions = game.legal_actions(state)
        return self.rng.choice(actions)


@dataclass
class MatchResult:
    """Result of a head-to-head match."""
    num_hands: int
    p0_total: float
    p1_total: float
    p0_mean: float
    p1_mean: float
    p0_std: float
    p1_std: float
    p0_ci_95: Tuple[float, float]
    p1_ci_95: Tuple[float, float]
    per_hand_results: List[float] = field(repr=False, default_factory=list)

    def __str__(self) -> str:
        return (
            f"Match ({self.num_hands} hands):\n"
            f"  P0: {self.p0_mean:+.4f} +/- {self.p0_std:.4f} "
            f"  95% CI: [{self.p0_ci_95[0]:+.4f}, {self.p0_ci_95[1]:+.4f}]\n"
            f"  P1: {self.p1_mean:+.4f} +/- {self.p1_std:.4f} "
            f"  95% CI: [{self.p1_ci_95[0]:+.4f}, {self.p1_ci_95[1]:+.4f}]"
        )


def play_hand(
    game: Game,
    p0_agent: Agent,
    p1_agent: Agent,
    rng: random.Random,
) -> float:
    """Play a single hand. Returns utility for player 0."""
    state = game.initial_state()

    while not game.is_terminal(state):
        player = game.current_player(state)

        if player == -1:
            outcomes = game.chance_outcomes(state)
            r = rng.random()
            cumulative = 0.0
            chosen = outcomes[0][0]
            for action, prob in outcomes:
                cumulative += prob
                if r < cumulative:
                    chosen = action
                    break
            state = game.next_state(state, chosen)
        elif player == 0:
            action = p0_agent.choose_action(game, state, 0)
            state = game.next_state(state, action)
        elif player == 1:
            action = p1_agent.choose_action(game, state, 1)
            state = game.next_state(state, action)
        else:
            raise ValueError(f"Unexpected player: {player}")

    return game.terminal_utility(state, 0)


def play_match(
    game: Game,
    p0_agent: Agent,
    p1_agent: Agent,
    num_hands: int = 10000,
    seed: Optional[int] = None,
    alternate_positions: bool = True,
) -> MatchResult:
    """Play a match between two agents.

    If alternate_positions=True, each "hand" plays two hands with swapped
    positions (AIVAT-style variance reduction).
    """
    rng = random.Random(seed)
    results: List[float] = []

    for _ in range(num_hands):
        v = play_hand(game, p0_agent, p1_agent, rng)

        if alternate_positions:
            v2 = play_hand(game, p1_agent, p0_agent, rng)
            v = (v + (-v2)) / 2.0

        results.append(v)

    p0_total = sum(results)
    p1_total = -p0_total

    n = len(results)
    p0_mean = p0_total / n
    p1_mean = -p0_mean

    if n > 1:
        p0_var = sum((r - p0_mean) ** 2 for r in results) / (n - 1)
        p0_std = math.sqrt(p0_var)
    else:
        p0_std = 0.0
    p1_std = p0_std

    if n > 0:
        se = p0_std / math.sqrt(n)
        p0_ci = (p0_mean - 1.96 * se, p0_mean + 1.96 * se)
        p1_ci = (-p0_ci[1], -p0_ci[0])
    else:
        p0_ci = (0.0, 0.0)
        p1_ci = (0.0, 0.0)

    return MatchResult(
        num_hands=n,
        p0_total=p0_total,
        p1_total=p1_total,
        p0_mean=p0_mean,
        p1_mean=p1_mean,
        p0_std=p0_std,
        p1_std=p1_std,
        p0_ci_95=p0_ci,
        p1_ci_95=p1_ci,
        per_hand_results=results,
    )


def expected_value(
    game: Game,
    profile: StrategyProfile,
) -> float:
    """Compute the exact expected value for player 0 under the strategy profile.

    Uses full tree traversal — exact for small games.
    """
    def ev(state, p0_prob: float, p1_prob: float) -> float:
        if game.is_terminal(state):
            return p0_prob * p1_prob * game.terminal_utility(state, 0)
        if game.current_player(state) == -1:
            v = 0.0
            for outcome, prob in game.chance_outcomes(state):
                v += prob * ev(game.next_state(state, outcome), p0_prob, p1_prob)
            return v
        player = game.current_player(state)
        key = game.infoset_key(state, player)
        action_probs = profile.get(key, {})
        actions = game.legal_actions(state)
        v = 0.0
        for a in actions:
            p = action_probs.get(a, 1.0 / len(actions))
            if player == 0:
                v += ev(game.next_state(state, a), p0_prob * p, p1_prob)
            else:
                v += ev(game.next_state(state, a), p0_prob, p1_prob * p)
        return v

    return ev(game.initial_state(), 1.0, 1.0)
