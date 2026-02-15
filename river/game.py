"""No-Limit Texas Hold'em River subgame.

Implements a configurable river betting game with:
- Arbitrary board cards and hand ranges
- Multiple bet sizing options
- Vectorized hand representation for efficient solving
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from river.cards import all_hole_cards, card_str, evaluate_7, parse_cards, parse_hand


@dataclass(frozen=True)
class Hand:
    cards: Tuple[int, int]
    weight: float
    strength: Tuple[int, ...]


@dataclass(frozen=True)
class Action:
    label: str
    amount: int = 0


@dataclass(frozen=True)
class RiverState:
    history: Tuple[str, ...]
    contrib: Tuple[int, int]
    player: Optional[int]
    checks: int
    raises: int
    terminal_winner: Optional[int]


@dataclass
class RiverConfig:
    board: Sequence[str]
    pot: int = 1000
    stacks: Tuple[int, int] = (9500, 9500)
    bet_sizes: Sequence[float] = (0.5, 1.0)
    oop_first_bets: Optional[Sequence[float]] = None
    ip_first_bets: Optional[Sequence[float]] = None
    oop_first_raises: Optional[Sequence[float]] = None
    ip_first_raises: Optional[Sequence[float]] = None
    oop_next_raises: Optional[Sequence[float]] = None
    ip_next_raises: Optional[Sequence[float]] = None
    include_all_in: bool = True
    max_raises: int = 1000
    ranges: Tuple[Optional[Sequence[str]], Optional[Sequence[str]]] = (None, None)
    range_weights: Tuple[Optional[Sequence[float]], Optional[Sequence[float]]] = (None, None)


class RiverHoldemGame:
    def __init__(self, config: RiverConfig) -> None:
        self.config = config
        self.board = parse_cards(config.board)
        self.base_pot = int(config.pot)
        self.stacks = (int(config.stacks[0]), int(config.stacks[1]))
        self.bet_sizes = list(config.bet_sizes)
        self.oop_first_bets = list(config.oop_first_bets) if config.oop_first_bets else self.bet_sizes
        self.ip_first_bets = list(config.ip_first_bets) if config.ip_first_bets else self.bet_sizes
        self.oop_first_raises = list(config.oop_first_raises) if config.oop_first_raises else self.bet_sizes
        self.ip_first_raises = list(config.ip_first_raises) if config.ip_first_raises else self.bet_sizes
        self.oop_next_raises = list(config.oop_next_raises) if config.oop_next_raises else self.bet_sizes
        self.ip_next_raises = list(config.ip_next_raises) if config.ip_next_raises else self.bet_sizes
        self.include_all_in = config.include_all_in
        self.max_raises = config.max_raises

        self.hands: List[List[Hand]] = []
        self.hand_weights: List[List[float]] = []
        self._legal_cache: Dict[Tuple[str, ...], List[Action]] = {}
        self._next_cache: Dict[Tuple[Tuple[str, ...], Action], RiverState] = {}
        for player in (0, 1):
            hand_list = self._build_hands(config.ranges[player], config.range_weights[player])
            self.hands.append(hand_list)
            weights = [hand.weight for hand in hand_list]
            total = sum(weights)
            if total <= 0:
                raise ValueError("Range weights must sum to > 0")
            self.hand_weights.append([w / total for w in weights])

    def _build_hands(
        self, hand_strings: Optional[Sequence[str]], weights: Optional[Sequence[float]]
    ) -> List[Hand]:
        if hand_strings is None:
            hole_cards = all_hole_cards(self.board)
        else:
            hole_cards = [parse_hand(hand) for hand in hand_strings]
        if weights is None:
            weights = [1.0 for _ in hole_cards]
        if len(weights) != len(hole_cards):
            raise ValueError("Weights must match number of hands")
        hands = []
        for cards, weight in zip(hole_cards, weights):
            if any(card in self.board for card in cards):
                continue
            strength = evaluate_7(list(cards) + list(self.board))
            hands.append(Hand(cards=cards, weight=float(weight), strength=strength))
        return hands

    def initial_state(self) -> RiverState:
        return RiverState(history=(), contrib=(0, 0), player=0, checks=0, raises=0, terminal_winner=None)

    def is_terminal(self, state: RiverState) -> bool:
        return state.player is None

    def current_player(self, state: RiverState) -> Optional[int]:
        return state.player

    def pot_total(self, state: RiverState) -> int:
        return self.base_pot + state.contrib[0] + state.contrib[1]

    def infoset_key(self, state: RiverState, player: int) -> str:
        if not state.history:
            return "root"
        return "/".join(state.history)

    def legal_actions(self, state: RiverState) -> List[Action]:
        if self.is_terminal(state):
            return []
        cached = self._legal_cache.get(state.history)
        if cached is not None:
            return cached
        player = state.player
        to_call = max(state.contrib) - state.contrib[player]
        remaining = self.stacks[player] - state.contrib[player]
        pot_total = self.pot_total(state)
        sizes = self.bet_sizes

        actions = []
        if to_call == 0:
            actions.append(Action("c", 0))
            if state.checks == 0 and player == 0:
                sizes = self.oop_first_bets
            elif state.checks == 1 and player == 1:
                sizes = self.ip_first_bets
            amounts = []
            for size in sizes:
                bet_amount = int(round(pot_total * size))
                if bet_amount <= 0:
                    continue
                bet_amount = min(bet_amount, remaining)
                if bet_amount > 0:
                    amounts.append(bet_amount)
            if self.include_all_in and remaining > 0:
                amounts.append(remaining)
            for amount in sorted(set(amounts)):
                if amount > 0:
                    actions.append(Action("b", amount))
            self._legal_cache[state.history] = actions
            return actions

        actions.append(Action("c", to_call))
        actions.append(Action("f", 0))
        if state.raises >= self.max_raises:
            self._legal_cache[state.history] = actions
            return actions

        if state.raises == 1:
            sizes = self.oop_first_raises if player == 0 else self.ip_first_raises
        elif state.raises > 1:
            sizes = self.oop_next_raises if player == 0 else self.ip_next_raises
        pot_after_call = pot_total + to_call
        amounts = []
        for size in sizes:
            raise_amount = int(round(pot_after_call * size))
            if raise_amount <= 0:
                continue
            total_add = to_call + raise_amount
            if total_add > remaining:
                total_add = remaining
                raise_amount = total_add - to_call
            if raise_amount > 0 and total_add > to_call:
                amounts.append(raise_amount)
        if self.include_all_in and remaining > to_call:
            amounts.append(remaining - to_call)
        for amount in sorted(set(amounts)):
            actions.append(Action("r", amount))
        self._legal_cache[state.history] = actions
        return actions

    def next_state(self, state: RiverState, action: Action) -> RiverState:
        cached = self._next_cache.get((state.history, action))
        if cached is not None:
            return cached
        player = state.player
        contrib = list(state.contrib)
        history = list(state.history)
        checks = state.checks
        raises = state.raises

        if action.label == "f":
            ns = RiverState(
                history=tuple(history + ["f"]),
                contrib=tuple(contrib),
                player=None,
                checks=checks,
                raises=raises,
                terminal_winner=1 - player,
            )
            self._next_cache[(state.history, action)] = ns
            return ns

        if action.label == "c":
            to_call = max(contrib) - contrib[player]
            if to_call == 0:
                history.append("c")
                checks += 1
                if checks >= 2:
                    ns = RiverState(
                        history=tuple(history), contrib=tuple(contrib),
                        player=None, checks=checks, raises=raises, terminal_winner=None,
                    )
                    self._next_cache[(state.history, action)] = ns
                    return ns
                ns = RiverState(
                    history=tuple(history), contrib=tuple(contrib),
                    player=1 - player, checks=checks, raises=raises, terminal_winner=None,
                )
                self._next_cache[(state.history, action)] = ns
                return ns
            contrib[player] += to_call
            history.append("c")
            ns = RiverState(
                history=tuple(history), contrib=tuple(contrib),
                player=None, checks=0, raises=raises, terminal_winner=None,
            )
            self._next_cache[(state.history, action)] = ns
            return ns

        to_call = max(contrib) - contrib[player]
        amount = action.amount
        if action.label == "r":
            contrib[player] += to_call + amount
        else:
            contrib[player] += amount
        if action.label == "b":
            history.append(f"b{amount}")
        else:
            history.append(f"r{amount}")
        raises += 1
        ns = RiverState(
            history=tuple(history), contrib=tuple(contrib),
            player=1 - player, checks=0, raises=raises, terminal_winner=None,
        )
        self._next_cache[(state.history, action)] = ns
        return ns
