"""Kuhn Poker game definition.

Kuhn Poker is a simplified poker game with 3 cards (J, Q, K) and 2 players.
Each player antes 1 chip, receives one card, then can check/bet in a single round.

Action encoding:
  'c' = check/call
  'b' = bet
  'f' = fold

Terminal histories: cc, bc, bf, cbc, cbf
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

NUM_CARDS = 3
CARD_RANKS = list(range(NUM_CARDS))
RANK_NAMES = {0: "J", 1: "Q", 2: "K"}
TERMINAL_HISTORIES = {"cc", "bc", "bf", "cbc", "cbf"}


@dataclass(frozen=True)
class KuhnState:
    """A state in Kuhn Poker.

    cards: (player0_card, player1_card) or None if not dealt yet.
    history: string of actions taken so far.
    """
    cards: Optional[Tuple[int, int]]
    history: str


class KuhnPoker:
    """Kuhn Poker game engine."""

    NUM_PLAYERS = 2
    NUM_CARDS = NUM_CARDS

    def initial_state(self) -> KuhnState:
        return KuhnState(cards=None, history="")

    def is_terminal(self, state: KuhnState) -> bool:
        return state.history in TERMINAL_HISTORIES

    def current_player(self, state: KuhnState) -> Optional[int]:
        """Return current player (0 or 1), -1 for chance, None for terminal."""
        if state.cards is None:
            return -1
        if self.is_terminal(state):
            return None
        return len(state.history) % 2

    def legal_actions(self, state: KuhnState) -> List[str]:
        if self.is_terminal(state):
            return []
        h = state.history
        if h in ("", "c"):
            return ["c", "b"]
        if h in ("b", "cb"):
            return ["c", "f"]
        return []

    def chance_outcomes(self, state: KuhnState) -> List[Tuple[Tuple[int, int], float]]:
        """Return all possible card dealings with uniform probability."""
        if state.cards is not None:
            return []
        outcomes = []
        for i in CARD_RANKS:
            for j in CARD_RANKS:
                if i != j:
                    outcomes.append(((i, j), 1.0 / 6.0))
        return outcomes

    def next_state(self, state: KuhnState, action) -> KuhnState:
        if state.cards is None:
            return KuhnState(cards=action, history="")
        return KuhnState(cards=state.cards, history=state.history + action)

    def infoset_key(self, state: KuhnState, player: int) -> str:
        """Information set key: private card + public history."""
        card = state.cards[player]
        return f"{RANK_NAMES[card]}|{state.history}"

    def terminal_utility(self, state: KuhnState, player: int) -> float:
        """Utility for `player` at a terminal state."""
        h = state.history
        if h == "bf":
            winner, pot, contrib = 0, 3, (2, 1)
        elif h == "cbf":
            winner, pot, contrib = 1, 3, (1, 2)
        elif h == "cc":
            pot, contrib = 2, (1, 1)
            winner = 0 if state.cards[0] > state.cards[1] else 1
        elif h in ("bc", "cbc"):
            pot, contrib = 4, (2, 2)
            winner = 0 if state.cards[0] > state.cards[1] else 1
        else:
            raise ValueError(f"Non-terminal history: {h}")

        if winner == player:
            return float(pot - contrib[player])
        return float(-contrib[player])

    # ---- Enumeration helpers for belief state tracking ----

    def all_deals(self) -> List[Tuple[int, int]]:
        """All possible (card0, card1) deals."""
        return [(i, j) for i in CARD_RANKS for j in CARD_RANKS if i != j]

    def all_histories(self) -> List[str]:
        """All non-chance histories (including terminal)."""
        return ["", "c", "b", "cb", "cc", "bc", "bf", "cbc", "cbf"]

    def all_infosets(self, player: int) -> List[str]:
        """All information set keys for a player."""
        infosets = []
        for card in CARD_RANKS:
            for h in self.all_histories():
                if h in TERMINAL_HISTORIES:
                    continue
                if h == "":
                    if player != 0:
                        continue
                elif h == "c":
                    if player != 1:
                        continue
                elif h == "b":
                    if player != 1:
                        continue
                elif h == "cb":
                    if player != 0:
                        continue
                infosets.append(f"{RANK_NAMES[card]}|{h}")
        return infosets
