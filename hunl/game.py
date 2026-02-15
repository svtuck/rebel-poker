"""HUNL Hold'em game engine implementing the Game protocol.

Wraps the Rust poker engine (PyO3 bindings) to provide a Game-protocol-compatible
interface for use with CFR solvers and the ReBeL training loop.

The game uses action abstraction to keep the tree tractable:
- Fixed bet sizes relative to pot (e.g., 0.5x, 1x pot)
- Fold/check/call always included
- All-in always included

Card dealing is handled as chance nodes. For HUNL with 1326 possible
hole card combinations per player, the full deal space is ~1.75M.
We use factored beliefs (per-player marginals over 1326 hands) rather
than tracking the joint distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import poker_engine as pe


# All 1326 two-card combinations, sorted by (c0, c1) with c0 < c1
ALL_HOLE_COMBOS: List[Tuple[int, int]] = [
    (c0, c1) for c0, c1 in combinations(range(52), 2)
]
COMBO_INDEX: Dict[Tuple[int, int], int] = {
    combo: i for i, combo in enumerate(ALL_HOLE_COMBOS)
}
NUM_COMBOS = 1326  # C(52, 2)


@dataclass(frozen=True)
class HUNLAction:
    """Wrapper around a poker action for the Game protocol.

    Attributes:
        action_type: 'fold', 'check', 'call', 'raise', 'all_in'
        amount: bet/raise amount (None for fold/check/call/all_in)
        label: human-readable string for this action
    """
    action_type: str
    amount: Optional[int] = None

    def __repr__(self) -> str:
        if self.amount is not None:
            return f"{self.action_type}:{self.amount}"
        return self.action_type

    def __hash__(self) -> int:
        return hash((self.action_type, self.amount))


@dataclass
class HUNLState:
    """Game state wrapping the Rust PyGameState.

    Attributes:
        rust_state: underlying Rust game state
        p0_hand: player 0's hole cards as (c0, c1) tuple, or None before deal
        p1_hand: player 1's hole cards, or None before deal
        board: community cards dealt so far
        phase: 'predeal', 'preflop', 'flop_deal', 'flop', 'turn_deal', 'turn',
               'river_deal', 'river', 'terminal'
        action_history: list of action labels for infoset key construction
    """
    rust_state: Any
    p0_hand: Optional[Tuple[int, int]] = None
    p1_hand: Optional[Tuple[int, int]] = None
    board: Tuple[int, ...] = ()
    phase: str = "predeal"
    action_history: Tuple[str, ...] = ()

    def used_cards(self) -> set:
        """Cards that are dealt and cannot appear in other positions."""
        used = set()
        if self.p0_hand:
            used.update(self.p0_hand)
        if self.p1_hand:
            used.update(self.p1_hand)
        used.update(self.board)
        return used


@dataclass
class HUNLConfig:
    """Configuration for the HUNL game.

    Attributes:
        bet_fractions: pot-relative bet sizes for action abstraction
        starting_stacks: chips each player starts with (200bb default)
        small_blind: small blind amount
        big_blind: big blind amount
    """
    bet_fractions: Sequence[float] = (0.5, 1.0)
    starting_stacks: int = 200
    small_blind: int = 1
    big_blind: int = 2


class HUNLGame:
    """Heads-Up No-Limit Hold'em implementing the Game protocol.

    This game uses the Rust poker engine for state management and action
    abstraction for tractable tree sizes. Board card dealing happens
    through chance nodes between betting streets.

    The game flow is:
    1. Chance: deal hole cards to both players (combined into one deal)
    2. Preflop betting (P0 = SB acts first)
    3. Chance: deal flop (3 cards)
    4. Flop betting (P0 = OOP acts first)
    5. Chance: deal turn (1 card)
    6. Turn betting
    7. Chance: deal river (1 card)
    8. River betting
    9. Showdown / fold
    """

    NUM_PLAYERS = 2

    def __init__(self, config: Optional[HUNLConfig] = None) -> None:
        self.config = config or HUNLConfig()
        self._action_abstraction = pe.PyActionAbstraction()
        # Cache for legal actions at each state to avoid recomputation
        self._action_cache: Dict[int, List[HUNLAction]] = {}

    def initial_state(self) -> HUNLState:
        """Return the initial state before dealing."""
        return HUNLState(
            rust_state=pe.PyGameState(),
            phase="predeal",
        )

    def is_terminal(self, state: HUNLState) -> bool:
        if state.phase == "predeal":
            return False
        if state.phase in ("flop_deal", "turn_deal", "river_deal"):
            return False
        return state.rust_state.is_terminal

    def current_player(self, state: HUNLState) -> Optional[int]:
        if self.is_terminal(state):
            return None
        if state.phase in ("predeal", "flop_deal", "turn_deal", "river_deal"):
            return -1  # chance node
        return state.rust_state.active_player

    def legal_actions(self, state: HUNLState) -> List[Any]:
        if self.is_terminal(state):
            return []
        if state.phase == "predeal":
            return []  # Use chance_outcomes instead
        if state.phase in ("flop_deal", "turn_deal", "river_deal"):
            return []  # Use chance_outcomes instead

        # Get abstract actions from Rust engine
        abs_actions = self._action_abstraction.abstract_actions(state.rust_state)
        return [
            HUNLAction(action_type=a.action_type, amount=a.amount)
            for a in abs_actions
        ]

    def chance_outcomes(self, state: HUNLState) -> List[Tuple[Any, float]]:
        """Return chance outcomes for deal nodes.

        For the predeal phase, returns all valid (p0_hand, p1_hand) combinations
        that don't share cards. Each deal has equal probability.

        For board dealing phases, returns all possible board cards.
        """
        if state.phase == "predeal":
            return self._predeal_outcomes()
        elif state.phase == "flop_deal":
            return self._flop_outcomes(state)
        elif state.phase == "turn_deal":
            return self._turn_outcomes(state)
        elif state.phase == "river_deal":
            return self._river_outcomes(state)
        return []

    def _predeal_outcomes(self) -> List[Tuple[Any, float]]:
        """All possible hole card deals for both players.

        Returns deals as ((c0, c1), (c2, c3)) tuples.
        Total: C(52,2) * C(50,2) / 2 if we counted ordered pairs,
        but we return each unique {hand_p0, hand_p1} unordered pair once.
        """
        outcomes = []
        all_combos = ALL_HOLE_COMBOS
        n = len(all_combos)

        for i in range(n):
            h0 = all_combos[i]
            for j in range(i + 1, n):
                h1 = all_combos[j]
                # Hands can't share cards
                if h0[0] == h1[0] or h0[0] == h1[1] or h0[1] == h1[0] or h0[1] == h1[1]:
                    continue
                outcomes.append(((h0, h1), 1.0))

        # Normalize probabilities
        total = len(outcomes)
        return [((h0, h1), 1.0 / total) for (h0, h1), _ in outcomes]

    def _flop_outcomes(self, state: HUNLState) -> List[Tuple[Any, float]]:
        """All possible flop cards (3 cards from remaining deck)."""
        used = state.used_cards()
        remaining = [c for c in range(52) if c not in used]
        outcomes = list(combinations(remaining, 3))
        prob = 1.0 / len(outcomes)
        return [(flop, prob) for flop in outcomes]

    def _turn_outcomes(self, state: HUNLState) -> List[Tuple[Any, float]]:
        """All possible turn cards (1 card from remaining deck)."""
        used = state.used_cards()
        remaining = [c for c in range(52) if c not in used]
        prob = 1.0 / len(remaining)
        return [(card, prob) for card in remaining]

    def _river_outcomes(self, state: HUNLState) -> List[Tuple[Any, float]]:
        """All possible river cards (1 card from remaining deck)."""
        used = state.used_cards()
        remaining = [c for c in range(52) if c not in used]
        prob = 1.0 / len(remaining)
        return [(card, prob) for card in remaining]

    def next_state(self, state: HUNLState, action: Any) -> HUNLState:
        if state.phase == "predeal":
            return self._apply_deal(state, action)
        elif state.phase in ("flop_deal", "turn_deal", "river_deal"):
            return self._apply_board_deal(state, action)
        else:
            return self._apply_action(state, action)

    def _apply_deal(self, state: HUNLState, deal: Tuple) -> HUNLState:
        """Apply the initial deal of hole cards."""
        h0, h1 = deal
        rs = pe.PyGameState()
        rs.deal_hole_cards(h0[0], h0[1], h1[0], h1[1])
        return HUNLState(
            rust_state=rs,
            p0_hand=h0,
            p1_hand=h1,
            board=(),
            phase="preflop",
            action_history=(),
        )

    def _apply_board_deal(self, state: HUNLState, cards: Any) -> HUNLState:
        """Deal community cards (flop/turn/river)."""
        rs = state.rust_state

        if state.phase == "flop_deal":
            c0, c1, c2 = cards
            rs.deal_flop(c0, c1, c2)
            new_board = state.board + (c0, c1, c2)
            new_phase = "flop"
        elif state.phase == "turn_deal":
            rs.deal_turn(cards)
            new_board = state.board + (cards,)
            new_phase = "turn"
        elif state.phase == "river_deal":
            rs.deal_river(cards)
            new_board = state.board + (cards,)
            new_phase = "river"
        else:
            raise ValueError(f"Invalid phase for board deal: {state.phase}")

        return HUNLState(
            rust_state=rs,
            p0_hand=state.p0_hand,
            p1_hand=state.p1_hand,
            board=new_board,
            phase=new_phase,
            action_history=state.action_history + ("/",),
        )

    def _apply_action(self, state: HUNLState, action: HUNLAction) -> HUNLState:
        """Apply a betting action."""
        # Find the matching Rust action
        rust_actions = state.rust_state.legal_actions()
        rust_action = None

        for ra in rust_actions:
            if action.action_type == ra.action_type:
                if action.amount is None and ra.amount is None:
                    rust_action = ra
                    break
                elif action.amount is not None and ra.amount is not None:
                    if action.amount == ra.amount:
                        rust_action = ra
                        break

        if rust_action is None:
            # For off-tree actions, find the closest legal action
            if action.action_type in ("raise", "bet"):
                best_ra = None
                best_diff = float("inf")
                target = action.amount or 0
                for ra in rust_actions:
                    if ra.action_type == "raise" and ra.amount is not None:
                        diff = abs(ra.amount - target)
                        if diff < best_diff:
                            best_diff = diff
                            best_ra = ra
                rust_action = best_ra

            if rust_action is None:
                # Fall back: match by type
                for ra in rust_actions:
                    if ra.action_type == action.action_type:
                        rust_action = ra
                        break
            if rust_action is None:
                # Last resort: check/call or first action
                for ra in rust_actions:
                    if ra.action_type in ("check", "call"):
                        rust_action = ra
                        break
                if rust_action is None:
                    rust_action = rust_actions[0]

        new_rs = state.rust_state.apply_action(rust_action)

        # Determine new phase
        new_street = new_rs.street
        if new_rs.is_terminal:
            new_phase = "terminal"
        elif state.phase == "preflop" and new_street == "preflop" and new_rs.active_player == 3:
            # active_player == 3 means waiting for board deal
            new_phase = "flop_deal"
        elif state.phase == "flop" and new_rs.active_player == 3:
            new_phase = "turn_deal"
        elif state.phase == "turn" and new_rs.active_player == 3:
            new_phase = "river_deal"
        else:
            new_phase = state.phase

        return HUNLState(
            rust_state=new_rs,
            p0_hand=state.p0_hand,
            p1_hand=state.p1_hand,
            board=state.board,
            phase=new_phase,
            action_history=state.action_history + (repr(action),),
        )

    def infoset_key(self, state: HUNLState, player: int) -> str:
        """Information set key: player's hand + public action history.

        Format: 'c0,c1|board|actions'
        """
        hand = state.p0_hand if player == 0 else state.p1_hand
        if hand is None:
            return "?"

        hand_str = f"{pe.py_card_to_string(hand[0])}{pe.py_card_to_string(hand[1])}"
        board_str = "".join(pe.py_card_to_string(c) for c in state.board)
        action_str = ",".join(state.action_history)
        return f"{hand_str}|{board_str}|{action_str}"

    def terminal_utility(self, state: HUNLState, player: int) -> float:
        return state.rust_state.terminal_utility(player)

    def hand_index(self, hand: Tuple[int, int]) -> int:
        """Get the index (0-1325) for a hand in ALL_HOLE_COMBOS."""
        key = tuple(sorted(hand))
        return COMBO_INDEX[key]


def valid_deals_for_board(board: Tuple[int, ...]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Enumerate all valid (p0_hand, p1_hand) deals given a board.

    Excludes any hand that uses a board card. Used for constructing
    PBS at a specific board state.
    """
    board_set = set(board)
    remaining = [c for c in range(52) if c not in board_set]
    valid_combos = list(combinations(remaining, 2))

    deals = []
    for i, h0 in enumerate(valid_combos):
        for h1 in valid_combos[i + 1:]:
            if h0[0] != h1[0] and h0[0] != h1[1] and h0[1] != h1[0] and h0[1] != h1[1]:
                deals.append((h0, h1))
    return deals


def hands_for_board(board: Tuple[int, ...]) -> List[Tuple[int, int]]:
    """All valid hole card combinations given a board."""
    board_set = set(board)
    return [(c0, c1) for c0, c1 in ALL_HOLE_COMBOS if c0 not in board_set and c1 not in board_set]
