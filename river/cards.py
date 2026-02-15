"""Card representation and hand evaluation for Texas Hold'em.

Cards are encoded as integers 0-51: card_id = suit * 13 + rank
Ranks: 2-A (indices 0-12), Suits: c, d, h, s (indices 0-3)

Hand evaluation returns tuples for comparison:
  (hand_type, *tiebreakers)
where hand_type is 0 (high card) through 8 (straight flush).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card_id(card: str) -> int:
    rank = RANKS.index(card[0])
    suit = SUITS.index(card[1])
    return suit * 13 + rank


def card_str(card: int) -> str:
    rank = card % 13
    suit = card // 13
    return f"{RANKS[rank]}{SUITS[suit]}"


def parse_cards(cards: Sequence[str]) -> List[int]:
    return [card_id(card) for card in cards]


def parse_hand(hand: str) -> Tuple[int, int]:
    if len(hand) != 4:
        raise ValueError(f"Hand must be 4 chars like AsKd, got {hand}")
    c1 = card_id(hand[:2])
    c2 = card_id(hand[2:])
    if c1 == c2:
        raise ValueError(f"Hand has duplicate card: {hand}")
    return tuple(sorted((c1, c2)))


def all_hole_cards(exclude: Iterable[int]) -> List[Tuple[int, int]]:
    exclude_set = set(exclude)
    deck = [c for c in range(52) if c not in exclude_set]
    return [tuple(sorted((a, b))) for a, b in combinations(deck, 2)]


def evaluate_5(cards: Sequence[int]) -> Tuple[int, ...]:
    ranks = [card % 13 + 2 for card in cards]
    suits = [card // 13 for card in cards]
    ranks_sorted = sorted(ranks, reverse=True)
    counts: Dict[int, int] = {}
    for rank in ranks:
        counts[rank] = counts.get(rank, 0) + 1
    count_items = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    counts_sorted = [count for _, count in count_items]
    ranks_by_count = [rank for rank, _ in count_items]

    is_flush = len(set(suits)) == 1
    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0
    if len(unique_ranks) == 5 and unique_ranks[0] - unique_ranks[-1] == 4:
        is_straight = True
        straight_high = unique_ranks[0]
    elif unique_ranks == [14, 5, 4, 3, 2]:
        is_straight = True
        straight_high = 5

    if is_straight and is_flush:
        return (8, straight_high)
    if counts_sorted == [4, 1]:
        return (7, ranks_by_count[0], ranks_by_count[1])
    if counts_sorted == [3, 2]:
        return (6, ranks_by_count[0], ranks_by_count[1])
    if is_flush:
        return (5, *ranks_sorted)
    if is_straight:
        return (4, straight_high)
    if counts_sorted == [3, 1, 1]:
        kickers = sorted(ranks_by_count[1:], reverse=True)
        return (3, ranks_by_count[0], *kickers)
    if counts_sorted == [2, 2, 1]:
        high_pair, low_pair = sorted(ranks_by_count[:2], reverse=True)
        return (2, high_pair, low_pair, ranks_by_count[2])
    if counts_sorted == [2, 1, 1, 1]:
        kickers = sorted(ranks_by_count[1:], reverse=True)
        return (1, ranks_by_count[0], *kickers)
    return (0, *ranks_sorted)


def evaluate_7(cards: Sequence[int]) -> Tuple[int, ...]:
    best: Tuple[int, ...] | None = None
    for combo in combinations(cards, 5):
        rank = evaluate_5(combo)
        if best is None or rank > best:
            best = rank
    if best is None:
        raise ValueError("No cards to evaluate")
    return best
