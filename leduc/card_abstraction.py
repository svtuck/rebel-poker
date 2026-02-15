"""Card abstraction for Leduc Poker using equity buckets.

In Leduc with 3 ranks x 2 suits and a community card, the natural
abstractions are:

Preflop (before board):
  - 3 buckets: J, Q, K (rank only; suits are symmetric)
  - Each rank has equal equity before the board

Postflop (after board):
  - 5 buckets based on hand strength relative to board:
    0: Low non-pair (rank < board rank, no pair)
    1: Mid non-pair (rank > board rank but not top, no pair)
    2: High non-pair (top rank, no pair)
    3: Low pair (paired with board, rank < other possible pair ranks)
    4: High pair (paired with board, highest possible)

  For Leduc's 3 ranks, the actual postflop buckets depend on
  the board card:

  Board = J(0):
    J -> pair (bucket 3 - low pair since J is lowest)
    Q -> high card Q (bucket 1)
    K -> high card K (bucket 2)

  Board = Q(1):
    J -> low non-pair (bucket 0)
    Q -> pair (bucket 3)
    K -> high non-pair (bucket 2)

  Board = K(2):
    J -> low non-pair (bucket 0)
    Q -> mid non-pair (bucket 1)
    K -> pair (bucket 4 - high pair since K is highest)

This abstraction captures the strategic essence: pairs beat non-pairs,
higher ranks beat lower, and pair rank matters among pairs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from leduc.game import RANKS, RANK_TO_STR, LeducState

# Number of preflop buckets (one per rank)
NUM_PREFLOP_BUCKETS = 3

# Number of postflop buckets
NUM_POSTFLOP_BUCKETS = 5


def preflop_bucket(card_rank: int) -> int:
    """Assign preflop equity bucket based on card rank.

    Returns 0 (J=weakest) to 2 (K=strongest).
    """
    return card_rank


def postflop_bucket(card_rank: int, board_rank: int) -> int:
    """Assign postflop equity bucket based on hand-board relationship.

    Returns:
        0: Low non-pair (weakest non-pair)
        1: Mid non-pair
        2: High non-pair (best non-pair)
        3: Low/mid pair
        4: High pair (best possible hand)
    """
    if card_rank == board_rank:
        # Pair with board
        if card_rank == max(RANKS):
            return 4  # High pair (K paired)
        return 3  # Low/mid pair (J or Q paired)
    else:
        # Non-pair: rank relative to other non-pairs
        # Count how many other ranks beat us (excluding board rank)
        non_board_ranks = [r for r in RANKS if r != board_rank]
        my_position = sorted(non_board_ranks).index(card_rank)
        # my_position: 0 = weakest non-pair, 1 = strongest non-pair
        if my_position == 0:
            return 0  # Low non-pair
        return 2  # High non-pair


def equity_against_uniform(card_rank: int, board_rank: Optional[int] = None) -> float:
    """Compute equity of a hand against a uniform opponent range.

    If board_rank is None, computes preflop equity.
    If board_rank is given, computes postflop equity.

    Returns win probability (0.0 to 1.0, with 0.5 for ties).
    """
    if board_rank is None:
        # Preflop: against uniform opponent range
        # P(win) = P(opponent has lower rank) + 0.5 * P(opponent has same rank)
        # With 6 cards, 2 per rank:
        # If I have rank r, opponent draws from 5 remaining cards:
        # - 2 cards of each other rank, 1 card of my rank
        wins = 0.0
        ties = 0.0
        total = 0.0
        for opp_rank in RANKS:
            # Number of cards opponent can have with this rank
            if opp_rank == card_rank:
                count = 1  # One other card of my rank
            else:
                count = 2  # Both cards of different rank
            total += count
            if card_rank > opp_rank:
                wins += count
            elif card_rank == opp_rank:
                ties += count
        return (wins + 0.5 * ties) / total
    else:
        # Postflop: against uniform opponent range, given board
        my_pair = (card_rank == board_rank)
        wins = 0.0
        ties = 0.0
        total = 0.0
        for opp_rank in RANKS:
            if opp_rank == card_rank:
                count = 1
            else:
                count = 2
            # But exclude the board card from opponent's possible cards
            # Board card is (board_rank, some_suit). Each rank has 2 suits.
            # Opponent can't hold the board card, so if opp_rank == board_rank,
            # only 1 card is available (the other suit).
            if opp_rank == board_rank:
                count = min(count, 1)

            total += count
            opp_pair = (opp_rank == board_rank)

            if my_pair and not opp_pair:
                wins += count
            elif not my_pair and opp_pair:
                pass  # opp wins
            elif my_pair and opp_pair:
                ties += count  # Same rank paired
            else:
                # Both non-pair: compare ranks
                if card_rank > opp_rank:
                    wins += count
                elif card_rank == opp_rank:
                    ties += count

        return (wins + 0.5 * ties) / total if total > 0 else 0.5


def hand_strength_vector(board_rank: Optional[int] = None) -> Dict[int, float]:
    """Compute equity for all possible hand ranks against uniform range.

    Returns {rank: equity} for each rank.
    """
    return {r: equity_against_uniform(r, board_rank) for r in RANKS}
