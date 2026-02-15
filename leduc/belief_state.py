"""Public Belief State (PBS) tracking for Leduc Poker using PyTorch.

Leduc has a richer chance structure than Kuhn:
- First chance: deal 2 private cards -> 30 outcomes (6 x 5)
- Second chance: deal 1 board card -> 4 outcomes per deal
- Total "full deals": 30 x 4 = 120 configurations

Private states: 3 ranks (J, Q, K). The PBS is [3, 2] -- same shape as Kuhn,
but the semantics change when a board card is present.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from leduc.game import (
    DECK,
    NUM_RANKS,
    RANKS,
    RANK_TO_STR,
    LeducPoker,
)

# Enumerate all possible private card deals: (card_p0, card_p1)
ALL_PRIVATE_DEALS = [
    (c0, c1) for i, c0 in enumerate(DECK) for j, c1 in enumerate(DECK) if i != j
]
NUM_PRIVATE_DEALS = len(ALL_PRIVATE_DEALS)  # 30
PRIVATE_DEAL_INDEX = {deal: idx for idx, deal in enumerate(ALL_PRIVATE_DEALS)}


def _board_cards_for_deal(deal):
    return [c for c in DECK if c != deal[0] and c != deal[1]]


# "Full deals": (card_p0, card_p1, board_card) -- all 120 configurations
ALL_FULL_DEALS = []
for _deal in ALL_PRIVATE_DEALS:
    for _board in _board_cards_for_deal(_deal):
        ALL_FULL_DEALS.append((_deal[0], _deal[1], _board))
NUM_FULL_DEALS = len(ALL_FULL_DEALS)  # 120
FULL_DEAL_INDEX = {d: idx for idx, d in enumerate(ALL_FULL_DEALS)}

# Map from private deal index -> list of full deal indices
PRIVATE_TO_FULL: Dict[int, List[int]] = {}
for _fd_idx, (_c0, _c1, _board) in enumerate(ALL_FULL_DEALS):
    _pd_idx = PRIVATE_DEAL_INDEX[(_c0, _c1)]
    PRIVATE_TO_FULL.setdefault(_pd_idx, []).append(_fd_idx)

# Map from full deal index -> private deal index
FULL_TO_PRIVATE: Dict[int, int] = {}
for _fd_idx, (_c0, _c1, _board) in enumerate(ALL_FULL_DEALS):
    FULL_TO_PRIVATE[_fd_idx] = PRIVATE_DEAL_INDEX[(_c0, _c1)]

NUM_PRIVATE_STATES = NUM_RANKS  # 3
NUM_PLAYERS = 2

# Map from rank to private deal indices where that player has that rank
RANK_TO_DEALS_P0 = {
    r: [i for i, d in enumerate(ALL_PRIVATE_DEALS) if d[0][0] == r] for r in RANKS
}
RANK_TO_DEALS_P1 = {
    r: [i for i, d in enumerate(ALL_PRIVATE_DEALS) if d[1][0] == r] for r in RANKS
}

# Map from rank to full deal indices where that player has that rank
RANK_TO_FULL_DEALS_P0 = {
    r: [i for i, (c0, c1, b) in enumerate(ALL_FULL_DEALS) if c0[0] == r] for r in RANKS
}
RANK_TO_FULL_DEALS_P1 = {
    r: [i for i, (c0, c1, b) in enumerate(ALL_FULL_DEALS) if c1[0] == r] for r in RANKS
}


def initial_chance_probs_private() -> torch.Tensor:
    """Uniform chance probability over the 30 private deals. Shape: [30]."""
    return torch.full((NUM_PRIVATE_DEALS,), 1.0 / NUM_PRIVATE_DEALS)


def initial_chance_probs_full() -> torch.Tensor:
    """Uniform chance probability over all 120 full deals. Shape: [120]."""
    return torch.full((NUM_FULL_DEALS,), 1.0 / NUM_FULL_DEALS)


def initial_pbs(device: str = "cpu") -> torch.Tensor:
    """Initial (uniform) public belief state. Shape: [3, 2]."""
    return torch.full(
        (NUM_PRIVATE_STATES, NUM_PLAYERS), 1.0 / NUM_PRIVATE_STATES,
        device=torch.device(device),
    )


def reach_to_pbs_private(
    reach_p0: torch.Tensor,
    reach_p1: torch.Tensor,
    chance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert per-private-deal reach [30] to factored PBS [3, 2]."""
    if chance is None:
        chance = initial_chance_probs_private().to(reach_p0.device)

    joint = chance * reach_p0 * reach_p1
    pbs = torch.zeros(NUM_PRIVATE_STATES, NUM_PLAYERS, device=reach_p0.device)

    for rank in RANKS:
        pbs[rank, 0] = sum(joint[i] for i in RANK_TO_DEALS_P0[rank])
        pbs[rank, 1] = sum(joint[i] for i in RANK_TO_DEALS_P1[rank])

    for p in range(NUM_PLAYERS):
        total = pbs[:, p].sum()
        if total > 0:
            pbs[:, p] /= total

    return pbs


def reach_to_pbs_full(
    reach_p0: torch.Tensor,
    reach_p1: torch.Tensor,
    chance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert per-full-deal reach [120] to factored PBS [3, 2]."""
    if chance is None:
        chance = initial_chance_probs_full().to(reach_p0.device)

    joint = chance * reach_p0 * reach_p1
    pbs = torch.zeros(NUM_PRIVATE_STATES, NUM_PLAYERS, device=reach_p0.device)

    for rank in RANKS:
        pbs[rank, 0] = sum(joint[i] for i in RANK_TO_FULL_DEALS_P0[rank])
        pbs[rank, 1] = sum(joint[i] for i in RANK_TO_FULL_DEALS_P1[rank])

    for p in range(NUM_PLAYERS):
        total = pbs[:, p].sum()
        if total > 0:
            pbs[:, p] /= total

    return pbs
