"""Public Belief State tracking for HUNL Hold'em.

Implements factored per-player beliefs over 1326 hole card combinations
as described in the ReBeL paper. Each player's belief is a probability
distribution over the 1326 possible 2-card hands, conditioned on:

1. The public action history (Bayesian update after each opponent action)
2. Board cards (zero out impossible hands containing board cards)

Key differences from Kuhn/Leduc PBS:
- num_private_states = 1326 (vs 3 for Kuhn, 6 for Leduc)
- Board cards cause hard constraints (hands containing board cards get prob 0)
- Beliefs must be updated incrementally as the game progresses
  (can't enumerate all deals like in small games)

PBS layout: [1326, 2] where pbs[h, p] = P(player p holds hand h | public info)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from hunl.game import ALL_HOLE_COMBOS, COMBO_INDEX, NUM_COMBOS, hands_for_board


class HUNLBeliefState:
    """Tracks factored per-player beliefs over 1326 hole card combinations.

    The belief state maintains a probability distribution for each player
    over all possible hole card combinations, updating after:
    - Board cards are revealed (zero out impossible hands)
    - Opponent takes an action (Bayesian update using strategy)

    Attributes:
        beliefs: [1326, 2] tensor of per-player beliefs
        board: current community cards
        valid_mask: [1326] boolean mask of hands not blocked by board
    """

    def __init__(
        self,
        board: Tuple[int, ...] = (),
        device: str = "cpu",
    ) -> None:
        self.board = board
        self.device = torch.device(device)

        # Initialize uniform beliefs over valid hands
        self.valid_mask = self._compute_valid_mask(board)
        num_valid = self.valid_mask.sum().item()

        self.beliefs = torch.zeros(NUM_COMBOS, 2, device=self.device)
        if num_valid > 0:
            for p in range(2):
                self.beliefs[:, p] = self.valid_mask.float() / num_valid

    def _compute_valid_mask(self, board: Tuple[int, ...]) -> torch.Tensor:
        """Compute which hands are possible given the board cards."""
        board_set = set(board)
        mask = torch.zeros(NUM_COMBOS, dtype=torch.bool, device=self.device)
        for i, (c0, c1) in enumerate(ALL_HOLE_COMBOS):
            if c0 not in board_set and c1 not in board_set:
                mask[i] = True
        return mask

    def update_board(self, new_board: Tuple[int, ...]) -> None:
        """Update beliefs after new community cards are revealed.

        Zeros out any hands that contain the new board cards, then
        renormalizes each player's belief distribution.
        """
        self.board = new_board
        self.valid_mask = self._compute_valid_mask(new_board)

        # Zero out impossible hands
        invalid = ~self.valid_mask
        self.beliefs[invalid, :] = 0.0

        # Renormalize
        for p in range(2):
            total = self.beliefs[:, p].sum()
            if total > 0:
                self.beliefs[:, p] /= total

    def bayesian_update(
        self,
        player: int,
        action_probs: torch.Tensor,
    ) -> None:
        """Bayesian update of opponent's beliefs after they take an action.

        When player `player` takes an action, their belief distribution
        is updated by multiplying by the probability of taking that action
        with each possible hand, then renormalizing.

        Args:
            player: which player took the action (0 or 1)
            action_probs: [1326] probability of the observed action for each hand
        """
        self.beliefs[:, player] *= action_probs

        # Renormalize
        total = self.beliefs[:, player].sum()
        if total > 0:
            self.beliefs[:, player] /= total

    def get_pbs(self) -> torch.Tensor:
        """Return current PBS as [1326, 2] tensor."""
        return self.beliefs.clone()

    def get_pbs_flat(self) -> torch.Tensor:
        """Return flattened PBS as [2652] tensor for network input."""
        return self.beliefs.flatten()

    @staticmethod
    def hand_conflict(h0: int, h1: int) -> bool:
        """Check if two hands share any cards."""
        c0 = ALL_HOLE_COMBOS[h0]
        c1 = ALL_HOLE_COMBOS[h1]
        return (c0[0] == c1[0] or c0[0] == c1[1] or
                c0[1] == c1[0] or c0[1] == c1[1])

    @staticmethod
    def build_conflict_mask() -> torch.Tensor:
        """Build [1326, 1326] boolean mask where True = hands share cards.

        This is used to enforce card removal: if we know player 0 holds
        hand h0, player 1 cannot hold any hand that shares a card with h0.
        """
        mask = torch.zeros(NUM_COMBOS, NUM_COMBOS, dtype=torch.bool)
        for i, (c0a, c0b) in enumerate(ALL_HOLE_COMBOS):
            for j, (c1a, c1b) in enumerate(ALL_HOLE_COMBOS):
                if c0a == c1a or c0a == c1b or c0b == c1a or c0b == c1b:
                    mask[i, j] = True
        return mask


def build_hunl_belief_config(board: Tuple[int, ...] = ()) -> Dict:
    """Build a belief configuration dict for HUNL at a given board state.

    This is used to create a BeliefConfig-compatible structure for the
    ReBeL training loop, adapted for HUNL's scale.

    Args:
        board: community cards already dealt

    Returns:
        Dict with keys matching BeliefConfig fields
    """
    board_set = set(board)
    valid_hands = [
        (i, (c0, c1))
        for i, (c0, c1) in enumerate(ALL_HOLE_COMBOS)
        if c0 not in board_set and c1 not in board_set
    ]

    # Build deals: all valid (hand_p0_idx, hand_p1_idx) pairs where hands don't conflict
    all_deals = []
    for i_idx, (i, h0) in enumerate(valid_hands):
        for j, h1 in valid_hands[i_idx + 1:]:
            if (h0[0] != h1[0] and h0[0] != h1[1] and
                    h0[1] != h1[0] and h0[1] != h1[1]):
                all_deals.append((i, j))

    return {
        "num_private_states": NUM_COMBOS,
        "num_players": 2,
        "num_valid_hands": len(valid_hands),
        "num_deals": len(all_deals),
        "valid_hand_indices": [i for i, _ in valid_hands],
        "board": board,
    }


def pbs_from_reach_probs(
    reach_p0: torch.Tensor,
    reach_p1: torch.Tensor,
    board: Tuple[int, ...] = (),
) -> torch.Tensor:
    """Convert per-hand reach probabilities to factored PBS.

    For HUNL, we don't track joint probabilities over all ~1.75M deals.
    Instead, we marginalize directly: each player's belief for hand h
    is proportional to their reach probability for that hand, weighted
    by the opponent's total reach for non-conflicting hands.

    Args:
        reach_p0: [1326] P0's reach probability per hand
        reach_p1: [1326] P1's reach probability per hand
        board: community cards for valid-hand masking

    Returns:
        pbs: [1326, 2] factored beliefs
    """
    board_set = set(board)
    pbs = torch.zeros(NUM_COMBOS, 2, device=reach_p0.device)

    # For player p, belief[h, p] ∝ reach_p[h] × Σ_{h' not conflicting with h} reach_{-p}[h']
    # This is the correct marginalization of the joint distribution.
    for h_idx, (c0, c1) in enumerate(ALL_HOLE_COMBOS):
        if c0 in board_set or c1 in board_set:
            continue

        # P0's belief: weighted by P1's reach over non-conflicting hands
        opp_weight = 0.0
        for h2_idx, (c2a, c2b) in enumerate(ALL_HOLE_COMBOS):
            if c2a in board_set or c2b in board_set:
                continue
            if c0 == c2a or c0 == c2b or c1 == c2a or c1 == c2b:
                continue
            opp_weight += reach_p1[h2_idx].item()
        pbs[h_idx, 0] = reach_p0[h_idx] * opp_weight

        # P1's belief: weighted by P0's reach over non-conflicting hands
        opp_weight = 0.0
        for h2_idx, (c2a, c2b) in enumerate(ALL_HOLE_COMBOS):
            if c2a in board_set or c2b in board_set:
                continue
            if c0 == c2a or c0 == c2b or c1 == c2a or c1 == c2b:
                continue
            opp_weight += reach_p0[h2_idx].item()
        pbs[h_idx, 1] = reach_p1[h_idx] * opp_weight

    # Normalize
    for p in range(2):
        total = pbs[:, p].sum()
        if total > 0:
            pbs[:, p] /= total

    return pbs


def pbs_from_reach_probs_fast(
    reach_p0: torch.Tensor,
    reach_p1: torch.Tensor,
    board: Tuple[int, ...] = (),
    conflict_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Vectorized version of pbs_from_reach_probs using the conflict matrix.

    Uses a precomputed [1326, 1326] conflict mask to avoid nested loops.
    Much faster for repeated calls during CFR.

    Args:
        reach_p0: [1326] P0's reach probability per hand
        reach_p1: [1326] P1's reach probability per hand
        board: community cards
        conflict_mask: [1326, 1326] precomputed conflict matrix (True = share cards)

    Returns:
        pbs: [1326, 2] factored beliefs
    """
    if conflict_mask is None:
        conflict_mask = HUNLBeliefState.build_conflict_mask()
    conflict_mask = conflict_mask.to(reach_p0.device)

    # Valid hand mask
    board_set = set(board)
    valid = torch.ones(NUM_COMBOS, dtype=torch.bool, device=reach_p0.device)
    for h_idx, (c0, c1) in enumerate(ALL_HOLE_COMBOS):
        if c0 in board_set or c1 in board_set:
            valid[h_idx] = False

    # Non-conflicting mask: ~conflict_mask AND valid for opponent
    # For each hand h, sum opponent reach over non-conflicting valid hands
    non_conflict = ~conflict_mask  # [1326, 1326]

    # Mask invalid hands
    valid_f = valid.float()
    reach_p0_masked = reach_p0 * valid_f
    reach_p1_masked = reach_p1 * valid_f

    # opp_weight_for_p0[h] = sum of reach_p1 over hands not conflicting with h and valid
    # = (non_conflict[h, :] AND valid) dot reach_p1
    valid_non_conflict = non_conflict & valid.unsqueeze(0)  # [1326, 1326]
    opp_weight_p0 = (valid_non_conflict.float() @ reach_p1_masked.unsqueeze(-1)).squeeze(-1)
    opp_weight_p1 = (valid_non_conflict.float() @ reach_p0_masked.unsqueeze(-1)).squeeze(-1)

    pbs = torch.zeros(NUM_COMBOS, 2, device=reach_p0.device)
    pbs[:, 0] = reach_p0_masked * opp_weight_p0
    pbs[:, 1] = reach_p1_masked * opp_weight_p1

    # Normalize
    for p in range(2):
        total = pbs[:, p].sum()
        if total > 0:
            pbs[:, p] /= total

    return pbs
