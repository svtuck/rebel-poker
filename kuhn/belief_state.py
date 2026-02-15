"""Public Belief State (PBS) tracking for Kuhn Poker using PyTorch.

In ReBeL, a Public Belief State is a tuple (s_pub, Delta(S_1), Delta(S_2))
where Delta(S_i) is a probability distribution over player i's private states,
conditioned on the public history of actions.

The key idea:
  PBS(h) = (h, belief_p0[card], belief_p1[card])
  where belief_pi[card] = P(player i holds card | history h)
                        proportional to sum over opponent cards of
                        reach_i(deal, h) * chance(deal)

This module computes per-player belief states in a vectorized way using
PyTorch tensors. The factored [NUM_PRIVATE_STATES, NUM_PLAYERS] representation
matches the ReBeL paper and is the foundation for value network training.

Tensor layout:
  - deals: [6] possible card deals (i,j) where i!=j, i,j in {0,1,2}
  - PBS: [NUM_PRIVATE_STATES, NUM_PLAYERS] = [3, 2] per-player beliefs
  - For each deal, we track reach probabilities for both players
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from kuhn.game import (
    CARD_RANKS,
    NUM_CARDS,
    RANK_NAMES,
    TERMINAL_HISTORIES,
    KuhnPoker,
)

# All 6 possible deals: (card_p0, card_p1) where cards differ
ALL_DEALS = [(i, j) for i in CARD_RANKS for j in CARD_RANKS if i != j]
NUM_DEALS = len(ALL_DEALS)  # 6
DEAL_INDEX = {deal: idx for idx, deal in enumerate(ALL_DEALS)}

# Per-player private state dimensions
NUM_PRIVATE_STATES = NUM_CARDS  # 3 (J, Q, K)
NUM_PLAYERS = 2

# Map from card to which deal indices contain that card for each player
CARD_TO_DEALS_P0 = {c: [i for i, d in enumerate(ALL_DEALS) if d[0] == c] for c in CARD_RANKS}
CARD_TO_DEALS_P1 = {c: [i for i, d in enumerate(ALL_DEALS) if d[1] == c] for c in CARD_RANKS}


def initial_chance_probs() -> torch.Tensor:
    """Uniform chance probability over all 6 deals. Shape: [6]."""
    return torch.full((NUM_DEALS,), 1.0 / NUM_DEALS)


def initial_pbs(device: str = "cpu") -> torch.Tensor:
    """Initial (uniform) public belief state. Shape: [NUM_PRIVATE_STATES, NUM_PLAYERS].

    At the root, each player is equally likely to hold any card.
    """
    return torch.full(
        (NUM_PRIVATE_STATES, NUM_PLAYERS), 1.0 / NUM_PRIVATE_STATES,
        device=torch.device(device),
    )


def reach_to_pbs(
    reach_p0: torch.Tensor,
    reach_p1: torch.Tensor,
    chance: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Convert per-deal reach probabilities to factored PBS [NUM_PRIVATE_STATES, NUM_PLAYERS].

    Marginalizes joint (reach * chance) over opponent's card to get each
    player's private-state belief.

    Args:
        reach_p0: [NUM_DEALS] reach probs for player 0
        reach_p1: [NUM_DEALS] reach probs for player 1
        chance: [NUM_DEALS] chance probs (default: uniform)
        device: torch device string

    Returns:
        pbs: [NUM_PRIVATE_STATES, NUM_PLAYERS] where pbs[c, p] = P(player p holds card c | history)
    """
    if chance is None:
        chance = initial_chance_probs().to(reach_p0.device)

    joint = chance * reach_p0 * reach_p1
    pbs = torch.zeros(NUM_PRIVATE_STATES, NUM_PLAYERS, device=reach_p0.device)

    for card in CARD_RANKS:
        # P0's belief: marginalize over P1's card
        pbs[card, 0] = sum(joint[i] for i in CARD_TO_DEALS_P0[card])
        # P1's belief: marginalize over P0's card
        pbs[card, 1] = sum(joint[i] for i in CARD_TO_DEALS_P1[card])

    # Normalize each player's column to sum to 1
    for p in range(NUM_PLAYERS):
        total = pbs[:, p].sum()
        if total > 0:
            pbs[:, p] /= total

    return pbs


class BeliefStateTracker:
    """Tracks public belief states across the Kuhn Poker game tree.

    For each public history, maintains:
      - reach_probs[player]: shape [6] -- reach probability for each deal
      - belief: shape [NUM_PRIVATE_STATES, NUM_PLAYERS] -- factored PBS

    The strategy is represented as a dict mapping infoset keys to
    action probability tensors.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.game = KuhnPoker()
        self.device = torch.device(device)
        self.chance_probs = initial_chance_probs().to(self.device)

        # Strategy tables: infoset_key -> tensor of action probs [num_actions]
        # These get populated from a CFR solution
        self._strategy: Dict[str, torch.Tensor] = {}

    def set_strategy_from_profile(
        self, profile: Dict[str, Dict[str, float]]
    ) -> None:
        """Load strategy from a CFR average strategy profile.

        Actions are stored in game order: ["c", "b"] or ["c", "f"].
        """
        self._strategy.clear()
        for key, action_probs in profile.items():
            # Determine action order from the history
            parts = key.split("|")
            history = parts[1] if len(parts) > 1 else ""
            if history in ("", "c"):
                actions = ["c", "b"]
            else:
                actions = ["c", "f"]
            probs = torch.tensor(
                [action_probs.get(a, 1.0 / len(actions)) for a in actions],
                device=self.device,
                dtype=torch.float32,
            )
            self._strategy[key] = probs

    def _get_strategy(self, infoset_key: str, num_actions: int) -> torch.Tensor:
        """Get strategy for an infoset, defaulting to uniform."""
        if infoset_key in self._strategy:
            return self._strategy[infoset_key]
        return torch.full(
            (num_actions,), 1.0 / num_actions, device=self.device
        )

    def compute_all_reach_probs(
        self,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute reach probabilities for all public histories.

        Returns dict: history -> (reach_p0[6], reach_p1[6])
        where reach_pi[d] is the probability that player i plays to
        reach this history given deal d.
        """
        reaches: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Root: both players have reach 1.0 for all deals
        ones = torch.ones(NUM_DEALS, device=self.device)
        reaches[""] = (ones.clone(), ones.clone())

        # BFS through the game tree by history length
        histories_by_depth = [[""], ["c", "b"], ["cc", "bc", "bf", "cb"], ["cbc", "cbf"]]

        for depth_histories in histories_by_depth[1:]:
            for h in depth_histories:
                parent_h = h[:-1]
                action = h[-1]

                if parent_h not in reaches:
                    continue

                parent_reach_p0, parent_reach_p1 = reaches[parent_h]
                player = len(parent_h) % 2

                # Determine legal actions at parent history
                if parent_h in ("", "c"):
                    actions = ["c", "b"]
                elif parent_h in ("b", "cb"):
                    actions = ["c", "f"]
                else:
                    continue

                action_idx = actions.index(action)

                # For the acting player, multiply reach by strategy prob
                # Strategy depends on private card, so we need per-deal probs
                action_probs = torch.ones(NUM_DEALS, device=self.device)

                card_to_deals = CARD_TO_DEALS_P0 if player == 0 else CARD_TO_DEALS_P1
                for card in CARD_RANKS:
                    infoset_key = f"{RANK_NAMES[card]}|{parent_h}"
                    strat = self._get_strategy(infoset_key, len(actions))
                    prob = strat[action_idx]
                    for deal_idx in card_to_deals[card]:
                        action_probs[deal_idx] = prob

                if player == 0:
                    new_reach_p0 = parent_reach_p0 * action_probs
                    new_reach_p1 = parent_reach_p1.clone()
                else:
                    new_reach_p0 = parent_reach_p0.clone()
                    new_reach_p1 = parent_reach_p1 * action_probs

                reaches[h] = (new_reach_p0, new_reach_p1)

        return reaches

    def compute_belief_states(
        self,
    ) -> Dict[str, torch.Tensor]:
        """Compute PBS for all public histories.

        Returns dict: history -> [NUM_PRIVATE_STATES, NUM_PLAYERS] factored belief tensor
        where pbs[card, player] = P(player holds card | history)
        """
        reaches = self.compute_all_reach_probs()
        beliefs: Dict[str, torch.Tensor] = {}

        for h, (reach_p0, reach_p1) in reaches.items():
            beliefs[h] = reach_to_pbs(reach_p0, reach_p1, self.chance_probs, str(self.device))

        return beliefs

    def compute_player_beliefs(
        self, history: str, player: int
    ) -> Dict[int, float]:
        """Compute player's belief about opponent's card at a history.

        This marginalizes the PBS over the player's own card to get
        P(opp_card | my_card, history).

        Returns: {opponent_card: probability} for each possible opponent card.
        """
        beliefs = self.compute_belief_states()
        if history not in beliefs:
            return {}

        pbs = beliefs[history]
        result = {}

        # The opponent is the other player
        opp = 1 - player
        for card in CARD_RANKS:
            result[card] = pbs[card, opp].item()

        return result

    def compute_counterfactual_values(
        self,
    ) -> Dict[str, torch.Tensor]:
        """Compute counterfactual values at terminal nodes for all deals.

        Returns dict: terminal_history -> values[6] (from player 0's perspective)
        """
        values: Dict[str, torch.Tensor] = {}
        game = self.game

        for h in TERMINAL_HISTORIES:
            v = torch.zeros(NUM_DEALS, device=self.device)
            for deal_idx, (c0, c1) in enumerate(ALL_DEALS):
                state = game.next_state(game.initial_state(), (c0, c1))
                for action in h:
                    state = game.next_state(state, action)
                v[deal_idx] = game.terminal_utility(state, 0)
            values[h] = v

        return values
