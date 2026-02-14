"""Public Belief State (PBS) tracking for Kuhn Poker using PyTorch.

In ReBeL, a Public Belief State is a probability distribution over the
joint private information (card deals), conditioned on the public history
of actions. The key idea is:

  PBS(h) = P(cards | history h) ∝ reach_0(cards, h) * reach_1(cards, h) * chance(cards)

where reach_i is player i's probability of playing to reach history h,
given that they hold card cards[i].

This module computes belief states in a vectorized way using PyTorch tensors,
operating over ALL card deals simultaneously. This is the foundation for
batch CFR and eventual neural network value/policy training.

Tensor layout:
  - deals: [6] possible card deals (i,j) where i!=j, i,j in {0,1,2}
  - For each deal, we track reach probabilities for both players
  - PBS at any public history = normalized product of reaches x chance probs
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

# Map from card to which deal indices contain that card for each player
CARD_TO_DEALS_P0 = {c: [i for i, d in enumerate(ALL_DEALS) if d[0] == c] for c in CARD_RANKS}
CARD_TO_DEALS_P1 = {c: [i for i, d in enumerate(ALL_DEALS) if d[1] == c] for c in CARD_RANKS}


def initial_chance_probs() -> torch.Tensor:
    """Uniform chance probability over all 6 deals. Shape: [6]."""
    return torch.full((NUM_DEALS,), 1.0 / NUM_DEALS)


class BeliefStateTracker:
    """Tracks public belief states across the Kuhn Poker game tree.

    For each public history, maintains:
      - reach_probs[player]: shape [6] -- reach probability for each deal
      - belief: shape [6] -- normalized PBS

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

        PBS(h)[d] = P(deal=d | history=h) ~ chance(d) * reach_0(d,h) * reach_1(d,h)

        Returns dict: history -> normalized belief tensor [6]
        """
        reaches = self.compute_all_reach_probs()
        beliefs: Dict[str, torch.Tensor] = {}

        for h, (reach_p0, reach_p1) in reaches.items():
            joint = self.chance_probs * reach_p0 * reach_p1
            total = joint.sum()
            if total > 0:
                beliefs[h] = joint / total
            else:
                beliefs[h] = torch.zeros(NUM_DEALS, device=self.device)

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

        for opp_card in CARD_RANKS:
            if player == 0:
                # Sum over deals where opponent has this card
                indices = CARD_TO_DEALS_P1[opp_card]
            else:
                indices = CARD_TO_DEALS_P0[opp_card]

            prob = sum(pbs[i].item() for i in indices)
            result[opp_card] = prob

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
