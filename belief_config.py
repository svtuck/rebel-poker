"""Game-agnostic belief state configuration for ReBeL.

This module defines the BeliefConfig dataclass that captures the game-specific
information needed by the ReBeL solver layer (value net, endgame solver,
trainer, data logger). Each game provides a BeliefConfig describing its
deal structure and private state dimensions.

The ReBeL layer never imports from kuhn.belief_state or leduc.belief_state
directly — it only works through BeliefConfig.

PBS (Public Belief State) layout:
  [num_private_states, num_players] — factored per-player beliefs
  pbs[c, p] = P(player p has private state c | public history)

Deal layout:
  A deal is a tuple of per-player private states, e.g. (card_p0, card_p1).
  There are num_deals possible deals, each with an associated chance probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from game_interface import Game


@dataclass
class BeliefConfig:
    """Configuration describing a game's belief state structure.

    This decouples the ReBeL solver layer from any specific game's
    belief state module (kuhn.belief_state, leduc.belief_state, etc.).

    Attributes:
        num_private_states: Number of possible private states per player
            (e.g., 3 for Kuhn = {J, Q, K}, 6 for Leduc = {Js, Jh, Qs, Qh, Ks, Kh})
        num_players: Number of players (typically 2)
        all_deals: List of (private_state_p0, private_state_p1) tuples
        deal_index: Maps deal tuple -> index in all_deals
        card_to_deals: Per-player mapping from private state -> list of deal indices
            card_to_deals[player][private_state] = [deal_idx, ...]
        chance_probs: Uniform probability over deals [num_deals]
    """
    num_private_states: int
    num_players: int
    all_deals: List[Tuple[int, ...]]
    deal_index: Dict[Tuple[int, ...], int] = field(default_factory=dict)
    card_to_deals: List[Dict[int, List[int]]] = field(default_factory=list)
    chance_probs: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if not self.deal_index:
            self.deal_index = {deal: idx for idx, deal in enumerate(self.all_deals)}

        if not self.card_to_deals:
            self.card_to_deals = []
            for p in range(self.num_players):
                mapping: Dict[int, List[int]] = {}
                for idx, deal in enumerate(self.all_deals):
                    card = deal[p]
                    mapping.setdefault(card, []).append(idx)
                self.card_to_deals.append(mapping)

        if self.chance_probs is None:
            self.chance_probs = torch.full(
                (self.num_deals,), 1.0 / self.num_deals
            )

    @property
    def num_deals(self) -> int:
        return len(self.all_deals)

    @property
    def pbs_dim(self) -> int:
        """Flattened PBS dimension for network I/O."""
        return self.num_private_states * self.num_players

    def initial_pbs(self, device: str = "cpu") -> torch.Tensor:
        """Uniform PBS at game root. Shape: [num_private_states, num_players]."""
        return torch.full(
            (self.num_private_states, self.num_players),
            1.0 / self.num_private_states,
            device=torch.device(device),
        )

    def reach_to_pbs(
        self,
        reach_p0: torch.Tensor,
        reach_p1: torch.Tensor,
        chance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert per-deal reach probabilities to factored PBS.

        Marginalizes joint (reach * chance) over opponent's card to get
        each player's private-state belief.

        Args:
            reach_p0: [num_deals] reach probs for player 0
            reach_p1: [num_deals] reach probs for player 1
            chance: [num_deals] chance probs (default: uniform)

        Returns:
            pbs: [num_private_states, num_players]
        """
        if chance is None:
            chance = self.chance_probs.to(reach_p0.device)

        reaches = [reach_p0, reach_p1]
        joint = chance * reach_p0 * reach_p1
        pbs = torch.zeros(
            self.num_private_states, self.num_players,
            device=reach_p0.device,
        )

        for p in range(self.num_players):
            for card, deal_indices in self.card_to_deals[p].items():
                pbs[card, p] = sum(joint[i] for i in deal_indices)

        # Normalize each player's column
        for p in range(self.num_players):
            total = pbs[:, p].sum()
            if total > 0:
                pbs[:, p] /= total

        return pbs

    def pbs_to_reach(
        self, pbs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert factored PBS to per-deal reach probabilities.

        Args:
            pbs: [num_private_states, num_players]

        Returns:
            reach_p0: [num_deals]
            reach_p1: [num_deals]
        """
        reach_p0 = torch.zeros(self.num_deals, device=pbs.device)
        reach_p1 = torch.zeros(self.num_deals, device=pbs.device)
        for deal_idx, deal in enumerate(self.all_deals):
            reach_p0[deal_idx] = pbs[deal[0], 0]
            reach_p1[deal_idx] = pbs[deal[1], 1]
        return reach_p0, reach_p1

    def deal_values_to_pbs_values(
        self, deal_values: torch.Tensor,
    ) -> torch.Tensor:
        """Convert per-deal values [num_deals] to PBS-space values.

        Returns [num_private_states, num_players] where:
          values[card, 0] = mean of deal_values where P0 holds card
          values[card, 1] = mean of -deal_values where P1 holds card
            (negated because deal_values are from P0's perspective)
        """
        pbs_values = torch.zeros(
            self.num_private_states, self.num_players,
            device=deal_values.device,
        )

        for card, deal_indices in self.card_to_deals[0].items():
            if deal_indices:
                pbs_values[card, 0] = sum(
                    deal_values[i] for i in deal_indices
                ) / len(deal_indices)

        for card, deal_indices in self.card_to_deals[1].items():
            if deal_indices:
                pbs_values[card, 1] = sum(
                    -deal_values[i] for i in deal_indices
                ) / len(deal_indices)

        return pbs_values

    def pbs_values_to_deal_values(
        self, pbs_values: torch.Tensor,
    ) -> torch.Tensor:
        """Convert PBS-space values to per-deal values (P0's perspective).

        Args:
            pbs_values: [num_private_states, num_players]

        Returns:
            deal_values: [num_deals] from P0's perspective
        """
        deal_values = torch.zeros(self.num_deals, device=pbs_values.device)
        for deal_idx, deal in enumerate(self.all_deals):
            deal_values[deal_idx] = pbs_values[deal[0], 0]
        return deal_values


def belief_config_from_game(game: Game) -> BeliefConfig:
    """Create a BeliefConfig by inspecting a game's chance outcomes.

    This works for any 2-player game where the initial chance node
    deals a tuple of private states (one per player).
    """
    initial = game.initial_state()
    outcomes = game.chance_outcomes(initial)

    # Extract deals and probabilities
    all_deals = []
    probs = []
    for deal, prob in outcomes:
        if isinstance(deal, (list, tuple)):
            all_deals.append(tuple(deal))
        else:
            all_deals.append((deal,))
        probs.append(prob)

    # Determine number of private states per player
    num_players = game.NUM_PLAYERS
    private_states_per_player: List[set] = [set() for _ in range(num_players)]
    for deal in all_deals:
        for p in range(num_players):
            private_states_per_player[p].add(deal[p])

    # All players should have the same number of private states
    num_private_states = max(max(s) + 1 for s in private_states_per_player)

    chance_probs = torch.tensor(probs, dtype=torch.float32)

    return BeliefConfig(
        num_private_states=num_private_states,
        num_players=num_players,
        all_deals=all_deals,
        chance_probs=chance_probs,
    )
