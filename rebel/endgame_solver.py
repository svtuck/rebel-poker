"""Poker endgame solver scaffold for ReBeL integration.

This module provides the framework for solving poker endgames (subgames)
using CFR, following the pattern from Noam Brown's poker_solver repo.

In ReBeL, the endgame solver:
1. Takes a Public Belief State at a given node
2. Runs CFR on the subgame rooted at that node
3. Uses a value network V(PBS) at leaf nodes instead of full rollouts
4. Returns the solved strategy for the subgame

The solver is game-agnostic: it uses the Game protocol to traverse the
game tree, enumerate chance outcomes, and compute utilities. Any game
implementing the Game protocol can use this solver.

References:
  - noambrown/poker_solver: River NLHE subgame solver
  - ReBeL paper (Brown & Sandholm, 2020): Section 3.2, Subgame solving
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from game_interface import Game


@dataclass
class SubgameConfig:
    """Configuration for a subgame solve."""
    iterations: int = 1000
    use_cfr_plus: bool = False
    discount_factor: float = 1.0


class SubgameSolver:
    """Solves a poker subgame from a given PBS using CFR.

    This is the core of ReBeL's search procedure. Given:
    - A game implementing the Game protocol
    - A public belief state (probability over deals)
    - A root history in the game tree
    - Optional: a value function for leaf evaluation

    It runs CFR on the subgame and returns the solved strategy.

    The solver discovers game structure (legal actions, terminal states,
    information sets) dynamically through the Game interface.
    """

    def __init__(
        self,
        game: Game,
        root_history: str,
        initial_beliefs: torch.Tensor,
        value_fn: Optional[Callable[[torch.Tensor, str], torch.Tensor]] = None,
        config: Optional[SubgameConfig] = None,
    ) -> None:
        self.game = game
        self.root_history = root_history
        self.initial_beliefs = initial_beliefs
        self.value_fn = value_fn
        self.config = config or SubgameConfig()
        self.num_deals = len(initial_beliefs)

        # Enumerate chance outcomes to build deal -> state mapping
        initial = game.initial_state()
        outcomes = game.chance_outcomes(initial)
        self.all_deals = [deal for deal, _ in outcomes]
        self._deal_to_idx = {deal: i for i, deal in enumerate(self.all_deals)}

        # Cache infoset -> actions mapping
        self._infoset_actions: Dict[str, List] = {}

        # Build infoset -> deal indices mapping
        self._infoset_deal_indices: Dict[str, List[int]] = {}
        for deal_idx, deal in enumerate(self.all_deals):
            state = game.next_state(initial, deal)
            self._build_infoset_map(state, deal_idx)

        # Regret and strategy tables for the subgame
        self.regret_sum: Dict[str, torch.Tensor] = {}
        self.strategy_sum: Dict[str, torch.Tensor] = {}

    def _build_infoset_map(self, state, deal_idx: int) -> None:
        """Recursively traverse to build infoset -> deal mapping."""
        game = self.game
        if game.is_terminal(state):
            return
        player = game.current_player(state)
        if player == -1:
            for outcome, _ in game.chance_outcomes(state):
                self._build_infoset_map(game.next_state(state, outcome), deal_idx)
            return
        key = game.infoset_key(state, player)
        self._infoset_deal_indices.setdefault(key, [])
        if deal_idx not in self._infoset_deal_indices[key]:
            self._infoset_deal_indices[key].append(deal_idx)
        if key not in self._infoset_actions:
            self._infoset_actions[key] = game.legal_actions(state)
        for action in game.legal_actions(state):
            self._build_infoset_map(game.next_state(state, action), deal_idx)

    def _get_strategy(self, key: str, num_actions: int) -> torch.Tensor:
        if key not in self.regret_sum:
            self.regret_sum[key] = torch.zeros(num_actions)
            self.strategy_sum[key] = torch.zeros(num_actions)

        positives = torch.clamp(self.regret_sum[key], min=0)
        total = positives.sum()
        if total > 0:
            return positives / total
        return torch.full((num_actions,), 1.0 / num_actions)

    def solve(self) -> Dict[str, Dict[str, float]]:
        """Run CFR on the subgame and return the average strategy.

        Returns: {infoset_key: {action: probability}}
        """
        game = self.game
        initial = game.initial_state()

        for iteration in range(self.config.iterations):
            reach_p0 = self.initial_beliefs.clone()
            reach_p1 = self.initial_beliefs.clone()

            # Build per-deal states at the root
            deal_states = [game.next_state(initial, deal) for deal in self.all_deals]
            self._cfr(deal_states, reach_p0, reach_p1)

        return self._extract_profile()

    def _cfr(
        self, states: List, reach_p0: torch.Tensor, reach_p1: torch.Tensor
    ) -> torch.Tensor:
        """CFR traversal for the subgame. Returns [num_deals] values."""
        game = self.game

        # Check if all states are terminal (they should all have same structure)
        if game.is_terminal(states[0]):
            v = torch.zeros(self.num_deals)
            for deal_idx, state in enumerate(states):
                v[deal_idx] = game.terminal_utility(state, 0)
            return v

        # Check if we should use value function for leaf evaluation
        if self.value_fn is not None and self._is_leaf(states):
            belief = self._compute_belief(reach_p0, reach_p1)
            # Use first state's history info for context
            return self.value_fn(belief, "")

        player = game.current_player(states[0])
        actions = game.legal_actions(states[0])
        num_actions = len(actions)

        action_values = []
        strategy_per_deal = []

        # Group deals by infoset for this node
        infoset_deals: Dict[str, List[int]] = {}
        for d_idx, state in enumerate(states):
            key = game.infoset_key(state, player)
            infoset_deals.setdefault(key, []).append(d_idx)

        for a_idx, action in enumerate(actions):
            s = torch.zeros(self.num_deals)
            for key, deal_indices in infoset_deals.items():
                strat = self._get_strategy(key, num_actions)
                for d_idx in deal_indices:
                    s[d_idx] = strat[a_idx]
            strategy_per_deal.append(s)

            child_states = [game.next_state(state, action) for state in states]
            if player == 0:
                child_values = self._cfr(child_states, reach_p0 * s, reach_p1)
            else:
                child_values = self._cfr(child_states, reach_p0, reach_p1 * s)
            action_values.append(child_values)

        action_values_t = torch.stack(action_values)
        strategy_t = torch.stack(strategy_per_deal)
        node_values = (strategy_t * action_values_t).sum(dim=0)

        # Update regrets
        opponent_reach = reach_p1 if player == 0 else reach_p0
        player_reach = reach_p0 if player == 0 else reach_p1

        for key, deal_indices in infoset_deals.items():
            strat = self._get_strategy(key, num_actions)

            for a_idx in range(num_actions):
                # Correct CFR regret: weighted sum of per-deal regrets
                # regret(I, a) = sum_{h in I} opp_reach(h) * (v(h,a) - v(h))
                regret = 0.0
                for i in deal_indices:
                    if player == 0:
                        regret += opponent_reach[i].item() * (
                            action_values_t[a_idx][i].item() - node_values[i].item()
                        )
                    else:
                        regret += opponent_reach[i].item() * (
                            node_values[i].item() - action_values_t[a_idx][i].item()
                        )

                self.regret_sum[key][a_idx] += regret

            p_r = sum(player_reach[i].item() for i in deal_indices)
            self.strategy_sum[key] += p_r * strat

        return node_values

    def _is_leaf(self, states: List) -> bool:
        """Check if this node is a leaf for depth-limited solving."""
        return False

    def _compute_belief(
        self, reach_p0: torch.Tensor, reach_p1: torch.Tensor
    ) -> torch.Tensor:
        """Compute PBS from reach probabilities."""
        joint = reach_p0 * reach_p1
        total = joint.sum()
        if total > 0:
            return joint / total
        return torch.zeros(self.num_deals)

    def _extract_profile(self) -> Dict[str, Dict[str, float]]:
        """Extract average strategy from accumulated strategy sums."""
        profile = {}
        for key, strat_sum in self.strategy_sum.items():
            total = strat_sum.sum()
            if total > 0:
                avg = strat_sum / total
            else:
                n = len(strat_sum)
                avg = torch.full((n,), 1.0 / n)

            actions = self._infoset_actions.get(key, [])
            if not actions:
                continue

            profile[key] = {a: avg[i].item() for i, a in enumerate(actions)}
        return profile
