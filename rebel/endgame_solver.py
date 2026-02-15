"""Poker endgame solver for ReBeL integration.

This module provides depth-limited subgame solving using CFR, following
the ReBeL paper (Brown & Sandholm, 2020, Section 3.2).

In ReBeL, the endgame solver:
1. Takes a Public Belief State at a given node
2. Runs CFR on the subgame rooted at that node
3. Uses a value network V(PBS) at leaf nodes instead of full rollouts
4. Returns the solved strategy and CFR values for training data

The solver is game-agnostic: it uses the Game protocol to traverse the
game tree and BeliefConfig for belief state operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from belief_config import BeliefConfig
from game_interface import Game


@dataclass
class SubgameConfig:
    """Configuration for a subgame solve."""
    iterations: int = 1000
    use_cfr_plus: bool = False
    discount_factor: float = 1.0
    max_depth: Optional[int] = None  # None = solve to terminal (no depth limit)


class SubgameSolver:
    """Solves a poker subgame from a given PBS using CFR.

    This is the core of ReBeL's search procedure. Given:
    - A game implementing the Game protocol
    - A BeliefConfig describing the game's belief structure
    - A public belief state [num_private_states, num_players]
    - A root history in the game tree
    - Optional: a value function for leaf evaluation
    - Optional: max_depth for depth-limited solving

    It runs CFR on the subgame and returns the solved strategy.
    """

    def __init__(
        self,
        game: Game,
        belief_config: BeliefConfig,
        root_history: str,
        initial_beliefs: torch.Tensor,
        value_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        config: Optional[SubgameConfig] = None,
    ) -> None:
        self.game = game
        self.belief_config = belief_config
        self.root_history = root_history
        self.initial_beliefs = initial_beliefs
        self.value_fn = value_fn
        self.config = config or SubgameConfig()
        self.num_deals = belief_config.num_deals

        # Enumerate chance outcomes to build deal -> state mapping
        initial = game.initial_state()
        outcomes = game.chance_outcomes(initial)
        self.all_deals = [deal for deal, _ in outcomes]
        self._deal_to_idx = {
            (tuple(deal) if isinstance(deal, (list, tuple)) else deal): i
            for i, deal in enumerate(self.all_deals)
        }

        # Cache infoset -> actions mapping
        self._infoset_actions: Dict[str, List] = {}

        # Build infoset -> deal indices mapping
        self._infoset_deal_indices: Dict[str, List[int]] = {}
        for deal_idx, deal in enumerate(self.all_deals):
            state = game.next_state(initial, deal)
            self._build_infoset_map(state, deal_idx)

        # Convert PBS to per-deal reach probabilities for CFR
        self._initial_reach_p0, self._initial_reach_p1 = belief_config.pbs_to_reach(
            initial_beliefs
        )

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
        """Run CFR on the subgame and return the average strategy."""
        game = self.game
        initial = game.initial_state()

        for iteration in range(self.config.iterations):
            reach_p0 = self._initial_reach_p0.clone()
            reach_p1 = self._initial_reach_p1.clone()

            deal_states = [game.next_state(initial, deal) for deal in self.all_deals]
            self._cfr(deal_states, reach_p0, reach_p1, depth=0)

        return self._extract_profile()

    def solve_with_values(self) -> Tuple[Dict[str, Dict[str, float]], torch.Tensor]:
        """Run CFR and return both the average strategy and root node values.

        Returns:
            profile: {infoset_key: {action: probability}}
            root_values: [num_private_states, num_players] values at root
        """
        game = self.game
        initial = game.initial_state()
        last_values = None

        for iteration in range(self.config.iterations):
            reach_p0 = self._initial_reach_p0.clone()
            reach_p1 = self._initial_reach_p1.clone()

            deal_states = [game.next_state(initial, deal) for deal in self.all_deals]
            last_values = self._cfr(deal_states, reach_p0, reach_p1, depth=0)

        # Convert per-deal values to PBS-space values
        root_pbs_values = self.belief_config.deal_values_to_pbs_values(last_values)
        return self._extract_profile(), root_pbs_values

    def _cfr(
        self, states: List, reach_p0: torch.Tensor, reach_p1: torch.Tensor,
        depth: int,
    ) -> torch.Tensor:
        """CFR traversal for the subgame. Returns [num_deals] values."""
        game = self.game

        # Terminal nodes
        if game.is_terminal(states[0]):
            v = torch.zeros(self.num_deals)
            for deal_idx, state in enumerate(states):
                v[deal_idx] = game.terminal_utility(state, 0)
            return v

        # Depth-limited leaf: use value function
        if self._is_leaf(depth):
            return self._evaluate_leaf(reach_p0, reach_p1)

        player = game.current_player(states[0])
        actions = game.legal_actions(states[0])
        num_actions = len(actions)

        action_values = []
        strategy_per_deal = []

        # Group deals by infoset
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
                child_values = self._cfr(child_states, reach_p0 * s, reach_p1, depth + 1)
            else:
                child_values = self._cfr(child_states, reach_p0, reach_p1 * s, depth + 1)
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

    def _is_leaf(self, depth: int) -> bool:
        """Check if this node is a leaf for depth-limited solving."""
        if self.config.max_depth is None:
            return False
        if self.value_fn is None:
            return False
        return depth >= self.config.max_depth

    def _evaluate_leaf(
        self, reach_p0: torch.Tensor, reach_p1: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate a leaf node using the value function.

        Converts reach probs to PBS, queries value net, converts back to deal values.
        """
        pbs = self.belief_config.reach_to_pbs(reach_p0, reach_p1)
        pbs_values = self.value_fn(pbs)
        return self.belief_config.pbs_values_to_deal_values(pbs_values)

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
