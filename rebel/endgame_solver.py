"""Poker endgame solver scaffold for ReBeL integration.

This module provides the framework for solving poker endgames (subgames)
using CFR, following the pattern from Noam Brown's poker_solver repo.

In ReBeL, the endgame solver:
1. Takes a Public Belief State at a given node
2. Runs CFR on the subgame rooted at that node
3. Uses a value network V(PBS) at leaf nodes instead of full rollouts
4. Returns the solved strategy for the subgame

For Kuhn Poker, the "endgame" is the entire game (it's small enough).
For larger games (Leduc, NLHE), this becomes essential for real-time play.

References:
  - noambrown/poker_solver: River NLHE subgame solver
  - ReBeL paper (Brown & Sandholm, 2020): Section 3.2, Subgame solving
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from kuhn.game import KuhnPoker, KuhnState, RANK_NAMES, CARD_RANKS
from kuhn.belief_state import ALL_DEALS, NUM_DEALS, CARD_TO_DEALS_P0, CARD_TO_DEALS_P1


@dataclass
class SubgameConfig:
    """Configuration for a subgame solve."""
    iterations: int = 1000
    use_cfr_plus: bool = False
    discount_factor: float = 1.0


class SubgameSolver:
    """Solves a poker subgame from a given PBS using CFR.

    This is the core of ReBeL's search procedure. Given:
    - A public belief state (probability over deals)
    - A root history in the game tree
    - Optional: a value function for leaf evaluation

    It runs CFR on the subgame and returns the solved strategy.
    """

    def __init__(
        self,
        game: KuhnPoker,
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

        # Regret and strategy tables for the subgame
        self.regret_sum: Dict[str, torch.Tensor] = {}
        self.strategy_sum: Dict[str, torch.Tensor] = {}

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
        for iteration in range(self.config.iterations):
            # Reach probs weighted by initial beliefs
            reach_p0 = self.initial_beliefs.clone()
            reach_p1 = self.initial_beliefs.clone()
            self._cfr(self.root_history, reach_p0, reach_p1)

        return self._extract_profile()

    def _cfr(
        self, history: str, reach_p0: torch.Tensor, reach_p1: torch.Tensor
    ) -> torch.Tensor:
        """CFR traversal for the subgame. Returns [NUM_DEALS] values."""
        from kuhn.belief_state import TERMINAL_HISTORIES

        if history in TERMINAL_HISTORIES:
            # Terminal node: compute payoffs
            v = torch.zeros(NUM_DEALS)
            for deal_idx, (c0, c1) in enumerate(ALL_DEALS):
                state = self.game.next_state(self.game.initial_state(), (c0, c1))
                for action in history:
                    state = self.game.next_state(state, action)
                v[deal_idx] = self.game.terminal_utility(state, 0)
            return v

        # Check if we should use value function for leaf evaluation
        if self.value_fn is not None and self._is_leaf(history):
            belief = self._compute_belief(reach_p0, reach_p1)
            return self.value_fn(belief, history)

        player = len(history) % 2
        if history in ("", "c"):
            actions = ["c", "b"]
        elif history in ("b", "cb"):
            actions = ["c", "f"]
        else:
            return torch.zeros(NUM_DEALS)

        num_actions = len(actions)
        action_values = []
        strategy_per_deal = []

        card_to_deals = CARD_TO_DEALS_P0 if player == 0 else CARD_TO_DEALS_P1

        for a_idx, action in enumerate(actions):
            s = torch.zeros(NUM_DEALS)
            for card in CARD_RANKS:
                key = f"{RANK_NAMES[card]}|{history}"
                strat = self._get_strategy(key, num_actions)
                for deal_idx in card_to_deals[card]:
                    s[deal_idx] = strat[a_idx]
            strategy_per_deal.append(s)

            child_history = history + action
            if player == 0:
                child_values = self._cfr(child_history, reach_p0 * s, reach_p1)
            else:
                child_values = self._cfr(child_history, reach_p0, reach_p1 * s)
            action_values.append(child_values)

        action_values_t = torch.stack(action_values)
        strategy_t = torch.stack(strategy_per_deal)
        node_values = (strategy_t * action_values_t).sum(dim=0)

        # Update regrets
        opponent_reach = reach_p1 if player == 0 else reach_p0
        player_reach = reach_p0 if player == 0 else reach_p1

        for card in CARD_RANKS:
            key = f"{RANK_NAMES[card]}|{history}"
            strat = self._get_strategy(key, num_actions)
            indices = card_to_deals[card]

            for a_idx in range(num_actions):
                # Correct CFR regret: weighted sum of per-deal regrets
                # regret(I, a) = Σ_{h∈I} opp_reach(h) * (v(h,a) - v(h))
                regret = 0.0
                for i in indices:
                    if player == 0:
                        regret += opponent_reach[i].item() * (
                            action_values_t[a_idx][i].item() - node_values[i].item()
                        )
                    else:
                        regret += opponent_reach[i].item() * (
                            node_values[i].item() - action_values_t[a_idx][i].item()
                        )

                self.regret_sum[key][a_idx] += regret

            p_r = sum(player_reach[i].item() for i in indices)
            self.strategy_sum[key] += p_r * strat

        return node_values

    def _is_leaf(self, history: str) -> bool:
        """Check if this node is a leaf for depth-limited solving."""
        # For Kuhn, no depth limit needed (game is tiny)
        return False

    def _compute_belief(
        self, reach_p0: torch.Tensor, reach_p1: torch.Tensor
    ) -> torch.Tensor:
        """Compute PBS from reach probabilities."""
        joint = reach_p0 * reach_p1
        total = joint.sum()
        if total > 0:
            return joint / total
        return torch.zeros(NUM_DEALS)

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

            parts = key.split("|")
            history = parts[1] if len(parts) > 1 else ""
            if history in ("", "c"):
                actions = ["c", "b"]
            else:
                actions = ["c", "f"]

            profile[key] = {a: avg[i].item() for i, a in enumerate(actions)}
        return profile
