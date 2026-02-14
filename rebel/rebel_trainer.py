"""ReBeL training loop for Kuhn Poker.

This implements the ReBeL algorithm's training pipeline:

1. Self-play with CFR at decision points
2. Log PBS + values at each decision point
3. Train value network V(PBS) on accumulated data
4. Use V(PBS) as leaf evaluator in subsequent CFR iterations

The key loop is:
  for epoch in range(num_epochs):
    1. Run CFR on full game (or subgame) using current V as leaf evaluator
    2. Collect training data: (PBS, CFR_values) pairs
    3. Train V to predict CFR_values from PBS
    4. Evaluate: compare V-predicted values to actual CFR values

For Kuhn Poker, this is overkill (the game is fully solvable by tabular CFR),
but it validates the approach before scaling to Leduc and NLHE.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from game_interface import Game
from kuhn.game import KuhnPoker, RANK_NAMES, CARD_RANKS
from kuhn.belief_state import (
    ALL_DEALS,
    CARD_TO_DEALS_P0,
    CARD_TO_DEALS_P1,
    NUM_DEALS,
    TERMINAL_HISTORIES,
    BeliefStateTracker,
)
from cfr.solver import CFRTrainer
from rebel.value_net import ValueNetwork, train_value_network
from rebel.data_logger import RebelDataLogger


class RebelTrainer:
    """Full ReBeL training pipeline for Kuhn Poker.

    This orchestrates the self-play, data collection, and neural
    network training loop.
    """

    def __init__(
        self,
        value_hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        cfr_iterations: int = 100,
        device: str = "cpu",
    ) -> None:
        self.game = KuhnPoker()
        self.device = torch.device(device)
        self.cfr_iterations = cfr_iterations
        self.lr = learning_rate

        # Value network
        self.value_net = ValueNetwork(hidden_dim=value_hidden_dim).to(self.device)

        # Data logger
        self.logger = RebelDataLogger()

        # CFR state
        self.regret_sum: Dict[str, torch.Tensor] = {}
        self.strategy_sum: Dict[str, torch.Tensor] = {}

        # Terminal values (precomputed)
        self._terminal_values = self._precompute_terminal_values()

        # Training history
        self.value_losses: List[float] = []
        self.exploitabilities: List[float] = []

    def _precompute_terminal_values(self) -> Dict[str, torch.Tensor]:
        values = {}
        for h in TERMINAL_HISTORIES:
            v = torch.zeros(NUM_DEALS, device=self.device)
            for deal_idx, (c0, c1) in enumerate(ALL_DEALS):
                state = self.game.next_state(self.game.initial_state(), (c0, c1))
                for action in h:
                    state = self.game.next_state(state, action)
                v[deal_idx] = self.game.terminal_utility(state, 0)
            values[h] = v
        return values

    def _get_strategy(self, key: str, num_actions: int) -> torch.Tensor:
        if key not in self.regret_sum:
            self.regret_sum[key] = torch.zeros(num_actions, device=self.device)
            self.strategy_sum[key] = torch.zeros(num_actions, device=self.device)
        positives = torch.clamp(self.regret_sum[key], min=0)
        total = positives.sum()
        if total > 0:
            return positives / total
        return torch.full((num_actions,), 1.0 / num_actions, device=self.device)

    def train(
        self,
        num_epochs: int = 10,
        cfr_iters_per_epoch: int = 100,
        value_train_epochs: int = 50,
    ) -> Dict[str, List[float]]:
        """Run the full ReBeL training loop.

        Returns training metrics.
        """
        chance = torch.full((NUM_DEALS,), 1.0 / NUM_DEALS, device=self.device)

        for epoch in range(num_epochs):
            self.logger.clear()

            # Phase 1: Run CFR, collecting PBS data
            for cfr_iter in range(cfr_iters_per_epoch):
                self._cfr_with_logging(
                    "", chance.clone(), chance.clone(), cfr_iter
                )

            # Phase 2: Train value network on collected data
            dataset = self.logger.to_dataset()
            if len(dataset["beliefs"]) > 0:
                losses = train_value_network(
                    self.value_net,
                    dataset,
                    epochs=value_train_epochs,
                    lr=self.lr,
                )
                if losses:
                    self.value_losses.append(losses[-1])

            # Phase 3: Evaluate
            profile = self._extract_profile()
            dummy = CFRTrainer(self.game)
            br0 = dummy._best_response_value(profile, 0)
            br1 = dummy._best_response_value(profile, 1)
            exp = 0.5 * (br0 + br1)
            self.exploitabilities.append(exp)

        return {
            "value_losses": self.value_losses,
            "exploitabilities": self.exploitabilities,
        }

    def _cfr_with_logging(
        self,
        history: str,
        reach_p0: torch.Tensor,
        reach_p1: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        """CFR traversal that also logs PBS data for value network training."""

        if history in TERMINAL_HISTORIES:
            return self._terminal_values[history]

        player = len(history) % 2
        if history in ("", "c"):
            actions = ["c", "b"]
        elif history in ("b", "cb"):
            actions = ["c", "f"]
        else:
            return torch.zeros(NUM_DEALS, device=self.device)

        num_actions = len(actions)
        action_values = []
        strategy_per_deal = []

        card_to_deals = CARD_TO_DEALS_P0 if player == 0 else CARD_TO_DEALS_P1

        for a_idx, action in enumerate(actions):
            s = torch.zeros(NUM_DEALS, device=self.device)
            for card in CARD_RANKS:
                key = f"{RANK_NAMES[card]}|{history}"
                strat = self._get_strategy(key, num_actions)
                for deal_idx in card_to_deals[card]:
                    s[deal_idx] = strat[a_idx]
            strategy_per_deal.append(s)

            child_history = history + action
            if player == 0:
                child_values = self._cfr_with_logging(
                    child_history, reach_p0 * s, reach_p1, iteration
                )
            else:
                child_values = self._cfr_with_logging(
                    child_history, reach_p0, reach_p1 * s, iteration
                )
            action_values.append(child_values)

        action_values_t = torch.stack(action_values)
        strategy_t = torch.stack(strategy_per_deal)
        node_values = (strategy_t * action_values_t).sum(dim=0)

        # Log PBS and values for this decision point
        joint = reach_p0 * reach_p1
        total = joint.sum()
        if total > 0:
            belief = joint / total
            strategy_dict = {}
            for card in CARD_RANKS:
                key = f"{RANK_NAMES[card]}|{history}"
                strategy_dict[key] = self._get_strategy(key, num_actions)

            self.logger.log_state(
                history=history,
                belief=belief,
                reach_p0=reach_p0,
                reach_p1=reach_p1,
                strategy=strategy_dict,
                values_p0=node_values,
                iteration=iteration,
            )

        # Update regrets
        opponent_reach = reach_p1 if player == 0 else reach_p0
        player_reach = reach_p0 if player == 0 else reach_p1

        for card in CARD_RANKS:
            key = f"{RANK_NAMES[card]}|{history}"
            strat = self._get_strategy(key, num_actions)
            indices = card_to_deals[card]

            for a_idx in range(num_actions):
                opp_vals = torch.stack([opponent_reach[i] for i in indices])
                act_vals = torch.stack([action_values_t[a_idx][i] for i in indices])
                node_vals = torch.stack([node_values[i] for i in indices])

                if player == 0:
                    regret = (opp_vals * (act_vals - node_vals)).sum()
                else:
                    regret = (opp_vals * (node_vals - act_vals)).sum()

                self.regret_sum[key][a_idx] += regret

            p_r = sum(player_reach[i].item() for i in indices)
            self.strategy_sum[key] += p_r * strat

        return node_values

    def _extract_profile(self) -> Dict[str, Dict[str, float]]:
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
