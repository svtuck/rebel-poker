"""Vectorized CFR operating on all chance outcomes in parallel.

Instead of traversing one deal at a time, this processes all possible
chance outcomes simultaneously using PyTorch tensors. This is the
computational pattern needed for ReBeL's batch value network training.

The solver is game-agnostic: it uses the Game protocol to enumerate
chance outcomes, legal actions, and information sets. Any game that
implements Game can use this solver.

Handles mid-tree chance nodes (e.g., Leduc board card deal) by expanding
each deal's chance outcomes into child states, recursing over the expanded
set, and contracting back via probability-weighted averaging.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from game_interface import Game


class VectorizedCFR:
    """CFR that operates on vectorized chance outcomes using PyTorch.

    For each chance outcome (deal), we track reach probabilities as
    tensors of shape [num_deals]. At each information set, the strategy
    is the same for all deals that map to that infoset, but reach
    probabilities differ per deal.

    This solver discovers the game structure by traversing from the
    initial state, so it works with any Game implementation. Mid-tree
    chance nodes (like Leduc's board card deal) are handled by expanding
    over outcomes weighted by probability.
    """

    def __init__(self, game: Game, device: str = "cpu") -> None:
        self.game = game
        self.device = torch.device(device)

        # Enumerate all chance outcomes (deals)
        initial = game.initial_state()
        assert game.current_player(initial) == -1, (
            "VectorizedCFR requires the game to start with a chance node"
        )
        outcomes = game.chance_outcomes(initial)
        self.all_deals = [deal for deal, _ in outcomes]
        self.deal_probs = torch.tensor(
            [prob for _, prob in outcomes], device=self.device, dtype=torch.float32
        )
        self.num_deals = len(self.all_deals)

        # Map from deal to index
        self._deal_to_idx: Dict = {deal: i for i, deal in enumerate(self.all_deals)}

        # Build infoset -> deal indices mapping by traversing the game tree
        # Maps (infoset_key) -> list of deal indices that belong to it
        self._infoset_deal_indices: Dict[str, List[int]] = {}
        self._build_infoset_mapping()

        # Regret and strategy accumulators
        self.regret_sum: Dict[str, torch.Tensor] = {}
        self.strategy_sum: Dict[str, torch.Tensor] = {}
        self.iteration = 0

        # Precompute terminal values
        self._terminal_cache: Dict = {}

    def _build_infoset_mapping(self) -> None:
        """Discover which deals map to which information sets."""
        game = self.game
        initial = game.initial_state()

        for deal_idx, deal in enumerate(self.all_deals):
            state = game.next_state(initial, deal)
            self._traverse_for_mapping(state, deal_idx)

    def _traverse_for_mapping(self, state, deal_idx: int) -> None:
        """Recursively traverse to find all infoset -> deal mappings."""
        game = self.game
        if game.is_terminal(state):
            return
        player = game.current_player(state)
        if player == -1:
            # Mid-tree chance node (e.g., Leduc board card deal)
            for outcome, _ in game.chance_outcomes(state):
                self._traverse_for_mapping(game.next_state(state, outcome), deal_idx)
            return

        key = game.infoset_key(state, player)
        self._infoset_deal_indices.setdefault(key, [])
        if deal_idx not in self._infoset_deal_indices[key]:
            self._infoset_deal_indices[key].append(deal_idx)

        for action in game.legal_actions(state):
            self._traverse_for_mapping(game.next_state(state, action), deal_idx)

    def _get_strategy(self, infoset_key: str, num_actions: int) -> torch.Tensor:
        """Current strategy via regret matching."""
        if infoset_key not in self.regret_sum:
            self.regret_sum[infoset_key] = torch.zeros(
                num_actions, device=self.device
            )
            self.strategy_sum[infoset_key] = torch.zeros(
                num_actions, device=self.device
            )

        positives = torch.clamp(self.regret_sum[infoset_key], min=0)
        total = positives.sum()
        if total > 0:
            return positives / total
        return torch.full((num_actions,), 1.0 / num_actions, device=self.device)

    def _cfr_vectorized(
        self, states: List, reach_p0: torch.Tensor, reach_p1: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized CFR over all deals simultaneously.

        Args:
            states: list of N states (parallel game trees)
            reach_p0: [N] reach probabilities for player 0
            reach_p1: [N] reach probabilities for player 1

        Returns: [N] expected values for player 0
        """
        game = self.game
        n = len(states)

        # Check if all states are terminal
        if game.is_terminal(states[0]):
            values = torch.zeros(n, device=self.device)
            for i, state in enumerate(states):
                values[i] = game.terminal_utility(state, 0)
            return values

        player = game.current_player(states[0])

        # Handle mid-tree chance nodes by expanding over outcomes
        if player == -1:
            return self._handle_chance_node(states, reach_p0, reach_p1)

        actions = game.legal_actions(states[0])
        num_actions = len(actions)

        # Build per-deal strategy probabilities for each action
        action_values = []
        strategy_per_deal = []

        for a_idx, action in enumerate(actions):
            s = torch.zeros(n, device=self.device)
            for i, state in enumerate(states):
                key = game.infoset_key(state, player)
                strat = self._get_strategy(key, num_actions)
                s[i] = strat[a_idx]
            strategy_per_deal.append(s)

            child_states = [game.next_state(state, action) for state in states]
            if player == 0:
                child_values = self._cfr_vectorized(
                    child_states, reach_p0 * s, reach_p1
                )
            else:
                child_values = self._cfr_vectorized(
                    child_states, reach_p0, reach_p1 * s
                )
            action_values.append(child_values)

        action_values_t = torch.stack(action_values)  # [num_actions, N]
        strategy_t = torch.stack(strategy_per_deal)    # [num_actions, N]

        # Node value: weighted sum of action values
        node_values = (strategy_t * action_values_t).sum(dim=0)  # [N]

        # Update regrets per infoset
        opponent_reach = reach_p1 if player == 0 else reach_p0
        player_reach = reach_p0 if player == 0 else reach_p1

        # Group states by infoset for efficient regret updates
        infoset_local_indices: Dict[str, List[int]] = {}
        for i, state in enumerate(states):
            key = game.infoset_key(state, player)
            infoset_local_indices.setdefault(key, []).append(i)

        for key, local_indices in infoset_local_indices.items():
            strat = self._get_strategy(key, num_actions)
            idx_tensor = torch.tensor(local_indices, device=self.device, dtype=torch.long)

            for a_idx in range(num_actions):
                if player == 0:
                    regret = (
                        (opponent_reach[idx_tensor] * action_values_t[a_idx][idx_tensor]).sum()
                        - (opponent_reach[idx_tensor] * node_values[idx_tensor]).sum()
                    )
                else:
                    regret = (
                        (opponent_reach[idx_tensor] * node_values[idx_tensor]).sum()
                        - (opponent_reach[idx_tensor] * action_values_t[a_idx][idx_tensor]).sum()
                    )
                self.regret_sum[key][a_idx] += regret

            total_player_reach = player_reach[idx_tensor].sum()
            self.strategy_sum[key] += total_player_reach * strat

        return node_values

    def _handle_chance_node(
        self,
        states: List,
        reach_p0: torch.Tensor,
        reach_p1: torch.Tensor,
    ) -> torch.Tensor:
        """Handle mid-tree chance nodes by expanding over outcomes.

        For each state, enumerates chance outcomes and recurses with the
        expanded state set. Values are contracted back via probability-weighted
        averaging. Reach probabilities pass through unchanged, matching
        scalar CFR semantics where chance probability only weights the
        returned value.
        """
        game = self.game
        n = len(states)

        # Build expanded states: for each original state, enumerate outcomes
        expanded_states: List = []
        expanded_reach_p0_list: List[float] = []
        expanded_reach_p1_list: List[float] = []
        # Track (original_idx, chance_prob) per expanded entry for contraction
        expansion_map: List[Tuple[int, float]] = []

        for i, state in enumerate(states):
            for outcome, prob in game.chance_outcomes(state):
                child = game.next_state(state, outcome)
                expanded_states.append(child)
                expanded_reach_p0_list.append(reach_p0[i].item())
                expanded_reach_p1_list.append(reach_p1[i].item())
                expansion_map.append((i, prob))

        exp_reach_p0 = torch.tensor(
            expanded_reach_p0_list, device=self.device, dtype=torch.float32
        )
        exp_reach_p1 = torch.tensor(
            expanded_reach_p1_list, device=self.device, dtype=torch.float32
        )

        expanded_values = self._cfr_vectorized(
            expanded_states, exp_reach_p0, exp_reach_p1
        )

        # Contract: probability-weighted sum back to original indices
        values = torch.zeros(n, device=self.device)
        for j, (orig_idx, prob) in enumerate(expansion_map):
            values[orig_idx] += prob * expanded_values[j]

        return values

    def train(self, iterations: int) -> List[float]:
        """Run vectorized CFR for given iterations.

        Returns exploitability at checkpoints.
        """
        game = self.game
        initial = game.initial_state()
        exploitabilities = []

        for i in range(iterations):
            self.iteration += 1

            # Build per-deal initial states
            deal_states = [game.next_state(initial, deal) for deal in self.all_deals]

            self._cfr_vectorized(
                deal_states,
                self.deal_probs.clone(),
                self.deal_probs.clone(),
            )

            if (i + 1) % max(1, iterations // 50) == 0 or i < 10:
                exp = self._exploitability()
                exploitabilities.append(exp)

        return exploitabilities

    def average_strategy_profile(self) -> Dict[str, Dict[str, float]]:
        """Return converged average strategy.

        Uses the game to determine action labels for each infoset.
        """
        game = self.game
        initial = game.initial_state()
        profile = {}

        # We need to map infoset keys back to their actions.
        # Build this mapping by traversing the game tree once.
        infoset_actions: Dict[str, List] = {}
        self._collect_infoset_actions(initial, infoset_actions)

        for key, strat_sum in self.strategy_sum.items():
            total = strat_sum.sum()
            if total > 0:
                avg = strat_sum / total
            else:
                n = len(strat_sum)
                avg = torch.full((n,), 1.0 / n)

            actions = infoset_actions.get(key, [])
            if not actions:
                continue

            profile[key] = {a: avg[i].item() for i, a in enumerate(actions)}

        return profile

    def _collect_infoset_actions(self, state, mapping: Dict[str, List]) -> None:
        """Traverse game tree to map infoset keys to their legal actions."""
        game = self.game
        if game.is_terminal(state):
            return
        player = game.current_player(state)
        if player == -1:
            for outcome, _ in game.chance_outcomes(state):
                self._collect_infoset_actions(game.next_state(state, outcome), mapping)
            return
        key = game.infoset_key(state, player)
        if key not in mapping:
            mapping[key] = game.legal_actions(state)
        for action in game.legal_actions(state):
            self._collect_infoset_actions(game.next_state(state, action), mapping)

    def _exploitability(self) -> float:
        """Compute exploitability using the scalar CFR trainer's method."""
        from cfr.solver import CFRTrainer
        profile = self.average_strategy_profile()
        dummy = CFRTrainer(self.game)
        br0 = dummy._best_response_value(profile, 0)
        br1 = dummy._best_response_value(profile, 1)
        return 0.5 * (br0 + br1)
