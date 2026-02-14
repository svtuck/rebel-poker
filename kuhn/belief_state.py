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
  - deals: [6] possible card deals (i,j) where i≠j, i,j ∈ {0,1,2}
  - For each deal, we track reach probabilities for both players
  - PBS at any public history = normalized product of reaches × chance probs
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
      - reach_probs[player]: shape [6] — reach probability for each deal
      - belief: shape [6] — normalized PBS

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

        PBS(h)[d] = P(deal=d | history=h) ∝ chance(d) * reach_0(d,h) * reach_1(d,h)

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


class VectorizedCFR:
    """CFR that operates on belief state tensors directly.

    Instead of traversing one deal at a time, this processes all 6 deals
    in parallel using PyTorch tensors. This is the computational pattern
    needed for ReBeL's batch value network training.

    The key insight: at each information set, the strategy is the same
    for all deals that map to that infoset (same card, same history).
    But the reach probabilities differ per deal.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.game = KuhnPoker()
        self.device = torch.device(device)
        self.chance_probs = initial_chance_probs().to(self.device)

        # Regret and strategy accumulators: infoset_key -> [num_actions] tensors
        self.regret_sum: Dict[str, torch.Tensor] = {}
        self.strategy_sum: Dict[str, torch.Tensor] = {}
        self.iteration = 0

        # Terminal values: precomputed [NUM_DEALS] for each terminal history
        self._terminal_values = self._precompute_terminal_values()

        # Build mappings for vectorized operations
        self._build_deal_masks()

    def _precompute_terminal_values(self) -> Dict[str, torch.Tensor]:
        """Precompute terminal utilities for all deals."""
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

    def _build_deal_masks(self) -> None:
        """Build boolean masks mapping cards to deal indices.

        self.card_masks[player][card] is a boolean tensor of shape [NUM_DEALS]
        that is True for deals where `player` holds `card`.
        """
        self.card_masks = {0: {}, 1: {}}
        for player in (0, 1):
            for card in CARD_RANKS:
                mask = torch.zeros(NUM_DEALS, dtype=torch.bool, device=self.device)
                card_to_deals = CARD_TO_DEALS_P0 if player == 0 else CARD_TO_DEALS_P1
                for idx in card_to_deals[card]:
                    mask[idx] = True
                self.card_masks[player][card] = mask

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

    def _expand_strategy_to_deals(
        self, history: str, player: int, action_idx: int, num_actions: int
    ) -> torch.Tensor:
        """Expand infoset strategy to per-deal action probabilities.

        For each deal, look up which card `player` holds, get that
        infoset's strategy, and return the probability of action_idx.

        Returns: [NUM_DEALS] tensor of action probabilities.
        """
        probs = torch.zeros(NUM_DEALS, device=self.device)
        card_to_deals = CARD_TO_DEALS_P0 if player == 0 else CARD_TO_DEALS_P1

        for card in CARD_RANKS:
            infoset_key = f"{RANK_NAMES[card]}|{history}"
            strat = self._get_strategy(infoset_key, num_actions)
            for deal_idx in card_to_deals[card]:
                probs[deal_idx] = strat[action_idx]

        return probs

    def _cfr_vectorized(
        self, history: str, reach_p0: torch.Tensor, reach_p1: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized CFR over all deals simultaneously.

        Args:
            history: public action history
            reach_p0: [NUM_DEALS] reach probabilities for player 0
            reach_p1: [NUM_DEALS] reach probabilities for player 1

        Returns: [NUM_DEALS] expected values for player 0
        """
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

        # Compute action values for all deals
        action_values = []
        strategy_per_deal = []  # [num_actions, NUM_DEALS]

        for a_idx, action in enumerate(actions):
            # Get per-deal strategy probabilities
            s = self._expand_strategy_to_deals(history, player, a_idx, num_actions)
            strategy_per_deal.append(s)

            # Update reach and recurse
            child_history = history + action
            if player == 0:
                child_values = self._cfr_vectorized(
                    child_history, reach_p0 * s, reach_p1
                )
            else:
                child_values = self._cfr_vectorized(
                    child_history, reach_p0, reach_p1 * s
                )
            action_values.append(child_values)

        # Stack: [num_actions, NUM_DEALS]
        action_values_t = torch.stack(action_values)
        strategy_t = torch.stack(strategy_per_deal)

        # Node value: sum over actions of strategy * action_value, per deal
        node_values = (strategy_t * action_values_t).sum(dim=0)  # [NUM_DEALS]

        # Update regrets per infoset
        opponent_reach = reach_p1 if player == 0 else reach_p0
        player_reach = reach_p0 if player == 0 else reach_p1

        card_to_deals = CARD_TO_DEALS_P0 if player == 0 else CARD_TO_DEALS_P1

        for card in CARD_RANKS:
            infoset_key = f"{RANK_NAMES[card]}|{history}"
            strat = self._get_strategy(infoset_key, num_actions)

            # Sum opponent reach over deals where this player has this card
            deal_indices = card_to_deals[card]
            mask = self.card_masks[player][card]

            for a_idx in range(num_actions):
                # Counterfactual regret: weighted by opponent reach
                # For player 1, values are from P0's perspective so we negate
                if player == 0:
                    regret = (
                        (opponent_reach[mask] * action_values_t[a_idx][mask]).sum()
                        - (opponent_reach[mask] * node_values[mask]).sum()
                    )
                else:
                    regret = (
                        (opponent_reach[mask] * node_values[mask]).sum()
                        - (opponent_reach[mask] * action_values_t[a_idx][mask]).sum()
                    )
                self.regret_sum[infoset_key][a_idx] += regret

            # Update strategy sum weighted by player reach
            total_player_reach = player_reach[mask].sum()
            self.strategy_sum[infoset_key] += total_player_reach * strat

        return node_values

    def train(self, iterations: int) -> List[float]:
        """Run vectorized CFR for given iterations.

        Returns exploitability at checkpoints.

        Reach probabilities are initialized to the chance probability (1/6)
        for each deal, which correctly incorporates the chance node's
        contribution. This means reach[d] = P(chance=d) * P(player actions).
        """
        exploitabilities = []

        for i in range(iterations):
            self.iteration += 1
            self._cfr_vectorized("", self.chance_probs.clone(), self.chance_probs.clone())

            if (i + 1) % max(1, iterations // 50) == 0 or i < 10:
                exp = self._exploitability()
                exploitabilities.append(exp)

        return exploitabilities

    def average_strategy_profile(self) -> Dict[str, Dict[str, float]]:
        """Return converged average strategy."""
        profile = {}
        for key, strat_sum in self.strategy_sum.items():
            total = strat_sum.sum()
            if total > 0:
                avg = strat_sum / total
            else:
                n = len(strat_sum)
                avg = torch.full((n,), 1.0 / n)

            # Determine actions from history
            parts = key.split("|")
            history = parts[1] if len(parts) > 1 else ""
            if history in ("", "c"):
                actions = ["c", "b"]
            else:
                actions = ["c", "f"]

            profile[key] = {a: avg[i].item() for i, a in enumerate(actions)}
        return profile

    def _exploitability(self) -> float:
        """Compute exploitability using the scalar CFR trainer's method."""
        from kuhn.cfr import CFRTrainer
        profile = self.average_strategy_profile()
        dummy = CFRTrainer(self.game)
        br0 = dummy._best_response_value(profile, 0)
        br1 = dummy._best_response_value(profile, 1)
        return 0.5 * (br0 + br1)

    def get_belief_state(self, history: str) -> torch.Tensor:
        """Get the public belief state at a given history.

        This uses the current strategy to compute reach probabilities
        and returns the normalized belief over deals.
        """
        tracker = BeliefStateTracker(device=str(self.device))
        profile = self.average_strategy_profile()
        tracker.set_strategy_from_profile(profile)
        beliefs = tracker.compute_belief_states()
        return beliefs.get(history, torch.zeros(NUM_DEALS, device=self.device))
