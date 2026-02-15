"""ReBeL training loop for Leduc Poker.

Extends the ReBeL approach to Leduc's multi-round structure with
community cards. The key differences from Kuhn:

1. Two betting rounds with a chance node (board card) between them
2. Larger game tree (288 information sets vs ~12 for Kuhn)
3. More deals: 30 private × 4 board = 120 full configurations
4. PBS needs to account for board card information

The training loop uses tabular CFR on the full Leduc game tree,
collecting PBS and value data for value network training.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from game_interface import Game
from leduc.game import LeducPoker, RANK_TO_STR, RANKS
from leduc.belief_state import (
    ALL_FULL_DEALS,
    ALL_PRIVATE_DEALS,
    NUM_FULL_DEALS,
    NUM_PLAYERS,
    NUM_PRIVATE_DEALS,
    NUM_PRIVATE_STATES,
    RANK_TO_FULL_DEALS_P0,
    RANK_TO_FULL_DEALS_P1,
    RANK_TO_DEALS_P0,
    RANK_TO_DEALS_P1,
    reach_to_pbs_full,
    reach_to_pbs_private,
)
from cfr.solver import CFRTrainer
from rebel.value_net import train_value_network
from rebel.data_logger import RebelDataLogger


# PBS dimension for Leduc: same as Kuhn (3 ranks × 2 players = 6)
LEDUC_PBS_DIM = NUM_PRIVATE_STATES * NUM_PLAYERS


class LeducValueNetwork(nn.Module):
    """Value network for Leduc Poker.

    Input: PBS flattened to R^6 (same shape as Kuhn since 3 ranks × 2 players)
    Output: values in R^6

    Larger hidden layers than Kuhn since Leduc has more complex value structure.
    """

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LEDUC_PBS_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, LEDUC_PBS_DIM),
        )

    def forward(self, pbs: torch.Tensor) -> torch.Tensor:
        return self.net(pbs)


def full_deal_values_to_pbs_values(
    node_values: torch.Tensor,
) -> torch.Tensor:
    """Convert per-full-deal values [120] to per-private-state values [3, 2].

    For each (rank, player), averages the value across all full deals where
    that player holds that rank.
    """
    pbs_values = torch.zeros(NUM_PRIVATE_STATES, NUM_PLAYERS, device=node_values.device)

    for rank in RANKS:
        p0_indices = RANK_TO_FULL_DEALS_P0[rank]
        if p0_indices:
            pbs_values[rank, 0] = sum(node_values[i] for i in p0_indices) / len(p0_indices)

        p1_indices = RANK_TO_FULL_DEALS_P1[rank]
        if p1_indices:
            pbs_values[rank, 1] = sum(-node_values[i] for i in p1_indices) / len(p1_indices)

    return pbs_values


class LeducRebelTrainer:
    """ReBeL training pipeline for Leduc Poker.

    Uses tabular CFR over the full game tree, collecting PBS data for
    value network training. The CFR solver handles all game traversal
    via the Game protocol.
    """

    def __init__(
        self,
        value_hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.game = LeducPoker()
        self.device = torch.device(device)
        self.lr = learning_rate

        # Value network
        self.value_net = LeducValueNetwork(hidden_dim=value_hidden_dim).to(self.device)

        # Data logger
        self.logger = RebelDataLogger()

        # Persistent CFR solver (accumulates across epochs)
        self._cfr = CFRTrainer(self.game)

        # Training history
        self.value_losses: List[float] = []
        self.exploitabilities: List[float] = []

    def train(
        self,
        num_epochs: int = 5,
        cfr_iters_per_epoch: int = 200,
        value_train_epochs: int = 50,
    ) -> Dict[str, List[float]]:
        """Run the full ReBeL training loop.

        Each epoch:
        1. Run CFR on Leduc for cfr_iters_per_epoch more iterations (cumulative)
        2. Extract PBS data from the converged strategy
        3. Train value network on the collected data
        4. Measure exploitability (should decrease as CFR accumulates)
        """
        for epoch in range(num_epochs):
            self.logger.clear()

            # Phase 1: Run more CFR iterations (cumulative)
            self._cfr.train(cfr_iters_per_epoch)
            profile = self._cfr.average_strategy_profile()

            # Phase 2: Collect PBS training data by traversing the game tree
            # with the converged strategy
            self._collect_pbs_data(profile)

            # Phase 3: Train value network
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

            # Phase 4: Evaluate exploitability
            exp = self._cfr.exploitability()
            self.exploitabilities.append(exp)

        return {
            "value_losses": self.value_losses,
            "exploitabilities": self.exploitabilities,
        }

    def _collect_pbs_data(self, profile: Dict[str, Dict[str, float]]) -> None:
        """Traverse the game tree with the given strategy, computing PBS at each
        decision point and logging it for value network training.

        We traverse over all full deals (card_p0, card_p1, board) and track
        reach probabilities. At each decision point, we compute the PBS and
        expected values.
        """
        game = self.game

        # Traverse over all full deals
        full_deal_reach_p0 = torch.ones(NUM_FULL_DEALS, device=self.device)
        full_deal_reach_p1 = torch.ones(NUM_FULL_DEALS, device=self.device)

        # We need to compute expected values for each full deal at each history.
        # Rather than a vectorized traversal, collect data by traversing each
        # full deal independently and aggregating.

        # For each non-terminal public history, collect:
        # - reach-weighted PBS
        # - reach-weighted values
        history_data: Dict[str, Dict] = {}

        for fd_idx, (c0, c1, board) in enumerate(ALL_FULL_DEALS):
            # Build the state after dealing cards and board
            state = game.initial_state()
            state = game.next_state(state, (c0, c1))

            self._traverse_deal(
                state, fd_idx, board, profile, 1.0, 1.0, history_data
            )

        # Convert aggregated data to PBS training points
        for history, data in history_data.items():
            reach_p0 = data["reach_p0"]
            reach_p1 = data["reach_p1"]
            values = data["values"]
            has_board = data["has_board"]

            # Compute PBS from reach
            if has_board:
                pbs = reach_to_pbs_full(reach_p0, reach_p1)
            else:
                # Preflop: marginalize over board cards to get private-deal reaches
                private_reach_p0 = torch.zeros(NUM_PRIVATE_DEALS, device=self.device)
                private_reach_p1 = torch.zeros(NUM_PRIVATE_DEALS, device=self.device)
                for fd_idx in range(NUM_FULL_DEALS):
                    pd_idx = _full_to_private_idx(fd_idx)
                    private_reach_p0[pd_idx] += reach_p0[fd_idx]
                    private_reach_p1[pd_idx] += reach_p1[fd_idx]
                pbs = reach_to_pbs_private(private_reach_p0, private_reach_p1)

            # Compute PBS values
            pbs_values = full_deal_values_to_pbs_values(values)

            # Log data point
            joint = reach_p0 * reach_p1
            if joint.sum() > 0:
                self.logger.log_state(
                    history=history,
                    belief=pbs,
                    reach_p0=reach_p0,
                    reach_p1=reach_p1,
                    strategy={},  # Not tracking per-infoset strategy here
                    values=pbs_values,
                    iteration=0,
                )

    def _traverse_deal(
        self,
        state,
        fd_idx: int,
        board_card: Tuple[int, int],
        profile: Dict[str, Dict[str, float]],
        reach_p0: float,
        reach_p1: float,
        history_data: Dict,
    ) -> float:
        """Traverse one full deal through the game tree.

        Returns the expected value for player 0 at this state.
        """
        game = self.game

        if game.is_terminal(state):
            return game.terminal_utility(state, 0)

        player = game.current_player(state)

        if player == -1:
            # Chance node (board card) - apply the known board card
            board_outcomes = game.chance_outcomes(state)
            # Find the matching outcome
            for outcome, prob in board_outcomes:
                if outcome == board_card:
                    child = game.next_state(state, outcome)
                    return self._traverse_deal(
                        child, fd_idx, board_card, profile,
                        reach_p0, reach_p1, history_data
                    )
            # Board card not in remaining (shouldn't happen)
            raise ValueError(f"Board card {board_card} not in chance outcomes")

        # Decision node: get strategy and traverse
        key = game.infoset_key(state, player)
        actions = game.legal_actions(state)
        action_probs = profile.get(key, {})

        # Record reach for this history
        history = state.history
        has_board = state.board is not None

        if history not in history_data:
            history_data[history] = {
                "reach_p0": torch.zeros(NUM_FULL_DEALS, device=self.device),
                "reach_p1": torch.zeros(NUM_FULL_DEALS, device=self.device),
                "values": torch.zeros(NUM_FULL_DEALS, device=self.device),
                "has_board": has_board,
            }

        history_data[history]["reach_p0"][fd_idx] = reach_p0
        history_data[history]["reach_p1"][fd_idx] = reach_p1

        node_value = 0.0
        for action in actions:
            p = action_probs.get(action, 1.0 / len(actions))
            child = game.next_state(state, action)

            if player == 0:
                child_value = self._traverse_deal(
                    child, fd_idx, board_card, profile,
                    reach_p0 * p, reach_p1, history_data
                )
            else:
                child_value = self._traverse_deal(
                    child, fd_idx, board_card, profile,
                    reach_p0, reach_p1 * p, history_data
                )
            node_value += p * child_value

        history_data[history]["values"][fd_idx] = node_value
        return node_value


# Helper for mapping full deal index to private deal index
from leduc.belief_state import FULL_TO_PRIVATE

def _full_to_private_idx(fd_idx: int) -> int:
    return FULL_TO_PRIVATE[fd_idx]
