"""ReBeL training loop for HUNL Hold'em.

Implements the ReBeL algorithm scaled to HUNL:

  for each iteration:
    1. Sample a random public state (board + action history)
    2. Construct PBS at that state (factored beliefs over 1326 hands)
    3. Run depth-limited CFR with value network at leaf nodes
    4. Collect (PBS, public_features, values) training pairs
    5. Train value network on accumulated data
    6. Measure loss decrease

The key challenge vs Kuhn/Leduc: we cannot enumerate the full game tree.
Instead, we sample public states and solve only the subgame rooted at
each sampled state. The value network generalizes across states.

This module provides:
- HUNLRebelConfig: hyperparameters for the training loop
- HUNLSubgameSolver: depth-limited CFR solver for HUNL subgames
- HUNLRebelTrainer: orchestrates the full training pipeline
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

import poker_engine as pe

from hunl.game import (
    ALL_HOLE_COMBOS,
    COMBO_INDEX,
    HUNLAction,
    HUNLConfig,
    HUNLGame,
    HUNLState,
    NUM_COMBOS,
    hands_for_board,
)
from hunl.belief_state import (
    HUNLBeliefState,
    pbs_from_reach_probs_fast,
)
from hunl.value_net import (
    HUNLValueNetConfig,
    HUNLValueNetwork,
    encode_public_features,
    encode_state_features,
    STREET_MAP,
)


@dataclass
class HUNLRebelConfig:
    """Configuration for HUNL ReBeL training."""
    # Training loop
    num_iters: int = 100
    cfr_iters_per_solve: int = 200
    samples_per_iter: int = 10

    # Value network
    value_hidden_dim: int = 512
    value_num_layers: int = 6
    value_lr: float = 1e-3
    value_train_epochs: int = 50
    value_batch_size: int = 64

    # CFR subgame solving
    max_depth: int = 4  # depth limit for CFR (number of actions)
    use_cfr_plus: bool = True

    # Replay buffer
    replay_buffer_size: int = 50000

    # Sampling
    sample_streets: Tuple[str, ...] = ("preflop", "flop", "turn", "river")
    device: str = "cpu"


class HUNLReplayBuffer:
    """Replay buffer storing (PBS, public_features, values) training triples."""

    def __init__(self, max_size: int = 50000) -> None:
        self._pbs: deque = deque(maxlen=max_size)
        self._features: deque = deque(maxlen=max_size)
        self._values: deque = deque(maxlen=max_size)

    def add(
        self,
        pbs: torch.Tensor,
        public_features: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Add a training triple."""
        self._pbs.append(pbs.detach().cpu().flatten())
        self._features.append(public_features.detach().cpu())
        self._values.append(values.detach().cpu().flatten())

    def to_dataset(self) -> Dict[str, torch.Tensor]:
        if not self._pbs:
            return {
                "pbs": torch.empty(0),
                "features": torch.empty(0),
                "values": torch.empty(0),
            }
        return {
            "pbs": torch.stack(list(self._pbs)),
            "features": torch.stack(list(self._features)),
            "values": torch.stack(list(self._values)),
        }

    def __len__(self) -> int:
        return len(self._pbs)


def sample_random_board(street: str) -> Tuple[int, ...]:
    """Sample a random board for a given street.

    Args:
        street: 'preflop', 'flop', 'turn', or 'river'

    Returns:
        Tuple of card IDs for the board
    """
    deck = list(range(52))
    random.shuffle(deck)

    if street == "preflop":
        return ()
    elif street == "flop":
        return tuple(sorted(deck[:3]))
    elif street == "turn":
        return tuple(sorted(deck[:4]))
    elif street == "river":
        return tuple(sorted(deck[:5]))
    else:
        raise ValueError(f"Unknown street: {street}")


def sample_random_state(
    game: HUNLGame,
) -> Tuple[HUNLState, Tuple[int, ...]]:
    """Sample a random public state by playing random actions.

    Returns a state at a random point in the game along with the board cards.
    This is used to generate diverse training data for the value network.

    Returns:
        (state, board) tuple
    """
    # Pick a random street to target
    target_street = random.choice(["preflop", "flop", "turn", "river"])

    # Deal random hole cards
    deck = list(range(52))
    random.shuffle(deck)
    h0 = tuple(sorted(deck[:2]))
    h1 = tuple(sorted(deck[2:4]))
    remaining = deck[4:]

    state = game.initial_state()
    state = game.next_state(state, (h0, h1))

    # Play random actions until we reach the target street or terminal
    while not game.is_terminal(state):
        player = game.current_player(state)

        if player == -1:
            # Chance node: deal board cards
            if state.phase == "flop_deal":
                flop = tuple(sorted(remaining[:3]))
                state = game.next_state(state, flop)
                if target_street == "flop":
                    break
            elif state.phase == "turn_deal":
                turn_card = remaining[3]
                state = game.next_state(state, turn_card)
                if target_street == "turn":
                    break
            elif state.phase == "river_deal":
                river_card = remaining[4]
                state = game.next_state(state, river_card)
                if target_street == "river":
                    break
            else:
                break
        else:
            # Player node: play random action
            actions = game.legal_actions(state)
            if not actions:
                break

            # Bias toward check/call to make deeper game trees more likely
            weights = []
            for a in actions:
                if a.action_type in ("check", "call"):
                    weights.append(3.0)
                elif a.action_type == "fold":
                    weights.append(0.5)
                else:
                    weights.append(1.0)

            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            action = random.choices(actions, weights=weights, k=1)[0]
            state = game.next_state(state, action)

            if target_street == "preflop" and state.phase != "preflop":
                break

    return state, state.board


class HUNLSubgameSolver:
    """Depth-limited CFR solver for HUNL subgames.

    Given a public state (board + action history + beliefs), runs CFR
    on the subgame rooted at that state, using the value network to
    evaluate leaf nodes at the depth limit.

    This is the core of ReBeL's search procedure scaled to HUNL.
    Unlike the small-game solver, this operates directly on hand indices
    (0-1325) rather than deal tuples, and uses vectorized operations.
    """

    def __init__(
        self,
        game: HUNLGame,
        state: HUNLState,
        initial_beliefs: torch.Tensor,  # [1326, 2]
        value_fn: Optional[Any] = None,
        max_depth: int = 4,
        cfr_iters: int = 200,
        use_cfr_plus: bool = True,
    ) -> None:
        self.game = game
        self.root_state = state
        self.beliefs = initial_beliefs.clone()
        self.value_fn = value_fn
        self.max_depth = max_depth
        self.cfr_iters = cfr_iters
        self.use_cfr_plus = use_cfr_plus

        # Determine valid hands at this board state
        board = state.board
        board_set = set(board)
        self.valid_hands = []
        self.valid_mask = torch.zeros(NUM_COMBOS, dtype=torch.bool)
        for i, (c0, c1) in enumerate(ALL_HOLE_COMBOS):
            if c0 not in board_set and c1 not in board_set:
                self.valid_hands.append(i)
                self.valid_mask[i] = True

        self.num_valid = len(self.valid_hands)

        # Precompute showdown matrix for fast terminal value computation
        self._precompute_showdown(board)

        # Regret and strategy accumulators keyed by (action_history_hash, player)
        self.regret_sum: Dict[str, torch.Tensor] = {}
        self.strategy_sum: Dict[str, torch.Tensor] = {}
        self.iteration = 0

        # Data collection
        self.collected_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def _precompute_showdown(self, board: Tuple[int, ...]) -> None:
        """Precompute vectorized showdown data for fast terminal value computation.

        Builds two [1326, 1326] matrices using tensor operations:
        - non_conflict_matrix: 1.0 if hands don't share cards, else 0.0
        - showdown_matrix: +1/-1/0 for win/loss/tie (only for river boards)

        Terminal values are then computed as matrix-vector products:
          values[h] = (showdown_matrix[h, :] · reach_opp) / (non_conflict[h, :] · reach_opp) * half_pot
        """
        board_list = list(board)

        # Per-hand card sets for conflict detection (used in _fold_values)
        self._hand_cards = {}
        for h_idx in self.valid_hands:
            self._hand_cards[h_idx] = set(ALL_HOLE_COMBOS[h_idx])

        # Build card membership matrix [1326, 52]: card_matrix[h, c] = 1 if hand h has card c
        card_matrix = torch.zeros(NUM_COMBOS, 52)
        for h_idx in self.valid_hands:
            c0, c1 = ALL_HOLE_COMBOS[h_idx]
            card_matrix[h_idx, c0] = 1.0
            card_matrix[h_idx, c1] = 1.0

        # Non-conflict: hands share zero cards
        # shared_cards[i,j] = card_matrix[i] · card_matrix[j] = count of shared cards
        shared_cards = card_matrix @ card_matrix.T
        self._non_conflict_matrix = (shared_cards == 0).float()

        # Zero out invalid hands (those containing board cards)
        valid_f = self.valid_mask.float()
        self._non_conflict_matrix *= valid_f.unsqueeze(0) * valid_f.unsqueeze(1)

        # Showdown matrix: sign of strength difference, masked by non-conflict
        self._showdown_matrix = torch.zeros(NUM_COMBOS, NUM_COMBOS)
        if len(board) >= 5:
            strengths = torch.zeros(NUM_COMBOS, dtype=torch.long)
            for h_idx in self.valid_hands:
                h = ALL_HOLE_COMBOS[h_idx]
                strengths[h_idx] = pe.py_eval7(list(h) + board_list)

            # Vectorized comparison: sign(s[i] - s[j]) * non_conflict[i,j]
            s_diff = strengths.unsqueeze(1).float() - strengths.unsqueeze(0).float()
            self._showdown_matrix = torch.sign(s_diff) * self._non_conflict_matrix

    def _get_strategy(self, key: str, num_actions: int) -> torch.Tensor:
        """Get current strategy via regret matching."""
        if key not in self.regret_sum:
            self.regret_sum[key] = torch.zeros(num_actions)
            self.strategy_sum[key] = torch.zeros(num_actions)

        positives = torch.clamp(self.regret_sum[key], min=0)
        total = positives.sum()
        if total > 0:
            return positives / total
        return torch.full((num_actions,), 1.0 / num_actions)

    def solve(self) -> Dict[str, Dict[str, float]]:
        """Run CFR and return average strategy profile."""
        # Initial reach probs from beliefs
        reach_p0 = self.beliefs[:, 0].clone()
        reach_p1 = self.beliefs[:, 1].clone()

        for i in range(self.cfr_iters):
            self.iteration = i
            self._cfr(
                self.root_state,
                reach_p0.clone(),
                reach_p1.clone(),
                depth=0,
            )

            if self.use_cfr_plus:
                for key in self.regret_sum:
                    self.regret_sum[key].clamp_(min=0)

        return self._extract_profile()

    def solve_with_values(
        self,
    ) -> Tuple[Dict[str, Dict[str, float]], torch.Tensor]:
        """Run CFR and return both strategy and root PBS values."""
        reach_p0 = self.beliefs[:, 0].clone()
        reach_p1 = self.beliefs[:, 1].clone()
        last_values = None

        for i in range(self.cfr_iters):
            self.iteration = i
            last_values = self._cfr(
                self.root_state,
                reach_p0.clone(),
                reach_p1.clone(),
                depth=0,
            )

        profile = self._extract_profile()

        # Convert per-hand values to PBS values
        pbs_values = torch.zeros(NUM_COMBOS, 2)
        if last_values is not None:
            pbs_values[:, 0] = last_values
            pbs_values[:, 1] = -last_values  # Zero-sum

        return profile, pbs_values

    def _cfr(
        self,
        state: HUNLState,
        reach_p0: torch.Tensor,  # [1326]
        reach_p1: torch.Tensor,  # [1326]
        depth: int,
    ) -> torch.Tensor:
        """CFR traversal returning [1326] values from P0's perspective."""
        game = self.game

        # Terminal node
        if game.is_terminal(state):
            return self._terminal_values(state, reach_p0, reach_p1)

        # Chance node (board deal)
        if state.phase in ("flop_deal", "turn_deal", "river_deal"):
            # For training, we fix the board at root and don't branch over boards
            # This is handled by the outer training loop sampling different boards
            return torch.zeros(NUM_COMBOS)

        # Depth-limited leaf
        if depth >= self.max_depth:
            if self.value_fn is not None:
                return self._evaluate_leaf(state, reach_p0, reach_p1)
            # Without a value net, return zeros (uniform EV assumption)
            return torch.zeros(NUM_COMBOS)

        player = game.current_player(state)
        actions = game.legal_actions(state)
        num_actions = len(actions)

        if num_actions == 0:
            return torch.zeros(NUM_COMBOS)

        # Get strategy for each valid hand
        # In HUNL, all hands at the same public state share the same infoset
        # (since the public history is the same), so we use a single strategy
        # per public state. The hand-specific part comes from the reach probs.
        history_key = ",".join(state.action_history)
        strategy = self._get_strategy(history_key, num_actions)

        # Compute per-action values
        action_values = torch.zeros(num_actions, NUM_COMBOS)
        for a_idx, action in enumerate(actions):
            child = game.next_state(state, action)
            if player == 0:
                action_values[a_idx] = self._cfr(
                    child,
                    reach_p0 * strategy[a_idx],
                    reach_p1,
                    depth + 1,
                )
            else:
                action_values[a_idx] = self._cfr(
                    child,
                    reach_p0,
                    reach_p1 * strategy[a_idx],
                    depth + 1,
                )

        # Node value = strategy-weighted sum of action values
        node_values = (strategy.unsqueeze(1) * action_values).sum(dim=0)

        # Update regrets
        if player == 0:
            opp_reach = reach_p1
        else:
            opp_reach = reach_p0

        # Regret for each action = opp_reach-weighted advantage
        for a_idx in range(num_actions):
            if player == 0:
                advantage = action_values[a_idx] - node_values
            else:
                advantage = node_values - action_values[a_idx]

            # Sum opponent-reach-weighted advantage over valid hands
            regret = (opp_reach * advantage * self.valid_mask.float()).sum().item()
            self.regret_sum[history_key][a_idx] += regret

        # Accumulate strategy
        player_reach = reach_p0 if player == 0 else reach_p1
        p_r = (player_reach * self.valid_mask.float()).sum().item()
        self.strategy_sum[history_key] += p_r * strategy

        # Collect training data at this node
        if self.iteration == self.cfr_iters - 1:
            joint = reach_p0 * reach_p1 * self.valid_mask.float()
            if joint.sum() > 1e-10:
                pbs = torch.zeros(NUM_COMBOS, 2)
                pbs[:, 0] = reach_p0 * self.valid_mask.float()
                pbs[:, 1] = reach_p1 * self.valid_mask.float()
                for p in range(2):
                    total = pbs[:, p].sum()
                    if total > 0:
                        pbs[:, p] /= total

                pbs_values = torch.zeros(NUM_COMBOS, 2)
                pbs_values[:, 0] = node_values
                pbs_values[:, 1] = -node_values

                features = encode_state_features(state)
                self.collected_data.append((pbs, features, pbs_values))

        return node_values

    def _terminal_values(
        self,
        state: HUNLState,
        reach_p0: torch.Tensor,
        reach_p1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute terminal values for all hands.

        For a fold, the folder loses their contribution.
        For a showdown, uses precomputed showdown matrix for O(n) per hand.
        """
        rs = state.rust_state

        # Check if this is a fold
        if state.action_history and state.action_history[-1].startswith("fold"):
            return self._fold_values(state)

        # Showdown: vectorized computation using precomputed matrices
        # values[h0] = (showdown_matrix[h0, :] · reach_p1) / (non_conflict[h0, :] · reach_p1) * half_pot
        pot = rs.pot
        half_pot = pot / 2.0

        # Matrix-vector products for all hands simultaneously
        ev_vec = self._showdown_matrix @ reach_p1       # [1326] weighted win/loss
        opp_total = self._non_conflict_matrix @ reach_p1  # [1326] opponent total reach

        values = torch.zeros(NUM_COMBOS)
        mask = opp_total > 1e-10
        values[mask] = (ev_vec[mask] / opp_total[mask]) * half_pot

        return values

    def _fold_values(self, state: HUNLState) -> torch.Tensor:
        """Values when one player folds."""
        rs = state.rust_state
        values = torch.zeros(NUM_COMBOS)

        # Determine who folded from terminal utility
        util_p0 = rs.terminal_utility(0)
        # All hands get the same value (fold is hand-independent)
        values[self.valid_mask] = util_p0

        return values

    def _evaluate_leaf(
        self,
        state: HUNLState,
        reach_p0: torch.Tensor,
        reach_p1: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate a leaf node using the value network."""
        # Construct PBS from reach probs
        pbs = torch.zeros(NUM_COMBOS, 2)
        pbs[:, 0] = reach_p0 * self.valid_mask.float()
        pbs[:, 1] = reach_p1 * self.valid_mask.float()
        for p in range(2):
            total = pbs[:, p].sum()
            if total > 0:
                pbs[:, p] /= total

        features = encode_state_features(state)

        # Query value network
        pbs_flat = pbs.flatten().unsqueeze(0)
        features_batch = features.unsqueeze(0)
        with torch.no_grad():
            pred = self.value_fn(pbs_flat, features_batch).squeeze(0)

        # Reshape to [1326, 2] and return P0 values
        pred_reshaped = pred.view(NUM_COMBOS, 2)
        return pred_reshaped[:, 0]

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

            # Map action indices back to action labels
            # We need the state to know the action labels
            profile[key] = {str(i): avg[i].item() for i in range(len(avg))}

        return profile


class HUNLRebelTrainer:
    """Full ReBeL training pipeline for HUNL Hold'em.

    Orchestrates the self-play + value network training loop:
    1. Sample random public states
    2. Solve subgames at those states using depth-limited CFR
    3. Collect training data from the solutions
    4. Train the value network
    5. Repeat
    """

    def __init__(
        self,
        config: Optional[HUNLRebelConfig] = None,
        game_config: Optional[HUNLConfig] = None,
    ) -> None:
        self.config = config or HUNLRebelConfig()
        self.game = HUNLGame(game_config)

        # Value network
        vnet_config = HUNLValueNetConfig(
            hidden_dim=self.config.value_hidden_dim,
            num_layers=self.config.value_num_layers,
        )
        self.value_net = HUNLValueNetwork(vnet_config)
        self.value_net.to(torch.device(self.config.device))

        # Replay buffer
        self.replay_buffer = HUNLReplayBuffer(
            max_size=self.config.replay_buffer_size,
        )

        # Training history
        self.value_losses: List[float] = []
        self.iter_stats: List[Dict] = []

    def _value_fn(
        self, pbs: torch.Tensor, features: torch.Tensor,
    ) -> torch.Tensor:
        """Value function wrapper for leaf evaluation."""
        self.value_net.eval()
        with torch.no_grad():
            pred = self.value_net(pbs, features)
        return pred

    def train(self, num_iters: Optional[int] = None) -> Dict[str, List]:
        """Run the ReBeL training loop.

        Args:
            num_iters: override config.num_iters

        Returns:
            Training metrics dict.
        """
        n_iters = num_iters or self.config.num_iters

        for iteration in range(n_iters):
            iter_data_count = 0

            # Phase 1: Sample and solve subgames
            solved_count = 0
            max_attempts = self.config.samples_per_iter * 10
            attempts = 0
            while solved_count < self.config.samples_per_iter and attempts < max_attempts:
                attempts += 1
                state, board = sample_random_state(self.game)

                if self.game.is_terminal(state):
                    continue

                if state.phase in ("flop_deal", "turn_deal", "river_deal"):
                    continue

                solved_count += 1

                # Construct initial beliefs (uniform over valid hands)
                beliefs = HUNLBeliefState(board=board)

                # Solve subgame
                solver = HUNLSubgameSolver(
                    game=self.game,
                    state=state,
                    initial_beliefs=beliefs.get_pbs(),
                    value_fn=self._value_fn if iteration > 0 else None,
                    max_depth=self.config.max_depth,
                    cfr_iters=self.config.cfr_iters_per_solve,
                    use_cfr_plus=self.config.use_cfr_plus,
                )

                solver.solve()

                # Collect training data
                for pbs, features, values in solver.collected_data:
                    self.replay_buffer.add(pbs, features, values)
                    iter_data_count += 1

            # Phase 2: Train value network
            loss = self._train_value_net()
            self.value_losses.append(loss)

            self.iter_stats.append({
                "iteration": iteration,
                "data_points": iter_data_count,
                "buffer_size": len(self.replay_buffer),
                "loss": loss,
            })

        return {
            "value_losses": self.value_losses,
            "iter_stats": self.iter_stats,
        }

    def _train_value_net(self) -> float:
        """Train value network on replay buffer data."""
        dataset = self.replay_buffer.to_dataset()
        if len(dataset["pbs"]) == 0:
            return float("inf")

        pbs_data = dataset["pbs"]
        features_data = dataset["features"]
        values_data = dataset["values"]

        n = len(pbs_data)
        device = torch.device(self.config.device)

        optimizer = torch.optim.Adam(
            self.value_net.parameters(),
            lr=self.config.value_lr,
        )
        loss_fn = nn.MSELoss()

        self.value_net.train()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.config.value_train_epochs):
            perm = torch.randperm(n)

            for start in range(0, n, self.config.value_batch_size):
                end = min(start + self.config.value_batch_size, n)
                idx = perm[start:end]

                batch_pbs = pbs_data[idx].to(device)
                batch_features = features_data[idx].to(device)
                batch_values = values_data[idx].to(device)

                pred = self.value_net(batch_pbs, batch_features)
                loss = loss_fn(pred, batch_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)
