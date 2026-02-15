"""ReBeL training loop (game-agnostic).

Implements the complete ReBeL algorithm (Brown & Sandholm, 2020):

  for each iteration:
    1. Sample a public state (or use root)
    2. Build PBS at that state
    3. Run depth-limited CFR with value net V at leaves
    4. Collect (PBS, CFR_values) training pairs from the solve
    5. Add pairs to replay buffer
    6. Train V on replay buffer
    7. Evaluate exploitability of the resulting strategy

The key difference from plain CFR: the value network learns to predict
counterfactual values from PBS, and feeds back into CFR as a leaf evaluator.
This creates a self-improving loop where the value net bootstraps better
CFR solves, which produce better training data for the value net.

The trainer is game-agnostic: it accepts any Game + BeliefConfig pair.
Game-specific details (deal structures, terminal histories, infoset keys)
are handled by the game and config objects.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from belief_config import BeliefConfig, belief_config_from_game
from cfr.solver import CFRTrainer
from game_interface import Game
from rebel.value_net import ValueNetwork, train_value_network
from rebel.endgame_solver import SubgameSolver, SubgameConfig


@dataclass
class RebelConfig:
    """Configuration for the ReBeL training loop."""
    num_iters: int = 100
    cfr_iters_per_solve: int = 200
    learning_rate: float = 1e-3
    batch_size: int = 32
    value_hidden_dim: int = 64
    value_train_epochs: int = 100
    replay_buffer_size: int = 10000
    max_depth: Optional[int] = None  # None = full solve (no depth limit)
    device: str = "cpu"


class ReplayBuffer:
    """Fixed-size replay buffer for (PBS, values) training pairs."""

    def __init__(self, max_size: int = 10000) -> None:
        self._beliefs: deque = deque(maxlen=max_size)
        self._values: deque = deque(maxlen=max_size)

    def add(self, belief: torch.Tensor, values: torch.Tensor) -> None:
        """Add a (PBS, values) pair. Both should be [num_private_states, num_players]."""
        self._beliefs.append(belief.detach().clone().flatten())
        self._values.append(values.detach().clone().flatten())

    def to_dataset(self) -> Dict[str, torch.Tensor]:
        """Convert buffer to tensors for training."""
        if not self._beliefs:
            return {"beliefs": torch.empty(0), "values": torch.empty(0)}
        return {
            "beliefs": torch.stack(list(self._beliefs)),
            "values": torch.stack(list(self._values)),
        }

    def __len__(self) -> int:
        return len(self._beliefs)


class RebelTrainer:
    """Full ReBeL training pipeline for any imperfect-information game.

    Orchestrates the self-play, data collection, and neural network
    training loop. The value network is used as a leaf evaluator
    during CFR, creating the feedback loop that is the core of ReBeL.

    Args:
        game: Any game implementing the Game protocol
        belief_config: BeliefConfig for the game (or auto-derived if not provided)
        config: RebelConfig with hyperparameters
    """

    def __init__(
        self,
        game: Optional[Game] = None,
        belief_config: Optional[BeliefConfig] = None,
        config: Optional[RebelConfig] = None,
        # Legacy kwargs for backward compatibility with existing tests
        value_hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        cfr_iterations: int = 100,
        device: str = "cpu",
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = RebelConfig(
                value_hidden_dim=value_hidden_dim,
                learning_rate=learning_rate,
                cfr_iters_per_solve=cfr_iterations,
                device=device,
            )

        # Default to Kuhn if no game provided (backward compat)
        if game is None:
            from kuhn.game import KuhnPoker
            game = KuhnPoker()

        self.game = game

        if belief_config is not None:
            self.belief_config = belief_config
        else:
            self.belief_config = belief_config_from_game(self.game)

        self.device = torch.device(self.config.device)

        # Value network: V(PBS) -> counterfactual values
        bc = self.belief_config
        self.value_net = ValueNetwork(
            pbs_dim=bc.pbs_dim,
            hidden_dim=self.config.value_hidden_dim,
        ).to(self.device)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=self.config.replay_buffer_size)

        # Training history
        self.value_losses: List[float] = []
        self.exploitabilities: List[float] = []

    def _value_fn(self, pbs: torch.Tensor) -> torch.Tensor:
        """Value function for leaf evaluation during CFR.

        Takes PBS [num_private_states, num_players], returns values in same shape.
        """
        self.value_net.eval()
        with torch.no_grad():
            flat = pbs.flatten().unsqueeze(0).to(self.device)
            pred = self.value_net(flat).squeeze(0)
        bc = self.belief_config
        return pred.view(bc.num_private_states, bc.num_players).cpu()

    def train(
        self,
        num_epochs: int = 10,
        cfr_iters_per_epoch: int = 100,
        value_train_epochs: int = 50,
    ) -> Dict[str, List[float]]:
        """Run the full ReBeL training loop.

        Args:
            num_epochs: number of ReBeL iterations (outer loop)
            cfr_iters_per_epoch: CFR iterations per subgame solve
            value_train_epochs: gradient steps per value net training phase

        Returns:
            Training metrics dict with 'value_losses' and 'exploitabilities'.
        """
        bc = self.belief_config
        cfr_iters = cfr_iters_per_epoch or self.config.cfr_iters_per_solve

        for epoch in range(num_epochs):
            # Phase 1: Run CFR solve (with value net at leaves if depth-limited)
            beliefs = bc.initial_pbs(str(self.device))

            value_fn = self._value_fn if self.config.max_depth is not None else None
            solver_config = SubgameConfig(
                iterations=cfr_iters,
                max_depth=self.config.max_depth,
            )
            solver = SubgameSolver(
                game=self.game,
                belief_config=bc,
                root_history="",
                initial_beliefs=beliefs,
                value_fn=value_fn,
                config=solver_config,
            )
            profile, root_values = solver.solve_with_values()

            # Collect training data from all decision points in this solve
            self._collect_training_data(solver, cfr_iters)

            # Phase 2: Train value network on replay buffer
            dataset = self.replay_buffer.to_dataset()
            if len(dataset["beliefs"]) > 0:
                losses = train_value_network(
                    self.value_net,
                    dataset,
                    epochs=value_train_epochs,
                    lr=self.config.learning_rate,
                    batch_size=self.config.batch_size,
                )
                if losses:
                    self.value_losses.append(losses[-1])

            # Phase 3: Evaluate exploitability of the strategy
            dummy = CFRTrainer(self.game)
            br0 = dummy._best_response_value(profile, 0)
            br1 = dummy._best_response_value(profile, 1)
            exp = 0.5 * (br0 + br1)
            self.exploitabilities.append(exp)

        return {
            "value_losses": self.value_losses,
            "exploitabilities": self.exploitabilities,
        }

    def _collect_training_data(
        self, solver: SubgameSolver, cfr_iters: int,
    ) -> None:
        """Collect (PBS, values) pairs from a completed CFR solve.

        Runs a single forward pass through the tree using the solver's
        converged average strategy.
        """
        game = self.game
        bc = self.belief_config
        initial = game.initial_state()

        ones = torch.ones(bc.num_deals)
        deal_states = [game.next_state(initial, deal) for deal in solver.all_deals]

        self._traverse_for_data(
            solver, deal_states, ones.clone(), ones.clone(), depth=0,
        )

    def _traverse_for_data(
        self,
        solver: SubgameSolver,
        states: list,
        reach_p0: torch.Tensor,
        reach_p1: torch.Tensor,
        depth: int,
    ) -> torch.Tensor:
        """Traverse the solved game tree to collect training data.

        Uses the solver's average strategy to compute expected values,
        then logs (PBS, values) pairs at each decision point.
        """
        game = solver.game
        bc = solver.belief_config

        if game.is_terminal(states[0]):
            v = torch.zeros(bc.num_deals)
            for d_idx, state in enumerate(states):
                v[d_idx] = game.terminal_utility(state, 0)
            return v

        # If depth-limited and at leaf, use value net
        if solver.config.max_depth is not None and depth >= solver.config.max_depth:
            if solver.value_fn is not None:
                pbs = bc.reach_to_pbs(reach_p0, reach_p1)
                pbs_values = solver.value_fn(pbs)
                return bc.pbs_values_to_deal_values(pbs_values)

        player = game.current_player(states[0])
        actions = game.legal_actions(states[0])
        num_actions = len(actions)
        profile = solver._extract_profile()

        # Group deals by infoset
        infoset_deals: Dict[str, List[int]] = {}
        for d_idx, state in enumerate(states):
            key = game.infoset_key(state, player)
            infoset_deals.setdefault(key, []).append(d_idx)

        # Compute strategy per deal from average strategy
        strategy_per_deal = []
        for a_idx, action in enumerate(actions):
            s = torch.zeros(bc.num_deals)
            for key, deal_indices in infoset_deals.items():
                action_probs = profile.get(key, {})
                prob = action_probs.get(action, 1.0 / num_actions)
                for d_idx in deal_indices:
                    s[d_idx] = prob
            strategy_per_deal.append(s)

        # Recurse to get action values
        action_values = []
        for a_idx, action in enumerate(actions):
            child_states = [game.next_state(state, action) for state in states]
            s = strategy_per_deal[a_idx]
            if player == 0:
                child_vals = self._traverse_for_data(
                    solver, child_states, reach_p0 * s, reach_p1, depth + 1,
                )
            else:
                child_vals = self._traverse_for_data(
                    solver, child_states, reach_p0, reach_p1 * s, depth + 1,
                )
            action_values.append(child_vals)

        action_values_t = torch.stack(action_values)
        strategy_t = torch.stack(strategy_per_deal)
        node_values = (strategy_t * action_values_t).sum(dim=0)

        # Log training data for this decision point
        joint = reach_p0 * reach_p1
        if joint.sum() > 1e-10:
            pbs = bc.reach_to_pbs(reach_p0, reach_p1)
            pbs_values = bc.deal_values_to_pbs_values(node_values)
            self.replay_buffer.add(pbs, pbs_values)

        return node_values

    def run_full_training(self) -> Dict[str, List[float]]:
        """Run the complete ReBeL training loop using config parameters.

        This is the primary entry point for the full training pipeline.
        """
        return self.train(
            num_epochs=self.config.num_iters,
            cfr_iters_per_epoch=self.config.cfr_iters_per_solve,
            value_train_epochs=self.config.value_train_epochs,
        )

    def get_strategy(self) -> Dict[str, Dict[str, float]]:
        """Get the current best strategy by running a final CFR solve."""
        bc = self.belief_config
        beliefs = bc.initial_pbs(str(self.device))
        value_fn = self._value_fn if self.config.max_depth is not None else None
        solver_config = SubgameConfig(
            iterations=self.config.cfr_iters_per_solve,
            max_depth=self.config.max_depth,
        )
        solver = SubgameSolver(
            game=self.game,
            belief_config=bc,
            root_history="",
            initial_beliefs=beliefs,
            value_fn=value_fn,
            config=solver_config,
        )
        return solver.solve()
