"""Data logger for ReBeL-style training (game-agnostic).

During CFR iterations, we log the data needed to train ReBeL's
value and policy networks:

1. Public Belief States (PBS) at each decision point
2. The CFR strategy (policy) at each PBS
3. The counterfactual values at each PBS
4. The reach probabilities that produced the PBS

This data can then be used to train:
  - V(PBS) -> expected value for each private state per player
  - pi(PBS) -> policy (action probabilities) given the belief state

The key ReBeL insight: instead of training on information states,
we train on public belief states. This allows the value network
to generalize across different private information configurations.

PBS is represented as [num_private_states, num_players] — the exact
dimensions depend on the game (configured via BeliefConfig).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class PBSDataPoint:
    """A single training data point for ReBeL.

    Attributes:
        history: public action history string
        belief: [num_private_states, num_players] factored PBS tensor
        reach_p0: [num_deals] player 0 reach probabilities
        reach_p1: [num_deals] player 1 reach probabilities
        strategy: dict mapping infoset_key -> [num_actions] strategy tensor
        values: [num_private_states, num_players] counterfactual values
        iteration: CFR iteration when this was recorded
    """
    history: str
    belief: torch.Tensor
    reach_p0: torch.Tensor
    reach_p1: torch.Tensor
    strategy: Dict[str, torch.Tensor]
    values: torch.Tensor
    iteration: int


class RebelDataLogger:
    """Logs PBS data during CFR for later neural network training.

    Usage:
        logger = RebelDataLogger()
        # During CFR iteration:
        logger.log_state(history, belief, reach_p0, reach_p1, strategy, values, iter)
        # After CFR:
        dataset = logger.to_dataset()
    """

    def __init__(self) -> None:
        self.data: List[PBSDataPoint] = []

    def log_state(
        self,
        history: str,
        belief: torch.Tensor,
        reach_p0: torch.Tensor,
        reach_p1: torch.Tensor,
        strategy: Dict[str, torch.Tensor],
        values: torch.Tensor,
        iteration: int,
    ) -> None:
        """Log a single PBS data point."""
        self.data.append(PBSDataPoint(
            history=history,
            belief=belief.detach().clone(),
            reach_p0=reach_p0.detach().clone(),
            reach_p1=reach_p1.detach().clone(),
            strategy={k: v.detach().clone() for k, v in strategy.items()},
            values=values.detach().clone(),
            iteration=iteration,
        ))

    def to_dataset(self) -> Dict[str, torch.Tensor]:
        """Convert logged data to tensors suitable for training.

        Returns:
            beliefs: [N, pbs_dim] -- flattened PBS for each logged state
            values: [N, pbs_dim] -- flattened counterfactual values
            iterations: [N] -- iteration indices
        """
        if not self.data:
            return {"beliefs": torch.empty(0), "values": torch.empty(0)}

        # Flatten [num_private_states, num_players] -> [pbs_dim] for network input
        beliefs = torch.stack([d.belief.flatten() for d in self.data])
        values = torch.stack([d.values.flatten() for d in self.data])
        iterations = torch.tensor([d.iteration for d in self.data])

        return {
            "beliefs": beliefs,
            "values": values,
            "iterations": iterations,
        }

    def clear(self) -> None:
        self.data.clear()

    def __len__(self) -> int:
        return len(self.data)

    def summary(self) -> str:
        """Human-readable summary of logged data."""
        if not self.data:
            return "No data logged."

        histories = set(d.history for d in self.data)
        iters = set(d.iteration for d in self.data)
        return (
            f"Logged {len(self.data)} data points\n"
            f"  Unique histories: {sorted(histories)}\n"
            f"  Iteration range: {min(iters)} - {max(iters)}\n"
        )


@dataclass
class RebelValueTarget:
    """Training target for the ReBeL value network.

    The value network learns: V(PBS) -> [values_per_private_state_per_player]
    """
    pbs: torch.Tensor        # [num_private_states, num_players] public belief state
    values: torch.Tensor     # [num_private_states, num_players] target values from CFR


@dataclass
class RebelPolicyTarget:
    """Training target for the ReBeL policy network.

    The policy network learns: pi(PBS, infoset) -> [action_probs]
    """
    pbs: torch.Tensor                    # [num_private_states, num_players] public belief state
    infoset_key: str                     # information set identifier
    action_probs: torch.Tensor           # [num_actions] target policy
