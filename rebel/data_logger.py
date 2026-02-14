"""Data logger for ReBeL-style training.

During CFR iterations, we log the data needed to train ReBeL's
value and policy networks:

1. Public Belief States (PBS) at each decision point
2. The CFR strategy (policy) at each PBS
3. The counterfactual values at each PBS
4. The reach probabilities that produced the PBS

This data can then be used to train:
  - V(PBS) -> expected value for each player given the belief state
  - π(PBS) -> policy (action probabilities) given the belief state

The key ReBeL insight: instead of training on information states,
we train on public belief states. This allows the value network
to generalize across different private information configurations.
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
        belief: [NUM_DEALS] PBS tensor
        reach_p0: [NUM_DEALS] player 0 reach probabilities
        reach_p1: [NUM_DEALS] player 1 reach probabilities
        strategy: dict mapping infoset_key -> [num_actions] strategy tensor
        values_p0: [NUM_DEALS] counterfactual values for player 0
        iteration: CFR iteration when this was recorded
    """
    history: str
    belief: torch.Tensor
    reach_p0: torch.Tensor
    reach_p1: torch.Tensor
    strategy: Dict[str, torch.Tensor]
    values_p0: torch.Tensor
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
        values_p0: torch.Tensor,
        iteration: int,
    ) -> None:
        """Log a single PBS data point."""
        self.data.append(PBSDataPoint(
            history=history,
            belief=belief.detach().clone(),
            reach_p0=reach_p0.detach().clone(),
            reach_p1=reach_p1.detach().clone(),
            strategy={k: v.detach().clone() for k, v in strategy.items()},
            values_p0=values_p0.detach().clone(),
            iteration=iteration,
        ))

    def to_dataset(self) -> Dict[str, torch.Tensor]:
        """Convert logged data to tensors suitable for training.

        Returns:
            beliefs: [N, NUM_DEALS] — PBS for each logged state
            values: [N, NUM_DEALS] — counterfactual values for player 0
            iterations: [N] — iteration indices
        """
        if not self.data:
            return {"beliefs": torch.empty(0), "values": torch.empty(0)}

        beliefs = torch.stack([d.belief for d in self.data])
        values = torch.stack([d.values_p0 for d in self.data])
        iterations = torch.tensor([d.iteration for d in self.data])

        return {
            "beliefs": beliefs,
            "values": values,
            "iterations": iterations,
        }

    def clear(self) -> None:
        self.data.clear()

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

    The value network learns: V(PBS) -> [values_per_deal]

    In ReBeL, the value network predicts the expected payoff
    for each possible private state (deal), given the public
    belief about what deals are possible.

    For Kuhn Poker with 6 deals, input is PBS ∈ R^6, output is V ∈ R^6.
    """
    pbs: torch.Tensor        # [NUM_DEALS] public belief state
    values: torch.Tensor     # [NUM_DEALS] target values from CFR


@dataclass
class RebelPolicyTarget:
    """Training target for the ReBeL policy network.

    The policy network learns: π(PBS, infoset) -> [action_probs]

    For each information set at a decision point, the policy
    maps the PBS + private info to action probabilities.
    """
    pbs: torch.Tensor                    # [NUM_DEALS] public belief state
    infoset_key: str                     # information set identifier
    action_probs: torch.Tensor           # [num_actions] target policy
