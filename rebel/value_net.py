"""ReBeL Value and Policy Networks for Kuhn Poker.

These are the neural network components that will eventually replace
CFR's tabular solution. In ReBeL:

1. The value network V(PBS) predicts expected payoffs for each possible
   private state per player
2. The policy network pi(PBS, private_info) predicts action probabilities

For Kuhn Poker, these are small networks suitable for validating the
approach before scaling to larger games.

The PBS is represented as [NUM_PRIVATE_STATES, NUM_PLAYERS] = [3, 2],
which is flattened to PBS_DIM = 6 for network input/output.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from kuhn.belief_state import NUM_PRIVATE_STATES, NUM_PLAYERS

# Flattened PBS dimension for network I/O
PBS_DIM = NUM_PRIVATE_STATES * NUM_PLAYERS  # 6 for Kuhn


class ValueNetwork(nn.Module):
    """Predicts counterfactual values from a public belief state.

    Input: PBS flattened to R^{PBS_DIM} (flattened [NUM_PRIVATE_STATES, NUM_PLAYERS])
    Output: V in R^{PBS_DIM} (expected value for each private state per player)

    Architecture is deliberately simple for Kuhn Poker.
    For larger games, this would be much deeper.
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PBS_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, PBS_DIM),
        )

    def forward(self, pbs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pbs: [batch, PBS_DIM] or [PBS_DIM] (flattened PBS)
        Returns:
            values: same shape as input
        """
        return self.net(pbs)


class PolicyNetwork(nn.Module):
    """Predicts action probabilities from PBS + private card information.

    Input: [PBS flattened in R^{PBS_DIM}, card_onehot in R^{NUM_PRIVATE_STATES}]
    Output: action_probs in R^{max_actions}

    The card encoding is a one-hot vector indicating which card the
    acting player holds. Combined with the PBS, this gives the full
    information needed for a decision.
    """

    MAX_ACTIONS = 2  # Kuhn Poker always has exactly 2 actions

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        input_dim = PBS_DIM + NUM_PRIVATE_STATES
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.MAX_ACTIONS),
        )

    def forward(
        self, pbs: torch.Tensor, card_onehot: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            pbs: [batch, PBS_DIM] (flattened PBS)
            card_onehot: [batch, NUM_PRIVATE_STATES]
        Returns:
            action_probs: [batch, MAX_ACTIONS] (softmax probabilities)
        """
        x = torch.cat([pbs, card_onehot], dim=-1)
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


def train_value_network(
    value_net: ValueNetwork,
    dataset: Dict[str, torch.Tensor],
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> list[float]:
    """Train the value network on CFR-generated data.

    Args:
        value_net: the value network to train
        dataset: output of RebelDataLogger.to_dataset()
        epochs: number of training epochs
        lr: learning rate
        batch_size: mini-batch size

    Returns:
        List of training losses per epoch
    """
    beliefs = dataset["beliefs"]  # [N, PBS_DIM]
    values = dataset["values"]    # [N, PBS_DIM]

    if len(beliefs) == 0:
        return []

    optimizer = torch.optim.Adam(value_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []

    n = len(beliefs)
    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(n)
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]

            pred = value_net(beliefs[idx])
            loss = loss_fn(pred, values[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)

    return losses
