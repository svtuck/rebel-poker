"""ReBeL Value and Policy Networks (game-agnostic).

These are the neural network components for ReBeL's search procedure:

1. The value network V(PBS) predicts expected payoffs for each possible
   private state per player
2. The policy network pi(PBS, private_info) predicts action probabilities

Network dimensions are parameterized by pbs_dim and num_private_states,
which come from the game's BeliefConfig. This allows the same network
architecture to work with Kuhn (pbs_dim=6), Leduc (pbs_dim=12), or
larger games.

PBS layout: [num_private_states, num_players] flattened to [pbs_dim]
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """Predicts counterfactual values from a public belief state.

    Input: PBS flattened to R^{pbs_dim}
    Output: V in R^{pbs_dim} (expected value for each private state per player)

    Args:
        pbs_dim: Flattened PBS dimension (num_private_states * num_players)
        hidden_dim: Hidden layer width
    """

    def __init__(self, pbs_dim: int = 6, hidden_dim: int = 64) -> None:
        super().__init__()
        self.pbs_dim = pbs_dim
        self.net = nn.Sequential(
            nn.Linear(pbs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pbs_dim),
        )

    def forward(self, pbs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pbs: [batch, pbs_dim] or [pbs_dim] (flattened PBS)
        Returns:
            values: same shape as input
        """
        return self.net(pbs)


class PolicyNetwork(nn.Module):
    """Predicts action probabilities from PBS + private card information.

    Input: [PBS flattened in R^{pbs_dim}, card_onehot in R^{num_private_states}]
    Output: action_probs in R^{max_actions}

    Args:
        pbs_dim: Flattened PBS dimension
        num_private_states: Number of private states per player
        max_actions: Maximum number of actions at any decision point
        hidden_dim: Hidden layer width
    """

    def __init__(
        self,
        pbs_dim: int = 6,
        num_private_states: int = 3,
        max_actions: int = 2,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.pbs_dim = pbs_dim
        self.num_private_states = num_private_states
        self.max_actions = max_actions
        input_dim = pbs_dim + num_private_states
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_actions),
        )

    def forward(
        self, pbs: torch.Tensor, card_onehot: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            pbs: [batch, pbs_dim] (flattened PBS)
            card_onehot: [batch, num_private_states]
        Returns:
            action_probs: [batch, max_actions] (softmax probabilities)
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
    beliefs = dataset["beliefs"]  # [N, pbs_dim]
    values = dataset["values"]    # [N, pbs_dim]

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
