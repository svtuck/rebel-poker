"""ReBeL Value and Policy Networks for Kuhn Poker.

These are the neural network components that will eventually replace
CFR's tabular solution. In ReBeL:

1. The value network V(PBS) predicts expected payoffs for each possible deal
2. The policy network π(PBS, private_info) predicts action probabilities

For Kuhn Poker, these are small networks suitable for validating the
approach before scaling to larger games.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from kuhn.belief_state import NUM_DEALS


class ValueNetwork(nn.Module):
    """Predicts counterfactual values from a public belief state.

    Input: PBS ∈ R^{NUM_DEALS} (probability distribution over deals)
    Output: V ∈ R^{NUM_DEALS} (expected value for each deal, player 0's perspective)

    Architecture is deliberately simple for Kuhn Poker.
    For larger games, this would be much deeper.
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_DEALS, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_DEALS),
        )

    def forward(self, pbs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pbs: [batch, NUM_DEALS] or [NUM_DEALS]
        Returns:
            values: same shape as input
        """
        return self.net(pbs)


class PolicyNetwork(nn.Module):
    """Predicts action probabilities from PBS + private card information.

    Input: [PBS ∈ R^{NUM_DEALS}, card_onehot ∈ R^{NUM_CARDS}]
    Output: action_probs ∈ R^{max_actions}

    The card encoding is a one-hot vector indicating which card the
    acting player holds. Combined with the PBS, this gives the full
    information needed for a decision.
    """

    NUM_CARDS = 3
    MAX_ACTIONS = 2  # Kuhn Poker always has exactly 2 actions

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        input_dim = NUM_DEALS + self.NUM_CARDS
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
            pbs: [batch, NUM_DEALS]
            card_onehot: [batch, NUM_CARDS]
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
    beliefs = dataset["beliefs"]  # [N, NUM_DEALS]
    values = dataset["values"]    # [N, NUM_DEALS]

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
