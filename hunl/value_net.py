"""Value network for HUNL ReBeL.

The value network V(PBS, public_features) predicts expected values for each
player's possible hands, given the public belief state. This is used as
the leaf evaluator in depth-limited CFR solving.

Input features:
- PBS: [1326, 2] factored beliefs flattened to [2652]
- Public features: pot size, effective stack, street, board card embeddings
  Total public features: ~20-30 dimensional

Output:
- Values: [1326, 2] predicted counterfactual values per hand per player

Architecture follows the ReBeL paper's design:
- 6-layer MLP with GeLU activation and LayerNorm
- Hidden dim 512 (paper uses 1536 but we scale down for feasibility)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from hunl.game import NUM_COMBOS


@dataclass
class HUNLValueNetConfig:
    """Configuration for the HUNL value network."""
    pbs_dim: int = NUM_COMBOS * 2  # 2652
    public_feature_dim: int = 20   # pot, stack, street, board features
    hidden_dim: int = 512
    num_layers: int = 6
    dropout: float = 0.0
    use_layer_norm: bool = True


class HUNLValueNetwork(nn.Module):
    """Predicts counterfactual values from PBS + public state.

    Architecture:
        [pbs_dim + public_feature_dim] -> LayerNorm -> GeLU MLP -> [pbs_dim]

    The network processes the full PBS (beliefs for all 1326 hands × 2 players)
    plus public game features, and outputs predicted values for each hand
    for each player.
    """

    def __init__(self, config: Optional[HUNLValueNetConfig] = None) -> None:
        super().__init__()
        self.config = config or HUNLValueNetConfig()
        c = self.config

        input_dim = c.pbs_dim + c.public_feature_dim
        output_dim = c.pbs_dim

        layers = []
        current_dim = input_dim

        for i in range(c.num_layers):
            layers.append(nn.Linear(current_dim, c.hidden_dim))
            if c.use_layer_norm:
                layers.append(nn.LayerNorm(c.hidden_dim))
            layers.append(nn.GELU())
            if c.dropout > 0:
                layers.append(nn.Dropout(c.dropout))
            current_dim = c.hidden_dim

        layers.append(nn.Linear(c.hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        pbs: torch.Tensor,
        public_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            pbs: [..., pbs_dim] flattened PBS
            public_features: [..., public_feature_dim] public state features
                If None, zeros are used (backward compat with small games)

        Returns:
            values: [..., pbs_dim] predicted counterfactual values
        """
        if public_features is None:
            public_features = torch.zeros(
                *pbs.shape[:-1], self.config.public_feature_dim,
                device=pbs.device, dtype=pbs.dtype,
            )

        x = torch.cat([pbs, public_features], dim=-1)
        return self.net(x)


def encode_public_features(
    pot: float,
    effective_stack: float,
    street: int,
    board_cards: Tuple[int, ...] = (),
    max_pot: float = 400.0,
    max_stack: float = 200.0,
) -> torch.Tensor:
    """Encode public game state into a fixed-size feature vector.

    Features:
    - pot / max_pot (normalized)
    - effective_stack / max_stack (normalized)
    - street one-hot (4 dims: preflop, flop, turn, river)
    - board card indicators (4 × 13 = 52 binary, but compressed to rank/suit counts)
      Actually we use a simpler encoding: number of board cards per rank (13) and
      suit count (4 suits), total 17 dims for board

    Total: 1 + 1 + 4 + 13 + 1 = 20 dims

    Args:
        pot: current pot size in chips
        effective_stack: min of both players' remaining stacks
        street: 0=preflop, 1=flop, 2=turn, 3=river
        board_cards: tuple of card IDs on the board
        max_pot: normalization factor for pot
        max_stack: normalization factor for stack

    Returns:
        features: [20] tensor
    """
    features = torch.zeros(20)

    # Pot and stack (normalized)
    features[0] = min(pot / max_pot, 2.0)
    features[1] = min(effective_stack / max_stack, 2.0)

    # Street one-hot
    if 0 <= street <= 3:
        features[2 + street] = 1.0

    # Board card rank histogram (13 dims) - count cards per rank
    for card in board_cards:
        rank = card % 13
        features[6 + rank] += 1.0

    # Number of suited board cards (1 dim)
    if board_cards:
        suit_counts = [0] * 4
        for card in board_cards:
            suit_counts[card // 13] += 1
        features[19] = max(suit_counts) / len(board_cards)

    return features


STREET_MAP = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


def encode_state_features(state) -> torch.Tensor:
    """Extract public features from an HUNLState.

    Args:
        state: HUNLState object

    Returns:
        features: [20] tensor of public features
    """
    rs = state.rust_state
    pot = rs.pot
    stacks = rs.stacks
    effective_stack = min(stacks[0], stacks[1])
    street = STREET_MAP.get(rs.street, 0)
    board_cards = state.board

    return encode_public_features(
        pot=pot,
        effective_stack=effective_stack,
        street=street,
        board_cards=board_cards,
    )
