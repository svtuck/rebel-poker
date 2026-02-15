"""Deep CFR: Neural network-based CFR.

Replaces tabular regret/strategy storage with neural networks that
generalize across similar information sets. Based on:
  Noam Brown et al., "Deep Counterfactual Regret Minimization" (ICML 2019)
  https://arxiv.org/abs/1811.00164

Components:
  - Advantage network: maps info set features -> regret estimates per action
  - Strategy network: maps info set features -> average strategy
  - Reservoir sampling for training data collection
  - External-sampling MCCFR with neural regret/strategy approximation

Also includes Single Deep CFR (SD-CFR) as a simplified variant.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from game_interface import Game


# ──────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────

class InfoSetFeaturizer:
    """Converts information set keys to fixed-size feature vectors.

    This base implementation uses a learned embedding table indexed by
    infoset key hash. Override for game-specific featurization.
    """

    def __init__(self, game: Game, device: str = "cpu") -> None:
        self.game = game
        self.device = torch.device(device)
        self._key_to_idx: Dict[str, int] = {}
        self._feature_dim: Optional[int] = None

    def feature_dim(self) -> int:
        """Return the dimension of feature vectors."""
        if self._feature_dim is not None:
            return self._feature_dim
        # Auto-detect by building index from a tree traversal
        self._enumerate_infosets()
        self._feature_dim = len(self._key_to_idx)
        return self._feature_dim

    def _enumerate_infosets(self) -> None:
        """Enumerate all infoset keys by traversing the game tree."""
        game = self.game
        visited = set()

        def traverse(state):
            if state in visited:
                return
            visited.add(state)

            if game.is_terminal(state):
                return
            player = game.current_player(state)
            if player == -1:
                for outcome, _ in game.chance_outcomes(state):
                    traverse(game.next_state(state, outcome))
                return
            key = game.infoset_key(state, player)
            if key not in self._key_to_idx:
                self._key_to_idx[key] = len(self._key_to_idx)
            for action in game.legal_actions(state):
                traverse(game.next_state(state, action))

        traverse(game.initial_state())

    def featurize(self, infoset_key: str) -> torch.Tensor:
        """Convert infoset key to one-hot feature vector."""
        dim = self.feature_dim()
        idx = self._key_to_idx.get(infoset_key)
        features = torch.zeros(dim, device=self.device)
        if idx is not None:
            features[idx] = 1.0
        return features

    def batch_featurize(self, keys: List[str]) -> torch.Tensor:
        """Convert a batch of infoset keys to feature matrix [batch, dim]."""
        return torch.stack([self.featurize(k) for k in keys])


# ──────────────────────────────────────────────────────────────
# Neural networks
# ──────────────────────────────────────────────────────────────

class AdvantageNetwork(nn.Module):
    """Predicts per-action advantage (regret) from info set features.

    Architecture: MLP with configurable hidden layers.
    Input: [batch, feature_dim]
    Output: [batch, num_actions]
    """

    def __init__(self, feature_dim: int, num_actions: int,
                 hidden_sizes: Tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        layers = []
        in_dim = feature_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StrategyNetwork(nn.Module):
    """Predicts average strategy from info set features.

    Architecture: MLP with softmax output.
    Input: [batch, feature_dim]
    Output: [batch, num_actions] (probabilities summing to 1)
    """

    def __init__(self, feature_dim: int, num_actions: int,
                 hidden_sizes: Tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        layers = []
        in_dim = feature_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


# ──────────────────────────────────────────────────────────────
# Reservoir sampling buffer
# ──────────────────────────────────────────────────────────────

class ReservoirBuffer:
    """Reservoir sampling buffer for training data.

    Maintains a fixed-size buffer of training samples. When the buffer
    is full, new samples replace existing ones with probability
    capacity / (num_seen), ensuring uniform sampling from the stream.
    """

    def __init__(self, capacity: int, seed: int = 42) -> None:
        self.capacity = capacity
        self.buffer: List[Tuple[str, int, torch.Tensor, torch.Tensor]] = []
        # Each entry: (infoset_key, iteration, features, target_values)
        self.num_seen = 0
        self.rng = random.Random(seed)

    def add(self, infoset_key: str, iteration: int,
            features: torch.Tensor, targets: torch.Tensor) -> None:
        """Add a sample using reservoir sampling."""
        self.num_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append((infoset_key, iteration, features.detach().cpu(), targets.detach().cpu()))
        else:
            idx = self.rng.randint(0, self.num_seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = (infoset_key, iteration, features.detach().cpu(), targets.detach().cpu())

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of (features, targets, iteration_weights)."""
        n = min(batch_size, len(self.buffer))
        samples = self.rng.sample(self.buffer, n)
        features = torch.stack([s[2] for s in samples])
        targets = torch.stack([s[3] for s in samples])
        iterations = torch.tensor([float(s[1]) for s in samples])
        return features, targets, iterations

    def __len__(self) -> int:
        return len(self.buffer)


# ──────────────────────────────────────────────────────────────
# Deep CFR Trainer
# ──────────────────────────────────────────────────────────────

@dataclass
class DeepCFRConfig:
    """Configuration for Deep CFR."""
    num_traversals: int = 300     # CFR iterations (data generation)
    advantage_buffer_size: int = 200000
    strategy_buffer_size: int = 200000
    advantage_train_steps: int = 2000
    strategy_train_steps: int = 2000
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden_sizes: Tuple[int, ...] = (128, 128)
    seed: int = 42


class DeepCFR:
    """Deep CFR trainer.

    Training loop:
    1. For each iteration t:
       a. Run external-sampling traversal for player 0 (using current advantage net)
       b. Run external-sampling traversal for player 1
       c. Collect (infoset, regret) samples into advantage buffer
       d. Collect (infoset, strategy) samples into strategy buffer
    2. After all iterations:
       a. Train advantage network on advantage buffer (weighted by iteration)
       b. Train strategy network on strategy buffer (weighted by iteration)
    3. The strategy network IS the final policy
    """

    def __init__(self, game: Game, config: Optional[DeepCFRConfig] = None,
                 device: str = "cpu",
                 featurizer: Optional[InfoSetFeaturizer] = None) -> None:
        self.game = game
        self.config = config or DeepCFRConfig()
        self.device = torch.device(device)

        # Feature extraction
        self.featurizer = featurizer or InfoSetFeaturizer(game, device=device)
        feat_dim = self.featurizer.feature_dim()

        # Determine max actions
        self._max_actions = self._find_max_actions()

        # Networks
        self.advantage_net = AdvantageNetwork(
            feat_dim, self._max_actions, self.config.hidden_sizes
        ).to(self.device)
        self.strategy_net = StrategyNetwork(
            feat_dim, self._max_actions, self.config.hidden_sizes
        ).to(self.device)

        # Buffers for each player
        self.advantage_buffers = [
            ReservoirBuffer(self.config.advantage_buffer_size, self.config.seed + p)
            for p in range(2)
        ]
        self.strategy_buffer = ReservoirBuffer(
            self.config.strategy_buffer_size, self.config.seed + 10
        )

        # RNG
        self.rng = random.Random(self.config.seed)

        # Per-infoset data for the current iteration (tabular fallback)
        self._current_regrets: Dict[str, List[float]] = {}
        self._action_counts: Dict[str, int] = {}

        self.iteration = 0

    def _find_max_actions(self) -> int:
        """Find the maximum number of actions across all infosets."""
        game = self.game
        max_a = 0
        visited = set()

        def traverse(state):
            nonlocal max_a
            if state in visited:
                return
            visited.add(state)
            if game.is_terminal(state):
                return
            player = game.current_player(state)
            if player == -1:
                for outcome, _ in game.chance_outcomes(state):
                    traverse(game.next_state(state, outcome))
                return
            actions = game.legal_actions(state)
            max_a = max(max_a, len(actions))
            for action in actions:
                traverse(game.next_state(state, action))

        traverse(game.initial_state())
        return max_a

    def _get_strategy_from_advantages(self, advantages: torch.Tensor,
                                      num_actions: int) -> List[float]:
        """Convert advantages to strategy via regret matching."""
        advs = advantages[:num_actions]
        positives = torch.clamp(advs, min=0)
        total = positives.sum()
        if total > 0:
            strategy = (positives / total).tolist()
        else:
            strategy = [1.0 / num_actions] * num_actions
        return strategy

    def _traverse(self, state, target_player: int, iteration: int,
                  player_reach: float) -> float:
        """External-sampling MCCFR traversal with neural advantage estimation.

        Args:
            state: current game state
            target_player: player whose regrets we're updating
            iteration: current iteration number (for weighting)
            player_reach: target player's reach probability

        Returns: expected value for target_player
        """
        game = self.game

        if game.is_terminal(state):
            return game.terminal_utility(state, target_player)

        player = game.current_player(state)

        if player == -1:
            # Sample chance outcome
            outcomes = game.chance_outcomes(state)
            probs = [p for _, p in outcomes]
            r = self.rng.random()
            cumulative = 0.0
            for i, p in enumerate(probs):
                cumulative += p
                if r <= cumulative:
                    return self._traverse(
                        game.next_state(state, outcomes[i][0]),
                        target_player, iteration, player_reach,
                    )
            return self._traverse(
                game.next_state(state, outcomes[-1][0]),
                target_player, iteration, player_reach,
            )

        actions = game.legal_actions(state)
        num_actions = len(actions)
        key = game.infoset_key(state, player)

        # Get current strategy from advantage network
        features = self.featurizer.featurize(key)
        with torch.no_grad():
            advantages = self.advantage_net(features.unsqueeze(0))[0]
        strategy = self._get_strategy_from_advantages(advantages, num_actions)

        if player == target_player:
            # Enumerate all actions
            action_values = []
            for a_idx, action in enumerate(actions):
                v = self._traverse(
                    game.next_state(state, action),
                    target_player, iteration,
                    player_reach * strategy[a_idx],
                )
                action_values.append(v)

            # Node value
            node_value = sum(s * v for s, v in zip(strategy, action_values))

            # Instantaneous regrets
            regrets = [v - node_value for v in action_values]

            # Pad to max_actions
            padded_regrets = regrets + [0.0] * (self._max_actions - num_actions)
            targets = torch.tensor(padded_regrets, dtype=torch.float32)

            # Add to advantage buffer
            self.advantage_buffers[target_player].add(
                key, iteration, features, targets
            )

            # Add to strategy buffer (weighted by reach * iteration)
            padded_strategy = strategy + [0.0] * (self._max_actions - num_actions)
            strat_targets = torch.tensor(padded_strategy, dtype=torch.float32)
            # Weight by iteration for linear averaging
            weighted_strat = strat_targets * iteration
            self.strategy_buffer.add(key, iteration, features, weighted_strat)

            return node_value

        else:
            # Sample opponent action
            r = self.rng.random()
            cumulative = 0.0
            for i, p in enumerate(strategy):
                cumulative += p
                if r <= cumulative:
                    return self._traverse(
                        game.next_state(state, actions[i]),
                        target_player, iteration, player_reach,
                    )
            return self._traverse(
                game.next_state(state, actions[-1]),
                target_player, iteration, player_reach,
            )

    def _train_advantage_network(self) -> float:
        """Train advantage network on collected advantage buffer."""
        total_loss = 0.0
        optimizer = torch.optim.Adam(
            self.advantage_net.parameters(), lr=self.config.learning_rate
        )

        # Combine both player buffers
        combined_buffer = ReservoirBuffer(
            self.config.advantage_buffer_size * 2, self.config.seed + 100
        )
        for p in range(2):
            for item in self.advantage_buffers[p].buffer:
                combined_buffer.buffer.append(item)
        combined_buffer.num_seen = len(combined_buffer.buffer)

        if len(combined_buffer) < self.config.batch_size:
            return 0.0

        self.advantage_net.train()
        for step in range(self.config.advantage_train_steps):
            features, targets, iterations = combined_buffer.sample(self.config.batch_size)
            features = features.to(self.device)
            targets = targets.to(self.device)
            iterations = iterations.to(self.device)

            # Iteration-weighted MSE loss (linear weighting as per paper)
            weights = iterations / iterations.sum()

            predictions = self.advantage_net(features)
            per_sample_loss = ((predictions - targets) ** 2).sum(dim=1)
            loss = (weights * per_sample_loss).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        self.advantage_net.eval()
        return total_loss / max(1, self.config.advantage_train_steps)

    def _train_strategy_network(self) -> float:
        """Train strategy network on collected strategy buffer."""
        if len(self.strategy_buffer) < self.config.batch_size:
            return 0.0

        optimizer = torch.optim.Adam(
            self.strategy_net.parameters(), lr=self.config.learning_rate
        )
        total_loss = 0.0

        self.strategy_net.train()
        for step in range(self.config.strategy_train_steps):
            features, targets, iterations = self.strategy_buffer.sample(self.config.batch_size)
            features = features.to(self.device)
            targets = targets.to(self.device)
            iterations = iterations.to(self.device)

            # Normalize targets to get strategy probs
            target_sums = targets.sum(dim=1, keepdim=True).clamp(min=1e-10)
            target_probs = targets / target_sums

            predictions = self.strategy_net(features)

            # Cross-entropy loss (weighted by iteration)
            weights = iterations / iterations.sum()
            per_sample_loss = -(target_probs * torch.log(predictions + 1e-10)).sum(dim=1)
            loss = (weights * per_sample_loss).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        self.strategy_net.eval()
        return total_loss / max(1, self.config.strategy_train_steps)

    def train(self, num_traversals: Optional[int] = None) -> Dict[str, List[float]]:
        """Run the full Deep CFR training loop.

        Returns dict with 'advantage_losses' and 'strategy_losses'.
        """
        n = num_traversals or self.config.num_traversals
        adv_losses = []
        strat_losses = []

        # Phase 1: Data collection via MCCFR traversals
        for t in range(1, n + 1):
            self.iteration = t
            for target_player in [0, 1]:
                self._traverse(
                    self.game.initial_state(), target_player, t, 1.0
                )

            # Periodically retrain advantage network
            if t % 50 == 0 or t == n:
                loss = self._train_advantage_network()
                adv_losses.append(loss)

        # Phase 2: Final training of strategy network
        strat_loss = self._train_strategy_network()
        strat_losses.append(strat_loss)

        return {"advantage_losses": adv_losses, "strategy_losses": strat_losses}

    def average_strategy_profile(self) -> Dict[str, Dict[str, float]]:
        """Return the learned strategy from the strategy network."""
        game = self.game
        profile = {}
        visited = set()

        def traverse(state):
            if game.is_terminal(state):
                return
            player = game.current_player(state)
            if player == -1:
                for outcome, _ in game.chance_outcomes(state):
                    traverse(game.next_state(state, outcome))
                return

            key = game.infoset_key(state, player)
            if key not in visited:
                visited.add(key)
                actions = game.legal_actions(state)
                features = self.featurizer.featurize(key)
                with torch.no_grad():
                    probs = self.strategy_net(features.unsqueeze(0))[0]
                profile[key] = {
                    a: probs[i].item() for i, a in enumerate(actions)
                }
            for action in game.legal_actions(state):
                traverse(game.next_state(state, action))

        traverse(game.initial_state())
        return profile

    def exploitability(self) -> float:
        """Compute exploitability of the learned strategy."""
        from cfr.solver import CFRTrainer
        profile = self.average_strategy_profile()
        dummy = CFRTrainer(self.game)
        br0 = dummy._best_response_value(profile, 0)
        br1 = dummy._best_response_value(profile, 1)
        return 0.5 * (br0 + br1)


# ──────────────────────────────────────────────────────────────
# Single Deep CFR (SD-CFR)
# ──────────────────────────────────────────────────────────────

@dataclass
class SDCFRConfig:
    """Configuration for Single Deep CFR."""
    num_traversals: int = 300
    buffer_size: int = 200000
    train_steps: int = 2000
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden_sizes: Tuple[int, ...] = (128, 128)
    seed: int = 42
    # How many past advantage networks to keep for strategy computation
    max_stored_networks: int = 50


class SingleDeepCFR:
    """Single Deep CFR (SD-CFR).

    Simplified variant of Deep CFR that eliminates the strategy network.
    Instead, it stores advantage networks from each training round and
    computes the final strategy by combining them (weighted by iteration).

    Key difference from Deep CFR:
    - No strategy buffer or strategy network
    - Stores snapshots of advantage networks
    - Final strategy computed from weighted combination of stored networks

    Reference: Steinberger (2019), "Single Deep Counterfactual Regret Minimization"
    """

    def __init__(self, game: Game, config: Optional[SDCFRConfig] = None,
                 device: str = "cpu",
                 featurizer: Optional[InfoSetFeaturizer] = None) -> None:
        self.game = game
        self.config = config or SDCFRConfig()
        self.device = torch.device(device)

        self.featurizer = featurizer or InfoSetFeaturizer(game, device=device)
        feat_dim = self.featurizer.feature_dim()
        self._max_actions = self._find_max_actions()

        # Current advantage network (retrained each round)
        self.advantage_net = AdvantageNetwork(
            feat_dim, self._max_actions, self.config.hidden_sizes
        ).to(self.device)

        # Stored (iteration, advantage_net_state_dict) snapshots
        self.stored_networks: List[Tuple[int, dict]] = []

        # Advantage buffer
        self.advantage_buffer = ReservoirBuffer(
            self.config.buffer_size, self.config.seed
        )

        self.rng = random.Random(self.config.seed)
        self.iteration = 0

    def _find_max_actions(self) -> int:
        game = self.game
        max_a = 0
        visited = set()

        def traverse(state):
            nonlocal max_a
            if state in visited:
                return
            visited.add(state)
            if game.is_terminal(state):
                return
            player = game.current_player(state)
            if player == -1:
                for outcome, _ in game.chance_outcomes(state):
                    traverse(game.next_state(state, outcome))
                return
            actions = game.legal_actions(state)
            max_a = max(max_a, len(actions))
            for action in actions:
                traverse(game.next_state(state, action))

        traverse(game.initial_state())
        return max_a

    def _get_strategy_from_advantages(self, advantages: torch.Tensor,
                                      num_actions: int) -> List[float]:
        advs = advantages[:num_actions]
        positives = torch.clamp(advs, min=0)
        total = positives.sum()
        if total > 0:
            return (positives / total).tolist()
        return [1.0 / num_actions] * num_actions

    def _traverse(self, state, target_player: int, iteration: int,
                  player_reach: float) -> float:
        """External-sampling traversal (same as Deep CFR)."""
        game = self.game

        if game.is_terminal(state):
            return game.terminal_utility(state, target_player)

        player = game.current_player(state)

        if player == -1:
            outcomes = game.chance_outcomes(state)
            probs = [p for _, p in outcomes]
            r = self.rng.random()
            cumulative = 0.0
            for i, p in enumerate(probs):
                cumulative += p
                if r <= cumulative:
                    return self._traverse(
                        game.next_state(state, outcomes[i][0]),
                        target_player, iteration, player_reach,
                    )
            return self._traverse(
                game.next_state(state, outcomes[-1][0]),
                target_player, iteration, player_reach,
            )

        actions = game.legal_actions(state)
        num_actions = len(actions)
        key = game.infoset_key(state, player)

        features = self.featurizer.featurize(key)
        with torch.no_grad():
            advantages = self.advantage_net(features.unsqueeze(0))[0]
        strategy = self._get_strategy_from_advantages(advantages, num_actions)

        if player == target_player:
            action_values = []
            for a_idx, action in enumerate(actions):
                v = self._traverse(
                    game.next_state(state, action),
                    target_player, iteration,
                    player_reach * strategy[a_idx],
                )
                action_values.append(v)

            node_value = sum(s * v for s, v in zip(strategy, action_values))
            regrets = [v - node_value for v in action_values]

            padded_regrets = regrets + [0.0] * (self._max_actions - num_actions)
            targets = torch.tensor(padded_regrets, dtype=torch.float32)

            self.advantage_buffer.add(key, iteration, features, targets)
            return node_value

        else:
            r = self.rng.random()
            cumulative = 0.0
            for i, p in enumerate(strategy):
                cumulative += p
                if r <= cumulative:
                    return self._traverse(
                        game.next_state(state, actions[i]),
                        target_player, iteration, player_reach,
                    )
            return self._traverse(
                game.next_state(state, actions[-1]),
                target_player, iteration, player_reach,
            )

    def _train_advantage_network(self) -> float:
        """Train advantage network on buffer."""
        if len(self.advantage_buffer) < self.config.batch_size:
            return 0.0

        optimizer = torch.optim.Adam(
            self.advantage_net.parameters(), lr=self.config.learning_rate
        )
        total_loss = 0.0

        self.advantage_net.train()
        for step in range(self.config.train_steps):
            features, targets, iterations = self.advantage_buffer.sample(self.config.batch_size)
            features = features.to(self.device)
            targets = targets.to(self.device)
            iterations = iterations.to(self.device)

            weights = iterations / iterations.sum()
            predictions = self.advantage_net(features)
            per_sample_loss = ((predictions - targets) ** 2).sum(dim=1)
            loss = (weights * per_sample_loss).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        self.advantage_net.eval()
        return total_loss / max(1, self.config.train_steps)

    def train(self, num_traversals: Optional[int] = None) -> Dict[str, List[float]]:
        """Run the SD-CFR training loop."""
        n = num_traversals or self.config.num_traversals
        losses = []
        snapshot_interval = max(1, n // self.config.max_stored_networks)

        for t in range(1, n + 1):
            self.iteration = t
            for target_player in [0, 1]:
                self._traverse(
                    self.game.initial_state(), target_player, t, 1.0
                )

            # Periodically train and snapshot
            if t % snapshot_interval == 0 or t == n:
                loss = self._train_advantage_network()
                losses.append(loss)

                # Store snapshot
                state_dict = {k: v.clone() for k, v in self.advantage_net.state_dict().items()}
                self.stored_networks.append((t, state_dict))

                # Trim old snapshots if needed
                if len(self.stored_networks) > self.config.max_stored_networks:
                    self.stored_networks = self.stored_networks[-self.config.max_stored_networks:]

        return {"losses": losses}

    def average_strategy_profile(self) -> Dict[str, Dict[str, float]]:
        """Compute average strategy from stored advantage network snapshots.

        For each infoset, computes the iteration-weighted average of strategies
        derived from each stored advantage network.
        """
        game = self.game
        profile = {}
        visited = set()
        feat_dim = self.featurizer.feature_dim()

        # Load all stored networks
        nets = []
        for t, state_dict in self.stored_networks:
            net = AdvantageNetwork(
                feat_dim, self._max_actions, self.config.hidden_sizes
            ).to(self.device)
            net.load_state_dict(state_dict)
            net.eval()
            nets.append((t, net))

        if not nets:
            # Fallback: use current network
            nets = [(self.iteration, self.advantage_net)]

        total_weight = sum(t for t, _ in nets)

        def traverse(state):
            if game.is_terminal(state):
                return
            player = game.current_player(state)
            if player == -1:
                for outcome, _ in game.chance_outcomes(state):
                    traverse(game.next_state(state, outcome))
                return

            key = game.infoset_key(state, player)
            if key not in visited:
                visited.add(key)
                actions = game.legal_actions(state)
                num_actions = len(actions)
                features = self.featurizer.featurize(key)

                # Average strategy across all stored networks
                avg_strategy = torch.zeros(num_actions)
                with torch.no_grad():
                    for t, net in nets:
                        advs = net(features.unsqueeze(0))[0][:num_actions]
                        positives = torch.clamp(advs, min=0)
                        total = positives.sum()
                        if total > 0:
                            strat = positives / total
                        else:
                            strat = torch.full((num_actions,), 1.0 / num_actions)
                        avg_strategy += t * strat

                avg_strategy /= total_weight
                # Renormalize
                s = avg_strategy.sum()
                if s > 0:
                    avg_strategy /= s
                else:
                    avg_strategy = torch.full((num_actions,), 1.0 / num_actions)

                profile[key] = {
                    a: avg_strategy[i].item() for i, a in enumerate(actions)
                }
            for action in game.legal_actions(state):
                traverse(game.next_state(state, action))

        traverse(game.initial_state())
        return profile

    def exploitability(self) -> float:
        """Compute exploitability."""
        from cfr.solver import CFRTrainer
        profile = self.average_strategy_profile()
        dummy = CFRTrainer(self.game)
        br0 = dummy._best_response_value(profile, 0)
        br1 = dummy._best_response_value(profile, 1)
        return 0.5 * (br0 + br1)
