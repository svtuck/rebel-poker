"""MLX-accelerated sparse marginalization for Apple Silicon.

This module provides GPU-accelerated implementations of the reach-guided
sparse marginalization using Apple's MLX framework. MLX provides Metal
GPU access with a NumPy-like API that's ideal for the tensor operations
in marginalization.

Key operations that benefit from GPU acceleration:
1. Reach probability updates: element-wise multiply of reach × likelihood
2. Joint computation: product of P reach vectors indexed by deal structure
3. Marginalization: scatter-add of joint weights into per-player beliefs
4. Pruning: threshold comparison and index selection

For games with >10k deals, the GPU parallelism gives significant speedups
over the CPU implementation, especially for the joint computation which
is embarrassingly parallel.

Falls back to PyTorch (CPU) if MLX is not available.

Usage:
    from sampling.mlx_marginalization import MLXSparseTracker
    tracker = MLXSparseTracker(cfg)  # Uses MLX if available, else PyTorch
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Tuple

import torch

# Try to import MLX — fall back gracefully
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


from sampling.marginalization import (
    GameConfig,
    enumerate_deals,
    deal_to_private_states,
    num_private_states,
)


class MLXSparseTracker:
    """GPU-accelerated reach-guided sparse marginalization.

    Uses MLX for Metal GPU acceleration on Apple Silicon. The key insight
    is that the reach update, joint computation, and marginalization are
    all data-parallel operations that map well to GPU execution.

    When MLX is not available, falls back to an optimized PyTorch
    implementation that uses vectorized tensor operations instead of
    Python loops.

    Architecture:
    - Reach probabilities are stored as dense [n_private_states] tensors per player
    - The deal structure is precomputed as index tensors:
      deal_state_indices[p] : [num_deals] -> private state index for player p
    - Joint computation: joint[d] = prod(reach[p][deal_state_indices[p][d]])
    - Marginalization: scatter_add of joint into belief vectors
    - Pruning: mask deals below threshold, compact active set

    All operations are vectorized — no Python loops over deals.
    """

    def __init__(
        self,
        cfg: GameConfig,
        prune_threshold: float = 1e-6,
        use_mlx: Optional[bool] = None,
    ):
        self.cfg = cfg
        self.prune_threshold = prune_threshold
        self.n_private = num_private_states(cfg)

        # Determine backend
        if use_mlx is None:
            self.use_mlx = HAS_MLX
        else:
            self.use_mlx = use_mlx and HAS_MLX

        # Enumerate deals and build index tensors
        deals = enumerate_deals(cfg)
        self.num_deals = len(deals)
        d2s = deal_to_private_states(deals, cfg)

        if self.use_mlx:
            self._init_mlx(d2s)
        else:
            self._init_torch(d2s)

    def _init_torch(self, d2s: Dict[int, List[int]]):
        """Initialize with PyTorch tensors (vectorized, no loops)."""
        # Deal-to-state index tensors: [num_deals] per player
        self.deal_indices_t = {
            p: torch.tensor(d2s[p], dtype=torch.long)
            for p in range(self.cfg.num_players)
        }

        # Per-player reach: [n_private_states]
        self.reach_t = [
            torch.ones(self.n_private) for _ in range(self.cfg.num_players)
        ]

        # Active deal mask: [num_deals] boolean
        self.active_mask_t = torch.ones(self.num_deals, dtype=torch.bool)

        # Precompute marginalization scatter indices for each player
        # For player p, marg_scatter[p] groups deals by private state
        self.marg_scatter_t = {}
        for p in range(self.cfg.num_players):
            self.marg_scatter_t[p] = self.deal_indices_t[p]

    def _init_mlx(self, d2s: Dict[int, List[int]]):
        """Initialize with MLX arrays for GPU acceleration."""
        self.deal_indices_m = {
            p: mx.array(d2s[p], dtype=mx.int32)
            for p in range(self.cfg.num_players)
        }
        self.reach_m = [
            mx.ones(self.n_private) for _ in range(self.cfg.num_players)
        ]
        self.active_mask_m = mx.ones(self.num_deals, dtype=mx.bool_)

    def observe_action(
        self, acting_player: int, action: int, policy: torch.Tensor
    ) -> None:
        """Update reach probabilities (vectorized, no Python loops)."""
        if self.use_mlx:
            self._observe_mlx(acting_player, action, policy)
        else:
            self._observe_torch(acting_player, action, policy)

    def _observe_torch(
        self, acting_player: int, action: int, policy: torch.Tensor
    ):
        """Vectorized PyTorch reach update + pruning."""
        likelihood = policy[:, action]  # [n_private_states]
        self.reach_t[acting_player] = self.reach_t[acting_player] * likelihood

        # Compute joint for active deals (vectorized)
        joint = self._compute_joint_torch()

        # Prune: deals below threshold × max
        if joint.numel() > 0:
            max_val = joint[self.active_mask_t].max() if self.active_mask_t.any() else torch.tensor(0.0)
            threshold = self.prune_threshold * max_val
            # Zero out and deactivate pruned deals
            below = joint < threshold
            self.active_mask_t = self.active_mask_t & ~below

    def _observe_mlx(
        self, acting_player: int, action: int, policy: torch.Tensor
    ):
        """MLX GPU-accelerated reach update."""
        likelihood = mx.array(policy[:, action].numpy())
        self.reach_m[acting_player] = self.reach_m[acting_player] * likelihood

        # Compute joint and prune (all on GPU)
        joint = self._compute_joint_mlx()
        active_joint = mx.where(self.active_mask_m, joint, mx.zeros_like(joint))
        max_val = mx.max(active_joint)
        threshold = self.prune_threshold * max_val
        self.active_mask_m = self.active_mask_m & (joint >= threshold)

    def _compute_joint_torch(self) -> torch.Tensor:
        """Compute joint distribution over deals using vectorized indexing."""
        # joint[d] = product over players of reach[p][deal_state[p][d]]
        joint = torch.ones(self.num_deals)
        for p in range(self.cfg.num_players):
            # Fancy indexing: reach[deal_indices] gives [num_deals] tensor
            joint = joint * self.reach_t[p][self.deal_indices_t[p]]
        return joint

    def _compute_joint_mlx(self):
        """Compute joint distribution on GPU via MLX."""
        joint = mx.ones(self.num_deals)
        for p in range(self.cfg.num_players):
            player_reach = self.reach_m[p][self.deal_indices_m[p]]
            joint = joint * player_reach
        return joint

    def marginal_belief(self, player: int) -> torch.Tensor:
        """Compute marginal belief via vectorized scatter-add."""
        if self.use_mlx:
            return self._marginal_mlx(player)
        return self._marginal_torch(player)

    def _marginal_torch(self, player: int) -> torch.Tensor:
        """Vectorized marginalization using scatter_add."""
        joint = self._compute_joint_torch()

        # Zero out inactive deals
        joint = joint * self.active_mask_t.float()

        # Normalize
        total = joint.sum()
        if total > 0:
            joint = joint / total

        # Scatter-add into belief vector
        belief = torch.zeros(self.n_private)
        belief.scatter_add_(0, self.marg_scatter_t[player], joint)

        return belief

    def _marginal_mlx(self, player: int) -> torch.Tensor:
        """MLX GPU marginalization."""
        joint = self._compute_joint_mlx()
        joint = mx.where(self.active_mask_m, joint, mx.zeros_like(joint))

        total = mx.sum(joint)
        if total > 0:
            joint = joint / total

        # MLX doesn't have scatter_add, so we use a loop over private states
        # (This is the main candidate for a custom Metal kernel)
        indices = self.deal_indices_m[player]
        belief_np = mx.zeros(self.n_private)

        # For each private state, sum the joint values of matching deals
        for s in range(self.n_private):
            mask = indices == s
            belief_np = belief_np.at[s].add(mx.sum(mx.where(mask, joint, mx.zeros_like(joint))))

        # Convert back to PyTorch for compatibility
        return torch.tensor(belief_np.tolist())

    def active_count(self) -> int:
        if self.use_mlx:
            return int(mx.sum(self.active_mask_m.astype(mx.int32)).item())
        return int(self.active_mask_t.sum().item())

    def sparsity_ratio(self) -> float:
        return self.active_count() / max(1, self.num_deals)


class VectorizedExactTracker:
    """Fully vectorized exact tracker (no Python loops over deals).

    This is the PyTorch-optimized version of ExactJointBeliefTracker.
    Instead of looping over deals in Python, all operations use
    vectorized tensor indexing and scatter operations.

    This alone provides significant speedup over the loop-based exact
    tracker for larger games, and serves as a baseline showing that
    much of the "exact is slow" perception comes from Python loops,
    not from the algorithmic cost.
    """

    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        self.n_private = num_private_states(cfg)
        deals = enumerate_deals(cfg)
        self.num_deals = len(deals)
        d2s = deal_to_private_states(deals, cfg)

        # Index tensors for vectorized operations
        self.deal_indices = {
            p: torch.tensor(d2s[p], dtype=torch.long)
            for p in range(cfg.num_players)
        }

        # Joint distribution
        self.joint = torch.ones(self.num_deals) / self.num_deals

    def observe_action(
        self, acting_player: int, action: int, policy: torch.Tensor
    ) -> None:
        """Vectorized Bayesian update."""
        # likelihood[d] = policy[state_of_acting_player_in_deal_d, action]
        states = self.deal_indices[acting_player]
        likelihood = policy[states, action]  # [num_deals]

        self.joint = self.joint * likelihood
        total = self.joint.sum()
        if total > 0:
            self.joint /= total

    def marginal_belief(self, player: int) -> torch.Tensor:
        """Vectorized marginalization via scatter_add."""
        belief = torch.zeros(self.n_private)
        belief.scatter_add_(0, self.deal_indices[player], self.joint)
        return belief

    def action_probability(
        self, acting_player: int, policy: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized P(action) computation."""
        states = self.deal_indices[acting_player]
        # policy_at_deals[d, a] = policy[state_for_deal_d, a]
        policy_at_deals = policy[states]  # [num_deals, num_actions]
        # P(a) = sum_d joint[d] * policy[state(d), a]
        return (self.joint.unsqueeze(1) * policy_at_deals).sum(dim=0)


# ---------------------------------------------------------------------------
# Benchmark: vectorized vs loop-based
# ---------------------------------------------------------------------------

def benchmark_vectorized_vs_loop():
    """Compare vectorized PyTorch vs loop-based Python for exact computation."""
    import time
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from tests.test_marginalization import (
        ExactJointBeliefTracker,
        generate_random_policy,
        l1_distance,
    )

    configs = [
        ("13-card 2P", GameConfig(deck_size=13, num_players=2, cards_per_player=1, num_actions=3)),
        ("20-card 2P", GameConfig(deck_size=20, num_players=2, cards_per_player=1, num_actions=4)),
        ("13-card 3P", GameConfig(deck_size=13, num_players=3, cards_per_player=1, num_actions=2)),
        ("10-card 2P 2-hand", GameConfig(deck_size=10, num_players=2, cards_per_player=2, num_actions=3)),
        ("12-card 2P 2-hand", GameConfig(deck_size=12, num_players=2, cards_per_player=2, num_actions=3)),
    ]

    print(f"{'Game':<25} {'Deals':>8} {'Loop(ms)':>10} {'Vec(ms)':>10} {'MLX(ms)':>10} {'Speedup':>10} {'L1':>10}")
    print("-" * 90)

    for name, cfg in configs:
        n_priv = num_private_states(cfg)
        deals = enumerate_deals(cfg)
        num_steps = 10

        # Generate actions
        rng = torch.Generator().manual_seed(42)
        loop_tracker = ExactJointBeliefTracker(cfg)
        actions = []

        for step in range(num_steps):
            player = step % cfg.num_players
            policy = generate_random_policy(n_priv, cfg.num_actions, rng)
            action_probs = loop_tracker.action_probability(player, policy)
            action = torch.multinomial(action_probs, 1, generator=rng).item()
            loop_tracker.observe_action(player, action, policy)
            actions.append((player, action, policy))

        # Time loop-based
        loop_tracker2 = ExactJointBeliefTracker(cfg)
        t0 = time.perf_counter()
        for player, action, policy in actions:
            loop_tracker2.observe_action(player, action, policy)
        for p in range(cfg.num_players):
            loop_tracker2.marginal_belief(p)
        loop_ms = (time.perf_counter() - t0) * 1000

        # Time vectorized
        vec_tracker = VectorizedExactTracker(cfg)
        t0 = time.perf_counter()
        for player, action, policy in actions:
            vec_tracker.observe_action(player, action, policy)
        for p in range(cfg.num_players):
            vec_tracker.marginal_belief(p)
        vec_ms = (time.perf_counter() - t0) * 1000

        # Time MLX sparse
        mlx_tracker = MLXSparseTracker(cfg, prune_threshold=1e-10)
        t0 = time.perf_counter()
        for player, action, policy in actions:
            mlx_tracker.observe_action(player, action, policy)
        for p in range(cfg.num_players):
            mlx_tracker.marginal_belief(p)
        mlx_ms = (time.perf_counter() - t0) * 1000

        # Verify correctness
        max_l1 = 0.0
        for p in range(cfg.num_players):
            l1 = l1_distance(loop_tracker.marginal_belief(p), vec_tracker.marginal_belief(p))
            max_l1 = max(max_l1, l1)

        speedup = loop_ms / vec_ms if vec_ms > 0 else float('inf')
        mlx_label = f"{mlx_ms:.1f}" if not mlx_tracker.use_mlx else f"{mlx_ms:.1f}*"

        print(f"{name:<25} {len(deals):>8} {loop_ms:>10.1f} {vec_ms:>10.1f} "
              f"{mlx_label:>10} {speedup:>10.1f}x {max_l1:>10.6f}")

    print(f"\n* = MLX GPU accelerated (HAS_MLX={HAS_MLX})")
    print("\nNote: The vectorized PyTorch version eliminates Python loops over deals,")
    print("which is the main bottleneck in the loop-based ExactJointBeliefTracker.")
    print("MLX would provide additional GPU parallelism on Apple Silicon.")


if __name__ == "__main__":
    benchmark_vectorized_vs_loop()
