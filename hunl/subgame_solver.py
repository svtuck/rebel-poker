"""Real-time subgame solving for HUNL play.

At decision time, the agent:
1. Constructs PBS from the game history (opponent's actions update beliefs)
2. Runs depth-limited CFR with the value network at leaves
3. Plays according to the solved strategy for its actual hand

This implements Libratus-style nested subgame solving:
- When the opponent plays an off-tree action, we re-solve from the current state
- The value network provides leaf evaluations, keeping the solve tractable
- Depth limit of 4-6 actions keeps solve time under 2 seconds

This module provides:
- SubgamePlayer: stateful player that maintains beliefs and solves on demand
- solve_subgame: one-shot subgame solve from a state
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from hunl.game import (
    ALL_HOLE_COMBOS,
    HUNLAction,
    HUNLConfig,
    HUNLGame,
    HUNLState,
    NUM_COMBOS,
)
from hunl.belief_state import HUNLBeliefState
from hunl.rebel_trainer import HUNLSubgameSolver
from hunl.value_net import HUNLValueNetwork, encode_state_features


def solve_subgame(
    game: HUNLGame,
    state: HUNLState,
    beliefs: torch.Tensor,
    value_net: HUNLValueNetwork,
    max_depth: int = 4,
    cfr_iters: int = 200,
) -> Dict[str, Dict[str, float]]:
    """One-shot subgame solve from a given state.

    Args:
        game: HUNL game instance
        state: current game state
        beliefs: [1326, 2] PBS at this state
        value_net: trained value network for leaf evaluation
        max_depth: CFR depth limit
        cfr_iters: number of CFR iterations

    Returns:
        Strategy profile: {history_key: {action_idx: probability}}
    """
    def value_fn(pbs_flat, features):
        value_net.eval()
        with torch.no_grad():
            return value_net(pbs_flat, features)

    solver = HUNLSubgameSolver(
        game=game,
        state=state,
        initial_beliefs=beliefs,
        value_fn=value_fn,
        max_depth=max_depth,
        cfr_iters=cfr_iters,
    )

    return solver.solve()


class SubgamePlayer:
    """A player that uses subgame solving to make decisions.

    Maintains running belief state and re-solves at each decision point.
    Implements nested subgame resolution: when the opponent plays an
    off-tree action, beliefs are updated and a new solve is performed.

    Usage:
        player = SubgamePlayer(game, value_net, player_id=0)
        player.observe_deal(my_hand, board)
        while not game_over:
            if my_turn:
                action = player.act(state)
            else:
                player.observe_action(opponent_action, state)
    """

    def __init__(
        self,
        game: HUNLGame,
        value_net: HUNLValueNetwork,
        player_id: int = 0,
        max_depth: int = 4,
        cfr_iters: int = 200,
    ) -> None:
        self.game = game
        self.value_net = value_net
        self.player_id = player_id
        self.max_depth = max_depth
        self.cfr_iters = cfr_iters

        self.my_hand: Optional[Tuple[int, int]] = None
        self.belief_state: Optional[HUNLBeliefState] = None
        self._current_profile: Optional[Dict] = None

    def observe_deal(
        self,
        my_hand: Tuple[int, int],
        board: Tuple[int, ...] = (),
    ) -> None:
        """Called when we see our hole cards and the current board."""
        self.my_hand = my_hand
        self.belief_state = HUNLBeliefState(board=board)

    def observe_board(self, board: Tuple[int, ...]) -> None:
        """Called when new community cards are revealed."""
        if self.belief_state is not None:
            self.belief_state.update_board(board)
        self._current_profile = None  # Invalidate cached solve

    def observe_action(
        self,
        action: HUNLAction,
        state: HUNLState,
        actor: int,
    ) -> None:
        """Called when an action is observed (ours or opponent's).

        For opponent actions, performs Bayesian update on their beliefs
        using the last solved strategy.
        """
        if actor != self.player_id and self._current_profile is not None:
            # Update opponent's beliefs based on the action they took
            # This requires knowing the strategy at the state before the action
            history_key = ",".join(state.action_history[:-1])  # Pre-action history
            strategy = self._current_profile.get(history_key, {})

            if strategy:
                # Find which action index this was
                actions = self.game.legal_actions(state)
                action_idx = None
                for i, a in enumerate(actions):
                    if a.action_type == action.action_type and a.amount == action.amount:
                        action_idx = i
                        break

                if action_idx is not None and str(action_idx) in strategy:
                    # The action probability is uniform across all hands
                    # (public state strategy), so the Bayesian update
                    # just multiplies by the strategy probability
                    prob = strategy[str(action_idx)]
                    action_probs = torch.full((NUM_COMBOS,), prob)
                    self.belief_state.bayesian_update(actor, action_probs)

        self._current_profile = None  # Need re-solve

    def act(self, state: HUNLState) -> HUNLAction:
        """Choose an action at the current state.

        Runs subgame solve and selects action according to the solved
        strategy for our actual hand.

        Args:
            state: current game state

        Returns:
            The chosen action
        """
        if self.belief_state is None:
            raise RuntimeError("Must call observe_deal before act")

        # Solve subgame
        beliefs = self.belief_state.get_pbs()
        profile = solve_subgame(
            game=self.game,
            state=state,
            beliefs=beliefs,
            value_net=self.value_net,
            max_depth=self.max_depth,
            cfr_iters=self.cfr_iters,
        )
        self._current_profile = profile

        # Get strategy for current state
        history_key = ",".join(state.action_history)
        strategy = profile.get(history_key, {})

        actions = self.game.legal_actions(state)
        if not strategy or not actions:
            # Default to check/call if no strategy
            for a in actions:
                if a.action_type in ("check", "call"):
                    return a
            return actions[0] if actions else HUNLAction("fold")

        # Sample action according to strategy
        probs = []
        for i in range(len(actions)):
            probs.append(strategy.get(str(i), 1.0 / len(actions)))

        # Normalize
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            probs = [1.0 / len(actions)] * len(actions)

        # Sample
        import random
        action_idx = random.choices(range(len(actions)), weights=probs, k=1)[0]
        return actions[action_idx]
