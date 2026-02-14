"""Abstract game interface for CFR and ReBeL.

Defines the protocol that any game must implement to be usable with
the CFR solver, belief state tracker, and ReBeL trainer. This allows
the same solver code to work with Kuhn Poker, Leduc Poker, Liar's Dice,
or any other imperfect-information game.

The interface uses Python's Protocol (structural subtyping) so games
don't need to explicitly inherit — they just need to implement the
required methods.
"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol, Tuple, runtime_checkable


# Type aliases — games define their own state and action types
State = Any
Action = Any


@runtime_checkable
class Game(Protocol):
    """Interface for an imperfect-information game.

    Any game that implements these methods can be used with:
    - CFRTrainer (vanilla tabular CFR)
    - SubgameSolver (depth-limited CFR for endgame solving)
    - VectorizedCFR / BeliefStateTracker (torch-based, game-specific)

    State and Action types are game-specific. The solver treats them
    as opaque values, only interacting through these methods.
    """

    NUM_PLAYERS: int

    def initial_state(self) -> State:
        """Return the initial game state (before any actions)."""
        ...

    def is_terminal(self, state: State) -> bool:
        """Return True if the state is terminal (game over)."""
        ...

    def current_player(self, state: State) -> Optional[int]:
        """Return the current player to act.

        Returns:
            0, 1, ... for player turns
            -1 for chance nodes
            None for terminal states
        """
        ...

    def legal_actions(self, state: State) -> List[Action]:
        """Return the list of legal actions at this state."""
        ...

    def chance_outcomes(self, state: State) -> List[Tuple[Action, float]]:
        """Return (action, probability) pairs for chance nodes.

        Only called when current_player(state) == -1.
        Probabilities must sum to 1.
        """
        ...

    def next_state(self, state: State, action: Action) -> State:
        """Return the state after taking an action."""
        ...

    def infoset_key(self, state: State, player: int) -> str:
        """Return a string key identifying the information set.

        Two states have the same infoset key for a player iff
        that player cannot distinguish between them (same private
        information and public history).
        """
        ...

    def terminal_utility(self, state: State, player: int) -> float:
        """Return the utility for `player` at a terminal state.

        Only called when is_terminal(state) is True.
        """
        ...
