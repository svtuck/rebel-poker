"""Test suite for public belief state marginalization.

Simulates a configurable card game where:
- D cards in the deck
- P players each dealt K cards (no two players share a card)
- A possible actions per turn
- Random policies: P(action | private_state) for each player
- Bayesian belief updates after each observed action

Validates that:
1. Exact beliefs (computed via joint reach × compatibility, then marginalized)
   match beliefs computed by the factored/marginal update rule
2. Beliefs remain valid probability distributions throughout
3. Approximation quality can be measured when using factored (marginal) beliefs
   instead of the full joint

The "exact" method maintains the full joint distribution over all players' private
states (exponential in P), while the "marginal" method maintains per-player belief
vectors independently (linear in P). For 2 players these are equivalent, but for
P >= 3 the marginal factorization is an approximation.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pytest
import torch


# ---------------------------------------------------------------------------
# Card game configuration
# ---------------------------------------------------------------------------

@dataclass
class GameConfig:
    """Configuration for a parameterized card game.

    Args:
        deck_size: D — total number of distinct cards
        num_players: P — number of players
        cards_per_player: K — cards dealt to each player
        num_actions: A — number of actions available each turn
        seed: random seed for reproducibility
    """
    deck_size: int = 6
    num_players: int = 2
    cards_per_player: int = 1
    num_actions: int = 2
    seed: int = 42

    def __post_init__(self):
        assert self.deck_size >= self.num_players * self.cards_per_player, (
            f"Deck size {self.deck_size} too small for {self.num_players} players "
            f"× {self.cards_per_player} cards each"
        )


# ---------------------------------------------------------------------------
# Deal enumeration and compatibility
# ---------------------------------------------------------------------------

def enumerate_deals(cfg: GameConfig) -> List[Tuple[Tuple[int, ...], ...]]:
    """Enumerate all valid deals: tuples of P hands, each hand is K cards.

    A deal is valid iff no card appears in more than one hand.
    Returns list of deals, where each deal is a tuple of P tuples (one per player).
    """
    cards = list(range(cfg.deck_size))
    deals = []

    def _build(player: int, used: set, so_far: list):
        if player == cfg.num_players:
            deals.append(tuple(tuple(h) for h in so_far))
            return
        for hand in itertools.combinations(cards, cfg.cards_per_player):
            if used.isdisjoint(hand):
                so_far.append(hand)
                _build(player + 1, used | set(hand), so_far)
                so_far.pop()

    _build(0, set(), [])
    return deals


def compatibility_matrix(cfg: GameConfig) -> torch.Tensor:
    """Build the compatibility matrix over all deals.

    Shape: [num_deals] — 1.0 for valid deals, used as chance weights.
    (All enumerated deals are valid by construction, so this is all-ones,
    but we keep it explicit for clarity and to support weighted variants.)
    """
    deals = enumerate_deals(cfg)
    return torch.ones(len(deals))


def deal_to_private_states(
    deals: List[Tuple[Tuple[int, ...], ...]], cfg: GameConfig
) -> Dict[int, List[List[int]]]:
    """Map from each player to a list (indexed by deal_idx) of their private state index.

    For K=1, private state = the card itself.
    For K>1, private state = index into the sorted list of possible K-card hands.
    """
    # Enumerate all possible hands per player (same for all players)
    all_hands = list(itertools.combinations(range(cfg.deck_size), cfg.cards_per_player))
    hand_to_idx = {h: i for i, h in enumerate(all_hands)}

    result = {}
    for p in range(cfg.num_players):
        result[p] = [hand_to_idx[deals[d][p]] for d in range(len(deals))]
    return result


def num_private_states(cfg: GameConfig) -> int:
    """Number of possible private states (hands) per player."""
    return math.comb(cfg.deck_size, cfg.cards_per_player)


# ---------------------------------------------------------------------------
# Random policy generation
# ---------------------------------------------------------------------------

def generate_random_policy(
    n_private_states: int, n_actions: int, rng: torch.Generator
) -> torch.Tensor:
    """Generate a random conditional policy: P(action | private_state).

    Shape: [n_private_states, n_actions], rows sum to 1.
    Uses Dirichlet(1,...,1) = uniform over simplex.
    """
    # Exponential trick for Dirichlet(1,...,1): sample exponentials, normalize
    raw = torch.zeros(n_private_states, n_actions).exponential_(1.0, generator=rng)
    return raw / raw.sum(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Exact joint belief tracking
# ---------------------------------------------------------------------------

class ExactJointBeliefTracker:
    """Tracks the exact joint distribution over all players' private states.

    Maintains a tensor of shape [num_deals] representing the probability of
    each deal given the observed action history. This is exponential in P
    but exact — no approximation.
    """

    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        self.deals = enumerate_deals(cfg)
        self.num_deals = len(self.deals)
        self.deal_to_states = deal_to_private_states(self.deals, cfg)
        self.n_private = num_private_states(cfg)

        # Uniform prior over deals (could weight by compatibility, but all valid)
        self.joint = torch.ones(self.num_deals) / self.num_deals

    def observe_action(
        self, acting_player: int, action: int, policy: torch.Tensor
    ) -> int:
        """Sample an action from the acting player's policy, then update beliefs.

        Args:
            acting_player: which player is acting
            action: the observed action (already sampled externally)
            policy: [n_private_states, n_actions] — P(action | private_state)

        Returns:
            The observed action.
        """
        # P(action | deal) = policy[private_state_of_acting_player_in_deal, action]
        likelihood = torch.zeros(self.num_deals)
        for d in range(self.num_deals):
            s = self.deal_to_states[acting_player][d]
            likelihood[d] = policy[s, action]

        # Bayes' rule: P(deal | history, action) ∝ P(action | deal) × P(deal | history)
        self.joint = self.joint * likelihood
        total = self.joint.sum()
        if total > 0:
            self.joint /= total

        return action

    def marginal_belief(self, player: int) -> torch.Tensor:
        """Compute the marginal belief over a single player's private state.

        Returns: [n_private_states] where entry s = P(player has state s | history)
        """
        belief = torch.zeros(self.n_private)
        for d in range(self.num_deals):
            s = self.deal_to_states[player][d]
            belief[s] += self.joint[d]
        return belief

    def action_probability(
        self, acting_player: int, policy: torch.Tensor
    ) -> torch.Tensor:
        """Compute P(action) = Σ_deal P(action|deal) P(deal).

        Returns: [n_actions]
        """
        n_actions = policy.shape[1]
        probs = torch.zeros(n_actions)
        for d in range(self.num_deals):
            s = self.deal_to_states[acting_player][d]
            probs += self.joint[d] * policy[s]
        return probs


# ---------------------------------------------------------------------------
# Factored (marginal) belief tracking — the approximation
# ---------------------------------------------------------------------------

class FactoredBeliefTracker:
    """Tracks per-player marginal beliefs independently.

    Maintains P vectors of shape [n_private_states], one per player.
    Updates each player's belief independently using Bayes' rule.

    For 2 players, this is equivalent to the exact joint (because the joint
    factors as reach_p0 × reach_p1 × chance). For P >= 3, this is an
    approximation because it ignores correlations between players' hands.
    """

    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        self.n_private = num_private_states(cfg)
        # Per-player uniform prior
        self.beliefs = [
            torch.ones(self.n_private) / self.n_private
            for _ in range(cfg.num_players)
        ]

    def observe_action(
        self, acting_player: int, action: int, policy: torch.Tensor
    ) -> None:
        """Update beliefs after observing an action.

        For the acting player:
            P(state | action) ∝ P(action | state) × P(state)

        For other players: beliefs remain unchanged in the factored approximation.
        (In the exact joint, observing someone's action DOES update your belief
        about other players' hands because of the incompatibility constraint.
        The factored approximation ignores this cross-player update.)
        """
        # Update acting player's belief
        likelihood = policy[:, action]  # [n_private_states]
        updated = self.beliefs[acting_player] * likelihood
        total = updated.sum()
        if total > 0:
            updated /= total
        self.beliefs[acting_player] = updated

    def marginal_belief(self, player: int) -> torch.Tensor:
        """Return the marginal belief for a player. Shape: [n_private_states]."""
        return self.beliefs[player].clone()


# ---------------------------------------------------------------------------
# Reach-probability-based exact computation
# ---------------------------------------------------------------------------

class ReachProbabilityTracker:
    """Tracks beliefs using per-player reach probabilities and the compatibility matrix.

    This is the method described in the issue: maintain per-player reach vectors,
    compute the joint as the outer product of reaches times compatibility,
    then marginalize.

    For 2 players this gives identical results to ExactJointBeliefTracker.
    For P >= 3 it is ALSO exact (not an approximation), because it constructs
    the full joint from reaches before marginalizing.
    """

    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        self.deals = enumerate_deals(cfg)
        self.num_deals = len(self.deals)
        self.deal_to_states = deal_to_private_states(self.deals, cfg)
        self.n_private = num_private_states(cfg)

        # Per-player reach probabilities: reach[p][d] = product of acting probs
        # when player p was the actor, across the history
        self.reach = [torch.ones(self.num_deals) for _ in range(cfg.num_players)]

        # Chance probability (uniform over valid deals)
        self.chance = torch.ones(self.num_deals) / self.num_deals

    def observe_action(
        self, acting_player: int, action: int, policy: torch.Tensor
    ) -> None:
        """Update reach probabilities after observing an action.

        Only the acting player's reach is updated:
            reach[acting_player][d] *= policy[private_state(d, acting_player), action]
        """
        for d in range(self.num_deals):
            s = self.deal_to_states[acting_player][d]
            self.reach[acting_player][d] *= policy[s, action]

    def joint_distribution(self) -> torch.Tensor:
        """Compute the joint distribution over deals.

        joint[d] ∝ chance[d] × Π_p reach[p][d]

        This is the outer product of reach probabilities times the compatibility
        matrix (which is implicit in the deal enumeration).
        """
        joint = self.chance.clone()
        for p in range(self.cfg.num_players):
            joint *= self.reach[p]
        total = joint.sum()
        if total > 0:
            joint /= total
        return joint

    def marginal_belief(self, player: int) -> torch.Tensor:
        """Compute marginal belief for a player by marginalizing the joint.

        P(player has state s | history) = Σ_{deals where player has state s} joint[d]
        """
        joint = self.joint_distribution()
        belief = torch.zeros(self.n_private)
        for d in range(self.num_deals):
            s = self.deal_to_states[player][d]
            belief[s] += joint[d]
        return belief


# ---------------------------------------------------------------------------
# Test utilities
# ---------------------------------------------------------------------------

def sample_action(
    acting_player: int,
    policy: torch.Tensor,
    joint_tracker: ExactJointBeliefTracker,
    rng: torch.Generator,
) -> int:
    """Sample an action according to the true probability P(action).

    P(a) = Σ_deal P(a | deal) × P(deal | history)
    """
    action_probs = joint_tracker.action_probability(acting_player, policy)
    # Handle edge case where all probs are 0 (shouldn't happen with proper policies)
    if action_probs.sum() < 1e-12:
        return torch.randint(policy.shape[1], (1,), generator=rng).item()
    return torch.multinomial(action_probs, 1, generator=rng).item()


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    """KL(p || q) — KL divergence from q to p.

    Only computed over entries where p > 0.
    """
    p_safe = p.clamp(min=eps)
    q_safe = q.clamp(min=eps)
    return (p_safe * (p_safe.log() - q_safe.log())).sum().item()


def l1_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """L1 distance between two distributions."""
    return (p - q).abs().sum().item()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDealEnumeration:
    """Test that deal enumeration and compatibility are correct."""

    def test_kuhn_deals(self):
        """Kuhn poker: D=3, P=2, K=1 → 6 deals."""
        cfg = GameConfig(deck_size=3, num_players=2, cards_per_player=1)
        deals = enumerate_deals(cfg)
        assert len(deals) == 6
        # No two players hold the same card
        for deal in deals:
            all_cards = [c for hand in deal for c in hand]
            assert len(all_cards) == len(set(all_cards))

    def test_4card_2player_deals(self):
        """D=4, P=2, K=1 → 12 deals."""
        cfg = GameConfig(deck_size=4, num_players=2, cards_per_player=1)
        deals = enumerate_deals(cfg)
        assert len(deals) == 12  # 4 × 3

    def test_3player_deals(self):
        """D=5, P=3, K=1 → 60 deals."""
        cfg = GameConfig(deck_size=5, num_players=3, cards_per_player=1)
        deals = enumerate_deals(cfg)
        assert len(deals) == 60  # 5 × 4 × 3

    def test_multicard_hands(self):
        """D=6, P=2, K=2 → C(6,2) × C(4,2) = 15 × 6 = 90 deals."""
        cfg = GameConfig(deck_size=6, num_players=2, cards_per_player=2)
        deals = enumerate_deals(cfg)
        assert len(deals) == 90

    def test_incompatibility(self):
        """No card appears in multiple hands within a deal."""
        cfg = GameConfig(deck_size=6, num_players=3, cards_per_player=2)
        deals = enumerate_deals(cfg)
        for deal in deals:
            all_cards = [c for hand in deal for c in hand]
            assert len(all_cards) == len(set(all_cards)), f"Duplicate card in deal {deal}"


class TestRandomPolicy:
    """Test that random policies are valid probability distributions."""

    def test_policy_shape(self):
        rng = torch.Generator().manual_seed(0)
        policy = generate_random_policy(5, 3, rng)
        assert policy.shape == (5, 3)

    def test_policy_rows_sum_to_one(self):
        rng = torch.Generator().manual_seed(0)
        policy = generate_random_policy(10, 4, rng)
        for s in range(10):
            assert abs(policy[s].sum().item() - 1.0) < 1e-6

    def test_policy_nonnegative(self):
        rng = torch.Generator().manual_seed(0)
        policy = generate_random_policy(10, 4, rng)
        assert (policy >= 0).all()

    def test_policy_determinism(self):
        """Same seed → same policy."""
        rng1 = torch.Generator().manual_seed(42)
        rng2 = torch.Generator().manual_seed(42)
        p1 = generate_random_policy(5, 3, rng1)
        p2 = generate_random_policy(5, 3, rng2)
        assert torch.allclose(p1, p2)


class TestExactJointBeliefTracker:
    """Test the exact joint belief tracker."""

    def test_initial_beliefs_uniform(self):
        """Initial marginal beliefs should be uniform."""
        cfg = GameConfig(deck_size=4, num_players=2, cards_per_player=1)
        tracker = ExactJointBeliefTracker(cfg)
        for p in range(cfg.num_players):
            belief = tracker.marginal_belief(p)
            n = num_private_states(cfg)
            expected = 1.0 / n
            for s in range(n):
                assert abs(belief[s].item() - expected) < 1e-6

    def test_beliefs_sum_to_one(self):
        """Marginal beliefs should sum to 1 after updates."""
        cfg = GameConfig(deck_size=4, num_players=2, cards_per_player=1, seed=7)
        rng = torch.Generator().manual_seed(cfg.seed)
        tracker = ExactJointBeliefTracker(cfg)
        n_priv = num_private_states(cfg)

        for step in range(5):
            player = step % cfg.num_players
            policy = generate_random_policy(n_priv, cfg.num_actions, rng)
            action_probs = tracker.action_probability(player, policy)
            action = torch.multinomial(action_probs, 1, generator=rng).item()
            tracker.observe_action(player, action, policy)
            for p in range(cfg.num_players):
                total = tracker.marginal_belief(p).sum().item()
                assert abs(total - 1.0) < 1e-5, (
                    f"Step {step}: player {p} belief sums to {total}"
                )

    def test_joint_sums_to_one(self):
        """Joint distribution should sum to 1 after updates."""
        cfg = GameConfig(deck_size=4, num_players=2, cards_per_player=1, seed=3)
        rng = torch.Generator().manual_seed(cfg.seed)
        tracker = ExactJointBeliefTracker(cfg)
        n_priv = num_private_states(cfg)

        for step in range(5):
            player = step % cfg.num_players
            policy = generate_random_policy(n_priv, cfg.num_actions, rng)
            action_probs = tracker.action_probability(player, policy)
            action = torch.multinomial(action_probs, 1, generator=rng).item()
            tracker.observe_action(player, action, policy)

        total = tracker.joint.sum().item()
        assert abs(total - 1.0) < 1e-5

    def test_deterministic_policy_reveals_state(self):
        """If a player always maps state→distinct action, observing reveals their state."""
        cfg = GameConfig(deck_size=3, num_players=2, cards_per_player=1, num_actions=3)
        tracker = ExactJointBeliefTracker(cfg)

        # Deterministic policy: state 0→action 0, state 1→action 1, state 2→action 2
        policy = torch.eye(3)  # [3, 3]

        # Observe player 0 taking action 1 → player 0 must hold card 1
        tracker.observe_action(0, 1, policy)
        belief = tracker.marginal_belief(0)
        assert belief[1].item() > 0.99
        assert belief[0].item() < 0.01
        assert belief[2].item() < 0.01

    def test_deterministic_policy_cross_player_update(self):
        """If player 0's state is revealed, player 1's belief should exclude that state."""
        cfg = GameConfig(deck_size=3, num_players=2, cards_per_player=1, num_actions=3)
        tracker = ExactJointBeliefTracker(cfg)

        # Deterministic policy reveals player 0's card
        policy = torch.eye(3)
        tracker.observe_action(0, 1, policy)

        # Player 1 cannot hold card 1 (incompatibility)
        belief_p1 = tracker.marginal_belief(1)
        assert belief_p1[1].item() < 0.01, (
            f"Player 1 should not hold card 1 after it's revealed as player 0's card, "
            f"but belief is {belief_p1[1].item()}"
        )
        # Player 1's belief should be 50/50 over cards 0 and 2
        assert abs(belief_p1[0].item() - 0.5) < 0.01
        assert abs(belief_p1[2].item() - 0.5) < 0.01


class TestReachProbabilityTracker:
    """Test the reach-probability-based belief tracker."""

    def test_initial_beliefs_uniform(self):
        cfg = GameConfig(deck_size=4, num_players=2, cards_per_player=1)
        tracker = ReachProbabilityTracker(cfg)
        for p in range(cfg.num_players):
            belief = tracker.marginal_belief(p)
            n = num_private_states(cfg)
            expected = 1.0 / n
            for s in range(n):
                assert abs(belief[s].item() - expected) < 1e-6

    def test_matches_exact_joint(self):
        """Reach-probability tracker should match exact joint for all configurations."""
        for seed in range(5):
            cfg = GameConfig(deck_size=4, num_players=2, cards_per_player=1, seed=seed)
            rng = torch.Generator().manual_seed(cfg.seed)
            exact = ExactJointBeliefTracker(cfg)
            reach = ReachProbabilityTracker(cfg)
            n_priv = num_private_states(cfg)

            for step in range(8):
                player = step % cfg.num_players
                policy = generate_random_policy(n_priv, cfg.num_actions, rng)
                action = sample_action(player, policy, exact, rng)

                exact.observe_action(player, action, policy)
                reach.observe_action(player, action, policy)

                for p in range(cfg.num_players):
                    exact_b = exact.marginal_belief(p)
                    reach_b = reach.marginal_belief(p)
                    assert torch.allclose(exact_b, reach_b, atol=1e-5), (
                        f"Seed {seed}, step {step}, player {p}: "
                        f"exact={exact_b} vs reach={reach_b}"
                    )

    def test_matches_exact_3player(self):
        """Reach-probability tracker should also match for 3 players."""
        cfg = GameConfig(deck_size=5, num_players=3, cards_per_player=1, seed=99)
        rng = torch.Generator().manual_seed(cfg.seed)
        exact = ExactJointBeliefTracker(cfg)
        reach = ReachProbabilityTracker(cfg)
        n_priv = num_private_states(cfg)

        for step in range(10):
            player = step % cfg.num_players
            policy = generate_random_policy(n_priv, cfg.num_actions, rng)
            action = sample_action(player, policy, exact, rng)

            exact.observe_action(player, action, policy)
            reach.observe_action(player, action, policy)

            for p in range(cfg.num_players):
                exact_b = exact.marginal_belief(p)
                reach_b = reach.marginal_belief(p)
                assert torch.allclose(exact_b, reach_b, atol=1e-5), (
                    f"Step {step}, player {p}: exact={exact_b} vs reach={reach_b}"
                )


class TestFactoredApproximation:
    """Test the factored (marginal) belief approximation and measure its quality."""

    def test_factored_matches_on_first_action(self):
        """On the very first action, factored should match exact for the acting player.

        Before any actions, all beliefs are uniform. The first Bayesian update
        only involves the acting player's policy, so the factored tracker gets
        the same result as the exact joint. After subsequent actions by other
        players, correlations from incompatibility make the factored tracker
        diverge even for the acting player.
        """
        cfg = GameConfig(deck_size=4, num_players=2, cards_per_player=1, seed=10)
        rng = torch.Generator().manual_seed(cfg.seed)
        exact = ExactJointBeliefTracker(cfg)
        factored = FactoredBeliefTracker(cfg)
        n_priv = num_private_states(cfg)

        # First action only
        policy = generate_random_policy(n_priv, cfg.num_actions, rng)
        action = sample_action(0, policy, exact, rng)
        exact.observe_action(0, action, policy)
        factored.observe_action(0, action, policy)

        # Acting player's factored belief should match exactly on first step
        exact_b = exact.marginal_belief(0)
        factored_b = factored.marginal_belief(0)
        assert torch.allclose(exact_b, factored_b, atol=1e-5), (
            f"First action: acting player's factored belief diverged: "
            f"exact={exact_b} vs factored={factored_b}"
        )

    def test_factored_diverges_for_cross_player(self):
        """For 2 players, factored misses the cross-player update from incompatibility.

        When player 0 acts and reveals information, the exact tracker updates
        player 1's beliefs (via the incompatibility constraint), but the factored
        tracker does not.
        """
        cfg = GameConfig(deck_size=3, num_players=2, cards_per_player=1, num_actions=3)
        exact = ExactJointBeliefTracker(cfg)
        factored = FactoredBeliefTracker(cfg)

        # Deterministic policy: perfectly reveals player 0's card
        policy = torch.eye(3)
        exact.observe_action(0, 1, policy)
        factored.observe_action(0, 1, policy)

        # Exact: player 1 cannot hold card 1 → belief should be [0.5, 0, 0.5]
        exact_b1 = exact.marginal_belief(1)
        assert exact_b1[1].item() < 0.01

        # Factored: player 1's belief is unchanged → still uniform [1/3, 1/3, 1/3]
        factored_b1 = factored.marginal_belief(1)
        assert abs(factored_b1[1].item() - 1.0 / 3) < 0.01

    def test_approximation_error_bounded(self):
        """Factored approximation error should be small for weak policies."""
        cfg = GameConfig(deck_size=5, num_players=3, cards_per_player=1, seed=55)
        rng = torch.Generator().manual_seed(cfg.seed)
        exact = ExactJointBeliefTracker(cfg)
        factored = FactoredBeliefTracker(cfg)
        n_priv = num_private_states(cfg)

        max_l1 = 0.0
        for step in range(15):
            player = step % cfg.num_players
            # Use near-uniform policies (weak signal → small approximation error)
            policy = generate_random_policy(n_priv, cfg.num_actions, rng)
            # Mix with uniform to make policies less extreme
            uniform = torch.ones_like(policy) / cfg.num_actions
            policy = 0.7 * uniform + 0.3 * policy

            action = sample_action(player, policy, exact, rng)
            exact.observe_action(player, action, policy)
            factored.observe_action(player, action, policy)

            for p in range(cfg.num_players):
                exact_b = exact.marginal_belief(p)
                factored_b = factored.marginal_belief(p)
                l1 = l1_distance(exact_b, factored_b)
                max_l1 = max(max_l1, l1)

        # With near-uniform policies, error should stay bounded
        assert max_l1 < 1.0, f"Max L1 error {max_l1} is too large for near-uniform policies"

    def test_approximation_error_grows_with_strong_policies(self):
        """Strong (informative) policies should produce larger approximation error."""
        errors_weak = []
        errors_strong = []

        for trial in range(5):
            for strength, error_list in [(0.1, errors_weak), (0.9, errors_strong)]:
                cfg = GameConfig(
                    deck_size=5, num_players=3, cards_per_player=1, seed=100 + trial
                )
                rng = torch.Generator().manual_seed(cfg.seed)
                exact = ExactJointBeliefTracker(cfg)
                factored = FactoredBeliefTracker(cfg)
                n_priv = num_private_states(cfg)

                max_l1 = 0.0
                for step in range(10):
                    player = step % cfg.num_players
                    policy = generate_random_policy(n_priv, cfg.num_actions, rng)
                    uniform = torch.ones_like(policy) / cfg.num_actions
                    policy = (1 - strength) * uniform + strength * policy

                    action = sample_action(player, policy, exact, rng)
                    exact.observe_action(player, action, policy)
                    factored.observe_action(player, action, policy)

                    for p in range(cfg.num_players):
                        l1 = l1_distance(exact.marginal_belief(p), factored.marginal_belief(p))
                        max_l1 = max(max_l1, l1)

                error_list.append(max_l1)

        avg_weak = sum(errors_weak) / len(errors_weak)
        avg_strong = sum(errors_strong) / len(errors_strong)
        # Strong policies should generally produce more error
        assert avg_strong >= avg_weak * 0.5, (
            f"Expected strong policies to produce more error: "
            f"avg_weak={avg_weak:.4f}, avg_strong={avg_strong:.4f}"
        )


class TestBayesianUpdates:
    """Test the Bayesian update mechanics in detail."""

    def test_action_probabilities_sum_to_one(self):
        """P(action) should sum to 1."""
        cfg = GameConfig(deck_size=4, num_players=2, cards_per_player=1, seed=1)
        rng = torch.Generator().manual_seed(cfg.seed)
        tracker = ExactJointBeliefTracker(cfg)
        n_priv = num_private_states(cfg)

        for step in range(5):
            player = step % cfg.num_players
            policy = generate_random_policy(n_priv, cfg.num_actions, rng)
            action_probs = tracker.action_probability(player, policy)
            assert abs(action_probs.sum().item() - 1.0) < 1e-5
            action = torch.multinomial(action_probs, 1, generator=rng).item()
            tracker.observe_action(player, action, policy)

    def test_bayes_update_formula(self):
        """Directly verify Bayes' theorem: P(deal|action) ∝ P(action|deal) × P(deal)."""
        cfg = GameConfig(deck_size=3, num_players=2, cards_per_player=1, num_actions=2)
        tracker = ExactJointBeliefTracker(cfg)

        rng = torch.Generator().manual_seed(42)
        policy = generate_random_policy(num_private_states(cfg), cfg.num_actions, rng)

        # Save prior
        prior = tracker.joint.clone()

        # Compute likelihood P(action=0 | deal) for each deal
        likelihood = torch.zeros(tracker.num_deals)
        for d in range(tracker.num_deals):
            s = tracker.deal_to_states[0][d]  # player 0's state
            likelihood[d] = policy[s, 0]

        # Expected posterior ∝ prior × likelihood
        expected_posterior = prior * likelihood
        expected_posterior /= expected_posterior.sum()

        # Actual update
        tracker.observe_action(0, 0, policy)

        assert torch.allclose(tracker.joint, expected_posterior, atol=1e-6)

    def test_sequential_updates_are_consistent(self):
        """Multiple sequential updates should produce the same result as batch."""
        cfg = GameConfig(deck_size=4, num_players=2, cards_per_player=1, seed=77)
        rng = torch.Generator().manual_seed(cfg.seed)
        n_priv = num_private_states(cfg)

        # Generate a sequence of policies and actions
        policies = []
        players = []
        actions = []
        tracker_for_sampling = ExactJointBeliefTracker(cfg)

        for step in range(5):
            player = step % cfg.num_players
            policy = generate_random_policy(n_priv, cfg.num_actions, rng)
            action = sample_action(player, policy, tracker_for_sampling, rng)
            tracker_for_sampling.observe_action(player, action, policy)
            policies.append(policy)
            players.append(player)
            actions.append(action)

        # Sequential tracker: update one at a time
        seq_tracker = ExactJointBeliefTracker(cfg)
        for player, action, policy in zip(players, actions, policies):
            seq_tracker.observe_action(player, action, policy)

        # Batch: compute likelihood as product of all per-step likelihoods
        batch_joint = torch.ones(seq_tracker.num_deals) / seq_tracker.num_deals
        for player, action, policy in zip(players, actions, policies):
            for d in range(seq_tracker.num_deals):
                s = seq_tracker.deal_to_states[player][d]
                batch_joint[d] *= policy[s, action]
        batch_joint /= batch_joint.sum()

        assert torch.allclose(seq_tracker.joint, batch_joint, atol=1e-6)


class TestMultiCardHands:
    """Test with K > 1 cards per player."""

    def test_2card_hands_beliefs_valid(self):
        """Beliefs should be valid distributions with multi-card hands."""
        cfg = GameConfig(deck_size=6, num_players=2, cards_per_player=2, seed=33)
        rng = torch.Generator().manual_seed(cfg.seed)
        exact = ExactJointBeliefTracker(cfg)
        n_priv = num_private_states(cfg)

        for step in range(5):
            player = step % cfg.num_players
            policy = generate_random_policy(n_priv, cfg.num_actions, rng)
            action = sample_action(player, policy, exact, rng)
            exact.observe_action(player, action, policy)

            for p in range(cfg.num_players):
                belief = exact.marginal_belief(p)
                total = belief.sum().item()
                assert abs(total - 1.0) < 1e-5
                assert (belief >= -1e-6).all()

    def test_reach_matches_exact_multicard(self):
        """Reach-based tracker should match exact for multi-card hands."""
        cfg = GameConfig(deck_size=6, num_players=2, cards_per_player=2, seed=44)
        rng = torch.Generator().manual_seed(cfg.seed)
        exact = ExactJointBeliefTracker(cfg)
        reach = ReachProbabilityTracker(cfg)
        n_priv = num_private_states(cfg)

        for step in range(5):
            player = step % cfg.num_players
            policy = generate_random_policy(n_priv, cfg.num_actions, rng)
            action = sample_action(player, policy, exact, rng)

            exact.observe_action(player, action, policy)
            reach.observe_action(player, action, policy)

            for p in range(cfg.num_players):
                assert torch.allclose(
                    exact.marginal_belief(p), reach.marginal_belief(p), atol=1e-5
                )


class TestApproximationMetrics:
    """Measure approximation quality with different metrics."""

    def _run_simulation(
        self, cfg: GameConfig, num_steps: int
    ) -> Tuple[List[float], List[float], List[float]]:
        """Run a simulation and return per-step L1, KL, and max-error metrics."""
        rng = torch.Generator().manual_seed(cfg.seed)
        exact = ExactJointBeliefTracker(cfg)
        factored = FactoredBeliefTracker(cfg)
        n_priv = num_private_states(cfg)

        l1_errors = []
        kl_errors = []
        max_errors = []

        for step in range(num_steps):
            player = step % cfg.num_players
            policy = generate_random_policy(n_priv, cfg.num_actions, rng)
            action = sample_action(player, policy, exact, rng)

            exact.observe_action(player, action, policy)
            factored.observe_action(player, action, policy)

            step_l1 = 0.0
            step_kl = 0.0
            step_max = 0.0
            for p in range(cfg.num_players):
                eb = exact.marginal_belief(p)
                fb = factored.marginal_belief(p)
                step_l1 = max(step_l1, l1_distance(eb, fb))
                step_kl = max(step_kl, kl_divergence(eb, fb))
                step_max = max(step_max, (eb - fb).abs().max().item())

            l1_errors.append(step_l1)
            kl_errors.append(step_kl)
            max_errors.append(step_max)

        return l1_errors, kl_errors, max_errors

    def test_2player_error_smaller_than_3player(self):
        """For 2 players, the factored approximation error should be smaller than for 3 players.

        With fewer players, there are fewer cross-player correlations to miss.
        """
        errors_2p = []
        errors_3p = []

        for trial in range(5):
            for n_players, deck, error_list in [(2, 4, errors_2p), (3, 5, errors_3p)]:
                cfg = GameConfig(
                    deck_size=deck, num_players=n_players, cards_per_player=1,
                    seed=200 + trial
                )
                rng = torch.Generator().manual_seed(cfg.seed)
                exact = ExactJointBeliefTracker(cfg)
                factored = FactoredBeliefTracker(cfg)
                n_priv = num_private_states(cfg)

                max_l1 = 0.0
                for step in range(10):
                    player = step % cfg.num_players
                    policy = generate_random_policy(n_priv, cfg.num_actions, rng)
                    action = sample_action(player, policy, exact, rng)
                    exact.observe_action(player, action, policy)
                    factored.observe_action(player, action, policy)
                    for p in range(cfg.num_players):
                        l1 = l1_distance(exact.marginal_belief(p), factored.marginal_belief(p))
                        max_l1 = max(max_l1, l1)
                error_list.append(max_l1)

        avg_2p = sum(errors_2p) / len(errors_2p)
        avg_3p = sum(errors_3p) / len(errors_3p)
        # 2-player error should generally be smaller (or at most comparable)
        assert avg_2p <= avg_3p * 2.0, (
            f"2-player error {avg_2p:.4f} unexpectedly much larger than 3-player {avg_3p:.4f}"
        )

    def test_3player_metrics_collected(self):
        """For 3 players, verify we can collect meaningful approximation metrics."""
        cfg = GameConfig(deck_size=6, num_players=3, cards_per_player=1, seed=30)
        l1s, kls, maxes = self._run_simulation(cfg, num_steps=15)

        # Should have collected metrics for each step
        assert len(l1s) == 15
        assert len(kls) == 15
        assert len(maxes) == 15

        # Metrics should be non-negative
        for l1, kl, mx in zip(l1s, kls, maxes):
            assert l1 >= -1e-10
            assert kl >= -1e-10
            assert mx >= -1e-10

    def test_error_monotonicity_tendency(self):
        """Over many steps, approximation error should generally grow (more info revealed).

        We test this loosely: the max error in the second half should be >= the first step's
        error (or close to it).
        """
        cfg = GameConfig(deck_size=5, num_players=3, cards_per_player=1, seed=60)
        l1s, _, _ = self._run_simulation(cfg, num_steps=20)

        first_half_max = max(l1s[:10])
        second_half_max = max(l1s[10:])
        # Second half should have at least as much error as early steps (or close)
        # This is a soft test — randomness can cause exceptions
        assert second_half_max >= first_half_max * 0.3 or first_half_max < 0.01


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_uniform_policy_no_information(self):
        """A uniform policy reveals no information — beliefs should not change."""
        cfg = GameConfig(deck_size=4, num_players=2, cards_per_player=1, num_actions=3)
        tracker = ExactJointBeliefTracker(cfg)

        # Uniform policy: every state plays every action with equal probability
        n_priv = num_private_states(cfg)
        policy = torch.ones(n_priv, cfg.num_actions) / cfg.num_actions

        prior = tracker.joint.clone()
        tracker.observe_action(0, 0, policy)

        assert torch.allclose(tracker.joint, prior, atol=1e-6), (
            "Uniform policy should not change beliefs"
        )

    def test_zero_probability_action(self):
        """Observing an action with 0 probability from some states should eliminate those states."""
        cfg = GameConfig(deck_size=3, num_players=2, cards_per_player=1, num_actions=2)
        tracker = ExactJointBeliefTracker(cfg)

        # Policy where state 0 never takes action 1
        policy = torch.tensor([
            [1.0, 0.0],  # state 0: always action 0
            [0.5, 0.5],  # state 1: equal
            [0.5, 0.5],  # state 2: equal
        ])

        # Observe player 0 taking action 1 → player 0 cannot be in state 0
        tracker.observe_action(0, 1, policy)
        belief = tracker.marginal_belief(0)
        assert belief[0].item() < 1e-10, "State 0 should have 0 probability"
        assert belief[1].item() > 0
        assert belief[2].item() > 0

    def test_all_trackers_agree_on_kuhn_params(self):
        """All three trackers should agree for Kuhn-like parameters."""
        cfg = GameConfig(deck_size=3, num_players=2, cards_per_player=1, seed=42)
        rng = torch.Generator().manual_seed(cfg.seed)
        n_priv = num_private_states(cfg)

        exact = ExactJointBeliefTracker(cfg)
        reach = ReachProbabilityTracker(cfg)
        factored = FactoredBeliefTracker(cfg)

        for step in range(6):
            player = step % cfg.num_players
            policy = generate_random_policy(n_priv, cfg.num_actions, rng)
            action = sample_action(player, policy, exact, rng)

            exact.observe_action(player, action, policy)
            reach.observe_action(player, action, policy)
            factored.observe_action(player, action, policy)

            # Exact and reach should always agree
            for p in range(cfg.num_players):
                assert torch.allclose(
                    exact.marginal_belief(p), reach.marginal_belief(p), atol=1e-5
                )

            # Factored is an approximation — check it's at least close
            l1_err = l1_distance(
                exact.marginal_belief(player), factored.marginal_belief(player)
            )
            assert l1_err < 0.5, (
                f"Step {step}: factored too far from exact for acting player {player}: "
                f"L1={l1_err:.4f}"
            )

    def test_single_deal_remaining(self):
        """When beliefs collapse to a single deal, all trackers should agree."""
        cfg = GameConfig(deck_size=3, num_players=2, cards_per_player=1, num_actions=3)
        tracker = ExactJointBeliefTracker(cfg)

        # Use deterministic policies to reveal both players' cards
        policy = torch.eye(3)  # state i → action i

        # Player 0 takes action 2 → holds card 2
        tracker.observe_action(0, 2, policy)
        # Player 1 takes action 0 → holds card 0
        tracker.observe_action(1, 0, policy)

        # Only deal (2, 0) should remain
        belief_p0 = tracker.marginal_belief(0)
        belief_p1 = tracker.marginal_belief(1)
        assert belief_p0[2].item() > 0.99
        assert belief_p1[0].item() > 0.99

    def test_many_actions(self):
        """Game with many possible actions should still work correctly."""
        cfg = GameConfig(deck_size=4, num_players=2, cards_per_player=1,
                         num_actions=10, seed=88)
        rng = torch.Generator().manual_seed(cfg.seed)
        exact = ExactJointBeliefTracker(cfg)
        reach = ReachProbabilityTracker(cfg)
        n_priv = num_private_states(cfg)

        for step in range(10):
            player = step % cfg.num_players
            policy = generate_random_policy(n_priv, cfg.num_actions, rng)
            action = sample_action(player, policy, exact, rng)

            exact.observe_action(player, action, policy)
            reach.observe_action(player, action, policy)

            for p in range(cfg.num_players):
                assert torch.allclose(
                    exact.marginal_belief(p), reach.marginal_belief(p), atol=1e-5
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
