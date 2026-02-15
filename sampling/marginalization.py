"""Sampling-based marginalization for scalable belief state computation.

The exact joint distribution over deals has size D!/(D-PK)! for D cards,
P players, K cards per player. For NLHE this is ~1.76M (2 players) or
astronomical for 3+. Sampling approaches trade exactness for scalability.

All trackers implement the same interface as ExactJointBeliefTracker:
  - observe_action(acting_player, action, policy) -> None
  - marginal_belief(player) -> Tensor[n_private_states]

And additionally expose:
  - effective_sample_size() -> float (ESS, measures sample diversity)
  - sparsity_ratio() -> float (fraction of deals with non-negligible weight)
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

@dataclass
class GameConfig:
    """Configuration for a parameterized card game (mirrors test_marginalization)."""
    deck_size: int = 6
    num_players: int = 2
    cards_per_player: int = 1
    num_actions: int = 2
    seed: int = 42

    def __post_init__(self):
        assert self.deck_size >= self.num_players * self.cards_per_player


# ---------------------------------------------------------------------------
# Deal enumeration utilities (shared across all trackers)
# ---------------------------------------------------------------------------

def enumerate_deals(cfg: GameConfig) -> List[Tuple[Tuple[int, ...], ...]]:
    """Enumerate all valid deals: tuples of P hands, each hand is K cards."""
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


def deal_to_private_states(
    deals: List[Tuple[Tuple[int, ...], ...]], cfg: GameConfig
) -> Dict[int, List[int]]:
    """Map each deal index to each player's private state index."""
    all_hands = list(itertools.combinations(range(cfg.deck_size), cfg.cards_per_player))
    hand_to_idx = {h: i for i, h in enumerate(all_hands)}
    result = {}
    for p in range(cfg.num_players):
        result[p] = [hand_to_idx[deals[d][p]] for d in range(len(deals))]
    return result


def num_private_states(cfg: GameConfig) -> int:
    return math.comb(cfg.deck_size, cfg.cards_per_player)


def num_total_deals(cfg: GameConfig) -> int:
    """Total number of valid deals."""
    n = cfg.deck_size
    result = 1
    for p in range(cfg.num_players):
        result *= math.comb(n - p * cfg.cards_per_player, cfg.cards_per_player)
    return result


# ---------------------------------------------------------------------------
# 1. Projection Sampling
# ---------------------------------------------------------------------------

class ProjectionSampler:
    """Approximate marginalization via projection onto per-player marginals.

    Instead of maintaining the full joint, we maintain per-player marginal
    beliefs and reconstruct an approximate joint by:
    1. Sampling deals proportional to the product of marginal probabilities
    2. Filtering for compatibility (no shared cards)
    3. Using importance weights to correct for the product-of-marginals bias

    This is effective when player hands are approximately independent —
    which is the case for large decks where the incompatibility constraint
    is weak (each player uses only a small fraction of the deck).

    The key insight: for NLHE with 1326 possible hands per player, the
    overlap constraint only excludes ~4% of hand combinations, so the
    product of marginals is a good proposal distribution.
    """

    def __init__(self, cfg: GameConfig, num_samples: int = 1000):
        self.cfg = cfg
        self.num_samples = num_samples
        self.n_private = num_private_states(cfg)
        self.deals = enumerate_deals(cfg)
        self.num_deals = len(self.deals)
        self.deal_to_states = deal_to_private_states(self.deals, cfg)

        # Per-player marginal beliefs (the projection)
        self.marginals = [
            torch.ones(self.n_private) / self.n_private
            for _ in range(cfg.num_players)
        ]

        # Per-player reach probabilities (for reconstructing the joint)
        self.reach = [torch.ones(self.n_private) for _ in range(cfg.num_players)]

        # Build deal-to-hand-index mapping for fast lookup
        # deal_hands[d] = (hand_idx_p0, hand_idx_p1, ...)
        self.deal_hands = []
        for d in range(self.num_deals):
            hands = tuple(self.deal_to_states[p][d] for p in range(cfg.num_players))
            self.deal_hands.append(hands)

        # Precompute compatibility: which hand combinations are valid deals
        # For each pair of hand indices, is it a valid deal?
        self._build_compatibility_index()

    def _build_compatibility_index(self):
        """Build a set of valid hand-index tuples for O(1) compatibility checks."""
        self.valid_deal_set = set(self.deal_hands)

    def observe_action(
        self, acting_player: int, action: int, policy: torch.Tensor
    ) -> None:
        """Update per-player reach and marginal beliefs after an action.

        Only the acting player's beliefs change (direct Bayesian update).
        Other players' beliefs are updated indirectly via the joint reconstruction
        when marginal_belief() is called.
        """
        # Update reach for the acting player
        likelihood = policy[:, action]  # [n_private_states]
        self.reach[acting_player] = self.reach[acting_player] * likelihood

        # Update marginal for acting player directly (fast, exact for this player)
        updated = self.marginals[acting_player] * likelihood
        total = updated.sum()
        if total > 0:
            updated /= total
        self.marginals[acting_player] = updated

    def marginal_belief(self, player: int) -> torch.Tensor:
        """Compute marginal belief via importance-weighted sampling from product of marginals.

        Steps:
        1. Sample hand combinations from the product of per-player marginals
        2. Filter for compatibility (valid deals)
        3. Compute importance weights to correct for proposal bias
        4. Marginalize over the sampled+weighted deals
        """
        # For small games, fall back to exact computation
        if self.num_deals <= self.num_samples * 2:
            return self._exact_marginal(player)

        return self._sampled_marginal(player)

    def _exact_marginal(self, player: int) -> torch.Tensor:
        """Exact marginal computation (for small games or validation)."""
        # Compute joint from reaches
        joint = torch.ones(self.num_deals)
        for d in range(self.num_deals):
            for p in range(self.cfg.num_players):
                s = self.deal_to_states[p][d]
                joint[d] *= self.reach[p][s]

        total = joint.sum()
        if total > 0:
            joint /= total

        belief = torch.zeros(self.n_private)
        for d in range(self.num_deals):
            s = self.deal_to_states[player][d]
            belief[s] += joint[d]
        return belief

    def _sampled_marginal(self, player: int) -> torch.Tensor:
        """Sampling-based marginal using product-of-marginals proposal."""
        belief = torch.zeros(self.n_private)

        # Proposal: product of per-player reach-weighted marginals
        proposal_marginals = []
        for p in range(self.cfg.num_players):
            m = self.reach[p].clone()
            total = m.sum()
            if total > 0:
                m /= total
            proposal_marginals.append(m)

        total_weight = 0.0

        for _ in range(self.num_samples):
            # Sample hand for each player from their marginal
            hands = []
            proposal_prob = 1.0
            for p in range(self.cfg.num_players):
                h = torch.multinomial(proposal_marginals[p], 1).item()
                hands.append(h)
                proposal_prob *= proposal_marginals[p][h].item()

            hand_tuple = tuple(hands)

            # Check compatibility
            if hand_tuple not in self.valid_deal_set:
                continue

            # Importance weight: true_prob / proposal_prob
            # true_prob ∝ product of reaches
            true_prob = 1.0
            for p in range(self.cfg.num_players):
                true_prob *= self.reach[p][hands[p]].item()

            if proposal_prob > 0:
                weight = true_prob / proposal_prob
            else:
                weight = 0.0

            total_weight += weight
            belief[hands[player]] += weight

        if total_weight > 0:
            belief /= total_weight

        return belief

    def effective_sample_size(self) -> float:
        """Estimate effective sample size from the most recent sampling."""
        # ESS ≈ (sum of weights)^2 / sum of weights^2
        # Higher ESS = more diverse samples = better approximation
        return float(self.num_samples)  # Placeholder

    def sparsity_ratio(self) -> float:
        """Fraction of deals with non-negligible probability."""
        joint = torch.ones(self.num_deals)
        for d in range(self.num_deals):
            for p in range(self.cfg.num_players):
                s = self.deal_to_states[p][d]
                joint[d] *= self.reach[p][s]
        total = joint.sum()
        if total > 0:
            joint /= total
        return (joint > 1e-8).float().mean().item()


# ---------------------------------------------------------------------------
# 2. Gibbs Sampling
# ---------------------------------------------------------------------------

class GibbsSampler:
    """Approximate marginalization via Gibbs sampling on the joint deal space.

    Gibbs sampling iteratively re-samples each player's hand conditioned on
    all other players' hands. The conditional distribution is:

        P(hand_p | hands_{-p}, history) ∝ reach_p[hand_p] × compat(hand_p, hands_{-p})

    where compat() = 1 if hand_p doesn't share cards with any other hand, 0 otherwise.

    Advantages over projection sampling:
    - Correctly handles correlations between players' hands
    - The compatibility constraint is automatically enforced
    - Works well when hands are strongly correlated (e.g., small decks)

    Disadvantages:
    - Requires burn-in period
    - Can mix slowly when beliefs are peaked (few valid transitions)
    - Sequential — each step depends on the previous state
    """

    def __init__(
        self,
        cfg: GameConfig,
        num_samples: int = 1000,
        burn_in: int = 100,
        thin: int = 1,
    ):
        self.cfg = cfg
        self.num_samples = num_samples
        self.burn_in = burn_in
        self.thin = thin
        self.n_private = num_private_states(cfg)
        self.deals = enumerate_deals(cfg)
        self.num_deals = len(self.deals)
        self.deal_to_states = deal_to_private_states(self.deals, cfg)

        # Per-player reach probabilities
        self.reach = [torch.ones(self.n_private) for _ in range(cfg.num_players)]

        # All possible hands
        self.all_hands = list(
            itertools.combinations(range(cfg.deck_size), cfg.cards_per_player)
        )

        # Map from hand tuple to set of cards for fast overlap checking
        self.hand_cards = {h: set(h) for h in self.all_hands}

        # Map from hand tuple to hand index
        self.hand_to_idx = {h: i for i, h in enumerate(self.all_hands)}

    def observe_action(
        self, acting_player: int, action: int, policy: torch.Tensor
    ) -> None:
        """Update reach probabilities for the acting player."""
        likelihood = policy[:, action]
        self.reach[acting_player] = self.reach[acting_player] * likelihood

    def marginal_belief(self, player: int) -> torch.Tensor:
        """Compute marginal belief via Gibbs sampling."""
        # For small games, use exact computation
        if self.num_deals <= (self.num_samples + self.burn_in) * 2:
            return self._exact_marginal(player)

        return self._gibbs_marginal(player)

    def _exact_marginal(self, player: int) -> torch.Tensor:
        """Exact marginal (for small games or validation)."""
        joint = torch.ones(self.num_deals)
        for d in range(self.num_deals):
            for p in range(self.cfg.num_players):
                s = self.deal_to_states[p][d]
                joint[d] *= self.reach[p][s]

        total = joint.sum()
        if total > 0:
            joint /= total

        belief = torch.zeros(self.n_private)
        for d in range(self.num_deals):
            s = self.deal_to_states[player][d]
            belief[s] += joint[d]
        return belief

    def _gibbs_marginal(self, player: int) -> torch.Tensor:
        """Gibbs sampling for marginal belief estimation."""
        belief = torch.zeros(self.n_private)

        # Initialize with a random valid deal
        current_hands = self._random_valid_deal()
        if current_hands is None:
            # No valid deals — return uniform
            return torch.ones(self.n_private) / self.n_private

        total_steps = self.burn_in + self.num_samples * self.thin
        sample_count = 0

        for step in range(total_steps):
            # Pick a random player to resample
            p = step % self.cfg.num_players

            # Compute the conditional: P(hand_p | hands_{-p})
            other_cards = set()
            for q in range(self.cfg.num_players):
                if q != p:
                    other_cards |= self.hand_cards[current_hands[q]]

            # Score each valid hand for player p
            scores = torch.zeros(len(self.all_hands))
            for h_idx, hand in enumerate(self.all_hands):
                if self.hand_cards[hand].isdisjoint(other_cards):
                    scores[h_idx] = self.reach[p][h_idx]

            total = scores.sum()
            if total > 0:
                scores /= total
                # Sample new hand for player p
                new_h_idx = torch.multinomial(scores, 1).item()
                new_hands = list(current_hands)
                new_hands[p] = self.all_hands[new_h_idx]
                current_hands = tuple(new_hands)

            # Collect sample after burn-in, with thinning
            if step >= self.burn_in and (step - self.burn_in) % self.thin == 0:
                h_idx = self.hand_to_idx[current_hands[player]]
                belief[h_idx] += 1.0
                sample_count += 1

        if sample_count > 0:
            belief /= sample_count

        return belief

    def _random_valid_deal(self) -> Optional[Tuple[Tuple[int, ...], ...]]:
        """Pick a random valid deal, weighted by reach probabilities."""
        if not self.deals:
            return None

        # Weight by product of reaches
        weights = torch.zeros(self.num_deals)
        for d in range(self.num_deals):
            w = 1.0
            for p in range(self.cfg.num_players):
                s = self.deal_to_states[p][d]
                w *= self.reach[p][s].item()
            weights[d] = w

        total = weights.sum()
        if total <= 0:
            # Fall back to uniform
            d_idx = torch.randint(self.num_deals, (1,)).item()
        else:
            d_idx = torch.multinomial(weights, 1).item()

        return self.deals[d_idx]

    def effective_sample_size(self) -> float:
        return float(self.num_samples)

    def sparsity_ratio(self) -> float:
        joint = torch.ones(self.num_deals)
        for d in range(self.num_deals):
            for p in range(self.cfg.num_players):
                s = self.deal_to_states[p][d]
                joint[d] *= self.reach[p][s]
        total = joint.sum()
        if total > 0:
            joint /= total
        return (joint > 1e-8).float().mean().item()


# ---------------------------------------------------------------------------
# 3. Reach-Guided Sparse Marginalization
# ---------------------------------------------------------------------------

class ReachGuidedSparse:
    """Sparse marginalization using reach probabilities to prune the deal space.

    The key insight: after several rounds of play, most deals have near-zero
    probability because the observed actions are inconsistent with many possible
    hands. Rather than maintaining the full joint (or sampling from it), we
    can prune deals whose reach probability falls below a threshold and only
    track the "active" set.

    This is particularly effective when policies are correlated — if certain
    hands are clearly strong (always bet) or clearly weak (always fold), then
    observing a bet quickly eliminates weak hands, making the active set small.

    The pruning criterion combines per-player reaches:
        weight(deal) = Π_p reach_p[hand_p(deal)]
    Deals with weight < threshold × max_weight are pruned.

    For games with correlated policies (which is the norm — good hands play
    differently from bad hands), this can reduce the effective deal space by
    orders of magnitude.

    Periodically, we can "reconstitute" the full joint from the active set
    to check that we haven't pruned too aggressively (the active set should
    capture >99% of the probability mass).
    """

    def __init__(
        self,
        cfg: GameConfig,
        prune_threshold: float = 1e-6,
        min_active_fraction: float = 0.01,
    ):
        self.cfg = cfg
        self.prune_threshold = prune_threshold
        self.min_active_fraction = min_active_fraction
        self.n_private = num_private_states(cfg)
        self.deals = enumerate_deals(cfg)
        self.num_deals = len(self.deals)
        self.deal_to_states = deal_to_private_states(self.deals, cfg)

        # Sparse representation: map from deal_index to weight
        # Initially all deals are active with equal weight
        self.active_weights: Dict[int, float] = {
            d: 1.0 / self.num_deals for d in range(self.num_deals)
        }

        # Per-player reach probabilities (maintained for reconstruction)
        self.reach = [torch.ones(self.n_private) for _ in range(cfg.num_players)]

        # Statistics
        self.total_pruned = 0
        self.prune_history: List[Tuple[int, int]] = []  # (step, active_count)

    def observe_action(
        self, acting_player: int, action: int, policy: torch.Tensor
    ) -> None:
        """Update weights and prune low-probability deals."""
        likelihood = policy[:, action]

        # Update reach
        self.reach[acting_player] = self.reach[acting_player] * likelihood

        # Update active weights
        new_weights = {}
        for d, w in self.active_weights.items():
            s = self.deal_to_states[acting_player][d]
            new_w = w * likelihood[s].item()
            if new_w > 0:
                new_weights[d] = new_w
        self.active_weights = new_weights

        # Normalize
        total = sum(self.active_weights.values())
        if total > 0:
            for d in self.active_weights:
                self.active_weights[d] /= total

        # Prune
        self._prune()

    def _prune(self):
        """Remove deals with negligible probability."""
        if not self.active_weights:
            return

        max_weight = max(self.active_weights.values())
        threshold = self.prune_threshold * max_weight

        # Ensure we keep at least min_active_fraction of deals
        min_active = max(1, int(self.min_active_fraction * self.num_deals))

        # Sort by weight, keep at least min_active
        sorted_deals = sorted(
            self.active_weights.items(), key=lambda x: x[1], reverse=True
        )

        new_weights = {}
        for i, (d, w) in enumerate(sorted_deals):
            if w >= threshold or i < min_active:
                new_weights[d] = w
            else:
                self.total_pruned += 1

        # Re-normalize
        total = sum(new_weights.values())
        if total > 0:
            for d in new_weights:
                new_weights[d] /= total

        before = len(self.active_weights)
        self.active_weights = new_weights
        after = len(self.active_weights)

        if before != after:
            self.prune_history.append((self.total_pruned, after))

    def marginal_belief(self, player: int) -> torch.Tensor:
        """Compute marginal belief from the sparse active set."""
        belief = torch.zeros(self.n_private)
        for d, w in self.active_weights.items():
            s = self.deal_to_states[player][d]
            belief[s] += w
        # Normalize (should already be ~1 but might have small numerical drift)
        total = belief.sum()
        if total > 0:
            belief /= total
        return belief

    def active_count(self) -> int:
        """Number of currently active (non-pruned) deals."""
        return len(self.active_weights)

    def coverage(self) -> float:
        """Fraction of total probability mass captured by the active set.

        Computes the probability mass of active deals under the full
        (unpruned) joint distribution. A value close to 1.0 means pruning
        hasn't lost significant mass.
        """
        # Reconstruct full joint from reaches
        full_joint = torch.ones(self.num_deals)
        for d in range(self.num_deals):
            for p in range(self.cfg.num_players):
                s = self.deal_to_states[p][d]
                full_joint[d] *= self.reach[p][s].item()

        total = full_joint.sum()
        if total <= 0:
            return 1.0

        active_mass = sum(full_joint[d].item() for d in self.active_weights)
        return active_mass / total.item()

    def effective_sample_size(self) -> float:
        """ESS based on weight distribution of active deals."""
        if not self.active_weights:
            return 0.0
        weights = list(self.active_weights.values())
        total = sum(weights)
        if total <= 0:
            return 0.0
        sum_sq = sum(w * w for w in weights)
        return (total * total) / sum_sq

    def sparsity_ratio(self) -> float:
        """Fraction of deals that are still active."""
        return len(self.active_weights) / max(1, self.num_deals)

    def reconstitute(self) -> Dict[int, float]:
        """Rebuild weights from reaches for all originally valid deals.

        Useful as a "garbage collection" step — if pruning has been too
        aggressive, this brings back deals that have become relevant again.
        """
        new_weights = {}
        for d in range(self.num_deals):
            w = 1.0
            for p in range(self.cfg.num_players):
                s = self.deal_to_states[p][d]
                w *= self.reach[p][s].item()
            if w > 0:
                new_weights[d] = w

        total = sum(new_weights.values())
        if total > 0:
            for d in new_weights:
                new_weights[d] /= total

        self.active_weights = new_weights
        self._prune()
        return self.active_weights


# ---------------------------------------------------------------------------
# 4. Reach-Correlated Sparse Sampler
# ---------------------------------------------------------------------------

class ReachCorrelatedSampler:
    """Smart sampler that exploits policy correlations to contain marginalization cost.

    When policies are correlated between players (e.g., all players with strong
    hands bet, all with weak hands fold), the effective joint distribution is
    concentrated on a small subset of deals. This sampler detects and exploits
    this structure.

    The approach:
    1. Maintain per-player reach vectors (compact: O(n_private) per player)
    2. Identify "hot" private states: states with significant reach probability
    3. Only enumerate deals composed of hot states
    4. Weight by the product of reaches

    This gives O(H^P) effective deals where H is the number of hot states
    per player, vs O(N^P) for the full enumeration where N = n_private_states.

    When policies are strongly correlated (e.g., after a bet, most weak hands
    are eliminated), H << N, giving massive savings.

    The hot-state threshold adapts based on the reach distribution:
    - Peaked distribution (few hot states): aggressive pruning
    - Flat distribution (many hot states): conservative pruning
    """

    def __init__(
        self,
        cfg: GameConfig,
        mass_threshold: float = 0.99,
    ):
        self.cfg = cfg
        self.mass_threshold = mass_threshold  # Keep states covering this much mass
        self.n_private = num_private_states(cfg)
        self.deals = enumerate_deals(cfg)
        self.num_deals = len(self.deals)
        self.deal_to_states = deal_to_private_states(self.deals, cfg)

        # Per-player reach probabilities
        self.reach = [torch.ones(self.n_private) for _ in range(cfg.num_players)]

        # All possible hands and card sets for compatibility
        self.all_hands = list(
            itertools.combinations(range(cfg.deck_size), cfg.cards_per_player)
        )
        self.hand_cards = [set(h) for h in self.all_hands]

    def observe_action(
        self, acting_player: int, action: int, policy: torch.Tensor
    ) -> None:
        likelihood = policy[:, action]
        self.reach[acting_player] = self.reach[acting_player] * likelihood

    def _hot_states(self, player: int) -> List[int]:
        """Identify private states that collectively cover mass_threshold of reach."""
        reach = self.reach[player]
        total = reach.sum()
        if total <= 0:
            return list(range(self.n_private))

        normalized = reach / total
        # Sort by probability descending
        sorted_indices = torch.argsort(normalized, descending=True)

        hot = []
        cumulative = 0.0
        for idx in sorted_indices:
            hot.append(idx.item())
            cumulative += normalized[idx].item()
            if cumulative >= self.mass_threshold:
                break

        return hot

    def marginal_belief(self, player: int) -> torch.Tensor:
        """Compute marginal belief by enumerating only deals over hot states."""
        # Get hot states for each player
        hot_per_player = [self._hot_states(p) for p in range(self.cfg.num_players)]

        # Enumerate compatible deals over hot states only
        belief = torch.zeros(self.n_private)
        total_weight = 0.0

        # Generate all combinations of hot states
        for combo in itertools.product(*hot_per_player):
            # Check compatibility: no card overlap
            cards_used = set()
            compatible = True
            for p_idx, h_idx in enumerate(combo):
                hand_c = self.hand_cards[h_idx]
                if not cards_used.isdisjoint(hand_c):
                    compatible = False
                    break
                cards_used |= hand_c

            if not compatible:
                continue

            # Weight = product of reaches
            weight = 1.0
            for p_idx, h_idx in enumerate(combo):
                weight *= self.reach[p_idx][h_idx].item()

            total_weight += weight
            belief[combo[player]] += weight

        if total_weight > 0:
            belief /= total_weight

        return belief

    def hot_state_counts(self) -> List[int]:
        """Number of hot states per player."""
        return [len(self._hot_states(p)) for p in range(self.cfg.num_players)]

    def effective_deal_count(self) -> int:
        """Approximate number of deals that would be enumerated."""
        counts = self.hot_state_counts()
        result = 1
        for c in counts:
            result *= c
        return result

    def sparsity_ratio(self) -> float:
        return self.effective_deal_count() / max(1, self.num_deals)
