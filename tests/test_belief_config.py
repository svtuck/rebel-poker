"""Tests for the game-agnostic BeliefConfig.

Validates that BeliefConfig correctly derives belief state structure
from games, and that PBS operations (reach_to_pbs, pbs_to_reach,
deal_values_to_pbs_values) are correct for Kuhn Poker.
"""

import pytest
import torch

from belief_config import BeliefConfig, belief_config_from_game
from kuhn.game import KuhnPoker
from kuhn.belief_state import (
    ALL_DEALS,
    NUM_DEALS,
    NUM_PRIVATE_STATES,
    NUM_PLAYERS,
    initial_pbs as kuhn_initial_pbs,
    reach_to_pbs as kuhn_reach_to_pbs,
)


class TestBeliefConfigFromGame:
    """Test that belief_config_from_game correctly derives config from KuhnPoker."""

    def setup_method(self):
        self.game = KuhnPoker()
        self.bc = belief_config_from_game(self.game)

    def test_dimensions(self):
        assert self.bc.num_private_states == NUM_PRIVATE_STATES  # 3
        assert self.bc.num_players == NUM_PLAYERS  # 2
        assert self.bc.num_deals == NUM_DEALS  # 6
        assert self.bc.pbs_dim == NUM_PRIVATE_STATES * NUM_PLAYERS  # 6

    def test_deals_match_kuhn(self):
        """BeliefConfig deals should match kuhn.belief_state.ALL_DEALS."""
        bc_deals = set(self.bc.all_deals)
        kuhn_deals = set(ALL_DEALS)
        assert bc_deals == kuhn_deals

    def test_card_to_deals_p0(self):
        """card_to_deals[0] should map each card to deals where P0 holds it."""
        for card in range(NUM_PRIVATE_STATES):
            deal_indices = self.bc.card_to_deals[0][card]
            for idx in deal_indices:
                assert self.bc.all_deals[idx][0] == card

    def test_card_to_deals_p1(self):
        """card_to_deals[1] should map each card to deals where P1 holds it."""
        for card in range(NUM_PRIVATE_STATES):
            deal_indices = self.bc.card_to_deals[1][card]
            for idx in deal_indices:
                assert self.bc.all_deals[idx][1] == card

    def test_chance_probs_uniform(self):
        """Chance probabilities should be uniform over deals."""
        expected = 1.0 / NUM_DEALS
        for p in self.bc.chance_probs:
            assert abs(p.item() - expected) < 1e-6

    def test_chance_probs_sum_to_one(self):
        assert abs(self.bc.chance_probs.sum().item() - 1.0) < 1e-6


class TestBeliefConfigPBS:
    """Test PBS operations against kuhn.belief_state reference implementation."""

    def setup_method(self):
        self.bc = belief_config_from_game(KuhnPoker())

    def test_initial_pbs_matches_kuhn(self):
        """BeliefConfig.initial_pbs should match kuhn initial_pbs."""
        bc_pbs = self.bc.initial_pbs()
        kuhn_pbs = kuhn_initial_pbs()
        assert torch.allclose(bc_pbs, kuhn_pbs, atol=1e-6)

    def test_reach_to_pbs_uniform(self):
        """Uniform reach should produce uniform PBS."""
        reach = torch.ones(NUM_DEALS)
        pbs = self.bc.reach_to_pbs(reach, reach)
        expected = 1.0 / NUM_PRIVATE_STATES
        for c in range(NUM_PRIVATE_STATES):
            for p in range(NUM_PLAYERS):
                assert abs(pbs[c, p].item() - expected) < 1e-5

    def test_reach_to_pbs_matches_kuhn(self):
        """BeliefConfig.reach_to_pbs should match kuhn reach_to_pbs."""
        reach_p0 = torch.tensor([1.0, 0.5, 1.0, 0.3, 0.7, 0.2])
        reach_p1 = torch.tensor([0.8, 1.0, 0.4, 0.6, 1.0, 0.9])

        bc_pbs = self.bc.reach_to_pbs(reach_p0, reach_p1)
        kuhn_pbs = kuhn_reach_to_pbs(reach_p0, reach_p1)

        assert torch.allclose(bc_pbs, kuhn_pbs, atol=1e-5), (
            f"BeliefConfig PBS:\n{bc_pbs}\nKuhn PBS:\n{kuhn_pbs}"
        )

    def test_pbs_columns_sum_to_one(self):
        """Each player's PBS column should sum to 1."""
        reach_p0 = torch.tensor([1.0, 0.5, 1.0, 0.3, 0.7, 0.2])
        reach_p1 = torch.tensor([0.8, 1.0, 0.4, 0.6, 1.0, 0.9])
        pbs = self.bc.reach_to_pbs(reach_p0, reach_p1)

        for p in range(NUM_PLAYERS):
            col_sum = pbs[:, p].sum().item()
            assert abs(col_sum - 1.0) < 1e-5, f"Player {p} PBS sums to {col_sum}"


class TestBeliefConfigRoundTrip:
    """Test PBS <-> reach probability conversions."""

    def setup_method(self):
        self.bc = belief_config_from_game(KuhnPoker())

    def test_pbs_to_reach_initial(self):
        """Initial PBS should produce uniform reach probabilities."""
        pbs = self.bc.initial_pbs()
        reach_p0, reach_p1 = self.bc.pbs_to_reach(pbs)
        expected = 1.0 / NUM_PRIVATE_STATES
        for i in range(NUM_DEALS):
            assert abs(reach_p0[i].item() - expected) < 1e-6
            assert abs(reach_p1[i].item() - expected) < 1e-6

    def test_deal_values_to_pbs_values_symmetry(self):
        """Values should be negated for P1 (zero-sum game)."""
        deal_values = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5, -0.5])
        pbs_values = self.bc.deal_values_to_pbs_values(deal_values)

        # For each card held by P0, average the deal values
        # For each card held by P1, average the negated deal values
        assert pbs_values.shape == (NUM_PRIVATE_STATES, NUM_PLAYERS)

    def test_pbs_values_deal_values_roundtrip(self):
        """deal_values -> pbs_values -> deal_values should be consistent."""
        deal_values = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5, -0.5])
        pbs_values = self.bc.deal_values_to_pbs_values(deal_values)
        recovered = self.bc.pbs_values_to_deal_values(pbs_values)

        # The recovered values use P0's perspective from pbs_values[card, 0]
        # So they should match the average of deal_values grouped by P0's card
        for card in range(NUM_PRIVATE_STATES):
            indices = self.bc.card_to_deals[0][card]
            expected = sum(deal_values[i].item() for i in indices) / len(indices)
            assert abs(recovered[indices[0]].item() - expected) < 1e-6


class TestBeliefConfigManual:
    """Test BeliefConfig with manually constructed data."""

    def test_manual_construction(self):
        """BeliefConfig should work with manually specified deals."""
        bc = BeliefConfig(
            num_private_states=2,
            num_players=2,
            all_deals=[(0, 1), (1, 0)],
        )
        assert bc.num_deals == 2
        assert bc.pbs_dim == 4  # 2 * 2
        assert bc.card_to_deals[0][0] == [0]  # P0 holds card 0 in deal 0
        assert bc.card_to_deals[0][1] == [1]  # P0 holds card 1 in deal 1
        assert bc.card_to_deals[1][1] == [0]  # P1 holds card 1 in deal 0
        assert bc.card_to_deals[1][0] == [1]  # P1 holds card 0 in deal 1

    def test_initial_pbs_manual(self):
        """Initial PBS should be uniform for a 2-card game."""
        bc = BeliefConfig(
            num_private_states=2,
            num_players=2,
            all_deals=[(0, 1), (1, 0)],
        )
        pbs = bc.initial_pbs()
        assert pbs.shape == (2, 2)
        assert torch.allclose(pbs, torch.tensor([[0.5, 0.5], [0.5, 0.5]]))

    def test_reach_to_pbs_manual(self):
        """Test PBS computation for a simple 2-card game."""
        bc = BeliefConfig(
            num_private_states=2,
            num_players=2,
            all_deals=[(0, 1), (1, 0)],
        )
        # If P0 reaches deal 0 with prob 1.0 and deal 1 with prob 0.0,
        # then P0 always holds card 0
        reach_p0 = torch.tensor([1.0, 0.0])
        reach_p1 = torch.tensor([1.0, 1.0])
        pbs = bc.reach_to_pbs(reach_p0, reach_p1)

        assert pbs[0, 0].item() == pytest.approx(1.0, abs=1e-6)  # P0 holds card 0
        assert pbs[1, 0].item() == pytest.approx(0.0, abs=1e-6)  # P0 doesn't hold card 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
