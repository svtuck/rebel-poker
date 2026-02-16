/// Expected value and counterfactual value computation for poker.
///
/// This is the innermost hot loop of CFR: given reach probabilities for both players
/// and a board, compute per-hand counterfactual values needed for regret updates.
///
/// The naive algorithm is O(N_HANDS^2) where N_HANDS = C(52,2) = 1326 for HUNL.
/// Card-overlap filtering reduces the constant but doesn't change complexity.
///
/// Optimized path: call `precompute_payoffs()` once per board to enable O(N) per-hand
/// counterfactual value computation using sorted hand ranks and prefix sums.

use crate::eval;

/// Total number of 2-card hands in a 52-card deck: C(52,2) = 1326
pub const NUM_HANDS: usize = 1326;

/// A 2-card hand, stored as (lo_card, hi_card) with lo < hi.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hand(pub u8, pub u8);

/// Precomputed hand enumeration and card-overlap data for a specific board.
pub struct EvContext {
    /// All 1326 possible 2-card hands, indexed 0..1326.
    pub hands: Vec<Hand>,
    /// For each card (0..52), which hand indices contain that card.
    pub card_to_hands: Vec<Vec<u16>>,
    /// Bitset: which hands are blocked by the board (contain a board card).
    pub board_blocked: Vec<bool>,
    /// For each hand index, list of opponent hand indices that share a card.
    pub hand_conflicts: Vec<Vec<u16>>,
    pub board: Vec<u8>,
    pub board_len: usize,
    /// Precomputed payoff matrix: payoff_matrix[h1 * NUM_HANDS + h2] = payoff(h1 vs h2)
    /// as i8 (+1, -1, 0). Invalid pairs (blocked or conflicting cards) are 0.
    /// None until `precompute_payoffs()` is called.
    payoff_matrix: Option<Vec<i8>>,
    /// Hand rank for each hand index (lower = better hand).
    /// Populated by `precompute_payoffs()` when board_len > 0.
    hand_ranks: Option<Vec<u16>>,
}

impl EvContext {
    /// Build the context for a given board.
    /// board: slice of community cards (0 for preflop, 3 flop, 4 turn, 5 river).
    pub fn new(board: &[u8]) -> Self {
        let hands = enumerate_hands();
        let card_to_hands = build_card_to_hands(&hands);

        let mut board_blocked = vec![false; NUM_HANDS];
        for &bc in board {
            for &hi in &card_to_hands[bc as usize] {
                board_blocked[hi as usize] = true;
            }
        }

        // Build conflict lists: for each hand, which other hands share a card.
        let mut hand_conflicts = vec![Vec::new(); NUM_HANDS];
        for h_idx in 0..NUM_HANDS {
            if board_blocked[h_idx] {
                continue;
            }
            let Hand(c0, c1) = hands[h_idx];
            let mut conflicts = Vec::new();
            for &hi in &card_to_hands[c0 as usize] {
                if hi as usize != h_idx && !board_blocked[hi as usize] {
                    conflicts.push(hi);
                }
            }
            for &hi in &card_to_hands[c1 as usize] {
                if hi as usize != h_idx && !board_blocked[hi as usize] {
                    if !conflicts.contains(&hi) {
                        conflicts.push(hi);
                    }
                }
            }
            hand_conflicts[h_idx] = conflicts;
        }

        EvContext {
            hands,
            card_to_hands,
            board_blocked,
            hand_conflicts,
            board: board.to_vec(),
            board_len: board.len(),
            payoff_matrix: None,
            hand_ranks: None,
        }
    }

    /// Precompute the payoff matrix for all hand pairs.
    /// After calling this, `compute_cf_values` will use the O(N) sorted prefix-sum
    /// algorithm instead of per-pair hand evaluation.
    pub fn precompute_payoffs(&mut self) {
        let mut matrix = vec![0i8; NUM_HANDS * NUM_HANDS];

        // Precompute the hand rank for every valid hand.
        // Each hand's rank only depends on the hand + board, not the opponent.
        let mut ranks = vec![0u16; NUM_HANDS];
        for h_idx in 0..NUM_HANDS {
            if self.board_blocked[h_idx] {
                continue;
            }
            let hand = self.hands[h_idx];
            ranks[h_idx] = match self.board_len {
                5 => {
                    let h7 = [hand.0, hand.1,
                        self.board[0], self.board[1], self.board[2],
                        self.board[3], self.board[4]];
                    eval::eval7(&h7)
                }
                3 => {
                    let h5 = [hand.0, hand.1,
                        self.board[0], self.board[1], self.board[2]];
                    eval::eval5(&h5)
                }
                4 => {
                    let h_cards = [hand.0, hand.1,
                        self.board[0], self.board[1], self.board[2], self.board[3]];
                    best_of_6(&h_cards)
                }
                _ => 0,
            };
        }

        // Build the payoff matrix using precomputed ranks.
        for h_idx in 0..NUM_HANDS {
            if self.board_blocked[h_idx] {
                continue;
            }
            let hand = self.hands[h_idx];
            let row_offset = h_idx * NUM_HANDS;
            let hr = ranks[h_idx];

            for opp_idx in 0..NUM_HANDS {
                if self.board_blocked[opp_idx] {
                    continue;
                }
                let opp_hand = self.hands[opp_idx];
                if cards_conflict(hand, opp_hand) {
                    continue;
                }

                let vr = ranks[opp_idx];
                let payoff = if self.board_len == 0 {
                    0i8
                } else if hr < vr {
                    1
                } else if hr > vr {
                    -1
                } else {
                    0
                };
                matrix[row_offset + opp_idx] = payoff;
            }
        }

        self.hand_ranks = Some(ranks);
        self.payoff_matrix = Some(matrix);
    }

    /// Compute scalar EV for each player.
    ///
    /// reach_p1, reach_p2: reach probability for each of the 1326 hands.
    /// Returns (ev_p1, ev_p2) where ev_p2 = -ev_p1 in a zero-sum game.
    pub fn compute_ev(&self, reach_p1: &[f64], reach_p2: &[f64]) -> (f64, f64) {
        assert_eq!(reach_p1.len(), NUM_HANDS);
        assert_eq!(reach_p2.len(), NUM_HANDS);

        let cf_p1 = self.compute_cf_values_one_player(reach_p2);
        let mut ev_p1 = 0.0f64;
        for i in 0..NUM_HANDS {
            if !self.board_blocked[i] {
                ev_p1 += reach_p1[i] * cf_p1[i];
            }
        }
        (ev_p1, -ev_p1)
    }

    /// Compute per-hand counterfactual values for both players.
    ///
    /// cf_value_p1[h1] = Σ_h2 reach_p2[h2] * payoff[h1][h2]
    /// cf_value_p2[h2] = Σ_h1 reach_p1[h1] * (-payoff[h1][h2])
    ///
    /// Board-blocked hands get cf_value = 0.
    pub fn compute_cf_values(
        &self,
        reach_p1: &[f64],
        reach_p2: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        assert_eq!(reach_p1.len(), NUM_HANDS);
        assert_eq!(reach_p2.len(), NUM_HANDS);

        let cf_p1 = self.compute_cf_values_one_player(reach_p2);
        let cf_p2 = self.compute_cf_values_one_player(reach_p1);
        (cf_p1, cf_p2)
    }

    /// Compute counterfactual values for one player.
    ///
    /// cf_value[h] = Σ_opp reach_opp[opp] * payoff(h, opp)
    fn compute_cf_values_one_player(&self, opp_reach: &[f64]) -> Vec<f64> {
        // Prefer sorted prefix-sum O(N) approach when hand ranks are available.
        if let Some(ref ranks) = self.hand_ranks {
            if self.board_len > 0 {
                return self.compute_cf_values_one_player_sorted(ranks, opp_reach);
            }
        }
        // Fall back to matrix multiply if payoff matrix is available.
        if let Some(ref matrix) = self.payoff_matrix {
            return self.compute_cf_values_one_player_matrix(matrix, opp_reach);
        }

        // Naive O(N²) with per-pair hand evaluation.
        let mut cf_values = vec![0.0f64; NUM_HANDS];

        for h_idx in 0..NUM_HANDS {
            if self.board_blocked[h_idx] {
                continue;
            }

            let mut cf_val = 0.0f64;
            let hand = self.hands[h_idx];

            for opp_idx in 0..NUM_HANDS {
                if self.board_blocked[opp_idx] {
                    continue;
                }

                let opp_hand = self.hands[opp_idx];

                if cards_conflict(hand, opp_hand) {
                    continue;
                }

                let opp_r = opp_reach[opp_idx];
                if opp_r == 0.0 {
                    continue;
                }

                let payoff = self.evaluate_payoff(h_idx, opp_idx);
                cf_val += opp_r * payoff;
            }

            cf_values[h_idx] = cf_val;
        }

        cf_values
    }

    /// Fast matrix-vector multiply using precomputed payoff matrix.
    fn compute_cf_values_one_player_matrix(&self, matrix: &[i8], opp_reach: &[f64]) -> Vec<f64> {
        let mut cf_values = vec![0.0f64; NUM_HANDS];

        for h_idx in 0..NUM_HANDS {
            if self.board_blocked[h_idx] {
                continue;
            }

            let row = &matrix[h_idx * NUM_HANDS..(h_idx + 1) * NUM_HANDS];
            let mut cf_val = 0.0f64;

            for opp_idx in 0..NUM_HANDS {
                cf_val += row[opp_idx] as f64 * opp_reach[opp_idx];
            }

            cf_values[h_idx] = cf_val;
        }

        cf_values
    }

    /// O(N) per hand using sorted ranks and prefix sums.
    ///
    /// For hand h with rank r:
    ///   cf_value[h] = (total reach of worse hands) - (total reach of better hands)
    ///                 adjusted for card-conflicting hands (~93 per hand)
    fn compute_cf_values_one_player_sorted(&self, ranks: &[u16], opp_reach: &[f64]) -> Vec<f64> {
        let mut cf_values = vec![0.0f64; NUM_HANDS];

        // Compute total reach of all valid hands.
        let mut total_reach = 0.0f64;
        for h_idx in 0..NUM_HANDS {
            if !self.board_blocked[h_idx] {
                total_reach += opp_reach[h_idx];
            }
        }

        // For each rank value, compute total reach of hands with that rank.
        let max_rank = 7463usize;
        let mut rank_reach = vec![0.0f64; max_rank + 1];
        for h_idx in 0..NUM_HANDS {
            if !self.board_blocked[h_idx] {
                rank_reach[ranks[h_idx] as usize] += opp_reach[h_idx];
            }
        }

        // Prefix sum: prefix_reach[r] = total reach of hands with rank < r (strictly better).
        let mut prefix_reach = vec![0.0f64; max_rank + 2];
        for r in 0..=max_rank {
            prefix_reach[r + 1] = prefix_reach[r] + rank_reach[r];
        }

        // For hand h with rank r:
        // - better_reach = prefix_reach[r] = reach of hands with rank < r (they beat h)
        // - worse_reach = total_reach - prefix_reach[r+1] = reach of hands with rank > r
        // - cf_value[h] = worse_reach - better_reach - conflict_adjustment
        for h_idx in 0..NUM_HANDS {
            if self.board_blocked[h_idx] {
                continue;
            }

            let r = ranks[h_idx] as usize;
            let better_reach = prefix_reach[r];
            let worse_reach = total_reach - prefix_reach[r + 1];

            let mut cf_val = worse_reach - better_reach;

            // Adjust for card conflicts: hands that share a card with h
            // were included in the prefix sums but shouldn't contribute.
            for &conflict_idx in &self.hand_conflicts[h_idx] {
                let cr = ranks[conflict_idx as usize] as usize;
                let c_reach = opp_reach[conflict_idx as usize];
                if cr < r {
                    // Was counted in better_reach (payoff -1), undo it
                    cf_val += c_reach;
                } else if cr > r {
                    // Was counted in worse_reach (payoff +1), undo it
                    cf_val -= c_reach;
                }
            }

            cf_values[h_idx] = cf_val;
        }

        cf_values
    }

    /// Evaluate the payoff for a (hero, villain) pair from hero's perspective.
    /// Returns +1.0 if hero wins, -1.0 if hero loses, 0.0 for tie.
    pub fn evaluate_payoff(&self, hero_idx: usize, villain_idx: usize) -> f64 {
        let hero = self.hands[hero_idx];
        let villain = self.hands[villain_idx];

        match self.board_len {
            0 => 0.0,
            5 => {
                let h7 = [hero.0, hero.1,
                    self.board[0], self.board[1], self.board[2],
                    self.board[3], self.board[4]];
                let v7 = [villain.0, villain.1,
                    self.board[0], self.board[1], self.board[2],
                    self.board[3], self.board[4]];
                let hr = eval::eval7(&h7);
                let vr = eval::eval7(&v7);
                if hr < vr { 1.0 } else if hr > vr { -1.0 } else { 0.0 }
            }
            3 | 4 => {
                match self.board_len {
                    3 => {
                        let h5 = [hero.0, hero.1,
                            self.board[0], self.board[1], self.board[2]];
                        let v5 = [villain.0, villain.1,
                            self.board[0], self.board[1], self.board[2]];
                        let hr = eval::eval5(&h5);
                        let vr = eval::eval5(&v5);
                        if hr < vr { 1.0 } else if hr > vr { -1.0 } else { 0.0 }
                    }
                    4 => {
                        let h_cards = [hero.0, hero.1,
                            self.board[0], self.board[1], self.board[2], self.board[3]];
                        let v_cards = [villain.0, villain.1,
                            self.board[0], self.board[1], self.board[2], self.board[3]];
                        let hr = best_of_6(&h_cards);
                        let vr = best_of_6(&v_cards);
                        if hr < vr { 1.0 } else if hr > vr { -1.0 } else { 0.0 }
                    }
                    _ => unreachable!(),
                }
            }
            _ => 0.0,
        }
    }
}

/// Enumerate all C(52,2) = 1326 two-card hands.
/// Returns them sorted with lo < hi.
pub fn enumerate_hands() -> Vec<Hand> {
    let mut hands = Vec::with_capacity(NUM_HANDS);
    for c0 in 0..52u8 {
        for c1 in (c0 + 1)..52u8 {
            hands.push(Hand(c0, c1));
        }
    }
    debug_assert_eq!(hands.len(), NUM_HANDS);
    hands
}

/// Build a mapping from card (0..52) to hand indices that contain it.
fn build_card_to_hands(hands: &[Hand]) -> Vec<Vec<u16>> {
    let mut card_to_hands = vec![Vec::new(); 52];
    for (i, &Hand(c0, c1)) in hands.iter().enumerate() {
        card_to_hands[c0 as usize].push(i as u16);
        card_to_hands[c1 as usize].push(i as u16);
    }
    card_to_hands
}

/// Check if two hands share any card.
#[inline]
fn cards_conflict(a: Hand, b: Hand) -> bool {
    a.0 == b.0 || a.0 == b.1 || a.1 == b.0 || a.1 == b.1
}

/// Evaluate best 5 of 6 cards.
fn best_of_6(cards: &[u8; 6]) -> eval::HandRank {
    let mut best = eval::HandRank::MAX;
    for skip in 0..6 {
        let mut hand = [0u8; 5];
        let mut j = 0;
        for i in 0..6 {
            if i != skip {
                hand[j] = cards[i];
                j += 1;
            }
        }
        let r = eval::eval5(&hand);
        if r < best {
            best = r;
        }
    }
    best
}

/// Get the hand index for a given (c0, c1) pair where c0 < c1.
pub fn hand_index(c0: u8, c1: u8) -> usize {
    debug_assert!(c0 < c1);
    let c0 = c0 as usize;
    let c1 = c1 as usize;
    c0 * 51 - c0 * (c0.wrapping_sub(1)) / 2 + c1 - c0 - 1
}

/// Get the hand index for an unordered card pair.
pub fn hand_index_unordered(a: u8, b: u8) -> usize {
    if a < b {
        hand_index(a, b)
    } else {
        hand_index(b, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::*;

    #[test]
    fn test_enumerate_hands_count() {
        let hands = enumerate_hands();
        assert_eq!(hands.len(), 1326);
    }

    #[test]
    fn test_enumerate_hands_sorted() {
        let hands = enumerate_hands();
        for h in &hands {
            assert!(h.0 < h.1);
        }
    }

    #[test]
    fn test_hand_index_roundtrip() {
        let hands = enumerate_hands();
        for (i, &Hand(c0, c1)) in hands.iter().enumerate() {
            assert_eq!(hand_index(c0, c1), i, "Failed for hand ({}, {})", c0, c1);
        }
    }

    #[test]
    fn test_cards_conflict() {
        assert!(cards_conflict(Hand(0, 1), Hand(1, 2)));
        assert!(cards_conflict(Hand(0, 1), Hand(0, 2)));
        assert!(!cards_conflict(Hand(0, 1), Hand(2, 3)));
    }

    #[test]
    fn test_board_blocking() {
        let ah = card(RANK_A, SUIT_HEARTS);
        let ctx = EvContext::new(&[ah, card(RANK_K, SUIT_SPADES), card(RANK_Q, SUIT_SPADES),
                                    card(RANK_J, SUIT_SPADES), card(RANK_T, SUIT_SPADES)]);
        let hands = enumerate_hands();
        for (i, h) in hands.iter().enumerate() {
            if h.0 == ah || h.1 == ah {
                assert!(ctx.board_blocked[i], "Hand containing Ah should be blocked");
            }
        }
    }

    #[test]
    fn test_card_overlap_exclusion() {
        let as_card = card(RANK_A, SUIT_SPADES);
        let board = [as_card, card(RANK_K, SUIT_HEARTS), card(RANK_Q, SUIT_HEARTS),
                     card(RANK_J, SUIT_HEARTS), card(RANK_T, SUIT_HEARTS)];
        let ctx = EvContext::new(&board);

        let mut reach = vec![0.0; NUM_HANDS];
        for i in 0..NUM_HANDS {
            if !ctx.board_blocked[i] {
                reach[i] = 1.0;
            }
        }

        let (cf_p1, cf_p2) = ctx.compute_cf_values(&reach, &reach);

        for (i, h) in ctx.hands.iter().enumerate() {
            if h.0 == as_card || h.1 == as_card {
                assert_eq!(cf_p1[i], 0.0, "Blocked hand should have zero cf_value");
                assert_eq!(cf_p2[i], 0.0, "Blocked hand should have zero cf_value");
            }
        }
    }

    #[test]
    fn test_consistency_ev_cf_values() {
        let board = [card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_HEARTS),
                     card(RANK_Q, SUIT_DIAMONDS), card(RANK_J, SUIT_CLUBS),
                     card(RANK_T, SUIT_SPADES)];
        let ctx = EvContext::new(&board);

        let mut reach_p1 = vec![0.0; NUM_HANDS];
        let mut reach_p2 = vec![0.0; NUM_HANDS];
        for i in 0..NUM_HANDS {
            if !ctx.board_blocked[i] {
                reach_p1[i] = (i as f64 + 1.0) / NUM_HANDS as f64;
                reach_p2[i] = ((NUM_HANDS - i) as f64) / NUM_HANDS as f64;
            }
        }

        let (ev_p1, ev_p2) = ctx.compute_ev(&reach_p1, &reach_p2);
        let (cf_p1, cf_p2) = ctx.compute_cf_values(&reach_p1, &reach_p2);

        let sum_p1: f64 = (0..NUM_HANDS)
            .filter(|&i| !ctx.board_blocked[i])
            .map(|i| reach_p1[i] * cf_p1[i])
            .sum();

        let sum_p2: f64 = (0..NUM_HANDS)
            .filter(|&i| !ctx.board_blocked[i])
            .map(|i| reach_p2[i] * cf_p2[i])
            .sum();

        assert!(
            (ev_p1 - sum_p1).abs() < 1e-10,
            "EV p1 ({}) should equal sum of reach*cf ({})",
            ev_p1, sum_p1
        );
        assert!(
            (ev_p2 - sum_p2).abs() < 1e-10,
            "EV p2 ({}) should equal sum of reach*cf ({})",
            ev_p2, sum_p2
        );
    }

    #[test]
    fn test_zero_sum() {
        let board = [card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_HEARTS),
                     card(RANK_Q, SUIT_DIAMONDS), card(RANK_J, SUIT_CLUBS),
                     card(RANK_T, SUIT_SPADES)];
        let ctx = EvContext::new(&board);

        let mut reach_p1 = vec![0.0; NUM_HANDS];
        let mut reach_p2 = vec![0.0; NUM_HANDS];
        for i in 0..NUM_HANDS {
            if !ctx.board_blocked[i] {
                reach_p1[i] = 1.0;
                reach_p2[i] = 1.0;
            }
        }

        let (ev_p1, ev_p2) = ctx.compute_ev(&reach_p1, &reach_p2);
        assert!(
            (ev_p1 + ev_p2).abs() < 1e-10,
            "Zero sum violated: {} + {} = {}",
            ev_p1, ev_p2, ev_p1 + ev_p2
        );
    }

    #[test]
    fn test_river_known_hands() {
        let board = [
            card(RANK_A, SUIT_CLUBS), card(RANK_K, SUIT_DIAMONDS),
            card(RANK_7, SUIT_HEARTS), card(RANK_4, SUIT_SPADES),
            card(RANK_2, SUIT_DIAMONDS),
        ];
        let ctx = EvContext::new(&board);

        let hero_idx = hand_index_unordered(card(RANK_A, SUIT_SPADES), card(RANK_A, SUIT_HEARTS));
        let villain_idx = hand_index_unordered(card(RANK_3, SUIT_CLUBS), card(RANK_5, SUIT_SPADES));

        assert!(!ctx.board_blocked[hero_idx]);
        assert!(!ctx.board_blocked[villain_idx]);

        let payoff = ctx.evaluate_payoff(hero_idx, villain_idx);
        assert_eq!(payoff, 1.0, "Three aces should beat K-high");

        let payoff_v = ctx.evaluate_payoff(villain_idx, hero_idx);
        assert_eq!(payoff_v, -1.0, "K-high should lose to three aces");

        let h1 = hand_index_unordered(card(RANK_Q, SUIT_SPADES), card(RANK_J, SUIT_HEARTS));
        let h2 = hand_index_unordered(card(RANK_Q, SUIT_HEARTS), card(RANK_J, SUIT_SPADES));
        assert!(!ctx.board_blocked[h1]);
        assert!(!ctx.board_blocked[h2]);
        let payoff_tie = ctx.evaluate_payoff(h1, h2);
        assert_eq!(payoff_tie, 0.0, "Same-rank hands should tie");
    }

    #[test]
    fn test_precomputed_matches_naive() {
        // Verify that precomputed/sorted approach gives identical results to naive
        let board = [card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_HEARTS),
                     card(RANK_Q, SUIT_DIAMONDS), card(RANK_J, SUIT_CLUBS),
                     card(RANK_T, SUIT_SPADES)];

        let ctx_naive = EvContext::new(&board);
        let mut ctx_fast = EvContext::new(&board);
        ctx_fast.precompute_payoffs();

        let mut reach_p1 = vec![0.0; NUM_HANDS];
        let mut reach_p2 = vec![0.0; NUM_HANDS];
        for i in 0..NUM_HANDS {
            if !ctx_naive.board_blocked[i] {
                reach_p1[i] = (i as f64 + 1.0) / NUM_HANDS as f64;
                reach_p2[i] = ((NUM_HANDS - i) as f64) / NUM_HANDS as f64;
            }
        }

        let (cf_naive_p1, cf_naive_p2) = ctx_naive.compute_cf_values(&reach_p1, &reach_p2);
        let (cf_fast_p1, cf_fast_p2) = ctx_fast.compute_cf_values(&reach_p1, &reach_p2);

        for i in 0..NUM_HANDS {
            assert!(
                (cf_naive_p1[i] - cf_fast_p1[i]).abs() < 1e-8,
                "cf_p1 mismatch at {}: naive={}, fast={}, diff={}",
                i, cf_naive_p1[i], cf_fast_p1[i], (cf_naive_p1[i] - cf_fast_p1[i]).abs()
            );
            assert!(
                (cf_naive_p2[i] - cf_fast_p2[i]).abs() < 1e-8,
                "cf_p2 mismatch at {}: naive={}, fast={}, diff={}",
                i, cf_naive_p2[i], cf_fast_p2[i], (cf_naive_p2[i] - cf_fast_p2[i]).abs()
            );
        }

        // EV check (prefix-sum reordering slightly changes fp accumulation)
        let (ev_naive_p1, _) = ctx_naive.compute_ev(&reach_p1, &reach_p2);
        let (ev_fast_p1, _) = ctx_fast.compute_ev(&reach_p1, &reach_p2);
        assert!(
            (ev_naive_p1 - ev_fast_p1).abs() < 1e-6,
            "EV mismatch: naive={}, fast={}, diff={}",
            ev_naive_p1, ev_fast_p1, (ev_naive_p1 - ev_fast_p1).abs()
        );
    }

    #[test]
    fn test_precomputed_zero_sum() {
        // Zero-sum property should hold with precomputed path
        let board = [card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_HEARTS),
                     card(RANK_Q, SUIT_DIAMONDS), card(RANK_J, SUIT_CLUBS),
                     card(RANK_T, SUIT_SPADES)];
        let mut ctx = EvContext::new(&board);
        ctx.precompute_payoffs();

        let mut reach_p1 = vec![0.0; NUM_HANDS];
        let mut reach_p2 = vec![0.0; NUM_HANDS];
        for i in 0..NUM_HANDS {
            if !ctx.board_blocked[i] {
                reach_p1[i] = 1.0;
                reach_p2[i] = 1.0;
            }
        }

        let (ev_p1, ev_p2) = ctx.compute_ev(&reach_p1, &reach_p2);
        assert!(
            (ev_p1 + ev_p2).abs() < 1e-8,
            "Zero sum violated: {} + {} = {}",
            ev_p1, ev_p2, ev_p1 + ev_p2
        );
    }

    #[test]
    fn test_precomputed_different_board() {
        // Test with a different board to verify generality
        let board = [
            card(RANK_2, SUIT_CLUBS), card(RANK_5, SUIT_DIAMONDS),
            card(RANK_8, SUIT_HEARTS), card(RANK_J, SUIT_SPADES),
            card(RANK_A, SUIT_CLUBS),
        ];

        let ctx_naive = EvContext::new(&board);
        let mut ctx_fast = EvContext::new(&board);
        ctx_fast.precompute_payoffs();

        let mut reach_p1 = vec![0.0; NUM_HANDS];
        let mut reach_p2 = vec![0.0; NUM_HANDS];
        for i in 0..NUM_HANDS {
            if !ctx_naive.board_blocked[i] {
                reach_p1[i] = ((i * 7 + 3) % 100) as f64 / 100.0;
                reach_p2[i] = ((i * 13 + 11) % 100) as f64 / 100.0;
            }
        }

        let (cf_naive_p1, cf_naive_p2) = ctx_naive.compute_cf_values(&reach_p1, &reach_p2);
        let (cf_fast_p1, cf_fast_p2) = ctx_fast.compute_cf_values(&reach_p1, &reach_p2);

        for i in 0..NUM_HANDS {
            assert!(
                (cf_naive_p1[i] - cf_fast_p1[i]).abs() < 1e-8,
                "cf_p1 mismatch at {}: naive={}, fast={}",
                i, cf_naive_p1[i], cf_fast_p1[i]
            );
            assert!(
                (cf_naive_p2[i] - cf_fast_p2[i]).abs() < 1e-8,
                "cf_p2 mismatch at {}: naive={}, fast={}",
                i, cf_naive_p2[i], cf_fast_p2[i]
            );
        }
    }

    #[test]
    fn test_symmetric_uniform_river() {
        let board = [
            card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_HEARTS),
            card(RANK_Q, SUIT_DIAMONDS), card(RANK_J, SUIT_CLUBS),
            card(RANK_T, SUIT_SPADES),
        ];
        let ctx = EvContext::new(&board);

        let mut reach = vec![0.0; NUM_HANDS];
        for i in 0..NUM_HANDS {
            if !ctx.board_blocked[i] {
                reach[i] = 1.0;
            }
        }

        let (ev_p1, _ev_p2) = ctx.compute_ev(&reach, &reach);
        assert!(
            ev_p1.abs() < 1e-10,
            "Uniform reach EV should be 0, got {}",
            ev_p1
        );
    }
}
