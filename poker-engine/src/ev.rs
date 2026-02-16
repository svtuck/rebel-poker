/// Expected value and counterfactual value computation for poker.
///
/// This is the innermost hot loop of CFR: given reach probabilities for both players
/// and a board, compute per-hand counterfactual values needed for regret updates.
///
/// The naive algorithm is O(N_HANDS^2) where N_HANDS = C(52,2) = 1326 for HUNL.
/// Card-overlap filtering reduces the constant but doesn't change complexity.

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
    /// For each hand index, bitset of opponent hand indices that share a card.
    /// Stored as Vec<Vec<u16>> for the naive implementation.
    pub hand_conflicts: Vec<Vec<u16>>,
    pub board: Vec<u8>,
    pub board_len: usize,
    /// Precomputed payoff matrix: payoff_matrix[h1 * NUM_HANDS + h2] = payoff(h1 vs h2)
    /// as i8 (+1, -1, 0). Invalid pairs (blocked or conflicting cards) are 0.
    /// None until `precompute_payoffs()` is called.
    payoff_matrix: Option<Vec<i8>>,
    /// Compact representation: indices of valid (non-blocked) hands, sorted by hand rank.
    /// Used for O(N log N) prefix-sum algorithm on river.
    valid_hands: Option<Vec<u16>>,
    /// Hand rank for each hand index (only meaningful for board_len == 5).
    /// Lower rank = better hand.
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
            // Hands that share c0 or c1
            let mut conflicts = Vec::new();
            for &hi in &card_to_hands[c0 as usize] {
                if hi as usize != h_idx && !board_blocked[hi as usize] {
                    conflicts.push(hi);
                }
            }
            for &hi in &card_to_hands[c1 as usize] {
                if hi as usize != h_idx && !board_blocked[hi as usize] {
                    // Avoid duplicates (a hand sharing both cards would appear twice)
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
            valid_hands: None,
            hand_ranks: None,
        }
    }

    /// Precompute the payoff matrix for all hand pairs.
    /// After calling this, `compute_cf_values` will use fast matrix-vector multiply
    /// instead of per-pair hand evaluation.
    /// The matrix is NUM_HANDS × NUM_HANDS i8 values ≈ 1.76 MB.
    pub fn precompute_payoffs(&mut self) {
        let mut matrix = vec![0i8; NUM_HANDS * NUM_HANDS];

        // First, precompute the hand rank for every valid hand.
        // This avoids redundant eval7/eval5 calls (each hand's rank only depends
        // on the hand + board, not the opponent).
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
                _ => 0, // Preflop: no ranks
            };
        }

        // Now build the payoff matrix using precomputed ranks.
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
                    0i8 // Preflop: can't evaluate
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
        // Zero-sum: ev_p2 = -ev_p1
        (ev_p1, -ev_p1)
    }

    /// Compute per-hand counterfactual values for both players.
    ///
    /// cf_value_p1[h1] = Σ_h2 reach_p2[h2] * payoff[h1][h2]
    /// cf_value_p2[h2] = Σ_h1 reach_p1[h1] * payoff_p2[h1][h2]
    ///                  = Σ_h1 reach_p1[h1] * (-payoff[h1][h2])
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
    /// where payoff(h, opp) is from h's perspective (+1 if h wins, -1 if h loses, 0 tie).
    fn compute_cf_values_one_player(&self, opp_reach: &[f64]) -> Vec<f64> {
        if let Some(ref matrix) = self.payoff_matrix {
            return self.compute_cf_values_one_player_fast(matrix, opp_reach);
        }

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

                // Check card conflict between the two hands
                if cards_conflict(hand, opp_hand) {
                    continue;
                }

                let opp_r = opp_reach[opp_idx];
                if opp_r == 0.0 {
                    continue;
                }

                // Evaluate showdown from hero's perspective
                let payoff = self.evaluate_payoff(h_idx, opp_idx);
                cf_val += opp_r * payoff;
            }

            cf_values[h_idx] = cf_val;
        }

        cf_values
    }

    /// Fast matrix-vector multiply using precomputed payoff matrix.
    /// cf_value[h] = Σ_opp payoff[h][opp] * reach_opp[opp]
    /// Since payoff values are {-1, 0, +1}, we split into add/subtract.
    #[inline(never)]
    fn compute_cf_values_one_player_fast(&self, matrix: &[i8], opp_reach: &[f64]) -> Vec<f64> {
        let mut cf_values = vec![0.0f64; NUM_HANDS];

        for h_idx in 0..NUM_HANDS {
            if self.board_blocked[h_idx] {
                continue;
            }

            let row = &matrix[h_idx * NUM_HANDS..(h_idx + 1) * NUM_HANDS];
            let mut win_sum = 0.0f64;
            let mut lose_sum = 0.0f64;

            for opp_idx in 0..NUM_HANDS {
                let p = row[opp_idx];
                if p > 0 {
                    win_sum += opp_reach[opp_idx];
                } else if p < 0 {
                    lose_sum += opp_reach[opp_idx];
                }
            }

            cf_values[h_idx] = win_sum - lose_sum;
        }

        cf_values
    }

    /// Evaluate the payoff for a (hero, villain) pair from hero's perspective.
    /// Returns +1.0 if hero wins, -1.0 if hero loses, 0.0 for tie.
    pub fn evaluate_payoff(&self, hero_idx: usize, villain_idx: usize) -> f64 {
        let hero = self.hands[hero_idx];
        let villain = self.hands[villain_idx];

        match self.board_len {
            0 => {
                // Preflop: no board, can't evaluate showdown.
                // For preflop EV we'd need to enumerate all possible boards,
                // which is not what this function does. Return 0 for now.
                // In practice, preflop EV computation should use a different method
                // (e.g., EHS or rollout). For the benchmark, we'll still include it
                // to measure the overhead of the loop.
                0.0
            }
            5 => {
                // River: full 7-card evaluation
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
                // Flop/Turn: would need rollout for proper EV.
                // For the naive implementation, we do best-of-5 with available cards.
                // Actually, for flop (3 cards) we have 5 cards total, use eval5.
                // For turn (4 cards) we have 6 cards, best 5 of 6.
                match self.board_len {
                    3 => {
                        // Flop: hero has 2 + board 3 = 5 cards exactly
                        let h5 = [hero.0, hero.1,
                            self.board[0], self.board[1], self.board[2]];
                        let v5 = [villain.0, villain.1,
                            self.board[0], self.board[1], self.board[2]];
                        let hr = eval::eval5(&h5);
                        let vr = eval::eval5(&v5);
                        if hr < vr { 1.0 } else if hr > vr { -1.0 } else { 0.0 }
                    }
                    4 => {
                        // Turn: hero has 2 + board 4 = 6 cards, best 5 of 6
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
    // C(6,5) = 6 combinations: skip each card in turn
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
/// This is the inverse of enumerate_hands().
pub fn hand_index(c0: u8, c1: u8) -> usize {
    debug_assert!(c0 < c1);
    // Hand index = sum_{i=0}^{c0-1} (51-i) + (c1 - c0 - 1)
    // = c0*51 - c0*(c0-1)/2 + c1 - c0 - 1
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
        // Board has Ah (card 50), so any hand containing 50 should be blocked
        let ah = card(RANK_A, SUIT_HEARTS);
        let ctx = EvContext::new(&[ah, card(RANK_K, SUIT_SPADES), card(RANK_Q, SUIT_SPADES),
                                    card(RANK_J, SUIT_SPADES), card(RANK_T, SUIT_SPADES)]);
        // Hands containing Ah should be blocked
        let hands = enumerate_hands();
        for (i, h) in hands.iter().enumerate() {
            if h.0 == ah || h.1 == ah {
                assert!(ctx.board_blocked[i], "Hand containing Ah should be blocked");
            }
        }
    }

    #[test]
    fn test_card_overlap_exclusion() {
        // If board has As, no hand containing As should contribute to EV
        let as_card = card(RANK_A, SUIT_SPADES);
        let board = [as_card, card(RANK_K, SUIT_HEARTS), card(RANK_Q, SUIT_HEARTS),
                     card(RANK_J, SUIT_HEARTS), card(RANK_T, SUIT_HEARTS)];
        let ctx = EvContext::new(&board);

        let mut reach = vec![0.0; NUM_HANDS];
        // Set uniform reach for non-blocked hands
        for i in 0..NUM_HANDS {
            if !ctx.board_blocked[i] {
                reach[i] = 1.0;
            }
        }

        let (cf_p1, cf_p2) = ctx.compute_cf_values(&reach, &reach);

        // Verify blocked hands have zero cf_values
        for (i, h) in ctx.hands.iter().enumerate() {
            if h.0 == as_card || h.1 == as_card {
                assert_eq!(cf_p1[i], 0.0, "Blocked hand should have zero cf_value");
                assert_eq!(cf_p2[i], 0.0, "Blocked hand should have zero cf_value");
            }
        }
    }

    #[test]
    fn test_consistency_ev_cf_values() {
        // ev_p1 should equal sum(reach_p1[h] * cf_p1[h]) for all valid h
        let board = [card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_HEARTS),
                     card(RANK_Q, SUIT_DIAMONDS), card(RANK_J, SUIT_CLUBS),
                     card(RANK_T, SUIT_SPADES)];
        let ctx = EvContext::new(&board);

        // Use non-uniform reach probabilities
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
            ev_p1,
            sum_p1
        );
        assert!(
            (ev_p2 - sum_p2).abs() < 1e-10,
            "EV p2 ({}) should equal sum of reach*cf ({})",
            ev_p2,
            sum_p2
        );
    }

    #[test]
    fn test_zero_sum() {
        // In a zero-sum game, ev_p1 + ev_p2 = 0
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
            ev_p1,
            ev_p2,
            ev_p1 + ev_p2
        );
    }

    #[test]
    fn test_river_known_hands() {
        // Board: Ac Kd 7h 4s 2d
        // Hero: As Ah (three aces) vs Villain: 3c 5s (nothing useful)
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

        // Three aces should beat K-high
        let payoff = ctx.evaluate_payoff(hero_idx, villain_idx);
        assert_eq!(payoff, 1.0, "Three aces should beat K-high");

        // From villain's perspective
        let payoff_v = ctx.evaluate_payoff(villain_idx, hero_idx);
        assert_eq!(payoff_v, -1.0, "K-high should lose to three aces");

        // Test a tie: hero and villain both have the same kicker situation
        // Board: Ac Kd 7h 4s 2d, both hold Qs Jh and Qh Js — same rank kickers, tie
        let h1 = hand_index_unordered(card(RANK_Q, SUIT_SPADES), card(RANK_J, SUIT_HEARTS));
        let h2 = hand_index_unordered(card(RANK_Q, SUIT_HEARTS), card(RANK_J, SUIT_SPADES));
        assert!(!ctx.board_blocked[h1]);
        assert!(!ctx.board_blocked[h2]);
        let payoff_tie = ctx.evaluate_payoff(h1, h2);
        assert_eq!(payoff_tie, 0.0, "Same-rank hands should tie");
    }

    #[test]
    fn test_precomputed_matches_naive() {
        // Verify that precomputed payoff matrix gives identical results to naive
        let board = [card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_HEARTS),
                     card(RANK_Q, SUIT_DIAMONDS), card(RANK_J, SUIT_CLUBS),
                     card(RANK_T, SUIT_SPADES)];

        let ctx_naive = EvContext::new(&board);
        let mut ctx_fast = EvContext::new(&board);
        ctx_fast.precompute_payoffs();

        // Non-uniform reach probabilities
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
                (cf_naive_p1[i] - cf_fast_p1[i]).abs() < 1e-10,
                "cf_p1 mismatch at {}: naive={}, fast={}",
                i, cf_naive_p1[i], cf_fast_p1[i]
            );
            assert!(
                (cf_naive_p2[i] - cf_fast_p2[i]).abs() < 1e-10,
                "cf_p2 mismatch at {}: naive={}, fast={}",
                i, cf_naive_p2[i], cf_fast_p2[i]
            );
        }

        // Also check EV
        let (ev_naive_p1, _) = ctx_naive.compute_ev(&reach_p1, &reach_p2);
        let (ev_fast_p1, _) = ctx_fast.compute_ev(&reach_p1, &reach_p2);
        assert!(
            (ev_naive_p1 - ev_fast_p1).abs() < 1e-10,
            "EV mismatch: naive={}, fast={}",
            ev_naive_p1, ev_fast_p1
        );
    }

    #[test]
    fn test_symmetric_uniform_river() {
        // With uniform reach and symmetric game, EV should be 0
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
        // With identical reach probabilities, the game is symmetric,
        // so EV should be exactly 0
        assert!(
            ev_p1.abs() < 1e-10,
            "Uniform reach EV should be 0, got {}",
            ev_p1
        );
    }
}
