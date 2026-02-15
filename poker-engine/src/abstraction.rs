/// Card abstraction for HUNL Texas Hold'em.
///
/// Reduces the ~10^161 information states to ~10^12-10^14 abstract states
/// that CFR can solve. Two key components:
///
/// 1. **Card abstraction** — Group strategically similar hands into buckets
///    - Preflop: 169 canonical starting hands
///    - Flop/Turn: Potential-aware histograms of next-round bucket transitions, k-means + EMD
///    - River: Direct EHS against opponent range, k-means
///
/// 2. **Action abstraction** — See state.rs ActionAbstraction

use crate::card;
use crate::eval;

// --- Preflop canonical hands ---

/// Total canonical preflop hands: 13 pairs + 78 suited + 78 offsuit = 169
pub const NUM_PREFLOP_BUCKETS: usize = 169;

/// Encode a 2-card hand into a canonical preflop bucket (0..168).
/// Canonical form: higher rank first. Pairs: 0..12, Suited: 13..90, Offsuit: 91..168.
pub fn preflop_bucket(card0: u8, card1: u8) -> u16 {
    let r0 = card::rank(card0);
    let r1 = card::rank(card1);
    let s0 = card::suit(card0);
    let s1 = card::suit(card1);

    let (high, low) = if r0 >= r1 { (r0, r1) } else { (r1, r0) };

    if high == low {
        // Pocket pair: index 0..12 (2s through As)
        high as u16
    } else if s0 == s1 {
        // Suited: 13 + triangular index
        13 + suited_index(high, low)
    } else {
        // Offsuit: 13 + 78 + triangular index
        91 + offsuit_index(high, low)
    }
}

/// Triangular index for suited hands: C(high, 2) + low for high > low
fn suited_index(high: u8, low: u8) -> u16 {
    // Maps (high, low) where high > low to 0..77
    // Index = (high * (high - 1)) / 2 + low
    ((high as u16) * (high as u16 - 1)) / 2 + low as u16
}

/// Same indexing for offsuit hands
fn offsuit_index(high: u8, low: u8) -> u16 {
    ((high as u16) * (high as u16 - 1)) / 2 + low as u16
}

// --- Expected Hand Strength (EHS) ---

/// Compute EHS for a given hand against all possible opponent hands.
/// EHS = P(win) + 0.5 * P(tie), averaged over all possible opponent holdings
/// and (optionally) remaining board cards.
///
/// `hero`: 2 hole cards
/// `board`: current board cards (0, 3, 4, or 5 cards)
/// `dead`: cards that can't be dealt (hero's cards + board)
///
/// For river (5 board cards): exact enumeration of opponent hands.
/// For earlier streets: Monte Carlo sampling of remaining board + opponent.
pub fn compute_ehs(hero: [u8; 2], board: &[u8]) -> f64 {
    let mut dead = [false; 52];
    dead[hero[0] as usize] = true;
    dead[hero[1] as usize] = true;
    for &b in board {
        dead[b as usize] = true;
    }

    match board.len() {
        5 => compute_ehs_river(hero, board, &dead),
        _ => compute_ehs_monte_carlo(hero, board, &dead, 500),
    }
}

/// Exact EHS on the river: enumerate all C(remaining, 2) opponent hands.
fn compute_ehs_river(hero: [u8; 2], board: &[u8], dead: &[bool; 52]) -> f64 {
    let hero_hand = [
        hero[0], hero[1], board[0], board[1], board[2], board[3], board[4],
    ];
    let hero_rank = eval::eval7(&hero_hand);

    let mut wins = 0u32;
    let mut ties = 0u32;
    let mut total = 0u32;

    for opp0 in 0..51u8 {
        if dead[opp0 as usize] {
            continue;
        }
        for opp1 in (opp0 + 1)..52u8 {
            if dead[opp1 as usize] {
                continue;
            }
            let opp_hand = [
                opp0, opp1, board[0], board[1], board[2], board[3], board[4],
            ];
            let opp_rank = eval::eval7(&opp_hand);
            total += 1;
            if hero_rank < opp_rank {
                wins += 1;
            } else if hero_rank == opp_rank {
                ties += 1;
            }
        }
    }

    if total == 0 {
        return 0.5;
    }
    (wins as f64 + 0.5 * ties as f64) / total as f64
}

/// Monte Carlo EHS: sample random boards + opponent hands.
fn compute_ehs_monte_carlo(
    hero: [u8; 2],
    board: &[u8],
    dead: &[bool; 52],
    num_samples: usize,
) -> f64 {
    // Build deck of remaining cards
    let mut remaining: Vec<u8> = Vec::with_capacity(52);
    for c in 0..52u8 {
        if !dead[c as usize] {
            remaining.push(c);
        }
    }

    let board_needed = 5 - board.len();
    let cards_needed = board_needed + 2; // remaining board + 2 opponent cards

    if remaining.len() < cards_needed {
        return 0.5;
    }

    // Deterministic sampling: enumerate combinations rather than random
    // For efficiency, we sample systematically using stride-based indexing
    let mut wins = 0u64;
    let mut ties = 0u64;
    let mut total = 0u64;

    // For river (board_needed=0): enumerate all opponent pairs
    if board_needed == 0 {
        return compute_ehs_river(hero, board, dead);
    }

    // For flop/turn: enumerate opponent hands for each sampled board completion
    // We'll do a partial enumeration: iterate over board completions and opponent pairs
    let n = remaining.len();

    if board_needed == 2 {
        // Turn → River: need 2 board cards
        for bi in 0..n {
            let bc0 = remaining[bi];
            for bj in (bi + 1)..n {
                let bc1 = remaining[bj];
                let full_board = [board[0], board[1], board[2], bc0, bc1];

                // Enumerate opponent hands
                for oi in 0..n {
                    if oi == bi || oi == bj {
                        continue;
                    }
                    let o0 = remaining[oi];
                    for oj in (oi + 1)..n {
                        if oj == bi || oj == bj {
                            continue;
                        }
                        let o1 = remaining[oj];
                        let hero_hand =
                            [hero[0], hero[1], full_board[0], full_board[1], full_board[2], full_board[3], full_board[4]];
                        let opp_hand =
                            [o0, o1, full_board[0], full_board[1], full_board[2], full_board[3], full_board[4]];
                        let hr = eval::eval7(&hero_hand);
                        let or = eval::eval7(&opp_hand);
                        total += 1;
                        if hr < or {
                            wins += 1;
                        } else if hr == or {
                            ties += 1;
                        }
                    }
                }
            }
        }
    } else if board_needed == 1 {
        // Turn: need 1 board card
        for bi in 0..n {
            let bc = remaining[bi];
            let full_board = [board[0], board[1], board[2], board[3], bc];

            for oi in 0..n {
                if oi == bi {
                    continue;
                }
                let o0 = remaining[oi];
                for oj in (oi + 1)..n {
                    if oj == bi {
                        continue;
                    }
                    let o1 = remaining[oj];
                    let hero_hand =
                        [hero[0], hero[1], full_board[0], full_board[1], full_board[2], full_board[3], full_board[4]];
                    let opp_hand =
                        [o0, o1, full_board[0], full_board[1], full_board[2], full_board[3], full_board[4]];
                    let hr = eval::eval7(&hero_hand);
                    let or = eval::eval7(&opp_hand);
                    total += 1;
                    if hr < or {
                        wins += 1;
                    } else if hr == or {
                        ties += 1;
                    }
                }
            }
        }
    } else {
        // Flop: need 3+ board cards — use Monte Carlo sampling
        // Full enumeration is too expensive for preflop (C(48,5)*C(43,2) ~= 1.5 billion)
        // Use systematic sampling with a simple PRNG
        let mut rng_state: u64 = 0xdeadbeef12345678;
        for _ in 0..num_samples {
            // Fisher-Yates-ish: pick cards_needed cards from remaining
            let mut sample = remaining.clone();
            for k in 0..cards_needed {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let idx = k + (rng_state as usize % (sample.len() - k));
                sample.swap(k, idx);
            }

            let mut full_board = [0u8; 5];
            for i in 0..board.len() {
                full_board[i] = board[i];
            }
            for i in 0..board_needed {
                full_board[board.len() + i] = sample[i];
            }
            let o0 = sample[board_needed];
            let o1 = sample[board_needed + 1];

            let hero_hand =
                [hero[0], hero[1], full_board[0], full_board[1], full_board[2], full_board[3], full_board[4]];
            let opp_hand =
                [o0, o1, full_board[0], full_board[1], full_board[2], full_board[3], full_board[4]];
            let hr = eval::eval7(&hero_hand);
            let or = eval::eval7(&opp_hand);
            total += 1;
            if hr < or {
                wins += 1;
            } else if hr == or {
                ties += 1;
            }
        }
    }

    if total == 0 {
        return 0.5;
    }
    (wins as f64 + 0.5 * ties as f64) / total as f64
}

// --- Bucket abstraction configuration ---

/// Card abstraction configuration.
/// Specifies the number of buckets per street.
#[derive(Debug, Clone)]
pub struct CardAbstraction {
    /// Number of buckets per street: [preflop, flop, turn, river]
    pub buckets_per_street: [usize; 4],
}

impl Default for CardAbstraction {
    fn default() -> Self {
        CardAbstraction {
            buckets_per_street: [169, 200, 200, 200],
        }
    }
}

impl CardAbstraction {
    pub fn new(preflop: usize, flop: usize, turn: usize, river: usize) -> Self {
        CardAbstraction {
            buckets_per_street: [preflop, flop, turn, river],
        }
    }

    /// Get the number of buckets for a given street.
    pub fn num_buckets(&self, street: usize) -> usize {
        self.buckets_per_street[street]
    }
}

// --- EHS histogram for potential-aware abstraction ---

/// Compute an EHS histogram for a hand: the distribution of EHS values
/// after dealing the remaining board cards.
///
/// Used for potential-aware abstraction on flop and turn.
/// The histogram has `num_bins` bins over [0, 1].
pub fn ehs_histogram(hero: [u8; 2], board: &[u8], num_bins: usize) -> Vec<f64> {
    let mut dead = [false; 52];
    dead[hero[0] as usize] = true;
    dead[hero[1] as usize] = true;
    for &b in board {
        dead[b as usize] = true;
    }

    let mut remaining: Vec<u8> = Vec::with_capacity(52);
    for c in 0..52u8 {
        if !dead[c as usize] {
            remaining.push(c);
        }
    }

    let mut histogram = vec![0.0f64; num_bins];
    let mut count = 0usize;

    match board.len() {
        3 => {
            // Flop → compute EHS for each possible turn card
            for &turn_card in &remaining {
                let turn_board = [board[0], board[1], board[2], turn_card];
                let ehs = compute_ehs(hero, &turn_board);
                let bin = ehs_to_bin(ehs, num_bins);
                histogram[bin] += 1.0;
                count += 1;
            }
        }
        4 => {
            // Turn → compute EHS for each possible river card
            for &river_card in &remaining {
                let river_board = [board[0], board[1], board[2], board[3], river_card];
                let ehs = compute_ehs(hero, &river_board);
                let bin = ehs_to_bin(ehs, num_bins);
                histogram[bin] += 1.0;
                count += 1;
            }
        }
        5 => {
            // River: single EHS value → spike histogram
            let ehs = compute_ehs(hero, board);
            let bin = ehs_to_bin(ehs, num_bins);
            histogram[bin] += 1.0;
            count = 1;
        }
        _ => {
            // Preflop: not used (169 canonical hands)
            return histogram;
        }
    }

    // Normalize to probability distribution
    if count > 0 {
        let total: f64 = histogram.iter().sum();
        if total > 0.0 {
            for v in &mut histogram {
                *v /= total;
            }
        }
    }

    histogram
}

fn ehs_to_bin(ehs: f64, num_bins: usize) -> usize {
    let bin = (ehs * num_bins as f64) as usize;
    std::cmp::min(bin, num_bins - 1)
}

// --- Earth Mover's Distance ---

/// Compute the Earth Mover's Distance between two 1D histograms.
/// Both histograms must be normalized (sum to 1) and have the same length.
/// EMD for 1D distributions = sum of |CDF_a - CDF_b|.
pub fn earth_movers_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut emd = 0.0;
    let mut cum_diff = 0.0;
    for i in 0..a.len() {
        cum_diff += a[i] - b[i];
        emd += cum_diff.abs();
    }
    emd
}

// --- K-means clustering with EMD ---

/// K-means clustering of histograms using Earth Mover's Distance.
/// Returns cluster assignments (indices into 0..k).
pub fn kmeans_emd(histograms: &[Vec<f64>], k: usize, max_iters: usize) -> Vec<u16> {
    let n = histograms.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    if k >= n {
        return (0..n as u16).collect();
    }

    let num_bins = histograms[0].len();

    // Initialize centroids using k-means++ strategy
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);

    // First centroid: middle element
    centroids.push(histograms[n / 2].clone());

    // Remaining centroids: pick furthest from existing centroids
    for _ in 1..k {
        let mut best_idx = 0;
        let mut best_dist = -1.0f64;
        for i in 0..n {
            let min_dist = centroids
                .iter()
                .map(|c| earth_movers_distance(&histograms[i], c))
                .fold(f64::INFINITY, f64::min);
            if min_dist > best_dist {
                best_dist = min_dist;
                best_idx = i;
            }
        }
        centroids.push(histograms[best_idx].clone());
    }

    let mut assignments = vec![0u16; n];

    for _iter in 0..max_iters {
        // Assignment step: assign each histogram to nearest centroid
        let mut changed = false;
        for i in 0..n {
            let mut best_cluster = 0u16;
            let mut best_dist = f64::INFINITY;
            for (c, centroid) in centroids.iter().enumerate() {
                let dist = earth_movers_distance(&histograms[i], centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = c as u16;
                }
            }
            if assignments[i] != best_cluster {
                assignments[i] = best_cluster;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update step: recompute centroids as mean of assigned histograms
        let mut new_centroids = vec![vec![0.0f64; num_bins]; k];
        let mut counts = vec![0usize; k];

        for i in 0..n {
            let c = assignments[i] as usize;
            counts[c] += 1;
            for j in 0..num_bins {
                new_centroids[c][j] += histograms[i][j];
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..num_bins {
                    new_centroids[c][j] /= counts[c] as f64;
                }
                centroids[c] = new_centroids[c].clone();
            }
        }
    }

    assignments
}

// --- Potential-aware abstraction: bucket transition histograms ---

/// Build potential-aware histogram for flop or turn.
/// Instead of raw EHS, this computes the probability distribution over
/// next-round buckets after dealing the next card.
///
/// `hero`: 2 hole cards
/// `board`: current board (3 for flop, 4 for turn)
/// `next_round_assignments`: bucket assignments for the next round
///   - For flop: maps (hero, turn_board) → turn bucket
///   - For river: maps (hero, river_board) → river bucket
/// `next_round_num_buckets`: number of buckets in the next round
///
/// Returns a histogram of length `next_round_num_buckets` showing the
/// probability of transitioning to each next-round bucket.
pub fn potential_aware_histogram(
    hero: [u8; 2],
    board: &[u8],
    next_card_to_bucket: &dyn Fn(u8) -> u16,
    next_round_num_buckets: usize,
) -> Vec<f64> {
    let mut dead = [false; 52];
    dead[hero[0] as usize] = true;
    dead[hero[1] as usize] = true;
    for &b in board {
        dead[b as usize] = true;
    }

    let mut histogram = vec![0.0f64; next_round_num_buckets];
    let mut count = 0usize;

    for c in 0..52u8 {
        if dead[c as usize] {
            continue;
        }
        let bucket = next_card_to_bucket(c) as usize;
        if bucket < next_round_num_buckets {
            histogram[bucket] += 1.0;
            count += 1;
        }
    }

    // Normalize
    if count > 0 {
        let total: f64 = histogram.iter().sum();
        if total > 0.0 {
            for v in &mut histogram {
                *v /= total;
            }
        }
    }

    histogram
}

// --- Full abstraction builder ---

/// Precomputed card abstraction: bucket assignments for every (hand, board) combination.
/// Used during CFR to look up bucket IDs quickly.
#[derive(Debug, Clone)]
pub struct PrecomputedAbstraction {
    pub config: CardAbstraction,
    /// River bucket assignments: indexed by some encoding of (hand, board)
    /// Stored as a flat lookup for known river situations
    pub river_ehs_bins: usize,
    pub turn_ehs_bins: usize,
    pub flop_ehs_bins: usize,
}

/// Compute river buckets for a specific board.
/// Takes all C(remaining,2) possible hero hands and clusters by EHS.
pub fn compute_river_buckets(
    board: &[u8; 5],
    num_buckets: usize,
    num_ehs_bins: usize,
) -> (Vec<[u8; 2]>, Vec<u16>) {
    let mut dead = [false; 52];
    for &b in board {
        dead[b as usize] = true;
    }

    // Enumerate all possible hero hands
    let mut hands: Vec<[u8; 2]> = Vec::new();
    for c0 in 0..51u8 {
        if dead[c0 as usize] {
            continue;
        }
        for c1 in (c0 + 1)..52u8 {
            if dead[c1 as usize] {
                continue;
            }
            hands.push([c0, c1]);
        }
    }

    if hands.is_empty() || num_buckets >= hands.len() {
        let assignments = (0..hands.len() as u16).collect();
        return (hands, assignments);
    }

    // Compute EHS histogram for each hand (on river, this is a spike)
    let histograms: Vec<Vec<f64>> = hands
        .iter()
        .map(|h| ehs_histogram(*h, board, num_ehs_bins))
        .collect();

    // Cluster
    let assignments = kmeans_emd(&histograms, num_buckets, 50);

    (hands, assignments)
}

/// Compute turn buckets for a specific board (3 flop + 1 turn card).
/// Uses potential-aware abstraction: histogram of river bucket transitions.
pub fn compute_turn_buckets(
    board: &[u8; 4],
    num_buckets: usize,
    river_num_buckets: usize,
    _river_ehs_bins: usize,
) -> (Vec<[u8; 2]>, Vec<u16>) {
    let mut dead = [false; 52];
    for &b in board {
        dead[b as usize] = true;
    }

    // Enumerate hero hands
    let mut hands: Vec<[u8; 2]> = Vec::new();
    for c0 in 0..51u8 {
        if dead[c0 as usize] {
            continue;
        }
        for c1 in (c0 + 1)..52u8 {
            if dead[c1 as usize] {
                continue;
            }
            hands.push([c0, c1]);
        }
    }

    if hands.is_empty() || num_buckets >= hands.len() {
        let assignments = (0..hands.len() as u16).collect();
        return (hands, assignments);
    }

    // For each hand, compute potential-aware histogram:
    // probability distribution over river buckets for each possible river card
    let histograms: Vec<Vec<f64>> = hands
        .iter()
        .map(|h| {
            // For each possible river card, compute EHS, bin it
            let next_card_to_bucket = |river_card: u8| -> u16 {
                let river_board = [board[0], board[1], board[2], board[3], river_card];
                let ehs = compute_ehs(*h, &river_board);
                // Map EHS to a river bucket (simple binning for now)
                let bin = (ehs * river_num_buckets as f64) as u16;
                std::cmp::min(bin, (river_num_buckets - 1) as u16)
            };
            potential_aware_histogram(*h, board, &next_card_to_bucket, river_num_buckets)
        })
        .collect();

    let assignments = kmeans_emd(&histograms, num_buckets, 50);
    (hands, assignments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::*;

    #[test]
    fn test_preflop_bucket_pairs() {
        // AA
        let bucket = preflop_bucket(card(RANK_A, SUIT_SPADES), card(RANK_A, SUIT_HEARTS));
        assert_eq!(bucket, 12); // RANK_A = 12

        // 22
        let bucket = preflop_bucket(card(RANK_2, SUIT_CLUBS), card(RANK_2, SUIT_DIAMONDS));
        assert_eq!(bucket, 0); // RANK_2 = 0
    }

    #[test]
    fn test_preflop_bucket_suited() {
        // AKs: both spades
        let b1 = preflop_bucket(card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_SPADES));
        let b2 = preflop_bucket(card(RANK_A, SUIT_HEARTS), card(RANK_K, SUIT_HEARTS));
        // Same canonical hand → same bucket
        assert_eq!(b1, b2);
        // Should be in suited range (13..91)
        assert!(b1 >= 13 && b1 < 91);
    }

    #[test]
    fn test_preflop_bucket_offsuit() {
        // AKo: different suits
        let b1 = preflop_bucket(card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_HEARTS));
        let b2 = preflop_bucket(card(RANK_A, SUIT_DIAMONDS), card(RANK_K, SUIT_CLUBS));
        assert_eq!(b1, b2);
        // Should be in offsuit range (91..169)
        assert!(b1 >= 91 && b1 < 169);
    }

    #[test]
    fn test_preflop_169_unique_buckets() {
        // Verify all 169 canonical hands are reachable
        let mut seen = [false; 169];
        for c0 in 0..51u8 {
            for c1 in (c0 + 1)..52u8 {
                let b = preflop_bucket(c0, c1) as usize;
                assert!(b < 169, "Bucket {} out of range for cards ({}, {})", b, c0, c1);
                seen[b] = true;
            }
        }
        let count = seen.iter().filter(|&&s| s).count();
        assert_eq!(count, 169, "Expected 169 unique buckets, got {}", count);
    }

    #[test]
    fn test_ehs_river_nut_hand() {
        // Royal flush should have EHS ≈ 1.0
        let hero = [card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_SPADES)];
        let board = [
            card(RANK_Q, SUIT_SPADES),
            card(RANK_J, SUIT_SPADES),
            card(RANK_T, SUIT_SPADES),
            card(RANK_2, SUIT_HEARTS),
            card(RANK_3, SUIT_DIAMONDS),
        ];
        let ehs = compute_ehs(hero, &board);
        assert!(ehs > 0.99, "Royal flush EHS should be ~1.0, got {}", ehs);
    }

    #[test]
    fn test_ehs_river_weak_hand() {
        // 7-2 offsuit on a paired board should be weak
        let hero = [card(RANK_7, SUIT_SPADES), card(RANK_2, SUIT_HEARTS)];
        let board = [
            card(RANK_A, SUIT_CLUBS),
            card(RANK_K, SUIT_DIAMONDS),
            card(RANK_Q, SUIT_HEARTS),
            card(RANK_J, SUIT_SPADES),
            card(RANK_9, SUIT_CLUBS),
        ];
        let ehs = compute_ehs(hero, &board);
        assert!(ehs < 0.3, "7-2o on AKQJ9 should be weak, got {}", ehs);
    }

    #[test]
    fn test_ehs_range() {
        // EHS should always be in [0, 1]
        let hero = [card(RANK_A, SUIT_SPADES), card(RANK_A, SUIT_HEARTS)];
        let board = [
            card(RANK_K, SUIT_CLUBS),
            card(RANK_Q, SUIT_DIAMONDS),
            card(RANK_J, SUIT_HEARTS),
            card(RANK_T, SUIT_SPADES),
            card(RANK_9, SUIT_CLUBS),
        ];
        let ehs = compute_ehs(hero, &board);
        assert!(ehs >= 0.0 && ehs <= 1.0, "EHS out of range: {}", ehs);
    }

    #[test]
    fn test_earth_movers_distance_identical() {
        let a = vec![0.25, 0.25, 0.25, 0.25];
        let emd = earth_movers_distance(&a, &a);
        assert!(emd.abs() < 1e-10, "EMD of identical distributions should be 0");
    }

    #[test]
    fn test_earth_movers_distance_opposite() {
        // All mass at start vs all mass at end
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0, 1.0];
        let emd = earth_movers_distance(&a, &b);
        // EMD = |1| + |1| + |1| = 3.0 (CDF differences at each bin)
        assert!((emd - 3.0).abs() < 1e-10, "EMD should be 3.0, got {}", emd);
    }

    #[test]
    fn test_kmeans_emd_basic() {
        // Two clear clusters
        let histograms = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.1, 0.9],
        ];
        let assignments = kmeans_emd(&histograms, 2, 50);
        // First two should be in same cluster, last two in another
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
        assert_ne!(assignments[0], assignments[2]);
    }

    #[test]
    fn test_ehs_histogram_river() {
        // On the river, histogram should be a spike at the EHS bin
        let hero = [card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_SPADES)];
        let board = [
            card(RANK_Q, SUIT_SPADES),
            card(RANK_J, SUIT_SPADES),
            card(RANK_T, SUIT_SPADES),
            card(RANK_2, SUIT_HEARTS),
            card(RANK_3, SUIT_DIAMONDS),
        ];
        let hist = ehs_histogram(hero, &board, 10);
        let sum: f64 = hist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Histogram should sum to 1, got {}", sum);
        // Royal flush → spike at highest bin
        assert!(hist[9] > 0.9, "Royal flush should spike in highest bin");
    }

    #[test]
    fn test_card_abstraction_config() {
        let abs = CardAbstraction::default();
        assert_eq!(abs.num_buckets(0), 169);
        assert_eq!(abs.num_buckets(1), 200);
        assert_eq!(abs.num_buckets(2), 200);
        assert_eq!(abs.num_buckets(3), 200);
    }
}
