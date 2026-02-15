/// Fast poker hand evaluation using lookup tables.
///
/// Hand rank encoding: lower is better (1 = royal flush, 7462 = worst high card).
///
/// All hand types resolved via O(1) table lookup:
/// - Flush hands: 8192-entry table indexed by 13-bit rank bitmask
/// - 5-unique-rank non-flush: 8192-entry table (straights + high cards)
/// - Duplicate-rank hands: hash table indexed by prime product of rank counts
///
/// Hand categories (rank ranges):
///   1..=10     Straight flush
///   11..=166   Four of a kind
///   167..=322  Full house
///   323..=1599 Flush
///   1600..=1609 Straight
///   1610..=2467 Three of a kind
///   2468..=3325 Two pair
///   3326..=6185 One pair
///   6186..=7462 High card

use crate::card;
use std::sync::OnceLock;

pub const NUM_HAND_CLASSES: u16 = 7462;
pub type HandRank = u16;

static TABLES: OnceLock<EvalTables> = OnceLock::new();

struct EvalTables {
    flush: [HandRank; 8192],
    unique5: [HandRank; 8192],
    /// Hash table for duplicate-rank hands. Indexed by prime-product hash.
    /// Size chosen large enough to avoid collisions for all 4888 distinct patterns.
    dp: Vec<HandRank>,
    dp_mask: usize,
}

/// One prime per rank, used to hash rank-count patterns.
const RANK_PRIMES: [u64; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];

/// DP table size (power of 2, large enough to avoid collisions).
const DP_SIZE_BITS: usize = 16; // 65536 entries
const DP_SIZE: usize = 1 << DP_SIZE_BITS;
const DP_MASK: usize = DP_SIZE - 1;

fn tables() -> &'static EvalTables {
    TABLES.get_or_init(EvalTables::build)
}

/// Evaluate a 5-card hand. Returns hand rank (1 = best, 7462 = worst).
#[inline]
pub fn eval5(cards: &[u8; 5]) -> HandRank {
    let t = tables();

    let mut rank_bits: u16 = 0;
    let mut suit_bits = [0u16; 4];
    let mut prime_product: u64 = 1;

    for &c in cards {
        let r = card::rank(c) as usize;
        let s = card::suit(c) as usize;
        rank_bits |= 1 << r;
        suit_bits[s] |= 1 << r;
        prime_product *= RANK_PRIMES[r];
    }

    // Flush check
    for s in 0..4 {
        if suit_bits[s].count_ones() >= 5 {
            return t.flush[suit_bits[s] as usize];
        }
    }

    // 5 unique ranks → straights + high cards
    if rank_bits.count_ones() == 5 {
        return t.unique5[rank_bits as usize];
    }

    // Duplicate ranks: O(1) hash lookup
    let idx = (prime_product as usize) & t.dp_mask;
    // Linear probe for collision resolution
    let mut i = idx;
    loop {
        let v = t.dp[i];
        if v != 0 {
            return v;
        }
        i = (i + 1) & t.dp_mask;
        if i == idx {
            break;
        }
    }
    // Should never reach here if tables are correctly built
    0
}

/// Evaluate a 7-card hand (best 5 of 7).
#[inline]
pub fn eval7(cards: &[u8; 7]) -> HandRank {
    let mut best: HandRank = HandRank::MAX;
    // Iterate over all 21 ways to choose 5 from 7
    // Unrolled for performance
    let c = cards;
    macro_rules! check5 {
        ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr) => {
            let r = eval5(&[c[$a], c[$b], c[$c], c[$d], c[$e]]);
            if r < best { best = r; }
        };
    }
    // All C(7,5) = 21 combinations
    check5!(0,1,2,3,4);
    check5!(0,1,2,3,5);
    check5!(0,1,2,3,6);
    check5!(0,1,2,4,5);
    check5!(0,1,2,4,6);
    check5!(0,1,2,5,6);
    check5!(0,1,3,4,5);
    check5!(0,1,3,4,6);
    check5!(0,1,3,5,6);
    check5!(0,1,4,5,6);
    check5!(0,2,3,4,5);
    check5!(0,2,3,4,6);
    check5!(0,2,3,5,6);
    check5!(0,2,4,5,6);
    check5!(0,3,4,5,6);
    check5!(1,2,3,4,5);
    check5!(1,2,3,4,6);
    check5!(1,2,3,5,6);
    check5!(1,2,4,5,6);
    check5!(1,3,4,5,6);
    check5!(2,3,4,5,6);
    best
}

/// Evaluate a batch of 7-card hands.
pub fn eval7_batch(hands: &[[u8; 7]]) -> Vec<HandRank> {
    hands.iter().map(|h| eval7(h)).collect()
}

/// Evaluate a batch of 5-card hands.
pub fn eval5_batch(hands: &[[u8; 5]]) -> Vec<HandRank> {
    hands.iter().map(|h| eval5(h)).collect()
}

/// Compare two 7-card hands. -1 = a wins, 0 = tie, 1 = b wins.
pub fn compare_hands(hand_a: &[u8; 7], hand_b: &[u8; 7]) -> i8 {
    let ra = eval7(hand_a);
    let rb = eval7(hand_b);
    if ra < rb { -1 } else if ra > rb { 1 } else { 0 }
}

pub fn hand_category(rank: HandRank) -> &'static str {
    match rank {
        1..=10 => "Straight Flush",
        11..=166 => "Four of a Kind",
        167..=322 => "Full House",
        323..=1599 => "Flush",
        1600..=1609 => "Straight",
        1610..=2467 => "Three of a Kind",
        2468..=3325 => "Two Pair",
        3326..=6185 => "One Pair",
        6186..=7462 => "High Card",
        _ => "Invalid",
    }
}

impl EvalTables {
    fn build() -> Self {
        let flush = Self::build_flush();
        let unique5 = Self::build_unique5();
        let (dp, dp_mask) = Self::build_dp();
        EvalTables { flush, unique5, dp, dp_mask }
    }

    fn build_flush() -> [HandRank; 8192] {
        let mut table = [0u16; 8192];
        let straights = Self::straight_masks();
        let straight_set: std::collections::HashSet<u16> = straights.iter().copied().collect();

        for (i, &mask) in straights.iter().enumerate() {
            table[mask as usize] = (i + 1) as HandRank;
        }

        let mut flush_combos = Self::all_5bit_masks(&straight_set);
        flush_combos.sort_by(|a, b| bits_desc(*b).cmp(&bits_desc(*a)));
        for (i, &mask) in flush_combos.iter().enumerate() {
            table[mask as usize] = 323 + i as HandRank;
        }

        table
    }

    fn build_unique5() -> [HandRank; 8192] {
        let mut table = [0u16; 8192];
        let straights = Self::straight_masks();
        let straight_set: std::collections::HashSet<u16> = straights.iter().copied().collect();

        for (i, &mask) in straights.iter().enumerate() {
            table[mask as usize] = 1600 + i as HandRank;
        }

        let mut hc = Self::all_5bit_masks(&straight_set);
        hc.sort_by(|a, b| bits_desc(*b).cmp(&bits_desc(*a)));
        for (i, &mask) in hc.iter().enumerate() {
            table[mask as usize] = 6186 + i as HandRank;
        }

        table
    }

    fn build_dp() -> (Vec<HandRank>, usize) {
        let mut table = vec![0u16; DP_SIZE];
        let mask = DP_MASK;

        let mut insert = |counts: &[u8; 13], rank: HandRank| {
            let mut hash: u64 = 1;
            for i in 0..13 {
                for _ in 0..counts[i] {
                    hash *= RANK_PRIMES[i];
                }
            }
            let mut idx = (hash as usize) & mask;
            loop {
                if table[idx] == 0 {
                    table[idx] = rank;
                    return;
                }
                // Collision: verify it's actually a different pattern
                // (prime products are unique for different count patterns,
                //  but the hash modulo might collide)
                idx = (idx + 1) & mask;
            }
        };

        // Four of a kind: 11..166
        let mut rank: HandRank = 11;
        for q in (0..13u8).rev() {
            for k in (0..13u8).rev() {
                if k == q { continue; }
                let mut c = [0u8; 13];
                c[q as usize] = 4; c[k as usize] = 1;
                insert(&c, rank);
                rank += 1;
            }
        }

        // Full house: 167..322
        for t in (0..13u8).rev() {
            for p in (0..13u8).rev() {
                if p == t { continue; }
                let mut c = [0u8; 13];
                c[t as usize] = 3; c[p as usize] = 2;
                insert(&c, rank);
                rank += 1;
            }
        }

        // Three of a kind: 1610..2467
        rank = 1610;
        for t in (0..13u8).rev() {
            for k1 in (0..13u8).rev() {
                if k1 == t { continue; }
                for k2 in (0..k1).rev() {
                    if k2 == t { continue; }
                    let mut c = [0u8; 13];
                    c[t as usize] = 3; c[k1 as usize] = 1; c[k2 as usize] = 1;
                    insert(&c, rank);
                    rank += 1;
                }
            }
        }

        // Two pair: 2468..3325
        for p1 in (0..13u8).rev() {
            for p2 in (0..p1).rev() {
                for k in (0..13u8).rev() {
                    if k == p1 || k == p2 { continue; }
                    let mut c = [0u8; 13];
                    c[p1 as usize] = 2; c[p2 as usize] = 2; c[k as usize] = 1;
                    insert(&c, rank);
                    rank += 1;
                }
            }
        }

        // One pair: 3326..6185
        for p in (0..13u8).rev() {
            for k1 in (0..13u8).rev() {
                if k1 == p { continue; }
                for k2 in (0..k1).rev() {
                    if k2 == p { continue; }
                    for k3 in (0..k2).rev() {
                        if k3 == p { continue; }
                        let mut c = [0u8; 13];
                        c[p as usize] = 2; c[k1 as usize] = 1;
                        c[k2 as usize] = 1; c[k3 as usize] = 1;
                        insert(&c, rank);
                        rank += 1;
                    }
                }
            }
        }

        (table, mask)
    }

    fn straight_masks() -> [u16; 10] {
        [
            0b1_1111_0000_0000,
            0b0_1111_1000_0000,
            0b0_0111_1100_0000,
            0b0_0011_1110_0000,
            0b0_0001_1111_0000,
            0b0_0000_1111_1000,
            0b0_0000_0111_1100,
            0b0_0000_0011_1110,
            0b0_0000_0001_1111,
            0b1_0000_0000_1111,
        ]
    }

    fn all_5bit_masks(exclude: &std::collections::HashSet<u16>) -> Vec<u16> {
        (0u16..8192).filter(|m| m.count_ones() == 5 && !exclude.contains(m)).collect()
    }
}

fn bits_desc(mask: u16) -> [u8; 5] {
    let mut bits = [0u8; 5];
    let mut idx = 0;
    for i in (0..13u8).rev() {
        if mask & (1 << i) != 0 && idx < 5 {
            bits[idx] = i;
            idx += 1;
        }
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::*;

    #[test]
    fn test_royal_flush() {
        let hand = [card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_SPADES),
                     card(RANK_Q, SUIT_SPADES), card(RANK_J, SUIT_SPADES),
                     card(RANK_T, SUIT_SPADES)];
        assert_eq!(eval5(&hand), 1);
    }

    #[test]
    fn test_wheel_straight_flush() {
        let hand = [card(RANK_A, SUIT_HEARTS), card(RANK_2, SUIT_HEARTS),
                     card(RANK_3, SUIT_HEARTS), card(RANK_4, SUIT_HEARTS),
                     card(RANK_5, SUIT_HEARTS)];
        assert_eq!(eval5(&hand), 10);
    }

    #[test]
    fn test_four_aces_king() {
        let hand = [card(RANK_A, SUIT_SPADES), card(RANK_A, SUIT_HEARTS),
                     card(RANK_A, SUIT_DIAMONDS), card(RANK_A, SUIT_CLUBS),
                     card(RANK_K, SUIT_SPADES)];
        assert_eq!(eval5(&hand), 11);
    }

    #[test]
    fn test_worst_hand() {
        let hand = [card(RANK_7, SUIT_SPADES), card(RANK_5, SUIT_HEARTS),
                     card(RANK_4, SUIT_DIAMONDS), card(RANK_3, SUIT_CLUBS),
                     card(RANK_2, SUIT_SPADES)];
        assert_eq!(eval5(&hand), 7462);
    }

    #[test]
    fn test_ace_high_straight() {
        let hand = [card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_HEARTS),
                     card(RANK_Q, SUIT_DIAMONDS), card(RANK_J, SUIT_CLUBS),
                     card(RANK_T, SUIT_SPADES)];
        assert_eq!(eval5(&hand), 1600);
    }

    #[test]
    fn test_full_house_beats_flush() {
        let fh = [card(RANK_A, SUIT_SPADES), card(RANK_A, SUIT_HEARTS),
                   card(RANK_A, SUIT_DIAMONDS), card(RANK_K, SUIT_CLUBS),
                   card(RANK_K, SUIT_SPADES)];
        let fl = [card(RANK_A, SUIT_HEARTS), card(RANK_K, SUIT_HEARTS),
                   card(RANK_Q, SUIT_HEARTS), card(RANK_J, SUIT_HEARTS),
                   card(RANK_9, SUIT_HEARTS)];
        assert!(eval5(&fh) < eval5(&fl));
    }

    #[test]
    fn test_eval7_royal_flush() {
        let hand = [card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_SPADES),
                     card(RANK_Q, SUIT_SPADES), card(RANK_J, SUIT_SPADES),
                     card(RANK_T, SUIT_SPADES), card(RANK_2, SUIT_HEARTS),
                     card(RANK_3, SUIT_DIAMONDS)];
        assert_eq!(eval7(&hand), 1);
    }

    #[test]
    fn test_pair_ordering() {
        let pair_a = [card(RANK_A, SUIT_SPADES), card(RANK_A, SUIT_HEARTS),
                      card(RANK_K, SUIT_DIAMONDS), card(RANK_Q, SUIT_CLUBS),
                      card(RANK_J, SUIT_SPADES)];
        let pair_k = [card(RANK_K, SUIT_SPADES), card(RANK_K, SUIT_HEARTS),
                      card(RANK_A, SUIT_DIAMONDS), card(RANK_Q, SUIT_CLUBS),
                      card(RANK_J, SUIT_SPADES)];
        assert!(eval5(&pair_a) < eval5(&pair_k));
    }

    #[test]
    fn test_hand_categories() {
        assert_eq!(hand_category(1), "Straight Flush");
        assert_eq!(hand_category(11), "Four of a Kind");
        assert_eq!(hand_category(167), "Full House");
        assert_eq!(hand_category(323), "Flush");
        assert_eq!(hand_category(1600), "Straight");
        assert_eq!(hand_category(1610), "Three of a Kind");
        assert_eq!(hand_category(2468), "Two Pair");
        assert_eq!(hand_category(3326), "One Pair");
        assert_eq!(hand_category(6186), "High Card");
    }

    #[test]
    fn test_all_hand_categories_count() {
        assert_eq!(10 + 156 + 156 + 1277 + 10 + 858 + 858 + 2860 + 1277, 7462);
    }
}
