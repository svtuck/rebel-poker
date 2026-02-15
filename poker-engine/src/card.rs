/// Card representation: packed into a u8.
/// card = rank * 4 + suit, where rank in 0..13 (2=0, 3=1, ..., A=12), suit in 0..4.
/// Total: 52 cards, values 0..51.

pub const NUM_CARDS: u8 = 52;
pub const NUM_RANKS: u8 = 13;
pub const NUM_SUITS: u8 = 4;

/// Rank constants (0-indexed: 2=0, 3=1, ..., A=12)
pub const RANK_2: u8 = 0;
pub const RANK_3: u8 = 1;
pub const RANK_4: u8 = 2;
pub const RANK_5: u8 = 3;
pub const RANK_6: u8 = 4;
pub const RANK_7: u8 = 5;
pub const RANK_8: u8 = 6;
pub const RANK_9: u8 = 7;
pub const RANK_T: u8 = 8;
pub const RANK_J: u8 = 9;
pub const RANK_Q: u8 = 10;
pub const RANK_K: u8 = 11;
pub const RANK_A: u8 = 12;

/// Suit constants
pub const SUIT_CLUBS: u8 = 0;
pub const SUIT_DIAMONDS: u8 = 1;
pub const SUIT_HEARTS: u8 = 2;
pub const SUIT_SPADES: u8 = 3;

#[inline(always)]
pub fn card(rank: u8, suit: u8) -> u8 {
    debug_assert!(rank < NUM_RANKS);
    debug_assert!(suit < NUM_SUITS);
    rank * 4 + suit
}

#[inline(always)]
pub fn rank(card: u8) -> u8 {
    card / 4
}

#[inline(always)]
pub fn suit(card: u8) -> u8 {
    card % 4
}

/// Parse a card string like "Ah", "2c", "Ts" into a u8.
pub fn parse_card(s: &str) -> Option<u8> {
    let bytes = s.as_bytes();
    if bytes.len() != 2 {
        return None;
    }
    let r = match bytes[0] {
        b'2' => RANK_2,
        b'3' => RANK_3,
        b'4' => RANK_4,
        b'5' => RANK_5,
        b'6' => RANK_6,
        b'7' => RANK_7,
        b'8' => RANK_8,
        b'9' => RANK_9,
        b'T' | b't' => RANK_T,
        b'J' | b'j' => RANK_J,
        b'Q' | b'q' => RANK_Q,
        b'K' | b'k' => RANK_K,
        b'A' | b'a' => RANK_A,
        _ => return None,
    };
    let s = match bytes[1] {
        b'c' | b'C' => SUIT_CLUBS,
        b'd' | b'D' => SUIT_DIAMONDS,
        b'h' | b'H' => SUIT_HEARTS,
        b's' | b'S' => SUIT_SPADES,
        _ => return None,
    };
    Some(card(r, s))
}

/// Format a card u8 as a 2-char string like "Ah".
pub fn card_to_string(c: u8) -> String {
    let r = match rank(c) {
        RANK_2 => '2',
        RANK_3 => '3',
        RANK_4 => '4',
        RANK_5 => '5',
        RANK_6 => '6',
        RANK_7 => '7',
        RANK_8 => '8',
        RANK_9 => '9',
        RANK_T => 'T',
        RANK_J => 'J',
        RANK_Q => 'Q',
        RANK_K => 'K',
        RANK_A => 'A',
        _ => '?',
    };
    let s = match suit(c) {
        SUIT_CLUBS => 'c',
        SUIT_DIAMONDS => 'd',
        SUIT_HEARTS => 'h',
        SUIT_SPADES => 's',
        _ => '?',
    };
    format!("{}{}", r, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_card_roundtrip() {
        for c in 0..52u8 {
            assert_eq!(card(rank(c), suit(c)), c);
        }
    }

    #[test]
    fn test_parse_card() {
        assert_eq!(parse_card("Ah"), Some(card(RANK_A, SUIT_HEARTS)));
        assert_eq!(parse_card("2c"), Some(card(RANK_2, SUIT_CLUBS)));
        assert_eq!(parse_card("Ts"), Some(card(RANK_T, SUIT_SPADES)));
        assert_eq!(parse_card("Kd"), Some(card(RANK_K, SUIT_DIAMONDS)));
    }

    #[test]
    fn test_card_to_string() {
        assert_eq!(card_to_string(card(RANK_A, SUIT_HEARTS)), "Ah");
        assert_eq!(card_to_string(card(RANK_2, SUIT_CLUBS)), "2c");
    }
}
