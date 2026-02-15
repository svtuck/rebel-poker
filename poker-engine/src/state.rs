/// HUNL Texas Hold'em game state representation.
///
/// Designed for vectorized CFR: states are small, copyable, and transitions are branchless
/// where possible. The solver holds many states simultaneously.

use crate::card;
use crate::eval;

/// Betting rounds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Street {
    Preflop = 0,
    Flop = 1,
    Turn = 2,
    River = 3,
}

/// Player actions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Fold,
    Check,
    Call,
    Raise(u32), // raise TO this amount (total chips wagered this street by this player)
    AllIn,
}

/// Compact game state for HUNL.
#[derive(Debug, Clone)]
pub struct GameState {
    /// Hole cards: [p0_card0, p0_card1, p1_card0, p1_card1], 0xFF = unknown
    pub hole_cards: [u8; 4],
    /// Board cards: up to 5, 0xFF = not dealt
    pub board: [u8; 5],
    /// Number of board cards dealt
    pub board_len: u8,
    /// Current street
    pub street: Street,
    /// Player stacks (remaining chips not yet wagered)
    pub stacks: [u32; 2],
    /// Amount each player has put in the pot total (across all streets)
    pub pot_contrib: [u32; 2],
    /// Amount each player has put in this street
    pub street_contrib: [u32; 2],
    /// Current bet to match this street (the highest wager this street)
    pub current_bet: u32,
    /// Who acts next: 0 or 1, or 2 = terminal, 3 = chance
    pub active_player: u8,
    /// Number of actions taken this street
    pub actions_this_street: u8,
    /// Is the hand over?
    pub is_terminal: bool,
    /// If terminal, who folded? 0xFF = showdown
    pub folded_player: u8,
    /// Size of the last raise this street (for min-raise calculation)
    pub last_raise_size: u32,
    /// Betting history encoded as compact bytes for info set keys
    pub history: Vec<u8>,
}

/// Initial stack size (default for HUNL)
pub const DEFAULT_STACK: u32 = 200; // 200 big blinds
pub const SMALL_BLIND: u32 = 1;
pub const BIG_BLIND: u32 = 2;

impl GameState {
    /// Create a new game state with default stacks.
    /// Blinds are posted: P0 = SB (1), P1 = BB (2).
    /// P0 (SB) acts first preflop.
    pub fn new(stack: u32) -> Self {
        GameState {
            hole_cards: [0xFF; 4],
            board: [0xFF; 5],
            board_len: 0,
            street: Street::Preflop,
            stacks: [stack - SMALL_BLIND, stack - BIG_BLIND],
            pot_contrib: [SMALL_BLIND, BIG_BLIND],
            street_contrib: [SMALL_BLIND, BIG_BLIND],
            current_bet: BIG_BLIND,
            active_player: 0, // SB acts first preflop
            actions_this_street: 0,
            is_terminal: false,
            folded_player: 0xFF,
            last_raise_size: BIG_BLIND, // The BB is effectively a raise of 1 BB
            history: Vec::new(),
        }
    }

    /// Deal hole cards to both players.
    pub fn deal_hole_cards(&mut self, p0: [u8; 2], p1: [u8; 2]) {
        self.hole_cards = [p0[0], p0[1], p1[0], p1[1]];
    }

    /// Deal the flop (3 cards).
    pub fn deal_flop(&mut self, cards: [u8; 3]) {
        self.board[0] = cards[0];
        self.board[1] = cards[1];
        self.board[2] = cards[2];
        self.board_len = 3;
        self.street = Street::Flop;
        self.reset_street_state();
    }

    /// Deal the turn (1 card).
    pub fn deal_turn(&mut self, c: u8) {
        self.board[3] = c;
        self.board_len = 4;
        self.street = Street::Turn;
        self.reset_street_state();
    }

    /// Deal the river (1 card).
    pub fn deal_river(&mut self, c: u8) {
        self.board[4] = c;
        self.board_len = 5;
        self.street = Street::River;
        self.reset_street_state();
    }

    /// Reset per-street tracking when a new street begins.
    fn reset_street_state(&mut self) {
        self.street_contrib = [0, 0];
        self.current_bet = 0;
        self.actions_this_street = 0;
        self.last_raise_size = BIG_BLIND; // min raise resets to 1 BB
        self.active_player = 0; // SB/P0 acts first postflop
    }

    /// Get total pot size.
    pub fn pot(&self) -> u32 {
        self.pot_contrib[0] + self.pot_contrib[1]
    }

    /// Get legal actions for the current player.
    pub fn legal_actions(&self) -> Vec<Action> {
        if self.is_terminal || self.active_player >= 2 {
            return Vec::new();
        }

        let player = self.active_player as usize;
        let to_call = self.current_bet.saturating_sub(self.street_contrib[player]);
        let stack = self.stacks[player];

        let mut actions = Vec::with_capacity(6);

        // Fold: only if facing a bet
        if to_call > 0 {
            actions.push(Action::Fold);
        }

        // Check or Call
        if to_call == 0 {
            actions.push(Action::Check);
        } else if to_call < stack {
            actions.push(Action::Call);
        }

        // Raise/All-in
        // Min raise = current_bet + last_raise_size (or BB, whichever is larger)
        let min_raise_to = self.current_bet + std::cmp::max(self.last_raise_size, BIG_BLIND);
        let max_raise_to = self.street_contrib[player] + stack; // our street contrib + remaining stack

        if to_call < stack {
            // We have chips beyond calling — can raise
            if min_raise_to < max_raise_to {
                // There's room for at least a min-raise that isn't all-in
                actions.push(Action::Raise(min_raise_to));
            }
            if max_raise_to > self.current_bet {
                actions.push(Action::AllIn);
            }
        } else if to_call >= stack && stack > 0 {
            // Can only call all-in (effectively an all-in call)
            actions.push(Action::AllIn);
        }

        actions
    }

    /// Apply an action and return the new state.
    pub fn apply_action(&self, action: Action) -> GameState {
        let mut next = self.clone();
        let player = self.active_player as usize;

        match action {
            Action::Fold => {
                next.is_terminal = true;
                next.folded_player = self.active_player;
                next.history.push(b'f');
            }
            Action::Check => {
                next.actions_this_street += 1;
                next.history.push(b'k');
                // Check-check ends the street (2 checks), or BB checks preflop after limp
                if next.actions_this_street >= 2 {
                    next.advance_street();
                } else {
                    next.active_player = 1 - self.active_player;
                }
            }
            Action::Call => {
                let to_call = self.current_bet - self.street_contrib[player];
                let actual_call = std::cmp::min(to_call, self.stacks[player]);
                next.stacks[player] -= actual_call;
                next.pot_contrib[player] += actual_call;
                next.street_contrib[player] += actual_call;
                next.actions_this_street += 1;
                next.history.push(b'c');

                // Preflop limp: SB calls BB, BB still gets option
                if self.street == Street::Preflop
                    && self.actions_this_street == 0
                    && self.current_bet == BIG_BLIND
                    && self.active_player == 0
                {
                    // SB limps — BB gets to act
                    next.active_player = 1;
                } else {
                    // Call closes the action — advance street
                    next.advance_street();
                }
            }
            Action::Raise(amount) => {
                let raise_size = amount - self.current_bet; // how much more than the current bet
                let to_put_in = amount - self.street_contrib[player];
                let actual = std::cmp::min(to_put_in, self.stacks[player]);
                next.stacks[player] -= actual;
                next.pot_contrib[player] += actual;
                next.street_contrib[player] += actual;
                next.current_bet = amount;
                next.last_raise_size = raise_size;
                next.actions_this_street += 1;
                next.history.push(b'r');
                next.active_player = 1 - self.active_player;
            }
            Action::AllIn => {
                let all_in_amount = self.stacks[player];
                next.pot_contrib[player] += all_in_amount;
                next.street_contrib[player] += all_in_amount;
                let new_total = next.street_contrib[player];
                if new_total > next.current_bet {
                    let raise_size = new_total - next.current_bet;
                    next.last_raise_size = std::cmp::max(raise_size, next.last_raise_size);
                    next.current_bet = new_total;
                }
                next.stacks[player] = 0;
                next.actions_this_street += 1;
                next.history.push(b'a');

                let opponent = 1 - player;
                // If both players are all-in, run out remaining streets
                if next.stacks[opponent] == 0 {
                    next.advance_street();
                } else {
                    // Opponent still has chips — they get to act
                    next.active_player = 1 - self.active_player;
                }
            }
        }

        next
    }

    /// Advance to the next street, or to showdown if on the river.
    fn advance_street(&mut self) {
        // Check if both players are all-in — if so, skip to showdown
        let both_all_in = self.stacks[0] == 0 && self.stacks[1] == 0;

        if both_all_in || self.street == Street::River {
            // Showdown
            self.is_terminal = true;
            self.active_player = 2;
            self.history.push(b'/');
            return;
        }

        match self.street {
            Street::Preflop => {
                self.active_player = 3; // chance node (deal flop)
            }
            Street::Flop => {
                self.active_player = 3; // chance node (deal turn)
            }
            Street::Turn => {
                self.active_player = 3; // chance node (deal river)
            }
            Street::River => unreachable!(), // handled above
        }
        self.history.push(b'/');
    }

    /// Get terminal utility for a player (in chips won/lost relative to starting stack).
    pub fn terminal_utility(&self, player: usize) -> f64 {
        assert!(self.is_terminal);

        if self.folded_player != 0xFF {
            // Someone folded
            let folder = self.folded_player as usize;
            if folder == player {
                -(self.pot_contrib[player] as f64)
            } else {
                self.pot_contrib[1 - player] as f64
            }
        } else {
            // Showdown: evaluate hands
            let p0_cards = self.player_hand(0);
            let p1_cards = self.player_hand(1);

            match (p0_cards, p1_cards) {
                (Some(h0), Some(h1)) => {
                    let r0 = eval::eval7(&h0);
                    let r1 = eval::eval7(&h1);
                    let total_pot = self.pot() as f64;
                    if r0 < r1 {
                        // P0 wins (lower rank = better)
                        if player == 0 {
                            total_pot - self.pot_contrib[0] as f64
                        } else {
                            -(self.pot_contrib[1] as f64)
                        }
                    } else if r0 > r1 {
                        // P1 wins
                        if player == 1 {
                            total_pot - self.pot_contrib[1] as f64
                        } else {
                            -(self.pot_contrib[0] as f64)
                        }
                    } else {
                        // Tie: split pot
                        let half = total_pot / 2.0;
                        half - self.pot_contrib[player] as f64
                    }
                }
                _ => 0.0, // Cards not dealt — shouldn't happen at showdown
            }
        }
    }

    /// Get the 7-card hand (2 hole + 5 board) for a player.
    fn player_hand(&self, player: usize) -> Option<[u8; 7]> {
        let h0 = self.hole_cards[player * 2];
        let h1 = self.hole_cards[player * 2 + 1];
        if h0 == 0xFF || self.board_len < 5 {
            return None;
        }
        Some([
            h0, h1, self.board[0], self.board[1], self.board[2], self.board[3], self.board[4],
        ])
    }

    /// Generate information set key for a player.
    /// Format: "AhKd||xrc/" (hole cards + visible board + action history)
    pub fn infoset_key(&self, player: usize) -> String {
        let h0 = self.hole_cards[player * 2];
        let h1 = self.hole_cards[player * 2 + 1];

        let mut key = String::with_capacity(64);

        // Hole cards (sorted for canonical form)
        let (c0, c1) = if h0 <= h1 { (h0, h1) } else { (h1, h0) };
        key.push_str(&card::card_to_string(c0));
        key.push_str(&card::card_to_string(c1));
        key.push('|');

        // Board cards
        for i in 0..self.board_len as usize {
            if i > 0 {
                key.push(' ');
            }
            key.push_str(&card::card_to_string(self.board[i]));
        }
        key.push('|');

        // Action history
        for &b in &self.history {
            key.push(b as char);
        }

        key
    }
}

// --- Action abstraction ---

/// Configuration for abstract action set.
/// Bet sizes are expressed as fractions of the pot.
#[derive(Debug, Clone)]
pub struct ActionAbstraction {
    /// Bet sizes as fractions of the pot (e.g., 0.25, 0.5, 1.0, 2.0)
    pub bet_fractions: Vec<f64>,
}

impl Default for ActionAbstraction {
    fn default() -> Self {
        ActionAbstraction {
            bet_fractions: vec![0.25, 0.5, 1.0, 2.0],
        }
    }
}

impl ActionAbstraction {
    /// Create an action abstraction with custom bet sizes.
    pub fn new(bet_fractions: Vec<f64>) -> Self {
        let mut fracs = bet_fractions;
        fracs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        fracs.dedup();
        ActionAbstraction {
            bet_fractions: fracs,
        }
    }

    /// Get abstract legal actions for the current game state.
    /// Returns: fold, check/call, pot-relative bet sizes, all-in.
    pub fn abstract_actions(&self, state: &GameState) -> Vec<Action> {
        if state.is_terminal || state.active_player >= 2 {
            return Vec::new();
        }

        let player = state.active_player as usize;
        let to_call = state.current_bet.saturating_sub(state.street_contrib[player]);
        let stack = state.stacks[player];

        let mut actions = Vec::with_capacity(self.bet_fractions.len() + 3);

        // Fold (only if facing a bet)
        if to_call > 0 {
            actions.push(Action::Fold);
        }

        // Check or Call
        if to_call == 0 {
            actions.push(Action::Check);
        } else if to_call < stack {
            actions.push(Action::Call);
        }

        if to_call >= stack {
            // Can only call all-in
            if stack > 0 {
                actions.push(Action::AllIn);
            }
            return actions;
        }

        // Pot-relative bet/raise sizes
        let pot_after_call = state.pot() + to_call; // pot if we call first
        let max_raise_to = state.street_contrib[player] + stack;
        let min_raise_to = state.current_bet + std::cmp::max(state.last_raise_size, BIG_BLIND);

        let mut added_sizes: Vec<u32> = Vec::new();

        for &frac in &self.bet_fractions {
            // Bet size = frac * pot_after_call, expressed as a raise-to amount
            let bet_amount = (frac * pot_after_call as f64).round() as u32;
            let raise_to = state.current_bet + bet_amount;

            // Clamp to [min_raise, max_raise)
            let clamped = std::cmp::max(raise_to, min_raise_to);
            if clamped >= max_raise_to {
                continue; // This would be all-in or larger
            }

            // Avoid duplicates
            if !added_sizes.contains(&clamped) {
                added_sizes.push(clamped);
                actions.push(Action::Raise(clamped));
            }
        }

        // Always include all-in
        if max_raise_to > state.current_bet {
            actions.push(Action::AllIn);
        }

        actions
    }

    /// Translate an off-tree action to the nearest abstract action.
    /// Used when the opponent makes a bet that isn't in our abstraction.
    /// Returns the closest abstract action by pot-relative distance.
    pub fn translate_action(&self, state: &GameState, actual_action: Action) -> Action {
        let abstract_actions = self.abstract_actions(state);

        if abstract_actions.is_empty() {
            return actual_action;
        }

        // If the exact action is in our abstraction, use it
        if abstract_actions.contains(&actual_action) {
            return actual_action;
        }

        // For non-raise actions, return as-is (fold/check/call are always in abstraction)
        let actual_amount = match actual_action {
            Action::Raise(a) => a,
            Action::AllIn => return Action::AllIn, // always in abstraction
            other => return other,
        };

        // Find the nearest abstract raise by absolute distance
        let mut best_action = Action::AllIn;
        let mut best_dist = u32::MAX;

        for &a in &abstract_actions {
            match a {
                Action::Raise(abstract_amount) => {
                    let dist = if actual_amount > abstract_amount {
                        actual_amount - abstract_amount
                    } else {
                        abstract_amount - actual_amount
                    };
                    if dist < best_dist {
                        best_dist = dist;
                        best_action = a;
                    }
                }
                Action::AllIn => {
                    let player = state.active_player as usize;
                    let all_in_to = state.street_contrib[player] + state.stacks[player];
                    let dist = if actual_amount > all_in_to {
                        actual_amount - all_in_to
                    } else {
                        all_in_to - actual_amount
                    };
                    if dist < best_dist {
                        best_dist = dist;
                        best_action = a;
                    }
                }
                _ => {}
            }
        }

        best_action
    }
}

// --- Vectorized batch operations ---

/// Apply the same action to multiple game states.
pub fn batch_apply_action(states: &[GameState], action: Action) -> Vec<GameState> {
    states.iter().map(|s| s.apply_action(action)).collect()
}

/// Get legal actions for multiple states (assuming they share the same action history).
pub fn batch_legal_actions(states: &[GameState]) -> Vec<Action> {
    if states.is_empty() {
        return Vec::new();
    }
    states[0].legal_actions()
}

/// Evaluate terminal utilities for a batch of terminal states.
pub fn batch_terminal_utility(states: &[GameState], player: usize) -> Vec<f64> {
    states.iter().map(|s| s.terminal_utility(player)).collect()
}

/// Generate info set keys for a batch of states.
pub fn batch_infoset_keys(states: &[GameState], player: usize) -> Vec<String> {
    states.iter().map(|s| s.infoset_key(player)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::*;

    #[test]
    fn test_new_state() {
        let state = GameState::new(200);
        assert_eq!(state.stacks[0], 199); // SB posted
        assert_eq!(state.stacks[1], 198); // BB posted
        assert_eq!(state.pot_contrib[0], 1);
        assert_eq!(state.pot_contrib[1], 2);
        assert_eq!(state.street_contrib[0], 1);
        assert_eq!(state.street_contrib[1], 2);
        assert_eq!(state.current_bet, 2);
        assert_eq!(state.active_player, 0);
        assert!(!state.is_terminal);
    }

    #[test]
    fn test_fold_preflop() {
        let mut state = GameState::new(200);
        state.deal_hole_cards(
            [card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_SPADES)],
            [card(RANK_2, SUIT_HEARTS), card(RANK_7, SUIT_DIAMONDS)],
        );

        let next = state.apply_action(Action::Fold);
        assert!(next.is_terminal);
        assert_eq!(next.folded_player, 0);
        // P0 folded, loses SB (1 chip)
        assert_eq!(next.terminal_utility(0), -1.0);
        assert_eq!(next.terminal_utility(1), 1.0);
    }

    #[test]
    fn test_call_raise_fold() {
        let mut state = GameState::new(200);
        state.deal_hole_cards(
            [card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_SPADES)],
            [card(RANK_2, SUIT_HEARTS), card(RANK_7, SUIT_DIAMONDS)],
        );

        // SB raises to 6
        let s1 = state.apply_action(Action::Raise(6));
        assert_eq!(s1.active_player, 1);
        assert_eq!(s1.current_bet, 6);
        assert_eq!(s1.street_contrib[0], 6);
        assert_eq!(s1.pot_contrib[0], 6);
        assert_eq!(s1.stacks[0], 194);

        // BB folds
        let s2 = s1.apply_action(Action::Fold);
        assert!(s2.is_terminal);
        assert_eq!(s2.terminal_utility(0), 2.0); // wins BB's 2
        assert_eq!(s2.terminal_utility(1), -2.0); // loses BB's 2
    }

    #[test]
    fn test_limp_check_to_flop() {
        let mut state = GameState::new(200);
        state.deal_hole_cards(
            [card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_SPADES)],
            [card(RANK_2, SUIT_HEARTS), card(RANK_7, SUIT_DIAMONDS)],
        );

        // SB calls (limp)
        let s1 = state.apply_action(Action::Call);
        assert_eq!(s1.active_player, 1); // BB gets to act
        assert_eq!(s1.street_contrib[0], 2); // SB put in 2 total
        assert_eq!(s1.pot_contrib[0], 2);

        // BB checks
        let s2 = s1.apply_action(Action::Check);
        assert_eq!(s2.active_player, 3); // chance node — deal flop
        assert!(!s2.is_terminal);
    }

    #[test]
    fn test_full_hand_to_showdown() {
        let mut state = GameState::new(200);
        // P0: As Ks (strong)
        // P1: 2h 7d (weak)
        state.deal_hole_cards(
            [card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_SPADES)],
            [card(RANK_2, SUIT_HEARTS), card(RANK_7, SUIT_DIAMONDS)],
        );

        // Preflop: SB calls, BB checks
        let s1 = state.apply_action(Action::Call);
        let s2 = s1.apply_action(Action::Check);
        assert_eq!(s2.active_player, 3); // chance

        // Deal flop: Tc 9s 2c
        let mut s3 = s2.clone();
        s3.deal_flop([
            card(RANK_T, SUIT_CLUBS),
            card(RANK_9, SUIT_SPADES),
            card(RANK_2, SUIT_CLUBS),
        ]);
        assert_eq!(s3.active_player, 0);
        assert_eq!(s3.street, Street::Flop);
        assert_eq!(s3.street_contrib, [0, 0]); // reset for new street

        // Flop: check-check
        let s4 = s3.apply_action(Action::Check);
        let s5 = s4.apply_action(Action::Check);
        assert_eq!(s5.active_player, 3); // chance

        // Deal turn: 3h
        let mut s6 = s5.clone();
        s6.deal_turn(card(RANK_3, SUIT_HEARTS));

        // Turn: check-check
        let s7 = s6.apply_action(Action::Check);
        let s8 = s7.apply_action(Action::Check);
        assert_eq!(s8.active_player, 3); // chance

        // Deal river: 4d
        let mut s9 = s8.clone();
        s9.deal_river(card(RANK_4, SUIT_DIAMONDS));

        // River: check-check → showdown
        let s10 = s9.apply_action(Action::Check);
        let s11 = s10.apply_action(Action::Check);
        assert!(s11.is_terminal);

        // P0 has A-K high, P1 has pair of 2s (2h on board)
        // P1 wins with pair of 2s
        let u0 = s11.terminal_utility(0);
        let u1 = s11.terminal_utility(1);
        assert!(u0 < 0.0, "P0 should lose (A-K high vs pair of 2s)");
        assert!(u1 > 0.0, "P1 should win");
        // Zero-sum
        assert!((u0 + u1).abs() < 0.001);
    }

    #[test]
    fn test_all_in_preflop() {
        let mut state = GameState::new(200);
        state.deal_hole_cards(
            [card(RANK_A, SUIT_SPADES), card(RANK_A, SUIT_HEARTS)],
            [card(RANK_K, SUIT_SPADES), card(RANK_K, SUIT_HEARTS)],
        );

        // SB shoves all-in
        let s1 = state.apply_action(Action::AllIn);
        assert_eq!(s1.stacks[0], 0);
        assert_eq!(s1.pot_contrib[0], 200);
        assert_eq!(s1.active_player, 1); // BB to act

        // BB calls all-in
        let s2 = s1.apply_action(Action::AllIn);
        assert_eq!(s2.stacks[1], 0);
        // Both all-in — should be terminal (need board cards dealt to evaluate)
        assert!(s2.is_terminal);
    }

    #[test]
    fn test_infoset_key() {
        let mut state = GameState::new(200);
        state.deal_hole_cards(
            [card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_SPADES)],
            [card(RANK_2, SUIT_HEARTS), card(RANK_7, SUIT_DIAMONDS)],
        );

        let key0 = state.infoset_key(0);
        let key1 = state.infoset_key(1);

        assert!(key0.contains("As"));
        assert!(key0.contains("Ks"));
        assert!(key1.contains("2h"));
        assert!(key1.contains("7d"));
        assert!(!key0.contains("2h"));
        assert!(!key1.contains("As"));
    }

    #[test]
    fn test_street_contrib_reset() {
        let mut state = GameState::new(200);
        state.deal_hole_cards(
            [card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_SPADES)],
            [card(RANK_2, SUIT_HEARTS), card(RANK_7, SUIT_DIAMONDS)],
        );

        // Preflop: SB raises to 6
        let s1 = state.apply_action(Action::Raise(6));
        assert_eq!(s1.street_contrib[0], 6);

        // BB calls
        let s2 = s1.apply_action(Action::Call);
        assert_eq!(s2.street_contrib[1], 6);
        assert_eq!(s2.pot_contrib[0], 6);
        assert_eq!(s2.pot_contrib[1], 6);

        // Deal flop
        let mut s3 = s2.clone();
        s3.deal_flop([
            card(RANK_T, SUIT_CLUBS),
            card(RANK_9, SUIT_SPADES),
            card(RANK_2, SUIT_CLUBS),
        ]);

        // Street contrib should be reset
        assert_eq!(s3.street_contrib, [0, 0]);
        assert_eq!(s3.current_bet, 0);
        // But pot_contrib preserved
        assert_eq!(s3.pot_contrib[0], 6);
        assert_eq!(s3.pot_contrib[1], 6);
    }

    #[test]
    fn test_min_raise_size() {
        let mut state = GameState::new(200);
        state.deal_hole_cards(
            [card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_SPADES)],
            [card(RANK_2, SUIT_HEARTS), card(RANK_7, SUIT_DIAMONDS)],
        );

        // SB raises to 6 (raise of 4 on top of BB's 2)
        let s1 = state.apply_action(Action::Raise(6));
        assert_eq!(s1.last_raise_size, 4); // 6 - 2 = 4

        // BB 3-bets: min reraise is 6 + 4 = 10
        let actions = s1.legal_actions();
        let raises: Vec<u32> = actions
            .iter()
            .filter_map(|a| match a {
                Action::Raise(amt) => Some(*amt),
                _ => None,
            })
            .collect();
        assert!(raises.contains(&10), "Min 3-bet should be 10, got {:?}", raises);
    }

    // --- Action abstraction tests ---

    #[test]
    fn test_default_abstraction() {
        let abs = ActionAbstraction::default();
        assert_eq!(abs.bet_fractions, vec![0.25, 0.5, 1.0, 2.0]);
    }

    #[test]
    fn test_abstract_actions_preflop() {
        let abs = ActionAbstraction::default();
        let mut state = GameState::new(200);
        state.deal_hole_cards(
            [card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_SPADES)],
            [card(RANK_2, SUIT_HEARTS), card(RANK_7, SUIT_DIAMONDS)],
        );

        let actions = abs.abstract_actions(&state);
        // Should have: Fold, Call, various raises, AllIn
        assert!(actions.contains(&Action::Fold));
        assert!(actions.contains(&Action::Call));
        assert!(actions.contains(&Action::AllIn));
        // Should have some raises
        let num_raises = actions.iter().filter(|a| matches!(a, Action::Raise(_))).count();
        assert!(num_raises > 0, "Should have at least one raise");
    }

    #[test]
    fn test_action_translation() {
        let abs = ActionAbstraction::new(vec![0.5, 1.0]);
        let mut state = GameState::new(200);
        state.deal_hole_cards(
            [card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_SPADES)],
            [card(RANK_2, SUIT_HEARTS), card(RANK_7, SUIT_DIAMONDS)],
        );

        // Exact match should return same
        let fold_translated = abs.translate_action(&state, Action::Fold);
        assert_eq!(fold_translated, Action::Fold);

        // Off-tree raise should map to nearest abstract raise
        let off_tree = abs.translate_action(&state, Action::Raise(7));
        match off_tree {
            Action::Raise(_) | Action::AllIn => {} // should map to some abstract size
            _ => panic!("Off-tree raise should map to a raise or all-in"),
        }
    }

    #[test]
    fn test_zero_sum() {
        let mut state = GameState::new(200);
        state.deal_hole_cards(
            [card(RANK_A, SUIT_SPADES), card(RANK_K, SUIT_SPADES)],
            [card(RANK_2, SUIT_HEARTS), card(RANK_7, SUIT_DIAMONDS)],
        );

        // Any terminal state should be zero-sum
        let fold_state = state.apply_action(Action::Fold);
        let u0 = fold_state.terminal_utility(0);
        let u1 = fold_state.terminal_utility(1);
        assert!((u0 + u1).abs() < 0.001, "Game must be zero-sum: {} + {} = {}", u0, u1, u0 + u1);
    }
}
