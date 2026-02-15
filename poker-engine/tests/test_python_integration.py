"""Integration tests for poker_engine Python bindings."""

import time
import poker_engine as pe


class TestCardFunctions:
    def test_card_roundtrip(self):
        for c in range(52):
            r = pe.py_rank(c)
            s = pe.py_suit(c)
            assert pe.py_card(r, s) == c

    def test_parse_all_cards(self):
        ranks = "23456789TJQKA"
        suits = "cdhs"
        for i, r in enumerate(ranks):
            for j, s in enumerate(suits):
                card_str = f"{r}{s}"
                c = pe.py_parse_card(card_str)
                assert c is not None, f"Failed to parse {card_str}"
                assert pe.py_rank(c) == i, f"Wrong rank for {card_str}"
                assert pe.py_suit(c) == j, f"Wrong suit for {card_str}"

    def test_card_to_string_roundtrip(self):
        for c in range(52):
            s = pe.py_card_to_string(c)
            assert pe.py_parse_card(s) == c, f"Roundtrip failed for card {c} -> {s}"


class TestHandEvaluation:
    def test_straight_flushes(self):
        """All 10 straight flush ranks: 1 (royal) to 10 (wheel)."""
        # Royal flush: A-K-Q-J-T of spades
        rf = [pe.py_parse_card(c) for c in ["As", "Ks", "Qs", "Js", "Ts"]]
        assert pe.py_eval5(rf) == 1

        # Steel wheel: 5-4-3-2-A of hearts
        wheel = [pe.py_parse_card(c) for c in ["5h", "4h", "3h", "2h", "Ah"]]
        assert pe.py_eval5(wheel) == 10

    def test_four_of_a_kind(self):
        """Four aces with K kicker is best quad (rank 11)."""
        hand = [pe.py_parse_card(c) for c in ["As", "Ah", "Ad", "Ac", "Ks"]]
        rank = pe.py_eval5(hand)
        assert rank == 11
        assert pe.py_hand_category(rank) == "Four of a Kind"

    def test_full_house(self):
        """Aces full of kings is best full house (rank 167)."""
        hand = [pe.py_parse_card(c) for c in ["As", "Ah", "Ad", "Ks", "Kh"]]
        rank = pe.py_eval5(hand)
        assert rank == 167
        assert pe.py_hand_category(rank) == "Full House"

    def test_flush(self):
        """A-K-Q-J-9 of hearts is best non-straight flush (rank 323)."""
        hand = [pe.py_parse_card(c) for c in ["Ah", "Kh", "Qh", "Jh", "9h"]]
        rank = pe.py_eval5(hand)
        assert rank == 323
        assert pe.py_hand_category(rank) == "Flush"

    def test_straight(self):
        """Ace-high straight offsuit (rank 1600)."""
        hand = [pe.py_parse_card(c) for c in ["As", "Kh", "Qd", "Jc", "Ts"]]
        rank = pe.py_eval5(hand)
        assert rank == 1600
        assert pe.py_hand_category(rank) == "Straight"

    def test_three_of_a_kind(self):
        hand = [pe.py_parse_card(c) for c in ["As", "Ah", "Ad", "Ks", "Qh"]]
        rank = pe.py_eval5(hand)
        assert pe.py_hand_category(rank) == "Three of a Kind"
        assert 1610 <= rank <= 2467

    def test_two_pair(self):
        hand = [pe.py_parse_card(c) for c in ["As", "Ah", "Ks", "Kh", "Qd"]]
        rank = pe.py_eval5(hand)
        assert pe.py_hand_category(rank) == "Two Pair"
        assert 2468 <= rank <= 3325

    def test_one_pair(self):
        hand = [pe.py_parse_card(c) for c in ["As", "Ah", "Ks", "Qh", "Jd"]]
        rank = pe.py_eval5(hand)
        assert pe.py_hand_category(rank) == "One Pair"
        assert 3326 <= rank <= 6185

    def test_high_card(self):
        hand = [pe.py_parse_card(c) for c in ["As", "Kh", "Qd", "Jc", "9s"]]
        rank = pe.py_eval5(hand)
        assert pe.py_hand_category(rank) == "High Card"
        assert 6186 <= rank <= 7462

    def test_worst_hand(self):
        hand = [pe.py_parse_card(c) for c in ["7s", "5h", "4d", "3c", "2s"]]
        assert pe.py_eval5(hand) == 7462

    def test_hand_ordering(self):
        """Better hands have lower rank numbers."""
        straight_flush = pe.py_eval5([pe.py_parse_card(c) for c in ["As", "Ks", "Qs", "Js", "Ts"]])
        quads = pe.py_eval5([pe.py_parse_card(c) for c in ["As", "Ah", "Ad", "Ac", "Ks"]])
        full_house = pe.py_eval5([pe.py_parse_card(c) for c in ["As", "Ah", "Ad", "Ks", "Kh"]])
        flush = pe.py_eval5([pe.py_parse_card(c) for c in ["Ah", "Kh", "Qh", "Jh", "9h"]])
        straight = pe.py_eval5([pe.py_parse_card(c) for c in ["As", "Kh", "Qd", "Jc", "Ts"]])
        trips = pe.py_eval5([pe.py_parse_card(c) for c in ["As", "Ah", "Ad", "Ks", "Qh"]])
        two_pair = pe.py_eval5([pe.py_parse_card(c) for c in ["As", "Ah", "Ks", "Kh", "Qd"]])
        pair = pe.py_eval5([pe.py_parse_card(c) for c in ["As", "Ah", "Ks", "Qh", "Jd"]])
        high_card = pe.py_eval5([pe.py_parse_card(c) for c in ["As", "Kh", "Qd", "Jc", "9s"]])

        assert straight_flush < quads < full_house < flush < straight < trips < two_pair < pair < high_card

    def test_eval7_best_5_of_7(self):
        """7-card eval should find the best 5-card hand."""
        cards = [pe.py_parse_card(c) for c in ["As", "Ks", "Qs", "Js", "Ts", "2h", "3d"]]
        assert pe.py_eval7(cards) == 1  # Royal flush

    def test_eval5_batch(self):
        rf = [pe.py_parse_card(c) for c in ["As", "Ks", "Qs", "Js", "Ts"]]
        worst = [pe.py_parse_card(c) for c in ["7s", "5h", "4d", "3c", "2s"]]
        results = pe.py_eval5_batch([rf, worst])
        assert results == [1, 7462]

    def test_eval7_batch(self):
        h1 = [pe.py_parse_card(c) for c in ["As", "Ks", "Qs", "Js", "Ts", "2h", "3d"]]
        h2 = [pe.py_parse_card(c) for c in ["7s", "5h", "4d", "3c", "2s", "8d", "9c"]]
        results = pe.py_eval7_batch([h1, h2])
        assert results[0] == 1

    def test_compare_hands(self):
        h1 = [pe.py_parse_card(c) for c in ["As", "Ks", "Qs", "Js", "Ts", "2h", "3d"]]
        h2 = [pe.py_parse_card(c) for c in ["7s", "5h", "4d", "3c", "2s", "8d", "9c"]]
        assert pe.py_compare_hands(h1, h2) == -1  # h1 wins


class TestGameState:
    def test_initial_state(self):
        gs = pe.PyGameState(200)
        assert gs.pot == 3
        assert gs.stacks == (199, 198)
        assert gs.active_player == 0
        assert gs.is_terminal == False
        assert gs.street == "preflop"

    def test_fold_preflop(self):
        gs = pe.PyGameState(200)
        gs.deal_hole_cards(
            pe.py_parse_card("As"), pe.py_parse_card("Ks"),
            pe.py_parse_card("2h"), pe.py_parse_card("7d"),
        )
        folded = gs.apply_action(pe.PyAction.fold())
        assert folded.is_terminal
        assert folded.terminal_utility(0) == -1.0  # SB loses 1
        assert folded.terminal_utility(1) == 1.0   # BB wins 1

    def test_legal_actions_preflop(self):
        gs = pe.PyGameState(200)
        gs.deal_hole_cards(
            pe.py_parse_card("As"), pe.py_parse_card("Ks"),
            pe.py_parse_card("2h"), pe.py_parse_card("7d"),
        )
        actions = gs.legal_actions()
        action_types = [a.action_type for a in actions]
        assert "fold" in action_types
        assert "call" in action_types or "check" in action_types
        assert "all_in" in action_types

    def test_infoset_key_privacy(self):
        """Each player's infoset key should only contain their own hole cards."""
        gs = pe.PyGameState(200)
        gs.deal_hole_cards(
            pe.py_parse_card("As"), pe.py_parse_card("Ks"),
            pe.py_parse_card("2h"), pe.py_parse_card("7d"),
        )
        key0 = gs.infoset_key(0)
        key1 = gs.infoset_key(1)
        # P0 sees As,Ks but not 2h,7d
        assert "As" in key0 or "Ks" in key0
        assert "2h" not in key0
        assert "7d" not in key0
        # P1 sees 2h,7d but not As,Ks
        assert "2h" in key1 or "7d" in key1
        assert "As" not in key1
        assert "Ks" not in key1

    def test_action_types(self):
        fold = pe.PyAction.fold()
        assert fold.action_type == "fold"
        assert fold.amount is None

        check = pe.PyAction.check()
        assert check.action_type == "check"

        call = pe.PyAction.call()
        assert call.action_type == "call"

        raise_to = pe.PyAction.raise_to(10)
        assert raise_to.action_type == "raise"
        assert raise_to.amount == 10

        all_in = pe.PyAction.all_in()
        assert all_in.action_type == "all_in"


class TestBatchOperations:
    def test_batch_apply_action(self):
        states = []
        for _ in range(100):
            gs = pe.PyGameState(200)
            gs.deal_hole_cards(
                pe.py_parse_card("As"), pe.py_parse_card("Ks"),
                pe.py_parse_card("2h"), pe.py_parse_card("7d"),
            )
            states.append(gs)

        folded = pe.py_batch_apply_action(states, pe.PyAction.fold())
        assert len(folded) == 100
        assert all(s.is_terminal for s in folded)

    def test_batch_terminal_utility(self):
        states = []
        for _ in range(50):
            gs = pe.PyGameState(200)
            gs.deal_hole_cards(
                pe.py_parse_card("As"), pe.py_parse_card("Ks"),
                pe.py_parse_card("2h"), pe.py_parse_card("7d"),
            )
            states.append(gs.apply_action(pe.PyAction.fold()))

        utils = pe.py_batch_terminal_utility(states, 0)
        assert len(utils) == 50
        assert all(u == -1.0 for u in utils)

    def test_batch_infoset_keys(self):
        states = []
        for _ in range(20):
            gs = pe.PyGameState(200)
            gs.deal_hole_cards(
                pe.py_parse_card("As"), pe.py_parse_card("Ks"),
                pe.py_parse_card("2h"), pe.py_parse_card("7d"),
            )
            states.append(gs)

        keys = pe.py_batch_infoset_keys(states, 0)
        assert len(keys) == 20
        # All should be the same since same cards + same history
        assert len(set(keys)) == 1


class TestConstants:
    def test_constants(self):
        assert pe.NUM_CARDS == 52
        assert pe.NUM_RANKS == 13
        assert pe.NUM_SUITS == 4
        assert pe.SMALL_BLIND == 1
        assert pe.BIG_BLIND == 2


class TestBenchmarks:
    """Rough performance sanity checks (not precise benchmarks)."""

    def test_eval5_throughput(self):
        """Sanity check: eval5 should be fast."""
        rf = [pe.py_parse_card(c) for c in ["As", "Ks", "Qs", "Js", "Ts"]]
        n = 100_000
        start = time.time()
        for _ in range(n):
            pe.py_eval5(rf)
        elapsed = time.time() - start
        rate = n / elapsed
        print(f"\neval5 throughput: {rate:,.0f} evals/sec (from Python)")
        # Should be at least 500K/sec even with Python overhead
        assert rate > 100_000, f"eval5 too slow: {rate:.0f}/sec"

    def test_eval7_throughput(self):
        """Sanity check: eval7 should be fast."""
        cards = [pe.py_parse_card(c) for c in ["As", "Ks", "Qs", "Js", "Ts", "2h", "3d"]]
        n = 50_000
        start = time.time()
        for _ in range(n):
            pe.py_eval7(cards)
        elapsed = time.time() - start
        rate = n / elapsed
        print(f"\neval7 throughput: {rate:,.0f} evals/sec (from Python)")
        assert rate > 50_000, f"eval7 too slow: {rate:.0f}/sec"

    def test_eval7_batch_throughput(self):
        """Batch eval should amortize Python overhead."""
        cards = [pe.py_parse_card(c) for c in ["As", "Ks", "Qs", "Js", "Ts", "2h", "3d"]]
        batch_size = 10_000
        batch = [cards] * batch_size
        n_batches = 10
        start = time.time()
        for _ in range(n_batches):
            pe.py_eval7_batch(batch)
        elapsed = time.time() - start
        rate = (batch_size * n_batches) / elapsed
        print(f"\neval7_batch throughput: {rate:,.0f} evals/sec")
        assert rate > 500_000, f"eval7_batch too slow: {rate:.0f}/sec"

    def test_state_transition_throughput(self):
        """State transitions should be fast."""
        gs = pe.PyGameState(200)
        gs.deal_hole_cards(
            pe.py_parse_card("As"), pe.py_parse_card("Ks"),
            pe.py_parse_card("2h"), pe.py_parse_card("7d"),
        )
        action = pe.PyAction.fold()
        n = 100_000
        start = time.time()
        for _ in range(n):
            gs.apply_action(action)
        elapsed = time.time() - start
        rate = n / elapsed
        print(f"\nState transition throughput: {rate:,.0f} transitions/sec (from Python)")
        assert rate > 100_000, f"State transitions too slow: {rate:.0f}/sec"
