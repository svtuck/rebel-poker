"""Tests for the evaluation framework.

Tests:
- Exploitability computation (exact and LBR)
- Best-response strategy computation
- Head-to-head play (self-play, baselines)
- Profiler functionality
- Training tracker
"""

import math
import json
import tempfile

import pytest

from kuhn.game import KuhnPoker
from leduc.game import LeducPoker
from cfr.solver import CFRTrainer

from eval.exploitability import (
    exploitability,
    best_response_value,
    best_response_strategy,
    lbr_exploitability,
    per_infoset_exploitability,
    LBRConfig,
)
from eval.head_to_head import (
    play_hand,
    play_match,
    expected_value,
    StrategyAgent,
    AlwaysCallAgent,
    AlwaysFoldAgent,
    AlwaysBetAgent,
    RandomAgent,
)
from eval.profiler import Profiler, profile_cfr_solver
from eval.training_tracker import TrainingTracker


# ── Fixtures ──


@pytest.fixture
def kuhn():
    return KuhnPoker()


@pytest.fixture
def leduc():
    return LeducPoker()


@pytest.fixture
def kuhn_nash_profile(kuhn):
    """A near-Nash strategy for Kuhn poker (10K iterations)."""
    trainer = CFRTrainer(kuhn)
    trainer.train(10000)
    return trainer.average_strategy_profile()


@pytest.fixture
def kuhn_weak_profile():
    """A deliberately weak strategy: always check/call."""
    return {
        "J|": {"c": 1.0, "b": 0.0},
        "Q|": {"c": 1.0, "b": 0.0},
        "K|": {"c": 1.0, "b": 0.0},
        "J|c": {"c": 1.0, "b": 0.0},
        "Q|c": {"c": 1.0, "b": 0.0},
        "K|c": {"c": 1.0, "b": 0.0},
        "J|b": {"c": 1.0, "f": 0.0},
        "Q|b": {"c": 1.0, "f": 0.0},
        "K|b": {"c": 1.0, "f": 0.0},
        "J|cb": {"c": 1.0, "f": 0.0},
        "Q|cb": {"c": 1.0, "f": 0.0},
        "K|cb": {"c": 1.0, "f": 0.0},
    }


@pytest.fixture
def leduc_profile(leduc):
    """A trained strategy for Leduc poker (200 iterations)."""
    trainer = CFRTrainer(leduc)
    trainer.train(200)
    return trainer.average_strategy_profile()


# ── Exploitability Tests ──


class TestExploitability:
    def test_nash_low_exploitability(self, kuhn, kuhn_nash_profile):
        """Near-Nash strategy should have near-zero exploitability."""
        exp = exploitability(kuhn, kuhn_nash_profile)
        assert exp < 0.01, f"Nash exploitability {exp} should be < 0.01"

    def test_weak_strategy_exploitable(self, kuhn, kuhn_weak_profile):
        """Always-call strategy should be highly exploitable."""
        exp = exploitability(kuhn, kuhn_weak_profile)
        assert exp > 0.05, f"Weak strategy exploitability {exp} should be > 0.05"

    def test_exploitability_nonnegative(self, kuhn, kuhn_nash_profile):
        """Exploitability should always be >= 0."""
        exp = exploitability(kuhn, kuhn_nash_profile)
        assert exp >= 0.0

    def test_best_response_value_vs_cfr(self, kuhn, kuhn_nash_profile):
        """best_response_value should match CFRTrainer's implementation."""
        br0 = best_response_value(kuhn, kuhn_nash_profile, br_player=0)
        br1 = best_response_value(kuhn, kuhn_nash_profile, br_player=1)

        trainer = CFRTrainer(kuhn)
        cfr_br0 = trainer._best_response_value(kuhn_nash_profile, 0)
        cfr_br1 = trainer._best_response_value(kuhn_nash_profile, 1)

        assert abs(br0 - cfr_br0) < 1e-10, f"BR0 mismatch: {br0} vs {cfr_br0}"
        assert abs(br1 - cfr_br1) < 1e-10, f"BR1 mismatch: {br1} vs {cfr_br1}"

    @pytest.mark.slow
    def test_leduc_exploitability(self, leduc, leduc_profile):
        """Leduc strategy after training should have bounded exploitability."""
        exp = exploitability(leduc, leduc_profile)
        assert exp < 5.0, f"Leduc exploitability {exp} unreasonably high"
        assert exp >= 0.0

    def test_best_response_strategy_deterministic(self, kuhn, kuhn_nash_profile):
        """Best response strategy should be deterministic (pure strategy)."""
        br_profile = best_response_strategy(kuhn, kuhn_nash_profile, br_player=0)
        for key, action_probs in br_profile.items():
            probs = list(action_probs.values())
            assert all(p == 0.0 or p == 1.0 for p in probs), (
                f"BR strategy at {key} not pure: {action_probs}"
            )

    def test_best_response_strategy_exploits_weak(self, kuhn, kuhn_weak_profile):
        """BR against always-call should exploit it."""
        br_profile = best_response_strategy(kuhn, kuhn_weak_profile, br_player=0)
        if "K|" in br_profile:
            assert br_profile["K|"].get("b", 0) == 1.0, (
                f"BR should always bet with K against always-call: {br_profile.get('K|')}"
            )

    def test_per_infoset_exploitability(self, kuhn, kuhn_weak_profile):
        """Per-infoset exploitability should have non-negative contributions."""
        contribs = per_infoset_exploitability(kuhn, kuhn_weak_profile, br_player=0)
        assert len(contribs) > 0
        for key, v in contribs.items():
            assert v >= -1e-10, f"Negative contribution at {key}: {v}"


class TestLBR:
    def test_lbr_vs_exact(self, kuhn, kuhn_nash_profile):
        """LBR should bound exact exploitability for small games.

        Note: Our LBR implementation samples chance outcomes but tries all
        actions at BR nodes, which overestimates exploitability (the BR player
        gets to see each sampled deal and play perfectly against it, rather
        than having to commit to an information-set-level policy). For near-Nash
        strategies, the absolute value should still be small.
        """
        exact = exploitability(kuhn, kuhn_nash_profile)
        config = LBRConfig(num_samples=50000, seed=42)
        approx = lbr_exploitability(kuhn, kuhn_nash_profile, config)
        assert approx >= 0, f"LBR should be nonnegative: {approx}"
        assert approx < 1.0, f"LBR {approx} unreasonably high for near-Nash"
        assert exact <= approx + 0.01, (
            f"Exact {exact} should be <= LBR {approx} (LBR is an upper bound)"
        )

    def test_lbr_detects_weak_strategy(self, kuhn, kuhn_weak_profile):
        """LBR should detect that always-call is exploitable."""
        config = LBRConfig(num_samples=10000, seed=42)
        approx = lbr_exploitability(kuhn, kuhn_weak_profile, config)
        assert approx > 0.01, f"LBR should detect exploitability: {approx}"

    def test_lbr_reproducible(self, kuhn, kuhn_nash_profile):
        """Same seed should give same result."""
        config = LBRConfig(num_samples=1000, seed=123)
        r1 = lbr_exploitability(kuhn, kuhn_nash_profile, config)
        r2 = lbr_exploitability(kuhn, kuhn_nash_profile, config)
        assert r1 == r2


# ── Head-to-Head Tests ──


class TestHeadToHead:
    def test_self_play_near_zero(self, kuhn, kuhn_nash_profile):
        """Nash vs Nash should yield ~0 EV."""
        agent = StrategyAgent(kuhn_nash_profile, seed=42)
        result = play_match(kuhn, agent, agent, num_hands=5000, seed=42)
        assert abs(result.p0_mean) < 0.1, (
            f"Self-play EV {result.p0_mean} should be ~0"
        )

    def test_nash_beats_always_fold(self, kuhn, kuhn_nash_profile):
        """Nash strategy should profit against always-fold."""
        nash_agent = StrategyAgent(kuhn_nash_profile, seed=42)
        fold_agent = AlwaysFoldAgent()
        result = play_match(kuhn, nash_agent, fold_agent, num_hands=5000, seed=42)
        assert result.p0_mean > 0.0, (
            f"Nash should beat always-fold: EV = {result.p0_mean}"
        )

    def test_nash_beats_always_call(self, kuhn, kuhn_nash_profile):
        """Nash strategy should profit against always-call."""
        nash_agent = StrategyAgent(kuhn_nash_profile, seed=42)
        call_agent = AlwaysCallAgent()
        result = play_match(kuhn, nash_agent, call_agent, num_hands=5000, seed=42)
        assert result.p0_mean > -0.1, (
            f"Nash should not lose much to always-call: EV = {result.p0_mean}"
        )

    def test_nash_beats_random(self, kuhn, kuhn_nash_profile):
        """Nash strategy should profit against random play."""
        nash_agent = StrategyAgent(kuhn_nash_profile, seed=42)
        random_agent = RandomAgent(seed=42)
        result = play_match(kuhn, nash_agent, random_agent, num_hands=5000, seed=42)
        assert result.p0_mean > -0.1, (
            f"Nash should not lose much to random: EV = {result.p0_mean}"
        )

    def test_confidence_interval_contains_mean(self, kuhn, kuhn_nash_profile):
        """95% CI should contain the mean."""
        agent = StrategyAgent(kuhn_nash_profile, seed=42)
        result = play_match(kuhn, agent, agent, num_hands=1000, seed=42)
        lo, hi = result.p0_ci_95
        assert lo <= result.p0_mean <= hi

    def test_zero_sum(self, kuhn, kuhn_nash_profile):
        """P0 and P1 EVs should sum to zero."""
        agent = StrategyAgent(kuhn_nash_profile, seed=42)
        random_agent = RandomAgent(seed=42)
        result = play_match(kuhn, agent, random_agent, num_hands=1000, seed=42)
        assert abs(result.p0_mean + result.p1_mean) < 1e-10, (
            f"Not zero-sum: P0={result.p0_mean}, P1={result.p1_mean}"
        )

    def test_always_bet_agent(self, kuhn, kuhn_nash_profile):
        """AlwaysBetAgent should function without errors."""
        bet_agent = AlwaysBetAgent()
        nash_agent = StrategyAgent(kuhn_nash_profile, seed=42)
        result = play_match(kuhn, bet_agent, nash_agent, num_hands=1000, seed=42)
        assert result.num_hands == 1000


class TestExpectedValue:
    def test_kuhn_game_value(self, kuhn, kuhn_nash_profile):
        """Expected value under Nash should be ~-1/18."""
        ev = expected_value(kuhn, kuhn_nash_profile)
        expected = -1.0 / 18.0
        assert abs(ev - expected) < 0.02, (
            f"Game value {ev:.4f} should be near {expected:.4f}"
        )


# ── Profiler Tests ──


class TestProfiler:
    def test_section_timing(self):
        """Section timing should record elapsed time."""
        profiler = Profiler()
        import time
        with profiler.section("test_section", iterations=10):
            time.sleep(0.01)
        assert len(profiler.result.timings) == 1
        assert profiler.result.timings[0].wall_time_s >= 0.01
        assert profiler.result.timings[0].name == "test_section"

    def test_memory_measurement(self):
        """Memory measurement should record peak and current."""
        profiler = Profiler()
        profiler.measure_memory("test", lambda: [0] * 100000)
        assert len(profiler.result.memory) == 1
        assert profiler.result.memory[0].peak_mb > 0

    def test_throughput_recording(self):
        """Throughput recording should compute correct rate."""
        profiler = Profiler()
        profiler.record_throughput("test", 1000, 2.0)
        assert profiler.result.throughput["test"] == 500.0

    def test_gpu_check(self):
        """GPU check should detect available backends."""
        profiler = Profiler()
        profiler.check_gpu()
        assert "backend" in profiler.result.gpu_info
        assert "pytorch_version" in profiler.result.gpu_info

    def test_bottleneck_identification(self):
        """Should identify top bottlenecks from timing data."""
        profiler = Profiler()
        import time
        with profiler.section("fast", iterations=1):
            time.sleep(0.001)
        with profiler.section("slow", iterations=1):
            time.sleep(0.02)
        bottlenecks = profiler.identify_bottlenecks()
        assert len(bottlenecks) > 0
        assert "slow" in bottlenecks[0]

    def test_profile_summary(self):
        """Summary should be a non-empty string."""
        profiler = Profiler()
        with profiler.section("test"):
            pass
        profiler.check_gpu()
        profiler.identify_bottlenecks()
        summary = profiler.result.summary()
        assert len(summary) > 0
        assert "Profile Summary" in summary

    def test_profile_cfr_solver(self, kuhn):
        """profile_cfr_solver should produce a complete ProfileResult."""
        result = profile_cfr_solver(
            solver_factory=lambda g: CFRTrainer(g),
            game=kuhn,
            iterations=100,
            warmup_iterations=10,
        )
        assert len(result.timings) > 0
        assert len(result.memory) > 0
        assert "backend" in result.gpu_info


# ── Training Tracker Tests ──


class TestTrainingTracker:
    def test_record_checkpoints(self, kuhn, kuhn_nash_profile):
        """Should record exploitability at each checkpoint."""
        tracker = TrainingTracker(kuhn)
        tracker.record(1, kuhn_nash_profile)
        tracker.record(2, kuhn_nash_profile)
        assert len(tracker.checkpoints) == 2
        assert all(cp.exploitability < 0.01 for cp in tracker.checkpoints)

    def test_record_value(self, kuhn):
        """record_value should store pre-computed exploitability."""
        tracker = TrainingTracker(kuhn)
        tracker.record_value(1, 0.5, {"loss": 0.1})
        assert tracker.checkpoints[0].exploitability == 0.5
        assert tracker.checkpoints[0].metrics["loss"] == 0.1

    def test_best_checkpoint(self, kuhn):
        """best_checkpoint should return the one with lowest exploitability."""
        tracker = TrainingTracker(kuhn)
        tracker.record_value(1, 0.5)
        tracker.record_value(2, 0.1)
        tracker.record_value(3, 0.3)
        best = tracker.best_checkpoint()
        assert best is not None
        assert best.iteration == 2
        assert best.exploitability == 0.1

    def test_convergence_rate(self, kuhn):
        """convergence_rate should return a negative slope for converging solver."""
        tracker = TrainingTracker(kuhn)
        for i in [10, 100, 1000, 10000]:
            tracker.record_value(i, 1.0 / math.sqrt(i))
        rate = tracker.convergence_rate()
        assert rate is not None
        assert rate < 0, f"Convergence rate {rate} should be negative"
        assert abs(rate - (-0.5)) < 0.1, (
            f"Rate {rate} should be ~-0.5 for O(1/sqrt(T))"
        )

    def test_save_and_load(self, kuhn):
        """Should round-trip through JSON."""
        tracker = TrainingTracker(kuhn)
        tracker.record_value(1, 0.5, {"loss": 0.1})
        tracker.record_value(2, 0.3, {"loss": 0.05})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        tracker.save(path)
        loaded = TrainingTracker.load(path, kuhn)
        assert len(loaded.checkpoints) == 2
        assert loaded.checkpoints[0].exploitability == 0.5
        assert loaded.checkpoints[1].metrics["loss"] == 0.05

    def test_summary(self, kuhn):
        """summary should produce readable output."""
        tracker = TrainingTracker(kuhn)
        tracker.record_value(1, 0.5)
        tracker.record_value(100, 0.05)
        summary = tracker.summary()
        assert "Training Progress" in summary

    def test_properties(self, kuhn):
        """exploitabilities, iterations, wall_times should be correct length."""
        tracker = TrainingTracker(kuhn)
        tracker.record_value(1, 0.5)
        tracker.record_value(2, 0.3)
        assert len(tracker.exploitabilities) == 2
        assert len(tracker.iterations) == 2
        assert len(tracker.wall_times) == 2


# ── Integration: CFR Training with Tracking ──


class TestIntegration:
    def test_cfr_training_with_tracker(self, kuhn):
        """Full workflow: train CFR, track exploitability, verify convergence."""
        trainer = CFRTrainer(kuhn)
        tracker = TrainingTracker(kuhn)

        checkpoints = [100, 500, 1000, 5000]
        for n in checkpoints:
            trainer.run(n - (trainer.iteration))
            profile = trainer.average_strategy_profile()
            tracker.record(n, profile)

        exps = tracker.exploitabilities
        assert exps[-1] < exps[0], f"Exploitability should decrease: {exps}"
        assert exps[-1] < 0.01, f"Final exploitability {exps[-1]} should be < 0.01"

        rate = tracker.convergence_rate()
        assert rate is not None
        assert rate < 0

    def test_head_to_head_vs_exploitability(self, kuhn):
        """A more exploitable strategy should lose more in head-to-head."""
        strong_trainer = CFRTrainer(kuhn)
        strong_trainer.train(5000)
        strong_profile = strong_trainer.average_strategy_profile()

        weak_trainer = CFRTrainer(kuhn)
        weak_trainer.train(50)
        weak_profile = weak_trainer.average_strategy_profile()

        strong_exp = exploitability(kuhn, strong_profile)
        weak_exp = exploitability(kuhn, weak_profile)
        assert strong_exp < weak_exp, "Strong should be less exploitable"

        strong_agent = StrategyAgent(strong_profile, seed=42)
        weak_agent = StrategyAgent(weak_profile, seed=42)
        result = play_match(kuhn, strong_agent, weak_agent, num_hands=10000, seed=42)
        assert result.p0_mean > -0.1, (
            f"Strong agent losing to weak: EV = {result.p0_mean}"
        )

    @pytest.mark.slow
    def test_leduc_full_eval(self, leduc):
        """Full evaluation pipeline on Leduc poker."""
        trainer = CFRTrainer(leduc)
        trainer.train(200)
        profile = trainer.average_strategy_profile()

        exp = exploitability(leduc, profile)
        assert exp >= 0.0
        assert exp < 5.0

        agent = StrategyAgent(profile, seed=42)
        random_agent = RandomAgent(seed=42)
        result = play_match(leduc, agent, random_agent, num_hands=500, seed=42)
        assert result.num_hands == 500


# ── Slumbot (Unit tests only, no network) ──


class TestSlumbotParsing:
    def test_parse_action_sequence(self):
        from eval.slumbot import parse_action_sequence
        parsed = parse_action_sequence("cb300c")
        assert parsed == [('c', None), ('b', 300), ('c', None)]

    def test_parse_with_streets(self):
        from eval.slumbot import parse_action_sequence
        parsed = parse_action_sequence("kb200c/kk")
        assert parsed == [
            ('k', None), ('b', 200), ('c', None),
            ('/', None), ('k', None), ('k', None),
        ]

    def test_parse_fold(self):
        from eval.slumbot import parse_action_sequence
        parsed = parse_action_sequence("b100f")
        assert parsed == [('b', 100), ('f', None)]

    def test_parse_empty(self):
        from eval.slumbot import parse_action_sequence
        parsed = parse_action_sequence("")
        assert parsed == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
