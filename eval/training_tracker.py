"""Track exploitability and metrics over training iterations.

Provides a TrainingTracker that records exploitability, loss, and
other metrics at each training checkpoint, with serialization
support for later analysis.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from game_interface import Game

StrategyProfile = Dict[str, Dict[str, float]]


@dataclass
class Checkpoint:
    """A single training checkpoint."""
    iteration: int
    wall_time_s: float
    exploitability: float
    metrics: Dict[str, float] = field(default_factory=dict)


class TrainingTracker:
    """Track and record metrics over training iterations.

    Usage:
        tracker = TrainingTracker(game)
        for epoch in range(num_epochs):
            # ... train ...
            profile = solver.average_strategy_profile()
            tracker.record(epoch, profile, extra_metrics={"loss": loss_val})
        tracker.save("training_log.json")
        print(tracker.summary())
    """

    def __init__(
        self,
        game: Game,
        compute_exploitability_fn=None,
    ) -> None:
        self.game = game
        self.checkpoints: List[Checkpoint] = []
        self._start_time = time.perf_counter()

        if compute_exploitability_fn is not None:
            self._compute_exp = compute_exploitability_fn
        else:
            from eval.exploitability import exploitability
            self._compute_exp = lambda g, p: exploitability(g, p)

    def record(
        self,
        iteration: int,
        profile: StrategyProfile,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> Checkpoint:
        """Record a training checkpoint. Computes exploitability and records wall time."""
        wall_time = time.perf_counter() - self._start_time
        exp = self._compute_exp(self.game, profile)

        metrics = extra_metrics or {}
        cp = Checkpoint(
            iteration=iteration,
            wall_time_s=wall_time,
            exploitability=exp,
            metrics=metrics,
        )
        self.checkpoints.append(cp)
        return cp

    def record_value(
        self,
        iteration: int,
        exploitability: float,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> Checkpoint:
        """Record a checkpoint with a pre-computed exploitability value."""
        wall_time = time.perf_counter() - self._start_time
        metrics = extra_metrics or {}
        cp = Checkpoint(
            iteration=iteration,
            wall_time_s=wall_time,
            exploitability=exploitability,
            metrics=metrics,
        )
        self.checkpoints.append(cp)
        return cp

    @property
    def exploitabilities(self) -> List[float]:
        return [cp.exploitability for cp in self.checkpoints]

    @property
    def iterations(self) -> List[int]:
        return [cp.iteration for cp in self.checkpoints]

    @property
    def wall_times(self) -> List[float]:
        return [cp.wall_time_s for cp in self.checkpoints]

    def best_checkpoint(self) -> Optional[Checkpoint]:
        """Return the checkpoint with lowest exploitability."""
        if not self.checkpoints:
            return None
        return min(self.checkpoints, key=lambda cp: cp.exploitability)

    def convergence_rate(self) -> Optional[float]:
        """Estimate convergence rate (exploitability reduction per iteration).

        Returns the slope of log(exploitability) vs log(iteration),
        or None if insufficient data.
        """
        valid = [(cp.iteration, cp.exploitability)
                 for cp in self.checkpoints
                 if cp.iteration > 0 and cp.exploitability > 0]
        if len(valid) < 2:
            return None

        log_iters = [math.log(i) for i, _ in valid]
        log_exps = [math.log(e) for _, e in valid]

        n = len(valid)
        sum_x = sum(log_iters)
        sum_y = sum(log_exps)
        sum_xy = sum(x * y for x, y in zip(log_iters, log_exps))
        sum_x2 = sum(x * x for x in log_iters)

        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-12:
            return None
        slope = (n * sum_xy - sum_x * sum_y) / denom
        return slope

    def summary(self) -> str:
        """Generate a human-readable summary of training progress."""
        if not self.checkpoints:
            return "No checkpoints recorded."

        lines = [
            "=" * 60,
            "  Training Progress",
            "=" * 60,
            f"  {'Iter':>8} {'Wall Time':>10} {'Exploitability':>15}",
            f"  {'-'*8} {'-'*10} {'-'*15}",
        ]

        for cp in self.checkpoints:
            lines.append(
                f"  {cp.iteration:>8} {cp.wall_time_s:>9.1f}s {cp.exploitability:>15.6f}"
            )
            if cp.metrics:
                for k, v in cp.metrics.items():
                    lines.append(f"  {'':>8} {k}: {v:.6f}")

        best = self.best_checkpoint()
        if best:
            lines.append(f"\n  Best: iter {best.iteration}, "
                         f"exploitability = {best.exploitability:.6f}")

        rate = self.convergence_rate()
        if rate is not None:
            lines.append(f"  Convergence rate (log-log slope): {rate:.3f}")
            if rate < -0.4:
                lines.append("  (Good: faster than O(1/sqrt(T)) convergence)")
            elif rate < -0.2:
                lines.append("  (OK: approximately O(1/sqrt(T)) convergence)")
            else:
                lines.append("  (Slow: slower than expected convergence)")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "checkpoints": [
                {
                    "iteration": cp.iteration,
                    "wall_time_s": cp.wall_time_s,
                    "exploitability": cp.exploitability,
                    "metrics": cp.metrics,
                }
                for cp in self.checkpoints
            ],
        }

    def save(self, path: str) -> None:
        """Save training log to JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str, game: Game) -> "TrainingTracker":
        """Load a training log from JSON."""
        data = json.loads(Path(path).read_text())
        tracker = cls(game)
        for cp_data in data["checkpoints"]:
            tracker.checkpoints.append(Checkpoint(
                iteration=cp_data["iteration"],
                wall_time_s=cp_data["wall_time_s"],
                exploitability=cp_data["exploitability"],
                metrics=cp_data.get("metrics", {}),
            ))
        return tracker
