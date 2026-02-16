"""Evaluation framework for rebel-poker.

Provides:
- exploitability: Best-response computation (exact + LBR sampling)
- head_to_head: Self-play and baseline opponent evaluation
- profiler: Wall-clock, memory, and throughput profiling
- slumbot: Slumbot API client for HUNL evaluation
- training_tracker: Exploitability tracking over training iterations
"""
