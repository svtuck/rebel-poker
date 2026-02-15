# ReBeL Scaling Plan: Kuhn → Leduc → Beyond

A technical roadmap for achieving a converging ReBeL training loop on Kuhn Poker and scaling to Leduc Poker. Based on an implementation audit, correctness verification, literature review of the ReBeL paper and 6 open-source implementations, and compute analysis for Apple Silicon.

---

## 1. Implementation Audit

### File-by-File Status

| File | Purpose | Status | Issues |
|------|---------|--------|--------|
| `kuhn/game.py` | Kuhn Poker game engine: states, actions, terminal utilities | **Working** | None. Correct zero-sum utilities, proper chance outcomes. |
| `kuhn/belief_state.py` | PBS tracking + `BeliefStateTracker` + `VectorizedCFR` | **Partial** | Uses joint `[6]` deal distribution, not factored `[3,2]` per-player PBS. See §3. `VectorizedCFR` is Kuhn-specific (hardcoded histories). |
| `cfr/solver.py` | Game-agnostic vanilla CFR via `Game` protocol | **Working** | Correctly converges. Exploitability computation via best response is sound. |
| `cfr/vectorized.py` | Game-agnostic vectorized CFR using PyTorch | **Working** | Discovers game structure dynamically. Converges to match scalar CFR. |
| `game_interface.py` | Abstract `Game` protocol (structural subtyping) | **Working** | Clean design. Any game implementing this protocol works with `cfr/solver.py` and `cfr/vectorized.py`. |
| `rebel/value_net.py` | Value network `V(PBS)` and policy network `π(PBS, card)` | **Partial** | Architecture exists but uses `PBS_DIM = 6` (flattened joint), not the factored representation. Policy network defined but never integrated into training loop. |
| `rebel/rebel_trainer.py` | ReBeL training loop: CFR → collect data → train V → repeat | **Partial** | Loop runs and exploitability decreases, but: (1) Kuhn-specific (hardcoded histories), (2) no policy network warm-start, (3) no replay buffer, (4) value network not used as leaf evaluator during CFR. |
| `rebel/data_logger.py` | Logs `(PBS, values)` pairs during CFR | **Working** | Correctly collects and flattens data. Supports the factored PBS format in its interface. |
| `rebel/endgame_solver.py` | Subgame solver with optional `V(PBS)` at leaves | **Partial** | Structure is correct but uses `[6]` joint beliefs. `_is_leaf()` always returns False (no depth-limited solving). Kuhn-specific histories. |
| `tests/test_kuhn_cfr.py` | Test suite (18 tests) | **Working** | All 18 tests pass. Validates CFR convergence, Nash properties, game value, belief states, vectorized CFR, subgame solver, and ReBeL trainer. |
| `ui/server.py` | Flask web UI for exploring solutions | **Working** | Functional API endpoints for strategy, beliefs, game tree. |
| `docs/apple_silicon_gpu.md` | GPU acceleration roadmap | **Reference** | Documents MLX, MPS, and Metal options. Good background. |
| `pyproject.toml` | Project metadata and dependencies | **Working** | Depends on `torch>=2.0`, `flask>=3.0`, `pytest>=7.0`. |
| `cfr/__init__.py` | Package exports | **Working** | Re-exports `CFRTrainer`, `InfoSet`, `VectorizedCFR`. |

### Architecture Summary

The codebase has a clean separation:
- **Game-agnostic layer:** `game_interface.py`, `cfr/solver.py`, `cfr/vectorized.py` — work with any `Game` implementation
- **Kuhn-specific layer:** `kuhn/game.py`, `kuhn/belief_state.py` — game logic and PBS tracking
- **ReBeL layer:** `rebel/` — training loop, value net, data logger, endgame solver

The game-agnostic CFR solver is the strongest part of the codebase. The ReBeL layer is the weakest — it works but has significant gaps before it can be called a correct ReBeL implementation.

---

## 2. Correctness Verification

### 2.1 CFR Solver (cfr/solver.py)

**Result: Correct.** Converges to the Kuhn Poker Nash equilibrium.

At 100,000 iterations:
- **Exploitability:** 0.000633 (near-zero, confirming Nash convergence)
- **Game value:** -0.05555 (matches analytical -1/18 = -0.05556)

Nash equilibrium strategy (Kuhn parameterization with α ≈ 0.22):

| Infoset | Strategy | Expected (α = 0.22) | Match? |
|---------|----------|---------------------|--------|
| P0 J bet | α = 0.220 | α ∈ [0, 1/3] | ✓ |
| P0 Q bet | 0.000 | 0 | ✓ |
| P0 K bet | 0.663 | 3α = 0.660 | ✓ |
| P1 J\|b fold | 1.000 | 1 | ✓ |
| P1 Q\|b call | 0.336 | 1/3 | ✓ |
| P1 K\|b call | 1.000 | 1 | ✓ |
| P1 J\|c bet | 0.332 | 1/3 | ✓ |
| P1 Q\|c bet | 0.000 | 0 | ✓ |
| P1 K\|c bet | 1.000 | 1 | ✓ |
| P0 J\|cb call | 0.000 | 0 | ✓ |
| P0 Q\|cb call | 0.555 | α + 1/3 = 0.553 | ✓ |
| P0 K\|cb call | 1.000 | 1 | ✓ |

Reference: Kuhn, H.W. (1950). "A simplified two-person poker." *Contributions to the Theory of Games*, 1:97-103.

### 2.2 Vectorized CFR (cfr/vectorized.py)

**Result: Correct.** At 10,000 iterations, exploitability = 0.002317. Strategies match scalar CFR within tolerance (< 0.15 difference at all infosets). The game-agnostic design correctly discovers Kuhn's game structure dynamically.

### 2.3 Endgame Solver (rebel/endgame_solver.py)

**Result: Correct for full-game solving.** When given the full game from root with uniform beliefs, converges to near-Nash (exploitability < 0.05 at 5,000 iterations). However, depth-limited solving is not implemented (`_is_leaf()` always returns `False`).

### 2.4 ReBeL Training Loop (rebel/rebel_trainer.py)

**Result: Functional but incomplete.**

5-epoch training run (200 CFR iterations/epoch, 50 value network epochs):
```
Exploitabilities: [0.0173, 0.0126, 0.0093, 0.0083, 0.0073]
Value losses:     [0.091,  0.075,  0.076,  0.071,  0.077]
```

Exploitability decreases monotonically from 0.017 to 0.007, confirming the training loop produces improving strategies. However, this improvement comes primarily from the CFR iterations themselves, not from the value network informing the search. The value network is trained but never used as a leaf evaluator — the loop is effectively running standalone CFR with a sidecar value network that learns from CFR outputs but doesn't feed back into it.

**Missing pieces for a true ReBeL loop:**
1. Value network as leaf evaluator during CFR (depth-limited solving)
2. Policy network warm-start for CFR iterations
3. Replay buffer for training stability
4. Self-play game generation (currently runs full-tree CFR, not episodic self-play)

### 2.5 All Tests

```
18 passed in 199.23s
```

All tests pass, including game properties, CFR convergence, Nash equilibrium validation, belief state tracking, vectorized CFR, subgame solver, and ReBeL trainer.

---

## 3. PBS Representation Fix (Issue #65)

### Current Implementation

The codebase uses a **joint distribution over all 6 deals** as the belief state:

```python
# kuhn/belief_state.py (current)
belief = torch.zeros(NUM_DEALS)  # [6] — one prob per (card0, card1) pair
joint = chance_probs * reach_p0 * reach_p1
belief = joint / joint.sum()
```

This represents P(card₀=i, card₁=j | history) as a single 6-dimensional vector.

### Correct Representation (ReBeL Paper)

The ReBeL paper defines PBS as **factored per-player beliefs**:

> β(h) = (h, Δ(S₁), Δ(S₂))

where Δ(Sᵢ) is a probability distribution over player i's private states.

(Brown et al., 2020, "Combining Deep Reinforcement Learning and Search for Imperfect-Information Games," §3, arXiv:2007.13544)

For Kuhn Poker, this is a `[3, 2]` matrix:

```
pbs[card, player] = P(player holds card | history)
```

Each column sums to 1 independently.

### Why This Matters

1. **Scalability:** For Leduc (6 private cards), the joint representation would be 30 values (6×5 ordered deals) vs 12 values (6 per player) factored. For NLHE, it's 1,326×1,325 = ~1.76M joint vs 2×1,326 = 2,652 factored.

2. **Generalization:** The value network should learn that similar per-player belief vectors produce similar values, regardless of the specific joint deal correlation. The factored representation enables this generalization.

3. **Paper fidelity:** The factored PBS is a core theoretical contribution. Using the joint representation means the value network isn't learning the right function.

### What Needs to Change

**File: `kuhn/belief_state.py`**

The `reach_to_pbs()` function already exists but returns `[3, 2]` — however, the `BeliefStateTracker` still computes and returns `[NUM_DEALS]` tensors in several places. The `compute_belief_states()` method should use `reach_to_pbs()` consistently.

Changes needed:
1. `compute_belief_states()` should return `Dict[str, torch.Tensor]` where tensors are `[NUM_PRIVATE_STATES, NUM_PLAYERS]`
2. `compute_counterfactual_values()` should return values indexed by private state per player, not by deal

**File: `rebel/value_net.py`**

```python
# Current
PBS_DIM = NUM_PRIVATE_STATES * NUM_PLAYERS  # 6 (coincidentally same as NUM_DEALS)

# Correct (already has the right value, but for the wrong reason)
# Input: [3, 2] flattened = 6
# Output: [3, 2] flattened = 6 (values per private state per player)
```

The dimensionality happens to be the same for Kuhn (6 deals = 3 cards × 2 players), but the *semantics* differ. The value network should predict V[card, player] = expected value if player holds card, not V[deal] = value for a specific (card₀, card₁) pair.

**File: `rebel/endgame_solver.py`**

The `_pbs_to_reach()` method needs to convert `[NUM_PRIVATE_STATES, NUM_PLAYERS]` PBS to per-deal reach probabilities for internal CFR computation. This already exists but the solver's interface should accept `[3, 2]` tensors.

**File: `rebel/rebel_trainer.py`**

The `deal_values_to_pbs_values()` function already converts deal-indexed values to per-private-state-per-player values. The `_cfr_with_logging()` method already calls `reach_to_pbs()`. Main change: ensure the value network receives and predicts in the factored format.

### Implementation Checklist for Issue #65

- [ ] Update `BeliefStateTracker.compute_belief_states()` to return `[3, 2]` tensors
- [ ] Update `BeliefStateTracker.compute_counterfactual_values()` to return per-player values
- [ ] Verify `reach_to_pbs()` marginalization is correct (sum over opponent's cards weighted by opponent reach)
- [ ] Update `SubgameSolver.__init__` to accept `[NUM_PRIVATE_STATES, NUM_PLAYERS]` initial beliefs
- [ ] Update `SubgameSolver._pbs_to_reach()` to convert from factored PBS to per-deal reaches
- [ ] Update value network training to use factored PBS as input/output
- [ ] Update tests to validate `[3, 2]` tensor shapes
- [ ] Verify that CFR convergence is unchanged (same exploitability)

---

## 4. Scaling Roadmap: Kuhn → Leduc → Larger Games

### 4.1 Game Complexity Comparison

| Metric | Kuhn Poker | Leduc Hold'em | Turn Endgame Hold'em | HUNL |
|--------|-----------|---------------|---------------------|------|
| Deck | 3 cards | 6 cards (3 ranks × 2 suits) | 52 cards | 52 cards |
| Rounds | 1 | 2 | 2 (turn + river) | 4 |
| Information sets | 12 | 288 | ~10⁸ | ~10¹¹ |
| Deals (ordered) | 6 | 30 (private) × 4 (board) = 120 | ~10⁶ | ~10⁶ |
| PBS dimensions (factored) | 3 × 2 = 6 | 6 × 2 + public = ~20 | 1326 × 2 + public = ~2700 | 1326 × 2 + public = ~2700 |
| Tabular CFR time | < 1 second | Seconds | Hours | Infeasible |

Sources: Kuhn (1950); Waugh et al. (2009), AAMAS 2010; Brown et al. (2020), arXiv:2007.13544

### 4.2 Leduc Poker: What Changes

**Game rules:**
- Deck: {J, Q, K} × 2 suits = 6 cards
- Round 1: Each player dealt 1 private card, then bet/check/raise with 2-chip raises, 2-bet max
- Round 2: 1 community card revealed, then bet/check/raise with 4-chip raises, 2-bet max
- Showdown: Pair beats high card. Same rank pair is best. Otherwise highest card wins.
- Ante: 1 chip each

**Implementation steps:**

1. **`leduc/game.py`** — New file implementing the `Game` protocol:
   - `LeducState`: cards tuple + history string + board card
   - `chance_outcomes()`: Two chance nodes — deal (round 1) and board card (round 2)
   - `legal_actions()`: Check/Bet/Call/Raise/Fold depending on history within each round
   - `terminal_utility()`: Pair > high card > tie
   - `infoset_key()`: `{private_card}|{board_card_or_none}|{betting_history}`

2. **No changes to `cfr/solver.py`** — Already game-agnostic via `Game` protocol. Will work immediately with Leduc.

3. **`leduc/belief_state.py`** — New file:
   - PBS dimension: 6 (private cards per player) × 2 players = 12, plus public state encoding
   - `reach_to_pbs()`: Marginalize over 30 possible private card deals
   - Board card encoding: one-hot over 6 possible community cards (or zero vector if round 1)

4. **`rebel/value_net.py`** — Parameterize for game size:
   - Input dim: `num_private_states * num_players + public_state_dim`
   - For Leduc: 6×2 + 7 (board one-hot + round indicator) = 19
   - Hidden dim: 128 (2 layers sufficient for 288 info sets)

5. **Tests:** Add `tests/test_leduc.py` validating game properties, CFR convergence, and known Leduc equilibrium game value (-0.086 chips/hand for player 1, per Waugh et al.)

### 4.3 Beyond Leduc

**Turn Endgame Hold'em (TEH)** is the next logical step after Leduc, used in the ReBeL paper as an intermediate benchmark. It starts partway through a NLHE hand (at the turn card) and plays out the remaining 2 streets.

**Full HUNL** requires:
- Card abstraction (grouping similar hands)
- Action abstraction (discretizing bet sizes)
- Monte Carlo CFR (MCCFR) instead of full traversal
- Much larger networks (6 layers × 1536 units per the paper)
- Distributed training (90 DGX-1 machines in the original paper)

This is out of scope for the near term but the architecture should support it.

---

## 5. Training Loop Design

### 5.1 Correct ReBeL Training Loop

The current implementation is missing key components. Here is the complete loop as described in the paper:

```
Algorithm: ReBeL Training (Brown et al., 2020, Algorithm 1)

Input: Game G, value network V_θ, policy network π_φ
Output: Trained V_θ, π_φ

Initialize replay buffer B = ∅

for epoch = 1 to N:
    # ---- Phase 1: Self-Play Data Generation ----
    for game = 1 to games_per_epoch:
        Play a full game using the following at each decision point:

        1. Observe current PBS β (factored per-player beliefs)
        2. WARM-START: Initialize CFR strategies using π_φ(β)
        3. Run T iterations of CFR on the subgame rooted at β:
           - At leaf nodes (depth limit D), use V_θ(β_leaf) as terminal values
           - Accumulate regrets and strategies as normal
        4. Compute average strategy σ̄ from CFR iterations
        5. Compute values v = CFR_value(β) from the T-iteration solve
        6. Store (β, v) in replay buffer B
        7. Sample action from σ̄ (with ε exploration)
        8. Update beliefs β based on action taken

    # ---- Phase 2: Train Value Network ----
    for batch in sample(B, num_batches):
        loss = MSE(V_θ(β_batch), v_batch)
        θ ← θ - α∇loss

    # ---- Phase 3: Train Policy Network ----
    for batch in sample(B, num_batches):
        loss = CrossEntropy(π_φ(β_batch, card), σ̄_batch)
        φ ← φ - α∇loss

    # ---- Phase 4: Evaluate ----
    Compute exploitability of current strategy
```

### 5.2 What's Missing in Current Implementation

| Component | Paper | Current Code | Priority |
|-----------|-------|-------------|----------|
| Value network as leaf evaluator | V(PBS) at depth limit | Not used during CFR | **P0** — core to ReBeL |
| Policy network warm-start | π(PBS) initializes CFR | Policy net defined but not in loop | **P1** — critical for convergence stability |
| Depth-limited solving | CFR to depth D, then V(PBS) | `_is_leaf()` returns False | **P0** — needed for value net to matter |
| Replay buffer | Circular buffer, sample uniformly | Data cleared each epoch | **P1** — stabilizes training |
| Self-play game generation | Play full games episodically | Full-tree CFR traversal | **P2** — needed for Leduc+, fine for Kuhn |
| Exploration | ε-greedy action selection | None | **P2** — needed for diverse training data |
| Learning rate schedule | Halve every N epochs | Fixed lr | **P2** |

### 5.3 Recommended Hyperparameters

#### Kuhn Poker

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Value net architecture | 2 layers × 64 units, ReLU | 12 info sets, 6-dim PBS; tiny game |
| Policy net architecture | 2 layers × 64 units, softmax output | Same reasoning |
| Learning rate | 1e-3, halve every 200 epochs | Fast convergence for small game |
| Batch size | 32 | Small dataset |
| Replay buffer size | 10,000 | ~100 epochs of data |
| CFR iterations per subgame | 100 | Sufficient for Kuhn convergence |
| Depth limit | None (full game) | Kuhn is too small for depth-limited solving |
| Training epochs | 500 | Should converge well before this |
| Games per epoch | 100 | Each game produces ~4 decision points |
| Optimizer | Adam | Standard choice |
| Exploration ε | 0.25 | Matches paper |

Expected convergence: Exploitability < 0.01 within 100 epochs, < 0.001 within 500.

#### Leduc Poker

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Value net architecture | 2 layers × 128 units, GeLU + LayerNorm | 288 info sets, ~20-dim PBS |
| Policy net architecture | 2 layers × 128 units, softmax output | Same |
| Learning rate | 3e-4, halve every 400 epochs | Matches paper's Liar's Dice config |
| Batch size | 256 | More data per epoch |
| Replay buffer size | 100,000 | Larger game needs more history |
| CFR iterations per subgame | 256 | More iterations needed for larger game |
| Depth limit | 1-2 rounds ahead | Test with depth 1, then 2 |
| Training epochs | 1,000 | Based on Liar's Dice convergence |
| Games per epoch | 1,000 | More diverse training data |
| Optimizer | Adam | Standard |
| Exploration ε | 0.25 | Matches paper |

Expected convergence: Exploitability < 0.05 within 200 epochs, < 0.01 within 1,000.

### 5.4 Measuring Convergence

**Exploitability** is the primary metric:

```
exploitability = 0.5 × (BR_value(player0) + BR_value(player1))
```

where `BR_value(player_i)` is the expected value when player i plays a best response against the learned strategy and the opponent plays the learned strategy.

For Kuhn: Compute exactly via full tree traversal (already implemented in `cfr/solver.py`).
For Leduc: Same approach, slightly slower but still exact.
For larger games: Must use sampling-based estimation.

**Convergence criteria:**
- Kuhn: exploitability < 0.001 (game value is -1/18 ≈ 0.056, so 0.001 is < 2% of game value)
- Leduc: exploitability < 0.01 (game value ≈ 0.086)

**Additional metrics:**
- Value network MSE loss (should decrease over training)
- Policy network cross-entropy loss (should decrease)
- Strategy entropy at key infosets (should stabilize)

---

## 6. Compute Estimates

### Hardware: Apple Silicon M2 Ultra (Mac Studio)

- 24 CPU cores (16 performance + 8 efficiency)
- 76 GPU cores
- 192 GB unified memory
- PyTorch MPS backend for GPU acceleration
- MLX for custom Metal kernels (future)

### Training Time Estimates

| Game | Method | Estimated Time | Bottleneck |
|------|--------|---------------|-----------|
| Kuhn | Tabular CFR (100k iterations) | < 1 second | N/A |
| Kuhn | ReBeL training (500 epochs) | 5-10 minutes | Value net training |
| Leduc | Tabular CFR (100k iterations) | Seconds | N/A |
| Leduc | ReBeL training (1,000 epochs) | 2-8 hours | CFR tree traversal |
| Liar's Dice (1x6f) | ReBeL training | Hours (single GPU in paper) | Similar to Leduc |
| HUNL | ReBeL training | Weeks on cluster | Infeasible on single machine |

Sources: Empirical measurement for Kuhn (this project). Liar's Dice estimate from Brown et al. (2020), which used 60 CPU threads + single GPU for 1,000 epochs × 25,600 examples. Leduc extrapolated from Liar's Dice (similar info set count).

### PyTorch MPS Acceleration

For the value/policy network training, MPS provides GPU acceleration on Apple Silicon:

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

CFR tree traversal is CPU-bound (recursive, branching). The vectorized CFR (`cfr/vectorized.py`) can benefit from MPS for tensor operations but the tree structure limits parallelism. For Leduc, the game tree is small enough that CPU is likely faster than the overhead of MPS dispatch.

The main GPU win comes at Leduc+ scale when training the value network on larger batches.

---

## 7. Open-Source Implementation Survey

### Implementations Analyzed

| Repository | Games | Stars | Key Lesson |
|-----------|-------|-------|------------|
| [facebookresearch/rebel](https://github.com/facebookresearch/rebel) | Liar's Dice only | 689 | Official ReBeL code. Poker not open-sourced. C++/Python split. Value net: 2 layers × 256 for Liar's Dice, 6 layers × 1536 for poker. **Archived.** |
| [google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel) | 80+ games incl. Kuhn, Leduc | ~5,000 | Best reference for CFR variants. Has vanilla CFR, CFR+, MCCFR, DCFR, LCFR. No ReBeL implementation. C++/Python. |
| [noambrown/poker_solver](https://github.com/noambrown/poker_solver) | Kuhn, Leduc, NLHE river | 132 | By the ReBeL author. CFR/CFR+/MCCFR/FP/DCFR. Python + C++. Range-based representation. Most architecturally relevant reference for this project. |
| [lifrordi/DeepStack-Leduc](https://github.com/lifrordi/DeepStack-Leduc) | No-limit Leduc | — | DeepStack algorithm on Leduc. Demonstrates PBS-based value net training. Lua/Torch. Good for understanding depth-limited solving. |
| [EricSteinberger/PokerRL](https://github.com/EricSteinberger/PokerRL) | NLHE | 508 | Deep CFR, NFSP, distributed training with Ray. Pinned to PyTorch 0.4.1 (outdated). |
| [datamllab/rlcard](https://github.com/datamllab/rlcard) | Leduc, Texas Hold'em, etc. | 3,400 | Pre-trained Leduc CFR model. Good for quick prototyping. Pure Python. |

### Key Lessons from the Literature

1. **Policy network is critical.** The facebookresearch/rebel Issue #25 documents a reimplementation attempt that skipped the policy network and warm-start. Result: value network training error diverged epoch-over-epoch. The policy network reduces the number of CFR iterations needed per subgame solve. ([GitHub Issue #25](https://github.com/facebookresearch/rebel/issues/25))

2. **Rust for data generation, Python for training.** Both the official ReBeL code and noambrown/poker_solver use a native/Python split (C++ in their case). CFR tree traversal is performance-critical; neural network training uses standard PyTorch. For Kuhn and Leduc, Python-only is fine. Our native backend will use Rust with pyo3 bindings.

3. **Start with tabular CFR as ground truth.** OpenSpiel's approach of having exact tabular solvers alongside approximate methods is valuable. Always verify the neural network approach against the tabular solution.

4. **Leduc is the right stepping stone.** It's small enough for exact solving (288 info sets) but large enough to require the factored PBS representation and test the full pipeline. Liar's Dice (used in the official ReBeL code) is a comparable benchmark.

---

## 8. Concrete Next Steps

### Phase 1: Fix PBS and Complete Kuhn ReBeL (Issues #65, #67)

1. Implement factored PBS `[NUM_PRIVATE_STATES, NUM_PLAYERS]` throughout the codebase (Issue #65)
2. Add depth-limited solving to `SubgameSolver` — use value network at leaf nodes
3. Integrate policy network into training loop for warm-starting CFR
4. Add replay buffer to `RebelTrainer`
5. Verify: exploitability converges to < 0.001 on Kuhn with the complete pipeline
6. Benchmark: training time on M2 Ultra

### Phase 2: Implement Leduc Poker

7. Implement `leduc/game.py` following the `Game` protocol
8. Verify `cfr/solver.py` converges on Leduc (should work immediately)
9. Implement `leduc/belief_state.py` with 6-dim per-player PBS
10. Scale value/policy networks (2 layers × 128 units)
11. Run ReBeL training loop on Leduc
12. Verify: exploitability < 0.01 on Leduc

### Phase 3: Optimize and Scale

13. Profile and optimize CFR traversal for Leduc (vectorized operations)
14. Enable PyTorch MPS for value net training
15. Experiment with CFR+ and DCFR variants
16. Consider MLX for custom GPU kernels if performance-bound
17. Document results and compare to OpenSpiel/rlcard baselines

---

## References

1. Brown, N., Bakhtin, A., Lerer, A., Gong, Q. (2020). "Combining Deep Reinforcement Learning and Search for Imperfect-Information Games." NeurIPS 2020. https://arxiv.org/abs/2007.13544
2. Kuhn, H.W. (1950). "A simplified two-person poker." *Contributions to the Theory of Games*, 1:97-103.
3. Zinkevich, M. et al. (2007). "Regret Minimization in Games with Incomplete Information." NIPS 2007. https://papers.nips.cc/paper/3306
4. Waugh, K. et al. (2009). "A Practical Use of Imperfect Recall." AAMAS 2010. https://poker.cs.ualberta.ca/publications/AAMAS10.pdf
5. Lanctot, M. et al. (2019). "OpenSpiel: A Framework for Reinforcement Learning in Games." https://arxiv.org/abs/1908.09453
6. facebookresearch/rebel. https://github.com/facebookresearch/rebel
7. google-deepmind/open_spiel. https://github.com/google-deepmind/open_spiel
8. noambrown/poker_solver. https://github.com/noambrown/poker_solver
9. lifrordi/DeepStack-Leduc. https://github.com/lifrordi/DeepStack-Leduc
10. EricSteinberger/PokerRL. https://github.com/EricSteinberger/PokerRL
11. datamllab/rlcard. https://github.com/datamllab/rlcard
12. facebookresearch/rebel Issue #25. https://github.com/facebookresearch/rebel/issues/25
