# rebel-poker

A from-scratch implementation of the [ReBeL algorithm](https://arxiv.org/abs/2007.13544) (Brown et al., 2020) for imperfect-information games, starting with Kuhn Poker and designed to scale toward Leduc Poker and beyond.

ReBeL (**Re**cursive **Be**lief-based **L**earning) is the first algorithm to combine deep reinforcement learning with search in imperfect-information games while provably converging to a Nash equilibrium. It generalizes the AlphaZero paradigm — self-play plus search plus learned evaluation — from perfect-information games like chess and Go to games with hidden information like poker.

This README explains the theory behind ReBeL, traces its intellectual lineage, surveys the work it has influenced, and maps each concept to the corresponding code in this project.

---

## Table of Contents

1. [ReBeL Core Concepts](#1-rebel-core-concepts)
   - [The Problem: Imperfect Information](#the-problem-imperfect-information)
   - [Public Belief States](#public-belief-states-pbs)
   - [The Self-Play + Search Paradigm](#the-self-play--search-paradigm)
   - [Depth-Limited Solving with Value Networks](#depth-limited-solving-with-value-networks)
   - [The Training Loop](#the-training-loop)
   - [Why PBS Is the Right Abstraction](#why-pbs-is-the-right-abstraction)
2. [Papers That ReBeL Builds On](#2-papers-that-rebel-builds-on)
3. [Papers That Reference or Extend ReBeL](#3-papers-that-reference-or-extend-rebel)
4. [How This Project Maps to the Paper](#4-how-this-project-maps-to-the-paper)
5. [Running the Code](#5-running-the-code)
6. [References](#6-references)

---

## 1. ReBeL Core Concepts

### The Problem: Imperfect Information

In perfect-information games (chess, Go), both players see the full board state. A game position is a **sufficient statistic** for decision-making — you don't need to reason about what your opponent knows or believes.

In imperfect-information games (poker, negotiation, many real-world settings), players hold **private information** that others cannot observe. This creates a fundamental challenge: the optimal action depends not just on the game state but on what you *believe* about the hidden information, and what your opponent believes about *your* hidden information, and so on.

Naive tree search doesn't work here. In chess, you can evaluate a position by searching ahead and backing up values. In poker, if you search ahead from a specific hand (say you hold King-Queen), you're implicitly revealing that hand to the search tree — the resulting strategy would be exploitable because it would play differently based on information the opponent doesn't have.

ReBeL's key insight is that you can restore the search paradigm by searching over **beliefs** instead of **states**.

### Public Belief States (PBS)

A Public Belief State (PBS) is a probability distribution over the hidden information (private cards), conditioned on everything that is publicly observable (the history of bets, checks, folds).

Formally, for a two-player game, a PBS at public history *h* is:

> **β(h) = (h, Δ(S₁), Δ(S₂))**

where Δ(Sᵢ) is a probability distribution over player *i*'s private states, computed from the players' strategies and the chance probabilities:

```
P(cards | history h) ∝ reach₀(cards, h) × reach₁(cards, h) × P_chance(cards)
```

Here, `reachᵢ(cards, h)` is the probability that player *i* would play to reach history *h* given that they hold `cards[i]`, according to their current strategy.

**How PBS differs from information sets:** An information set groups all game states that a single player cannot distinguish. PBS is a *public* concept — it represents what an omniscient observer would believe about the hidden state, given only public actions. Crucially, PBS factors into per-player beliefs: you track each player's card distribution independently, rather than tracking a joint distribution over all possible card assignments.

**Concrete example (Kuhn Poker):** At the start of the game with 3 cards (J, Q, K) and 2 players, each player is equally likely to hold any card. After player 0 bets, the PBS shifts: player 0 is now more likely to hold K (strong hand bets more often) and more likely to hold J (bluffs sometimes), but less likely to hold Q (rarely bets). The PBS encodes this updated belief.

### The Self-Play + Search Paradigm

AlphaZero's breakthrough was combining three ideas: (1) self-play to generate training data, (2) search (MCTS) at each decision point, and (3) a learned value function to evaluate leaf positions.

ReBeL adapts this paradigm to imperfect-information games:

| AlphaZero | ReBeL |
|-----------|-------|
| Game state | Public Belief State (PBS) |
| MCTS for search | CFR for search |
| Value network V(state) | Value network V(PBS) |
| Self-play on game tree | Self-play on public game tree |

The critical substitution is replacing **Monte Carlo Tree Search** with **Counterfactual Regret Minimization (CFR)** as the search algorithm. MCTS assumes a single game state and backs up scalar values; CFR operates over belief distributions and handles the strategic reasoning needed when information is hidden.

### Depth-Limited Solving with Value Networks

In a large game like no-limit hold'em, you cannot run CFR to completion — the game tree is too large. ReBeL uses **depth-limited solving**: run CFR on a shallow subgame (a few actions deep), and at the leaves, instead of rolling out to terminal states, query a neural network:

```
V(PBS) → expected payoff for each possible private state
```

This is analogous to AlphaZero using its value network at MCTS leaf nodes instead of playing random rollouts to the end of the game.

The value network takes a PBS as input (a probability distribution over deals) and predicts the expected payoff for each possible private state (each possible card a player might hold). This provides the "intuition" that makes shallow search effective.

### The Training Loop

ReBeL's training loop alternates between **generating data** (via CFR self-play) and **improving the value network**:

```
for epoch in 1..N:
    1. SELF-PLAY WITH SEARCH
       - Play games using CFR at each decision point
       - At each PBS encountered, run T iterations of CFR
       - Use V(PBS) to evaluate leaf nodes during CFR
       - Record (PBS, computed_values) pairs as training data

    2. TRAIN VALUE NETWORK
       - Train V(PBS) to predict the CFR-computed values
       - Loss = MSE between V(PBS) and actual CFR values
       - This teaches the network what different belief states are "worth"

    3. EVALUATE
       - Measure exploitability of the resulting strategy
       - The value network should improve each epoch,
         leading to better CFR solutions, leading to better training data
```

This creates a virtuous cycle: better value estimates → better CFR solutions → better training data → better value estimates. The paper proves that this process converges to a Nash equilibrium in two-player zero-sum games.

### Why PBS Is the Right Abstraction

PBS is a **sufficient statistic** for the public game tree. This means:

1. **No information loss:** Given a PBS, you can compute the optimal strategy without knowing the exact sequence of private events that led to it. Two different game histories that produce the same PBS are strategically equivalent.

2. **Enables generalization:** Different game situations that produce similar belief distributions should have similar values. A neural network trained on PBS can generalize across situations in a way that a lookup table indexed by game history cannot.

3. **Factored representation scales:** Because PBS factors into per-player beliefs, the representation size grows linearly with the number of private states per player, not combinatorially. For no-limit hold'em, the factored PBS is a [C(52,2)] = [1,326] vector per player (2,652 total), versus a [1,326 × 1,325] ≈ 1.76M-dimensional joint distribution.

4. **Sound game-theoretic foundation:** The ReBeL paper proves that the value of a subgame can be expressed as a function of the PBS at its root, which is what makes depth-limited solving with a learned V(PBS) theoretically justified.

---

## 2. Papers That ReBeL Builds On

ReBeL stands on the shoulders of two decades of research in computational game theory and AI. Here is the intellectual lineage, with each paper's key contribution and its relationship to ReBeL.

### Counterfactual Regret Minimization (CFR)

> **Zinkevich, M., Johanson, M., Bowling, M., Piccione, C.** "Regret Minimization in Games with Incomplete Information." *NIPS 2007.*
> [NeurIPS proceedings](https://papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information)

CFR is the foundational algorithm for solving imperfect-information games. It works by iterating over the game tree, tracking "regret" for each action at each information set (how much better that action would have been in hindsight), and updating the strategy to favor actions with higher regret. Over many iterations, the *average* strategy converges to a Nash equilibrium.

**Relation to ReBeL:** CFR is the search algorithm inside ReBeL. Where AlphaZero uses MCTS to search, ReBeL runs CFR at each decision point. The key innovation of ReBeL is wrapping CFR in a learning loop with a value network, rather than using CFR as a standalone solver.

### Monte Carlo CFR (MCCFR)

> **Lanctot, M., Waugh, K., Zinkevich, M., Bowling, M.** "Monte Carlo Sampling for Regret Minimization in Extensive Games." *NIPS 2009.*
> [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html)

MCCFR introduced sampling-based variants of CFR (external sampling, outcome sampling, etc.) that don't need to traverse the entire game tree on each iteration. Each iteration is cheaper, though more iterations are needed. This made CFR practical for much larger games.

**Relation to ReBeL:** MCCFR variants can be used as the search subroutine within ReBeL to handle games where full CFR traversals are too expensive. Pluribus (a precursor to ReBeL) used Monte Carlo CFR for its blueprint strategy computation.

### CFR+

> **Tammelin, O.** "Solving Large Imperfect Information Games Using CFR+." *arXiv preprint, 2014.*
> [arXiv:1407.5042](https://arxiv.org/abs/1407.5042)

CFR+ modifies the regret update rule: instead of allowing regrets to go negative and then clamping during strategy computation, CFR+ floors regrets to zero *immediately*. This seemingly small change (regret matching+ instead of regret matching) leads to dramatically faster convergence — often 10× or more. CFR+ was the engine behind Cepheus, the first program to essentially solve heads-up limit Texas hold'em.

**Relation to ReBeL:** CFR+ can be used in place of vanilla CFR within the ReBeL search procedure for faster convergence. The project's `SubgameConfig` includes a `use_cfr_plus` flag for this purpose.

### DeepStack

> **Moravčík, M., Schmid, M., Burch, N., Lisý, V., Morrill, D., Bard, N., Davis, T., Waugh, K., Johanson, M., Bowling, M.** "DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker." *Science, 356(6337):508-513, 2017.*
> [arXiv:1701.01724](https://arxiv.org/abs/1701.01724)

DeepStack was the first AI to combine neural networks with depth-limited solving in poker. It introduced the idea of **continual re-solving**: at each decision point, solve a depth-limited subgame using a neural network to evaluate the leaves, rather than relying on a pre-computed blueprint strategy. DeepStack defeated professional poker players in heads-up no-limit Texas hold'em.

**Relation to ReBeL:** DeepStack is a direct precursor. ReBeL's depth-limited solving with V(PBS) at leaf nodes follows the same pattern. The key difference is that ReBeL formalizes this into a general RL + search framework with convergence guarantees, while DeepStack used a more ad-hoc training procedure and was specific to poker.

### Libratus

> **Brown, N., Sandholm, T.** "Superhuman AI for Heads-Up No-Limit Poker: Libratus Beats Top Professionals." *Science, 359(6374):418-424, 2018.*
> [DOI:10.1126/science.aao1733](https://www.science.org/doi/10.1126/science.aao1733)

Libratus defeated top human professionals at heads-up no-limit Texas hold'em. Its architecture combined three modules: (1) a pre-computed blueprint strategy using abstraction + CFR, (2) real-time **safe subgame solving** that refines the strategy during play, and (3) a self-improvement module that patches exploitable parts of the strategy between matches.

**Relation to ReBeL:** Libratus demonstrated that subgame solving during play is essential for superhuman poker. ReBeL's search procedure descends from Libratus's subgame solver, but ReBeL replaces the pre-computed blueprint with a learned value function, eliminating the need for game-specific abstractions.

### Pluribus

> **Brown, N., Sandholm, T.** "Superhuman AI for Multiplayer Poker." *Science, 365(6456):885-890, 2019.*
> [DOI:10.1126/science.aay2400](https://www.science.org/doi/10.1126/science.aay2400)

Pluribus extended superhuman poker AI to six-player no-limit Texas hold'em — a much harder setting because Nash equilibria aren't as well-defined in multiplayer games. It used MCCFR for blueprint computation and a novel depth-limited search procedure during play.

**Relation to ReBeL:** Pluribus showed that depth-limited search with learned evaluation can work in multiplayer settings. ReBeL generalizes the search framework to work with formal PBS and proves convergence guarantees for the two-player case.

### DREAM

> **Steinberger, E., Lerer, A., Brown, N.** "DREAM: Deep Regret Minimization with Advantage Baselines and Model-free Learning." *arXiv preprint, 2020.*
> [arXiv:2006.10410](https://arxiv.org/abs/2006.10410)

DREAM is a model-free deep RL algorithm for imperfect-information games. Unlike CFR (which requires a game model to traverse the tree), DREAM learns purely from sampled episodes, using advantage baselines to reduce variance. It converges to Nash equilibrium in two-player zero-sum games without needing a simulator.

**Relation to ReBeL:** DREAM and ReBeL were developed concurrently at Facebook AI Research. They represent two different approaches to applying deep learning to imperfect-information games: DREAM is model-free (no search, pure learning), while ReBeL is model-based (search + learning). ReBeL's approach of combining search with learning generally achieves stronger performance, especially with limited training compute.

### Safe Subgame Solving

> **Brown, N., Sandholm, T.** "Safe and Nested Subgame Solving for Imperfect-Information Games." *NIPS 2017.*
> [arXiv:1705.02955](https://arxiv.org/abs/1705.02955)

This paper introduced the theory of safe subgame solving — how to re-solve a subgame during play without introducing exploitability. In perfect-information games, you can solve any subtree independently. In imperfect-information games, naive subgame solving can create exploitable strategies because it ignores the opponent's ability to reach the subgame with different beliefs. Safe solving adds constraints that preserve game-theoretic soundness.

**Relation to ReBeL:** ReBeL's subgame solving procedure inherits the safety guarantees from this work. The endgame solver in this project follows the safe solving pattern: it takes an initial PBS (encoding what the opponent might hold) and solves the subgame respecting those beliefs.

### Value Functions for Depth-Limited Solving

> **Kovařík, V., Schmid, M., Burch, N., Bowling, M., Lisý, V.** "Value Functions for Depth-Limited Solving in Zero-Sum Imperfect-Information Games." *Artificial Intelligence, 2023.*
> [arXiv:1906.06412](https://arxiv.org/abs/1906.06412)

This paper provides the theoretical foundation for using value networks in depth-limited solving. It proves that public information is both necessary and sufficient for computing optimal value functions — justifying the PBS-based approach used by DeepStack and ReBeL.

**Relation to ReBeL:** This work provides the formal proof that V(PBS) is a well-defined and sufficient value function for depth-limited subgame solving, which is a core assumption of the ReBeL framework.

---

## 3. Papers That Reference or Extend ReBeL

### Student of Games

> **Schmid, M., Moravčík, M., Burch, N., Kadlec, R., Davidson, J., Waugh, K., Bard, N., Timbers, F., Lanctot, M., Holland, G.Z., Davoodi, E., Christianson, A., Bowling, M.** "Student of Games: A Unified Learning Algorithm for Both Perfect and Imperfect Information Games." *Science Advances, 9(46), 2023.*
> [arXiv:2112.03178](https://arxiv.org/abs/2112.03178) | [DOI:10.1126/sciadv.adg3256](https://www.science.org/doi/10.1126/sciadv.adg3256)

*Note: This paper was originally published as "Player of Games" (arXiv v1, Dec 2021) and renamed to "Student of Games" when published in Science Advances (v2, Nov 2023).*

Student of Games unifies MCTS (for perfect-information games) and CFR-based search (for imperfect-information games) into a single algorithm. It achieves strong performance across chess, Go, poker, and Scotland Yard, demonstrating that a single learning algorithm can handle both game types.

**Relation to ReBeL:** Student of Games extends ReBeL's paradigm by combining it with MCTS-style search. Where ReBeL uses CFR exclusively, Student of Games uses a hybrid search that reduces to MCTS in perfect-information settings and to CFR-like search in imperfect-information settings. It validates the PBS-based approach as a unifying framework.

### RL-CFR

> **Li, B., Fang, Z., Huang, L.** "RL-CFR: Improving Action Abstraction for Imperfect Information Extensive-Form Games with Reinforcement Learning." *ICML 2024, PMLR 235:27752-27770.*
> [arXiv:2403.04344](https://arxiv.org/abs/2403.04344)

RL-CFR extends the ReBeL framework with **dynamic action abstraction** via reinforcement learning. Instead of using a fixed set of bet sizes (a critical limitation of prior poker AI), RL-CFR learns which bet sizes to include in the abstraction during search.

**Relation to ReBeL:** Directly builds on ReBeL's search procedure, adding learned abstraction on top. Outperforms ReBeL's replication and the Slumbot benchmark in heads-up no-limit Texas hold'em.

### Look-Ahead Search on Top of Policy Networks

> **Kubíček, O., Burch, N., Lisý, V.** "Look-Ahead Search on Top of Policy Networks in Imperfect Information Games." *IJCAI 2024.*
> [arXiv:2312.15220](https://arxiv.org/abs/2312.15220)

This work adds test-time search to policy-gradient algorithms (like those used in DREAM) by incorporating a learned critic for depth-limited look-ahead. It bridges the gap between model-free policy learning and ReBeL-style model-based search.

**Relation to ReBeL:** Demonstrates that ReBeL's search-at-test-time paradigm can be applied on top of policy-gradient methods, not just CFR-based self-play.

### LAMIR

> **Kubíček, O., Lisý, V.** "Look-Ahead Reasoning with a Learned Model in Imperfect Information Games." *arXiv preprint, 2025.*
> [arXiv:2510.05048](https://arxiv.org/abs/2510.05048)

LAMIR extends the MuZero paradigm to imperfect-information games by learning an abstracted game model for look-ahead reasoning, rather than requiring a perfect simulator.

**Relation to ReBeL:** While ReBeL requires a known game model for its CFR search, LAMIR learns the model, potentially enabling ReBeL-like search in games where the rules aren't fully specified.

---

## 4. How This Project Maps to the Paper

### File-to-Concept Mapping

| File | ReBeL Concept | Paper Reference |
|------|---------------|-----------------|
| `kuhn/game.py` | Game definition: states, actions, terminal utilities, information sets | Foundation — defines the extensive-form game |
| `kuhn/cfr.py` | Vanilla CFR implementation: regret matching, strategy accumulation, exploitability via best response | §3.1 — CFR as the search subroutine |
| `kuhn/belief_state.py` — `BeliefStateTracker` | PBS computation: reach probabilities, belief normalization, player beliefs | §3 — Public Belief States |
| `kuhn/belief_state.py` — `VectorizedCFR` | Vectorized CFR operating over all deals simultaneously using PyTorch tensors | §3.1 — batch computation pattern needed for GPU acceleration |
| `rebel/value_net.py` — `ValueNetwork` | V(PBS) → expected payoffs per deal | §3.2 — value network for leaf evaluation |
| `rebel/value_net.py` — `PolicyNetwork` | π(PBS, private_info) → action probabilities | §3.2 — policy network |
| `rebel/rebel_trainer.py` | The full ReBeL training loop: CFR self-play → collect (PBS, value) pairs → train V → repeat | §4 — Algorithm 1 (ReBeL training) |
| `rebel/data_logger.py` | Data collection during CFR: logs (history, PBS, reach_probs, strategy, values) at each decision point | §4 — training data generation |
| `rebel/endgame_solver.py` | Subgame solver: runs CFR from a given PBS with optional V(PBS) at leaves | §3.2 — depth-limited subgame solving |
| `tests/test_kuhn_cfr.py` | Validation against known Nash equilibrium properties of Kuhn Poker | Verification — game value = -1/18, known strategy properties |
| `ui/server.py` | Web interface for exploring solutions, belief states, and the game tree | Visualization (not in paper) |
| `docs/apple_silicon_gpu.md` | GPU acceleration roadmap for Apple Silicon (MLX, MPS, Metal) | Implementation-specific (not in paper) |

### Key Implementation Details

**CFR with reach probabilities** (`kuhn/cfr.py:61-112`): The scalar CFR implementation follows the standard formulation — recursive tree traversal, regret accumulation weighted by opponent reach, strategy accumulation weighted by player reach. The `_best_response_value` method computes exploitability by finding the optimal counter-strategy.

**Vectorized CFR** (`kuhn/belief_state.py:237-481`): Instead of traversing one card deal at a time, this processes all 6 deals simultaneously as a `[6]` tensor. Each CFR operation (strategy expansion, reach propagation, regret update) operates on the full deal vector. This is the computation pattern that maps to GPU parallelism for larger games.

**PBS computation** (`kuhn/belief_state.py:164-184`): The belief state at history *h* is computed as:
```python
joint = chance_probs * reach_p0 * reach_p1
belief = joint / joint.sum()
```
This normalizes the product of chance and reach probabilities into a probability distribution over deals.

**ReBeL training loop** (`rebel/rebel_trainer.py:99-144`): Implements the core algorithm:
1. Run CFR with data logging for `cfr_iters_per_epoch` iterations
2. Collect (PBS, values) pairs via the data logger
3. Train the value network on MSE loss against CFR values
4. Evaluate exploitability of the resulting strategy

**Value network** (`rebel/value_net.py:23-51`): A simple 3-layer MLP: `[NUM_DEALS] → 64 → 64 → [NUM_DEALS]`. Takes a PBS (probability distribution over deals) and predicts expected payoffs for each deal. For Kuhn Poker this is deliberately simple; larger games would use much deeper networks.

### Known Limitations and the PBS Fix (Issue #65)

The current implementation represents the PBS as a **joint distribution over all 6 card deals** — a `[6]` vector where each entry corresponds to a (card₀, card₁) pair like (J,Q), (J,K), etc.

[Issue #65](https://github.com/svtuck/task-ingestion/issues/65) documents that the ReBeL paper defines PBS differently: as **factored per-player beliefs**, i.e., `(Δ(S₁), Δ(S₂))` — a `[3]` vector per player representing P(J), P(Q), P(K) for each player independently. This gives a `[3, 2]` matrix for Kuhn Poker.

The joint representation is:
- **Correct for Kuhn Poker** — with only 6 deals, the joint and factored representations are both tractable and produce the same strategies
- **Won't scale** — for no-limit hold'em, the joint would be C(52,2)² ≈ 1.76M values vs. 2 × C(52,2) = 2,652 in the factored form
- **Not what the paper describes** — the factored PBS is a core theoretical contribution of ReBeL

Fixing this representation is tracked in issue #65 and is a prerequisite for scaling beyond Kuhn Poker.

### What's Implemented vs. What's Next

| Component | Status |
|-----------|--------|
| Kuhn Poker game engine | Done |
| Scalar (vanilla) CFR | Done |
| Vectorized CFR (PyTorch tensors) | Done |
| PBS tracking and computation | Done (joint representation) |
| Value network V(PBS) | Done |
| Policy network π(PBS, card) | Done (architecture only, not trained in loop) |
| ReBeL training loop | Done |
| Endgame/subgame solver | Done |
| Data logging for training | Done |
| Web UI for exploration | Done |
| PBS factored representation (issue #65) | Not yet |
| Leduc Poker game | Not yet |
| CFR+ variant | Scaffolded (`use_cfr_plus` flag) |
| GPU acceleration (Apple Silicon) | Documented, not yet implemented |
| Larger game support (NLHE) | Future |

---

## 5. Running the Code

### Prerequisites

```bash
pip install torch flask pytest
```

Or install from `pyproject.toml`:

```bash
pip install -e .
```

### Run the Solver

```bash
# Run CFR (scalar + vectorized) with default 10,000 iterations
python run_kuhn.py

# Run with custom iteration count
python run_kuhn.py 50000
```

This will display:
- Strategy profiles from both scalar and vectorized CFR
- Public belief states at every game history
- Terminal values and ReBeL training data preview
- Exploitability measurements (should converge toward 0)

### Run the Web UI

```bash
python ui/server.py
```

Opens a Flask server at `http://localhost:5050` with an API for exploring solutions, belief states, and the game tree.

### Run Tests

```bash
pytest tests/ -v
```

Tests validate:
- Kuhn Poker game properties (zero-sum, correct legal actions)
- CFR convergence to near-zero exploitability
- Known Nash equilibrium strategy properties (K always calls, J always folds vs. bet, etc.)
- Game value ≈ -1/18 for player 0
- Belief state properties (sum to 1, update correctly after bets)
- Vectorized CFR matches scalar CFR
- Subgame solver convergence
- ReBeL training loop runs and produces bounded exploitability

---

## 6. References

Full citations for all papers discussed above, ordered chronologically.

1. **Zinkevich, M., Johanson, M., Bowling, M., Piccione, C.** (2007). "Regret Minimization in Games with Incomplete Information." *Advances in Neural Information Processing Systems 20 (NIPS 2007).* [NeurIPS proceedings](https://papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information)

2. **Lanctot, M., Waugh, K., Zinkevich, M., Bowling, M.** (2009). "Monte Carlo Sampling for Regret Minimization in Extensive Games." *Advances in Neural Information Processing Systems 22 (NIPS 2009).* [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2009/hash/00411460f7c92d2124a67ea0f4cb5f85-Abstract.html)

3. **Tammelin, O.** (2014). "Solving Large Imperfect Information Games Using CFR+." *arXiv preprint.* [arXiv:1407.5042](https://arxiv.org/abs/1407.5042)

4. **Brown, N., Sandholm, T.** (2017). "Safe and Nested Subgame Solving for Imperfect-Information Games." *Advances in Neural Information Processing Systems 30 (NIPS 2017).* [arXiv:1705.02955](https://arxiv.org/abs/1705.02955)

5. **Moravčík, M., Schmid, M., Burch, N., Lisý, V., Morrill, D., Bard, N., Davis, T., Waugh, K., Johanson, M., Bowling, M.** (2017). "DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker." *Science, 356(6337):508-513.* [arXiv:1701.01724](https://arxiv.org/abs/1701.01724)

6. **Brown, N., Sandholm, T.** (2018). "Superhuman AI for Heads-Up No-Limit Poker: Libratus Beats Top Professionals." *Science, 359(6374):418-424.* [DOI:10.1126/science.aao1733](https://www.science.org/doi/10.1126/science.aao1733)

7. **Brown, N., Sandholm, T.** (2019). "Superhuman AI for Multiplayer Poker." *Science, 365(6456):885-890.* [DOI:10.1126/science.aay2400](https://www.science.org/doi/10.1126/science.aay2400)

8. **Steinberger, E., Lerer, A., Brown, N.** (2020). "DREAM: Deep Regret Minimization with Advantage Baselines and Model-free Learning." *arXiv preprint.* [arXiv:2006.10410](https://arxiv.org/abs/2006.10410)

9. **Brown, N., Bakhtin, A., Lerer, A., Gong, Q.** (2020). "Combining Deep Reinforcement Learning and Search for Imperfect-Information Games." *Advances in Neural Information Processing Systems 33 (NeurIPS 2020).* [arXiv:2007.13544](https://arxiv.org/abs/2007.13544)

10. **Kovařík, V., Schmid, M., Burch, N., Bowling, M., Lisý, V.** (2023). "Value Functions for Depth-Limited Solving in Zero-Sum Imperfect-Information Games." *Artificial Intelligence.* [arXiv:1906.06412](https://arxiv.org/abs/1906.06412)

11. **Schmid, M., Moravčík, M., Burch, N., Kadlec, R., Davidson, J., Waugh, K., Bard, N., Timbers, F., Lanctot, M., Holland, G.Z., Davoodi, E., Christianson, A., Bowling, M.** (2023). "Student of Games: A Unified Learning Algorithm for Both Perfect and Imperfect Information Games." *Science Advances, 9(46).* [arXiv:2112.03178](https://arxiv.org/abs/2112.03178)

12. **Li, B., Fang, Z., Huang, L.** (2024). "RL-CFR: Improving Action Abstraction for Imperfect Information Extensive-Form Games with Reinforcement Learning." *ICML 2024, PMLR 235:27752-27770.* [arXiv:2403.04344](https://arxiv.org/abs/2403.04344)

13. **Kubíček, O., Burch, N., Lisý, V.** (2024). "Look-Ahead Search on Top of Policy Networks in Imperfect Information Games." *IJCAI 2024.* [arXiv:2312.15220](https://arxiv.org/abs/2312.15220)

14. **Kubíček, O., Lisý, V.** (2025). "Look-Ahead Reasoning with a Learned Model in Imperfect Information Games." *arXiv preprint.* [arXiv:2510.05048](https://arxiv.org/abs/2510.05048)
