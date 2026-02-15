# rebel-poker

From-scratch implementation of the [ReBeL algorithm](https://arxiv.org/abs/2007.13544) (Brown et al., 2020) for imperfect-information games. Combines deep RL with search using **Public Belief States (PBS)** — probability distributions over hidden information conditioned on public history — as the abstraction for both CFR search and value network training.

**Roadmap:** svtuck/task-ingestion#88 (Kuhn → Leduc → HUNL)

## Architecture

Three-layer design with strict separation between game logic and solver logic:

### 1. Game-Agnostic Core
| File | Purpose |
|------|---------|
| `game_interface.py` | `Game` Protocol — any game implements this to work with all solvers |
| `belief_config.py` | `BeliefConfig` — deal enumeration, PBS ↔ reach conversion, value aggregation |
| `cfr/solver.py` | `CFRTrainer` — vanilla CFR, CFR+, DCFR, alternating updates |
| `cfr/vectorized.py` | `VectorizedCFR` — PyTorch-based CFR over all deals simultaneously |
| `cfr/mccfr.py` | External-sampling Monte Carlo CFR |
| `cfr/deep_cfr.py` | Neural network CFR (advantage + strategy networks) |

### 2. ReBeL Framework (`rebel/`)
| File | Purpose |
|------|---------|
| `rebel/rebel_trainer.py` | `RebelTrainer` — self-play CFR + value network training loop |
| `rebel/value_net.py` | `ValueNetwork`, `PolicyNetwork` — MLP architectures for PBS → values |
| `rebel/endgame_solver.py` | `SubgameSolver` — depth-limited subgame solving with optional value net at leaves |
| `rebel/data_logger.py` | `RebelDataLogger` — collects (PBS, values) pairs during CFR for training |
| `rebel/leduc_trainer.py` | Leduc-specific ReBeL trainer (imports from `leduc/` — see note below) |

### 3. Game Implementations
| Directory | Game | Status |
|-----------|------|--------|
| `kuhn/` | Kuhn Poker (3 cards, 2 players) | Working. Nash value = -1/18 |
| `leduc/` | Leduc Poker (6 cards, 3 ranks × 2 suits, 2 rounds) | Game logic works, ReBeL loop incomplete |
| `hunl/` | Heads-Up No-Limit Hold'em (uses Rust engine via PyO3) | Scaffolded, not trained |
| `river/` | River-only subgame solver (configurable bets/ranges) | Working |

## Key Abstractions

### Game Protocol (`game_interface.py`)

Structural subtyping — games don't inherit, they just implement the methods:

```python
class Game(Protocol):
    NUM_PLAYERS: int
    def initial_state() -> State
    def is_terminal(state) -> bool
    def current_player(state) -> Optional[int]  # 0/1 for players, -1 for chance, None for terminal
    def legal_actions(state) -> List[Action]
    def chance_outcomes(state) -> List[Tuple[Action, float]]
    def next_state(state, action) -> State
    def infoset_key(state, player) -> str
    def terminal_utility(state, player) -> float
```

### BeliefConfig (`belief_config.py`)

Game-agnostic PBS configuration. Captures deal structure and provides conversion utilities:

```python
@dataclass
class BeliefConfig:
    num_private_states: int      # Cards per player (3 for Kuhn, 6 for Leduc)
    all_deals: List[Tuple[int,...]]  # All possible card assignments
    chance_probs: torch.Tensor   # Prior probability over deals
```

Key methods: `initial_pbs()`, `reach_to_pbs()`, `pbs_to_reach()`, `deal_values_to_pbs_values()`, `pbs_values_to_deal_values()`.

## rebel/ Must Stay Game-Agnostic

The core files in `rebel/` (`rebel_trainer.py`, `value_net.py`, `endgame_solver.py`, `data_logger.py`) must **not** import from `kuhn/`, `leduc/`, or any game-specific module. They operate through the `Game` protocol and `BeliefConfig` only.

**Exception:** `rebel/leduc_trainer.py` is a game-specific trainer that imports from `leduc/`. This is acceptable as a game-specific entry point, but the pattern should be: game-specific trainers live in their game directory or are clearly marked as game-specific. New game-specific code should go in the game's directory (e.g., `kuhn/`, `leduc/`), not in `rebel/`.

## Open Issues

| Issue | Title | Status |
|-------|-------|--------|
| [#90](https://github.com/svtuck/task-ingestion/issues/90) | Integrate Leduc poker into unified pipeline | Open |
| [#91](https://github.com/svtuck/task-ingestion/issues/91) | Rust game engine with PyO3 bindings | Open |
| [#92](https://github.com/svtuck/task-ingestion/issues/92) | HUNL game engine + card/action abstraction | Open |
| [#93](https://github.com/svtuck/task-ingestion/issues/93) | GPU-accelerated CFR + Deep CFR | Open |
| [#94](https://github.com/svtuck/task-ingestion/issues/94) | ReBeL training loop for HUNL | Open |
| [#95](https://github.com/svtuck/task-ingestion/issues/95) | Evaluation framework — exploitability + head-to-head | Open |
| [#96](https://github.com/svtuck/task-ingestion/issues/96) | Poker solver optimization: Python-to-native perf gap | Open |
| [#98](https://github.com/svtuck/task-ingestion/issues/98) | Extend VectorizedCFR for mid-tree chance nodes | Open |

## Commands

```bash
# Install
pip install -e .

# Run tests (164 tests)
pytest tests/ -v

# Skip slow tests
pytest tests/ -m "not slow"

# Run specific test file
pytest tests/test_kuhn_cfr.py -v

# Solve Kuhn poker
python run_kuhn.py           # Default iterations
python run_kuhn.py 50000     # Custom

# Solve Leduc poker
python run_leduc.py              # Quick demo (1000 iters)
python run_leduc.py --benchmark  # Full benchmark (10K iters)
python run_leduc.py --rebel      # ReBeL training loop

# Web UI
python ui/server.py          # http://localhost:5050

# Benchmarks
python benchmarks/bench_cfr_solvers.py
python benchmarks/bench_river_cfr.py

# Rust poker engine (requires Rust toolchain)
cd poker-engine && maturin develop
```

## Dependencies

- Python >= 3.9
- `torch >= 2.0` — tensor ops, neural networks
- `flask >= 3.0` — web UI
- `pytest >= 7.0` — testing
- `maturin` + Rust toolchain — only for HUNL (PyO3 bindings)

## Directory Structure

```
rebel-poker/
├── game_interface.py          # Game Protocol
├── belief_config.py           # BeliefConfig abstraction
├── run_kuhn.py                # Kuhn solver entry point
├── run_leduc.py               # Leduc solver entry point
├── pyproject.toml             # Dependencies, pytest config
│
├── kuhn/                      # Kuhn Poker
│   ├── game.py                #   Game engine (3 cards, actions: c/b/f)
│   └── belief_state.py        #   PBS tracking (6 deals, 3 private states)
│
├── leduc/                     # Leduc Poker
│   ├── game.py                #   Game engine (6 cards, 2 rounds)
│   ├── belief_state.py        #   PBS tracking (30 private, 120 full deals)
│   └── card_abstraction.py    #   Hand strength bucketing
│
├── hunl/                      # Heads-Up No-Limit Hold'em
│   ├── game.py                #   Rust engine wrapper (PyO3)
│   ├── belief_state.py        #   1326 hole card combos
│   └── value_net.py           #   HUNL-specific network architecture
│
├── river/                     # River subgame solver
│   ├── game.py                #   Configurable board/stacks/bets
│   ├── cards.py               #   Card utilities
│   ├── vector_cfr.py          #   Vectorized solver
│   └── vector_eval.py         #   Hand evaluation
│
├── cfr/                       # CFR algorithms (game-agnostic)
│   ├── solver.py              #   CFRTrainer (vanilla, CFR+, DCFR)
│   ├── vectorized.py          #   VectorizedCFR (PyTorch)
│   ├── mccfr.py               #   External-sampling MCCFR
│   ├── batched_mccfr.py       #   Batched MCCFR
│   ├── deep_cfr.py            #   Neural network CFR
│   ├── fictitious_play.py     #   Fictitious play
│   └── matrix_cfr.py          #   Matrix-form CFR
│
├── rebel/                     # ReBeL framework (game-agnostic core)
│   ├── rebel_trainer.py       #   Main training loop
│   ├── value_net.py           #   ValueNetwork, PolicyNetwork
│   ├── endgame_solver.py      #   Depth-limited subgame solver
│   ├── data_logger.py         #   (PBS, values) data collection
│   └── leduc_trainer.py       #   Leduc-specific trainer (exception — imports leduc/)
│
├── sampling/                  # Marginalization research
│   ├── marginalization.py     #   Exact + sampling approaches
│   ├── large_benchmark.py     #   Scaling benchmarks
│   └── mlx_marginalization.py #   Apple Silicon GPU acceleration
│
├── ui/                        # Flask web UI (localhost:5050)
│   ├── server.py              #   API endpoints + server
│   └── templates/             #   HTML templates
│
├── benchmarks/                # Performance benchmarks
│   ├── bench_cfr_solvers.py
│   └── bench_river_cfr.py
│
├── tests/                     # 164 tests
│   ├── test_kuhn_cfr.py       #   CFR convergence, Nash properties
│   ├── test_belief_config.py  #   BeliefConfig conversions
│   ├── test_leduc.py          #   Leduc game logic
│   ├── test_marginalization.py
│   ├── test_sampling_marginalization.py
│   ├── test_vectorized_tracker.py
│   └── test_parity.py         #   Scalar vs vectorized CFR match
│
├── docs/                      # Documentation
│   ├── scaling-plan.md        #   Technical roadmap + implementation audit
│   ├── apple_silicon_gpu.md   #   GPU acceleration strategy
│   └── poker-solver-research.md
│
└── poker-engine/              # Rust poker engine (PyO3/maturin)
```

## Coding Conventions

- **Indentation:** 4 spaces
- **Type hints:** Used throughout — all function signatures have type annotations
- **Docstrings:** Google style (Args/Returns sections)
- **Naming:** PascalCase for classes, snake_case for functions/variables, UPPER_CASE for constants
- **Imports:** `from __future__ import annotations` at top, then stdlib → third-party → local
- **Data classes:** `@dataclass` for all configuration objects (`CFRConfig`, `SubgameConfig`, `RebelConfig`, `BeliefConfig`)
- **Protocols:** Structural subtyping via `typing.Protocol` — no base class inheritance for game interface

## Known Limitations

- PBS representation uses **joint deal distribution** in some places rather than **factored per-player beliefs** (issue #65, closed). The factored form (`[num_private_states, num_players]`) is needed for scaling beyond Kuhn.
- `endgame_solver.py` depth-limiting not fully implemented (`_is_leaf()` always returns `False`)
- PolicyNetwork defined but not integrated into the training loop
- No GPU acceleration yet — see `docs/apple_silicon_gpu.md` for MLX/MPS/Metal roadmap
- Kuhn-specific hardcoding remains in parts of `rebel_trainer.py` (terminal history checks)
