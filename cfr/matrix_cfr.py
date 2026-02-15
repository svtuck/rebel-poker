"""Matrix-based GPU-accelerated CFR.

Converts the game tree into sparse matrices and computes CFR iterations
using matrix-vector operations on GPU (PyTorch). Based on the GPU-CFR paper:
  https://arxiv.org/abs/2408.14778

Key idea: Instead of recursive tree traversal, represent the game tree as
sparse adjacency matrices and compute reach probabilities / counterfactual
values via sparse matrix-vector multiplication, level by level.

Supports PyTorch backends: CPU, MPS (Apple Silicon), CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from game_interface import Game


@dataclass
class MatrixCFRConfig:
    """Configuration for matrix-based CFR."""
    use_plus: bool = False       # CFR+ regret clipping
    use_dcfr: bool = False       # Discounted CFR
    dcfr_alpha: float = 1.5
    dcfr_beta: float = 0.0
    dcfr_gamma: float = 2.0


@dataclass(frozen=True)
class _NodeInfo:
    """Internal node metadata collected during tree enumeration."""
    state: object
    player: int           # -1 for chance, 0/1 for players, -2 for terminal
    depth: int
    infoset_key: str      # "" for chance/terminal
    actions: tuple         # legal actions at this node
    terminal_utility_p0: float  # only for terminal nodes
    chance_prob: float     # probability of chance action leading here


class GameTree:
    """Flattened game tree with sparse matrix representation.

    Enumerates all game tree nodes (one per deal × history), assigns each
    an integer index, and builds sparse matrices for level-by-level CFR.
    """

    def __init__(self, game: Game, device: str = "cpu") -> None:
        self.game = game
        self.device = torch.device(device)

        # Node storage
        self.nodes: List[_NodeInfo] = []
        self.node_index: Dict = {}  # state -> node_idx

        # Tree structure
        self.max_depth = 0
        self.num_nodes = 0

        # Infoset tracking
        self.infoset_to_id: Dict[str, int] = {}
        self.infoset_actions: Dict[int, Tuple] = {}  # infoset_id -> actions
        self.num_infosets = 0

        # Action-node mapping
        # For each node, which action indices (in its infoset) lead to which child nodes
        self.parent_child_pairs: List[Tuple[int, int, int, float]] = []
        # (parent_idx, child_idx, action_idx_in_infoset, chance_prob)

        # Build the tree
        self._enumerate(game)
        self._build_matrices()

    def _enumerate(self, game: Game) -> None:
        """BFS enumeration of the entire game tree."""
        root = game.initial_state()
        self._enumerate_recursive(root, 0, 1.0)

    def _enumerate_recursive(self, state, depth: int, chance_prob: float) -> int:
        """Recursively enumerate nodes. Returns node index."""
        game = self.game

        # Check if already visited
        if state in self.node_index:
            return self.node_index[state]

        node_idx = len(self.nodes)
        self.node_index[state] = node_idx

        if game.is_terminal(state):
            info = _NodeInfo(
                state=state, player=-2, depth=depth,
                infoset_key="", actions=(),
                terminal_utility_p0=game.terminal_utility(state, 0),
                chance_prob=chance_prob,
            )
            self.nodes.append(info)
            self.max_depth = max(self.max_depth, depth)
            return node_idx

        player = game.current_player(state)

        if player == -1:
            # Chance node
            info = _NodeInfo(
                state=state, player=-1, depth=depth,
                infoset_key="", actions=(),
                terminal_utility_p0=0.0, chance_prob=chance_prob,
            )
            self.nodes.append(info)
            for action, prob in game.chance_outcomes(state):
                child_state = game.next_state(state, action)
                child_idx = self._enumerate_recursive(child_state, depth + 1, prob)
                self.parent_child_pairs.append((node_idx, child_idx, 0, prob))
            self.max_depth = max(self.max_depth, depth)
            return node_idx

        # Player node
        actions = tuple(game.legal_actions(state))
        key = game.infoset_key(state, player)

        if key not in self.infoset_to_id:
            iset_id = len(self.infoset_to_id)
            self.infoset_to_id[key] = iset_id
            self.infoset_actions[iset_id] = actions
            self.num_infosets = iset_id + 1

        info = _NodeInfo(
            state=state, player=player, depth=depth,
            infoset_key=key, actions=actions,
            terminal_utility_p0=0.0, chance_prob=chance_prob,
        )
        self.nodes.append(info)

        for a_idx, action in enumerate(actions):
            child_state = game.next_state(state, action)
            child_idx = self._enumerate_recursive(child_state, depth + 1, 1.0)
            self.parent_child_pairs.append((node_idx, child_idx, a_idx, 1.0))

        self.max_depth = max(self.max_depth, depth)
        return node_idx

    def _build_matrices(self) -> None:
        """Build sparse matrices for level-by-level CFR operations."""
        self.num_nodes = len(self.nodes)
        dev = self.device

        # Node properties as tensors
        self.node_player = torch.tensor(
            [n.player for n in self.nodes], device=dev, dtype=torch.long
        )
        self.node_depth = torch.tensor(
            [n.depth for n in self.nodes], device=dev, dtype=torch.long
        )
        self.node_infoset = torch.tensor(
            [self.infoset_to_id.get(n.infoset_key, -1) for n in self.nodes],
            device=dev, dtype=torch.long,
        )
        self.terminal_values_p0 = torch.tensor(
            [n.terminal_utility_p0 for n in self.nodes], device=dev, dtype=torch.float32
        )
        self.is_terminal = (self.node_player == -2)
        self.is_chance = (self.node_player == -1)
        self.is_player = (self.node_player >= 0)

        # Group nodes by depth for level-by-level traversal
        self.depth_nodes: Dict[int, torch.Tensor] = {}
        for d in range(self.max_depth + 1):
            mask = self.node_depth == d
            self.depth_nodes[d] = mask.nonzero(as_tuple=True)[0]

        # Build parent-child relationship tensors
        # For each edge: parent_idx, child_idx, action_idx, prob
        if self.parent_child_pairs:
            parents = torch.tensor([p[0] for p in self.parent_child_pairs], device=dev, dtype=torch.long)
            children = torch.tensor([p[1] for p in self.parent_child_pairs], device=dev, dtype=torch.long)
            action_indices = torch.tensor([p[2] for p in self.parent_child_pairs], device=dev, dtype=torch.long)
            edge_probs = torch.tensor([p[3] for p in self.parent_child_pairs], device=dev, dtype=torch.float32)
        else:
            parents = torch.tensor([], device=dev, dtype=torch.long)
            children = torch.tensor([], device=dev, dtype=torch.long)
            action_indices = torch.tensor([], device=dev, dtype=torch.long)
            edge_probs = torch.tensor([], device=dev, dtype=torch.float32)

        self.edge_parents = parents
        self.edge_children = children
        self.edge_action_indices = action_indices
        self.edge_probs = edge_probs  # chance probs (1.0 for player actions)

        # Build per-depth edge groups for level-by-level ops
        edge_parent_depths = self.node_depth[parents] if len(parents) > 0 else torch.tensor([], device=dev, dtype=torch.long)
        self.depth_edges: Dict[int, torch.Tensor] = {}
        for d in range(self.max_depth + 1):
            mask = edge_parent_depths == d
            self.depth_edges[d] = mask.nonzero(as_tuple=True)[0]

        # Max actions across all infosets
        self.max_actions = max(
            (len(acts) for acts in self.infoset_actions.values()), default=1
        )

        # Build infoset -> node mapping for regret/strategy aggregation
        self._build_infoset_node_map()

    def _build_infoset_node_map(self) -> None:
        """Map each infoset to its member nodes."""
        dev = self.device
        self.infoset_node_lists: Dict[int, torch.Tensor] = {}
        for i in range(self.num_nodes):
            iset = self.node_infoset[i].item()
            if iset >= 0:
                self.infoset_node_lists.setdefault(iset, []).append(i)
        for iset in self.infoset_node_lists:
            self.infoset_node_lists[iset] = torch.tensor(
                self.infoset_node_lists[iset], device=dev, dtype=torch.long
            )


class MatrixCFR:
    """GPU-accelerated CFR using matrix operations.

    Instead of recursive tree traversal, this flattens the game tree into
    sparse matrices and performs CFR iterations via:
    1. Forward pass: compute reach probabilities level-by-level
    2. Backward pass: compute counterfactual values level-by-level
    3. Regret update: element-wise operations on GPU tensors
    """

    def __init__(self, game: Game, config: Optional[MatrixCFRConfig] = None,
                 device: str = "cpu") -> None:
        self.config = config or MatrixCFRConfig()
        self.device = torch.device(device)
        self.game = game

        # Build game tree
        self.tree = GameTree(game, device=device)

        # Per-infoset regret and strategy accumulators: [num_infosets, max_actions]
        n_isets = self.tree.num_infosets
        max_a = self.tree.max_actions
        dev = self.device

        self.regret_sum = torch.zeros(n_isets, max_a, device=dev)
        self.strategy_sum = torch.zeros(n_isets, max_a, device=dev)

        # Action mask: which actions are valid per infoset
        self.action_mask = torch.zeros(n_isets, max_a, device=dev, dtype=torch.bool)
        for iset_id, actions in self.tree.infoset_actions.items():
            self.action_mask[iset_id, :len(actions)] = True

        self.iteration = 0

    def _current_strategy(self) -> torch.Tensor:
        """Compute current strategy via regret matching.

        Returns: [num_infosets, max_actions] strategy tensor.
        """
        positives = torch.clamp(self.regret_sum, min=0.0)
        positives = positives * self.action_mask.float()
        totals = positives.sum(dim=1, keepdim=True)

        # Count valid actions per infoset for uniform fallback
        num_actions = self.action_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        uniform = self.action_mask.float() / num_actions

        # Use regret-matched where sum > 0, uniform otherwise
        has_positive = (totals > 0).float()
        strategy = has_positive * (positives / totals.clamp(min=1e-10)) + (1 - has_positive) * uniform
        return strategy

    def _average_strategy(self) -> torch.Tensor:
        """Compute average strategy from accumulated strategy sums.

        Returns: [num_infosets, max_actions] average strategy tensor.
        """
        strat = self.strategy_sum * self.action_mask.float()
        totals = strat.sum(dim=1, keepdim=True)
        num_actions = self.action_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        uniform = self.action_mask.float() / num_actions
        has_sum = (totals > 0).float()
        return has_sum * (strat / totals.clamp(min=1e-10)) + (1 - has_sum) * uniform

    def _iterate(self) -> None:
        """Run one CFR iteration using matrix operations."""
        self.iteration += 1
        tree = self.tree
        dev = self.device
        N = tree.num_nodes

        strategy = self._current_strategy()  # [num_infosets, max_actions]

        # Compute action probability for each edge
        # For player edges: look up strategy[infoset][action_idx]
        # For chance edges: use edge_probs
        edge_strategy = torch.ones(len(tree.edge_parents), device=dev)

        if len(tree.edge_parents) > 0:
            parent_infosets = tree.node_infoset[tree.edge_parents]
            parent_players = tree.node_player[tree.edge_parents]

            # Player action edges
            player_mask = parent_players >= 0
            if player_mask.any():
                player_edges = player_mask.nonzero(as_tuple=True)[0]
                isets = parent_infosets[player_edges]
                acts = tree.edge_action_indices[player_edges]
                edge_strategy[player_edges] = strategy[isets, acts]

            # Chance edges
            chance_mask = parent_players == -1
            if chance_mask.any():
                chance_edges = chance_mask.nonzero(as_tuple=True)[0]
                edge_strategy[chance_edges] = tree.edge_probs[chance_edges]

        # --- Forward pass: reach probabilities ---
        # reach_p0[node], reach_p1[node] = product of player's action probs along path
        reach_p0 = torch.zeros(N, device=dev)
        reach_p1 = torch.zeros(N, device=dev)
        reach_p0[0] = 1.0  # root
        reach_p1[0] = 1.0

        for d in range(tree.max_depth):
            edge_indices = tree.depth_edges.get(d)
            if edge_indices is None or len(edge_indices) == 0:
                continue

            parents = tree.edge_parents[edge_indices]
            children = tree.edge_children[edge_indices]
            probs = edge_strategy[edge_indices]
            parent_players = tree.node_player[parents]

            # P0 reach: multiply by action prob only if player 0 acts
            # P1 reach: multiply by action prob only if player 1 acts
            # Chance: multiply both reaches by chance prob
            p0_factor = torch.ones_like(probs)
            p1_factor = torch.ones_like(probs)

            p0_acts = parent_players == 0
            p1_acts = parent_players == 1
            chance_acts = parent_players == -1

            p0_factor[p0_acts] = probs[p0_acts]
            p0_factor[chance_acts] = probs[chance_acts]
            p1_factor[p1_acts] = probs[p1_acts]
            p1_factor[chance_acts] = probs[chance_acts]

            # Scatter to children (handle multiple parents mapping to same child)
            reach_p0.scatter_add_(0, children, reach_p0[parents] * p0_factor)
            reach_p1.scatter_add_(0, children, reach_p1[parents] * p1_factor)

        # --- Backward pass: counterfactual values ---
        # values[node] = expected value for player 0
        values = torch.zeros(N, device=dev)
        values[tree.is_terminal] = tree.terminal_values_p0[tree.is_terminal]

        for d in range(tree.max_depth, -1, -1):
            edge_indices = tree.depth_edges.get(d)
            if edge_indices is None or len(edge_indices) == 0:
                continue

            parents = tree.edge_parents[edge_indices]
            children = tree.edge_children[edge_indices]
            probs = edge_strategy[edge_indices]

            # Parent value = sum over children of (action_prob * child_value)
            values.scatter_add_(0, parents, probs * values[children])

        # --- Regret computation ---
        # For each player node and each action:
        #   instant_regret = opponent_reach * (action_value - node_value)
        if len(tree.edge_parents) > 0:
            parent_players = tree.node_player[tree.edge_parents]
            player_edge_mask = parent_players >= 0

            if player_edge_mask.any():
                pe = player_edge_mask.nonzero(as_tuple=True)[0]
                p_parents = tree.edge_parents[pe]
                p_children = tree.edge_children[pe]
                p_action_idx = tree.edge_action_indices[pe]
                p_infosets = tree.node_infoset[p_parents]
                p_players = tree.node_player[p_parents]

                # Action values = child values
                action_values = values[p_children]
                node_values = values[p_parents]

                # Opponent reach
                opp_reach = torch.where(
                    p_players == 0, reach_p1[p_parents], reach_p0[p_parents]
                )

                # Regret from player 0's perspective:
                # If player 0 acts: regret = opp_reach * (action_value - node_value)
                # If player 1 acts: regret = opp_reach * (node_value - action_value)
                # (because values are in P0's perspective)
                regret = torch.where(
                    p_players == 0,
                    opp_reach * (action_values - node_values),
                    opp_reach * (node_values - action_values),
                )

                # Accumulate regrets per infoset-action
                # Use scatter_add to aggregate across nodes in same infoset
                flat_idx = p_infosets * self.tree.max_actions + p_action_idx
                regret_flat = self.regret_sum.view(-1)

                if self.config.use_plus:
                    # For CFR+, first accumulate instant regrets in a temp buffer
                    temp_regret = torch.zeros_like(regret_flat)
                    temp_regret.scatter_add_(0, flat_idx, regret)
                    regret_flat += temp_regret
                    # Clip to non-negative
                    self.regret_sum.clamp_(min=0.0)
                else:
                    regret_flat.scatter_add_(0, flat_idx, regret)

                # Strategy accumulation
                player_reach = torch.where(
                    p_players == 0, reach_p0[p_parents], reach_p1[p_parents]
                )
                strat_weight = player_reach * edge_strategy[pe]

                strat_flat = self.strategy_sum.view(-1)
                strat_flat.scatter_add_(0, flat_idx, strat_weight)

        # DCFR discounting
        if self.config.use_dcfr:
            t = float(self.iteration)
            pos_base = t ** self.config.dcfr_alpha
            neg_base = t ** self.config.dcfr_beta
            pos_scale = pos_base / (pos_base + 1.0)
            neg_scale = neg_base / (neg_base + 1.0)
            strat_scale = (t / (t + 1.0)) ** self.config.dcfr_gamma

            pos_mask = self.regret_sum > 0
            neg_mask = self.regret_sum < 0
            self.regret_sum[pos_mask] *= pos_scale
            self.regret_sum[neg_mask] *= neg_scale
            self.strategy_sum *= strat_scale

    def run(self, iterations: int) -> None:
        """Run CFR for the given number of iterations."""
        for _ in range(iterations):
            self._iterate()

    def train(self, iterations: int) -> List[float]:
        """Run CFR and return exploitability at checkpoints."""
        exploitabilities = []
        for i in range(iterations):
            self._iterate()
            if (i + 1) % max(1, iterations // 50) == 0 or i < 10:
                exp = self.exploitability()
                exploitabilities.append(exp)
        return exploitabilities

    def average_strategy_profile(self) -> Dict[str, Dict[str, float]]:
        """Return the average strategy as {infoset_key: {action: prob}}."""
        avg = self._average_strategy()  # [num_infosets, max_actions]
        profile = {}

        # Reverse the infoset_to_id mapping
        id_to_key = {v: k for k, v in self.tree.infoset_to_id.items()}

        for iset_id in range(self.tree.num_infosets):
            key = id_to_key[iset_id]
            actions = self.tree.infoset_actions[iset_id]
            probs = avg[iset_id, :len(actions)]
            profile[key] = {a: probs[i].item() for i, a in enumerate(actions)}

        return profile

    def exploitability(self) -> float:
        """Compute exploitability of the current average strategy."""
        from cfr.solver import CFRTrainer
        profile = self.average_strategy_profile()
        dummy = CFRTrainer(self.game)
        br0 = dummy._best_response_value(profile, 0)
        br1 = dummy._best_response_value(profile, 1)
        return 0.5 * (br0 + br1)
