"""Bridge between Python game representation and Rust CFR trainer.

Extracts the game tree and evaluation data from Python objects,
serializes them into flat arrays, and constructs a RustCFRTrainer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from river.game import Action, RiverHoldemGame, RiverState
from river.vector_cfr import VectorCFRConfig
from river.vector_eval import (
    StrengthSummary,
    action_tokens,
    build_blocked_indices,
    build_strength_summary,
)


def _build_game_tree(game: RiverHoldemGame):
    """Walk the game tree and serialize all nodes into flat arrays."""
    nodes_list = []
    node_map = {}

    infoset_map: Dict[int, Dict[str, int]] = {0: {}, 1: {}}
    infoset_keys: List[List[str]] = [[], []]
    infoset_actions: List[List[List[str]]] = [[], []]
    infoset_num_actions: List[List[int]] = [[], []]

    def get_infoset_idx(player: int, state: RiverState) -> int:
        key = game.infoset_key(state, player)
        if key in infoset_map[player]:
            return infoset_map[player][key]
        idx = len(infoset_map[player])
        infoset_map[player][key] = idx
        infoset_keys[player].append(key)
        actions = game.legal_actions(state)
        infoset_num_actions[player].append(len(actions))
        infoset_actions[player].append(action_tokens(actions))
        return idx

    def walk(state: RiverState) -> int:
        key = state.history
        if key in node_map:
            return node_map[key]

        idx = len(nodes_list)
        node_map[key] = idx
        nodes_list.append(None)

        if game.is_terminal(state):
            pot_total = game.pot_total(state)
            nodes_list[idx] = {
                "type": 0,
                "pot_total": float(pot_total),
                "contrib": (float(state.contrib[0]), float(state.contrib[1])),
                "terminal_winner": state.terminal_winner if state.terminal_winner is not None else -1,
                "player": -1,
                "children": [],
                "infoset_idx": -1,
            }
            return idx

        player = game.current_player(state)
        infoset_idx = get_infoset_idx(player, state)
        actions = game.legal_actions(state)
        children = []
        for action in actions:
            child_state = game.next_state(state, action)
            child_idx = walk(child_state)
            children.append(child_idx)

        nodes_list[idx] = {
            "type": 1,
            "pot_total": 0.0,
            "contrib": (0.0, 0.0),
            "terminal_winner": -1,
            "player": player,
            "children": children,
            "infoset_idx": infoset_idx,
        }
        return idx

    root = walk(game.initial_state())
    assert root == 0

    return {
        "nodes_list": nodes_list,
        "infoset_keys": infoset_keys,
        "infoset_actions": infoset_actions,
        "infoset_num_actions": infoset_num_actions,
    }


def _build_showdown_data(game: RiverHoldemGame):
    """Build showdown evaluation data for both players."""
    data = {}
    for player in (0, 1):
        opp_hands = game.hands[1 - player]
        opp_summary = build_strength_summary(opp_hands)

        sorted_indices = opp_summary.sorted_indices

        sorted_pos = [0] * len(opp_hands)
        for pos, orig_idx in enumerate(sorted_indices):
            sorted_pos[orig_idx] = pos

        hand_strength_range = []
        for hand in game.hands[player]:
            r = opp_summary.strength_ranges.get(hand.strength)
            if r is not None:
                hand_strength_range.append(r)
            else:
                lo, hi = 0, len(opp_summary.strengths_sorted)
                target = hand.strength
                while lo < hi:
                    mid = (lo + hi) // 2
                    if opp_summary.strengths_sorted[mid] < target:
                        lo = mid + 1
                    else:
                        hi = mid
                hand_strength_range.append((lo, lo))

        blocked = build_blocked_indices(game.hands[player], opp_summary)

        data[player] = {
            "sorted_indices": sorted_indices,
            "sorted_pos": sorted_pos,
            "hand_strength_range": hand_strength_range,
            "blocked": blocked,
        }

    return data


def create_rust_trainer(game: RiverHoldemGame, config: VectorCFRConfig):
    """Create a RustCFRTrainer from a Python game and config."""
    try:
        from rust_cfr import RustCFRTrainer
    except ImportError:
        raise ImportError(
            "rust_cfr not found. Build it with: "
            "cd rust_cfr && maturin develop --release"
        )

    tree_data = _build_game_tree(game)
    showdown_data = _build_showdown_data(game)

    nodes_list = tree_data["nodes_list"]
    game_data = {
        "num_hands": [len(game.hands[0]), len(game.hands[1])],
        "hand_weights_0": list(game.hand_weights[0]),
        "hand_weights_1": list(game.hand_weights[1]),
        "blocked_0": showdown_data[0]["blocked"],
        "blocked_1": showdown_data[1]["blocked"],
        "opp_sorted_0": showdown_data[0]["sorted_indices"],
        "opp_sorted_1": showdown_data[1]["sorted_indices"],
        "hand_strength_range_0": showdown_data[0]["hand_strength_range"],
        "hand_strength_range_1": showdown_data[1]["hand_strength_range"],
        "opp_sorted_pos_0": showdown_data[0]["sorted_pos"],
        "opp_sorted_pos_1": showdown_data[1]["sorted_pos"],
        "node_types": [n["type"] for n in nodes_list],
        "node_pot_totals": [n["pot_total"] for n in nodes_list],
        "node_contribs": [n["contrib"] for n in nodes_list],
        "node_terminal_winners": [n["terminal_winner"] for n in nodes_list],
        "node_players": [n["player"] for n in nodes_list],
        "node_children": [n["children"] for n in nodes_list],
        "node_infoset_idx": [n["infoset_idx"] for n in nodes_list],
        "num_infosets": [len(tree_data["infoset_num_actions"][0]), len(tree_data["infoset_num_actions"][1])],
        "infoset_num_actions": tree_data["infoset_num_actions"],
    }

    config_data = {
        "use_plus": config.use_plus,
        "linear_weighting": config.linear_weighting,
        "alternating": config.alternating,
        "use_dcfr": config.use_dcfr,
        "dcfr_alpha": config.dcfr_alpha,
        "dcfr_beta": config.dcfr_beta,
        "dcfr_gamma": config.dcfr_gamma,
    }

    trainer = RustCFRTrainer(game_data, config_data)
    return trainer, tree_data["infoset_keys"], tree_data["infoset_actions"]
