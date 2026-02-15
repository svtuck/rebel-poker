"""Flask server for the poker solver UI.

Provides a web interface for exploring Kuhn Poker CFR solutions,
public belief states, and strategy profiles.
"""

from __future__ import annotations

import json
import sys
import os

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kuhn.game import KuhnPoker, RANK_NAMES
from cfr.solver import CFRTrainer
from cfr.vectorized import VectorizedCFR
from kuhn.belief_state import (
    BeliefStateTracker,
    ALL_DEALS,
    CARD_TO_DEALS_P0,
    CARD_TO_DEALS_P1,
    NUM_DEALS,
    NUM_PRIVATE_STATES,
    NUM_PLAYERS,
)

app = Flask(__name__, template_folder="templates", static_folder="static")

# Global solver state
solver_state = {
    "trainer": None,
    "profile": None,
    "tracker": None,
    "iterations": 0,
}


def get_deal_label(deal):
    return f"{RANK_NAMES[deal[0]]}{RANK_NAMES[deal[1]]}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/solve", methods=["POST"])
def solve():
    """Run CFR for the specified number of iterations."""
    data = request.json or {}
    iterations = data.get("iterations", 10000)

    trainer = CFRTrainer(KuhnPoker())
    trainer.train(iterations)
    profile = trainer.average_strategy_profile()

    tracker = BeliefStateTracker()
    tracker.set_strategy_from_profile(profile)

    solver_state["trainer"] = trainer
    solver_state["profile"] = profile
    solver_state["tracker"] = tracker
    solver_state["iterations"] = iterations

    return jsonify({
        "status": "ok",
        "iterations": iterations,
        "exploitability": trainer.exploitability(),
    })


@app.route("/api/strategy")
def strategy():
    """Return the full strategy profile."""
    if solver_state["profile"] is None:
        return jsonify({"error": "No solution computed yet"}), 400

    profile = solver_state["profile"]
    result = {}
    for key in sorted(profile.keys()):
        card, history = key.split("|")
        player = len(history) % 2 if history != "" else 0

        # Determine action labels
        if history in ("", "c"):
            action_labels = {"c": "Check", "b": "Bet"}
        else:
            action_labels = {"c": "Call", "f": "Fold"}

        result[key] = {
            "card": card,
            "history": history or "(root)",
            "player": player,
            "actions": {
                action_labels.get(a, a): round(p, 4)
                for a, p in profile[key].items()
            },
            "raw_actions": profile[key],
        }

    return jsonify(result)


@app.route("/api/beliefs")
def beliefs():
    """Return belief states for all histories."""
    if solver_state["tracker"] is None:
        return jsonify({"error": "No solution computed yet"}), 400

    tracker = solver_state["tracker"]
    belief_data = tracker.compute_belief_states()
    reach_data = tracker.compute_all_reach_probs()

    card_labels = [RANK_NAMES[c] for c in range(NUM_PRIVATE_STATES)]
    deal_labels = [get_deal_label(d) for d in ALL_DEALS]

    histories = ["", "c", "b", "cb", "cc", "bc", "bf", "cbc", "cbf"]
    terminal = {"cc", "bc", "bf", "cbc", "cbf"}

    result = {
        "card_labels": card_labels,
        "deal_labels": deal_labels,
        "histories": [],
    }

    for h in histories:
        if h not in belief_data:
            continue

        pbs = belief_data[h]  # [NUM_PRIVATE_STATES, NUM_PLAYERS]
        reach_p0, reach_p1 = reach_data[h]

        entry = {
            "history": h or "(root)",
            "is_terminal": h in terminal,
            "player": len(h) % 2 if h and h not in terminal else None,
            "pbs": {
                "p0": [round(pbs[c, 0].item(), 6) for c in range(NUM_PRIVATE_STATES)],
                "p1": [round(pbs[c, 1].item(), 6) for c in range(NUM_PRIVATE_STATES)],
            },
            "reach_p0": [round(r, 6) for r in reach_p0.tolist()],
            "reach_p1": [round(r, 6) for r in reach_p1.tolist()],
        }

        # For terminal nodes, add payoffs
        if h in terminal:
            payoffs = []
            game = KuhnPoker()
            for deal_idx, (c0, c1) in enumerate(ALL_DEALS):
                state = game.next_state(game.initial_state(), (c0, c1))
                for action in h:
                    state = game.next_state(state, action)
                payoffs.append({
                    "p0": game.terminal_utility(state, 0),
                    "p1": game.terminal_utility(state, 1),
                })
            entry["payoffs"] = payoffs

        result["histories"].append(entry)

    return jsonify(result)


@app.route("/api/tree")
def game_tree():
    """Return the game tree structure for visualization."""
    if solver_state["profile"] is None:
        return jsonify({"error": "No solution computed yet"}), 400

    profile = solver_state["profile"]
    tracker = solver_state["tracker"]
    beliefs = tracker.compute_belief_states()
    reach_data = tracker.compute_all_reach_probs()

    terminal = {"cc", "bc", "bf", "cbc", "cbf"}
    game = KuhnPoker()

    # Precompute terminal utilities for all deals at all terminal histories
    terminal_utils = {}  # history -> [NUM_DEALS] utilities from P0's perspective
    for h in terminal:
        utils = []
        for deal_idx, (c0, c1) in enumerate(ALL_DEALS):
            state = game.next_state(game.initial_state(), (c0, c1))
            for action in h:
                state = game.next_state(state, action)
            utils.append(game.terminal_utility(state, 0))
        terminal_utils[h] = utils

    def build_node(history):
        """Build tree node. Returns (node_dict, deal_evs) where deal_evs[d] is
        the EV from P0's perspective for deal d at this node."""
        is_term = history in terminal
        player = None if is_term else len(history) % 2

        node = {
            "history": history or "(root)",
            "is_terminal": is_term,
            "player": player,
        }

        # Add PBS
        if history in beliefs:
            pbs = beliefs[history]
            node["pbs"] = {
                "p0": [round(pbs[c, 0].item(), 4) for c in range(NUM_PRIVATE_STATES)],
                "p1": [round(pbs[c, 1].item(), 4) for c in range(NUM_PRIVATE_STATES)],
            }

        # Add deal probabilities (joint, from reach probs)
        if history in reach_data:
            reach_p0, reach_p1 = reach_data[history]
            chance = 1.0 / NUM_DEALS
            joint = [(reach_p0[d].item() * reach_p1[d].item() * chance)
                     for d in range(NUM_DEALS)]
            total = sum(joint)
            if total > 0:
                deal_probs = [(d, joint[d] / total) for d in range(NUM_DEALS)]
                # Sort by probability descending
                deal_probs.sort(key=lambda x: -x[1])
                node["deal_probs"] = [
                    {"deal": get_deal_label(ALL_DEALS[d]), "prob": round(p, 6)}
                    for d, p in deal_probs if p > 1e-8
                ]

        if is_term:
            deal_evs = terminal_utils[history]
            # Overall EV weighted by joint probability
            if history in beliefs:
                pbs = beliefs[history]
                ev_p0 = 0.0
                for deal_idx, (c0, c1) in enumerate(ALL_DEALS):
                    joint_prob = pbs[c0, 0].item() * pbs[c1, 1].item()
                    ev_p0 += joint_prob * deal_evs[deal_idx]
                node["ev_p0"] = round(ev_p0, 4)
            return node, deal_evs

        # Determine actions
        if history in ("", "c"):
            actions = [("c", "Check"), ("b", "Bet")]
        elif history in ("b", "cb"):
            actions = [("c", "Call"), ("f", "Fold")]
        else:
            return node, [0.0] * NUM_DEALS

        # Add strategy info for this node
        strategies = {}
        for card in range(3):
            card_name = RANK_NAMES[card]
            key = f"{card_name}|{history}"
            if key in profile:
                strategies[card_name] = {
                    a_label: round(profile[key].get(a, 0), 4)
                    for a, a_label in actions
                }
        node["strategies"] = strategies

        # Build children and compute deal-level EVs
        children = []
        child_deal_evs = {}  # action_code -> [NUM_DEALS] evs
        for action, label in actions:
            child_history = history + action
            child, child_evs = build_node(child_history)
            child["action"] = label
            child["action_code"] = action
            children.append(child)
            child_deal_evs[action] = child_evs
        node["children"] = children

        # Compute per-deal EV at this decision node:
        # EV(deal d) = sum_a strategy(card_of_acting_player(d), a) * child_EV(a, d)
        card_to_deals = CARD_TO_DEALS_P0 if player == 0 else CARD_TO_DEALS_P1

        deal_evs = [0.0] * NUM_DEALS
        for card in range(3):
            card_name = RANK_NAMES[card]
            key = f"{card_name}|{history}"
            strat = profile.get(key, {})
            for deal_idx in card_to_deals[card]:
                ev = 0.0
                for action_code, _ in actions:
                    action_prob = strat.get(action_code, 1.0 / len(actions))
                    ev += action_prob * child_deal_evs[action_code][deal_idx]
                deal_evs[deal_idx] = ev

        # Compute per-hand EVs for both players
        # For each player, for each card they could hold, average the deal EVs
        # weighted by conditional probability of opponent's card
        hand_evs = {"p0": {}, "p1": {}}
        if history in reach_data:
            reach_p0, reach_p1 = reach_data[history]
            chance = 1.0 / NUM_DEALS
            for card in range(3):
                card_name = RANK_NAMES[card]
                # P0 holding this card: deals where c0 == card
                p0_deals = CARD_TO_DEALS_P0[card]
                weight_sum = 0.0
                ev_sum = 0.0
                for d in p0_deals:
                    w = reach_p0[d].item() * reach_p1[d].item() * chance
                    weight_sum += w
                    ev_sum += w * deal_evs[d]
                if weight_sum > 1e-10:
                    hand_evs["p0"][card_name] = round(ev_sum / weight_sum, 4)

                # P1 holding this card: deals where c1 == card
                p1_deals = CARD_TO_DEALS_P1[card]
                weight_sum = 0.0
                ev_sum = 0.0
                for d in p1_deals:
                    w = reach_p0[d].item() * reach_p1[d].item() * chance
                    weight_sum += w
                    # P1's EV is negative of P0's EV (zero-sum)
                    ev_sum += w * (-deal_evs[d])
                if weight_sum > 1e-10:
                    hand_evs["p1"][card_name] = round(ev_sum / weight_sum, 4)

        node["hand_evs"] = hand_evs

        return node, deal_evs

    tree, _ = build_node("")
    return jsonify(tree)


@app.route("/api/info")
def info():
    """Return general info about the current solution."""
    if solver_state["trainer"] is None:
        return jsonify({"solved": False})

    return jsonify({
        "solved": True,
        "iterations": solver_state["iterations"],
        "exploitability": solver_state["trainer"].exploitability(),
        "game_value_p0": -1.0 / 18.0,
        "num_infosets": len(solver_state["profile"]),
    })


if __name__ == "__main__":
    # Auto-solve on startup
    print("Solving Kuhn Poker (10000 iterations)...")
    trainer = CFRTrainer(KuhnPoker())
    trainer.train(10000)
    profile = trainer.average_strategy_profile()
    tracker = BeliefStateTracker()
    tracker.set_strategy_from_profile(profile)
    solver_state["trainer"] = trainer
    solver_state["profile"] = profile
    solver_state["tracker"] = tracker
    solver_state["iterations"] = 10000
    print(f"Done. Exploitability: {trainer.exploitability():.6f}")

    app.run(debug=True, port=5050)
