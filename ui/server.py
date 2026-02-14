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
    NUM_DEALS,
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

    deal_labels = [get_deal_label(d) for d in ALL_DEALS]

    histories = ["", "c", "b", "cb", "cc", "bc", "bf", "cbc", "cbf"]
    terminal = {"cc", "bc", "bf", "cbc", "cbf"}

    result = {
        "deal_labels": deal_labels,
        "histories": [],
    }

    for h in histories:
        if h not in belief_data:
            continue

        belief = belief_data[h].tolist()
        reach_p0, reach_p1 = reach_data[h]

        entry = {
            "history": h or "(root)",
            "is_terminal": h in terminal,
            "player": len(h) % 2 if h and h not in terminal else None,
            "belief": [round(b, 6) for b in belief],
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

    terminal = {"cc", "bc", "bf", "cbc", "cbf"}

    def build_node(history):
        is_term = history in terminal
        player = len(history) % 2 if history and not is_term else None

        node = {
            "history": history or "(root)",
            "is_terminal": is_term,
            "player": player,
        }

        if history in beliefs:
            b = beliefs[history]
            node["belief"] = [round(x, 4) for x in b.tolist()]

        if is_term:
            # Add expected payoff weighted by belief
            if history in beliefs:
                game = KuhnPoker()
                ev_p0 = 0.0
                b = beliefs[history]
                for deal_idx, (c0, c1) in enumerate(ALL_DEALS):
                    state = game.next_state(game.initial_state(), (c0, c1))
                    for action in history:
                        state = game.next_state(state, action)
                    ev_p0 += b[deal_idx].item() * game.terminal_utility(state, 0)
                node["ev_p0"] = round(ev_p0, 4)
            return node

        # Add children
        if history in ("", "c"):
            actions = [("c", "Check"), ("b", "Bet")]
        elif history in ("b", "cb"):
            actions = [("c", "Call"), ("f", "Fold")]
        else:
            return node

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

        children = []
        for action, label in actions:
            child_history = history + action
            child = build_node(child_history)
            child["action"] = label
            child["action_code"] = action
            children.append(child)
        node["children"] = children

        return node

    tree = build_node("")
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
