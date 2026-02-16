"""Slumbot API client for HUNL evaluation.

Slumbot (slumbot.com) is a public heads-up no-limit hold'em bot.
This client plays hands against Slumbot via its HTTP API and
collects statistics.

API endpoints:
  POST /api/new_hand  — start a new hand
  POST /api/act       — send an action

Action encoding:
  'k' = check, 'c' = call, 'f' = fold, 'bN' = bet/raise to N chips
  '/' separates streets

Game parameters:
  Blinds: 50/100, Stack: 20,000 chips
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

SLUMBOT_HOST = "slumbot.com"
NEW_HAND_URL = f"https://{SLUMBOT_HOST}/api/new_hand"
ACT_URL = f"https://{SLUMBOT_HOST}/api/act"
NUM_STREETS = 4
SMALL_BLIND = 50
BIG_BLIND = 100
STACK_SIZE = 20000


@dataclass
class HandResult:
    """Result of a single hand against Slumbot."""
    winnings: float
    client_pos: int
    hole_cards: List[str]
    board: List[str]
    action_sequence: str


@dataclass
class SlumbotMatchResult:
    """Aggregate result of playing N hands against Slumbot."""
    num_hands: int
    total_winnings: float
    mean_winnings: float
    std_winnings: float
    ci_95: Tuple[float, float]
    hands: List[HandResult] = field(repr=False, default_factory=list)

    def __str__(self) -> str:
        return (
            f"Slumbot Match ({self.num_hands} hands):\n"
            f"  Total: {self.total_winnings:+.0f} chips\n"
            f"  Mean:  {self.mean_winnings:+.2f} chips/hand\n"
            f"  Std:   {self.std_winnings:.2f}\n"
            f"  95% CI: [{self.ci_95[0]:+.2f}, {self.ci_95[1]:+.2f}]"
        )


def parse_action_sequence(action: str) -> List[Tuple[str, Optional[int]]]:
    """Parse Slumbot action string into a list of (action_type, amount) tuples.

    Examples:
        "cb300c" -> [('c', None), ('b', 300), ('c', None)]
        "kb200c/kk" -> [('k', None), ('b', 200), ('c', None), ('/', None), ('k', None), ('k', None)]
    """
    parsed = []
    i = 0
    while i < len(action):
        ch = action[i]
        if ch == 'b':
            j = i + 1
            while j < len(action) and action[j].isdigit():
                j += 1
            amount = int(action[i+1:j]) if j > i + 1 else 0
            parsed.append(('b', amount))
            i = j
        elif ch in ('k', 'c', 'f', '/'):
            parsed.append((ch, None))
            i += 1
        else:
            i += 1
    return parsed


class SlumbotClient:
    """HTTP client for playing against Slumbot.

    Requires the 'requests' library: pip install requests
    """

    def __init__(
        self,
        action_callback: Callable[[Dict[str, Any]], str],
        delay_between_hands: float = 0.5,
    ) -> None:
        if not HAS_REQUESTS:
            raise ImportError(
                "The 'requests' library is required for Slumbot API access. "
                "Install with: pip install requests"
            )
        self.action_callback = action_callback
        self.delay = delay_between_hands
        self.token: Optional[str] = None

    def new_hand(self) -> Dict[str, Any]:
        """Start a new hand."""
        data = {}
        if self.token:
            data["token"] = self.token
        response = requests.post(NEW_HAND_URL, json=data)
        response.raise_for_status()
        result = response.json()
        if "token" in result:
            self.token = result["token"]
        return result

    def act(self, action: str) -> Dict[str, Any]:
        """Send an action."""
        data = {"incr": action}
        if self.token:
            data["token"] = self.token
        response = requests.post(ACT_URL, json=data)
        response.raise_for_status()
        result = response.json()
        if "token" in result:
            self.token = result["token"]
        return result

    def play_hand(self) -> HandResult:
        """Play a single hand against Slumbot."""
        response = self.new_hand()
        client_pos = response.get("client_pos", 0)
        hole_cards = response.get("hole_cards", [])
        board = response.get("board", [])
        action_seq = response.get("action", "")

        while "winnings" not in response:
            hand_state = {
                "hole_cards": hole_cards,
                "board": response.get("board", board),
                "action": response.get("action", action_seq),
                "client_pos": client_pos,
            }
            action = self.action_callback(hand_state)
            response = self.act(action)
            if "board" in response:
                board = response["board"]
            if "action" in response:
                action_seq = response["action"]

        return HandResult(
            winnings=response["winnings"],
            client_pos=client_pos,
            hole_cards=hole_cards,
            board=board,
            action_sequence=action_seq,
        )

    def play_match(self, num_hands: int = 1000) -> SlumbotMatchResult:
        """Play multiple hands and compute statistics."""
        hands: List[HandResult] = []
        for i in range(num_hands):
            if i > 0:
                time.sleep(self.delay)
            hand = self.play_hand()
            hands.append(hand)

        winnings = [h.winnings for h in hands]
        total = sum(winnings)
        n = len(winnings)
        mean = total / n if n > 0 else 0.0

        if n > 1:
            var = sum((w - mean) ** 2 for w in winnings) / (n - 1)
            std = math.sqrt(var)
        else:
            std = 0.0

        se = std / math.sqrt(n) if n > 0 else 0.0
        ci = (mean - 1.96 * se, mean + 1.96 * se)

        return SlumbotMatchResult(
            num_hands=n,
            total_winnings=total,
            mean_winnings=mean,
            std_winnings=std,
            ci_95=ci,
            hands=hands,
        )


def always_call_callback(hand_state: Dict[str, Any]) -> str:
    """Simple callback that always calls or checks."""
    action = hand_state.get("action", "")
    parsed = parse_action_sequence(action)
    if parsed and parsed[-1][0] == 'b':
        return "c"
    return "k"


def always_fold_callback(hand_state: Dict[str, Any]) -> str:
    """Simple callback that always folds (or checks if no bet)."""
    action = hand_state.get("action", "")
    parsed = parse_action_sequence(action)
    if parsed and parsed[-1][0] == 'b':
        return "f"
    return "k"
