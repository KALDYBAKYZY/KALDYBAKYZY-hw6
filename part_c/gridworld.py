import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "data" / "maps.json").read_text())

ACTIONS = CONFIG["actions"]
DISCOUNT = CONFIG["discount"]
TILES = CONFIG["tiles"]
MOVES = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
PERPENDICULAR = {
    "up": ("left", "right"),
    "down": ("left", "right"),
    "left": ("up", "down"),
    "right": ("up", "down"),
}
ARROWS = {"up": "↑", "right": "→", "down": "↓", "left": "←"}


class GridWorld:
    def __init__(self, name, rows):
        self.name = name
        self.rows = rows
        self.height = len(rows)
        self.width = len(rows[0])
        self.states = [
            (row, col)
            for row in range(1, self.height + 1)
            for col in range(1, self.width + 1)
            if TILES[rows[row - 1][col - 1]].get("traversable", True)
        ]

    def tile(self, state):
        row, col = state
        return self.rows[row - 1][col - 1]

    def is_terminal(self, state):
        return TILES[self.tile(state)].get("terminal", False)

    def reward(self, state):
        return TILES[self.tile(state)]["reward"]

    def start(self):
        return next(state for state in self.states if self.tile(state) == "S")

    def move(self, state, direction):
        row_step, col_step = MOVES[direction]
        row, col = state[0] + row_step, state[1] + col_step
        inside = 1 <= row <= self.height and 1 <= col <= self.width
        if not inside or not TILES[self.rows[row - 1][col - 1]].get("traversable", True):
            return state
        return (row, col)

    def direction_probabilities(self, state, action):
        rule = TILES[self.tile(state)]
        if rule.get("movement") == "slippery":
            side_a, side_b = PERPENDICULAR[action]
            side = rule["perpendicular_each"]
            return [(action, rule["intended"]), (side_a, side), (side_b, side)]
        return [(action, 1.0)]

    def outcomes(self, state, action):
        if self.is_terminal(state):
            return []
        results = []
        for direction, probability in self.direction_probabilities(state, action):
            next_state = self.move(state, direction)
            results.append((probability, next_state, self.reward(next_state)))
        return results


def paper_world():
    return GridWorld("paper", CONFIG["paper_map"])


def extended_world():
    return GridWorld("extended", CONFIG["extended_map"])