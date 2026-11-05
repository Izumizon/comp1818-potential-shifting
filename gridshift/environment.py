"""Deterministic position-only Gridworld with explicit reconstruction choices.

Actions: 0 up, 1 right, 2 down, 3 left. Invalid movement consumes a step.
Goal termination takes priority over a simultaneous collection cutoff.
The cutoff is an external truncation, not an observed finite-horizon task.
"""
from dataclasses import dataclass
from collections import deque
import numpy as np

DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass(frozen=True)
class GridSpec:
    size: int = 25
    start: int = 0
    goal: int | None = None
    walls: tuple[int, ...] = ()
    reward: str = "goal"
    max_steps: int = 250
    name: str = "open25"

    def __post_init__(self):
        if not isinstance(self.size, int) or self.size < 2:
            raise ValueError("size must be an integer >= 2")
        if self.goal is None:
            object.__setattr__(self, "goal", self.size**2 - 1)
        object.__setattr__(self, "walls", tuple(sorted(set(self.walls))))
        if self.reward not in ("goal", "step") or self.max_steps < 1:
            raise ValueError("invalid reward or collection cap")
        for state in (self.start, self.goal, *self.walls):
            if not isinstance(state, int) or not 0 <= state < self.size**2:
                raise ValueError("invalid state")
        if self.start == self.goal or self.start in self.walls or self.goal in self.walls:
            raise ValueError("start and goal must be distinct accessible states")


def model(spec: GridSpec):
    """Return next-state, original-reward and true-terminal arrays.

    Goal/wall rows are inert. Gridworld.step never acts from these rows.
    """
    count = spec.size**2
    nxt = np.empty((count, 4), dtype=np.int64)
    rewards = np.zeros((count, 4), dtype=np.float64)
    terminal = np.zeros(count, dtype=np.bool_)
    terminal[spec.goal] = True
    walls = set(spec.walls)
    for s in range(count):
        row, col = divmod(s, spec.size)
        for a, (dr, dc) in enumerate(DELTAS):
            target = s
            if s != spec.goal and s not in walls:
                nr, nc = row + dr, col + dc
                if 0 <= nr < spec.size and 0 <= nc < spec.size:
                    candidate = nr * spec.size + nc
                    if candidate not in walls:
                        target = candidate
                rewards[s, a] = (-1.0 if spec.reward == "step"
                                  else float(target == spec.goal))
            nxt[s, a] = target
    return nxt, rewards, terminal


def distances(spec: GridSpec):
    """Offline evaluator's BFS distances, never an agent update target."""
    nxt, _, _ = model(spec)
    dist = np.full(spec.size**2, np.inf)
    dist[spec.goal] = 0.0
    queue = deque([spec.goal])
    walls = set(spec.walls)
    while queue:
        s = queue.popleft()
        for target in nxt[s] if s != spec.goal else _goal_neighbours(spec):
            target = int(target)
            if target not in walls and not np.isfinite(dist[target]):
                dist[target] = dist[s] + 1
                queue.append(target)
    return dist


def _goal_neighbours(spec):
    row, col = divmod(spec.goal, spec.size)
    return [r * spec.size + c for dr, dc in DELTAS
            if 0 <= (r := row + dr) < spec.size and 0 <= (c := col + dc) < spec.size
            and r * spec.size + c not in spec.walls]


class Gridworld:
    def __init__(self, spec=None, *, size=3, max_steps=4):
        self.spec = spec if spec is not None else GridSpec(size=size, max_steps=max_steps,
                                                          name=f"open{size}")
        self.next_states, self.rewards, self.terminals = model(self.spec)
        self.reset()

    def reset(self):
        self.state = self.spec.start
        self.steps = 0
        self.terminated = False
        self.truncated = False
        return self.state

    def step(self, action):
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= action < 4:
            raise ValueError("action must be an integer in 0..3")
        if self.terminated or self.truncated:
            raise RuntimeError("episode ended; call reset")
        old = self.state
        self.state = int(self.next_states[old, action])
        self.steps += 1
        self.terminated = bool(self.terminals[self.state])
        self.truncated = self.steps >= self.spec.max_steps and not self.terminated
        return (self.state, float(self.rewards[old, action]), self.terminated,
                self.truncated, {"steps": self.steps})
