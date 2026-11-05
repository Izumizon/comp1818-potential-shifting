"""Readable reference updates, independent of the accelerated experiment loop."""
import numpy as np


def epsilon_greedy(values, epsilon, rng):
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    values = np.asarray(values)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("expected four finite action values")
    if rng.random() < epsilon:
        return int(rng.integers(4))
    ties = np.flatnonzero(values == values.max())
    return int(rng.choice(ties))


def q_update(q, state, action, reward, next_state, terminated, truncated,
             alpha=0.1, gamma=0.95, cutoff="bootstrap"):
    if not np.issubdtype(q.dtype, np.floating):
        raise TypeError("Q must have floating-point dtype")
    if q.ndim != 2 or q.shape[1] != 4 or not np.all(np.isfinite(q)):
        raise ValueError("invalid Q table")
    if not (0 < alpha <= 1 and 0 <= gamma < 1):
        raise ValueError("invalid alpha/gamma")
    if cutoff not in ("bootstrap", "episodic_cutoff"):
        raise ValueError("unknown cutoff protocol")
    if not (0 <= state < len(q) and 0 <= next_state < len(q) and 0 <= action < 4):
        raise ValueError("invalid state/action")
    if not np.isfinite(reward):
        raise ValueError("reward must be finite")
    stop = terminated or (truncated and cutoff == "episodic_cutoff")
    target = float(reward) if stop else float(reward) + gamma * q[next_state].max()
    q[state, action] += alpha * (target - q[state, action])
    return float(q[state, action])


def shaping_reward(phi, state, next_state, gamma, terminated=False,
                   truncated=False, cutoff="bootstrap"):
    if cutoff not in ("bootstrap", "episodic_cutoff"):
        raise ValueError("unknown cutoff protocol")
    stop = terminated or (truncated and cutoff == "episodic_cutoff")
    return gamma * (0.0 if stop else phi[next_state]) - phi[state]
