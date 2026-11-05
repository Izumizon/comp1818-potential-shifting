"""Fast loop and a Python reference use the same pregenerated PCG64 streams.

Evaluation has separate fixed streams, local state and no Q updates. It cannot
consume training randomness. Numba changes execution speed, not the algorithm.
"""
import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def wrap(fn):
            fn.py_func = fn
            return fn
        return wrap


@njit(cache=True)
def choose(values, epsilon, exploration_u, choice_u):
    if exploration_u < epsilon:
        return min(3, int(choice_u * 4))
    maximum = values[0]
    for a in range(1, 4):
        maximum = max(maximum, values[a])
    count = 0
    for a in range(4):
        count += int(values[a] == maximum)
    rank = min(count - 1, int(choice_u * count))
    for a in range(4):
        if values[a] == maximum:
            if rank == 0:
                return a
            rank -= 1
    return 0


@njit(cache=True)
def evaluate(q, nxt, rewards, terminal, start, cap, gamma, epsilon, uniforms):
    # Columns: discounted original return, original return, steps, success, cutoff.
    output = np.zeros((uniforms.shape[0], 5), dtype=np.float64)
    for episode in range(uniforms.shape[0]):
        state = start
        discount = 1.0
        for step in range(cap):
            action = choose(q[state], epsilon, uniforms[episode, step, 0],
                            uniforms[episode, step, 1])
            ns = nxt[state, action]
            reward = rewards[state, action]
            output[episode, 0] += discount * reward
            output[episode, 1] += reward
            discount *= gamma
            output[episode, 2] = step + 1
            state = ns
            if terminal[ns]:
                output[episode, 3] = 1.0
                break
        if not terminal[state]:
            output[episode, 4] = 1.0
    return output


@njit(cache=True)
def train(q, nxt, rewards, terminal, phi, start, cap, alpha, gamma,
          epsilon, eval_epsilon, eval_every, stop_on_cutoff,
          training_uniforms, evaluation_uniforms):
    results = np.zeros((evaluation_uniforms.shape[0], evaluation_uniforms.shape[1], 5))
    state, elapsed, checkpoint = start, 0, 0
    goals, cutoffs = 0, 0
    for step in range(training_uniforms.shape[0]):
        action = choose(q[state], epsilon, training_uniforms[step, 0],
                        training_uniforms[step, 1])
        ns = nxt[state, action]
        elapsed += 1
        terminated = terminal[ns]
        truncated = elapsed >= cap and not terminated
        stop = terminated or (truncated and stop_on_cutoff)
        future_phi = 0.0 if stop else phi[ns]
        shaped = rewards[state, action] + gamma * future_phi - phi[state]
        future_q = 0.0
        if not stop:
            future_q = q[ns, 0]
            for a in range(1, 4):
                future_q = max(future_q, q[ns, a])
        q[state, action] += alpha * (shaped + gamma * future_q - q[state, action])
        state = ns
        if terminated or truncated:
            goals += int(terminated)
            cutoffs += int(truncated)
            state, elapsed = start, 0
        if (step + 1) % eval_every == 0:
            results[checkpoint] = evaluate(q, nxt, rewards, terminal, start, cap,
                                           gamma, eval_epsilon, evaluation_uniforms[checkpoint])
            checkpoint += 1
    return results, q, goals, cutoffs
