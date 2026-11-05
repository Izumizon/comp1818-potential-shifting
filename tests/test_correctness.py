"""Behavioural, algebraic and independent-reference checks; no expected scores."""
import numpy as np
import pytest
from gridshift.environment import Gridworld, GridSpec, model, distances
from gridshift.learner import q_update, epsilon_greedy, shaping_reward
from gridshift.potentials import exponential, corrupt, smooth, shift, valid_nonterminal
from gridshift.engine import train, evaluate, choose
from gridshift.experiment import RunConfig, get_map, seed_for, suite_configs, run_id
from numerical_checks import compare_arrays, compare_observations


def test_reset_clears_episode_without_changing_spec():
    env = Gridworld()
    env.state, env.steps, env.terminated, env.truncated = 8, 4, True, True
    assert env.reset() == 0
    assert (env.steps, env.terminated, env.truncated) == (0, False, False)
    assert (env.spec.size, env.spec.max_steps) == (3, 4)


@pytest.mark.parametrize("action,expected", [(0, 0), (1, 1), (2, 3), (3, 0)])
def test_first_transition_and_boundaries(action, expected):
    env = Gridworld()
    assert env.step(action)[:4] == (expected, 0.0, False, False)
    assert env.steps == 1


def test_goal_on_cap_has_priority():
    env = Gridworld()
    for action in [1, 1, 2, 2]:
        result = env.step(action)
    assert result[:4] == (8, 1.0, True, False)
    with pytest.raises(RuntimeError): env.step(0)


def test_truncation_preserves_actual_state():
    env = Gridworld()
    for action in [1, 0, 0, 0]: result = env.step(action)
    assert result[:4] == (1, 0.0, False, True)
    with pytest.raises(RuntimeError): env.step(0)


@pytest.mark.parametrize("action", [-1, 4, 1.0, True, "right"])
def test_invalid_actions_do_not_change_state(action):
    env = Gridworld()
    with pytest.raises(ValueError): env.step(action)
    assert env.state == 0 and env.steps == 0


def test_walls_and_original_step_reward():
    spec = GridSpec(size=3, walls=(1,), reward="step", max_steps=10)
    env = Gridworld(spec)
    assert env.step(1)[:2] == (0, -1.0)
    for a in [2, 2, 1, 1]: result = env.step(a)
    assert result[:4] == (8, -1.0, True, False)
    assert distances(spec)[0] == 4


@pytest.mark.parametrize("terminal,truncated,cutoff,expected", [
    (False, False, "bootstrap", .436), (True, False, "bootstrap", .46),
    (False, True, "bootstrap", .436), (False, True, "episodic_cutoff", .36)])
def test_hand_updates(terminal, truncated, cutoff, expected):
    q = np.zeros((9, 4)); q[0, 1] = .4; q[1, 0] = .8
    before = q.copy()
    q_update(q, 0, 1, 1.0 if terminal else 0.0, 1, terminal, truncated, cutoff=cutoff)
    assert q[0, 1] == pytest.approx(expected)
    q[0, 1] = before[0, 1]
    np.testing.assert_array_equal(q, before)


def test_terminal_update_ignores_large_next_values_and_integer_q_rejected():
    q = np.full((9, 4), 999.0); q[0, 0] = .4
    assert q_update(q, 0, 0, 1.0, 8, True, False) == pytest.approx(.46)
    with pytest.raises(TypeError): q_update(np.zeros((9, 4), dtype=int), 0, 0, 0, 1, False, False)


def test_fractional_rewards_not_silently_truncated():
    q = np.zeros((9, 4))
    assert q_update(q, 0, 0, -.1, 1, False, False) == pytest.approx(-.01)


def test_epsilon_greedy_respects_unique_maximum_and_seed():
    assert epsilon_greedy([0, 3, 0, 0], 0, np.random.default_rng(3)) == 1
    a = np.random.default_rng(7); b = np.random.default_rng(7)
    assert [epsilon_greedy([0, 0, 0, 0], .05, a) for _ in range(30)] == [epsilon_greedy([0, 0, 0, 0], .05, b) for _ in range(30)]


def test_potential_factor_terminal_override_and_corruption_histogram():
    spec = GridSpec(size=5)
    large = exponential(spec, "literal"); unit = exponential(spec, "unit")
    valid = valid_nonterminal(spec)
    np.testing.assert_allclose(large[valid], 32*unit[valid], rtol=1e-12, atol=1e-12)
    assert large[spec.goal] == unit[spec.goal] == 0
    broken = corrupt(large, spec, .75, np.random.default_rng(1))
    np.testing.assert_array_equal(np.sort(broken[valid]), np.sort(large[valid]))
    assert not np.array_equal(broken, large)


@pytest.mark.parametrize("name", ["open10", "detour10"])
def test_graph_smoothing_is_synchronous_mean_preserving_and_bounded(name):
    spec = get_map(name)
    valid = valid_nonterminal(spec)
    phi = exponential(spec)
    phi = corrupt(phi, spec, .75, np.random.default_rng(4))
    before = phi.copy(); result = smooth(phi, spec)
    np.testing.assert_array_equal(phi, before)
    assert result[valid].mean() == pytest.approx(phi[valid].mean(), abs=1e-14)
    assert result[valid].min() >= phi[valid].min()-1e-14
    assert result[valid].max() <= phi[valid].max()+1e-14
    np.testing.assert_array_equal(smooth(phi, spec, 0), phi)
    assert result[spec.goal] == 0 and np.isfinite(distances(spec)[spec.start])


@pytest.mark.parametrize("q0", [-1, 0, 1])
def test_shift_identity_only_for_nonterminal_transitions(q0):
    spec = GridSpec(size=3)
    phi = exponential(spec)
    psi, b = shift(phi, spec, q0, .95, 0)
    assert psi[spec.goal] == 0
    assert (shaping_reward(psi, 0, 1, .95)-shaping_reward(phi, 0, 1, .95)) == pytest.approx(b)
    assert shaping_reward(psi, 7, 8, .95, True) == pytest.approx(-psi[7])


def test_telescoping_including_truncation_boundary():
    phi = np.array([.4, .8, .2, 0.0]); gamma = .9
    total = sum(gamma**i*shaping_reward(phi, i, i+1, gamma, i==2) for i in range(3))
    assert total == pytest.approx(-phi[0])
    partial = shaping_reward(phi, 0, 1, gamma, truncated=True)
    assert partial == pytest.approx(-phi[0]+gamma*phi[1])
    assert shaping_reward(phi, 0, 1, gamma, truncated=True, cutoff="episodic_cutoff") == pytest.approx(-phi[0])


def test_potential_initialisation_equivalence_transition_by_transition():
    spec = GridSpec(size=3, max_steps=100)
    nxt, rewards, terminal = model(spec)
    phi, _ = shift(exponential(spec), spec, 1.0, .95, 0)
    shaped = np.ones((9, 4)); shaped[8] = 0
    raw = shaped + phi[:, None]
    rng = np.random.default_rng(22)
    for _ in range(150):
        s = int(rng.integers(8)); a = int(rng.integers(4)); ns = int(nxt[s, a])
        r = rewards[s, a]; done = bool(terminal[ns])
        q_update(shaped, s, a, r+shaping_reward(phi, s, ns, .95, done), ns, done, False)
        q_update(raw, s, a, r, ns, done, False)
        np.testing.assert_allclose(raw, shaped+phi[:, None], rtol=1e-12, atol=1e-12)


def test_policy_invariance_with_independent_value_iteration():
    spec = GridSpec(size=3)
    nxt, rewards, terminal = model(spec); gamma = .95
    phi = np.array([10., -4., 2., 7., -3., 8., 9., .2, 0.])
    def solve(r):
        v = np.zeros(9)
        for _ in range(10000):
            q = r+gamma*v[nxt]*(~terminal[nxt])
            q[terminal] = 0
            new = q.max(axis=1)
            if np.max(np.abs(new-v)) < 1e-12: return q
            v = new
        raise AssertionError("oracle did not converge")
    raw = solve(rewards)
    shaped_rewards = rewards+gamma*phi[nxt]-phi[:, None]
    shaped_rewards[terminal] = 0
    shaped = solve(shaped_rewards)
    # Independent value iteration stops at 1e-12, so use an absolute solver
    # allowance rather than NumPy's implicit 1e-7 relative tolerance.
    np.testing.assert_allclose(shaped[:8], raw[:8]-phi[:8, None], rtol=0, atol=1e-10)
    for s in range(8):
        np.testing.assert_array_equal(np.isclose(raw[s], raw[s].max()), np.isclose(shaped[s], shaped[s].max()))


def test_evaluation_never_mutates_q_and_repeats_exactly():
    spec = GridSpec(size=3, max_steps=5)
    nxt, r, terminal = model(spec)
    q = np.random.default_rng(1).normal(size=(9, 4)); before = q.copy()
    u = np.random.default_rng(5).random((3, 5, 2))
    first = evaluate(q, nxt, r, terminal, 0, 5, .95, .05, u)
    np.testing.assert_array_equal(q, before)
    np.testing.assert_array_equal(first, evaluate(q, nxt, r, terminal, 0, 5, .95, .05, u))


@pytest.mark.parametrize("stop_on_cutoff", [False, True])
def test_compiled_training_matches_independent_readable_updates(stop_on_cutoff):
    spec = GridSpec(size=3, max_steps=5)
    nxt, r, terminal = model(spec)
    phi, _ = shift(exponential(spec), spec, -1, .95, 0)
    initial = np.full((9, 4), -1.0); initial[8] = 0
    u = np.random.default_rng(2).random((40, 2))
    eu = np.random.default_rng(3).random((4, 2, 5, 2))
    records, fast, _, _ = train(initial.copy(), nxt, r, terminal, phi, 0, 5, .1, .95,
                                .05, .05, 10, stop_on_cutoff, u, eu)
    slow = initial.copy(); s = 0; elapsed = 0; checkpoint = 0
    cutoff = "episodic_cutoff" if stop_on_cutoff else "bootstrap"
    for i in range(40):
        a = choose(slow[s], .05, u[i, 0], u[i, 1]); ns = int(nxt[s, a])
        elapsed += 1; term = bool(terminal[ns]); trunc = elapsed >= 5 and not term
        reward = r[s, a]+shaping_reward(phi, s, ns, .95, term, trunc, cutoff)
        q_update(slow, s, a, reward, ns, term, trunc, cutoff=cutoff)
        s = ns
        if term or trunc: s, elapsed = 0, 0
        if (i+1) % 10 == 0:
            expected = evaluate(slow, nxt, r, terminal, 0, 5, .95, .05, eu[checkpoint])
            compare_observations(records[checkpoint][None, ...], expected[None, ...]); checkpoint += 1
    # Independently expressed arithmetic may round differently; discrete
    # trajectories remain exact, while Q arithmetic gets the explicit tolerance.
    compare_arrays(fast, slow)


def test_evaluation_schedule_does_not_change_training():
    spec = GridSpec(size=3, max_steps=5); nxt, r, terminal = model(spec)
    u = np.random.default_rng(3).random((40, 2)); phi = exponential(spec)
    def run(every):
        return train(np.zeros((9, 4)), nxt, r, terminal, phi, 0, 5, .1, .95, .05,
                     .05, every, False, u, np.zeros((40//every, 2, 5, 2)))[1]
    np.testing.assert_array_equal(run(5), run(10))


def test_configuration_stream_roles_and_no_baseline_duplication():
    assert seed_for(0, "training", "open10") != seed_for(0, "evaluation", "open10")
    configs = suite_configs("core")
    assert len(configs) == len({run_id(c) for c in configs})
    with pytest.raises(ValueError): RunConfig(steps=501)
