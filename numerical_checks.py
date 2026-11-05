"""Numerical verification contracts, distinct from bit identity.

Float64 agreement: |actual-reference| <= 1e-12*|reference| + 1e-12.
This allows arithmetic/reduction roundoff, not statistical outcome differences.
Configuration identity and discrete outcomes remain exact. If near ties change
actions and episode outcomes, the reproducibility check still fails.
"""
import json
import numpy as np

RTOL = 1e-12
ATOL = 1e-12


def assert_identity(actual, expected):
    """Identity includes JSON types; config floats are never approximated."""
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True), 'identity mismatch'


def compare_arrays(actual, reference, *, discrete=False, require_bit_exact=False):
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert actual.shape == reference.shape, 'shape mismatch'
    assert actual.dtype == reference.dtype, 'dtype mismatch'
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)), 'nonfinite value'
    bit_exact = actual.tobytes(order='C') == reference.tobytes(order='C')
    if discrete:
        np.testing.assert_array_equal(actual, reference)
    else:
        np.testing.assert_allclose(actual, reference, rtol=RTOL, atol=ATOL, equal_nan=False)
    if require_bit_exact:
        assert bit_exact, 'numeric agreement without bit-exact identity'
    return {'bit_exact': bit_exact, 'comparison': 'exact discrete' if discrete else 'tolerance float64',
            'max_abs_error': float(np.max(np.abs(actual-reference))) if actual.size else 0.0}


def compare_observations(actual, reference, *, require_bit_exact=False):
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert actual.ndim == reference.ndim == 3 and actual.shape[-1] == reference.shape[-1] == 5
    # Undiscounted rewards are integer counts in both implemented tasks.
    discrete = compare_arrays(actual[..., 1:], reference[..., 1:], discrete=True,
                              require_bit_exact=require_bit_exact)
    floating = compare_arrays(actual[..., 0], reference[..., 0], require_bit_exact=require_bit_exact)
    return {'bit_exact': discrete['bit_exact'] and floating['bit_exact'],
            'discrete': discrete, 'discounted_return': floating}


def compare_metrics(actual, reference):
    assert actual.keys() == reference.keys(), 'metric names changed'
    counts = {'training_steps', 'evaluation_steps', 'checkpoints', 'evaluation_episodes',
              'training_goals', 'training_cutoffs'}
    bit_exact = True
    for name in actual:
        if name in counts:
            assert_identity(actual[name], reference[name])
        else:
            # Floating reductions of strictly checked success flags also get a tolerance.
            result = compare_arrays(np.asarray(actual[name]), np.asarray(reference[name]))
            bit_exact &= result['bit_exact']
    return {'bit_exact': bit_exact}
