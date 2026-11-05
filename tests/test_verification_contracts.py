"""Regression checks that distinguish roundoff from genuine reproducibility failures."""
import numpy as np
import pytest
from numerical_checks import assert_identity, compare_arrays, compare_metrics, compare_observations


def test_one_ulp_difference_is_numeric_agreement_but_not_bit_identity():
    a = np.array([1.0]); b = np.nextafter(a, np.inf)
    assert compare_arrays(a, b)['bit_exact'] is False
    with pytest.raises(AssertionError, match='bit-exact'):
        compare_arrays(a, b, require_bit_exact=True)


def test_meaningful_float_error_is_rejected():
    with pytest.raises(AssertionError):
        compare_arrays(np.array([1.00000001]), np.array([1.0]))


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_values_are_always_rejected(bad):
    with pytest.raises(AssertionError, match='nonfinite'):
        compare_arrays(np.array([bad]), np.array([bad]))


@pytest.mark.parametrize('other', [np.ones((1, 1)), np.ones(1, dtype=np.float32)])
def test_shape_and_dtype_are_strict(other):
    with pytest.raises(AssertionError):
        compare_arrays(np.ones(1, dtype=np.float64), other)


def test_signed_zero_numeric_equality_does_not_imply_bit_identity():
    assert compare_arrays(np.array([0.0]), np.array([-0.0]))['bit_exact'] is False


@pytest.mark.parametrize('column', [1, 2, 3, 4])
def test_discrete_observation_columns_do_not_get_float_tolerance(column):
    original = np.array([[[.95, 1., 2., 1., 0.]]])
    changed = original.copy(); changed[0, 0, column] += 1e-13
    with pytest.raises(AssertionError):
        compare_observations(changed, original)


def test_discounted_return_roundoff_is_separate_from_discrete_contract():
    original = np.array([[[.95, 1., 2., 1., 0.]]])
    changed = original.copy(); changed[0, 0, 0] = np.nextafter(.95, np.inf)
    result = compare_observations(changed, original)
    assert not result['bit_exact'] and result['discrete']['bit_exact']


@pytest.mark.parametrize('changed', [{'gamma': np.nextafter(.95, 1.)}, {'gamma': 1}])
def test_configuration_values_are_not_approximately_compared(changed):
    with pytest.raises(AssertionError, match='identity'):
        assert_identity(changed, {'gamma': .95})


def test_configuration_integer_float_types_are_not_collapsed():
    with pytest.raises(AssertionError, match='identity'):
        assert_identity({'seed': 1.0}, {'seed': 1})


def test_metric_counts_are_exact_and_metric_schema_is_strict():
    with pytest.raises(AssertionError):
        compare_metrics({'training_steps': 20}, {'training_steps': 21})
    with pytest.raises(AssertionError):
        compare_metrics({'mean_success': 0.0}, {})
