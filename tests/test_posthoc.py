"""Check the interaction sign, pairing and weighting against known synthetic cells."""
import pytest
from analyse_posthoc import interaction


def toy_records():
    # Two fixed cells per seed. Their interactions are +4 and -2, yielding +1.
    records = []
    for seed in range(10):
        for name, values in [('open10', [1., 3., 2., 8.]), ('detour10', [2., 3., 4., 3.])]:
            for method, value in zip(('pbrs', 'shifted', 'smooth_pbrs', 'smooth_shifted'), values):
                records.append({'config': {'map_name': name, 'normalisation': 'unit',
                    'corruption': 0., 'q_initial': 1., 'seed': seed, 'method': method},
                    'metrics': {'mean_checkpoint_return': value}})
    return records


def test_interaction_uses_complete_paired_quartets_and_seed_block_weighting():
    summary, cells = interaction(toy_records())
    assert summary['mean'] == summary['ci_low'] == summary['ci_high'] == 1.
    assert summary['paired_quartets'] == len(cells) == 20
    assert summary['n_seed_blocks'] == 10 and summary['cells_per_seed'] == 2


def test_interaction_rejects_missing_comparison_arm():
    with pytest.raises(KeyError):
        interaction(toy_records()[1:])


def test_interaction_rejects_duplicate_factor_cells():
    records = toy_records()
    with pytest.raises(AssertionError, match='duplicate'):
        interaction(records + [records[0]])
