"""Reviewer-requested exploratory interaction; uses saved data, no new training.

This analysis was proposed after viewing the primary results. It neither
replaces the frozen primary contrast nor supplies a confirmatory synergy test.
"""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
from gridshift.analysis import load_runs, core_key, interval, write_csv


def interaction(core, normalisation=None, q_initial=None):
    table = {core_key(r['config']): r['metrics']['mean_checkpoint_return'] for r in core}
    assert len(table) == len(core), 'duplicate factor keys'
    cells = []
    blocks = defaultdict(list)
    for key in sorted(table):
        map_name, norm, fraction, q0, seed, method = key
        if method != 'smooth_shifted':
            continue
        if normalisation is not None and norm != normalisation:
            continue
        if q_initial is not None and q0 != q_initial:
            continue
        get = lambda arm: table[(map_name, norm, fraction, q0, seed, arm)]
        shifted_gain = get('smooth_shifted')-get('shifted')
        unshifted_gain = get('smooth_pbrs')-get('pbrs')
        value = shifted_gain-unshifted_gain
        blocks[seed].append(value)
        cells.append({'map_name': map_name, 'normalisation': norm, 'corruption': fraction,
                      'q_initial': q0, 'seed': seed, 'smoothing_gain_shifted': shifted_gain,
                      'smoothing_gain_unshifted': unshifted_gain, 'difference_in_differences': value})
    assert set(blocks) == set(range(10))
    assert len({len(v) for v in blocks.values()}) == 1
    statistics = interval([np.mean(blocks[s]) for s in sorted(blocks)])
    statistics.update({'paired_quartets': len(cells), 'cells_per_seed': len(cells)//10})
    return statistics, cells


def main():
    core = [r for r in load_runs('results') if r['config']['phase'] == 'core']
    primary, cells = interaction(core)
    summary = json.loads(Path('analysis/summary.json').read_text())
    np.testing.assert_allclose(primary['mean'], summary['extension_primary']['mean']-
                               summary['smoothing_unshifted']['mean'], rtol=1e-12, atol=1e-12)
    result = {'status': 'post-hoc exploratory; reviewer requested after primary outcomes',
              'formula': '(smooth_shifted - shifted) - (smooth_pbrs - pbrs)',
              'weighting': 'equal fixed-factor cell means within seed, then ten paired seed blocks',
              'interval': '10000 percentile bootstrap resamples; descriptive, conditional on fixed tasks',
              'pooled': primary,
              'four_arm_summary': {arm: {
                  'runs': sum(r['config']['method'] == arm for r in core),
                  'mean_checkpoint_return': float(np.mean([r['metrics']['mean_checkpoint_return']
                      for r in core if r['config']['method'] == arm])),
                  'zero_final_success_runs': sum(r['metrics']['final_success'] == 0
                      for r in core if r['config']['method'] == arm)}
                  for arm in ('pbrs', 'shifted', 'smooth_pbrs', 'smooth_shifted')},
              'by_normalisation': {n: interaction(core, normalisation=n)[0] for n in ('literal', 'unit')},
              'by_initialisation': {str(q): interaction(core, q_initial=q)[0] for q in (-1., 0., 1.)}}
    Path('analysis/posthoc_interaction.json').write_text(json.dumps(result, indent=2))
    write_csv('analysis/posthoc_interaction_cells.csv', cells)
    write_csv('analysis/posthoc_interaction_seed_blocks.csv',
              [{'seed': seed, 'interaction': value} for seed, value in enumerate(primary['seed_block_values'])])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
