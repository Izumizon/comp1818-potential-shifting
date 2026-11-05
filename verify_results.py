"""Audit saved outcomes with numerical and exact identity contracts.

Only six selected runs are re-executed. Previously retained reruns are preserved.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from gridshift.analysis import load_runs
from gridshift.experiment import RunConfig, execute, fingerprint, run_id
from numerical_checks import (RTOL, ATOL, assert_identity, compare_arrays,
                              compare_metrics, compare_observations)


def main(rerun_directory=None, require_bit_exact=False):
    records = load_runs('results')
    assert len(records) == 1800
    expected = {}
    for phase in ('core', 'reconstruction'):
        manifest = json.loads((Path('results')/phase/'manifest.json').read_text())
        assert manifest['engine_sha256'] == fingerprint()
        for config in manifest['configurations']:
            rid = run_id(RunConfig(**config))
            assert rid not in expected, 'duplicate manifest run'
            expected[rid] = config
    assert {r['run_id'] for r in records} == expected.keys(), 'manifest/result identity mismatch'
    episode_count = 0
    actual_steps = 0
    for rec in records:
        c = rec['config']
        assert_identity(c, expected[rec['run_id']])
        assert rec['engine_sha256'] == fingerprint()
        with np.load(rec['_path']) as f:
            obs, q = f['observations'], f['q_final']
            assert np.all(np.isfinite(obs)) and np.all(np.isfinite(q))
            expected_shape = (c['steps']//c['eval_every'], c['eval_episodes'], 5)
            assert obs.shape == expected_shape
            compare_arrays(f['checkpoints'], np.arange(1, expected_shape[0]+1)*c['eval_every'], discrete=True)
            assert np.all((obs[..., 2] >= 1) & (obs[..., 2] <= 250))
            compare_arrays(obs[..., 1:], np.rint(obs[..., 1:]), discrete=True)
            assert np.all(np.isin(obs[..., 3:], [0, 1]))
            np.testing.assert_array_equal(obs[..., 3]+obs[..., 4], 1)
            if c['reward'] == 'goal':
                np.testing.assert_array_equal(obs[..., 1], obs[..., 3])
                compare_arrays(obs[..., 0], obs[..., 3]*c['gamma']**(obs[..., 2]-1))
            else:
                np.testing.assert_array_equal(obs[..., 1], -obs[..., 2])
                compare_arrays(obs[..., 0], -(1-c['gamma']**obs[..., 2])/(1-c['gamma']))
            compare_metrics(rec['metrics'], {
                **rec['metrics'], 'mean_checkpoint_return': float(obs[..., 0].mean()),
                'final_return': float(obs[-1, :, 0].mean()),
                'final_success': float(obs[-1, :, 3].mean()),
                'mean_success': float(obs[..., 3].mean()),
                'evaluation_steps': int(obs[..., 2].sum()),
                'training_steps': c['steps'], 'checkpoints': expected_shape[0],
                'evaluation_episodes': expected_shape[0]*expected_shape[1]})
            episode_count += obs.shape[0]*obs.shape[1]
            actual_steps += int(obs[..., 2].sum())
    selections = [next(r for r in records if r['config']['phase'] == 'core'
                       and r['config']['method'] == method and r['config']['seed'] == 3)
                  for method in ('unshaped', 'pbrs', 'shifted', 'smooth_pbrs', 'smooth_shifted')]
    selections.append(next(r for r in records if r['config']['phase'] == 'reconstruction'
                           and r['config']['cutoff'] == 'episodic_cutoff' and r['config']['reward'] == 'step'))
    if rerun_directory is None:
        # Repeated verification creates new evidence rather than replacing it.
        directory = Path('verification/r8-reruns')
        suffix = 2
        while directory.exists():
            directory = Path(f'verification/r8-reruns-{suffix}')
            suffix += 1
    else:
        directory = Path(rerun_directory)
    assert not directory.exists(), 'Preserve reruns: choose a new --rerun-directory'
    reruns = []
    for old in selections:
        new = execute(RunConfig(**old['config']), output=directory, resume=False)
        assert_identity(old['config'], new['config'])
        assert old['run_id'] == new['run_id'] and old['engine_sha256'] == new['engine_sha256']
        report = {'run_id': old['run_id'], 'metrics': compare_metrics(new['metrics'], old['metrics'])}
        if require_bit_exact:
            assert report['metrics']['bit_exact'], 'metric numeric agreement without bit identity'
        with np.load(old['_path']) as a, np.load(directory/(old['run_id']+'.npz')) as b:
            report['observations'] = compare_observations(b['observations'], a['observations'],
                                                         require_bit_exact=require_bit_exact)
            for key in ('q_final', 'potential', 'checkpoints'):
                report[key] = compare_arrays(b[key], a[key], discrete=key == 'checkpoints',
                                             require_bit_exact=require_bit_exact)
        reruns.append(report)
    # Zero bias means the same computation; preserve its strict bit-exact control.
    lookup = {tuple(sorted(r['config'].items())): r for r in records}
    zero_pairs = 0
    for r in records:
        c = r['config']
        if c['q_initial'] != 0 or c['reward'] != 'goal' or c['method'] not in ('pbrs', 'smooth_pbrs'):
            continue
        other = dict(c)
        other['method'] = 'shifted' if c['method'] == 'pbrs' else 'smooth_shifted'
        pair = lookup[tuple(sorted(other.items()))]
        with np.load(r['_path']) as a, np.load(pair['_path']) as b:
            for key in ('observations', 'q_final', 'potential', 'checkpoints'):
                compare_arrays(a[key], b[key], require_bit_exact=True)
        zero_pairs += 1
    result = {'checked_runs': len(records), 'checked_evaluation_episodes': episode_count,
              'checked_evaluation_transitions': actual_steps, 'fresh_deterministic_reruns': len(selections),
              'identical_zero_shift_pairs': zero_pairs, 'status': 'passed',
              'floating_tolerances': {'rtol': RTOL, 'atol': ATOL},
              'strict_contract': 'config/types, IDs, engine hashes, integer reward totals, lengths, flags, counts, checkpoints',
              'require_bit_exact_reruns': require_bit_exact, 'reruns': reruns,
              'scope': 'tight numerical agreement plus separately reported bit identity; not arbitrary correctness proof'}
    (directory/'audit.json').write_text(json.dumps(result, indent=2))
    Path('verification/results_check_r8.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun-directory', help='New output directory; by default choose a fresh numbered directory')
    parser.add_argument('--require-bit-exact', action='store_true')
    args = parser.parse_args()
    main(args.rerun_directory, args.require_bit_exact)
