# Milestone 2 pilot gate

Three reserved-seed pilot runs executed successfully in 4.747 seconds of runner
wall time. Their raw observations, configurations, hashes and timings are in
`results/pilot/`. The last configuration triggered another Numba signature
compilation; pilot wall time must not be read as a steady-state step rate.

The separate executed cost profile (`pilot_cost_profile.json`) measured:

- 500,000 training-only transitions in 0.0069798 seconds;
- 1,000,000 evaluation-only transitions in 0.0190066 seconds;
- warmup/cache loading 0.41946 seconds.

At those measured kernel rates, the maximum step-count projections are about
15.30 seconds for the 300-run reconstruction suite and 4.61 seconds for the
1,500-run core suite. These projections exclude preparation, PCG64 random-array
generation, metadata/NPZ IO, imports and system contention. Actual suite times
will be recorded separately. Evaluation costs are included in both estimates.

The bounded compute gate is passed. Proceed with the frozen complete matrices,
not a selection based on pilot success. The pilot seed (9001) is absent from all
reported reconstruction/core seeds. No performance conclusion is drawn from
the pilot. Existing mathematical/source-fidelity limitations remain unchanged.
