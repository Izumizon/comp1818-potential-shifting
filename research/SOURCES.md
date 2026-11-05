# Research and source-fidelity ledger

These are genuine research sources used by the project. Bibliographic entries
are in `report/references.bib`. Mathematical claims are qualified in the report
and frozen protocol; linked sources are not endorsements of this implementation.

| Source | Role and boundary |
|---|---|
| [Müller & Kudenko (2025), full preprint](https://arxiv.org/abs/2502.01307) | Selected potential-shifting method and tabular settings. The combined exponential normalisation and exact author cutoff implementation remain ambiguous; this project retains both explicit scales and labels reconstructed details. The conference item is an extended abstract, not falsely cited as a full conference paper. |
| [Müller, Berg & Kudenko (2025)](https://doi.org/10.1007/s00521-024-10615-2) | Prior incomplete/incorrect symbolic-plan guidance. The present frozen scalar-field corruption and graph operation do not constitute a claim to have discovered imperfect-guidance shaping. |
| [Ng, Harada & Russell (1999)](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) | Potential-based reward transformation and policy invariance under its assumptions. No finite-budget or capped-metric invariance claim follows. |
| [Wiewiora (2003)](https://doi.org/10.1613/jair.1190) | Static shaping/Q-initialisation equivalence. Tested transition by transition; initialisation is retained as a comparison factor. |
| [Kondor & Lafferty (2002), author publication page](https://people.cs.uchicago.edu/~risi/papers/papers.html) | Established graph diffusion context. The project uses one Laplacian Euler step; it does not implement or claim a new diffusion kernel. |
| [Farama, Handling Time Limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/) | Modern continuing-task bootstrap convention. Kept distinct from the source-like zero-endpoint cutoff sensitivity condition. |

## Reconstructed versus contributed work

- Published component: potential shaping and prescribed constant nonterminal
  potential shift; source-linked hyperparameters are recorded in `PROTOCOL.md`.
- Reconstruction choices: exact grid transitions, action tie-breaking, seeds,
  two fully specified exponential pipelines and explicit cutoff implementations.
- Empirical extension: frozen value-permutation corruption with all baseline
  initialisations/scales retained and original-return evaluation.
- Method modification: one stationary synchronous graph-smoothing step before
  shifting, with the same shift, potential information and training budgets.
- No novel-algorithm claim, exact author-code reproduction claim or invented
  performance comparison with the imperfect-plan paper is made.

The implementation was generated for this project, not copied from author code
or presented as student-authored. `AI_DISCLOSURE.md` describes its provenance.
