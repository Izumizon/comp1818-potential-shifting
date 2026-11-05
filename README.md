# COMP1818 potential-shifting investigation

**Final delivery after reviewer #9:** the reviewed methodology and original
experimental evidence are unchanged. Implementation and generated dataset links
are public and checked without owner authentication. Institutional AI-policy
and teaching-team topic/extension confirmation remain open. No Moodle submission
has been performed.

- Implementation: https://github.com/Izumizon/comp1818-potential-shifting
- Dataset: https://github.com/Izumizon/comp1818-potential-shifting/tree/main/data
- Saved results and models: the repository's `results/` directory.

`verification/public_access_r9.json` records unauthenticated access and byte
comparisons. `verification/REVIEWER_09_RECORD.md` distinguishes the reported
cross-environment review from same-host checks. Earlier archives are preserved;
historical records retain their original dates/counts and are not current claims
about publication status.

This is an AI-generated research project and report prepared under the expanded
AI permission reported by the user. It contains executed experiments, not a
student-authored implementation or a guaranteed high-mark submission. Read
`AI_DISCLOSURE.md` and the final verification report before submitting anything.

## What was implemented

- A configurable deterministic Gridworld with goal/step rewards, walls,
  explicit termination/truncation flags and position-only state.
- Tabular Q-learning and seeded epsilon-greedy control with uniform tie-breaking.
- Static potential shaping, the published non-terminal potential shift, two
  explicit exponential normalisations and separate cutoff protocols.
- A stationary graph-smoothing method modification and its shift/smoothing
  factorial ablation. No scientific novelty is claimed.
- Independent training/evaluation random streams, complete configurations,
  final Q-tables, raw evaluation episodes and deterministic analysis.
- Random-policy baselines and an explicitly informed BFS reference.

The exact author normalisation and cutoff implementation remain unresolved.
This project is a controlled reconstruction and extension, not a claim of exact
reproduction. See `PROTOCOL.md` for the precise formulas, scope and limitations.

## Set up and run (Python 3.12)

Extract the archive, open a terminal in its `COMP1818-project-r9` directory, and
first verify the delivered files with `python verify_package.py`. This uses only
the standard library. Use a fresh environment for experiments. On Windows:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pytest -q --junitxml=verification/new-tests.xml
.venv\Scripts\python.exe run.py --suite pilot
.venv\Scripts\python.exe benchmark.py
```

After inspecting the pilot cost, reproduce the frozen suites:

```powershell
.venv\Scripts\python.exe run.py --suite reconstruction
.venv\Scripts\python.exe run.py --suite core
.venv\Scripts\python.exe run.py --suite random
.venv\Scripts\python.exe export_data.py
.venv\Scripts\python.exe -m gridshift.analysis
.venv\Scripts\python.exe verify_results.py
.venv\Scripts\python.exe analyse_posthoc.py
```

Existing complete runs resume only when the engine fingerprint matches. To
actually regenerate evidence without replacing the delivered files, pass
`--output reproduced-results` to `run.py`; then call
`analyse('reproduced-results', 'reproduced-analysis')` from
`gridshift.analysis`. Do not delete or relabel old results when changing code.
`verify_results.py` deliberately re-executes six selected runs and retains those
reruns in a fresh numbered `verification/r8-reruns*` directory. It checks exact
configuration identity and discrete observations; floating arithmetic uses
explicit rtol=atol=1e-12. Per-array bit equality is reported separately. Optional
`--require-bit-exact` additionally requires byte identity. This is a stricter
host-specific diagnostic, not a cross-platform promise. The historical
`verification/reruns/` is unchanged. `analyse_posthoc.py` computes the reviewer-
requested difference-in-differences from saved outcomes, explicitly exploratory.

Numba JIT/cache loading is included in the first applicable call, not a claim
that every subsequent run recompiles. NumPy PCG64 draws are pregenerated in
separate streams; the readable and accelerated update paths are tested against
one another. Exact reruns were checked on this host. Floating-point and compiler
differences may prevent cross-platform bitwise equality. Independent review
reported exact discrete observations and four tolerance-only Q/potential
comparisons across environments; universal bit-exact reproducibility is not
claimed. Those separate reviewer logs were not supplied here.

## Evidence and file map

| File or directory | Purpose |
|---|---|
| `gridshift/environment.py` | Transition/reward model and explicit episode contract |
| `gridshift/learner.py` | Readable independent Q-update and shaping reference |
| `gridshift/potentials.py` | Both normalisations, corruption, smoothing, shift and diagnostics |
| `gridshift/engine.py` | Accelerated training/evaluation kernels |
| `gridshift/experiment.py` | Frozen configuration matrices, seeds and per-run recording |
| `tests/` | Executed behavioural and mathematical checks |
| `results/*/manifest.json` | Complete configuration lists, written before execution |
| `results/*/*.npz` | Actual evaluation outcomes, final models and potentials |
| `results/*/*.json` | Actual settings, metrics, timings and engine fingerprints |
| `data/` | Generated environment dataset and informed symbolic reference |
| `analysis/` | Recomputed run tables, confidence summaries and exported figures |
| `report/` | Official-class LaTeX source and compiled review PDF |
| `verification/` | Test records, pilot, raw-outcome audit and final verification |

No external downloaded training dataset is used. The simulator definitions,
transition/reward arrays, configs and seeds are the research dataset. Both maps
and final Q-tables are included. The supplied audio/XOR/RL teaching notebooks are
background material and are not falsely presented as this implementation.

## Read the result without overstating it

The primary outcome is mean checkpoint discounted original-task return within
the 250-step evaluation window. Confidence intervals resample ten paired seed
blocks after averaging the fixed factors inside each block. They do not turn
checkpoints or episodes into independent training runs. The graph extension
depends on known simulator adjacency and can damage useful guidance. Interpret
normalisation, initialisation and corruption separately using the full tables;
the pooled average is not a universal improvement claim.

BFS uses exact task knowledge and no training. Its role is to show the limits of
claiming practical benefit for RL on fully known tiny maps. It is not a fair
model-free learning-speed competitor. The source-style cutoff condition has a
separate limitation record and is excluded from stationary-invariance claims.

## Report and outstanding submission actions

The official `COMPXXXX.cls` and `.bst` are preserved unchanged in `report/`.
With a LaTeX installation, compile `report/main.tex` (BibTeX required, or use
Tectonic). From the project directory, generate the source with
`python build_report.py`, change directory to `report`, then run
`tectonic main.tex` or `latexmk -pdf main.tex`. The delivered PDF was compiled
with Tectonic 0.17.0. The body including AI reflection is checked against four pages;
references and the separate declaration follow. Figures are exported by
Matplotlib and tables/equations are typeset, not screenshots.

The report contains the genuine implementation/data links above. Build with
`python build_report.py --require-published`; this requires populated HTTPS
URLs. Run `python verify_publication.py --output verification/new-public-check.json`
to recheck access without owner authentication. Access can change after delivery;
check it again before submitting.

Confirm the reported AI-policy amendment and topic/method-extension scope with
the teaching team, then independently review and understand the entire work.
These confirmations remain open items in the final readiness checklist. No mark
prediction or submission-ready claim is made.
