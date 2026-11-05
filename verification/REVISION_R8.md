# Reviewer #8 — technical revision

The original project and delivery ZIP are preserved. This revision changes the
verification/reporting layer, not the training engine, frozen configurations,
environment dataset or original experimental outcomes. No full matrix was rerun.

## Actual changes

- `numerical_checks.py`, `verify_results.py`: distinguish numerical agreement
  (rtol=atol=1e-12, explicit finite/shape/dtype checks) from byte identity. Exact
  checks remain for configuration values/types, IDs, engine hashes, checkpoint
  indices, counts, integer reward totals, lengths and termination/truncation
  flags. Fresh reruns record per-array bit identity and maximum absolute error.
  A `--require-bit-exact` diagnostic is available. Existing rerun directories
  cannot be overwritten; automatic numbered directories preserve new evidence.
- `tests/test_correctness.py`: independent accelerated/readable arithmetic uses
  explicit tolerances while episode outcomes stay exact. Remaining algebraic
  `allclose` calls no longer inherit NumPy's looser relative default. Exact
  repeated-evaluation/no-mutation and zero-shift negative controls remain strict.
- New verification-contract tests deliberately inject one-ULP roundoff, signed
  zero, meaningful numerical error, nonfinite values, dtype/shape differences,
  discrete changes and configuration changes. These establish what is accepted
  and rejected, rather than adjusting a threshold to hide an observed failure.
- `analyse_posthoc.py`, its tests and `analysis/posthoc_interaction*`: reproducible
  four-arm difference-in-differences with complete paired cells and seed-block
  weighting. It is labelled post-hoc and does not replace the frozen primary
  contrast. The original summaries and figures remain unchanged.
- `build_report.py` and `report/main.tex`: expanded source comparison, interaction
  equation/interpretation, complete four-arm means/failure counts, and limitations
  concerning cutoff diagnostics, metrics and task generalisation. The fourth
  counted page is used for critical discussion. The final counted page is
  balanced without changing the original class, margins, font size or spacing.

## Executed tests and audits

- **54/54 tests passed** in the final source-tree run (1.87 s), recorded in
  `tests_r8.xml`: the original 34 cases plus 20 new verification/interaction cases.
  A separate clean source copy also passed **54/54 in 6.40 s**, recorded in
  `clean-copy-tests_r8.xml` and `clean-copy-tests_r8.txt`.
- All **1,800 saved main runs**, **3,750,000 evaluation episodes** and
  **722,007,156 evaluation transitions** passed the revised audit, including
  strict manifest/configuration identity and explicit arithmetic tolerances.
- Six selected fresh reruns passed. Their observations, final Q, potentials and
  checkpoints were also **bit-exact on this host**, maximum absolute error zero.
  All metric reductions agreed bit-exactly too. No tolerance-only mismatch was
  observed in these selected runs; cross-platform equality remains unproven.
- All **260 zero-prescribed-shift pairs** retained strict bit equality, now also
  checked for potential/checkpoint arrays. `results_check_r8.json` and
  `r8-reruns/audit.json` retain the actual outcomes.

The audit reran six selected configurations only. It did not rerun, tune, remove
or replace the 1,800-run matrix. Inspection/validation scripts had one Windows
console encoding failure while printing a PDF ligature; ASCII-safe console
output corrected that utility. No experimental result was changed by it.

## Corrected interpretation

The primary smoothing-with-shift contrast remains **+0.0113357107**. Smoothing
without shifting changes MCR by **+0.0039768868**. The reviewer-requested
interaction is therefore:

`(smooth_shifted − shifted) − (smooth_pbrs − pbrs) = +0.0073588239`.

Across 360 matched quartets, factors are first averaged within each of ten seed
blocks. The descriptive 10,000-resample percentile interval is
**[+0.0044335153, +0.0109820012]**. It is **−0.0002454363** at literal scale,
**+0.0149630841** at unit scale, and exactly zero at Q_initial=0. These subsets
are exploratory; no multiplicity-adjusted significance or universal synergy is
claimed. The result estimates how the smoothing effect differs between shift
arms in this design. It neither proves shifting is necessary nor supplies
independent confirmation, since the same saved data were inspected again.

All four arms remain visible: mean MCR is 0.0313293 (PBRS), 0.0927626 (shifted),
0.0353062 (smoothed PBRS), and 0.1040983 (smoothed shifted). Runs with zero final-
checkpoint success are respectively **304, 201, 299 and 189 of 360**. These are
observed last-checkpoint failures, not proof of permanent non-convergence.

The report distinguishes qualitative consistency with initialisation/bias
dependence from empirical reconstruction under missing-detail assumptions.
Exact reproduction is not established. The paper's Figure 3 uses episode-length
curves/standard errors, while our primary statistic averages discounted return;
no invented author numerical baseline, speedup ratio or curve error is given.
Our step-reward runs mostly succeed and do not reproduce the source's reported
cutoff-failure pattern. Retained aggregate evaluations cannot diagnose its
oscillation mechanism. The source comparison is tied to arXiv:2502.01307v1,
Sections 6.1/6.2.1 and Figure 3, rechecked during this revision.

## Remaining hard gates and risks

The revised PDF remains a review document. Genuine accessible implementation
and dataset URLs are absent; `build_report.py --require-published` must reject
this state. Publication and teaching-team access verification remain manual.
The reported updated AI permission and acceptance of the method extension still
require confirmation. AI authorship/disclosure remains truthful and unchanged.

Potential normalisation/source cutoff ambiguity, two fixed maps, ten joint
training/corruption seed blocks, known-graph side information and capped metrics
still limit interpretation. Negative literal-scale and null/intermediate-
corruption results are retained. No marks are predicted and submission readiness
is not claimed.

`final_qa_r8.json` records the final page/format, gate and preservation checks;
the revised archive has its own SHA-256 manifest. Earlier verification reports
describe the preserved earlier delivery and should not be mistaken for this
revision's status.

Final PDF verification passed: **four counted pages including the 79-word AI
reflection**, followed by references and the declaration (six pages total).
All six rendered pages were inspected. The original class/bibliography style
are byte-identical to the supplied template; no overfull boxes or unresolved
references were found. The publication gate rejected the pending URLs as
expected. All 3,744 original delivered files and the original archive remain
unchanged; 3,712 original engine/evidence files also match in the revision.
