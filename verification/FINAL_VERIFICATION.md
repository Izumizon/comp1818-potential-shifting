# Final technical handover — 4 November 2026

## Delivery status

Implementation, executed experiments, analysis, LaTeX and compiled review PDF are
complete locally. **This is a review package, not yet a submission-ready PDF:**
the required external implementation and dataset links have not been published.
`PUBLICATION.md` gives the exact remaining publication/build steps. No Moodle
submission or external publication has been performed.

The project was AI-generated under the updated permission explicitly reported
by the user. The original supplied brief contains a stricter policy; a separate
official amendment was not supplied. Confirm the reported amendment and the
method-extension scope with the teaching team before submission. The student
must independently inspect, understand and be able to defend the work. No mark
prediction or guarantee is made.

## Actual verification evidence

| Check | Executed outcome | Evidence |
|---|---|---|
| Behavioural/mathematical tests | 34 passed in 6.48 s | `tests.xml` |
| Fresh source-copy test execution | 34 passed in 5.53 s | `clean-copy-tests.xml`, `clean-copy-tests.txt` |
| Saved main-run audit | All 1,800 runs and 3,750,000 evaluation episodes passed arithmetic and fingerprint checks | `results_check.json`, `../verify_results.py` |
| Deterministic fresh reruns | Six reruns exactly reproduce observations, final Q, potentials and checkpoints on this host | `reruns/`, `../verify_results.py` |
| Prescribed zero-shift control | 260 paired comparisons have identical saved Q/outcome arrays | `results_check.json` |
| Informed BFS diagnostic | Six legal shortest paths agree with all 600 retained reference episodes | `final_qa.json`, `../data/` |
| Official template integrity | Class and bibliography style byte-identical to supplied ZIP | `final_qa.json` |
| PDF | Five A4 pages, selectable text, unencrypted; three counted report pages including reflection; references and declaration follow | `final_qa.json`, `../report/coursework-report-review.pdf` |
| Visual inspection | All five rendered pages inspected; no overlapping or clipped content | `final_qa.json` |
| Publication gate | `build_report.py --require-published` correctly rejects currently missing URLs without changing LaTeX | `final_qa.json` |

The test suite covers boundaries, walls, rewards, reset/episode flags, hand Q
updates, floating point values, seeded action selection, transformations,
shaping/shift identities, synchronous smoothing, initialisation equivalence,
tiny-MDP policy invariance, accelerated/reference agreement, and separation of
training/evaluation randomness. These checks are evidence within their tested
scope, not a proof of arbitrary correctness.

One final QA assertion initially failed because extracted PDF text contains an
`fl` ligature in “reflection”. Unicode normalisation corrected the assertion;
the report itself required no change. No experimental failures were concealed:
all failed learning conditions are included in the raw results and summaries.

## Executed workload and retained artefacts

A reserved-seed pilot preceded the frozen main matrices. Three pilot runs took
4.747 s of runner wall time; measured warm training/evaluation kernel rates and
the compute gate are retained in `MILESTONE-02.md` and `pilot_cost_profile.json`.

The reconstruction suite has 300 distinct configurations, each with 250,000
training transitions and 10 evaluation episodes every 250 transitions. The
core suite has 1,500 configurations, each with 50,000 training transitions and
10 episodes every 1,000 transitions. Every evaluation episode is capped at 250
steps. Main totals are **150,000,000 training transitions, 3,750,000 evaluation
episodes and 722,007,156 actual evaluation transitions**. Runner wall times were
33.667 s and 25.503 s; the suites ran concurrently, so these are not additive
measurements of a single serial execution. JIT, preparation and IO affect costs.

The archive includes complete per-run configurations, engine fingerprints,
observations, final Q-tables, potentials, checkpoints, generated environment
transition/reward arrays, random baselines, informed BFS reference, deterministic
analysis, exported figures, sources and tests. No external training dataset was
used. The frozen protocol and post-run BFS addendum distinguish planned primary
comparisons from later diagnostics. Cache files and local runtime installations
are excluded from the archive.

## What the evidence supports

The prespecified smoothed-shifted minus shifted contrast is +0.0113357 in mean
checkpoint discounted original-task return; its ten-seed-block percentile
bootstrap interval is [0.0069070, 0.0169621]. This is an absolute return change,
not an accuracy percentage. Literal-scale mean contrast is -0.0004181, with an
interval ending at zero; the unit-scale contrast is +0.0230895. The 25% selected
corruption interval spans zero. All factors and failure cases are retained.

These are conditional results for two fixed maps and joint training/corruption
seed variation. The study does not establish a universal improvement, a new
algorithm, separate corruption/training variance components, or real-world RL
benefit. BFS solves these fully known maps directly and has an acknowledged
information advantage.

## Source fidelity and material limits

The published shift is implemented separately from an explicitly reconstructed
potential pipeline. `32**h` and `32**(h-1)` differ by 32; both are retained without
rescaling original rewards. Author normalisation, some transition conventions,
exact cutoff mask, tie-breaking and seeds remain unresolved. Selected matches
or successful learning therefore do not establish exact reproduction.

States encode position only. The bootstrap protocol retains next potential and
value at an external cutoff; the source-like sensitivity protocol zeros both.
They are not described as interchangeable. Stationary policy-invariance claims
are limited to the underlying goal-terminated task. There is no time-augmented
625 × 251 learner, and no stationary finite-horizon claim for the capped,
position-only variant.

The extension is one fixed synchronous graph-smoothing step before the published
shift. It uses established mathematics and requires known simulator adjacency.
It can flatten useful guidance. Corruption changes spatial smoothness alongside
directional consistency, so it does not identify a universal reliability effect.

## Evidence against the official weighted rubric

This is an evidence map, not a prediction of marks or a claim that every
criterion is necessarily satisfied.

| Official category | Evidence available | Remaining weakness |
|---|---|---|
| Problem domain 20%: clarity/exactness 10%, logic/format 10% | Explicit question, related research, equations, declared assumptions; original class; short report including AI reflection | Exact source construction unresolved; student understanding must be demonstrated |
| Development 40%: components/parameters 10%, tuning/hybridisation/modification 10%, advanced features/implementations 10%, result quality/execution 10% | Controlled Q-initialisation/scale/cutoff comparisons, stationary method modification and factorial ablation, seeded pipeline, mathematical tests and actual saved runs | Strength fixed rather than optimised; modification is modest and its coursework suitability needs scope confirmation; no publication-level novelty claimed |
| Conclusion/review 40%: previous-work comparison 10%, coherent summary 15%, critical discussion 15% | Source-like sensitivity results, prior imperfect-plan work distinguished, conditional conclusion, negative/null cases, trivial and informed references, uncertainty and information limitations | No exact replication of author curves; small fixed task set limits external validity |

## Remaining student actions

1. Review the source, tests, protocol and full factor results; confirm the reported
   AI-policy amendment and method-extension scope with the teaching team.
2. Publish the implementation/dataset artefacts with accessible permissions.
3. Insert genuine URLs in `report/publication.json`, run the required-publication
   build and compile again using the unchanged template.
4. Recheck page count, link access, anonymity and declaration after that build.
5. Submit the single final PDF yourself under the applicable Moodle rules by
   18 November 2026, 17:00 UK time. Earlier preparation packs remain preserved.
