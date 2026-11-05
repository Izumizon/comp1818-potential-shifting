# Frozen protocol and source-fidelity record

This specification was written before the reconstruction/core experiment suites.
Pilot seed 9001 is reserved for execution-cost checks and excluded from reported
confirmatory run seeds. Pilot observations are retained; no condition will be
removed because its result is unfavourable. Configuration manifests are emitted
before execution and bind the runs to a SHA-256 fingerprint of the engine files.

## Source versus explicit reconstruction

Source: Müller and Kudenko (2025), full preprint arXiv:2502.01307v1. Tabular
settings: alpha .1, epsilon .05, gamma .95; initial Q values -1, 0, +1;
exponential base 32; original goal reward 1 and intermediate reward 0, or
alternative original reward -1 on every transition. Their grid is corner-to-
corner with a 250-step episode cap. The selected reconstruction uses 250,000
training transitions and ten epsilon-.05 evaluation episodes every 250 steps,
with five independent training seeds. No initial step-zero evaluation is added.

Unresolved: the full exponential/normalisation pipeline, some transition
conventions, exact cutoff bootstrap mask, action tie-breaking and seed list.
This implementation cannot claim exact reproduction or recovery of all
published results. The 25x25 size is explicit in the paper's motivating example;
its use for the main reconstruction is an assumption. The reconstruction uses
deterministic cardinal moves, boundary self-loops, no walls, seeded uniform
greedy tie-breaking, and a 250-step evaluation safety cap. The exact author
evaluation cap is not independently verified.

Let d be Manhattan distance to the goal, D its maximum over accessible states,
and h=1-d/D. Preserve both definitions as a sensitivity factor:

- literal: E=32**h on non-terminal states;
- unit: E=32**(h-1) on non-terminal states.

The second inserts an additional factor 1/32 and is not asserted to be the
authors' code. All goal and wall entries are zero. Corruption and smoothing
operate before shifting; terminal values are overridden again after shifting.
No hidden re-normalisation follows the shift.

For a non-terminal state, Psi=E+b/(gamma-1), where
b=(1-gamma)*Q_initial-r_intermediate. Shaped reward is the original reward plus
gamma*Psi(next)-Psi(current), with the boundary convention below. Original task
rewards are never scaled, and reported evaluation rewards are never shaped.

## Two cutoff conventions, not conflated

- `bootstrap`: the goal is true termination (zero next potential/value).
  A non-goal collection cutoff retains next potential and bootstraps. Resets
  collect another training segment; the target is the discounted goal-terminated
  task. Static-potential policy invariance applies to that underlying task under
  its usual assumptions, not to finite-budget learning or capped evaluation.
- `episodic_cutoff`: a non-goal cap forces zero endpoint potential and suppresses
  bootstrapping. This reconstructs the kind of cutoff treatment discussed in the
  paper, but the exact author update remains unverified. Because time is absent
  from the learner state, do not claim a stationary finite-horizon MDP or invoke
  the ordinary stationary policy-invariance theorem for this capped protocol.

Both return separate termination/truncation flags. Goal termination wins if
goal and collection cap coincide. The core study uses only `bootstrap`; the
second convention is confined to the labelled reconstruction sensitivity study.
No time-augmented learner is used or implied.

## Method modification: stationary local graph smoothing

The extension replaces E by S(E)=E-L*E/4, one synchronous Euler diffusion step
on the undirected non-terminal grid graph, before the same prescribed shift.
L is the combinatorial graph Laplacian. The strength is fixed at one, not tuned
on evaluation results. This operator preserves the non-terminal mean, cannot
expand its range and smooths local variation. It can also remove useful guidance
or create ties; no performance improvement is assumed. It is an implementation
extension using an established graph operation, not a claimed new algorithm.

The full simulator adjacency is supplied as construction-time side information
to every shaping arm. Smoothing does not use BFS distances, original reward
outcomes, training trajectories or evaluation data. Comparisons with unshaped
learning do not isolate the value of side information; the shifted-versus-
unshifted and smoothed-versus-unsmoothed comparisons do. No claim of parity with
EPBRS's symbolic-plan representations or its long-sequence tasks is made.

Corruption permutes a prespecified fraction (0, .25, .75) of non-terminal values.
It freezes the permutation for each run and preserves the value histogram. The
same corruption instance is used across methods, normalisations, Q initialisations
and cutoff protocols for each map and seed. Actual changed fraction is logged:
permutation fixed points mean it need not equal the selected fraction.

Directional consistency compares signs of potential changes against signs of
offline BFS-distance reductions on a fixed, uniformly weighted set of legal
non-terminal edges. Terminal edges, walls, self-loops and equal-distance edges
are excluded. Potential ties count separately. BFS is evaluator-only, not an
input to the learner or smoothing. Corruption changes smoothness as well as
consistency; do not interpret this as a universal scalar reliability intervention.

## Suites and prespecified outcomes

`pilot`: three seed-9001 runs, each 5,000 training steps, covering the 25x25
source-like schedule and a smoothed/corrupted 10x10 case. A separate cost profile
must justify the full workload before large suites run.

`reconstruction`: open25, both rewards, both cutoff rules, both normalisations,
Q_initial {-1,0,+1}, seeds 0..4, methods unshaped/pbrs/shifted. Unshaped runs are
not duplicated across irrelevant normalisation settings: 300 distinct runs.

`core`: open10 and detour10, goal reward, bootstrap cutoff, both normalisations,
three corruption fractions, all three initialisations, seeds 0..9, 50,000
training transitions. Evaluate ten greedy episodes every 1,000 training steps,
cap 250. Methods: pbrs, shifted, smooth_pbrs, smooth_shifted plus unshaped.
Unshaped is not duplicated over corruption/normalisation: 1,500 distinct runs.

Random policies: ten seeds, 100 episodes each, for each map and reward, saved
separately. They do not undergo fictitious training.

Primary per-run outcome: mean original discounted return across all scheduled
checkpoints and evaluation episodes (MCR). Also retain original undiscounted
returns, episode lengths, successes and cutoffs for every evaluation episode.
This is a capped observational metric, not an empirical proof of invariance.
No threshold-based post-hoc definition of convergence is introduced.

Primary extension contrast: smooth_shifted minus shifted MCR, paired within
seed/map/normalisation/corruption/initialisation. Summarise the complete fixed
factor set within each seed first, then quantify uncertainty over the ten
independent seed blocks. Secondary views report factors separately. Do not
treat checkpoints, evaluation episodes or factor cells as independent training
replicates. Report all directions of effect, failures and null conditions.

Seeds include both training and corruption variation: the block result estimates
variation over those joint random choices, not their separately identified
variance components. Two hand-designed maps do not support a population claim
over all mazes. Bootstrap intervals are descriptive for this fixed design;
no uncorrected battery of significance claims will be used.

## Scope and external dependencies

The empirical extension and method composition still need teaching-team scope
confirmation. The user reports an updated AI policy permitting full generation;
no separate official amendment was supplied. Preserve full AI disclosure and
confirm that reported policy before actual submission. These are submission
risks, not reasons to misrepresent authorship or stop reversible development.
