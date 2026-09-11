# A stochastic parameter recovery benchmark starts in the repository as six frozen published problems scored in simulations, because the algorithm work it measures is here and nothing measures it yet (issue #663, small version)

## Status

Accepted and implemented (2026-09-11). The "small version" issue #663 asks for first: enough
problems and a scoring protocol to start scoring the stochastic-fitting work of #659, #660,
#661 and #662, while the full suite of twenty to thirty problems is assembled.

## The problem

For fitting differential equation models there is an accepted benchmark collection (PEtab,
Hass and colleagues). For fitting stochastic rule-based models there is nothing: no agreed
problems, no reference answers, no scoring protocol. Four open issues claim that some change
makes stochastic fits better, and none of those claims is measurable. The pieces of a benchmark
exist here, in the recovery test tier (`tests/recovery_harness.py`, `tests/test_recovery.py`),
but that tier is aimed at differential equation models and is wired as a test, not as
something a result can be reported against.

A stochastic benchmark can be better than the ODE one: the true parameters are chosen, the
data are simulated from them, and the score is whether a fit gets them back. The right answer
is known exactly.

## The decision

### Where it lives, for now

`benchmarks/stochastic_recovery/`, next to the sampler benchmarks. The issue is right that the
finished suite should not live inside one of the tools it scores, and the directory is built
to leave: `protocol.py` (definitions, scoring, aggregation) imports nothing from PyBNF, the
problem directories are data, and `harness.py` depends only on the installed `pybnf` package.
It copies the inline dask doubles from `tests/integration_harness.py` rather than importing
them, so nothing under `tests/` is needed. The public home is
[wshlavacek/stochbench](https://github.com/wshlavacek/stochbench), seeded the same day with
the same six problems and the same `protocol.py`, laid out after the PEtab benchmark
collection; PyBNF keeps its copy of the problems for its tests and runner, and the runner is
PyBNF's own.

### Every parameter is marked identifiable, on measured leverage

The definition's `identifiable` flag exists so that a parameter the data cannot determine is
reported but not scored. Whether the data determine a parameter is measured, not guessed:
`harness.leverage` scores the truth several times to get the objective's noise there, then each
parameter alone at half and double its true value, and reports the shift in units of that
noise. Every parameter of the six problems moves the objective by more than its noise in at
least one direction, most by hundreds of standard deviations, so all are scored. The baseline
then shows which of them the methods miss, and those are the parameters whose leverage is ten
to fifty standard deviations next to siblings at a thousand: narrow directions of a steep
landscape, an optimizer's problem and exactly what the benchmark should expose. Flags are not
revised to match results; a flag changes only if the leverage measurement says the data do not
see the parameter.

### What a problem is

A directory with a frozen `problem.json` (format version 1), the model it names, and the data
file it names. The JSON records the true value, search bounds and identifiability of every free
parameter; the simulate method, suffix and sampling grid; the observables; how many replicates
the data average and the replicate-index offset they were drawn at; the sigma floor; the
simulation budget a fit gets; and the replicates per evaluation the baseline methods use. Six
problems, all from `BNGL-Models/models`, all published with a citation, chosen for a range of
size (three to six free parameters), noise (single-molecule promoters to hundreds of
receptors), dynamics (transients, noise-driven switching, noise-driven cycles) and simulator
(five SSA, one NFsim):

| id | model | free | method |
|---|---|---:|---|
| Shahrezaei_PNAS2008 | three-stage gene expression, Shahrezaei and Swain 2008 | 4 | ssa |
| Lin_PhysRevE2016 | bursty positive autoregulation, Lin and Doering 2016 | 4 | ssa |
| McKane_PhysRevLett2005 | demographic-noise predator-prey cycles, McKane and Newman 2005 | 4 | ssa |
| Hlavacek_PNAS2001 | kinetic proofreading in receptor signaling, Hlavacek et al. 2001 | 4 | ssa |
| Munsky_Science2012 | three two-state promoters, Munsky et al. 2012 | 6 | ssa |
| Yang_PhysRevE2008 | trivalent-ligand bivalent-receptor aggregation, Yang et al. 2008 | 3 | nf |

Each model is the library model with the free parameters bound through PyBNF's
`name name__FREE` alias form, the observables that carry no information dropped, and the
published stationary-distribution protocols replaced by a transient window from a fixed
initial state, since a fit to time-course means needs a transient. Every adaptation is
written in the model header and the JSON.

### The data are committed, and drawn where no fit can reach

The data file holds, at each sampling time, the mean over the replicates and a `_SD` column
holding the standard deviation across them, floored at five percent of the observable's peak
mean so a point every trajectory agrees on (the initial condition) does not get infinite
weight. The file is committed. It is regenerable from the definition and the simulator, and a
test checks that regeneration is deterministic, but the committed file is the benchmark:
should the simulator's random stream change, the problem does not.

Under PyBNF's default `stochastic_seed = auto` a trajectory's seed comes from the parameter
values and the replicate index, so a fit evaluating the true parameters draws replicates 0, 1,
2, ... of exactly the process the data came from, and would reproduce the data's own
trajectories instead of drawing fresh ones. The data are drawn at replicate indices from one
million up, past anything a fit reaches (a fit's indices stay below a few hundred, including
the confirmation stage's).

### Scoring

Written in `protocol.py`'s docstring and applied by its functions. Error is
`|log10(estimate / true)|` per parameter, in decades, since every parameter is searched on a
log scale; a fit's error is the largest over the parameters marked identifiable. Success at the
loose tolerance is a factor of two on every identifiable parameter and is the headline; the
tight tolerance is 26 percent. Cost is simulations, not evaluations, and every simulation
counts: the search, the end-of-fit confirmation that decides a stochastic fit's answer (#659),
and the replicates PyBNF runs for its information criteria. Simulations-to-success is read off
a trace of the fit's reported best against simulations spent, and reported only for a fit whose
final answer is within the loose tolerance, because the reported best can enter the tolerance
and leave it again. Every (problem, method) pair runs from several fit seeds; the success rate
over seeds is the primary statistic.

### The budget is enforced by the runner, in simulations

Every method gets the same simulation budget per problem, so the comparison is at equal cost.
The runner counts every job the run loop executes and returns `'STOP'` from the algorithm's
decision hook once the budget is spent; `max_iterations` is set past any budget and DE's
convergence stop is turned off, so the budget is the binding stop for every method alike. The
inline client's futures are lazy (a job runs when its result is consumed, not when it is
submitted), or a whole generation would run before the check could stop anything. The
end-of-fit confirmation (ten candidates, ten replicates) runs after the stop and is counted;
it is part of the protocol because it is how PyBNF decides a stochastic fit's answer.

### The baseline methods

Differential evolution, scatter search with and without its noise handling (#660), and CMA-ES
with and without its uncertainty handling (#661). Each pair differs only in the one toggle, so
the difference between them is that feature's measured effect, which is what #663 exists to
provide. No simplex refine: a simplex on a noisy objective is not a polish.

## Consequences

* The claims of #659, #660 and #661 can be measured on six problems from today. The baseline
  results are committed next to the problems (`results/baseline_v1.json` and its table in the
  README) as the reference every later change compares against. The first baseline already
  says something: CMA-ES's uncertainty handling (#661) raises the success rate on two problems
  and costs nothing on the other four, while scatter search's noise handling (#660) helps on
  one problem and appears to hurt on two, at the edge of what five seeds resolve. No method
  exceeds 80 percent on any problem, and two problems defeat every method at the frozen
  budget; the failures land on the narrow directions of the objective (a parameter whose
  leverage is ten to fifty noise standard deviations where its siblings' is a thousand), not on
  parameters the data cannot see.
* Every record's trace carries the per-parameter errors of the reported best as it changed, so
  a results file can be scored again (`protocol.rescore`) should a definition's identifiability
  flags be revised, without running anything.
* A fit of an SSA problem costs two to four minutes on one core at the frozen budget, the
  NFsim problem fifteen to twenty-five; the whole baseline (six problems, five methods, five
  seeds) takes under two hours on eight cores. Small enough to rerun for an algorithm change.
* The default test tier checks that the frozen definitions, the models and the committed data
  agree with each other and that the scoring functions do what the protocol says, without a
  backend. The `recovery` tier checks, through the real backend, that every identifiable
  parameter has leverage on the frozen objective, that data generation is deterministic, and
  that a budgeted fit produces a complete record whose counts add up.
* Not decided here, on purpose: the full problem set, methods that are not ours, and a
  permanent identifier. Those are the rest of #663.

## Alternatives considered

* **Single-trajectory data.** A fit to one noisy trajectory is a different, harder question
  (which trajectory?) than recovering the parameters that generated a population; replicate
  means with replicate spread are what PyBNF's `smoothing` and `chi_sq` are built for, and
  what the published stochastic fitting jobs in `BNGL-Models/pybnf-jobs` do.
* **Sigma as the standard error of the mean.** Scaling every sigma by the same factor changes
  no ranking, and the fit's replicate count is the method's choice, not the problem's. The
  trajectory spread is a property of the process and is frozen with the data.
* **Budget in evaluations.** A method that runs more replicates per parameter set would look
  cheaper than it is. Simulations are what the machine pays for.
* **Living in `tests/`.** A test tier cannot be reported against, and the issue's point is
  that the suite must eventually leave the repository; a directory that depends only on the
  installed package can.
* **Twenty problems now.** The issue asks for quality over quantity and for a small version
  first; six problems that run in an hour are more use today than twenty that do not exist.
