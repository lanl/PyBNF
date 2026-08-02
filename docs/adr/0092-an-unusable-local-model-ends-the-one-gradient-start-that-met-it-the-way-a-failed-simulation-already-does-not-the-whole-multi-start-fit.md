# An unusable local model ends the one gradient start that met it, the way a failed simulation already does, not the whole multi-start fit (issue #528)

**Status: Accepted and implemented (2026-08-02).** A gradient runner now asks whether the local
model assembled at an evaluated point is one it can actually step from — every array it consumes
finite — before it steps. An unusable model is handled exactly as a failed simulation already is:
**mid-search the trial is rejected** (the trust region shrinks, the line search backtracks) and that
start continues; **at the start point that one start terminates**, naming what was unusable, and
every other concurrent start keeps running. The two LAPACK calls in the step math are additionally
wrapped, so a factorization that fails on finite-but-pathological input takes the same route. A fit
whose models are all finite is byte-identical.

## The problem

`Laske_PLOSComputBiol2019` (k = 13, n = 42, from the Grein et al. 2026 benchmark subset) run as a
20-start `gntr` fit died after nine seconds:

```console
$ pybnf -c Laske_PLOSComputBiol2019.conf -o
[ERROR][rank 0][.../cvodes.c:8040][cvHandleFailure] At t = 0.34694 and h = 4.8226e-09, the error
test failed repeatedly or with |h| = hmin.
Job gntr_13 failed
Sorry, an unknown error occurred: numpy.linalg.LinAlgError: SVD did not converge
```

with the exception thrown from the augmented-Jacobian SVD that opens every trust-region iteration:

```python
j_aug = np.zeros((m + n, n))
j_aug[:m] = J_h
j_aug[m:] = np.diag(np.sqrt(diag_h))
u, s, vt = np.linalg.svd(j_aug, full_matrices=False)   # <- no finiteness guard
```

LAPACK's `gesdd` does not converge on non-finite input and reports that the only way it can, as
`LinAlgError`. The input was non-finite because the point's **derivatives** were, not its objective.
The traceback places it exactly: the point returned simulation data (a failed simulation takes the
guarded `grad = None` path instead and never reaches here), and `eigh` had already succeeded on the
EFIM Hessian two frames up, so the Hessian was finite and the NaN entered through the assembled
gradient — a stiff parameter set whose ODE solve completes and scores while the forward sensitivity
system it carries alongside diverges.

### The omission was asymmetric

The same runner already guards the *objective*, one method away:

```python
if not np.isfinite(f_new):
    self.Delta = 0.25 * step_h_norm      # shrink and re-solve; no accept
    return self._reject_or_budget()
```

and the same *fit* already handles a point that fails outright. The reported run's log shows that
machinery working correctly for a different start:

```
GNTR start 13/20 stopping: start point failed to simulate (a non-integrable point); no
objective/gradient to descend from
```

That is the intended shape (#492): one start dies, the other 19 continue, the concurrent multi-start
base keeps the global best. A start whose *derivatives* were bad instead of its *score* fell through
both guards.

### It is a whole-fit abort on all three gradient methods, not one lost start

Nothing between the runner and `pybnf.main` catches this, so the exception unwinds
`got_result` → `run` → `main` and the run ends with no result at all. Nineteen healthy starts are
discarded because of one — which inverts the reason multi-start exists. On a benchmark sweep this
turns "19/20 starts converged" into a crash report.

The reported traceback is `gntr`'s, but the defect is not: every gradient leaf aborts the whole fit
on the same input, only by different exceptions. Driving one poisoned start through a real 4-start
decay fit (the new end-to-end test) on the pre-fix code:

| `job_type` | what one non-finite model does to the whole fit |
|---|---|
| `gntr` | `LinAlgError: SVD did not converge` (`_build_scaling`), or `LinAlgError: Eigenvalues did not converge` if the NaN is in the EFIM Hessian, which `eigh` meets first |
| `trf` | `LinAlgError: SVD did not converge` (`_build_scaling`) |
| `lbfgs` | no LAPACK failure — the NaN gradient makes a NaN direction, and the point it proposes dies as `OutOfBoundsException: Free parameter k cannot be assigned the value nan` |

## The decision

### The question is "is this model usable", asked of the model the leaf actually consumes

A new predicate on `GradientRunner`, alongside the existing `_failed_start`:

```python
def _model_is_usable(self, grad):
    return grad is not None and self._all_finite(grad.gradient)
```

The base checks the scalar gradient every assembled `GradientResult` carries, which is exactly what
`lbfgs` steps from. Each trust-region leaf overrides it with the arrays *it* steps from, because
checking anything else would be checking a symptom:

* `_TRFRunner` — the residual `r` and its Jacobian `J`. The gradient `g = Jᵀr` and the whole
  Coleman–Li scaling are derived from them, so they are the root.
* `_GNTRRunner` — the scalar gradient and the EFIM Hessian. Its pseudo residual model is *derived*
  inside `_set_model` from those two, so the check has to happen upstream of that derivation — and
  in the reported case it must, since a NaN gradient with a finite Hessian passes `eigh` cleanly and
  only surfaces later, in the SVD.

A **missing** Hessian is deliberately not the predicate's business: that is an internal wiring error
(a `gntr` runner driven off the residual path), which `_set_model` already raises loudly, and it must
keep raising rather than being quietly reclassified as a bad point.

### Where the check fires decides what happens, and that is already settled by #492

Nothing new is invented for the response, because the runner already has two well-defined answers for
a point it cannot use, and which applies depends only on whether there is an earlier iterate to fall
back to:

* **Mid-search — reject the trial.** For `trf`/`gntr` the model check joins the objective check in
  the one branch (`not np.isfinite(f_new) or not self._model_is_usable(grad)`), so the trust region
  shrinks and re-solves from the previous point's cached SVD. For `lbfgs` it joins the Armijo test,
  so the line search backtracks along the same direction. The search *backs off toward the region
  where the model behaves*, which is where it came from.

  The trial is rejected even when it improved the objective — it is, in the new test, exactly the
  trial the clean search accepts. Accepting it would move the iterate onto a point with no usable
  curvature, and for L-BFGS-B fold a NaN curvature pair into the limited-memory history, making every
  later direction NaN. The improvement is not lost to the *fit*: every evaluated point lands in the
  trajectory before the runner ever sees it, so the global best already holds it.

* **At the start point — end this start.** There is no earlier iterate. A new
  `_failed_model(detail)` terminates the start with a reason naming what was unusable, and the
  concurrent multi-start base does what it already does for `_failed_start`: logs
  `<METHOD> start i/N stopping: <reason>`, decrements the live count, and keeps every other start
  running.

### A start that never stepped reports no objective, and says why it has none

`_failed_model` records the same `inf` penalty `_failed_start` does, even though its point *did*
score. The tempting alternative — keep the finite start-point score, since the point really was
evaluated — is wrong for the one consumer that reads a terminated runner's `fval`:
`ProfileLikelihoodAlgorithm`'s grid point. A profile point's value is
`min over the free parameters at fixed θ_k`. If the inner re-optimization never took a step, the
start value is an **upper bound** on that minimum, not the minimum. Entered as the profile it
inflates that point's Δχ², and the track's crossing test (`dchi2 >= threshold`) does not consult the
per-point `success` flag — so an un-optimized point can declare a threshold crossing that the data
does not support and close the confidence interval too narrowly. Under-reporting uncertainty is the
one direction a profile must not fail in. `inf` puts the point outside the finite filter in CI
extraction, exactly where a point carrying no profile value belongs.

That leaves the two failures indistinguishable to a consumer that has to *explain* the stop, so the
runner also records which it was, in a new `failure` field (`'simulation'` / `'model'` / `None`).
The profile track reads it to end a direction with the accurate wall:

```
reached a non-integrable point (simulation failed)
reached a point the inner re-optimization has no usable local model at (the slice scored, but
its derivatives did not)
```

The alternative — string-matching the inner runner's `stop_reason` — would couple the track to
message wording that exists to be read by humans and reworded freely.

### The LAPACK calls are wrapped too, because finite is not the same as factorizable

The finiteness check is the primary guard and the one that fires on the reported problem, but it is
not sufficient by itself, in two directions:

* `gesdd` can fail to converge on a finite but pathological matrix, independently of any NaN;
* conversely, the guarantee that non-finite input *raises* is a LAPACK implementation detail, not a
  numpy contract — a build that returned NaN singular values instead would defeat a try/except alone.

So both are in place. `_build_scaling` re-checks the *assembled* `j_aug`/`f_aug` (which also catches
an overflow introduced by the Coleman–Li scaling itself, downstream of the model check) and wraps the
SVD; `_GNTRRunner._set_model` wraps the `eigh`. Each raises a private `_UnusableModel`, caught in
`_TRFRunner.got` — the state machine's single entry point, and therefore the one place a failure
raised three frames down can become a clean per-start termination instead of an unwind through
`got_result`.

### The blast radius is not widened to a blanket try/except

An obvious alternative is to wrap `ConcurrentMultiStartOptimizer.got_result` so that *any* exception
from a runner ends that start. It is rejected: the gradient path deliberately raises `PybnfError`
from inside `got_result` for conditions that *should* end the fit — `trf`'s exact-least-squares
refusal, `gntr`'s unassembled-Hessian wiring error, an objective the assembly cannot differentiate.
Those are configuration and programming errors that every start will hit identically, and swallowing
them per-start would turn a clear refusal into 20 silent terminations and an empty result. Only the
two specific numerical failures above are converted, at the point they occur.

## Consequences

* **A gradient fit no longer dies of one bad start.** The reported `Laske` shape — one of 20 starts
  reaching a stiff point whose sensitivities diverge — now costs that start and nothing else.
* **The stop reason says which object was unusable**, per method: `least-squares residual model`
  (`trf`), `Fisher model (gradient + EFIM Hessian)` (`gntr`), `gradient` (`lbfgs`), each followed by
  *"(the point scored, but its derivatives did not)"* — the distinction from a failed simulation,
  which is what a user reading the log needs in order to tell the two apart.
* **`lbfgs` stops proposing NaN points.** The NaN-PSet abort above is gone with the same guard, so
  the leaf that reported no LAPACK failure is fixed by the same change that fixes the two that did.
* **`profile_likelihood` survives the same wall**, since it drives the same runners: a slice whose
  derivatives diverge used to abort the whole job with the identical `LinAlgError`. It now ends that
  one direction, at a point that contributes no profile value, and names the wall it hit.
* **A fit whose models are all finite is unchanged** — the guards only add checks, on arrays the step
  math is about to consume anyway (an `isfinite` scan of `J` is negligible beside the SVD of the
  augmented matrix built from it).
* **Nothing is configurable.** There is no key to make an unusable model fatal again; a whole-fit
  abort was never a behavior anyone chose.
* **A fit can now end with every start stopped this way** and still report — the trajectory's best
  scored point. That is the honest outcome (those points *were* scored), and the per-start log lines
  say what happened, but it does mean a badly placed box no longer announces itself as a crash. The
  documentation note added to `docs/gradient_fitting.rst` says so explicitly.

## Verification

* `tests/test_gradient_runner.py` — offline, backend-free, parametrized over all three leaves:
  termination at a start point whose model is not finite (at the `inf` penalty, tagged
  `failure == 'model'`); back-off from a
  mid-search unusable trial with convergence to the same minimum, where the spoiled trial is
  *the one the clean search accepts* and no point the runner proposes is ever non-finite; the
  `LinAlgError` belt-and-braces for both `svd` and `eigh`; a non-finite EFIM Hessian caught before
  `eigh`; the wiring error still raising; and the real `GradientOptimizer._advance` returning `DONE`
  rather than propagating. Eleven of the twelve fail on the pre-fix code (the twelfth is the
  wiring-error guard, which must pass both before and after).
* `tests/test_gradient_optimizer.py` — end to end (`recovery` tier), `trf` / `lbfgs` / `gntr`: a real
  4-start decay fit with start 0's assembled gradient poisoned at every point it visits completes,
  runs all four starts to termination, and recovers the truth from the survivors. On the pre-fix code
  all three reproduce the reported abort.
* `tests/test_profile_likelihood.py` — a track walking into a slice that scores but whose free-column
  sensitivities are NaN stops at that wall with the model-specific reason, records the wall point
  unsuccessful and non-finite, and does not run on to the point cap. It sits beside the existing
  non-integrable-slice test, whose behavior is unchanged. On the pre-fix code it dies with the same
  `LinAlgError` — the profile job was a third casualty of the one missing guard.

## References

* Issue #528 — the report, with the `Laske_PLOSComputBiol2019` reproducer and traceback.
* ADR-0007 — the run-loop contract (picklable step machines, no `run()` override) the runners honor.
* #492 — the failed-simulation handling this extends: `grad = None` at a non-integrable point.
* #481 / #500 — the `gntr` EFIM runner and the shared concurrent multi-start base.
