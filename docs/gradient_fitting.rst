.. _gradient_fitting:

Gradient-based fitting (forward sensitivities)
==============================================

For a deterministic ODE (network) model, PyBNF can compute the **exact gradient** of its
objective with respect to the free parameters — not a finite-difference approximation — by
carrying each simulation's *forward output sensitivities* :math:`\partial g / \partial\theta`
through to the objective. This is the foundation a gradient-based optimizer (quasi-Newton /
trust-region least-squares) stands on: it lets the fit follow the true downhill direction
instead of probing it parameter-by-parameter.

This page describes the gradient *plumbing* — what it computes, the objective
configurations it supports, how to enable it, and what it costs — and the three
gradient-based optimizers that consume it (see *Running a gradient fit* below).

.. note::

   The gradient path is **edition-2 only** and requires a **deterministic ODE** simulation of a
   **reaction network** (a ``.bngl`` model that generates a network, run with ``method=>"ode"``).
   These requirements are **enforced**, not merely documented: a gradient ``job_type`` on a
   non-edition-2 config, a non-bngsim model, or a model that bngsim cannot differentiate is
   refused up front with a message that names **which** requirement is unmet, then points back at
   a metaheuristic ``job_type`` on an indented ``->`` line beneath it. A non-ODE simulation
   *method* (SSA / NFsim) is refused at the first sensitivity-bearing simulation. A model with
   **discrete events** (a discrete jump in the dynamics) *is* supported: bngsim applies the
   event's own sensitivity jump at each fire, so a dosing or stimulation schedule fits on the
   gradient path — see *Discrete events* below for which event shapes it covers and what an older
   bngsim does instead. The gradient is computed in
   PyBNF's native parameter space and then transformed once into the sampling space the optimizer
   walks (see *Parameter scales* below), so a log-scaled parameter composes for free.


Running a gradient fit
----------------------

Three optimizers consume the gradient, all opt-in via ``job_type``:

* ``job_type = trf`` — a **Trust-Region-Reflective least-squares** optimizer
  (Branch–Coleman–Li, matching ``scipy.optimize.least_squares(method="trf")``). It
  consumes the residual vector + residual Jacobian and approximates the Hessian as
  :math:`J^{\mathsf T}J`, which is far better-conditioned on a least-squares problem than feeding a
  scalar gradient to a generic quasi-Newton method. This is the workhorse for the common Gaussian /
  sum-of-squares case. Bounds are handled by the **Coleman–Li reflective transformation**: a
  trust-region step that would leave the box is reflected off the bound it crosses, and an affine
  scaling :math:`D(x)` derived from the distance to the bounds keeps the model valid as the iterate
  approaches a bound — so it converges cleanly onto a **bound-active** optimum (sliding along an
  active face) rather than stalling against it, and its first-order optimality test reads as optimal
  on an active face. It requires an **exact least-squares residual** (a Gaussian or fixed-scale
  Student-t objective, no constraints); a fit whose objective is not an exact sum of squares is
  refused with a pointer to ``lbfgs``.

* ``job_type = lbfgs`` — a bounded limited-memory quasi-Newton optimizer (**L-BFGS-B**,
  Byrd–Lu–Nocedal–Zhu). It consumes the **scalar** gradient, so it handles precisely the objectives
  ``trf`` refuses: an estimated noise scale — including a **prediction-dependent** scale
  (``sigma = prediction_formula …``, the combined additive+proportional error model, whose scale
  rides the same forward sensitivity as the residual) — the Laplace / count families, and active
  constraint penalties.

* ``job_type = gntr`` — a **general-objective Fisher/Gauss-Newton trust-region** optimizer. It gives
  ``trf``'s trust-region step quality — the well-conditioned :math:`J^{\mathsf T}J`-style curvature —
  for the very objectives ``trf`` refuses and only ``lbfgs`` could handle. Its Hessian is the
  **expected-Fisher / Gauss-Newton information**
  :math:`H = \sum_i \kappa_i\, s_i s_i^{\mathsf T}` (plus estimated-noise and constraint blocks),
  built from the same forward sensitivities :math:`s_i = \partial \mathrm{pred}_i/\partial\theta`
  plus small analytic per-family curvature factors — no second-order sensitivities. It consumes the
  *same scalar gradient* as ``lbfgs``; only the curvature model differs. Internally it feeds
  :math:`(g, H)` through the same Coleman–Li reflective machinery ``trf`` uses (so on a Gaussian
  least-squares fit it reduces to ``trf``'s step exactly), but with the general Fisher Hessian.
  This cut supports an estimated-σ Gaussian (``chi_sq_dynamic``), a fixed-scale Laplace, a
  fixed-dispersion negative-binomial (mean-centered), and a Gaussian fit with static-hinge
  constraints; a coupled corner it cannot yet build the Fisher Hessian for (a mean-on-log-scale
  estimated scale, a **prediction-dependent** scale — whose scale couples to the location, so the
  noise block is not diagonal — a free-dispersion / median count family, an estimated Student-t df,
  or an estimated constraint scale) is refused with a pointer to ``lbfgs``, which fits it.

All three run natively inside PyBNF's distributed propose/score loop (one objective evaluation is one
scheduler job) rather than through a blocking ``scipy`` driver, so backup/resume work exactly as for
every other ``job_type``. They are also registered as **refiners** (``refine_method = trf`` / ``lbfgs``
/ ``gntr``), so a gradient step can polish a metaheuristic's best fit.

**Local multi-start.** A gradient method is purely *local*: it descends into whatever basin its
start point lands in. To guard against a bad basin on a multimodal or bound-active landscape, a
standalone gradient fit over a bounded-prior box runs **N independent starts concurrently** and
keeps the global best. ``N`` reuses ``population_size`` (consistent with the metaheuristics, where it
is the parallel-population size):

* ``population_size = 1`` — a single start from the box center (the historical behavior).
* ``population_size = N`` — start 0 is the box center; the remaining ``N − 1`` are Latin-hypercube
  samples drawn across the prior box from the seeded ``random_seed``, so the scatter is reproducible.

The N starts run as N concurrent jobs (matching every other method's parallelism), each advancing its
own step machine, and the best fit found across all of them is the result. Multi-start applies only
to a standalone box-start fit: when the optimizer runs as a **refiner** (an explicit start point is
injected) it always runs a single start, since the job there is to polish the one best fit, not to
re-scatter. ``max_iterations`` is the per-start iteration budget.

**A bad start is survivable.** Stiff corners of a parameter box can defeat the solver, and a fit is
expected to walk into them: a point may fail to integrate at all, or integrate while its forward
sensitivities diverge, leaving a finite objective with a non-finite gradient. Neither ends the fit.
The affected start stops where it is — the log records why, e.g.::

    GNTR start 13/20 stopping: start point failed to simulate (a non-integrable point); no
        objective/gradient to descend from
    GNTR start 7/20 stopping: the Fisher model (gradient + EFIM Hessian) at the start point is
        not finite (the point scored, but its derivatives did not); no usable local model to
        descend from

— while every other start keeps running and the global best is taken across the survivors. Mid-search
the same conditions are gentler still: the trial is simply rejected (the trust region shrinks, or the
line search backtracks) and that start continues from its current iterate. A run in which some starts
stopped this way still reports a fit; if *every* start stops this way, the box is likely placed where
the model cannot be integrated.

**Convergence tuning.** Both optimizers stop when a first-order optimality (gradient) tolerance or a
step tolerance is met, or the per-start iteration budget is exhausted:

* ``trf`` — ``trf_grad_tol`` (first-order optimality on the scaled gradient, default ``1e-8``) and
  ``trf_step_tol`` (accepted step negligible relative to the point, default ``1e-8``). The initial
  trust radius is derived from the start point; there is no separate radius knob.
* ``lbfgs`` — ``lbfgs_grad_tol`` (gradient tolerance, default ``1e-6``), ``lbfgs_step_tol`` (step
  tolerance, default ``1e-8``), ``lbfgs_history`` (number of stored correction pairs, default
  ``10``), and the line-search constants ``lbfgs_c1`` (Armijo sufficient-decrease, default ``1e-4``)
  and ``lbfgs_backtrack`` (step-length reduction factor, :math:`0 < \beta < 1`, default ``0.5``).
* ``gntr`` — ``gntr_grad_tol`` (first-order optimality on the scaled gradient, default ``1e-8``) and
  ``gntr_step_tol`` (accepted step negligible, default ``1e-8``), identical in meaning to ``trf``'s
  (it *is* a trust-region step), plus ``gntr_ridge`` — a small relative Levenberg ridge added to the
  Fisher Hessian before the step solve (default ``1e-10``), large enough to keep the Hessian strictly
  positive definite, small enough not to perturb the Newton step.

For all three, ``<method>_max_iterations`` caps the iterations per start and defaults to the global
``max_iterations``.


.. _multiple_shooting:

Multiple shooting (``job_type = ms``)
--------------------------------------

Every method above fits by **single shooting**: simulate the whole time course from the model's
initial conditions and compare it with the data. On a model whose difficulty *is* horizon length,
that transcription can put the answer out of reach of any search. The motivating case is an
oscillator: on ``Borghans_BiophysChem1997``, a correctly-shaped trajectory whose period is wrong by
more than about 3 % scores **worse than fitting no dynamics at all**, so a flat line is the ceiling
on essentially the whole parameter box, and fifteen independent global searches terminate at it.

``job_type = ms`` changes the transcription rather than the search. Each experiment's time course
is cut at knots; each segment is integrated from its **own** start state, and continuity between
segments is imposed as an equality constraint rather than assumed. Over one short segment a period
error cannot accumulate, so the information moves out of a residual term that saturates and into
continuity defects, which carry a direction. The segment-start states are internal to the method —
searched, bounded and differentiated, but never reported as fit results.

**The score you get is an ordinary one.** Every iterate is *certified*: the auxiliary states are
discarded, the parameters are re-simulated with ordinary single shooting, and that score is what is
reported and ranked. A run that leaves continuity unconverged therefore scores as what it actually
is, and the number in ``sorted_params_final.txt`` is directly comparable with any other fit's.

**The ladder is the mechanism.** A run does not fix a segment count; it solves a sequence of
transcriptions, coarsening toward the ordinary unsegmented problem (``4 → 2 → 1`` by default) and
warm-starting each rung from the previous one. The stage trace is printed as it goes, and on the
motivating problem it *is* the result — every segmented stage scored worse than a flat line, and
the coarsening is what converted them.

**Requirements and limits.** ``ms`` needs everything the gradient path needs, plus a model
with an enumerated ODE state to restart from — which means either bngsim path. A knot carries
the model's **state**, so both a generated network (``.net``, via BNGL) and an SBML/Antimony
model with ``sbml_backend = bngsim`` are supported; a network-free (NFsim) model never
enumerates a state and is refused. It refuses, by name and up front, a fit it
would otherwise quietly change — a dose-response scan or pre-equilibration protocol (no time axis
to cut, or a measured phase that already starts from a carried state), and any scored quantity that
is a function of a whole series (an analytic per-series ``scale``, a data ``normalization``, a
``cumulative`` column, or BPSL constraints). An analytically profiled noise scale
(``noise_profiling = 1``) is fine and is the recommended pairing: it is profiled over the pooled
data residuals, which continuity defects never enter. Segments are simulated serially, so ``ms``
does not scale across a cluster the way the metaheuristics do, and a stopped run cannot be resumed
with ``-r``.

**A note on size, for network models.** The transcription's width scales with the model's
**state**, not with the number of fitted parameters: the finest rung adds
``(m − 1) × n_species`` internal variables per experiment, and the initial-condition
sensitivity system is ``n_species`` wide. On a handful of species that is nothing; on a
combinatorially expanded reaction network it can dwarf the fit's own dimension
(``egfr_ground.net``, 356 species, at ``m = 4`` adds ~1068). A run prints the number it is
adding as it starts, so the cost is visible before it is paid; lower ``ms_segments`` if it is
more than the problem warrants.

**Tuning.** The defaults come from measurement on the motivating problem, not from convention, and
are worth leaving alone unless you have a reason:

* ``ms_segments`` (default ``4``) — the **finest** rung of the ladder, and ``ms_coarsening``
  (default ``2``) the factor between rungs. Starting with many short segments is the wrong end:
  under partial observability an over-segmented stage is under-determined and routinely certifies
  worse than its own start.
* ``ms_penalty`` (default ``10``) and ``ms_penalty_growth`` (default ``5``) — the augmented
  Lagrangian's initial penalty and its growth factor, capped at ``ms_max_penalty``. Deliberately
  **tight**: a loose start was measured both worse *and* twice as expensive, because the inner
  solve on a nearly-unconstrained subproblem never converges and burns its whole budget.
* ``ms_feasibility_tol`` / ``ms_optimality_tol`` (both ``1e-6``) — the scaled continuity defect and
  the projected-gradient optimality a run must reach together to report convergence. The optimality
  tolerance is looser than ``gntr_grad_tol`` on purpose: it measures the *augmented* Lagrangian's
  gradient, which carries a factor of the penalty.
* ``ms_inner_iterations`` (default ``50``) — trust-region iterations per outer iteration; an
  approximate inner solve is what the method is designed around. ``ms_max_iterations`` caps outer
  iterations per rung and defaults to the global ``max_iterations``.
* ``ms_aux_decades`` (default ``6``) — the half-width, in decades, of a segment-start state's box
  around its own magnitude.

A minimal run::

    edition = 2
    job_type = ms
    sbml_backend = bngsim
    model: model.xml
    experiment: myexp, data: mydata.exp
    noise_profiling = 1
    output_dir = output/ms

The design, the measurements behind each default, and what this cut deliberately leaves out are
recorded in ADR-0109 (the reusable transcription layer) and ADR-0110 (this consumer).


Profile likelihood (identifiability + confidence intervals)
-----------------------------------------------------------

``job_type = profile_likelihood`` is a standalone job that turns the gradient path into a
**Data2Dynamics-style** identifiability analysis (Raue et al., *Bioinformatics* 25(15):1923–1929,
2009). For each fitted parameter :math:`\theta_k` it fixes :math:`\theta_k` to a grid of values
around the optimum :math:`\theta^\*` and **re-optimizes all the other parameters** at each grid
point, tracing the profile :math:`\chi^2_{\mathrm{PL}}(\theta_k) = \min_{j\neq k}\chi^2(\theta)`.
It shares every requirement and gate of the ``trf`` / ``lbfgs`` methods (edition 2, a deterministic
ODE network, bngsim forward sensitivities). The inner re-optimizations reuse the same two engines,
and the job picks between them automatically from the objective's structure: an **exact
least-squares** objective (a fixed-scale Gaussian / Student-t, no constraints) profiles with the
trust-region ``trf`` step, while any other objective (an estimated noise scale, the Laplace / count
families, active constraint penalties) profiles with ``lbfgs`` on the scalar gradient. You do not
choose the engine — a one-line note at the start of the run reports which was selected.

The job runs in two phases:

#. **Find** :math:`\theta^\*`. If every parameter declares an ``initial_value:`` (the optimum from
   a fit you already ran), those values are taken as :math:`\theta^\*` and the fit is skipped.
   Otherwise the job first runs a multi-start trust-region **polish** over the bounded-prior box
   (``population_size`` starts, ``max_iterations`` budget) to locate :math:`\theta^\*`.
#. **Profile.** Each parameter is walked outward from :math:`\theta^\*` in both directions on an
   **adaptive** ``log10``-space grid (the step shrinks where the profile steepens, grows where it is
   flat), warm-starting each grid point's re-optimization from its neighbour. A direction stops when
   the profile crosses the :math:`\Delta\chi^2` threshold (the :math:`\chi^2` quantile at the
   configured confidence level, 1 dof), reaches a parameter bound, or hits a per-direction point cap.
   The profiles are independent, so they are farmed across the scheduler concurrently — one
   directional walk per parameter per direction, up to ``profile_likelihood_max_parallel`` at a time
   (default: all of them) — rather than run serially. A cap only queues the excess walks; none is
   dropped, so coverage is never silently truncated.

**Configuring a run.** Profile likelihood is a ``job_type`` on the ordinary edition-2 surface —
the same ``model:`` / ``experiment:`` / free-parameter lines any gradient fit uses, with the
run-selector set to ``profile_likelihood`` and a handful of ``profile_likelihood_*`` knobs. A
minimal config that polishes to the optimum and then profiles every free parameter::

    edition = 2
    model: model.bngl
    experiment: myexp, data: mydata.exp
    output_dir = output/pl

    bngl_backend = bngsim
    job_type = profile_likelihood
    objective = chi_sq                 # a per-point _SD column in the .exp -> exact least squares

    population_size = 20               # multi-start polish: 20 starts to locate theta*
    max_iterations = 200               # per-start polish budget

    profile_likelihood_confidence = 0.95
    profile_likelihood_step = 0.05     # initial adaptive grid step (sampling space)
    profile_likelihood_max_points = 40 # per-direction grid-point cap

    loguniform_var = k1 1e-4 1e2       # the free parameters to profile
    loguniform_var = k2 1e-4 1e2

To profile only a subset, name them: ``profile_likelihood_params = k1, k2``. If you have **already
fitted** the model, skip the polish by giving each parameter its optimum as an ``initial_value:`` on
a ``parameter:`` record — the job then takes those as :math:`\theta^\*` and profiles around them
without re-fitting::

    parameter: k1, lower: 1e-4, upper: 1e2, initial_value: 0.017
    parameter: k2, lower: 1e-4, upper: 1e2, initial_value: 3.1

**Reading the results.** From each finished profile the job extracts the confidence interval at the
configured level and assigns an identifiability class. It writes three kinds of artifact to
``Results/``:

* ``profile_likelihood_summary.txt`` — one row per parameter: the best-fit value :math:`\theta^\*_k`
  (the centre the profile was traced around), the CI endpoints, per-endpoint *at-bound* flags, the
  classification, and any coverage ``notes``. This is the table to read first.
* ``profile_<name>.txt`` — the profile *curve* for one parameter: each grid point's parameter value,
  the re-optimized objective, its :math:`\Delta\chi^2` above the optimum, and the inner
  re-optimization's iteration count + convergence flag. Plot the :math:`\Delta\chi^2` column against
  the parameter column to see the profile shape; where it crosses the horizontal threshold line is
  the CI edge.
* ``profile_likelihood.png`` — the same picture, drawn for you: one :math:`\Delta\chi^2` panel per
  parameter with the threshold, CI, and optimum marked. Only written when matplotlib is installed
  (the optional ``pybnf[plot]`` extra); its absence is logged and the text artifacts are unaffected.

The **classification** summarizes the profile shape (Raue *et al.* 2009):

* **identifiable** — the profile crosses the :math:`\Delta\chi^2` threshold on *both* sides, giving a
  finite two-sided CI that brackets :math:`\theta^\*_k`. The parameter is pinned down by the data.
* **practically non-identifiable** — the profile rises but does **not** cross the threshold on at
  least one side before it runs into a parameter bound (or the point cap). The CI is *open* on that
  side; PyBNF reports the endpoint clamped **at the bound** and flags it (``ci_low_at_bound`` /
  ``ci_high_at_bound``) rather than silently closing the interval, so a one-sided or bound-limited CI
  reads as exactly that. More/better-placed data — or a wider bound, if the true value may lie beyond
  it — is what tightens such a parameter. A side that stopped only because it hit the grid-point cap
  (not a genuine plateau) is called out in the ``notes`` column with a pointer to raise
  ``profile_likelihood_max_points``.
* **structurally non-identifiable** — the profile is **flat**: the parameter can move with no
  objective response because another parameter (or combination) compensates exactly. This is a
  property of the model + observables, not the data volume; it is resolved by adding an observable
  that breaks the degeneracy, fixing one of the confounded parameters, or reparameterizing to the
  identifiable combination.

A direction can also stop at a **wall**: a fixed value beyond which the model no longer integrates,
or one where it integrates and scores but its sensitivities do not, leaving the inner
re-optimization nothing to descend. Either way that grid point contributes no profile value (it is
recorded unsuccessful and dropped from the CI, rather than entered as an un-optimized upper bound
that could fabricate a threshold crossing), the direction stops at the wall, and the ``stops`` note
says which wall it was. Such a side reads as *not cleanly crossed* — the honest result when the
profile cannot be continued far enough to cross the threshold.

Every per-point profile record rides PyBNF's ordinary backup/resume, so a run can be resumed or
extended without recomputing a finished profile.

The knobs are ``profile_likelihood_confidence`` (the CI level), ``profile_likelihood_params`` (the
subset to profile; default all), ``profile_likelihood_step`` / ``profile_likelihood_min_step`` /
``profile_likelihood_max_step`` / ``profile_likelihood_dchi2_target`` (the adaptive grid),
``profile_likelihood_max_points`` (the per-direction cap), ``profile_likelihood_reopt_max_iterations``
(the per-grid-point re-optimization budget), and ``profile_likelihood_max_parallel`` (the max
concurrent directional walks; ``0`` = all of them).


What it computes
----------------

For the default Gaussian objective (``chi_sq`` and its modern ``noise_model`` equivalents), each
scored observation :math:`i` contributes a **standardized residual**

.. math::

   \rho_i = \frac{\hat y_i(\theta) - y_i}{\sigma_i},
   \qquad \text{loss} = \tfrac{1}{2}\sum_i \rho_i^2 ,

exactly the quantity ``chi_sq`` already sums. PyBNF assembles, summed across every experiment:

* the **residual vector** :math:`\rho` and the **residual Jacobian** :math:`J_{ij} =
  (1/\sigma_i)\,\partial\hat y_i/\partial\theta_j` — the form a trust-region least-squares
  solver (``scipy.least_squares``) consumes directly; and
* the **scalar gradient** :math:`\nabla F = J^{\mathsf T}\rho` — the form a quasi-Newton method
  (L-BFGS-B) consumes.

With a **fixed** σ the data fit is the whole objective, so both forms are built from the
*same* :math:`\rho` and :math:`J` and agree by construction: the optimizer walks precisely the
surface PyBNF reports, with the same :math:`\sigma`-weighting, the same column selection, and the
same per-point bootstrap weights as the scalar objective.


Estimated σ (a free-parameter noise scale)
------------------------------------------

A **fitted** σ — the edition-2 ``noise_model = normal, sigma = fit <param>`` surface, where
``<param>`` is an ordinary free parameter declared by id (no legacy ``__FREE`` marker) — keeps
the Gaussian normalizer, so the per-point loss is :math:`(\hat y_i - y_i)^2/(2\sigma^2) +
\log\sigma` and the gradient gains a column for the noise parameter:

.. math::

   \frac{\partial\,\text{loss}}{\partial\sigma}
   = -\frac{(\hat y_i - y_i)^2}{\sigma^3} + \frac{1}{\sigma}
   = \frac{1 - \rho_i^2}{\sigma}.

The free σ carries no model column (it is unbound from the simulation), so this column comes
entirely from the normalizer and the σ-dependence of the data fit — never from the sensitivity
tensor. Because :math:`\log\sigma` is **not** a sum of squares, it cannot be represented in the
residual/Jacobian form; PyBNF therefore folds the σ column into the **scalar gradient only** and
leaves the residual Jacobian a faithful least-squares model of the data fit alone. The result's
``least_squares_exact`` flag is ``False`` whenever an estimated σ is present — the signal that a
trust-region least-squares step must consume the scalar gradient (quasi-Newton / L-BFGS) rather
than the bare residual form. A fixed-σ fit is unaffected (the flag stays ``True``).


Log / lognormal noise scale
---------------------------

The Gaussian noise can be additive on a **log scale** rather than the linear one — the
edition-2 ``noise_model = lognormal`` surface (log10) — modelling multiplicative error. The
gradient handles it as a strict generalization: with the prediction taken to be the **median**,
the standardized residual simply lives in the additive (log) space,

.. math::

   \rho_i = \frac{f(\hat y_i) - f(y_i)}{\sigma_i},
   \qquad f = \log_{10}\ (\text{or}\ \ln),

so :math:`\tfrac12\rho_i^2` is still the per-point data fit. The native sensitivity
:math:`\partial\hat y_i/\partial\theta` is unchanged; only the per-point residual derivative
picks up the scale's chain factor,

.. math::

   \frac{\partial\rho_i}{\partial\hat y_i} = \frac{f'(\hat y_i)}{\sigma_i}
   = \frac{1}{\hat y_i\,\sigma_i\,\ln b},

with :math:`b = 10` for log10 and :math:`b = e` for the natural-log variant. The linear scale is
the :math:`f' = 1` special case, so an ordinary additive-error fit is byte-for-byte unchanged. A
log scale composes with an estimated σ (``noise_model = lognormal, sigma = fit <param>``): the
σ column :math:`(1-\rho_i^2)/\sigma` is identical once :math:`\rho` is read in log space.

A log scale only has support for **positive** values (:math:`f = \log_{10}` requires
:math:`\hat y_i > 0` and :math:`y_i > 0`). At a non-positive prediction or observation the
gradient path does **not** raise — it propagates a non-finite value, exactly as the scalar
objective returns a non-finite score for the same out-of-support point. That keeps the gradient
consistent with the objective it differentiates and gives a trust-region step its usual signal to
reject the point, rather than aborting the whole assembly.

A **mean** (rather than median) prediction adds the family's moment correction; it is now
supported too (see *Asymmetric and non-Gaussian families* below).


Asymmetric and non-Gaussian families
------------------------------------

A noise family yields an **exact least-squares residual/Jacobian** — the form a trust-region
solver (Levenberg–Marquardt / TRF) minimizes directly — exactly when its data fit reformulates as
a *smooth* half-square. The Gaussian does (:math:`\text{data fit} = \tfrac12\rho^2`). Among the
robust families, **Student-t** (``noise_model = student_t``) does too, but **Laplace**
(``noise_model = laplace``) does not.

**Student-t: an exact square-root-loss residual** (issue #459). With :math:`z=(\hat y-y)/\sigma`,
the signed residual

.. math::

   r = \operatorname{sign}(z)\,\sqrt{2\,\text{data fit}}
     = \operatorname{sign}(z)\,\sqrt{(\nu+1)\,\log\!\bigl(1+z^2/\nu\bigr)}

satisfies **both** :math:`\tfrac12 r^2 = \text{data fit}` (so ``scipy.least_squares`` minimizes the
*true* Student-t loss, not a frozen-weight IRLS surrogate) **and**
:math:`r\,\partial r/\partial\hat y = \partial\,\text{data fit}/\partial\hat y` (so its
residual-Jacobian reproduces the objective gradient). It is **smooth through** :math:`z=0`
(:math:`r\sim\sqrt{(\nu+1)/\nu}\,z` near the origin — an odd, infinitely differentiable function of
:math:`z` — behaving like a Gaussian residual at the center and downweighting the tails as
:math:`z` grows), with :math:`\partial r/\partial\hat y = \sqrt{(\nu+1)/\nu}\,f'(\hat y)/\sigma` at
the center. So a **fixed-scale Student-t fit is** ``least_squares_exact`` — the Gaussian's
exact-least-squares status recovered for the robust family, and the LM/TRF path fits it directly.

**Laplace stays scalar-only.** Its L1 data fit :math:`|z|/b` gives
:math:`\sqrt{2\,\text{data fit}}\sim\sqrt{|z|}`, a **cusp with infinite slope at** :math:`z=0` (and
the IRLS weight :math:`1/|z|\to\infty` there), so least-absolute-deviation is *inherently* not
cleanly least-squares — no residual a trust-region solver could minimize. (A smoothed pseudo-Huber
surrogate would be a separate, explicitly opt-in approximation of the loss, not exposed here.) The
count family likewise has no least-squares residual. Such a family's gradient is assembled from the
**universal** scalar form

.. math::

   \nabla F = \sum_i w_i \,\frac{\partial\,\text{data fit}_i}{\partial\hat y_i}\,
              \frac{\partial\hat y_i}{\partial\theta},
   \qquad
   \text{Laplace:}\quad \frac{\partial\,\text{data fit}}{\partial\hat y}
     = \frac{\operatorname{sign}(\hat y - y)}{b}\,f'(\hat y),

non-smooth at the kink (:math:`\hat y = y`), where PyBNF takes the **subgradient 0** (the symmetric
least-absolute-deviation choice). A no-residual family makes the result's ``least_squares_exact``
flag ``False`` (the residual/Jacobian then model only the Gaussian / Student-t columns, if any) —
the signal that a trust-region step must consume the scalar gradient.

An **estimated** noise parameter composes for every family: Laplace's scale :math:`b`, and
Student-t's :math:`\sigma` **and** :math:`\nu` (the first two-parameter estimated-noise gradient),
each adding a scalar column for its retained normalizer. A normalizer is never a square, so an
estimated-scale fit is ``least_squares_exact`` ``False`` for *any* family — including Student-t,
whose data-fit residual still stacks while its :math:`\log\sigma` / df-block normalizer columns
ride the scalar gradient.

**Mean centering.** The prediction may be interpreted as the distribution's **mean** rather than
the median (``location = mean``); the gradient subtracts the family's moment correction in additive
space. The correction is prediction-independent, so it is free on the derivative side and a no-op
on the linear scale (where mean = median for these symmetric families). It bites only on a log
scale, where a mean prediction models the original-space mean of a log-normal / log-Laplace.

The **negative-binomial** family is the one asymmetric family not yet differentiable: its default
median centering inverts a continuous CDF for the distribution mean (a root-find), so its gradient
needs implicit differentiation through that inversion — a named follow-up (issue #458).


Trajectory transforms and normalization
---------------------------------------

The scored prediction need not be the raw simulated observable: PyBNF can form it through a
per-observable **trajectory transform** before scoring, and the gradient threads each transform's
own derivative so it stays exact.

* **Cumulative → incident** (``cumulative``): a cumulative count is differenced to its
  per-interval increment, :math:`\hat p_i = \hat y_i - \hat y_{i-1}` (row 0 keeps its raw value),
  so the sensitivity is the matching difference of sensitivity rows,
  :math:`\partial\hat p_i/\partial\theta = \partial\hat y_i/\partial\theta -
  \partial\hat y_{i-1}/\partial\theta`.

* **Per-measurement scale/offset**: a row-varying ``observableParameters`` measurement model is a
  general formula :math:`\hat p_i = f(\hat y_i,\dots;\,a,\dots)` over sim-output columns and
  per-row scale/offset tokens. Its sensitivity is the formula's exact symbolic gradient, chained
  through each referenced column's sensitivity plus any **estimated** scale/offset parameter it
  names. Unlike an estimated σ (which lands only on the scalar gradient), such a parameter genuinely
  enters :math:`\partial\hat p/\partial\theta`, so it has a real residual-Jacobian column (a square),
  and the residual form stays exact.

* **Normalization** (``normalization`` — ``init`` / ``peak`` / ``zero`` / ``unit`` / ``floor``):
  the predicted
  column is rescaled by a normalizer :math:`N(\theta)` read off the moving trajectory (its peak,
  initial value, z-score, or unit range), so :math:`\partial(\hat y_i/N)/\partial\theta` is a
  quotient/chain rule that **couples rows** — e.g. for ``peak``,
  :math:`\partial(\hat y_i/N)/\partial\theta = (\partial\hat y_i/\partial\theta -
  n_i\,\partial\hat y_p/\partial\theta)/N` with :math:`p` the peak row and :math:`n_i` the
  normalized value. The transform is applied at the data level (it overwrites the raw column), so
  the few facts the chain rule needs — the divisor and its reference row(s) — are recorded when the
  column is normalized; ``zero`` couples *every* row through the standard deviation, and ``floor``
  (:math:`\hat y_i + \rho\max\hat y`) is additive, so every row picks up the same
  :math:`\rho\,\partial\hat y_p/\partial\theta` term. All five are
  threaded, and any combination of these transforms composes (normalization is applied first, then
  the cumulative/per-measurement transform on top, exactly as scoring does). A **chain** of two or
  more of them on one column (``floor 0.03, peak``) composes as well: each stage's rule is that
  same closed form read in *its own* inputs — the previous stage's per-row sensitivities — so the
  gradient folds the chain forward, one wrapper per stage, with the values a stage produced riding
  along on its record when the next transform overwrites them (#539).

* **Analytic per-series scale** (``normalization <obs> = scale``): the scored value is
  :math:`c^{*}(\theta)\,\hat y_i(\theta)`, where :math:`c^{*}` is profiled out of the *whole*
  matched series at scoring time, so its sensitivity is a product rule whose second term is shared
  by every point of the column:

  .. math::

     \frac{\partial(c^{*}\hat y_i)}{\partial\theta}
       = c^{*}\frac{\partial \hat y_i}{\partial\theta} + \hat y_i\frac{\partial c^{*}}{\partial\theta},
     \qquad
     \frac{\partial c^{*}}{\partial\theta} =
     \begin{cases}
       -\,c^{*}\displaystyle\sum_i w_i \frac{\partial \hat y_i/\partial\theta}{\hat y_i}\Big/\sum_i w_i
         & \text{(log family)}\\[2ex]
       \displaystyle\sum_i w_i\,(y_i - 2c^{*}\hat y_i)\,\frac{\partial \hat y_i}{\partial\theta}
         \Big/ \sum_i w_i \hat y_i^{2} & \text{(linear family)}
     \end{cases}

  — the closed-form derivative of the profiling condition itself, summed over exactly the points
  the profiling includes. This term does **not** drop out by the envelope theorem: the profiling is
  σ-unweighted (and family-aware only in the log/linear split), so :math:`c^{*}` is not in general
  the objective's own minimizer over the scale. The profiled scale is resolved per experiment, so a
  column scaled in one experiment is an ordinary column in another.


Constraint penalties
--------------------

A fit may add **qualitative / inequality constraints** (a ``.prop`` / ``.con`` file) whose penalty
is added to the objective — *"Stot > 90 at time = 2"*, *"A < B always"*, and the like. A
constraint penalty is a piecewise (``weight``) or Gaussian-CDF (``confidence`` / ``tolerance``)
function of an at-/between-time **readout** :math:`q_1 - q_2`, evaluated at the worst-case point
:math:`i^\*` of its enforcement interval, so its gradient is that readout's forward sensitivity
times the local penalty slope:

.. math::

   \frac{\partial(\text{penalty})}{\partial\theta}
   = \underbrace{f'(\Delta)}_{\text{local slope}}\;
     \Bigl(\frac{\partial q_{1}}{\partial\theta} - \frac{\partial q_{2}}{\partial\theta}\Bigr)_{i^\*},
   \qquad \Delta = \max_i\,(q_{1,i} - q_{2,i}),

evaluated at the achieving row :math:`i^\*` (Danskin's theorem; the *best* point if the constraint
is enforced ``once``). For the **static** model :math:`f' = \text{weight}` where the constraint is
violated and **0** where it is satisfied or pinned to a ``min_penalty`` floor (the non-smooth
boundary takes the subgradient 0, like the Laplace kink). For the **likelihood** model
:math:`f'(\Delta) = (p_\max - p_\min)\,\phi(-\Delta/k)/(k\,p_{\text{adj}})` — smooth everywhere.
A constant operand contributes no sensitivity. PyBNF assembles the summed constraint gradient
(:func:`~pybnf.gradient.assemble_constraint_gradient`), in sampling space, ready to add to the
objective gradient. Like an estimated-σ normalizer, a penalty is **not** a sum of squares, so a fit
with active constraints is not ``least_squares_exact`` (its gradient is consumed on the scalar
path).


Measurement-model layer (SBML / Antimony)
-----------------------------------------

A scored observable need not be a raw simulation output. The **measurement-model layer**
(``observableFormula``, the new-era PEtab / SBML path) materializes each observable as an
expression :math:`g = f(\hat y_1, \dots;\,w, \dots)` over the simulation's output columns and the
parameter set — a *post-simulation* transform, applied identically for BNGL and SBML/Antimony — so
its sensitivity is the formula's exact symbolic gradient chained through each referenced column's
forward sensitivity:

.. math::

   \frac{\partial g}{\partial\theta}
   = \sum_{\text{columns } c} \frac{\partial f}{\partial c}\,\frac{\partial c}{\partial\theta}
   \;+\; \sum_{\text{parameters } w} \frac{\partial f}{\partial w},

with the column terms reading the same routing-folded, normalization-aware sensitivities the rest
of the assembly uses, and a parameter named *directly* in the formula (an observation-model scale /
offset estimated as a fit parameter) contributing its :math:`\partial f/\partial w` straight to its
own column. Like a per-measurement scale, such a parameter genuinely enters
:math:`\partial g/\partial\theta`, so it has a real residual-Jacobian column (a square) and the
residual form stays exact; a fixed model constant and the independent variable contribute nothing.

This is what lets a small **SBML / Antimony** model fit on the gradient path: that backend exposes
the same forward output sensitivities as the network ODE backend (per-``species:`` and, with
``print_functions``, per-function), and the measurement layer differentiates the
``observableFormula`` over them. A **bare-name** observable (the formula is just one species /
observable) needs no measurement model — it scores that column directly through its forward
sensitivity.


Pre-equilibration / steady state
--------------------------------

A **pre-equilibration** experiment (``preequilibrate:``) runs the model unmeasured to steady state,
switches a condition, then measures the transient from that equilibrated state — one simulation,
two phases, species state carried over with no reset between them (ADR-0052). The measured
trajectory's initial condition *is* the steady state :math:`x^\*(\theta)`, which itself depends on
the free parameters, so the measurement phase's forward sensitivities must start not from zero but
from the **steady-state sensitivity** :math:`\partial x^\*/\partial\theta`:

.. math::

   \frac{\partial x^\*}{\partial\theta}
   = -\Bigl(\frac{\partial f}{\partial x}\Bigr)^{-1}\frac{\partial f}{\partial\theta},
   \qquad f(x^\*,\theta) = 0,

the implicit-function-theorem derivative of the steady-state condition. The backend computes this
seed and threads it across the pre-equilibration boundary (it integrates the equilibration phase's
sensitivities to their steady value and uses that as the measurement phase's initial sensitivity),
so the assembly reads the measurement-phase tensor exactly as for any other experiment — no
special case in the objective math. The effect is sharp where a parameter sets the equilibrium but
is *switched out of* the measurement-phase dynamics: its entire measured-trajectory gradient flows
through the seed, and would read identically zero without it.

This is handled automatically: when the gradient path is active, the measurement phase of a
pre-equilibration protocol seeds its (parameter-axis) sensitivities from the equilibration phase's
steady-state sensitivity. The equilibration phase is a deterministic ODE run requesting the same
parameter sensitivities (which the gradient path does by construction). A free parameter bound
*only* to an **initial condition** is not carried across the boundary — a stable steady state is
independent of its initial conditions (its steady-state sensitivity is zero), so there is nothing to
seed; that combination is refused rather than reported as a (degenerate) zero.

The measured phase may be a **dose-response scan** rather than a time course — the
preincubate → wash → dose-scan protocol (``preequilibrate:`` + ``condition:`` +
``type: parameter_scan``), where every dose starts from the carried post-intervention state. Each
dose's initial sensitivity is that state's :math:`\partial x/\partial\theta`, and the per-dose
tensors stack down the dose axis exactly as for a fresh-from-seed scan. The **intervention** in
between (a species ``setConcentration``) assigns an initial condition of its own, so its
:math:`\partial x_k(0)/\partial\theta` is that assignment's derivative, not the equilibration's:
zero for a literal amount, and the exact derivative for an amount written over model parameters
(including through derived ones). An intervention amount outside the arithmetic grammar is refused
by name. This needs ``bngsim >= 0.12.0``; an older build refuses the scored scan with an upgrade
hint. Scanning a parameter the fit is *differentiating* is refused — the carried derivative was
taken at the pre-scan value while each dose pins the same symbol — so scan the dose/condition
parameter the data sweeps.

The **pre-equilibration condition's own** species perturbation (a ``preequilibrate:`` condition
with a species target) is the same assignment written one phase earlier, before anything has run,
and it is differentiated the same way: PyBNF declares the assignment's
:math:`\partial x_k(0)/\partial\theta` to the backend so the run is seeded from it (ADR-0101). This
matters only for a **fixed-duration** equilibration (``equil_t_end:``) — a steady-state
equilibration relaxes the assigned amount away, so the measured phase genuinely does not depend on
it. In both positions, an amount that reads a fitted parameter which no requested sensitivity
column carries is refused by name rather than contributing a zero row: write such a dose over model
parameters the fit varies (``"A()" = 2*k_deg``, or through a derived id), not over a free parameter
that binds no model parameter.

Species interventions are the one place the two backends differ. On an SBML/Antimony model
(``sbml_backend = bngsim``) the parameter-axis seeding above works exactly as described, but a
**measurement condition that writes a species amount** mid-protocol is refused on the gradient path:
the write retires the carried sensitivity matrix, and that backend has no rebuild for the write's
own seed row (ADR-0104). Fit such a protocol with a gradient-free ``job_type``, express the
intervention as a parameter the species' initial value reads, or use a BNGL model.


Discrete events
---------------

A **discrete event** — an SBML ``event``, and so also an Antimony ``at (…): …`` — is a discrete
jump in the dynamics: it reinitialises the integrator state at a trigger, which is how a dosing
or stimulation schedule is usually written. A forward sensitivity carried across such a jump is
right only if the solver applies the event's own jump to it at each fire,

.. math::

   s^+ = \frac{\partial h}{\partial x}
         \left(s^- + f^-\frac{\partial t^*}{\partial p}\right)
         + \frac{\partial h}{\partial p}
         - f^+\frac{\partial t^*}{\partial p}

for a state assignment :math:`x^+ = h(x^-, p)` firing at :math:`t^*`. On a bngsim **newer than
0.12.1** it does, so an event-bearing model fits on ``trf`` / ``lbfgs`` / ``gntr`` like any
other. The event shapes it covers are:

* a **fixed trigger time** (``time >= 5``), where the crossing does not move with the parameters
  (:math:`\partial t^*/\partial p = 0`) and the jump is the assignment Jacobian alone;
* a trigger whose **threshold is a fitted constant** (``time >= t_on`` with ``t_on`` estimated),
  where the crossing time is resolved before the run;
* a **state-dependent** trigger (``A < 30``) that reduces to a single relational comparison,
  whose crossing bngsim differentiates in flight by the implicit function theorem.

What it declines — an event with an **execution delay**, and a trigger that is *not* a single
relational comparison (a conjunction, a negation, an equality), whose true-set boundary has no
one differentiable surface — is refused per simulation with a message naming the reason, so the
fit stops with an actionable error rather than a wrong gradient. Use a metaheuristic ``job_type``
for those.

**On bngsim 0.12.1 or older** the refusal is instead *blanket* and arrives up front, at
construction. The floor is set by silent wrongness, not by a missing feature: such a build can
answer an event it cannot actually differentiate without saying so. A trigger reading the state
came back as a finite tensor with the event's contribution missing rather than being refused
(lanl/bngsim#52, through 0.11.x); an assignment reading the state — ``A := A + dose``, the
repeat-dosing idiom — dropped the carried term and restarted the assigned row from zero
(lanl/bngsim#144, through 0.12.1); and a solver root that fires nothing rewound the state but not
the sensitivity history (lanl/bngsim#146, through 0.12.1). PyBNF declines the whole class on
those builds rather than run a fit to completion on a wrong gradient, and the message names the
upgrade.

**Discontinuity triggers are not events.** A forcing pulse or a piecewise-time schedule written
as ``if(t >= tau)`` in a rate law breaks the integrator step but never jumps the state, so it
never reached the differentiability gate above. It is differentiable on a current bngsim,
switch time included: the crossing contributes a term where the switch fires, so ``tau`` is
itself estimable. (This is newer than it looks — through bngsim 0.12.1 an ``if()`` in a rate law
declined bngsim's sensitivity codegen and the fit was refused per simulation, which is what
tutorial Lesson 6 used to be about.)


The capability gate (what is supported)
---------------------------------------

The gradient is assembled only for a configuration whose derivative is unambiguous and exact
today — a **Gaussian, Laplace, or Student-t noise family** with the prediction interpreted as the
**median or the mean**, additive on **any noise scale** (linear, or a log scale — log10 / natural
log; see *Log / lognormal noise scale* above), with each noise parameter either **fixed** (read
from the data / a constant) or estimated as a **single free parameter** (see *Estimated σ* and
*Asymmetric and non-Gaussian families* above), the prediction formed through any of the
per-observable **trajectory transforms** (cumulative→incident, a per-measurement scale/offset,
normalization — floor and multi-transform chains included — or an analytic per-series scale; see
*Trajectory transforms and normalization* above), and observables materialized
through a **measurement-model layer** (the SBML/Antimony / ``observableFormula`` path; see
*Measurement-model layer* above). Any other configuration raises a clear ``GradientNotSupported``
naming what is missing, so a caller can fall back to a gradient-free step rather than trust a wrong
derivative. Not yet supported (each a separate, additive follow-up):

* the **negative-binomial** family (its median CDF-inversion implicit derivative — issue #458);
* an estimated noise scale given by an **expression** over several free parameters, or a row-varying
  per-measurement σ (the formula chain rule is a later sub-layer);
* a **mean** prediction on a **log** scale *together with* an estimated noise parameter (there the
  moment correction depends on the noise parameter, coupling the estimated-scale column).

**Pre-equilibration / steady-state** sensitivities *are* supported (see above). Every other
objective continues to fit exactly as before; the gradient path is purely additive and inactive
unless explicitly enabled.


Enabling sensitivities
----------------------

The gradient path is opt-in and gated on the simulator backend. Enabling it does three things:

#. **Capability check.** The bngsim backend must expose forward output sensitivities; a build
   without them refuses with an actionable message (the scalar fit is unaffected).
#. **Routing.** Each free parameter is matched to the model entity of the same id: a kinetic /
   global parameter is requested on the *parameter* sensitivity axis, while a free parameter that
   is a species' initial value is requested on the *initial-condition* axis. A per-experiment
   condition (a ``condition:`` perturbation) contributes the exact chain-rule factor for that
   experiment, and a parameter pinned by the condition is dropped from the request.

   A free parameter may also bind **no** model id and reach the model only through a condition
   that sets an entity to its value — ``condition: cA, perturbations: I0_ = I0_CA``, the usual way
   to fit a per-condition initial condition. Its column is then the sum over everything that
   target reaches: the target's own sensitivity axis, plus one term for every initial value the
   target *seeds*, each scaled by that seeding's derivative. Both are exact, and neither is
   assumed to be 1 — a target that seeds two species with opposite signs
   (``I_ = I0_``, ``S_ = N_ - I0_``) contributes both, and a target that feeds a parameter another
   quantity is derived from (``beta_N = R0_*gamma_/N_``) contributes that parameter's axis scaled
   by a factor re-evaluated at every fit point. A seeding expression outside the arithmetic
   grammar (a function call), or one reached through an assignment rule, is refused with a
   message naming the target rather than silently contributing a wrong column.
#. **Solve.** Each simulation then integrates the model **and** its sensitivities, attaching the
   native :math:`\partial g/\partial\theta` tensor to the simulated data for the assembly to read.

.. warning::

   A free parameter that *only* sets a species' initial value is live on the gradient path's
   initial-condition axis, but a plain time-course ``execute`` does not, by itself, re-evaluate
   species initializers from the current parameters. A gradient-based initial-condition fit must
   synchronise the species initial concentrations (as the steady-state scan path already does)
   so the parameter genuinely moves the trajectory.


Cost
----

Forward sensitivities make each simulation solve an augmented system: alongside the :math:`N`
state equations it integrates :math:`N\times P` sensitivity equations, where :math:`P` is the
number of parameters whose sensitivity is requested for that experiment. The practical wall-clock
cost of a sensitivity-bearing solve scales roughly as :math:`(1 + P)` times a plain solve — so a
single sensitivity solve replaces the :math:`P{+}1` separate solves a forward-difference gradient
would need, at comparable cost but with an **exact** derivative and no step-size tuning. Only the
parameters actually free in a given experiment are requested (a condition-pinned parameter is
dropped), so :math:`P` is the live count, not the total.


Parameter scales
----------------

The sensitivities and the residual Jacobian are assembled in PyBNF's **native** parameter space.
The transform into the **sampling** space the optimizer walks (the :math:`\theta\leftrightarrow u`
map a ``log10`` / ``ln`` parameter scale defines) is applied exactly once at the end: the residual
is scale-invariant, and each Jacobian column is multiplied by :math:`\mathrm{d}\theta/\mathrm{d}u`.
Every scale supplies that factor in closed form — one for a linear parameter,
:math:`\ln(10)\,\theta` for ``log10``, :math:`\theta` for ``ln`` — so gradient fitting needs no
autodiff and no optional dependency, whatever scale the parameters are declared on.

.. seealso::

   :ref:`API reference <gradient_module>` — the :py:mod:`pybnf.gradient` module
   docstrings for the sensitivity routing and gradient-assembly layers.
