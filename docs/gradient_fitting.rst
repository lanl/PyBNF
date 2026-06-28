.. _gradient_fitting:

Gradient-based fitting (forward sensitivities)
==============================================

For a deterministic ODE (network) model, PyBNF can compute the **exact gradient** of its
objective with respect to the free parameters — not a finite-difference approximation — by
carrying each simulation's *forward output sensitivities* :math:`\partial g / \partial\theta`
through to the objective. This is the foundation a gradient-based optimizer (quasi-Newton /
trust-region least-squares) stands on: it lets the fit follow the true downhill direction
instead of probing it parameter-by-parameter.

This page describes the gradient *plumbing* available today — what it computes, the one
objective configuration it supports so far, how to enable it, and what it costs. The
gradient-based optimizer that consumes it is being added separately; until it lands, the
machinery here is exercised through PyBNF's test suite and the :mod:`pybnf.gradient` API.

.. note::

   The gradient path is **edition-2 only** and requires a **deterministic ODE** simulation of a
   **reaction network** (a ``.bngl`` model that generates a network, run with ``method=>"ode"``).
   It is computed in PyBNF's native parameter space and then transformed once into the sampling
   space the optimizer walks (see *Parameter scales* below), so a log-scaled parameter composes
   for free.


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

Only a Gaussian has :math:`\text{data fit} = \tfrac12\rho^2`, so only a Gaussian column yields an
exact least-squares residual/Jacobian. The robust families — **Laplace** (``noise_model =
laplace``) and **Student-t** (``noise_model = student_t``) — have a data fit that is *not* a sum of
squares, so the gradient is assembled from the **universal** scalar form

.. math::

   \nabla F = \sum_i w_i \,\frac{\partial\,\text{data fit}_i}{\partial\hat y_i}\,
              \frac{\partial\hat y_i}{\partial\theta},

with the per-family slope

.. math::

   \text{Laplace:}\quad \frac{\partial\,\text{data fit}}{\partial\hat y}
     = \frac{\operatorname{sign}(\hat y - y)}{b}\,f'(\hat y),
   \qquad
   \text{Student-t:}\quad \frac{\partial\,\text{data fit}}{\partial\hat y}
     = \frac{(\nu+1)\,z}{\nu + z^2}\,\frac{f'(\hat y)}{\sigma},\ \ z=\frac{\hat y - y}{\sigma}.

The Laplace data fit is non-smooth at the kink (:math:`\hat y = y`); PyBNF takes the **subgradient
0** there (the symmetric least-absolute-deviation choice). The Student-t factor
:math:`(\nu+1)/(\nu+z^2)` is the IRLS weight that downweights an outlier. Because neither carries a
least-squares residual, the result's ``least_squares_exact`` flag is ``False`` (the residual /
Jacobian then model only the Gaussian columns, if any) — the signal that a trust-region step must
consume the scalar gradient. An **estimated** noise parameter composes here exactly as for the
Gaussian: Laplace's scale :math:`b`, and Student-t's :math:`\sigma` **and** :math:`\nu` (the first
two-parameter estimated-noise gradient), each adding a scalar column for its retained normalizer.

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

* **Normalization** (``normalization`` — ``init`` / ``peak`` / ``zero`` / ``unit``): the predicted
  column is rescaled by a normalizer :math:`N(\theta)` read off the moving trajectory (its peak,
  initial value, z-score, or unit range), so :math:`\partial(\hat y_i/N)/\partial\theta` is a
  quotient/chain rule that **couples rows** — e.g. for ``peak``,
  :math:`\partial(\hat y_i/N)/\partial\theta = (\partial\hat y_i/\partial\theta -
  n_i\,\partial\hat y_p/\partial\theta)/N` with :math:`p` the peak row and :math:`n_i` the
  normalized value. The transform is applied at the data level (it overwrites the raw column), so
  the few facts the chain rule needs — the divisor and its reference row(s) — are recorded when the
  column is normalized; ``zero`` couples *every* row through the standard deviation. All four are
  threaded, and any combination of these transforms composes (normalization is applied first, then
  the cumulative/per-measurement transform on top, exactly as scoring does).


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


The capability gate (what is supported)
---------------------------------------

The gradient is assembled only for a configuration whose derivative is unambiguous and exact
today — a **Gaussian, Laplace, or Student-t noise family** with the prediction interpreted as the
**median or the mean**, additive on **any noise scale** (linear, or a log scale — log10 / natural
log; see *Log / lognormal noise scale* above), with each noise parameter either **fixed** (read
from the data / a constant) or estimated as a **single free parameter** (see *Estimated σ* and
*Asymmetric and non-Gaussian families* above), and the prediction formed through any of the
per-observable **trajectory transforms** (cumulative→incident, a per-measurement scale/offset, or
normalization; see *Trajectory transforms and normalization* above). Any other configuration raises
a clear ``GradientNotSupported`` naming what is missing, so a caller can fall back to a
gradient-free step rather than trust a wrong derivative. Not yet supported (each a separate,
additive follow-up):

* the **negative-binomial** family (its median CDF-inversion implicit derivative — issue #458);
* an estimated noise scale given by an **expression** over several free parameters, or a row-varying
  per-measurement σ (the formula chain rule is a later sub-layer);
* a **mean** prediction on a **log** scale *together with* an estimated noise parameter (there the
  moment correction depends on the noise parameter, coupling the estimated-scale column);
* an **SBML / measurement-model** observable layer; and
* **pre-equilibration / steady-state** sensitivities.

Every other objective continues to fit exactly as before; the gradient path is purely additive
and inactive unless explicitly enabled.


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
A linear parameter contributes a factor of one (and needs no extra dependency); a log-scaled
parameter's factor is obtained by autodiff of its scale, which requires the optional
``pybnf[jax]`` extra (install with ``pip install pybnf[jax]``).
