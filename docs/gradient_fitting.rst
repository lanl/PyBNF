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


The capability gate (what is supported)
---------------------------------------

The gradient is assembled only for the configuration whose derivative is unambiguous and
exact today — the **default Gaussian noise family, additive on the linear scale, with the
prediction interpreted as the median**, and the σ either **fixed** (read from the data / a
constant) or estimated as a **single free parameter** (``sigma = fit <param>``; see *Estimated
σ* above). Any other configuration raises a clear ``GradientNotSupported`` naming
what is missing, so a caller can fall back to a gradient-free step rather than trust a wrong
derivative. Not yet supported (each a separate, additive follow-up):

* an estimated σ given by an **expression** over several free parameters, or a row-varying
  per-measurement σ (the formula chain rule is a later sub-layer);
* a **log / lognormal** noise scale, or a **mean** (rather than median) location;
* an **asymmetric** family (Laplace, negative-binomial, Student-t, …);
* a per-observable **trajectory transform** — cumulative→incident differencing, or a
  per-measurement scale/offset;
* an **SBML / measurement-model** observable layer; and
* **constraint** penalties.

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
