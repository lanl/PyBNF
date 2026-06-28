.. _analytical_objectives:

Analytical and user-defined objectives
======================================

PyBNF can optimize or sample an objective that is a **closed-form function of the free
parameters**, with **no BNGL or SBML model file and no simulator**. You write the objective
directly in the ``.conf`` file — as a math expression or a Python callable — and PyBNF's full
optimizer / sampler / parallel / prior machinery applies unchanged. This makes PyBNF useful well
beyond biological-model calibration: fitting analytical test functions, sampling an arbitrary
posterior, prototyping on a closed-form approximation, or fitting a small analytical model
(dose-response / Hill, Michaelis–Menten steady state, growth curves, mixture models) without
writing a mechanistic model.

All of these forms require a modern :ref:`edition <edition>` (``edition = 2``).

.. note::

   **The convention is a negative log-likelihood (lower is better).** You supply the *cost* — a
   negative log-likelihood, or any quantity to be minimized — exactly as for every other PyBNF
   objective. The optimizers minimize it; the Bayesian samplers treat it as :math:`-\log p(\text{data}
   \mid \theta)` and assemble the posterior :math:`\log p(\theta \mid \text{data}) = \log p(\theta) -
   \text{objective}`, so a prior on a parameter (a ``normal_var`` / ``parameter:`` record) folds in
   automatically. There is no sign to get backwards: write the cost, not the log-likelihood.

There are three ways to declare the objective, in increasing generality:

* a **built-in analytical target** chosen from a menu (``objective = banana, …``),
* an inline **math expression** (``objective = expression``), and
* a **Python callable** (``objective = callable``).


Built-in analytical targets
---------------------------

PyBNF ships five standard test geometries, declared inline on the objective line — no separate
file:

======================  ====================================================================
Target                  Geometry
======================  ====================================================================
``gaussian``            Axis-aligned Gaussian (diagonal variance; a *separable* objective)
``rotated_gaussian``    Correlated Gaussian with a full covariance (non-separable)
``rotated_quartic``     Smooth, non-separable, non-quadratic, trap-free curved valley (2-D)
``banana``              Rosenbrock / banana-shaped distribution (any dimension)
``multimodal``          Mixture of Gaussians with configurable modes
======================  ====================================================================

The target's constants ride the objective line; a vector field (a mean or variance) is a
space-separated list, and the ``multimodal`` mixture components are given as repeated ``mode:``
records::

    edition   = 2
    objective = banana, a = 1, b = 100
    objective = gaussian, mean = 0 0, variance = 1 1
    objective = rotated_gaussian, mean = 0 0, variances = 2 0.5, angle = 0.5236
    objective = rotated_quartic, mean = 0 0, angle = 0.5236, coeff = 0.01 1

::

    edition   = 2
    objective = multimodal
    mode: weight = 0.5, mean = -4 -4, variance = 0.5 0.5
    mode: weight = 0.5, mean =  4  4, variance = 0.5 0.5

For a menu target the coordinates are anonymous, so each free parameter binds to a coordinate **by
the integer index in its name**: a parameter ending in ``1`` is coordinate 1, one ending in ``2`` is
coordinate 2, and so on (any prefix works — ``x1``/``p1``/``theta1``). The index set must be exactly
``1..D`` for the target's dimension ``D``; a missing index or a name without one is a pointed error.
The defaults are echoed at run start, so the geometry is never silently assumed.

::

    edition   = 2
    objective = banana, a = 1, b = 100
    job_type  = de
    uniform_var = x1 -5 5
    uniform_var = x2 -5 5
    population_size = 20
    max_iterations  = 200

(The five targets are also reachable from a ``.target`` JSON file via ``model: name.target`` +
``objective = score`` — the original developer surface, retained for back-compatibility and for a
fully general covariance matrix.)


Bring your own: ``objective = expression``
------------------------------------------

An inline math expression declares the objective as a function of the free parameters, written as
**PEtab math** on a companion ``expression`` line::

    edition     = 2
    objective   = expression
    expression  = 0.5*((1 - x1)^2 + 100*(x2 - x1^2)^2)     # the Rosenbrock cost
    job_type    = de
    uniform_var = x1 -5 5
    uniform_var = x2 -5 5
    population_size = 20
    max_iterations  = 300

The expression's symbols bind to the declared free parameters **by name** (``x1`` → the parameter
``x1``); declaration order is irrelevant. A declared parameter the expression does not reference is
simply unconstrained by the likelihood (its prior still samples it). An undeclared symbol, or an
unparseable expression, is a pointed error at config load — never a surprise mid-run.

.. note::

   **PEtab math uses** ``^`` **for exponentiation, not** ``**``. The expression supports the usual
   operators, parentheses, and function calls (``log``, ``exp``, ``sin``, …). The expression form
   requires the optional PEtab/sympy extra: ``pip install pybnf[petab]``.

The expression form is the recommended default for any objective a single formula can express; the
callable form below is the escape hatch for the rest.


Bring your own: ``objective = callable``
----------------------------------------

When a single expression cannot capture the objective — a ``logsumexp`` mixture, a loop over
groups or replicates, a ``scipy.stats`` density, a hand-rolled pooling term — supply a **Python
callable** instead::

    edition   = 2
    objective = callable
    callable  = mymodule:negative_log_likelihood

The ``callable`` value is an entry point ``<module>:<function>``. The left side is either an
**importable dotted module** (``mypkg.mymodule``, resolved on ``PYTHONPATH``) or a **file path**
(``path/to/model.py``); the right side is the function name. The function is resolved and validated
at config load, so a missing module, a wrong name, or a non-callable is caught immediately.

The function must have the signature::

    def negative_log_likelihood(params, data=None):
        # params : {parameter_name: value}   -- bind-by-name, the declared free parameters
        # data   : {experiment_name: Data}   -- bound .exp files, or None (see below)
        return float_cost                    # the NLL / cost; lower is better

``params`` is a plain ``{name: value}`` dict of the current parameter set. The callable returns the
scalar cost. PyBNF imports your Python (the same trust model as a ``postprocess`` script), so the
callable can do anything Python can.

.. note::

   A general Python callable is **not differentiable** by PyBNF, so ``objective = callable`` works
   with the gradient-free optimizers and samplers (``de`` / ``am`` / ``dream`` / …) but **not** with
   :ref:`job_type = hmc <alg-hmc>`; for HMC use ``objective = expression`` or a built-in target.


Binding experimental data
-------------------------

A bring-your-own objective that fits a curve to measurements (a Hill curve, a growth model)
needs the data. Declare it with a top-level ``data`` key — a comma list of ``.exp`` files, each one
**experiment**::

    data = dose_response.exp
    data = replicate1.exp, replicate2.exp

The ``data`` key is valid only with ``objective = expression`` or ``objective = callable`` (any
other objective binds data through a model / experiment). The two forms consume it differently:

* A **callable** receives the whole set as its ``data`` argument: a ``{experiment_name: Data}``
  mapping keyed by each file's stem (``dose_response.exp`` → ``data["dose_response"]``), or ``None``
  when no ``data`` key is present. The callable reduces it however it likes — sum over one
  experiment, pool across many, weight, whatever. Each :class:`~pybnf.data.Data` exposes its columns
  by header: ``d["time"]`` / ``d["obs"]`` return the column arrays.

* An **expression** becomes a **per-observation** contribution over the parameters *and* the data
  columns: the ``.exp`` column headers join the parameters as symbols, and PyBNF evaluates the
  expression once per data row and **sums** the result over every row and every bound experiment.
  This is the standard "sum of per-point NLL" form. For example, a Gaussian curve fit::

      edition     = 2
      objective   = expression
      expression  = 0.5*(y - vmax*x/(km + x))^2     # x, y are the data columns
      data        = michaelis_menten.exp            # columns: x  y
      job_type    = de
      uniform_var = vmax 0 10
      uniform_var = km   0 10

  Here ``vmax`` and ``km`` are free parameters and ``x``/``y`` are the ``.exp`` columns; the
  expression is the per-point squared residual, summed over the data. A data column whose name
  collides with a free parameter, or a referenced column missing from a bound experiment, is a
  pointed error at config load. (A callable handles non-per-observation reductions — coupled
  points, custom weighting — that the per-observation expression cannot.)


Bayesian inference and priors
-----------------------------

Because the objective is an NLL, **every analytical / bring-your-own objective is a likelihood**,
and PyBNF's Bayesian samplers (:ref:`am <alg-am>`, :ref:`dream <alg-dream>`, ``p_dream``) sample its
posterior with no extra work. Declare informative priors with the parameter keywords
(:ref:`normal_var <objective_key>` and the rest of the prior catalog, or the new-era
``parameter:`` record) and the sampler assembles :math:`\log p(\theta \mid \text{data}) = \log
p(\theta) - \text{objective}`. A flat (``uniform_var``) prior over a wide box recovers the maximum
likelihood / least-squares estimate; an informative prior gives the full Bayesian posterior. So an
analytical Bayesian model is just ``objective = expression`` (or ``callable``) + a prior + a
sampler — no new code::

    edition     = 2
    objective   = expression
    expression  = 0.5*(y - vmax*x/(km + x))^2
    data        = michaelis_menten.exp
    job_type    = am
    normal_var  = vmax 5 2          # informative prior
    uniform_var = km   0 10
    population_size = 4
    max_iterations  = 50000

The posterior samples are written in the standard format and load straight into ArviZ when
:ref:`output_inference_data <output_inference_data>` ``= 1`` (``pip install pybnf[arviz]``).


.. _alg-hmc-on-analytical:

Gradient-based sampling (HMC)
-----------------------------

For an analytical or ``expression`` target, PyBNF can sample the posterior with **Hamiltonian Monte
Carlo (NUTS)** — :ref:`job_type = hmc <alg-hmc>`. Because the closed-form objective is
differentiable, HMC follows the posterior's gradient and mixes far better than the gradient-free
samplers on correlated or curved geometries. HMC is the reference sampler for benchmarking the
gradient-free methods, and it requires the optional ``pip install pybnf[jax]`` extra. It applies to
the built-in targets and to ``objective = expression`` (including a data-bound curve fit), but not
to ``objective = callable`` (a general callable is not differentiable). See :ref:`HMC <alg-hmc>` for
details and the HMC-specific keys.


Worked examples
---------------

**Optimize the Rosenbrock function.** No model, no data; recover the mode at ``(1, 1)``::

    edition     = 2
    objective   = expression
    expression  = 0.5*((1 - x1)^2 + 100*(x2 - x1^2)^2)
    job_type    = de
    uniform_var = x1 -5 5
    uniform_var = x2 -5 5
    population_size = 20
    max_iterations  = 500

**Fit a Michaelis–Menten curve to data with a callable.** ``mm.py`` next to the config::

    # mm.py
    import numpy as np

    def nll(params, data):
        d = data["mm"]                                  # the mm.exp experiment
        pred = params["vmax"] * d["x"] / (params["km"] + d["x"])
        return 0.5 * float(np.sum((d["y"] - pred) ** 2))

::

    edition     = 2
    objective   = callable
    callable    = mm.py:nll
    data        = mm.exp
    job_type    = de
    uniform_var = vmax 0 10
    uniform_var = km   0 10
    population_size = 20
    max_iterations  = 200

**Sample a banana posterior with HMC** (requires ``pybnf[jax]``)::

    edition     = 2
    objective   = banana, a = 1, b = 8
    job_type    = hmc
    uniform_var = x1 -12 12
    uniform_var = x2 -12 12
    population_size = 4
    num_warmup   = 1000
    num_samples  = 2000
    target_accept = 0.95
