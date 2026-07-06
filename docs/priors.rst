.. _priors:

Priors and Parameter Initialization
===================================

Every free parameter carries a **prior** — a probability distribution over its
values. The prior does double duty: a Bayesian sampler uses it as the prior in
the usual sense, and an optimizer uses it as the distribution its
:ref:`initial population <param-init>` is drawn from. A prior is an orthogonal
**distribution family × scale**: the *family* (Normal, Uniform, Gamma, …) fixes
the shape, the *scale* (linear or base-10 logarithmic) fixes the space the
parameter is sampled, proposed, and stored in. The family is always evaluated in
that scale.

The ``*_var`` keyword
---------------------

A prior is declared with a ``*_var`` keyword whose name encodes the family and
the scale. The naming is regular: a family with base ``b`` yields ``b_var``
(linear) and ``logb_var`` (log10). So the Normal family gives ``normal_var`` and
``lognormal_var``; Gamma gives ``gamma_var`` and ``loggamma_var``; and so on.
The value gives the parameter id followed by the family's parameters::

    uniform_var  = k1__FREE 0.01 100        # id, lower, upper
    normal_var   = k2__FREE 1.0 0.3         # id, mean, sd
    gamma_var    = k3__FREE 2.0 0.5         # id, shape, scale
    exponential_var = k4__FREE 1.0          # id, scale (one-parameter family)

The full per-keyword syntax is under the :doc:`configuration reference
<config_keys>`.

Parameter scale
---------------

The **scale** is the space the parameter is sampled, proposed, and stored in —
**linear** or **log10**. The prior and the proposal arithmetic share it, and the
posterior target is defined directly in this scale with no change of variables.
The ``log`` prefix on a keyword selects log10; a bare "log" always means log10
across PyBNF, matching the noise-model :ref:`additive scale <noise_models>`.
Log-scale priors are the right choice for a rate constant or concentration that
ranges over orders of magnitude. (The natural-log scale is reachable only
through the labelled :ref:`parameter record <parameter-record>` below, not the
positional ``*_var`` grammar.)

Support and reflecting bounds
-----------------------------

A family's **support** — the region where its density is nonzero — is intrinsic
to the family: Uniform is finite, Normal and Laplace are unbounded, and the
positive families (Gamma, Exponential, half-Normal, …) are bounded below at
zero.

**Reflecting bounds** are a separate idea. They are a box a proposal is folded
back into during a fit, and they exist only for the box-shaped
``uniform_var`` / ``loguniform_var`` priors, which take an optional trailing flag
— ``b`` (or blank) keeps the parameter **bounded**, so a proposal that would
leave the box is reflected back in; ``u`` makes it **unbounded**, letting the
search leave the initial range::

    uniform_var = x__FREE 10 30 u    # sample in [10, 30], but allow moves outside

The other finite-support families (for example Beta on ``[0, 1]``) draw from
their own density rather than a reflecting box, so they take no such flag.

Distribution families
---------------------

The families below are all reachable through the positional ``*_var`` grammar.
Each row lists the linear keyword; every family also has the ``log``-prefixed
log10 form (``lognormal_var``, ``loggamma_var``, …).

.. list-table::
   :header-rows: 1
   :widths: 22 26 24 28

   * - Family (linear keyword)
     - Parameters
     - Support
     - Notes
   * - ``uniform_var``
     - lower, upper
     - finite box
     - The box-bounded prior; takes the ``b`` / ``u`` reflecting-bounds flag.
       ``loguniform_var`` is the log10 form.
   * - ``normal_var``
     - mean, sd
     - :math:`(-\infty, \infty)`
     - Gaussian.
   * - ``laplace_var``
     - location, scale
     - :math:`(-\infty, \infty)`
     - Heavier-tailed than Normal.
   * - ``cauchy_var``
     - location, scale
     - :math:`(-\infty, \infty)`
     - Very heavy tails.
   * - ``gumbel_var``
     - location, scale
     - :math:`(-\infty, \infty)`
     - Extreme-value.
   * - ``logistic_var``
     - location, scale
     - :math:`(-\infty, \infty)`
     - Symmetric, slightly heavier-tailed than Normal.
   * - ``gamma_var``
     - shape, scale
     - :math:`(0, \infty)`
     - Positive; PEtab-catalog parity.
   * - ``inv_gamma_var``
     - shape, scale
     - :math:`(0, \infty)`
     - Conjugate prior for a variance.
   * - ``weibull_var``
     - shape, scale
     - :math:`(0, \infty)`
     - Lifetime / time-to-event.
   * - ``beta_var``
     - alpha, beta
     - :math:`[0, 1]`
     - The canonical prior for a fraction.
   * - ``exponential_var``
     - scale
     - :math:`(0, \infty)`
     - One-parameter.
   * - ``chisquare_var``
     - dof
     - :math:`(0, \infty)`
     - One-parameter.
   * - ``rayleigh_var``
     - scale
     - :math:`(0, \infty)`
     - One-parameter.
   * - ``half_normal_var``
     - scale
     - :math:`(0, \infty)`
     - The right half of a zero-centered Normal; a mild positive scale prior.
   * - ``half_cauchy_var``
     - scale
     - :math:`(0, \infty)`
     - The right half of a zero-centered Cauchy; a weakly-informative scale
       prior.

The positive-support and log-scale families are natural priors for an
:ref:`estimated noise parameter <noise_models>` — a standard deviation or
dispersion that must stay positive.

No prior: start points
----------------------

The keywords ``var`` (linear) and ``logvar`` (log10) give a parameter a single
start value and **no** prior distribution. They are the start points for the
start-point optimizers — Simplex, Powell, and CMA-ES. A no-prior parameter still
carries a scale and is varied during the fit; it simply contributes nothing to
the log prior and cannot be prior-sampled::

    var    = k__FREE 1.5
    logvar = k__FREE 0.001

.. _parameter-record:

Multi-parameter priors: the parameter record
---------------------------------------------

The positional ``*_var`` grammar carries at most two distribution parameters, so
a family with three — like **Student-t** (degrees of freedom, location, scale) —
has no positional keyword. These are authored instead through the edition-2
labelled ``parameter:`` record, which names each field, and which is also where
the natural-log parameter scale is selected. Tutorial lesson `32_prior_gallery
<https://github.com/lanl/PyBNF/tree/main/examples/tutorial/32_prior_gallery>`__
walks through the full catalog, including the Student-t record.

See also
--------

- :ref:`Initialization <param-init>` — how the prior seeds an optimizer's
  starting population, including Latin-hypercube sampling.
- :doc:`config_keys` — the exact per-keyword configuration syntax.
- :ref:`noise_models` — the companion reference; an estimated noise parameter is
  a free parameter and takes a prior like any other.
- :doc:`petab` — PyBNF imports and exports PEtab v2 priors, whose catalog these
  families mirror.
- :ref:`API reference <priors_module>` — the :py:mod:`pybnf.priors` module
  docstrings for the ``Prior`` families and their scale/bounding infrastructure.
