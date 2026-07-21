.. _noise_models:

Noise Models and Objective Functions
====================================

Every fit scores a candidate parameter set with an **objective function**. The
number it returns — the *objective value*, lower is better — is what an
optimizer minimizes and what a Bayesian sampler reads as the data term of the
negative log-posterior.

Many of PyBNF's objective functions are **noise models**: probabilistic
observation models that map a deterministic model prediction plus one or more
noise parameters to a distribution over the observed data, whose negative
log-likelihood *is* the objective value. The remaining objectives (``sos``,
``sod``, ``norm_sos``, ``ave_norm_sos``) are plain losses, not likelihoods.

This page is the conceptual reference for the menu. The exact configuration
syntax lives with the :ref:`objfunc <objective>` key (whole-fit default) and the
:ref:`noise_model <noise_model_key>` key (whole-fit or per-observable); the
closed-form residuals, Jacobians, and differentiability that let the
gradient-based optimizers and HMC use these families are covered in
:doc:`gradient_fitting`.

Two shapes of noise model
-------------------------

A **per-point noise model** has a log-likelihood that factors into a sum of
independent per-observation terms — ``chi_sq`` (Gaussian), ``lognormal`` / ``lnnormal``
(Gaussian on log10 / natural log), ``laplace`` (Laplace), ``neg_bin`` (negative binomial),
and ``student_t`` (Student-t). It is
defined by three orthogonal axes, paired with a **noise-parameter source**:

- **Distribution family** — Gaussian, Laplace, Student-t, or negative binomial;
  the shape of the observation noise.
- **Additive scale** — the scale the noise is additive on: ``LINEAR`` (additive
  on the observable itself, the ordinary Gaussian), ``LOG10`` (additive on
  log10 — multiplicative, lognormal error), or ``LN`` (additive on the natural
  log). One log base across PyBNF: a bare "log" always means log10, so the
  ``lognormal`` objective is log10; the explicit ``lnnormal`` family asks for
  natural log (``Gaussian(LN)``).
- **Location interpretation** — which summary of the distribution the prediction
  is taken to be: ``median`` (PEtab v2's convention, and the no-correction
  default) or ``mean`` (which adds the family's moment correction on a log
  scale). It matters only when the noise is asymmetric on the prediction's
  scale; for a linear-additive Gaussian, mean and median coincide.

A **column-joint noise model** couples the per-observation contributions across
a whole data column, so the likelihood does not factor point by point. Today the
only member is ``kl``, the multinomial cross-entropy. (The related geometric
``wasserstein`` distance is a column-joint *objective*, not a likelihood; both
are selected with the ``profile_objective`` key rather than ``objfunc``.)

The noise-parameter source
--------------------------

A noise model's dispersion — a Gaussian's :math:`\sigma`, a Laplace's scale
:math:`b`, a negative binomial's dispersion :math:`r` — is not part of the
family; it is supplied separately by a **noise-parameter source**. The source's
kind is load-bearing: a *fixed* source contributes only the family's data-fit
term (its normalizer is a parameter-independent constant that drops out), while
an *estimated* source contributes the full negative log-likelihood **including
the normalizer** — the term that keeps a free scale from running off to
infinity. This one rule is what makes each legacy objective its exact
family-times-source default.

The sources, in the vocabulary of the :ref:`noise_model <noise_model_key>` key:

- **read from a data column** (``read_exp_file <suffix>``) — read the value per
  point from the experimental column ``<observable><suffix>`` (conventionally
  ``_SD``). Fixed.
- **estimate as a free parameter** (``fit <name>__FREE``) — a single free
  parameter, declared and prior-sampled like any other. Estimated.
- **fix at a constant** (``fix_at <number>``). Fixed.
- **relative** (``relative [<cv>]``) — a constant coefficient of variation,
  :math:`\sigma = \mathrm{cv}\cdot|\mathrm{value}|`; the heteroscedastic model
  the legacy ``norm_sos`` fits.
- **column mean** (``column_mean``) — one scale per column, the observable's
  experimental column mean; the model the legacy ``ave_norm_sos`` fits.
- **formula** (``formula <expr>``) — an expression over free parameters (and,
  row by row, PEtab noise placeholders); the PEtab ``noiseFormula`` source.
- **prediction formula** (``prediction_formula <expr>``) — an expression whose
  :math:`\sigma` scales with the **simulated output**, the combined
  additive+proportional error model :math:`\sigma = \sigma_\text{abs} +
  \sigma_\text{rel}\cdot y` where :math:`y` is the observable's predicted value.
  Symbols resolve either from the fit (the estimated coefficients) or from the
  current simulation column of that name (a model species/observable/function).
  Estimated. Gradient-free score path only.

The objective-function codes
----------------------------

The legacy ``objfunc`` values below each pin one point on the axes above. They
remain valid, and in edition-2 you can reach the same models — and combinations
the legacy codes never named — through the more explicit ``noise_model`` key.
The five plain least-squares losses (``sos``, ``sod``, ``norm_sos``,
``ave_norm_sos``, and the fixed-:math:`\sigma` ``chi_sq``) already have their
formulas under :ref:`Objective functions <objective>`; they are not restated
here.

.. list-table::
   :header-rows: 1
   :widths: 20 34 46

   * - ``objfunc``
     - Family × source
     - Notes
   * - ``chi_sq``
     - Gaussian; :math:`\sigma` read from ``_SD`` (fixed)
     - The chi-square data fit. Fixed :math:`\sigma`, so the Gaussian
       normalizer drops.
   * - ``chi_sq_dynamic``
     - Gaussian; :math:`\sigma` estimated (``sigma__FREE``)
     - Estimated :math:`\sigma`, so the ``+log`` :math:`\sigma` normalizer is
       retained.
   * - ``lognormal``
     - Gaussian on **LOG10**, median; :math:`\sigma` from ``_SD`` (fixed)
     - Multiplicative error; observations and predictions must be positive.
   * - ``laplace``
     - Laplace; scale :math:`b` estimated (``b__FREE``)
     - Heavy-tailed, outlier-robust (least-absolute-deviation); the
       :math:`\log(2b)` normalizer is retained.
   * - ``sos``
     - Gaussian; :math:`\sigma` fixed at 1
     - Sum of squares. A loss, not a calibrated likelihood.
   * - ``sod``
     - Laplace; scale fixed at 1
     - Sum of absolute differences.
   * - ``norm_sos``
     - Gaussian; ``relative`` scale
     - Each residual normalized by its own data value.
   * - ``ave_norm_sos``
     - Gaussian; ``column_mean`` scale
     - Each residual normalized by the column average.
   * - ``neg_bin``
     - Negative binomial; dispersion :math:`r` fixed (``neg_bin_r``)
     - Overdispersed counts. Self-normalizing PMF.
   * - ``neg_bin_dynamic``
     - Negative binomial; dispersion :math:`r` estimated (``r__FREE``)
     - Counts with an estimated dispersion; recognizes the legacy ``_Cum``
       column convention (see below).
   * - ``kl``
     - Multinomial cross-entropy (column-joint)
     - Compares the shape of a whole column at once, not point by point.
   * - ``direct_pass``
     - Passthrough
     - Reads a single ``score`` cell and ignores the data — the seam for an
       :doc:`analytical or bring-your-own objective <analytical_objectives>`.

``student_t`` — the heavy-tailed, outlier-robust Gaussian with a tail-heaviness
knob :math:`\nu` (degrees of freedom) — is the one per-point family without a
legacy ``objfunc`` code; reach it through ``noise_model = student_t`` (its two
parameters, :math:`\sigma` and :math:`\nu`, are sourced independently, so a fit
may estimate zero, one, or both). See :doc:`gradient_fitting` for its
differentiable square-root-loss residual.

``lnnormal`` is the modern natural-log sibling of ``lognormal``. Both use the same
Gaussian kernel and ``sigma`` sources; only the additive scale differs:
``lognormal`` scores ``log10(prediction) - log10(observation)``, while ``lnnormal``
scores ``ln(prediction) - ln(observation)`` and interprets ``sigma`` in natural-log units.
The distinction also carries into normalized pointwise likelihoods: ``lnnormal`` has the
change-of-variables Jacobian ``-ln(observation)``, whereas log10 additionally contributes
``-ln(ln(10))``. PEtab v2 ``noiseDistribution = log-normal`` maps exactly to ``lnnormal``.

A noise scale that depends on the prediction
--------------------------------------------

Most sources above give a :math:`\sigma` that is a constant, a data column, or an
expression over free parameters — a value that does **not** depend on the model's
output. Real measurements often violate that: instrument noise frequently has a
*combined* additive-plus-proportional form, a small floor plus a term that grows
with the signal,

.. math::

   \sigma_i \;=\; \sigma_\text{abs} \;+\; \sigma_\text{rel}\cdot y_i,

where :math:`y_i` is the observable's **predicted** value at point :math:`i`. This is
a likelihood parameter that is a *function of the system state*: as the fit moves and
the trajectory changes, every point's noise scale changes with it. The
``prediction_formula`` source expresses exactly this. Its expression may name

- **free parameters** — the estimated coefficients (:math:`\sigma_\text{abs}`,
  :math:`\sigma_\text{rel}`), resolved from the current parameter set; and
- **model entities** — any species, observable, or function the backend outputs,
  read from the *current simulation* at the scored point.

So ``noise_model y = gaussian, sigma = prediction_formula sd_abs__FREE + sd_rel__FREE
* y`` fits a Gaussian whose standard deviation scales with the predicted ``y``. It is
an *estimated* source (it references estimated coefficients), so the family's
normalizer is retained — the term that couples the residual and the scale is what lets
the data inform :math:`\sigma_\text{abs}` and :math:`\sigma_\text{rel}` jointly.

This differs from ``relative`` and ``column_mean``, which scale with the *measured*
data (a fixed quantity), and from ``formula``, whose expression sees only the free
parameters. ``prediction_formula`` is the only source that reads the simulation, and
it is what PEtab's affine ``noiseFormula`` (e.g. Raia_CancerResearch2011) imports as.

A prediction-dependent scale is differentiable on the **scalar gradient** path: an
L-BFGS fit (``job_type = lbfgs``) threads the scale formula's chain rule — the scale's
dependence on the prediction rides the same forward sensitivity as the residual —
through the gradient, so it optimizes the combined additive+proportional error model
directly. Because the retained ``+log σ`` normalizer is not a sum of squares, the fit
is not least-squares-exact, so the trust-region path (``job_type = trf``) refuses and
points at ``lbfgs``, and the **EFIM Fisher** path (``job_type = gntr``) likewise refuses
(the scale couples to the location, so the noise block is not diagonal) — a later
addition. Every gradient-free optimizer and sampler that evaluates the objective
directly is unaffected.

The cumulative prediction transform
------------------------------------

The ``cumulative`` flag on a per-observable ``noise_model`` line is a
**prediction transform**, not a noise family: a column declared cumulative has
its simulated prediction differenced row to row (a cumulative total becomes the
per-interval increment, with the first row kept as-is) before scoring. Because
it changes *how the prediction is formed from the simulation*, it is orthogonal
to the noise family and composes with any per-point model (``normal``,
``laplace``, ``neg_bin``, …). The older ``_Cum`` column-name convention,
recognized only by ``objfunc = neg_bin_dynamic``, is a compatibility bridge for
the same transform and is deliberately not widened to other families.

Choosing per observable
-----------------------

A single global ``objfunc`` (or a whole-fit ``noise_model`` line) applies the
same model to every data column. In edition-2 you can instead override
individual observables — a ``noise_model <observable> = …`` line per column — so
one fit can, say, score a well-calibrated readout with ``chi_sq`` and a noisy
count readout with ``neg_bin``. Any observable without its own line falls back to
the whole-fit default. The full grammar, with worked examples, is under the
:ref:`noise_model <noise_model_key>` key.

See also
--------

- :ref:`Objective functions <objective>` — the least-squares formulas.
- :ref:`noise_model <noise_model_key>` and :ref:`objfunc <objective>` —
  configuration syntax.
- :doc:`gradient_fitting` — residuals, Jacobians, and which families the
  gradient optimizers and HMC support (estimated :math:`\sigma`, Student-t,
  lognormal, Laplace).
- :ref:`priors` — the companion reference for a free parameter's prior, which
  includes any estimated noise parameter.
- :doc:`analytical_objectives` — supplying a closed-form log-likelihood
  directly, with no simulator in the loop.
- :ref:`API reference <noise_module>` — the :py:mod:`pybnf.noise` module
  docstrings for the per-point noise-model kernels.
