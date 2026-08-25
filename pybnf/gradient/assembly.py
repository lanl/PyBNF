"""Gaussian objective gradient + residual-Jacobian assembly (#449, #385).

Step C of the #385 gradient-plumbing epic. Given #447's per-experiment forward
output-sensitivity tensor (``Data.output_sensitivities``) and #448's per-experiment
routing (:class:`pybnf.gradient.routing.ExperimentRouting`), assemble -- for the
default Gaussian, LINEAR-scale, fixed-sigma objective, summed across experiments --
both forms of the gradient:

1. the **scalar** ``dF/du`` (for quasi-Newton / L-BFGS-B), and
2. the **residual vector + residual-Jacobian** (for trust-region least-squares,
   #386's primary path).

Convention pin (issue #449)
---------------------------
PyBNF's Gaussian loss is ``data_fit = (pred - obs)**2/(2 sigma**2)`` with
``mu = pred`` for the default (LINEAR, MEDIAN) family, so per scored point ``i``:

* residual ``rho_i = (pred_i - obs_i)/sigma_i``,  ``loss = 1/2 ||rho||**2``;
* residual-Jacobian (native param space) ``J_ij = (1/sigma_i) * factor_j *
  d pred_i/d theta_j`` -- ``d pred/d theta`` from #447's tensor, ``factor_j`` from
  #448's routing;
* scalar gradient ``dF/d theta = J^T rho`` (**not** ``2 J^T rho``).

``scipy.least_squares`` minimizes ``1/2 ||rho||**2`` with the same ``rho``/``J``, so
the residual form and the scalar form agree by construction -- the optimizer walks
the surface PyBNF reports. The per-point bootstrap weight ``w_i`` (1.0 unless
bootstrapping) is folded in as ``sqrt(w_i)`` on both ``rho_i`` and ``J_i``, so
``1/2 ||rho||**2`` stays ``sum_i w_i * data_fit_i`` -- exactly what ``evaluate``
sums (``eval_point * weight``).

Native -> sampling space (once, ADR-0029)
-----------------------------------------
``rho`` is scale-invariant; the Jacobian moves to the sampling space the optimizer
walks by ``J -> J @ diag(d theta/d u)``, applied **once** at the end. ``d theta/d u``
comes from each parameter's scale in closed form (``Scale.d_inverse``,
``priors/scale.py``): ``1`` for LINEAR (short-circuited), ``ln(10)*10**u`` for log10,
``exp(u)`` for ln. No autodiff, so no gradient fit built on the shipped scales needs
the optional ``pybnf[jax]`` extra (ADR-0087, #524); a custom scale that supplies only
an ``inverse_jax`` still falls back to ``jax.grad`` (the house pattern, ADR-0019).

Estimated noise scale (layer D, #451)
-------------------------------------
An estimated sigma -- the edition-2 ``noise_model = normal, sigma = fit <param>`` surface
(ADR-0021/0034), a freely-named free parameter; equivalently ``chi_sq_dynamic``'s legacy
``sigma__FREE`` default -- keeps the Gaussian normalizer, so the per-point loss is
``(pred-obs)**2/(2 sigma**2) + log sigma`` and gains a sigma column ``d loss/d sigma =
-(pred-obs)**2/sigma**3 + 1/sigma`` (``objective.noise_grad_point``). The routing binds
that free parameter by id (ADR-0034); estimated noise is matched by source *type*
(``FreeParameterSigma``), never by a name convention. ``+log sigma`` is **not** a sum
of squares, so it cannot live in the residual/Jacobian form: this assembly adds the
sigma column straight to the **scalar** gradient and leaves the residual-Jacobian a
faithful least-squares model of the *data fit* alone -- flagged by
``GradientResult.least_squares_exact`` (``False`` once any estimated scale is present),
so #386's trust-region path knows to use the scalar gradient (L-BFGS) for an
estimated-sigma fit. The free sigma routes to ``NONE`` in #448 (no model column), so
its gradient comes entirely from this normalizer + the sigma-dependence of the data
fit, never from the sensitivity tensor.

The estimated-scale column generalizes past a single free parameter: ``noise_grad_point`` returns
the full ``sum_p (dL/dp)*(dp/dtheta)`` vector, where each source supplies ``dp/dtheta``
(``sigma_sensitivity``). A single free sigma has ``dsigma/dname = 1`` and no sim coupling, so its
column is byte-identical to the historical scalar one. A **composite** sigma is the general chain
rule ``dsigma/dtheta = sum_symbol (dsigma/dsymbol)*(dsymbol/dtheta)``: a **PSet-only** formula
(``FormulaSigma`` / ``PerMeasurementFormulaSigma``, ADR-0044/0045/#505) puts ``dsigma/dcoeff`` on each
coefficient's column (the per-measurement variant binding a row token from ``exp_data``/``exp_row``);
a **prediction-dependent** sigma (``sigma = sigma_abs + sigma_rel*y``, a ``PredictionFormulaSigma``,
ADR-0075/0079) additionally chains ``dsigma/dprediction`` through the SAME ``raw_sens`` forward
sensitivity the residual rides -- so, unlike the others, it also perturbs the model-parameter columns.
Every estimated scale stays on the scalar path (the retained normalizer is not a square).

Trajectory transforms + normalization (layer F, #453)
-----------------------------------------------------
``_prediction`` may form the scored value from the raw observable through a per-observable
transform: a **cumulative -> incident** difference (ADR-0051), a **per-measurement** scale/
offset formula (ADR-0045), or an upstream ``Data``-level **normalization** (ADR-0053/0066). Each
makes ``∂pred/∂θ`` differ from the raw observable sensitivity, so the assembly reads a
``raw_sens(col, row)`` accessor (the #447 tensor, routing-factor-folded, with normalization's
own quotient/chain rule threaded in -- ``_normalized_sensitivity``, which for a *chain* of
``Data``-level transforms wraps the tensor accessor once per stage, each stage's rule read in
the values that stage consumed and produced, #539/ADR-0102) and hands it to the
objective's :meth:`~pybnf.objective.SummationObjective.prediction_sensitivity` seam, which
mirrors ``_prediction`` branch for branch (cumulative differences sensitivity rows; a per-
measurement formula chains its symbolic gradient through each referenced column's sensitivity
plus any estimated placeholder it names -- which, unlike a free σ, *does* enter ``∂pred/∂θ`` and
so lands in the residual-Jacobian). A plain column collapses to the raw sensitivity, so the
no-transform path is byte-identical.

The two ADR-0066 primitives close the same way (#533). A **floor** (``x' = x + rho*max(x)``) is
additive and separable, so ``∂x'_i/∂θ = s_i + rho*s_argmax`` -- one more term in
``_normalized_sensitivity``. An **analytic per-series scale** is the one transform that is not
per-point: its ``c*`` is profiled from the whole matched (sim, data) series, so the scored value
is ``c*(θ)·s_i(θ)`` and its sensitivity is the product rule ``c*·∂s_i/∂θ + s_i·∂c*/∂θ``. The
series-wide ``∂c*/∂θ`` is the closed-form derivative of the profiling condition
(:meth:`~pybnf.objective.SummationObjective.analytic_scale_sensitivity`) -- a geometric-mean
ratio for a log family, the least-squares optimum for a linear one -- computed once per
experiment here and shared by every point of the column. It does **not** vanish by the envelope
theorem: the profiling criterion is family-aware but σ-unweighted, so it is not in general the
objective's own minimizer over ``c``. Resolving which columns *this* experiment scales needs the
experiment's ``data_key``, which travels as the optional 4th element of each ``experiments``
item; without it a scaled column is refused rather than silently differentiated unscaled.

Asymmetric / non-Gaussian families (layer G, #454; least-squares residual #459)
-------------------------------------------------------------------------------
A family yields an **exact least-squares residual/Jacobian** iff its data fit reformulates as a
*smooth* half-square. Two do: the **Gaussian** (``data_fit = 1/2 rho**2``) and -- the #459
follow-up -- the **Student-t**, whose exact square-root-loss residual ``r = sign(z) sqrt(2
data_fit) = sign(z) sqrt((ν+1) log1p(z²/ν))`` satisfies both ``1/2 r² == data_fit`` *and*
``r · d r/d pred == d(data_fit)/d pred`` and is smooth through ``z=0`` (``r ~ sqrt((ν+1)/ν) z``,
downweighting the tails). Both route through ``residual_point`` (``NoiseModel.residual`` /
``d_residual_d_prediction``), contributing a residual row + native Jacobian row, so a fixed-scale
Student-t fit is ``least_squares_exact`` -- the Gaussian's exact-least-squares status recovered
for the robust family, and #386's LM/TRF can fit it directly.

**Laplace** has no such clean residual: its L1 data fit ``|·|/b`` gives ``sqrt(2 data_fit) ~
sqrt|z|``, a cusp with infinite slope at ``z=0``, so it stays scalar-only (the count family
likewise). Such a family's objective gradient is assembled from the **universal** scalar form
``sum_i w_i · d(data_fit_i)/d(pred_i) · d(pred_i)/d θ`` -- the per-family slope ``d(data_fit)/d(pred)``
(``objective.data_fit_grad_point`` -> ``NoiseModel.d_data_fit_d_prediction``) chained through the
same layer-F ``prediction_sensitivity`` ``d pred/d θ``, accumulated into a separate
``data_fit_gradient`` vector exactly as the estimated-noise column accumulates into
``noise_gradient``; the result is flagged not ``least_squares_exact`` (the residual/Jacobian then
model only the Gaussian/Student-t columns, if any), the signal #386's trust-region path uses the
scalar gradient (L-BFGS). The estimated-noise column generalizes per family throughout: Laplace's
``b``, and Student-t's ``sigma`` **and** ``df`` (the first multi-parameter estimated-noise
gradient, ADR-0058) -- a retained normalizer is never a square, so an estimated-scale fit is
inexact for *any* family (its data-fit residual, if it has one, still stacks; the normalizer
column rides ``noise_gradient``). The prediction (the MEDIAN or, layer G, the MEAN) and the noise
scale (linear or log, layer E) compose throughout; ``objective.has_least_squares_residual`` routes
each column. An all-Gaussian fit never touches ``data_fit_gradient`` (it stays zero), so that path
is byte-identical. The **negative-binomial**
count family rides the same scalar ``data_fit_gradient`` path (#458): its prediction slope is the
NB score chained through the median CDF-inversion implicit derivative, and its estimated dispersion
``r`` -- self-normalizing PMF, so the whole column is in the data fit -- the ``noise_gradient``
path. The dispersion column closes for MEAN and MEDIAN alike: a MEDIAN mean is solved from ``r``,
so its column folds in ``d mean/d r`` (the betainc first-parameter implicit derivative), the count
analogue of the mean-on-log offset coupling the location-scale families fold in (#385/#458).

Constraint / qualitative penalties (layer I, #456)
--------------------------------------------------
A fit may add qualitative / inequality constraints (BPSL ``.con`` / ``.prop`` files) whose
penalty is added to the objective. :func:`assemble_constraint_gradient` is the sibling assembler:
each constraint's penalty is a piecewise (static) or Gaussian-CDF (likelihood) function of an at-/
between-time readout ``q1 - q2``, so its gradient is that readout's forward sensitivity (read via
a ``(model, suffix, observable)``-keyed accessor over the #447 tensor + #448 routing) times the
local penalty slope (:meth:`~pybnf.constraint.Constraint.penalty_gradient`). Like an estimated-
noise normalizer, a penalty is not a sum of squares, so it lives on the scalar gradient only: a
fit with active constraints is not ``least_squares_exact``, and #386 adds this term to the
objective gradient.

Measurement-model layer (layer H, #455)
---------------------------------------
A scored observable may be a materialized **measurement-model** column (ADR-0036): an expression
``observableFormula`` the objective's :class:`~pybnf.measurement.base.MeasurementLayer` adds to the
trajectory before scoring (the SBML/Antimony / new-era PEtab path, where the SBML backend exposes
the same ``species:`` forward sensitivities the net backend does for observables). Such a column is
not in the #447 tensor, so the ``raw_sens`` accessor recognizes it and delegates to the model's
:meth:`~pybnf.measurement.base.MeasurementModel.prediction_sensitivity` -- the formula's exact chain
rule over each referenced column's sensitivity (read back through this same accessor, so routing /
normalization fold in) plus any fit parameter the formula names directly (an observation-model
scale/offset, which -- like a per-measurement scale -- enters ``∂pred/∂θ`` and lands in the
residual-Jacobian). A plain (non-measurement) column collapses to the tensor/normalized sensitivity,
so the no-measurement path is byte-identical.
"""

from dataclasses import dataclass

import numpy as np

from .errors import GradientNotSupported
from .routing import PARAM, IC, NONE
from ..printing import PybnfError


@dataclass
class GradientResult:
    """The assembled gradient of a Gaussian objective at one parameter point.

    ``residual`` is the stacked standardized residual ``rho`` (one entry per scored
    observation across all experiments, ``sqrt(weight)``-folded). ``jacobian`` is the
    matching ``(n_obs, n_param)`` residual-Jacobian **in sampling space** (the
    ``d theta/d u`` transform already applied). ``gradient`` is the scalar
    ``dF/d u`` over the free parameters, in ``param_names`` order.

    With a **fixed**-scale fit whose families all carry an exact least-squares residual -- the
    **Gaussian** (any scale/location) and, #459, the **Student-t** (its smooth square-root-loss
    residual) -- the data fit IS the whole objective, so the residual and scalar forms agree by
    construction (``gradient == jacobian.T @ residual``, ``0.5||rho||**2 == evaluate``) and
    ``least_squares_exact`` is ``True``. It is ``False`` once the residual-Jacobian is no longer
    the whole story:

    * an **estimated** noise parameter (layer D/G, #451/#454) -- its retained normalizer
      (``+log sigma``, ``log(2 b)``, the df-block) is not a square, so it is folded into the
      scalar ``gradient`` only and the residual-Jacobian models the data fit alone; or
    * a family with **no clean least-squares residual** (Laplace, whose L1 data fit is the cusp
      ``sqrt|z|``; the count family, layer G #454/#459) -- its data fit is not a smooth sum of
      squares, so it carries no residual at all and its whole data-fit gradient is on the scalar
      path.

    Either way the scalar ``gradient`` is complete (``jacobian.T @ residual`` over the
    residual-bearing columns -- Gaussian / Student-t -- if any, plus the data-fit and noise
    columns), and ``least_squares_exact = False`` is the signal that #386's trust-region step must
    consume it rather than the bare residual.
    """
    residual: np.ndarray      # (n_obs,)
    jacobian: np.ndarray      # (n_obs, n_param), sampling space
    gradient: np.ndarray      # (n_param,) = J^T rho + estimated-noise columns
    param_names: list         # free-parameter order of the columns / gradient
    least_squares_exact: bool = True   # False once an estimated sigma is present
    #: The expected-Fisher / Gauss-Newton Hessian (n_param, n_param), sampling space --
    #: attached only on the EFIM trust-region path (``job_type = gntr``, #481/#488) by
    #: :func:`assemble_gradient_and_fisher_hessian`; ``None`` for ``trf`` / ``lbfgs``, which
    #: never form it.
    hessian: np.ndarray = None


def assemble_gaussian_gradient(objective, experiments, free_params):
    """Assemble the scalar gradient and residual-Jacobian, summed across experiments.

    ``objective`` is the fit's :class:`~pybnf.objective.LikelihoodObjective`; it supplies
    each residual-bearing point's residual through ``residual_point`` (a Gaussian or, #459, a
    Student-t -- the families whose data fit is a smooth half-square), each no-residual (Laplace /
    count) point's data-fit gradient through ``data_fit_grad_point`` (routed by
    ``has_least_squares_residual``), and any estimated-noise gradient column through
    ``noise_grad_point`` (each gating the supported configuration -- a Gaussian / Laplace /
    Student-t / negative-binomial family, MEDIAN or MEAN, any noise scale, noise fixed or single
    free parameters -- raising :class:`GradientNotSupported` otherwise). ``experiments`` is an
    iterable of
    ``(sim_data, exp_data, routing[, data_key])`` items -- one per scored model/condition; each
    ``sim_data`` must carry the #447 ``output_sensitivities`` payload (the gradient
    path active), and ``routing`` is that experiment's
    :class:`~pybnf.gradient.routing.ExperimentRouting`. The optional 4th element is that
    experiment's ``data_key`` (the suffix ``evaluate`` scores it under), needed only to resolve an
    analytic per-series ``scale`` (ADR-0066, #533): omitting it is byte-identical for every other
    fit, and refuses a scaled column rather than differentiating it unscaled. ``free_params`` is
    the ordered
    list of :class:`~pybnf.pset.FreeParameter` defining the ``u``-vector: it fixes the
    column order of the Jacobian and the entries of the scalar gradient, and supplies
    each parameter's scale (current value -> ``d theta/d u``).

    Returns a :class:`GradientResult`. The per-experiment routing is built once by the
    caller (#386) -- it depends only on model structure, conditions, and free-parameter
    ids, never on the parameter values -- so this per-evaluation assembly only reads
    the freshly simulated sensitivity tensors.
    """
    return _assemble_gradient(objective, experiments, free_params, include_fisher=False)


def assemble_gradient_and_fisher_hessian(objective, experiments, free_params):
    """Assemble a :class:`GradientResult` and attach its expected-Fisher Hessian in one pass.

    This is the ``gntr`` objective-assembly path (#488). It produces the same residual,
    Jacobian, scalar gradient, and Fisher Hessian as calling :func:`assemble_gaussian_gradient`
    and :func:`assemble_fisher_hessian` separately, but walks each scored point only once. The
    shared point walk resolves the simulation row, filters missing observations, builds the raw
    sensitivity accessor, and calls ``prediction_sensitivity`` once before feeding both the
    gradient and curvature accumulators.
    """
    return _assemble_gradient(objective, experiments, free_params, include_fisher=True)


def _assemble_gradient(objective, experiments, free_params, include_fisher):
    """Implementation shared by the gradient-only and combined gradient/Fisher assemblers."""
    names = [p.name for p in free_params]
    index = {name: j for j, name in enumerate(names)}
    n_param = len(free_params)

    # An estimated free noise scale (a free sigma) reads its value from the objective's
    # per-evaluation pset map (ADR-0021); seed it from the current free-parameter point
    # so the loss is scored at u. Merged over any existing map so a prior evaluate's
    # fixed parameters survive (a fixed-sigma fit never reads it -- harmless there).
    existing = getattr(objective, '_pset_values', None) or {}
    objective._pset_values = {**existing, **{p.name: p.value for p in free_params}}
    _seed_profiled_noise(objective, experiments)

    rho_rows = []
    jac_rows = []
    # The estimated-noise (sigma) columns of the scalar gradient -- accumulated apart
    # from the residual-Jacobian because the normalizer ``+log sigma`` is not a square
    # (layer D, #451). Zero for a fixed-sigma fit.
    noise_gradient = np.zeros(n_param)
    # The no-residual-family data-fit gradient (layer G, #454/#459): a Laplace (its L1 data
    # fit is the cusp sqrt|z|) or count column has no least-squares residual, so its data fit
    # contributes its scalar gradient ``sum_i w_i * d(data_fit_i)/d(pred_i) * d(pred_i)/d theta``
    # straight here. Zero for an all-Gaussian / Student-t fit (the residual-bearing path).
    data_fit_gradient = np.zeros(n_param)
    hessian = np.zeros((n_param, n_param)) if include_fisher else None
    least_squares_exact = True
    for sim_data, exp_data, routing, *rest in experiments:
        if _accumulate_experiment(objective, sim_data, exp_data, routing, index, n_param,
                                  rho_rows, jac_rows, noise_gradient, data_fit_gradient,
                                  hessian=hessian, data_key=rest[0] if rest else None):
            least_squares_exact = False

    rho = np.asarray(rho_rows, dtype=float)
    jac = np.asarray(jac_rows, dtype=float).reshape(len(rho_rows), n_param)

    # Native -> sampling space, applied exactly once (ADR-0029): rho is invariant, each
    # Jacobian column scales by d theta_j/d u_j at the current value, and the two scalar
    # gradient accumulators (a free sigma's noise column; an asymmetric family's data-fit
    # column) take the same per-parameter chain factor.
    factors = _sampling_scale_factors(free_params)
    jac = jac * factors[np.newaxis, :]
    noise_gradient = noise_gradient * factors
    data_fit_gradient = data_fit_gradient * factors
    if hessian is not None:
        hessian = hessian * np.outer(factors, factors)

    gradient = jac.T @ rho + data_fit_gradient + noise_gradient
    return GradientResult(residual=rho, jacobian=jac, gradient=gradient,
                          param_names=names, least_squares_exact=least_squares_exact,
                          hessian=hessian)


def _seed_profiled_noise(objective, experiments):
    """Put every analytically profiled noise scale at its MLE before the point walk (ADR-0108,
    #562), so the seams below differentiate the loss at ``sigma_hat(theta)`` -- the value the
    fit's own scoring uses at this point.

    A profiled scale is not among ``free_params`` (the search never carries it), so the
    ``_pset_values`` seeding above cannot supply it; without this the noise sources would read a
    stale value from a previous evaluation, or none at all. The gradient itself needs no new
    term: ``sigma_hat`` minimizes the objective over the scale, so by the envelope theorem
    ``d/dtheta NLL*(theta) = partial/partial theta NLL(theta, sigma)`` at ``sigma = sigma_hat``
    -- which is exactly the assembled frozen-scale gradient with the profiled columns dropped
    (``objective.noise_grad_point``).

    A no-op for a fit that profiles nothing. A degenerate group (an unbounded or non-finite
    profile) has no gradient to assemble, so it refuses rather than differentiating a value that
    does not exist."""
    if not getattr(objective, '_profiled_noise_params', None):
        return
    triples = [(sim_data, exp_data, rest[0] if rest else None)
               for sim_data, exp_data, _routing, *rest in experiments]
    if not objective._resolve_profiled_noise(triples):
        raise GradientNotSupported(
            "An analytically profiled noise scale (noise_profiling = 1, ADR-0108) is degenerate "
            "at this point -- its group's residual is zero or not finite, so the profiled "
            "likelihood has no finite value and no gradient. Score this point on the "
            "gradient-free path, or drop noise_profiling for this fit.")


def _accumulate_experiment(objective, sim_data, exp_data, routing, index, n_param,
                           rho_rows, jac_rows, noise_gradient, data_fit_gradient, hessian=None,
                           data_key=None):
    """Append one experiment's per-point residual and native-space Jacobian rows (for a
    least-squares column -- Gaussian or Student-t, #459), accumulate any estimated-noise gradient
    columns into ``noise_gradient``, and accumulate a no-residual family's scalar data-fit gradient
    into ``data_fit_gradient`` (layer G, #454).

    Mirrors ``SummationObjective.evaluate``'s point loop exactly -- same independent
    variable, same comparable-column intersection, same NaN skip, same
    ``_sim_row_for`` row match -- so the gradient is assembled over precisely the
    points PyBNF scores. Columns are walked in sorted order for a deterministic
    observation axis (matching ``evaluate_pointwise``). Returns ``True`` iff this
    experiment made the result not ``least_squares_exact`` -- an estimated-noise column or a
    no-residual (Laplace / count) family was present (so the caller can clear the flag)."""
    inexact = False
    quantity = "gradient and Hessian" if hessian is not None else "gradient"
    for point in _iter_scored_points(
            objective, sim_data, exp_data, routing, index, n_param, quantity, data_key):
        if _accumulate_gradient_point(
                objective, sim_data, exp_data, index, point, rho_rows, jac_rows,
                noise_gradient, data_fit_gradient):
            inexact = True
        if hessian is not None:
            _accumulate_fisher_point(objective, sim_data, exp_data, index, point, hessian)
    return inexact


def _iter_scored_points(objective, sim_data, exp_data, routing, index, n_param, quantity,
                        data_key=None):
    """Yield each scored point and its native-space prediction sensitivity once (#488).

    This is the point-selection scaffold shared by the gradient-only, Fisher-only, and combined
    ``gntr`` assemblers: independent-variable resolution, comparable-column intersection, row
    matching, NaN filtering, raw-sensitivity access, and the trajectory-transform chain rule all
    live here. Each item is ``(sim_row, rownum, col_name, weight, dpred_dtheta, raw_sens)``;
    ``raw_sens`` travels with the point because estimated-noise gradient and Fisher blocks use the
    same accessor when differentiating their scale sources.

    ``data_key`` is the experiment's scoring key (the ``evaluate`` ``data_key`` -- the suffix),
    needed only to resolve a per-series **analytic scale** (ADR-0066, #533): its profiled ``c*``
    and ``∂c*/∂θ`` are properties of the whole series, so they are computed once here and handed
    to every point of the column. A fit that scales nothing walks no extra points.
    """
    sens = sim_data.output_sensitivities
    if sens is None:
        raise GradientNotSupported(
            "An experiment carries no forward-sensitivity tensor; enable the gradient "
            "path (apply_routing) on every scored model before assembling the %s." % quantity)

    indvar = min(exp_data.cols, key=exp_data.cols.get)
    comparable = set(sim_data.cols) | set(objective._per_measurement_models)
    compare_cols = set(exp_data.cols).intersection(comparable)
    compare_cols.discard(indvar)

    # The per-column sensitivity accessor (#453): ∂(column as _prediction sees it)/∂θ -- the
    # #447 tensor read at a row, routing-factor-folded, NONE/pinned parameters at 0, with any
    # Data-level normalization and measurement-model chain rules folded in. Built once per
    # experiment; the combined gradient/Fisher path then reuses it for both consumers.
    raw_sens = _raw_sensitivity_accessor(objective, sim_data, sens, routing, index, n_param, indvar)

    # The per-series analytic scale's (c*, ∂c*/∂θ) for each column this experiment scales
    # (ADR-0066, #533) -- one profiling walk, shared by every point of the column, exactly the
    # points ``evaluate`` profiles from. ``{}`` for an unscaled experiment (the common case), and
    # ``None`` when the caller supplied no data_key, which ``prediction_sensitivity`` refuses
    # rather than silently differentiating an unscaled prediction.
    scale_terms = (objective.analytic_scale_sensitivity(
        sim_data, exp_data, indvar, compare_cols, data_key, raw_sens, index)
        if data_key is not None else None)
    if scale_terms is not None:
        # Point the objective's own scoring seam at THIS experiment's factors, exactly as
        # ``evaluate`` does at the top of its loop: the residual / data-fit / Fisher consumers
        # below read the prediction through ``_prediction``, which multiplies by them. Without
        # this the residual would carry whichever experiment was scored last. ``{}`` for an
        # unscaled experiment, which is what an unscaled fit already holds.
        objective._scale_factors = {col: c for col, (c, _dc) in scale_terms.items()}

    for rownum in range(exp_data.data.shape[0]):
        sim_row = objective._sim_row_for(sim_data, exp_data, indvar, rownum, show_warnings=False)
        for col_name in sorted(compare_cols):
            observation = exp_data.data[rownum, exp_data.cols[col_name]]
            if np.isnan(observation):
                continue
            weight = exp_data.weights[rownum, exp_data.cols[col_name]]
            # ∂pred/∂θ through the objective's transform seam (plain / cumulative / per-
            # measurement / analytic scale; #453/#533), so every consumer differentiates exactly
            # what is scored. A pinned parameter and a model-unbound nuisance carry 0 in raw_sens.
            dpred_dtheta = objective.prediction_sensitivity(
                sim_data, sim_row, col_name, exp_data, rownum, raw_sens, index,
                scale_terms=scale_terms)
            yield sim_row, rownum, col_name, weight, dpred_dtheta, raw_sens


def _accumulate_gradient_point(objective, sim_data, exp_data, index, point,
                               rho_rows, jac_rows, noise_gradient, data_fit_gradient):
    """Consume one :func:`_iter_scored_points` item for the scalar/residual gradient."""
    sim_row, rownum, col_name, weight, dpred_dtheta, raw_sens = point
    sqrt_w = np.sqrt(weight)
    # Layer D/G (#451/#454), ADR-0079: an estimated noise scale contributes its full
    # ``sum_p (dL/dp) * (dp/dtheta)`` vector directly to the scalar gradient. The normalizer is
    # not a square, so this stays off the residual-Jacobian. ``None`` means fixed noise.
    noise_vec = objective.noise_grad_point(
        sim_data, exp_data, sim_row, rownum, col_name, raw_sens, index)
    inexact = noise_vec is not None
    if noise_vec is not None:
        noise_gradient += weight * noise_vec

    if objective.has_least_squares_residual(col_name):
        # Gaussian and Student-t data fits are smooth half-squares: append their exact residual
        # and native Jacobian rows, sqrt(weight)-folded.
        rho, drho_dpred = objective.residual_point(
            sim_data, exp_data, sim_row, rownum, col_name)
        rho_rows.append(sqrt_w * rho)
        jac_rows.append(sqrt_w * drho_dpred * dpred_dtheta)
    else:
        # Laplace and count families have no clean least-squares residual; accumulate their
        # complete data-fit gradient on the scalar path.
        dfit_dpred = objective.data_fit_grad_point(
            sim_data, exp_data, sim_row, rownum, col_name)
        data_fit_gradient += weight * dfit_dpred * dpred_dtheta
        inexact = True
    return inexact


def _accumulate_fisher_point(objective, sim_data, exp_data, index, point, hessian):
    """Consume one :func:`_iter_scored_points` item for its expected-Fisher terms."""
    sim_row, rownum, col_name, weight, dpred_dtheta, raw_sens = point
    # Location block: w_i * kappa_i * outer(s_i, s_i).
    kappa = objective.location_fisher_point(sim_data, exp_data, sim_row, rownum, col_name)
    if kappa:
        hessian += (weight * kappa) * np.outer(dpred_dtheta, dpred_dtheta)
    # Noise block (ADR-0080): ``sum_p I_scale_p * outer(g_i^p, g_i^p)``. A single free
    # sigma supplies a diagonal unit-vector block; prediction-dependent scales may couple axes.
    noise_block = objective.noise_fisher_point(
        sim_data, exp_data, sim_row, rownum, col_name, raw_sens, index)
    if noise_block is not None:
        hessian += weight * noise_block


def iter_fisher_points(objective, experiments, free_params):
    """Yield each scored point's **own** expected-Fisher block, instead of their sum (#574).

    :func:`assemble_fisher_hessian` adds every point's rank-1 terms into one matrix. Optimal
    experimental design needs the terms kept apart: a design is a *subset* of measurements, so
    scoring one means adding up the blocks of the points it contains and leaving the rest out.
    Because the Fisher information is a plain sum over points, that is all a design score is.

    Each item is ``(experiment_index, exp_row, col_name, block)``, where ``block`` is the
    ``(n_param, n_param)`` matrix that point contributes -- the same location + noise terms
    :func:`_accumulate_fisher_point` adds, already weight-folded and already in sampling space
    (the ``d theta/d u`` factors applied on both axes, ADR-0029), so the blocks add straight onto
    a Hessian :func:`assemble_fisher_hessian` returned. Summing every yielded block reproduces
    that Hessian.

    ``experiments`` and ``free_params`` are exactly what the other assemblers take. The point
    walk is the shared :func:`_iter_scored_points` scaffold, so the points, the row matching, the
    NaN skip and the transform chain rule are identical to the ones the fit scores."""
    names = [p.name for p in free_params]
    index = {name: j for j, name in enumerate(names)}
    n_param = len(free_params)

    # Seed the estimated-noise / profiled-scale reads from this point exactly as
    # assemble_fisher_hessian does, so a free sigma resolves the same way here.
    existing = getattr(objective, '_pset_values', None) or {}
    objective._pset_values = {**existing, **{p.name: p.value for p in free_params}}
    _seed_profiled_noise(objective, experiments)

    factors = _sampling_scale_factors(free_params)
    sampling = np.outer(factors, factors)
    for exp_index, (sim_data, exp_data, routing, *rest) in enumerate(experiments):
        data_key = rest[0] if rest else None
        for point in _iter_scored_points(objective, sim_data, exp_data, routing, index,
                                         n_param, "Fisher information", data_key):
            block = np.zeros((n_param, n_param))
            _accumulate_fisher_point(objective, sim_data, exp_data, index, point, block)
            yield exp_index, point[1], point[2], block * sampling


def assemble_fisher_hessian(objective, experiments, free_params):
    """Assemble the expected-Fisher / Gauss-Newton **Hessian** ``H`` (n_param x n_param),
    summed across experiments. This standalone API produces the same curvature the combined
    :func:`assemble_gradient_and_fisher_hessian` path feeds to the EFIM trust-region optimizer
    (``job_type = gntr``, #481/#488).

    ``H = sum_i w_i [ kappa_i * outer(s_i, s_i)  +  sum_p I_scale_p * outer(g_i^p, g_i^p) ]``
    where ``s_i = d(prediction_i)/d(theta)`` is the same forward sensitivity the gradient uses
    (through the objective's ``prediction_sensitivity`` seam), ``kappa_i`` the per-point location
    Fisher (``objective.location_fisher_point``: ``(d rho/d pred)**2`` for a residual-bearing
    Gaussian/Student-t column, the family's ``location_fisher`` for Laplace/count), ``I_scale_p``
    each estimated noise parameter's expected Fisher, and ``g_i^p = d(noise_param_p)/d(theta)`` its
    scale sensitivity -- the whole ``sum_p I_scale_p * outer(g_i^p, g_i^p)`` noise block returned by
    ``objective.noise_fisher_point`` (ADR-0080). Every rank-1 term is PSD (``kappa >= 0``,
    ``I_scale >= 0``), so ``H`` is PSD by construction. For a bare **free** sigma ``g_i^p`` is the
    unit vector ``e_p`` (the noise parameter is model-unbound, 0 in ``s_i``), so its block is the
    historical diagonal entry ``I_scale * outer(e_p, e_p)`` and the estimated-sigma Hessian is
    block-diagonal, exactly the Fisher predicts. For a **prediction-dependent** sigma
    (``sigma = sigma_abs + sigma_rel*y``, ADR-0075/0079) ``g_i^p`` also carries model-parameter
    columns (the scale rides the prediction), so ``outer(g_i^p, g_i^p)`` produces the genuine
    location↔scale coupling off the diagonal -- a strict superset of the diagonal cut.

    ``experiments`` is the same ``(sim_data, exp_data, routing[, data_key])`` iterable
    :func:`assemble_gaussian_gradient` consumes, ``free_params`` the same ordered free-
    parameter list. Mirrors that assembler's point loop exactly (same points, same
    ``raw_sens`` accessor, same ``prediction_sensitivity``), so the Hessian is formed over
    precisely the points the gradient and objective score. The native -> sampling transform
    (ADR-0029) is the gradient's ``d theta/d u`` factor applied on **both** axes:
    ``H <- diag(f) H diag(f)``. Raises :class:`GradientNotSupported` for a configuration whose
    Fisher this cut does not assemble (a MEDIAN-centered count, a MEAN-on-log estimated scale,
    ...), so the fit refuses the EFIM step with a pointer to the L-BFGS-B path."""
    names = [p.name for p in free_params]
    index = {name: j for j, name in enumerate(names)}
    n_param = len(free_params)

    # An estimated free noise scale reads its value from the objective's per-evaluation pset
    # map (ADR-0021); seed it from the current point exactly as assemble_gaussian_gradient
    # does, so the Fisher is formed at u (idempotent -- the gradient assembly already seeded it).
    existing = getattr(objective, '_pset_values', None) or {}
    objective._pset_values = {**existing, **{p.name: p.value for p in free_params}}
    _seed_profiled_noise(objective, experiments)

    hessian = np.zeros((n_param, n_param))
    for sim_data, exp_data, routing, *rest in experiments:
        _accumulate_experiment_fisher(objective, sim_data, exp_data, routing, index, n_param,
                                      hessian, data_key=rest[0] if rest else None)

    # Native -> sampling space, applied once on both axes (ADR-0029): the same per-parameter
    # d theta/d u factor the gradient scales its columns by, here as an outer product.
    factors = _sampling_scale_factors(free_params)
    return hessian * np.outer(factors, factors)


def _accumulate_experiment_fisher(objective, sim_data, exp_data, routing, index, n_param,
                                  hessian, data_key=None):
    """Accumulate one experiment's per-point Fisher rank-1 terms into ``hessian`` (the
    curvature twin of :func:`_accumulate_experiment`). Same independent variable, same
    comparable-column intersection, same NaN skip, same ``_sim_row_for`` row match, same
    sorted-column walk -- so the Hessian is assembled over precisely the points the gradient
    is. The location block reads ``kappa_i`` (``location_fisher_point``) and the noise block the
    per-point matrix ``sum_p I_scale_p * outer(g_i^p, g_i^p)`` (``noise_fisher_point``, ADR-0080)."""
    for point in _iter_scored_points(
            objective, sim_data, exp_data, routing, index, n_param, "Hessian", data_key):
        _accumulate_fisher_point(objective, sim_data, exp_data, index, point, hessian)


def _sensitivity(sens, selector, contribution, sim_row, free_param):
    """The native forward sensitivity ``d(observable)/d(routed entity)`` at one time row.

    Reads the parameter axis (``sensitivity_params``) for a PARAM contribution and the
    initial-condition axis (``sensitivity_ic``) for an IC contribution, addressing the entity by
    the contribution's ``key`` (parameter id, or species for an IC). ``free_param`` names the
    routed free parameter for the diagnostic when a requested column is absent."""
    if contribution.target == PARAM:
        axis, labels = 'parameter', sens.param_names
    else:  # IC
        axis, labels = 'ic', sens.ic_species
    if contribution.key not in labels:
        raise GradientNotSupported(
            "Free parameter '%s' routes to %s '%s', but the simulation's "
            "sensitivity tensor has no such column (axis labels: %s). Apply the same "
            "routing to the model before running it (apply_routing)."
            % (free_param, axis, contribution.key, ', '.join(map(str, labels)) or '(none)'))
    column = sens.slice_for(selector, axis=axis)   # (n_times, n_axis)
    return column[sim_row, labels.index(contribution.key)]


def _raw_sensitivity_accessor(objective, sim_data, sens, routing, index, n_param, indvar):
    """Build ``raw_sens(col_name, row) -> (n_param,)``: native-space ``∂(that column as
    ``_prediction`` reads it)/∂θ`` (#453), the seam the objective's
    :meth:`~pybnf.objective.SummationObjective.prediction_sensitivity` composes transforms on.

    The base is the #447 forward tensor read at one row, each routed parameter's column scaled
    by its condition factor, with a pinned (factor 0) or model-unbound (``NONE``, e.g. a free
    σ) parameter left at 0 -- exactly the per-parameter Jacobian the assembly built inline
    before, now vectorised. The independent variable is θ-independent (sensitivity 0). When the
    column was normalized (ADR-0053), the normalizer's own θ-dependence is threaded here so the
    caller sees ``∂(normalized column)/∂θ`` and every downstream transform composes correctly
    (scoring applies normalize -> ``_prediction``).

    A **materialized measurement-model column** (ADR-0036, layer H #455) -- an expression
    ``observableFormula`` the objective's :class:`~pybnf.measurement.base.MeasurementLayer`
    added to the trajectory before scoring (the SBML/Antimony / new-era PEtab path) -- is not in
    the sensitivity tensor; its derivative is the formula's chain rule over the raw columns it
    references, so it is delegated to the model's ``prediction_sensitivity``, which calls back
    into this same accessor for each referenced column (a species/observable that *is* in the
    tensor, normalization-folded). A plain column has no measurement model and collapses to the
    tensor/normalized sensitivity, so the no-measurement path is byte-identical.

    The routing is checked here -- once per experiment, before any point is read -- for the one
    structural property ``tensor_sens`` relies on and cannot itself detect: each route names
    each native column exactly once, so summing a route's contributions counts each column
    exactly once (:meth:`~pybnf.gradient.routing.ExperimentRouting.check_column_multiplicity`,
    #537). A column added twice is a clean integer multiple of the true derivative, which no
    objective value can reveal -- the fit just walks a scaled surface -- so it is asserted
    rather than assumed."""
    routing.check_column_multiplicity()
    _check_axes_are_not_the_same_derivative(sens, routing)
    norm = sim_data.normalization or {}
    normalized = {}   # col -> the folded chain accessor, built on first use and reused per row
    measurement = getattr(objective, 'measurement', None)
    measurement_models = ({mm.observable_id: mm for mm in measurement.models}
                          if measurement else {})
    pset_values = getattr(objective, '_pset_values', None) or {}

    def tensor_sens(col_name, row):
        if col_name == indvar:
            return np.zeros(n_param)   # the independent variable does not move with θ
        vec = np.zeros(n_param)
        selector = _selector_for(sens, col_name)
        for name, route in routing.routes.items():
            total = 0.0
            for c in route.contributions:
                if c.target == NONE or c.factor == 0.0:
                    continue
                total += c.factor * _sensitivity(sens, selector, c, row, name)
            vec[index[name]] = total
        return vec

    def raw_sens(col_name, row):
        model = measurement_models.get(col_name)
        if model is not None:
            return model.prediction_sensitivity(sim_data, row, pset_values, raw_sens, index)
        records = norm.get(col_name)
        if not records:
            return tensor_sens(col_name, row)
        if col_name not in normalized:
            normalized[col_name] = _normalized_sensitivity(records, col_name, sim_data, tensor_sens)
        return normalized[col_name](col_name, row)

    return raw_sens


def _check_axes_are_not_the_same_derivative(sens, routing):
    """Refuse a route whose parameter axis and initial-condition axis are the *same* number (#537).

    A free parameter that seeds a species initial value can reach the trajectory on two axes:
    its own ``sensitivity_params`` column, and the ``sensitivity_ic`` column of the species it
    seeds. The router keeps both when the model says the ODE right-hand side also reads the
    parameter (ADR-0097, #535), because then the parameter axis carries the right-hand-side path
    and the initial-condition terms carry the seeding -- two genuinely different quantities whose
    sum is the derivative.

    That is a statement about the *backend*, not a theorem. bngsim seeds ``∂x(0)/∂p`` into the
    parameter axis as well (lanl/bngsim#43, widened to compound initialAssignment expressions by
    lanl/bngsim#147), and where it does, the two axes are not complementary halves -- they are
    the same derivative, and summing them doubles the column. On
    ``Raia_CancerResearch2011`` the two are byte-identical, and forcing both into the route
    reproduces #537's signature exactly: ``init_Rec_i`` at 2.00000x its central difference, every
    other column untouched.

    So it is checked rather than assumed, numerically and per experiment: for a route holding
    both, compare the parameter slice against the **sum of the initial-condition terms the route
    is about to add**, each with its own chain-rule factor. Agreement means the parameter axis
    is *entirely* seeding -- the right-hand-side path is zero -- so the two are one derivative
    and the sum would double it.

    Comparing against the weighted sum rather than a single slice is what makes this work for a
    non-unit seed. Where the coefficient is exactly 1 the two columns are the same IVP and
    CVODES returns them bit-for-bit (Raia); where it is not (``X(0) = a*X0`` gives
    ``d_param[X0] = 3.0 * d_ic[X]``) they agree to roundoff instead, which an equality test
    misses -- lanl/bngsim#155. Hence a tight relative tolerance: two genuinely independent
    derivatives of a live model do not track each other to 1e-12 across every selector and row.

    What this **cannot** catch is a parameter that seeds *and* drives the right-hand side, where
    ``d_param`` is ``RHS + seeding`` and overlaps the initial-condition terms only partially --
    no comparison of totals can see a partial overlap. That case is gated at routing time
    instead (:func:`~pybnf.gradient.routing.route_for_model`) and closes properly once
    lanl/bngsim#155 exposes the seed matrix. ``Fiedler_BMCSystBiol2016`` is that shape and
    passes here, correctly.
    """
    if sens.d_param is None or sens.d_ic is None:
        return   # only one axis was computed; nothing can be summed twice
    for name, route in routing.routes.items():
        own = next((c for c in route.contributions
                    if c.target == PARAM and c.key == name and c.requested), None)
        if own is None or own.key not in sens.param_names:
            continue
        ic_terms = [c for c in route.contributions
                    if c.target == IC and c.requested and c.key in sens.ic_species]
        if not ic_terms:
            continue
        # Both sides carry this experiment's condition factor, so scale the parameter slice by
        # its own contribution's factor to compare like with like.
        p_col = own.factor * sens.d_param[:, :, sens.param_names.index(own.key)]
        ic_sum = sum(c.factor * sens.d_ic[:, :, sens.ic_species.index(c.key)] for c in ic_terms)
        scale = float(np.abs(p_col).max()) if p_col.shape == ic_sum.shape else 0.0
        if scale == 0.0 or np.abs(p_col - ic_sum).max() > 1e-12 * scale:
            continue
        raise PybnfError(
            "Free parameter '%s' routes to both the parameter axis '%s' and the "
            "initial-condition axis/axes %s, but the simulation returns the same numbers for "
            "the two (agreeing to %.1e relative) -- they are one derivative, not the "
            "right-hand-side and seeding halves of one, so the assembly would sum it twice and "
            "report this column at an integer multiple of its true value. The backend seeds "
            "d(x(0))/d('%s') into the parameter axis, which already carries the whole "
            "derivative for a parameter the right-hand side does not otherwise read; the "
            "routing should drop that axis (ADR-0097). See lanl/PyBNF#537, lanl/bngsim#155."
            % (name, own.key, ', '.join(repr(c.key) for c in ic_terms),
               float(np.abs(p_col - ic_sum).max()) / scale, name))


def _normalized_sensitivity(records, col_name, sim_data, tensor_sens):
    """Fold a column's normalization **chain** into one ``∂(normalized col)/∂θ`` accessor
    (ADR-0053/0066, #539).

    ``normalization`` rescales the predicted column by a θ-dependent ``N(θ)`` read off the
    moving trajectory, so ``∂(raw/N)/∂θ`` is a quotient/chain rule coupling the scored row with
    the row(s) ``N`` is read from. A chain (``floor 0.03, peak``) applies two or more such
    transforms in order, and each one's rule is that same closed form read in *its own* inputs:
    the previous stage's per-row sensitivities, and its own output values. So the fold is
    literal -- start from the raw #447 tensor accessor and wrap it once per stage, each wrapper
    reading the accessor the previous stage returned. A single transform is the one-iteration
    case and threads exactly the rule it always did.

    The values a stage's rule reads are its **output** column: retained on the record when a
    later transform overwrote them (:class:`~pybnf.data.NormalizationRecord.values`), and for
    the last stage read back from the now-rescaled ``Data`` -- so a column normalized once
    retains nothing. See :class:`~pybnf.data.NormalizationRecord` for each method's closed form.

    Returns an accessor with ``tensor_sens``'s own ``(col_name, row)`` signature, which is what
    lets each stage call its predecessor exactly as the un-chained rule called the tensor."""
    final = sim_data.data[:, sim_data.cols[col_name]]
    stage_sens = tensor_sens
    for i, record in enumerate(records):
        values = record.values if i < len(records) - 1 else final
        if values is None:
            # Only reachable if a caller normalized the column twice without going through
            # Data's own normalize_* methods, which hand each overwritten stage its values.
            raise GradientNotSupported(
                "Column '%s' went through a chain of normalizations (%s) but stage %d ('%s') "
                "kept none of the values it produced, which its own chain rule reads, so the "
                "chain cannot be folded (issue #539). Use a single normalization per column, or "
                "a gradient-free step."
                % (col_name, ' -> '.join(r.method for r in records), i + 1, record.method))
        stage_sens = _stage_sensitivity(record, values, stage_sens)
    return stage_sens


def _stage_sensitivity(record, normed, previous):
    """One stage of a normalization chain: ``∂(this stage's output)/∂θ`` from ``previous``, the
    accessor for the values it consumed, and ``normed``, the values it produced.

    Memoized per row, because a stage reads its predecessor at rows other than the scored one --
    the reference row a divisor is read from, and for ``zero`` *every* row. So a ``zero`` mid-
    chain asks the stage below it for the whole column, once per scored row, and that stage asks
    the one below it; without the memo the work would multiply down the chain per scored row.
    With it each (stage, row) is computed once per experiment per evaluation. The tensor at the
    bottom is not memoized, so a single transform costs exactly what it always did."""
    cache = {}   # keyed by row alone: a stage belongs to the one column its chain normalizes

    def stage(col_name, row):
        if row not in cache:
            cache[row] = _stage_rule(record, col_name, row, normed, previous)
        return cache[row]

    return stage


def _stage_rule(record, col_name, row, normed, tensor_sens):
    """The closed-form rule for one recorded transform (:class:`~pybnf.data.NormalizationRecord`),
    read in the values it consumed (``tensor_sens``, the previous stage's accessor -- the #447
    tensor itself for the first transform of a column) and the values it produced (``normed``)."""
    s_i = tensor_sens(col_name, row)
    if record.method == 'floor':
        # x' = x + rho*max(x) (ADR-0066): additive and separable, so ∂x'_i/∂θ = s_i + rho*s_argmax
        # -- the scored row's own sensitivity plus rho times the row the max is read from
        # (recorded as ref_row; the argmax is unchanged by a constant shift). Like ``peak``'s
        # divisor, the max is differentiated at its achieving row (a.e. valid).
        return s_i + record.rho * tensor_sens(col_name, record.ref_row)
    if record.method == 'zero':
        return _zscore_sensitivity(record, col_name, row, tensor_sens, normed, s_i)
    # peak / init / unit: a two-row (+ optional baseline) quotient rule.
    s_base = (tensor_sens(col_name, record.baseline_row)
              if record.baseline_row is not None else 0.0)
    s_ref = tensor_sens(col_name, record.ref_row)
    n_i = normed[row]
    return ((s_i - s_base) - record.sign * n_i * (s_ref - s_base)) / record.scale


def _zscore_sensitivity(record, col_name, row, tensor_sens, normed, s_i):
    """``∂/∂θ`` of a z-score-normalized column (subtract mean μ, divide by std σ) -- the one
    method that couples **every** row through σ (ADR-0053). With ``s_bar`` the per-row mean of
    the sensitivities of the values it consumed (the #447 tensor's, or -- mid-chain -- the
    previous stage's) and σ = ``record.scale``::

        ∂n_i/∂θ = (s_i - s_bar)/σ - n_i·(∂σ/∂θ)/σ,
        ∂σ/∂θ  = Σ_k n_k (s_k - s_bar)/(K - ddof)

    (``n_k = (raw_k - μ)/σ`` is the recorded normalized value, so ``(raw_k - μ) = σ·n_k``
    cancels the σ in ``∂σ/∂θ``). A σ of 0 means ``Data`` left the column un-divided
    (``n_i = raw_i - μ``), so ``∂n_i/∂θ = s_i - s_bar``."""
    nrows = len(normed)
    all_s = np.array([tensor_sens(col_name, k) for k in range(nrows)])   # (K, n_param)
    s_bar = all_s.mean(axis=0)
    if record.scale == 0.0:
        return s_i - s_bar
    dsigma = (normed[:, np.newaxis] * (all_s - s_bar)).sum(axis=0) / (nrows - record.ddof)
    return (s_i - s_bar) / record.scale - normed[row] * dsigma / record.scale


def _selector_for(sens, col_name):
    """The typed sensitivity selector for an objective column name.

    A scored (or measurement-formula-referenced) column is a BNGL model observable
    (``observable:<name>``), an ``expression:<name>`` global function (with
    ``print_functions``), or -- on the SBML/Antimony backend (layer H #455) -- a
    ``species:<name>``; the sensitivity tensor labels its columns the same way (#447/#455).
    Raises :class:`GradientNotSupported` if none was computed -- the gradient path needs a
    sensitivity column for every scored observable (or column a measurement formula reads)."""
    for prefix in ('observable:', 'expression:', 'species:'):
        selector = prefix + col_name
        if selector in sens.selectors:
            return selector
    raise GradientNotSupported(
        "No forward-sensitivity column for scored observable '%s' (have: %s)."
        % (col_name, ', '.join(sens.selectors) or '(none)'))


def _sampling_scale_factors(free_params):
    """The ``d theta/d u`` Jacobian-diagonal for the native -> sampling transform.

    Identity (1.0) for every LINEAR parameter -- short-circuited, so the all-linear
    case never touches a scale. For a log-scaled parameter, the scale's own analytic
    derivative at the current ``u = forward(theta)`` (``priors/scale.py``), which keeps
    the log10/ln bases in one place and bit-consistent with the sampler."""
    factors = np.ones(len(free_params))
    for j, param in enumerate(free_params):
        if not param.log_space:
            continue
        u = param.to_sampling_space(param.value)
        factors[j] = _d_theta_d_u(param, u)
    return factors


def _d_theta_d_u(free_param, u):
    """``d theta/d u`` for one log-scaled parameter -- the scale's analytic derivative
    (``d_inverse``, ADR-0087), falling back to autodiff of ``inverse_jax``.

    Every built-in scale's inverse is an exponential, so its derivative is closed form
    (``ln(10)*10**u`` / ``exp(u)``) and this stays off jax. It used to ``jax.grad`` that
    scalar power, which JIT-compiled an XLA kernel *per rebuild of the Jacobian* -- ~40 ms
    the persistent cache then declined as too cheap to keep -- and made the optional
    ``pybnf[jax]`` extra a hard requirement of most gradient fits, since ``loguniform_var``
    is the usual way to declare a rate constant (#524). The autodiff route remains for a
    custom :class:`~pybnf.priors.scale.Scale` that supplies only an ``inverse_jax``."""
    try:
        return float(free_param.d_from_sampling_space(float(u)))
    except NotImplementedError:
        jax = _require_jax()
        return float(jax.grad(free_param.from_sampling_space_jax)(float(u)))


def _require_jax():
    """Import ``jax`` lazily for the sampling-space transform, or raise a pointed error.

    Reached only by a **custom** scale that defines an ``inverse_jax`` but no analytic
    ``d_inverse`` (ADR-0087), whose native -> sampling Jacobian factor is autodiffed
    instead (ADR-0029/0059); ``jax`` is the optional ``pybnf[jax]`` extra (ADR-0019), so
    a missing install surfaces as a :class:`PybnfError` naming the extra -- the house
    pattern (mirroring ``samplers/hmc._require_jax``) -- never a bare ``ImportError``. No
    fit using only the built-in linear/log10/ln scales reaches here."""
    try:
        import jax
    except ImportError as e:
        raise PybnfError(
            "Gradient assembly needs jax to autodiff a custom parameter scale that "
            "defines no analytic derivative (`d_inverse`), which is the optional 'jax' "
            "extra. Install it with `pip install pybnf[jax]` (or `uv pip install "
            "pybnf[jax]`), or give the scale a `d_inverse`. A fit using only the "
            "built-in linear / log10 / ln scales needs no extra."
        ) from e
    return jax


# ============================== constraint penalty gradient (layer I, #456) ===

def assemble_constraint_gradient(constraint_sets, sim_data_dict, routings, free_params):
    """The scalar gradient of the total constraint penalty w.r.t. the free parameters (layer I,
    #456/#385), in **sampling space** -- the term #386 adds to the objective gradient for a fit
    with active constraints.

    ``constraint_sets`` is an iterable of :class:`~pybnf.constraint.ConstraintSet`; each
    constraint's penalty is a piecewise (static) or Gaussian-CDF (likelihood) function of an at-/
    between-time readout, so its gradient is that readout's forward sensitivity times the local
    penalty slope (:meth:`~pybnf.constraint.Constraint.penalty_gradient`). ``sim_data_dict`` is the
    ``{model: {suffix: Data}}`` the penalties are scored on, each ``Data`` carrying the #447
    sensitivity tensor; ``routings`` maps ``(model, suffix) -> ExperimentRouting`` (#448) so a
    readout's sensitivity is factor-folded into the free-parameter axes exactly as the objective's
    is. ``free_params`` fixes the parameter (column) order and supplies the native->sampling
    transform applied once at the end.

    A penalty is not a sum of squares, so -- like an estimated noise normalizer (layer D) -- it
    lives on the scalar gradient only: a fit with active constraints is not ``least_squares_exact``,
    and #386 must consume the scalar gradient. Returns a ``(n_param,)`` vector (zeros for an empty
    constraint set)."""
    names = [p.name for p in free_params]
    index = {name: j for j, name in enumerate(names)}
    n_param = len(free_params)
    # Live parameter values, so a constraint with an estimated scale reads its
    # current value and contributes its d(penalty)/d(scale) column -- the constraint counterpart of
    # assemble_gaussian_gradient seeding the objective's _pset_values from the free-parameter point.
    pset_values = {p.name: p.value for p in free_params}
    raw_sens = _constraint_sensitivity_accessor(sim_data_dict, routings, index, n_param)
    grad = np.zeros(n_param)
    for cset in constraint_sets:
        for constraint in cset.constraints:
            grad += constraint.penalty_gradient(sim_data_dict, raw_sens, index, n_param,
                                                pset_values=pset_values)
    return grad * _sampling_scale_factors(free_params)


def assemble_constraint_hessian(constraint_sets, sim_data_dict, routings, free_params):
    """The Gauss-Newton **Hessian** of the total constraint penalty w.r.t. the free parameters
    (layer I curvature, #481/#456), in **sampling space** -- the constraint block the EFIM
    trust-region path (``job_type = gntr``) adds to :func:`assemble_fisher_hessian`'s data-fit
    Hessian, the curvature sibling of :func:`assemble_constraint_gradient`.

    Each constraint's penalty is ``P(q(theta))`` for an at-/between-time readout ``q``, so its
    exact Hessian is ``P''(q) * outer(grad q, grad q) + P'(q) * hess q``; the Gauss-Newton term
    drops the second-order sensitivity ``hess q`` (as the EFIM drops it for the data fit) and
    clamps ``P''`` to its positive part, giving the PSD ``max(P''(q), 0) * outer(grad q, grad q)``
    (:meth:`~pybnf.constraint.Constraint.penalty_curvature`). Reuses
    :func:`_constraint_sensitivity_accessor` for ``grad q`` exactly as the gradient does, and
    applies the same native -> sampling ``d theta/d u`` factor on both axes. A **piecewise-linear**
    (static hinge ``.con``) penalty has ``P'' == 0``, so it contributes no curvature -- correct, a
    linear penalty has none; its pull rides the gradient. Returns zeros for an empty constraint set;
    raises :class:`GradientNotSupported` for a penalty whose curvature this cut does not assemble
    (a clipped smooth penalty, an estimated constraint scale) -> the L-BFGS-B fallback."""
    names = [p.name for p in free_params]
    index = {name: j for j, name in enumerate(names)}
    n_param = len(free_params)
    pset_values = {p.name: p.value for p in free_params}
    raw_sens = _constraint_sensitivity_accessor(sim_data_dict, routings, index, n_param)
    hessian = np.zeros((n_param, n_param))
    for cset in constraint_sets:
        for constraint in cset.constraints:
            hessian += constraint.penalty_curvature(sim_data_dict, raw_sens, index, n_param,
                                                    pset_values=pset_values)
    factors = _sampling_scale_factors(free_params)
    return hessian * np.outer(factors, factors)


def _constraint_sensitivity_accessor(sim_data_dict, routings, index, n_param):
    """Build ``raw_sens(model, suffix, observable, row) -> (n_param,)``: the native-space forward
    sensitivity of one constraint readout, routing-factor-folded (#448), with a pinned (factor 0)
    or model-unbound (``NONE``) parameter left at 0 -- the constraint counterpart of the
    objective's ``raw_sens``, keyed by the full ``(model, suffix, observable)`` since a constraint
    may read any simulation's output. The independent variable does not move with theta
    (sensitivity 0); a constant operand is handled by the constraint (it never calls this).

    Every supplied routing is checked for the one-contribution-per-native-column invariant the
    summation below relies on, exactly as the objective's accessor checks its own (#537)."""
    for routing in routings.values():
        routing.check_column_multiplicity()

    def raw_sens(model, suffix, observable, row):
        sim_data = sim_data_dict[model][suffix]
        sens = sim_data.output_sensitivities
        if sens is None:
            raise GradientNotSupported(
                "Constraint reads observable '%s' from model '%s' suffix '%s', whose simulation "
                "carries no forward-sensitivity tensor; enable the gradient path (apply_routing) "
                "on every model a constraint reads before assembling the constraint gradient."
                % (observable, model, suffix))
        if observable == sim_data.indvar:
            return np.zeros(n_param)
        if (model, suffix) not in routings:
            raise GradientNotSupported(
                "Constraint reads observable '%s' from model '%s' suffix '%s', but no routing "
                "was supplied for it; provide an ExperimentRouting for every (model, suffix) a "
                "constraint reads (routings[(model, suffix)])." % (observable, model, suffix))
        routing = routings[(model, suffix)]
        selector = _selector_for(sens, observable)
        vec = np.zeros(n_param)
        for name, route in routing.routes.items():
            total = 0.0
            for c in route.contributions:
                if c.target == NONE or c.factor == 0.0:
                    continue
                total += c.factor * _sensitivity(sens, selector, c, row, name)
            vec[index[name]] = total
        return vec
    return raw_sens
