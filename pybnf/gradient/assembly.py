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
is one autodiff of each parameter's scale ``inverse_jax`` (``priors/scale.py``) -- no
hand-written per-scale derivative. A LINEAR parameter has ``d theta/d u = 1`` and is
short-circuited, so the common (all-linear) path needs no jax; only a log-scaled
parameter pulls in the optional ``pybnf[jax]`` extra (the house pattern, ADR-0019).

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

Trajectory transforms + normalization (layer F, #453)
-----------------------------------------------------
``_prediction`` may form the scored value from the raw observable through a per-observable
transform: a **cumulative -> incident** difference (ADR-0051), a **per-measurement** scale/
offset formula (ADR-0045), or an upstream ``Data``-level **normalization** (ADR-0053). Each
makes ``∂pred/∂θ`` differ from the raw observable sensitivity, so the assembly reads a
``raw_sens(col, row)`` accessor (the #447 tensor, routing-factor-folded, with normalization's
own quotient/chain rule threaded in -- ``_normalized_sensitivity``) and hands it to the
objective's :meth:`~pybnf.objective.SummationObjective.prediction_sensitivity` seam, which
mirrors ``_prediction`` branch for branch (cumulative differences sensitivity rows; a per-
measurement formula chains its symbolic gradient through each referenced column's sensitivity
plus any estimated placeholder it names -- which, unlike a free σ, *does* enter ``∂pred/∂θ`` and
so lands in the residual-Jacobian). A plain column collapses to the raw sensitivity, so the
no-transform path is byte-identical.

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
from .routing import PARAM, NONE
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
    #: attached only on the EFIM trust-region path (``fit_type = gntr``, #481) by
    #: :func:`assemble_fisher_hessian`; ``None`` for ``trf`` / ``lbfgs``, which never form it.
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
    ``(sim_data, exp_data, routing)`` triples -- one per scored model/condition; each
    ``sim_data`` must carry the #447 ``output_sensitivities`` payload (the gradient
    path active), and ``routing`` is that experiment's
    :class:`~pybnf.gradient.routing.ExperimentRouting`. ``free_params`` is the ordered
    list of :class:`~pybnf.pset.FreeParameter` defining the ``u``-vector: it fixes the
    column order of the Jacobian and the entries of the scalar gradient, and supplies
    each parameter's scale (current value -> ``d theta/d u``).

    Returns a :class:`GradientResult`. The per-experiment routing is built once by the
    caller (#386) -- it depends only on model structure, conditions, and free-parameter
    ids, never on the parameter values -- so this per-evaluation assembly only reads
    the freshly simulated sensitivity tensors.
    """
    names = [p.name for p in free_params]
    index = {name: j for j, name in enumerate(names)}
    n_param = len(free_params)

    # An estimated free noise scale (a free sigma) reads its value from the objective's
    # per-evaluation pset map (ADR-0021); seed it from the current free-parameter point
    # so the loss is scored at u. Merged over any existing map so a prior evaluate's
    # fixed parameters survive (a fixed-sigma fit never reads it -- harmless there).
    existing = getattr(objective, '_pset_values', None) or {}
    objective._pset_values = {**existing, **{p.name: p.value for p in free_params}}

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
    least_squares_exact = True
    for sim_data, exp_data, routing in experiments:
        if _accumulate_experiment(objective, sim_data, exp_data, routing, index, n_param,
                                  rho_rows, jac_rows, noise_gradient, data_fit_gradient):
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

    gradient = jac.T @ rho + data_fit_gradient + noise_gradient
    return GradientResult(residual=rho, jacobian=jac, gradient=gradient,
                          param_names=names, least_squares_exact=least_squares_exact)


def _accumulate_experiment(objective, sim_data, exp_data, routing, index, n_param,
                           rho_rows, jac_rows, noise_gradient, data_fit_gradient):
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
    sens = sim_data.output_sensitivities
    if sens is None:
        raise GradientNotSupported(
            "An experiment carries no forward-sensitivity tensor; enable the gradient "
            "path (apply_routing) on every scored model before assembling the gradient.")

    indvar = min(exp_data.cols, key=exp_data.cols.get)
    comparable = set(sim_data.cols) | set(objective._per_measurement_models)
    compare_cols = set(exp_data.cols).intersection(comparable)
    compare_cols.discard(indvar)

    # The per-column sensitivity accessor (#453): ∂(column as _prediction sees it)/∂θ -- the
    # #447 tensor read at a row, routing-factor-folded, NONE/pinned parameters at 0, with any
    # Data-level normalization chain rule (ADR-0053) folded in, and any materialized
    # measurement-model column (ADR-0036, layer H #455) differentiated through its formula's
    # chain rule. The objective's prediction_sensitivity seam threads each trajectory transform
    # (cumulative / per-measurement) on top of it; for a plain column it returns exactly the
    # per-parameter Jacobian this loop built inline before (byte-identical), now vectorised.
    raw_sens = _raw_sensitivity_accessor(objective, sim_data, sens, routing, index, n_param, indvar)

    inexact = False
    for rownum in range(exp_data.data.shape[0]):
        sim_row = objective._sim_row_for(sim_data, exp_data, indvar, rownum, show_warnings=False)
        for col_name in sorted(compare_cols):
            observation = exp_data.data[rownum, exp_data.cols[col_name]]
            if np.isnan(observation):
                continue
            weight = exp_data.weights[rownum, exp_data.cols[col_name]]
            sqrt_w = np.sqrt(weight)
            # ∂pred/∂θ through the objective's transform seam (plain / cumulative / per-
            # measurement; #453), so the gradient differentiates exactly what is scored.
            # A pinned (factor 0) parameter and a model-unbound nuisance (a free sigma; layer
            # D) carry 0 in raw_sens, so this stays the data fit's dependence alone -- an
            # estimated noise parameter's own column is handled below on the scalar path.
            dpred_dtheta = objective.prediction_sensitivity(
                sim_data, sim_row, col_name, exp_data, rownum, raw_sens, index)
            # Layer D/G (#451/#454): each estimated free noise parameter contributes
            # d(loss)/d(param) straight to the scalar gradient (its normalizer is not a square,
            # so it stays off the residual-Jacobian). Weighted by the full per-point weight,
            # exactly as ``evaluate`` weights the per-point loss.
            for pname, dloss_dparam in objective.noise_grad_point(
                    sim_data, exp_data, sim_row, rownum, col_name).items():
                if pname not in index:
                    raise GradientNotSupported(
                        "Observable '%s' estimates its noise scale as free parameter "
                        "'%s', but '%s' is not among the gradient's free parameters "
                        "(%s). An estimated noise scale must be a declared free "
                        "parameter." % (col_name, pname, pname, ', '.join(index) or '(none)'))
                noise_gradient[index[pname]] += weight * dloss_dparam
                inexact = True
            if objective.has_least_squares_residual(col_name):
                # A family whose data fit is a smooth half-square -- a Gaussian (data_fit =
                # 1/2 rho**2) or a Student-t (its exact sqrt-loss residual, #459): it contributes
                # an exact residual row + native Jacobian row (sqrt(w)-folded), and its data-fit
                # gradient comes from J^T rho. The Gaussian path is byte-identical to pre-layer-G.
                rho, drho_dpred = objective.residual_point(
                    sim_data, exp_data, sim_row, rownum, col_name)
                rho_rows.append(sqrt_w * rho)
                jac_rows.append(sqrt_w * drho_dpred * dpred_dtheta)
            else:
                # A family with no clean least-squares residual (Laplace, whose L1 data fit is the
                # cusp sqrt|z|; the count family; layer G, #454/#459): its data-fit gradient goes
                # straight to the scalar accumulator (full per-point weight), and the result is not
                # least_squares_exact.
                dfit_dpred = objective.data_fit_grad_point(
                    sim_data, exp_data, sim_row, rownum, col_name)
                data_fit_gradient += weight * dfit_dpred * dpred_dtheta
                inexact = True
    return inexact


def assemble_fisher_hessian(objective, experiments, free_params):
    """Assemble the expected-Fisher / Gauss-Newton **Hessian** ``H`` (n_param x n_param),
    summed across experiments -- the curvature the EFIM trust-region path
    (``fit_type = gntr``, #481) consumes alongside the scalar gradient
    :func:`assemble_gaussian_gradient` already produces.

    ``H = sum_i w_i [ kappa_i * outer(s_i, s_i) ]  +  sum_estimated-noise w_i * I_scale *
    outer(e_p, e_p)`` where ``s_i = d(prediction_i)/d(theta)`` is the same forward
    sensitivity the gradient uses (through the objective's ``prediction_sensitivity``
    seam), ``kappa_i`` the per-point location Fisher (``objective.location_fisher_point``:
    ``(d rho/d pred)**2`` for a residual-bearing Gaussian/Student-t column, the family's
    ``location_fisher`` for Laplace/count), and ``I_scale`` each estimated noise parameter's
    Fisher (``objective.noise_fisher_point``). Every rank-1 term is PSD (``kappa >= 0``,
    ``I_scale >= 0``), so ``H`` is PSD by construction. The location and noise blocks are
    disjoint (a noise parameter never enters ``s_i``, which is 0 in its column), so the
    Gaussian estimated-sigma Hessian is block-diagonal, exactly the Fisher predicts.

    ``experiments`` is the same ``(sim_data, exp_data, routing)`` iterable
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

    hessian = np.zeros((n_param, n_param))
    for sim_data, exp_data, routing in experiments:
        _accumulate_experiment_fisher(objective, sim_data, exp_data, routing, index, n_param,
                                      hessian)

    # Native -> sampling space, applied once on both axes (ADR-0029): the same per-parameter
    # d theta/d u factor the gradient scales its columns by, here as an outer product.
    factors = _sampling_scale_factors(free_params)
    return hessian * np.outer(factors, factors)


def _accumulate_experiment_fisher(objective, sim_data, exp_data, routing, index, n_param,
                                  hessian):
    """Accumulate one experiment's per-point Fisher rank-1 terms into ``hessian`` (the
    curvature twin of :func:`_accumulate_experiment`). Same independent variable, same
    comparable-column intersection, same NaN skip, same ``_sim_row_for`` row match, same
    sorted-column walk -- so the Hessian is assembled over precisely the points the gradient
    is. The location block reads ``kappa_i`` (``location_fisher_point``) and the noise block
    each estimated parameter's ``I_scale`` (``noise_fisher_point``)."""
    sens = sim_data.output_sensitivities
    if sens is None:
        raise GradientNotSupported(
            "An experiment carries no forward-sensitivity tensor; enable the gradient "
            "path (apply_routing) on every scored model before assembling the Hessian.")

    indvar = min(exp_data.cols, key=exp_data.cols.get)
    comparable = set(sim_data.cols) | set(objective._per_measurement_models)
    compare_cols = set(exp_data.cols).intersection(comparable)
    compare_cols.discard(indvar)

    raw_sens = _raw_sensitivity_accessor(objective, sim_data, sens, routing, index, n_param, indvar)

    for rownum in range(exp_data.data.shape[0]):
        sim_row = objective._sim_row_for(sim_data, exp_data, indvar, rownum, show_warnings=False)
        for col_name in sorted(compare_cols):
            observation = exp_data.data[rownum, exp_data.cols[col_name]]
            if np.isnan(observation):
                continue
            weight = exp_data.weights[rownum, exp_data.cols[col_name]]
            # d(prediction)/d(theta) through the objective's transform seam -- the SAME
            # s_i the gradient's residual-Jacobian / data-fit column is built from.
            dpred_dtheta = objective.prediction_sensitivity(
                sim_data, sim_row, col_name, exp_data, rownum, raw_sens, index)
            # Location block: w_i * kappa_i * outer(s_i, s_i). A column whose location
            # curvature is 0 (a pinned/constant prediction) adds nothing.
            kappa = objective.location_fisher_point(sim_data, exp_data, sim_row, rownum, col_name)
            if kappa:
                hessian += (weight * kappa) * np.outer(dpred_dtheta, dpred_dtheta)
            # Noise block: each estimated noise parameter's Fisher on its own coordinate.
            for pname, i_scale in objective.noise_fisher_point(
                    sim_data, exp_data, sim_row, rownum, col_name).items():
                if pname not in index:
                    raise GradientNotSupported(
                        "Observable '%s' estimates its noise scale as free parameter '%s', "
                        "but '%s' is not among the gradient's free parameters (%s)."
                        % (col_name, pname, pname, ', '.join(index) or '(none)'))
                j = index[pname]
                hessian[j, j] += weight * i_scale


def _sensitivity(sens, selector, route, sim_row):
    """The native forward sensitivity ``d(observable)/d(routed entity)`` at one time row.

    Reads the parameter axis (``sensitivity_params``) for a PARAM route and the
    initial-condition axis (``sensitivity_ic``) for an IC route, addressing the
    entity by the route's ``key`` (parameter id, or species for an IC)."""
    if route.target == PARAM:
        axis, labels = 'parameter', sens.param_names
    else:  # IC
        axis, labels = 'ic', sens.ic_species
    if route.key not in labels:
        raise GradientNotSupported(
            "Free parameter '%s' routes to %s '%s', but the simulation's "
            "sensitivity tensor has no such column (axis labels: %s). Apply the same "
            "routing to the model before running it (apply_routing)."
            % (route.free_param, axis, route.key, ', '.join(map(str, labels)) or '(none)'))
    column = sens.slice_for(selector, axis=axis)   # (n_times, n_axis)
    return column[sim_row, labels.index(route.key)]


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
    tensor/normalized sensitivity, so the no-measurement path is byte-identical."""
    norm = sim_data.normalization or {}
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
            if route.factor == 0.0 or route.target == NONE:
                continue
            vec[index[name]] = route.factor * _sensitivity(sens, selector, route, row)
        return vec

    def raw_sens(col_name, row):
        model = measurement_models.get(col_name)
        if model is not None:
            return model.prediction_sensitivity(sim_data, row, pset_values, raw_sens, index)
        record = norm.get(col_name)
        if record is None:
            return tensor_sens(col_name, row)
        return _normalized_sensitivity(record, col_name, row, sim_data, tensor_sens)

    return raw_sens


def _normalized_sensitivity(record, col_name, row, sim_data, tensor_sens):
    """Thread a per-column normalizer's own derivative into ``∂(normalized col)/∂θ`` (ADR-0053).

    ``normalization`` rescales the predicted column by a θ-dependent ``N(θ)`` read off the
    moving trajectory, so ``∂(raw/N)/∂θ`` is a quotient/chain rule coupling the scored row with
    the row(s) ``N`` is read from. The raw per-row sensitivities ``s_k`` come from the
    (un-normalized) #447 tensor; the normalized values ``n_k`` are read back from the now-
    rescaled ``Data`` (so the raw values need not be retained). See
    :class:`~pybnf.data.NormalizationRecord` for each method's closed form."""
    normed = sim_data.data[:, sim_data.cols[col_name]]
    s_i = tensor_sens(col_name, row)
    if record.method == 'floor':
        # x' = x + rho*max(x): additive, so ∂x'_i/∂θ = s_i + rho*s_argmax -- a simple rule, but
        # the floor's gradient is deliberately DEFERRED (ADR-0066, #479); refuse rather than
        # silently return the wrong (peak/init) quotient below. The fit falls back to a
        # gradient-free step (ADR-0475).
        raise GradientNotSupported(
            "Floor normalization ('floor', #479) on column '%s' has a deferred gradient; use a "
            "gradient-free step." % col_name)
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
    the raw sensitivities and σ = ``record.scale``::

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

    Identity (1.0) for every LINEAR parameter -- the common case, computed without
    jax. For a log-scaled parameter, autodiff its scale's ``inverse_jax`` at the
    current ``u = forward(theta)`` (``priors/scale.py``), so no per-scale derivative
    is hand-written and the log10/ln bases stay bit-consistent with the sampler."""
    factors = np.ones(len(free_params))
    for j, param in enumerate(free_params):
        if not param.log_space:
            continue
        u = param.to_sampling_space(param.value)
        factors[j] = _d_theta_d_u(param, u)
    return factors


def _d_theta_d_u(free_param, u):
    """``d theta/d u`` for one log-scaled parameter via autodiff of ``inverse_jax``."""
    jax = _require_jax()
    return float(jax.grad(free_param.from_sampling_space_jax)(float(u)))


def _require_jax():
    """Import ``jax`` lazily for the sampling-space transform, or raise a pointed error.

    The native -> sampling Jacobian of a **log-scaled** parameter autodiffs the
    scale's ``inverse_jax`` (ADR-0029/0059); ``jax`` is the optional ``pybnf[jax]``
    extra (ADR-0019), so a missing install surfaces as a :class:`PybnfError` naming
    the extra -- the house pattern (mirroring ``samplers/hmc._require_jax``) -- never
    a bare ``ImportError``. An all-linear fit never reaches here."""
    try:
        import jax
    except ImportError as e:
        raise PybnfError(
            "Gradient assembly needs jax to transform a log-scaled parameter's "
            "gradient into sampling space, which is the optional 'jax' extra. Install "
            "it with `pip install pybnf[jax]` (or `uv pip install pybnf[jax]`). A fit "
            "with only linear-scale parameters needs no extra."
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
    trust-region path (``fit_type = gntr``) adds to :func:`assemble_fisher_hessian`'s data-fit
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
    (sensitivity 0); a constant operand is handled by the constraint (it never calls this)."""
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
            if route.factor == 0.0 or route.target == NONE:
                continue
            vec[index[name]] = route.factor * _sensitivity(sens, selector, route, row)
        return vec
    return raw_sens
