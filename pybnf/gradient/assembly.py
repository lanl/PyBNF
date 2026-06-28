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

    With a **fixed** sigma the data fit IS the whole objective, so the residual and
    scalar forms agree by construction (``gradient == jacobian.T @ residual``,
    ``0.5||rho||**2 == evaluate``) and ``least_squares_exact`` is ``True``. With an
    **estimated** sigma (layer D, #451) the retained ``+log sigma`` normalizer is not a
    square: it is folded into the scalar ``gradient`` only (``gradient == jacobian.T @
    residual + the noise columns``), the residual-Jacobian stays a model of the data fit
    alone (so ``0.5||rho||**2`` omits the normalizer and the sigma columns of
    ``jacobian`` are zero), and ``least_squares_exact`` is ``False`` -- the signal that
    a trust-region least-squares step must instead consume the scalar ``gradient``.
    """
    residual: np.ndarray      # (n_obs,)
    jacobian: np.ndarray      # (n_obs, n_param), sampling space
    gradient: np.ndarray      # (n_param,) = J^T rho + estimated-noise columns
    param_names: list         # free-parameter order of the columns / gradient
    least_squares_exact: bool = True   # False once an estimated sigma is present


def assemble_gaussian_gradient(objective, experiments, free_params):
    """Assemble the scalar gradient and residual-Jacobian, summed across experiments.

    ``objective`` is the fit's :class:`~pybnf.objective.LikelihoodObjective`; it
    supplies each point's residual through ``residual_point`` and any estimated-noise
    gradient column through ``noise_grad_point`` (which gate the Gaussian/LINEAR/MEDIAN
    cut-1 case -- fixed or single-free-parameter sigma -- raising
    :class:`GradientNotSupported` otherwise). ``experiments`` is an iterable of
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
    least_squares_exact = True
    for sim_data, exp_data, routing in experiments:
        if _accumulate_experiment(objective, sim_data, exp_data, routing,
                                  index, n_param, rho_rows, jac_rows, noise_gradient):
            least_squares_exact = False

    rho = np.asarray(rho_rows, dtype=float)
    jac = np.asarray(jac_rows, dtype=float).reshape(len(rho_rows), n_param)

    # Native -> sampling space, applied exactly once (ADR-0029): rho is invariant, each
    # Jacobian column scales by d theta_j/d u_j at the current value, and the scalar
    # noise gradient (a free sigma's column) takes the same per-parameter chain factor.
    factors = _sampling_scale_factors(free_params)
    jac = jac * factors[np.newaxis, :]
    noise_gradient = noise_gradient * factors

    gradient = jac.T @ rho + noise_gradient
    return GradientResult(residual=rho, jacobian=jac, gradient=gradient,
                          param_names=names, least_squares_exact=least_squares_exact)


def _accumulate_experiment(objective, sim_data, exp_data, routing,
                           index, n_param, rho_rows, jac_rows, noise_gradient):
    """Append one experiment's per-point residual and native-space Jacobian rows, and
    accumulate any estimated-noise (sigma) gradient columns into ``noise_gradient``.

    Mirrors ``SummationObjective.evaluate``'s point loop exactly -- same independent
    variable, same comparable-column intersection, same NaN skip, same
    ``_sim_row_for`` row match -- so the gradient is assembled over precisely the
    points PyBNF scores. Columns are walked in sorted order for a deterministic
    observation axis (matching ``evaluate_pointwise``). Returns ``True`` iff this
    experiment contributed an estimated-noise column (so the caller can clear the
    ``least_squares_exact`` flag)."""
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
    # Data-level normalization chain rule (ADR-0053) folded in. The objective's
    # prediction_sensitivity seam threads each trajectory transform (cumulative / per-
    # measurement) on top of it; for a plain column it returns exactly the per-parameter
    # Jacobian this loop built inline before (byte-identical), now vectorised.
    raw_sens = _raw_sensitivity_accessor(sim_data, sens, routing, index, n_param, indvar)

    had_estimated_noise = False
    for rownum in range(exp_data.data.shape[0]):
        sim_row = objective._sim_row_for(sim_data, exp_data, indvar, rownum, show_warnings=False)
        for col_name in sorted(compare_cols):
            observation = exp_data.data[rownum, exp_data.cols[col_name]]
            if np.isnan(observation):
                continue
            rho, drho_dpred = objective.residual_point(sim_data, exp_data, sim_row, rownum, col_name)
            weight = exp_data.weights[rownum, exp_data.cols[col_name]]
            sqrt_w = np.sqrt(weight)
            # ∂pred/∂θ through the objective's transform seam (plain / cumulative / per-
            # measurement; #453), so the Jacobian row differentiates exactly what is scored.
            # A pinned (factor 0) parameter and a model-unbound nuisance (a free sigma; layer
            # D) carry 0 in raw_sens, so this row stays the data fit's dependence alone -- an
            # estimated sigma's own column is handled below on the scalar path.
            dpred_dtheta = objective.prediction_sensitivity(
                sim_data, sim_row, col_name, exp_data, rownum, raw_sens, index)
            jac_row = sqrt_w * drho_dpred * dpred_dtheta
            # Layer D (#451): an estimated free noise scale contributes d loss/d sigma
            # straight to the scalar gradient (the +log sigma normalizer is not a square,
            # so it stays off the residual-Jacobian). Weighted by the full per-point
            # weight, exactly as ``evaluate`` weights the per-point loss.
            for pname, dloss_dparam in objective.noise_grad_point(
                    sim_data, exp_data, sim_row, rownum, col_name).items():
                if pname not in index:
                    raise GradientNotSupported(
                        "Observable '%s' estimates its noise scale as free parameter "
                        "'%s', but '%s' is not among the gradient's free parameters "
                        "(%s). An estimated noise scale must be a declared free "
                        "parameter." % (col_name, pname, pname, ', '.join(index) or '(none)'))
                noise_gradient[index[pname]] += weight * dloss_dparam
                had_estimated_noise = True
            rho_rows.append(sqrt_w * rho)
            jac_rows.append(jac_row)
    return had_estimated_noise


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


def _raw_sensitivity_accessor(sim_data, sens, routing, index, n_param, indvar):
    """Build ``raw_sens(col_name, row) -> (n_param,)``: native-space ``∂(that column as
    ``_prediction`` reads it)/∂θ`` (#453), the seam the objective's
    :meth:`~pybnf.objective.SummationObjective.prediction_sensitivity` composes transforms on.

    The base is the #447 forward tensor read at one row, each routed parameter's column scaled
    by its condition factor, with a pinned (factor 0) or model-unbound (``NONE``, e.g. a free
    σ) parameter left at 0 -- exactly the per-parameter Jacobian the assembly built inline
    before, now vectorised. The independent variable is θ-independent (sensitivity 0). When the
    column was normalized (ADR-0053), the normalizer's own θ-dependence is threaded here so the
    caller sees ``∂(normalized column)/∂θ`` and every downstream transform composes correctly
    (scoring applies normalize -> ``_prediction``)."""
    norm = sim_data.normalization or {}

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

    A scored column is a model observable (``observable:<name>``) or, with
    ``print_functions``, an expression (``expression:<name>``); the sensitivity
    tensor labels its columns the same way (#447). Raises
    :class:`GradientNotSupported` if neither was computed -- the gradient path needs a
    sensitivity column for every scored observable."""
    for prefix in ('observable:', 'expression:'):
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
