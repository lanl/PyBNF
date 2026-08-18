"""Marginal-time objective gradient assembly (ADR-0113, issue #588 -- ``time_error`` phase 2).

Phase 1 (ADR-0112) marginalizes the latent measurement time out of the likelihood by quadrature
over the *stored* trajectory, reusing every noise family's density -- and is gradient-free, so a
gradient ``job_type`` is refused at build. Phase 2 lifts that refusal without augmenting the ODE:
PyBNF's forward-sensitivity engine (#447) already delivers ``∂y(τ)/∂θ`` at every stored grid node,
so ``dz_k/dθ`` is a Python quadrature over the same nodes phase 1 integrates. The paper (Vanhoefer
et al., bioRxiv 2026.05.09.724053) augments the ODE with a state ``z_k`` only because its solver
(AMICI/CVODES) returns sensitivities of ODE *states* alone; PyBNF reaches the same sensitivity
directly (ADR-0113).

This module is the marginal-time sibling of :func:`pybnf.gradient.assembly.assemble_gaussian_gradient`.
It differs in exactly one way: a marginal-time datum is not scored against one matched trajectory
row but integrated over the whole window, so the per-experiment contribution comes from the
objective's :meth:`~pybnf.measurement.time_error.MarginalizedTimeObjective.marginal_gradient` (which
walks the trajectory) rather than the assembly's matched-row point loop. Everything else is shared:
the same ``raw_sens`` forward-sensitivity accessor (:func:`~pybnf.gradient.assembly._raw_sensitivity_accessor`,
routing-factor-folded, normalization / measurement-model chain rules threaded in) and the same
native -> sampling ``dθ/du`` transform applied once at the end (ADR-0029).

The marginal-time contribution ``−log z_k`` is the log of an integral, never a sum of squares, so it
carries no least-squares residual: the assembled :class:`~pybnf.gradient.assembly.GradientResult` is
always ``least_squares_exact = False`` (its residual / Jacobian are empty), which routes ``job_type =
trf`` -- which needs an exact residual -- to its refusal, while ``lbfgs`` consumes the scalar gradient
and ``gntr`` the per-datum-score Fisher this assembler also produces (``include_fisher``).
"""

import numpy as np

from .assembly import GradientResult, _raw_sensitivity_accessor, _sampling_scale_factors
from .errors import GradientNotSupported


def assemble_marginal_time_gradient(objective, experiments, free_params, include_fisher=False):
    """Assemble the scalar gradient (and, for ``gntr``, the Gauss-Newton Fisher) of a
    :class:`~pybnf.measurement.time_error.MarginalizedTimeObjective`, summed across experiments.

    ``objective`` is the fit's marginal-time objective; each experiment's contribution to
    ``∇_θ (−Σ_k log z_k)`` comes from its :meth:`~pybnf.measurement.time_error.MarginalizedTimeObjective.marginal_gradient`
    (the trajectory-quadrature the phase-1 ``evaluate`` differentiates exactly). ``experiments`` is
    the same ``(sim_data, exp_data, routing[, data_key])`` iterable
    :func:`~pybnf.gradient.assembly.assemble_gaussian_gradient` consumes -- one per scored
    model/condition, each ``sim_data`` carrying the #447 ``output_sensitivities`` tensor -- and
    ``free_params`` the ordered free-parameter list fixing the column order and each parameter's
    ``dθ/du`` scale factor.

    Returns a :class:`~pybnf.gradient.assembly.GradientResult` with an empty residual / Jacobian and
    ``least_squares_exact = False``; ``hessian`` is ``None`` unless ``include_fisher`` (then the
    per-datum-score outer product ``Σ_k w_k g_k g_kᵀ``, PSD by construction -- the empirical Fisher
    ``gntr`` steps from). The scalar ``gradient`` and Fisher are in sampling space ``u``.
    """
    names = [p.name for p in free_params]
    index = {name: j for j, name in enumerate(names)}
    n_param = len(free_params)

    # An estimated σ / σ_t reads its value from the objective's per-evaluation pset map (ADR-0021);
    # seed it from the current free-parameter point, exactly as assemble_gaussian_gradient does, and
    # merge over any existing map so a prior evaluate's fixed parameters survive.
    existing = getattr(objective, '_pset_values', None) or {}
    objective._pset_values = {**existing, **{p.name: p.value for p in free_params}}

    grad = np.zeros(n_param)
    hessian = np.zeros((n_param, n_param)) if include_fisher else None
    for sim_data, exp_data, routing, *_rest in experiments:
        sens = sim_data.output_sensitivities
        if sens is None:
            raise GradientNotSupported(
                "A time_error experiment carries no forward-sensitivity tensor; enable the gradient "
                "path (apply_routing) on every scored model before assembling the marginal-time "
                "gradient.")
        indvar = min(exp_data.cols, key=exp_data.cols.get)
        raw_sens = _raw_sensitivity_accessor(objective, sim_data, sens, routing, index, n_param, indvar)
        contribution = objective.marginal_gradient(
            sim_data, exp_data, raw_sens, index, n_param, include_fisher=include_fisher)
        if contribution is None:
            # An unscoreable experiment (a marginal contribution z_k underflowed). Its score is the
            # inf penalty, so gradient_at is not reached for it in a normal run (#492); defensive.
            raise GradientNotSupported(
                "A time_error datum is unscoreable at this point (a marginal contribution z_k "
                "underflowed to zero), so no finite gradient exists here. Score this point on the "
                "gradient-free path.")
        g_exp, h_exp = contribution
        grad += g_exp
        if include_fisher:
            hessian += h_exp

    # Native -> sampling space, applied once (ADR-0029): the scalar gradient by the per-parameter
    # dθ/du factor, the Fisher on both axes (diag(f) H diag(f)).
    factors = _sampling_scale_factors(free_params)
    grad = grad * factors
    if hessian is not None:
        hessian = hessian * np.outer(factors, factors)

    return GradientResult(residual=np.zeros(0), jacobian=np.zeros((0, n_param)), gradient=grad,
                          param_names=names, least_squares_exact=False, hessian=hessian)
