"""Gradient plumbing: free-parameter -> sensitivity routing and objective/constraint gradient assembly.

The #385 gradient-plumbing epic surfaces objective gradients and residual Jacobians from
bngsim's forward output-sensitivity tensor. This package hosts the PyBNF-side machinery:

* :mod:`pybnf.gradient.routing` (#448) -- the *pure mapping* from edition-2 free parameters to
  each experiment's ``sensitivity_params`` / ``sensitivity_ic`` request and per-parameter
  chain-rule factor (no objective math); and
* :mod:`pybnf.gradient.assembly` (#449 + the layer follow-ups) -- the objective gradient /
  residual-Jacobian assembly (``assemble_gaussian_gradient``) that combines #447's sensitivity
  tensor with #448's routing, summed across experiments, in sampling space (the form #386's
  optimizer consumes). Beyond the cut-1 fixed-sigma Gaussian (#449) it covers an estimated noise
  scale (#451), a log / lognormal scale (#452), per-observable trajectory transforms and
  normalization (#453), the asymmetric Laplace / Student-t families and mean centering (#454),
  and -- via the sibling ``assemble_constraint_gradient`` -- qualitative / inequality constraint
  penalties (#456). A configuration outside the supported set raises :class:`GradientNotSupported`,
  so a caller can fall back to a gradient-free step. For the EFIM trust-region optimizer
  (``job_type = gntr``, #481) it additionally assembles the expected-Fisher / Gauss-Newton
  **Hessian** in the same point walk as the scalar gradient
  (``assemble_gradient_and_fisher_hessian`` + the constraint sibling
  ``assemble_constraint_hessian``); ``assemble_fisher_hessian`` remains the standalone API.

The capability gate and the per-layer math are documented in ``docs/gradient_fitting.rst``.
"""

from .errors import GradientNotSupported
from .routing import (
    PARAM,
    IC,
    NONE,
    RouteContribution,
    ParamRoute,
    ExperimentRouting,
    classify_free_param,
    classify_condition_target,
    condition_factor,
    route_experiment,
    route_for_model,
    apply_routing,
    apply_routings,
)
from .assembly import (
    GradientResult,
    assemble_constraint_gradient,
    assemble_constraint_hessian,
    assemble_fisher_hessian,
    assemble_gradient_and_fisher_hessian,
    assemble_gaussian_gradient,
)

__all__ = [
    'GradientNotSupported',
    'PARAM',
    'IC',
    'NONE',
    'RouteContribution',
    'ParamRoute',
    'ExperimentRouting',
    'classify_free_param',
    'classify_condition_target',
    'condition_factor',
    'route_experiment',
    'route_for_model',
    'apply_routing',
    'apply_routings',
    'GradientResult',
    'assemble_gaussian_gradient',
    'assemble_gradient_and_fisher_hessian',
    'assemble_constraint_gradient',
    'assemble_fisher_hessian',
    'assemble_constraint_hessian',
]
