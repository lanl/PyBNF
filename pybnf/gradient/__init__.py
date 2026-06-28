"""Gradient plumbing: free-parameter -> sensitivity routing and (later) Jacobian assembly.

The #385 gradient-plumbing epic surfaces objective gradients and residual Jacobians from
bngsim's forward output-sensitivity tensor. This package hosts the PyBNF-side machinery:

* :mod:`pybnf.gradient.routing` (#448) -- the *pure mapping* from edition-2 free parameters to
  each experiment's ``sensitivity_params`` / ``sensitivity_ic`` request and per-parameter
  chain-rule factor (no objective math).

The objective gradient / residual-Jacobian assembly (#449) lands here too.
"""

from .routing import (
    PARAM,
    IC,
    NONE,
    ParamRoute,
    ExperimentRouting,
    classify_free_param,
    condition_factor,
    route_experiment,
    route_for_model,
    apply_routing,
)

__all__ = [
    'PARAM',
    'IC',
    'NONE',
    'ParamRoute',
    'ExperimentRouting',
    'classify_free_param',
    'condition_factor',
    'route_experiment',
    'route_for_model',
    'apply_routing',
]
