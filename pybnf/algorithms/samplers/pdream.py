"""PDreamAlgorithm — Preconditioned DREAM (the ``p_dream`` fit type).

As of ADR-0067 (Stage 1) the preconditioned proposal is no longer a separate
algorithm: it is the ``'whitened'`` proposal operator on :class:`DreamAlgorithm`
itself. ``p_dream`` is therefore just DREAM(ZS) with the proposal pinned to
``'whitened'`` — expressed here as a one-line config default plus a thin
subclass kept so the public ``job_type = p_dream`` code, the
``PDreamAlgorithm`` class name (re-exported by ``pybnf.algorithms``), and the
existing oracle tests all continue to resolve unchanged.
"""


from .dream import DreamAlgorithm, DreamConfig
from ...registry import register_fit_type

from typing import Any


class PDreamConfig(DreamConfig):
    """Config for preconditioned DREAM (p_dream), co-located with the method
    (ADR-0006). Pins the DREAM ``proposal`` operator to ``'whitened'`` and keeps
    the one preconditioning key ``precondition_adapt`` that p_dream owns
    (see [[test_config_unused_keys]]); everything else is inherited from
    :class:`DreamConfig`."""

    proposal: str = 'whitened'
    precondition_adapt: Any = None


@register_fit_type('p_dream', family='sampler', display_name='Preconditioned DREAM',
                   schema=PDreamConfig)
class PDreamAlgorithm(DreamAlgorithm):
    """
    P-DREAM: Preconditioned DREAM — DREAM(ZS) with the ``'whitened'`` proposal.

    All behaviour lives in :class:`DreamAlgorithm`, selected by ``proposal =
    'whitened'`` (pinned by :class:`PDreamConfig`): DE proposals are computed in a
    covariance-whitened space ``z = L_inv @ x`` learned online from the chain
    history (Adaptive-Metropolis style), so correlated posteriors are sampled in
    decorrelated coordinates. The proposal stays symmetric (DE differences from an
    external archive), so standard Metropolis-Hastings acceptance is valid.

    This subclass adds no code of its own; it exists only to keep ``p_dream`` a
    distinct, named public ``job_type`` over the unified engine (ADR-0067).
    """
