"""Analytic profiling of an observable's linear coefficients (``linear_profiling = 1``,
ADR-0132, #671): the classification and the solve.

A PEtab ``observableFormula`` such as ``Z_state*scale + offset`` declares two parameters
that enter the prediction **affinely**: for fixed dynamics the prediction is
``Phi(theta) . c + B(theta)`` with ``c = (scale, offset)``. Under a linear-scale Gaussian
likelihood the optimum over ``c`` is a weighted least-squares problem with a closed form,
so those coefficients need not be searched at all: at every evaluation the fit solves for
them from the data and scores the projection residual (variable projection, Golub and
Pereyra 1973). The search then carries only the dynamics, every draw a global method ranks
is coefficient-optimal, and a twenty-decade box on a quantity that has a closed form stops
being a search dimension.

This module holds the two pieces that are pure functions of their inputs:

* :func:`affine_roles` -- which of a formula's candidate parameters enter it affinely, how
  each enters (``scale`` / ``offset`` / ``affine``), and whether the formula is affine in all
  of them **jointly**. Separately-affine-but-not-jointly is the ``scale*(x + offset)`` case,
  which spans the right two columns but is quadratic in the declared pair; solving for the
  span and mapping back to the declared names is ill posed as ``scale -> 0``, so it is
  refused rather than approximated.
* :func:`solve_group` -- the weighted least-squares solve over one group of coefficients
  within their declared bounds. The unconstrained closed form does not know a parameter has
  a declared support, and on the ADR-0130 fixture it returned a negative ``scale`` for a
  ``loguniform`` parameter at a third of the sampled points, scoring better for it. So the
  solve is box-constrained: the unconstrained minimum-norm solution when it lies in the box
  (the pseudo-inverse fallback for a singular design, which happens whenever the simulated
  column is constant over the group), and a bounded-variable least-squares solve otherwise.
  Which bound held a coefficient is reported, never silent.

The design matrix itself is built by the objective, by evaluating the measurement model at
basis coefficient vectors rather than by parsing the formula a second time
(:meth:`~pybnf.objective.LikelihoodObjective._resolve_linear_coefficients`): with every
profiled coefficient at 0 the prediction is ``B``, and coefficient ``j`` at 1 gives
``Phi_j + B``. That is exact for an affine formula, formula-agnostic, and needs no new
compile machinery.
"""

import re
from collections import namedtuple

import numpy as np

#: A PEtab per-measurement placeholder symbol, whose token is bound per data row.
PLACEHOLDER = re.compile(r'(?:observable|noise)Parameter\d+_\w+')

#: One group of coefficients the fit solves for jointly: the coefficients that share an
#: observable, transitively. ``names`` is the sorted tuple of free-parameter names,
#: ``columns`` the frozenset of observable ids whose formulas read them, ``lower`` /
#: ``upper`` the declared bounds as arrays parallel to ``names``.
LinearGroup = namedtuple('LinearGroup', 'names columns lower upper')


def _parse(formula):
    from ..petab.formula import _parse as parse_petab, _require_petab_math
    return parse_petab(_require_petab_math(), formula, source='observableFormula')


def formula_symbol_names(formula):
    """The free-symbol names of a PEtab formula, sorted."""
    return sorted(str(s) for s in _parse(formula).free_symbols)


def affine_roles(formula, candidates):
    """How each candidate parameter enters ``formula``, and whether it is affine in all of
    them jointly.

    Returns ``(roles, jointly_affine)``. ``roles`` maps every candidate the formula reads to
    ``'scale'`` (the formula is ``A*p`` with nothing left when ``p = 0``), ``'offset'``
    (``p + B``), ``'affine'`` (``A*p + B`` otherwise) or ``'nonlinear'`` (a second
    derivative in ``p`` that is not identically zero). A candidate the formula does not read
    is absent. ``jointly_affine`` is whether every second partial over the affine
    candidates, cross terms included, vanishes identically -- the exact condition under which
    the formula is ``Phi . c + B`` in those coefficients.

    Symbols are resolved by **name** against the parsed expression's own symbols: PEtab's
    parser tags them with assumptions, so a bare ``sympy.Symbol(name)`` is a different object
    and every derivative against it would be identically zero.
    """
    import sympy as sp
    expr = _parse(formula)
    by_name = {str(s): s for s in expr.free_symbols}
    roles = {}
    for name in candidates:
        sym = by_name.get(name)
        if sym is None:
            continue
        if sp.simplify(sp.diff(expr, sym, sym)) != 0:
            roles[name] = 'nonlinear'
            continue
        slope = sp.simplify(sp.diff(expr, sym))
        rest = sp.simplify(expr.subs(sym, 0))
        if rest == 0:
            roles[name] = 'scale'
        elif slope == 1:
            roles[name] = 'offset'
        else:
            roles[name] = 'affine'
    affine = [by_name[n] for n, role in roles.items() if role != 'nonlinear']
    jointly = True
    for i, a in enumerate(affine):
        for b in affine[i + 1:]:
            if sp.simplify(sp.diff(expr, a, b)) != 0:
                jointly = False
    return roles, jointly


def solve_group(phi, target, weights, lower, upper, tol=1e-9):
    """The weighted least-squares coefficients of one group within their declared bounds.

    Minimizes ``sum_i weights_i * (target_i - phi_i . c)**2`` over ``c`` subject to
    ``lower <= c <= upper``. ``phi`` is ``(n, m)``, ``target`` and ``weights`` ``(n,)``,
    ``lower`` / ``upper`` ``(m,)`` with ``+-inf`` for an open side.

    Returns ``(coefficients, active)``: ``active[j]`` is ``'lower'`` / ``'upper'`` when that
    bound held coefficient ``j``, else ``None``. The unconstrained solve is the minimum-norm
    least-squares solution, so a singular design (a simulated column constant over the
    group) returns the pseudo-inverse answer rather than raising; the bounded solve runs
    only when the unconstrained one leaves the box, and is exact (bounded-variable least
    squares), not the unconstrained solution clipped -- clamping one coefficient moves
    where the others belong.
    """
    phi = np.asarray(phi, dtype=float)
    target = np.asarray(target, dtype=float)
    sw = np.sqrt(np.asarray(weights, dtype=float))
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    a = phi * sw[:, None]
    b = target * sw
    coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    active = [None] * len(coef)
    if np.all(coef >= lower - tol) and np.all(coef <= upper + tol):
        return coef, active
    from scipy.optimize import lsq_linear
    result = lsq_linear(a, b, bounds=(lower, upper), method='bvls')
    coef = np.asarray(result.x, dtype=float)
    scale = np.maximum(1.0, np.abs(coef))
    for j in range(len(coef)):
        if np.isfinite(lower[j]) and abs(coef[j] - lower[j]) <= tol * scale[j]:
            active[j] = 'lower'
        elif np.isfinite(upper[j]) and abs(coef[j] - upper[j]) <= tol * scale[j]:
            active[j] = 'upper'
    return coef, active
