"""Analytical, file-free objective models -- a negative log-likelihood computed directly
from the free parameters, with no external simulator. Used with objfunc = direct_pass
(``objective = score`` / a named target / an inline expression; ADR-0031/0050/0059).

Two non-simulator :class:`~pybnf.pset.Model` subclasses share the ``score``-column seam:

* :class:`AnalyticalModel` -- one of a closed *menu* of built-in benchmark targets, read
  from a ``.target`` JSON file or declared inline on the objective line (ADR-0059 item 6):

    gaussian         - Axis-aligned Gaussian (diagonal variance; a *separable* objective)
    rotated_gaussian - Correlated Gaussian with a full covariance Sigma (non-separable)
    rotated_quartic  - Smooth, non-separable, NON-quadratic, trap-free valley (2D)
    banana           - Rosenbrock/banana-shaped distribution (2D)
    multimodal       - Mixture of Gaussians with configurable modes

  Coordinates bind to free parameters **by name** (``coordinate_order``): coordinate i is
  the declared parameter whose name ends in index ``i+1`` (``x1``/``p1`` -> coordinate 1),
  independent of declaration order and lexical sort -- the bind-by-name contract (ADR-0034),
  replacing the old silent sorted-positional convention.

* :class:`ExpressionModel` -- a **bring-your-own** target (ADR-0050): the user writes the NLL
  as a PEtab-math expression over the declared free parameters on the config line
  (``objective = expression`` + ``expression = 0.5*((1 - x1)^2 + 100*(x2 - x1^2)^2)``), with
  no model file and no ``.exp``. The expression compiles to a numpy callable
  (``pybnf.petab.formula.compile_objective_expression``) and its free symbols bind to PSet
  values **by name** (``x1``, ``x2``) -- the bind-by-name fix the menu's sorted-positional
  convention did not need.
"""

import copy
import json
import logging
import re
import time
import numpy as np
from os.path import splitext, basename

from .data import Data
from .pset import Model
from .printing import PybnfError

logger = logging.getLogger(__name__)

#: Built-in analytical targets that can be declared *inline* on the objective line
#: (ADR-0059 item 6: ``objective = banana, a = 1, b = 100``), mapped to their constants'
#: documented defaults (applied + echoed at run start when the user omits them). These are
#: the scalar-constant targets; the matrix/mixture targets (rotated_gaussian / multimodal)
#: do not fit a config line and keep a ``.target`` JSON sidecar. The parser (parse.py) keeps
#: the names literal to avoid an import cycle; this dict is the canonical home.
INLINE_TARGET_DEFAULTS = {
    'banana': {'a': 1.0, 'b': 100.0},
}
INLINE_TARGET_TYPES = frozenset(INLINE_TARGET_DEFAULTS)


class AnalyticalModel(Model):
    """
    A model that computes a target score directly from free parameters.

    Reads a .target JSON file specifying the target type and parameters.
    Returns a Data object with a single 'score' column containing the NLL.
    """

    def __init__(self, target_file=None, pset=None, *, target_def=None, name=None):
        """Build from a ``.target`` JSON file (``target_file``), or directly from an
        in-memory ``target_def`` dict + ``name``.

        The in-memory path is the seam the named-objective grammar uses (ADR-0059): an
        ``objective = banana, a = 1, b = 100`` config line is parsed to a ``target_def``
        (``{'type': 'banana', 'a': 1.0, 'b': 100.0}``) and synthesized here with no JSON
        sidecar on disk -- the structured matrix/mixture targets still ship a ``.target``
        file. ``file_path`` (used as the per-evaluation model prefix) falls back to the
        name, since there is no file."""
        if target_def is not None:
            if name is None:
                raise ValueError('AnalyticalModel(target_def=...) also needs a name')
            self.target_def = target_def
            self.name = name
            self.file_path = target_file if target_file is not None else name
        else:
            if target_file is None:
                raise ValueError('AnalyticalModel needs either a target_file or a target_def')
            self.file_path = target_file
            self.name = splitext(basename(target_file))[0]
            with open(target_file, encoding='utf-8') as f:
                self.target_def = json.load(f)

        self.suffixes = ['target']
        self.stochastic = False
        self.has_observables = True
        self.param_names = set()  # All params come from the config, not the model file

        self.target_type = self.target_def['type']
        self._pset = pset

        # Pre-compute target-specific constants
        if self.target_type == 'gaussian':
            self._mean = np.array(self.target_def['mean'])
            self._var = np.array(self.target_def['variance'])
            self._inv_var = 1.0 / self._var
        elif self.target_type == 'rotated_gaussian':
            self._mean = np.array(self.target_def['mean'], dtype=float)
            self._cov = np.array(self.target_def['covariance'], dtype=float)
            # Precision Sigma^{-1}; symmetrize to clear any inversion round-off so
            # the quadratic form stays exactly symmetric.
            prec = np.linalg.inv(self._cov)
            self._prec = 0.5 * (prec + prec.T)
        elif self.target_type == 'rotated_quartic':
            self._mean = np.array(self.target_def['mean'], dtype=float)
            angle = float(self.target_def['angle'])
            c, s = np.cos(angle), np.sin(angle)
            self._rot = np.array([[c, -s], [s, c]])   # R(angle); r = R (x - mu)
            self._coeff = np.array(self.target_def['coeff'], dtype=float)  # (k1, k2)
        elif self.target_type == 'banana':
            self._a = self.target_def.get('a', 1.0)
            self._b = self.target_def.get('b', 100.0)
        elif self.target_type == 'multimodal':
            self._modes = []
            for mode in self.target_def['modes']:
                w = mode['weight']
                mu = np.array(mode['mean'])
                var = np.array(mode['variance'])
                self._modes.append((np.log(w), mu, 1.0 / var))
        else:
            raise ValueError(f'Unknown analytical target type: {self.target_type}')

    def copy_with_param_set(self, pset):
        m = copy.copy(self)
        m._pset = pset
        return m

    def nll_jax(self):
        """Return a JAX-traceable ``f(theta) -> NLL`` for this target (ADR-0059).

        The gradient-based reference sampler (``job_type = hmc``) differentiates
        this through ``jax.grad`` to drive blackjax NUTS. It is the JAX peer of
        :meth:`_compute_nll` -- same closed form, same precomputed constants
        (``_mean``/``_inv_var``/``_prec``), so the numpy ``execute()`` score path
        and the JAX log-density share one source of truth. ``theta`` is a JAX
        array of the parameter values in the order the sampler binds them.

        The first HMC slice hand-wrote the two closed-form-truth oracle targets
        -- ``gaussian`` (a diagonal quadratic form) and ``rotated_gaussian`` (a
        full-precision quadratic form). This second slice adds the two canonical
        *stress* geometries -- ``banana`` (a curved, non-Gaussian valley) and
        ``multimodal`` (a separated-mode mixture) -- so HMC can serve as the
        reference yardstick that scores the gradient-free samplers on the hard
        cases (ADR-0059's stated purpose). Each JAX branch mirrors its numpy peer
        (``_nll_banana`` / ``_nll_multimodal`` / ``_nll_rotated_quartic``) term for
        term off the *same* precomputed constants, so the score path and the JAX
        log-density stay one source of truth. Every menu ``target_type`` now has a
        JAX branch (ADR-0059 item 2 added ``rotated_quartic``); the BYO ``expression``
        target carries its own ``nll_jax`` on :class:`ExpressionModel`. The trailing
        ``raise`` is therefore a defensive guard for an unrecognized ``target_type``,
        not a deferred-slice boundary."""
        import jax.numpy as jnp
        if self.target_type == 'gaussian':
            mean = jnp.asarray(self._mean)
            inv_var = jnp.asarray(self._inv_var)

            def nll(theta):
                diff = theta - mean
                return 0.5 * jnp.sum(diff * diff * inv_var)
            return nll
        if self.target_type == 'rotated_gaussian':
            mean = jnp.asarray(self._mean)
            prec = jnp.asarray(self._prec)

            def nll(theta):
                diff = theta - mean
                return 0.5 * (diff @ prec @ diff)
            return nll
        if self.target_type == 'banana':
            # 0.5 * sum_i [(a - x_i)^2 + b (x_{i+1} - x_i^2)^2] -- the vectorized
            # peer of the _nll_banana loop (the slices x[:-1]/x[1:] are the
            # consecutive (x_i, x_{i+1}) pairs), exact for any dimension.
            a, b = self._a, self._b

            def nll(theta):
                x_i, x_next = theta[:-1], theta[1:]
                return 0.5 * jnp.sum((a - x_i) ** 2 + b * (x_next - x_i ** 2) ** 2)
            return nll
        if self.target_type == 'multimodal':
            # -logsumexp_k [ log w_k - 0.5 (x - mu_k)^T Sigma_k^{-1} (x - mu_k) ],
            # the JAX peer of _nll_multimodal: jax.scipy's logsumexp supplies the
            # same max-shift numerical stabilization the numpy branch hand-rolls.
            from jax.scipy.special import logsumexp
            modes = [(float(log_w), jnp.asarray(mu), jnp.asarray(inv_var))
                     for log_w, mu, inv_var in self._modes]

            def nll(theta):
                log_components = jnp.stack([
                    log_w - 0.5 * jnp.sum((theta - mu) ** 2 * inv_var)
                    for log_w, mu, inv_var in modes])
                return -logsumexp(log_components)
            return nll
        if self.target_type == 'rotated_quartic':
            # k1 * r1^4 + k2 * r2^2 with r = R(angle) (x - mu) -- the jnp peer of
            # _nll_rotated_quartic term for term (quartic along the first rotated axis,
            # quadratic along the second), off the same precomputed rotation / coeffs.
            mean = jnp.asarray(self._mean)
            rot = jnp.asarray(self._rot)
            k1, k2 = self._coeff

            def nll(theta):
                r = rot @ (theta - mean)
                return k1 * r[0] ** 4 + k2 * r[1] ** 2
            return nll
        # Defensive fallback: every menu target_type above has a JAX branch, and the BYO
        # expression target carries its own nll_jax (ExpressionModel), so this is reachable
        # only for an unrecognized target_type (config validates the menu names upstream).
        raise PybnfError(
            "job_type = hmc has no JAX log-density for the analytical target %r (ADR-0059). "
            "The supported menu targets are 'gaussian', 'rotated_gaussian', 'banana', "
            "'multimodal', and 'rotated_quartic'; a bring-your-own 'expression' target carries "
            "its own JAX log-density. Run a gradient-free sampler (am / dream / p_dream) instead."
            % self.target_type)

    def save(self, file_prefix, **kwargs):
        pass

    def get_suffixes(self):
        return self.suffixes

    def execute(self, folder, filename, timeout):
        """Compute the NLL score from the current parameter set."""
        # Small delay to prevent dask race condition with instant-completion tasks
        time.sleep(0.01)
        params = self._get_param_values()
        score = self._compute_nll(params)

        # Return Data with 'index' and 'score' columns (index is the independent variable)
        data = Data(arr=np.array([[0.0, score]]))
        data.cols = {'index': 0, 'score': 1}
        data.headers = {0: 'index', 1: 'score'}
        return {'target': data}

    def _get_param_values(self):
        """Extract parameter values as a numpy array in **coordinate order** (bind-by-name).

        The score path's half of the bind-by-name contract: the values are ordered by
        :meth:`coordinate_order` (the integer index in each parameter name), so the i-th
        element is the parameter the user named for coordinate i+1 -- not whichever name
        happened to sort first."""
        if self._pset is None:
            raise ValueError('AnalyticalModel has no parameter set')
        names = self.coordinate_order(self._pset.keys())
        return np.array([self._pset[n] for n in names])

    def _dimension(self, n_declared):
        """The target's coordinate dimension: intrinsic for the fixed-shape targets (the
        mean / mixture-component vector length), and the number of declared parameters for
        the any-dimension banana."""
        if self.target_type in ('gaussian', 'rotated_gaussian', 'rotated_quartic'):
            return len(self._mean)
        if self.target_type == 'multimodal':
            return len(self._modes[0][1])   # (log_w, mu, inv_var) -> mu length
        return n_declared                   # banana: generalizes to any dimension

    def coordinate_order(self, param_names):
        """The declared free-parameter names ordered by the integer index in each name
        (``x1`` -> coordinate 1, ``x2`` -> coordinate 2, ...) -- the bind-by-name contract
        (ADR-0034) that replaces the silent sorted-positional convention.

        A menu target's coordinates are anonymous (banana coordinate 0, gaussian ``mean[0]``,
        ...), so binding free parameters to them needs a deterministic, user-controllable rule.
        The integer suffix of each parameter name **is** that rule: it names the coordinate, so
        the order is independent of declaration order and of lexical sort (``x10`` correctly
        follows ``x9``, which a lexical sort got wrong). Any prefix works (``x1`` / ``p1`` /
        ``theta1``); the index set must be exactly ``1..D`` for the target's dimension ``D``
        (``D`` = the number of declared parameters for the any-dimension banana). Used by both
        the numpy ``execute`` score path (:meth:`_get_param_values`) and the JAX HMC path, so
        the two never disagree.

        Raises ``PybnfError`` when a name has no integer suffix, or the indices are not exactly
        ``1..D`` (wrong count, a gap, or a duplicate) -- naming the offending parameters and the
        expected coordinate names, instead of silently binding the wrong coordinate (the #425
        footgun the sorted convention hid)."""
        names = list(param_names)
        d = self._dimension(len(names))
        expected = ', '.join(f'x{i}' for i in range(1, d + 1))
        indexed, unindexed = [], []
        for n in names:
            m = re.search(r'(\d+)$', n)
            if m:
                indexed.append((int(m.group(1)), n))
            else:
                unindexed.append(n)
        if unindexed:
            raise PybnfError(
                f"Cannot bind free parameter(s) {sorted(unindexed)} to the "
                f"'{self.target_type}' target's coordinates.",
                f"A menu analytical target binds coordinates to parameters by the integer "
                f"index in the parameter name (ADR-0034 bind-by-name), so each name must end "
                f"in its coordinate index. Name the {d} coordinate(s) {expected} (any prefix "
                f"works, e.g. p1..p{d}); got {sorted(unindexed)} with no index.")
        indexed.sort()
        indices = [i for i, _ in indexed]
        if indices != list(range(1, d + 1)):
            raise PybnfError(
                f"The '{self.target_type}' target has {d} coordinate(s) ({expected}), but the "
                f"declared free parameters carry indices {indices}.",
                f"Declare exactly the coordinates {expected} (any prefix; the indices must be "
                f"1..{d} with no gaps or duplicates).")
        return [n for _, n in indexed]

    def _compute_nll(self, params):
        """Compute negative log-likelihood for the target distribution."""
        if self.target_type == 'gaussian':
            return self._nll_gaussian(params)
        elif self.target_type == 'rotated_gaussian':
            return self._nll_rotated_gaussian(params)
        elif self.target_type == 'rotated_quartic':
            return self._nll_rotated_quartic(params)
        elif self.target_type == 'banana':
            return self._nll_banana(params)
        elif self.target_type == 'multimodal':
            return self._nll_multimodal(params)
        else:
            # Unreachable in practice (__init__ already rejects unknown types),
            # but fail loud rather than return an implicit None if a new target
            # type is ever added to __init__ but not here.
            raise ValueError(f'Unknown analytical target type: {self.target_type}')

    def _nll_gaussian(self, params):
        """NLL of multivariate Gaussian: 0.5 * sum((x - mu)^2 / sigma^2)"""
        diff = params - self._mean
        return 0.5 * np.sum(diff ** 2 * self._inv_var)

    def _nll_rotated_gaussian(self, params):
        """NLL of a correlated (rotated) multivariate Gaussian with full
        covariance Sigma: ``0.5 * (x - mu)^T Sigma^{-1} (x - mu)``.

        Unlike the axis-aligned ``gaussian`` (diagonal variance, a *separable*
        objective whose principal axes are the coordinate axes), the off-diagonal
        precision couples the coordinates, so the quadratic bowl's principal axes
        are rotated off the coordinate axes. That is the textbook validator for
        conjugate-direction (Powell) and covariance-adapting (CMA-ES) methods:
        coordinate-only descent zig-zags, while those methods discover the
        rotation (#405). The mode is still ``mu`` (NLL 0 there).
        """
        diff = params - self._mean
        return 0.5 * float(diff @ self._prec @ diff)

    def _nll_rotated_quartic(self, params):
        """NLL of a smooth, non-separable, NON-quadratic, trap-free valley:
        ``k1 * r1**4 + k2 * r2**2`` where ``r = R(angle) (x - mu)``.

        Quartic along the first rotated axis, quadratic along the second; with
        ``k1 << k2`` this is a long, flat, curved valley. Its *only* stationary
        point is ``mu`` (the mode, NLL 0), so it is trap-free for a local
        optimizer — unlike the banana. Because it is non-quadratic, a single
        fixed-step parabolic line search is a poor 1-D model and converges slowly
        / stalls, whereas a bracketing + Brent line search follows the valley.
        This is the discriminating target for Powell's robustified line search
        (#406); the rotated *Gaussian* (quadratic) cannot discriminate, since a
        parabola fits a quadratic exactly.
        """
        r = self._rot @ (params - self._mean)
        return self._coeff[0] * r[0] ** 4 + self._coeff[1] * r[1] ** 2

    def _nll_banana(self, params):
        """
        NLL of Rosenbrock/banana distribution:
        -log p(x1, x2) = 0.5 * [(a - x1)^2 + b * (x2 - x1^2)^2]

        Generalizes to d dimensions as:
        -log p(x) = 0.5 * sum_{i=1}^{d-1} [(a - x_i)^2 + b * (x_{i+1} - x_i^2)^2]
        """
        a, b = self._a, self._b
        nll = 0.0
        for i in range(len(params) - 1):
            nll += 0.5 * ((a - params[i]) ** 2 + b * (params[i + 1] - params[i] ** 2) ** 2)
        return nll

    def _nll_multimodal(self, params):
        """
        NLL of a mixture of Gaussians:
        -log p(x) = -log sum_k w_k * N(x; mu_k, Sigma_k)
                   = -logsumexp(log(w_k) - 0.5 * (x - mu_k)^T Sigma_k^{-1} (x - mu_k))
        """
        log_components = []
        for log_w, mu, inv_var in self._modes:
            diff = params - mu
            log_density = log_w - 0.5 * np.sum(diff ** 2 * inv_var)
            log_components.append(log_density)
        # logsumexp for numerical stability
        max_log = max(log_components)
        log_sum = max_log + np.log(sum(np.exp(lc - max_log) for lc in log_components))
        return -log_sum


class ExpressionModel(Model):
    """A bring-your-own analytical objective: the user's closed-form NLL as a math expression
    over the free parameters, with no model file (ADR-0050, the #425 "Tier 1" surface).

    A sibling of :class:`AnalyticalModel` -- another non-simulator :class:`~pybnf.pset.Model`
    whose :meth:`execute` emits a one-cell ``score`` column the ``DirectPassObjective`` reads
    straight through (so no new objective, sampler, or run-loop code). The differences from the
    built-in *menu* are the two ADR-0050 wins:

    * **Bring-your-own** -- the target is the user's ``expression = ...`` config line, compiled
      to a numpy callable via the shared sympy backend
      (:func:`pybnf.petab.formula.compile_objective_expression`), not one of five hardcoded
      enum types.
    * **Bind-by-name** -- the expression's free symbols bind to PSet values *by declared name*
      (``x1`` -> ``pset['x1']``), not by sorted position. An expression names its variables, so
      positional binding would be a footgun; this is the fix the menu path deferred.

    A separate class (over an ``'expression'`` ``target_type`` on ``AnalyticalModel``) is the
    right call: bind-by-name is intrinsically different from the menu's sorted-positional
    ``_get_param_values`` / closed-form ``_compute_nll`` / ``nll_jax`` dispatch, and the state
    is different too (a compiled callable + ordered symbol names, not precomputed matrices and a
    type enum). ADR-0050 names the class. It still reuses the *whole* synthesis / injection /
    score path (config synthesizes + injects it fileless exactly like the menu target, and
    ``DirectPassObjective`` scores its ``score`` cell unchanged) -- that reuse lives at the
    seam, not in the model class.

    Holds the *expression string* (picklable) and the ordered free-symbol names; the lambdified
    callables are **not** picklable, so they are compiled lazily and dropped from the pickle state
    (recompiled once per dask worker). For ``job_type = hmc`` it also exposes :meth:`nll_jax` --
    the *same* sympy expression lambdified with the JAX backend (ADR-0059 item 2), so HMC
    ``jax.grad``s a user's bring-your-own log-density exactly as it does the built-in menu targets.
    """

    def __init__(self, formula, ordered_names, name, *, pset=None):
        """``formula`` is the PEtab-math NLL string; ``ordered_names`` the sorted free-symbol
        names it expects positionally (from :func:`compile_objective_expression`); ``name`` the
        synthesized model id (also the per-evaluation file prefix -- there is no file)."""
        self.formula = formula
        self._ordered_names = list(ordered_names)
        self.name = name
        self.file_path = name
        self.suffixes = ['expression']
        self.stochastic = False
        self.has_observables = True
        self.param_names = set()  # All params come from the config, not a model file
        self._pset = pset
        # The lambdify-generated callables are not picklable (no importable qualname), so each is
        # compiled on first use and re-derived after unpickling rather than carried across the
        # dask boundary -- see _compiled() / nll_jax() and __getstate__. `_func` is the numpy
        # score-path callable; `_jax_func` the JAX peer the HMC gradient path uses.
        self._func = None
        self._jax_func = None

    def _compiled(self):
        """The numpy callable for the expression, compiled lazily and memoized.

        ``_ordered_names`` is exactly the expression's free-symbol set, so passing it as the
        allowed namespace re-validates trivially (the real declared-parameter validation
        happened once at config load). One compile per process; ``copy_with_param_set`` shares
        the memoized callable, so a fit's many per-pset copies do not recompile."""
        if self._func is None:
            from .petab.formula import compile_objective_expression
            self._func, _ = compile_objective_expression(self.formula, self._ordered_names)
        return self._func

    def coordinate_order(self, param_names):
        """The expression's free-symbol names, in the order :meth:`nll_jax` consumes them
        (ADR-0059 item 2 / ADR-0034 bind-by-name).

        For an expression the coordinates *are* the named free symbols, so this is simply
        ``_ordered_names`` -- the symbol order the lambdified callable expects positionally. The
        gradient-based ``hmc`` sampler permutes its ``self.variables``-order ``u`` into this order
        (``HMCSampler._coordinate_permutation``) so a parameter binds to the symbol of the same
        name, not by declaration position. A declared parameter the expression does not reference
        is simply absent (the likelihood is flat in it; its prior still samples it). Every name is
        a declared free parameter (config validated), so ``param_names`` only sanity-bounds them."""
        missing = [n for n in self._ordered_names if n not in set(param_names)]
        if missing:                                     # defensive: config load already guarantees this
            raise PybnfError(
                f"The objective expression references {missing}, which are not among the declared "
                f"free parameters {sorted(param_names)} (ADR-0050 bind-by-name).")
        return list(self._ordered_names)

    def nll_jax(self):
        """Return a JAX-traceable ``f(theta) -> NLL`` for the expression (ADR-0059 item 2).

        The JAX peer of :meth:`_compiled`: the *same* sympy expression, lambdified with the JAX
        backend (``compile_objective_expression(..., backend='jax')``), so the numpy score path and
        the differentiable HMC log-density are one source of truth -- they cannot drift, sharing the
        parse, the validation, and the bind-by-name ``_ordered_names`` ordering. ``theta`` is a JAX
        array of the referenced symbols' values in ``_ordered_names`` (= :meth:`coordinate_order`)
        order, which is exactly what the HMC permutation delivers; it is unpacked into the callable's
        positional scalar arguments. Compiled once and memoized (HMC runs in-process, so this never
        crosses the dask boundary)."""
        if self._jax_func is None:
            from .petab.formula import compile_objective_expression
            self._jax_func, _ = compile_objective_expression(
                self.formula, self._ordered_names, backend='jax')
        jax_func = self._jax_func
        n = len(self._ordered_names)

        def nll(theta):
            return jax_func(*(theta[i] for i in range(n)))
        return nll

    def __getstate__(self):
        # Drop the lambdified callables (unpicklable); _compiled() / nll_jax() rebuild them from
        # the picklable formula string + ordered_names on the worker. (HMC keeps the model
        # in-process, so _jax_func is only ever built there; dropped here for the de/am/dream
        # dask path's safety.)
        state = self.__dict__.copy()
        state['_func'] = None
        state['_jax_func'] = None
        return state

    def copy_with_param_set(self, pset):
        m = copy.copy(self)
        m._pset = pset
        return m

    def save(self, file_prefix, **kwargs):
        pass

    def get_suffixes(self):
        return self.suffixes

    def execute(self, folder, filename, timeout):
        """Evaluate the expression NLL at the current parameter set (bind-by-name) and return
        it in a one-cell ``score`` column, mirroring :meth:`AnalyticalModel.execute`."""
        # Small delay to prevent dask race condition with instant-completion tasks (the
        # integration harness patches this module's time.sleep to a no-op).
        time.sleep(0.01)
        if self._pset is None:
            raise ValueError('ExpressionModel has no parameter set')
        func = self._compiled()
        try:
            args = [self._pset[name] for name in self._ordered_names]
        except KeyError as e:
            # Defensive: config-time validation already guarantees every free symbol is a
            # declared free parameter (hence present in the PSet), so this should be unreachable.
            raise PybnfError(
                f"The objective expression references free parameter {e}, which is not in the "
                f"parameter set. Declared free parameters bind to the expression's symbols by "
                f"name (ADR-0050).")
        score = float(func(*args))
        data = Data(arr=np.array([[0.0, score]]))
        data.cols = {'index': 0, 'score': 1}
        data.headers = {0: 'index', 1: 'score'}
        return {self.suffixes[0]: data}
