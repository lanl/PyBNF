"""Measurement-time uncertainty: marginalize the latent sampling time out of the likelihood
(ADR-0112, issue #587).

A datum ``(t_k, ȳ_k)`` is normally scored at the single prediction ``y(t_k, θ)`` read off the
trajectory at its reported time. When the sampling time is itself uncertain, the latent time
``τ_k ~ p(τ | t_k)`` is integrated out, so the per-observation likelihood becomes

    z_k(θ) = ∫_{t_0}^{t_max}  p(ȳ_k | y(τ, θ))  p(τ | t_k)  dτ            (ADR-0112, Eq. 18)

and the objective is ``J(θ) = −Σ_k log z_k(θ) − log p(θ)``. This module implements the phase-1
(quadrature-over-the-stored-trajectory) engine and the ``time_error`` clause that selects it.

The three pieces mirror the noise surface (ADR-0011/0021/0058) one-for-one:

    noise surface (existing)            time-error surface (this module)
    --------------------------          --------------------------------
    NoiseModel  (family kernel)         TimeErrorPrior  (time-prior shape)
    SigmaSource (fit/fix_at/…)          TimeErrorSource (fit/fix_at)
    LikelihoodObjective                 MarginalizedTimeObjective
    _build_noise_spec                   build_time_error_spec

so the clause reuses the *density* every noise family already computes
(:meth:`pybnf.noise.base.NoiseModel.log_density`) as the integrand, and adds only the outer
one-dimensional integral. Nothing is added to the model file; the truncated-normal ``erf``
normalizer lives here in Python, never in the model language (ADR-0112 "the quadrature engine").

Phase-1 scope (ADR-0112 "ship the quadrature engine first"): a **whole-fit** ``time_error``
clause; a noise scale that is **constant / free / a data column** (``fix_at`` / ``fit`` /
``read_exp_file`` -- σ does not vary with the latent time τ). A **per-observable** time prior, a
**prediction-dependent σ** (``relative`` / ``formula`` / ``prediction_formula``, whose σ would
vary across the integration window), and a **count family** integrand are deferred to follow-ups
and refused at build. Phase 2 (the augmented ODE that gives error-controlled integration and
``dz_k/dθ`` for gradient methods) is a separate ADR; the gradient seam raises
``NotImplementedError`` pointing at it.

Storage / dispatch. The clause is parsed by ``parse.py`` onto the ``noise_model`` line and
stored under its own ``('time_error', observable)`` structural key -- the ``cumulative`` pattern,
so the ``(family, fields, location)`` noise tuple is untouched. ``config.py``'s ``_load_obj_func``
detects the key, edition-gates it, and (unless ``σ_t`` is a fixed 0 -- the standard-likelihood
limit, kept as an ordinary :class:`~pybnf.objective.LikelihoodObjective`) swaps the built
per-point objective for a :class:`MarginalizedTimeObjective` via :func:`build_time_error_objective`.
"""

from abc import ABC, abstractmethod

import numpy as np

from ..printing import PybnfError
from ..objective import SummationObjective


# ============================================================================================
# The time prior  p(τ | t_k)  -- a small family, abstract on its 2nd member (ADR-0011 bar)
# ============================================================================================

class TimeErrorPrior(ABC):
    """A distribution over the latent sampling time ``τ`` given the reported time ``t_k``,
    truncated to the marginal's support ``[t_0, t_max]`` (ADR-0112, Eq. 15).

    Distinct from a :class:`pybnf.priors.base.Prior`: this is an *integration kernel* evaluated
    at data-driven centres ``t_k`` over a grid of ``τ`` values, never sampled as a search
    coordinate. Two operations are required -- the density over a grid and the truncation
    normalizer -- and every messy special function (the truncated-normal ``erf``) stays here in
    Python, out of the model language.
    """

    #: The clause literal that selects this member (``time_error = <name>``).
    name = None

    @abstractmethod
    def unnormalized_density(self, tau_grid, t_k, sigma_t):
        """The prior's *un-normalized* density ``∝ p(τ | t_k)`` evaluated on ``tau_grid``
        (a 1-D array of quadrature nodes). The normalizer (which folds the ``[t_0, t_max]``
        truncation) is applied separately by :meth:`log_normalizer`, so a caller can keep the
        integrand in the un-normalized-times-kernel form the augmented ODE (phase 2) also uses.
        """

    @abstractmethod
    def log_normalizer(self, t_k, sigma_t, t0, tmax):
        """``log`` of the constant that turns :meth:`unnormalized_density` into a proper density
        over ``[t_0, t_max]``. For the truncated normal this is where ``erf`` (via
        ``scipy.stats.norm.cdf``) enters, in Python -- never in the model."""


class TruncatedNormalTimeError(TimeErrorPrior):
    """The paper's default (Eq. 25): ``τ ~ N(t_k, σ_t²)`` truncated to ``[t_0, t_max]``."""

    name = 'truncated_normal'

    def unnormalized_density(self, tau_grid, t_k, sigma_t):
        # Gaussian kernel centred at t_k; the 1/(√(2π) σ_t Z_k) constant is in log_normalizer.
        z = (np.asarray(tau_grid, dtype=float) - t_k) / sigma_t
        return np.exp(-0.5 * z * z)

    def log_normalizer(self, t_k, sigma_t, t0, tmax):
        from scipy.stats import norm
        # 1/(√(2π) σ_t) · 1/(Φ((tmax−t_k)/σ_t) − Φ((t0−t_k)/σ_t)) ; return its log.
        z = norm.cdf((tmax - t_k) / sigma_t) - norm.cdf((t0 - t_k) / sigma_t)
        if z <= 0.0:
            # The reported time's window carries no mass inside the support -- an ill-posed
            # datum, refused rather than divided by zero (ADR-0112 "the refusals").
            raise PybnfError(
                f'time_error: truncated-normal support is empty for t_k={t_k}, sigma_t={sigma_t}',
                'The truncation interval [t_0, t_max] contains no probability mass for this '
                'measurement. Widen the support or check the reported time.')
        return -0.5 * np.log(2.0 * np.pi) - np.log(sigma_t) - np.log(z)


class UniformTimeError(TimeErrorPrior):
    """A flat timing-error window ``τ ~ U(t_k − w, t_k + w)`` (``σ_t`` read as the half-width
    ``w``), clipped to ``[t_0, t_max]`` -- the second member, for a bounded-but-unknown error."""

    name = 'uniform'

    def unnormalized_density(self, tau_grid, t_k, sigma_t):
        tau = np.asarray(tau_grid, dtype=float)
        return ((tau >= t_k - sigma_t) & (tau <= t_k + sigma_t)).astype(float)

    def log_normalizer(self, t_k, sigma_t, t0, tmax):
        lo, hi = max(t0, t_k - sigma_t), min(tmax, t_k + sigma_t)
        if hi <= lo:
            raise PybnfError(
                f'time_error: uniform window is empty for t_k={t_k}, half-width={sigma_t}',
                'The window [t_k−w, t_k+w] does not intersect [t_0, t_max].')
        return -np.log(hi - lo)


#: ``time_error = <token>`` -> its TimeErrorPrior. Two members clear the ADR-0011 "abstract on
#: the 2nd member" bar; a third (e.g. a log-normal delay) is a one-line addition.
_TIME_ERROR_PRIORS = {
    TruncatedNormalTimeError.name: TruncatedNormalTimeError,
    UniformTimeError.name: UniformTimeError,
}


# ============================================================================================
# The σ_t source  -- fit/fix_at, paralleling SigmaSource but consumed by the time prior
# ============================================================================================

class TimeErrorSource(ABC):
    """Where ``σ_t`` comes from at evaluation time. Parallels
    :class:`pybnf.noise.source.SigmaSource`, but feeds the time prior, not a family's data_fit.
    Only the two search-relevant verbs are meaningful for a time-error scale: estimate it
    (``fit``) or hold it (``fix_at``). A data-column or prediction-dependent σ_t is not defined."""

    #: True iff this source is a searched free parameter (so `k` counts it, ADR-0112).
    estimated = False

    @abstractmethod
    def value(self, pset_values):
        """The scalar ``σ_t`` for the current parameter set (``{name: value}``)."""

    def required_free_param(self):
        """The free-parameter name this source reads, or ``None`` (a fixed source)."""
        return None


class FixedTimeError(TimeErrorSource):
    """``sigma_t = fix_at <number>``. ``fix_at 0`` is the standard likelihood: the dispatch in
    :func:`build_time_error_objective` short-circuits it back to the plain per-point objective."""

    def __init__(self, value):
        self._value = float(value)

    @property
    def is_zero(self):
        return self._value == 0.0

    def value(self, pset_values):
        return self._value


class FreeParameterTimeError(TimeErrorSource):
    """``sigma_t = fit <param__FREE>``. One extra search dimension -- not ``n_t`` -- read from
    the PSet by name, exactly like a :class:`FreeParameterSigma` (ADR-0112 "sigma_t is a new
    measurement parameter")."""

    estimated = True
    is_zero = False

    def __init__(self, param_name):
        self.param_name = param_name

    def required_free_param(self):
        return self.param_name

    def value(self, pset_values):
        return float(pset_values[self.param_name])


def _build_time_error_source(verb, arg):
    """One ``sigma_t = <verb> [<arg>]`` field -> its TimeErrorSource. Only ``fit`` / ``fix_at``."""
    verb = verb.lower()
    if verb == 'fit':
        if arg is None:
            raise PybnfError('time_error: "sigma_t = fit" needs a parameter name',
                             'Use "sigma_t = fit st__FREE" to estimate the timing-error scale.')
        return FreeParameterTimeError(arg)
    if verb == 'fix_at':
        if arg is None:
            raise PybnfError('time_error: "sigma_t = fix_at" needs a number',
                             'Use "sigma_t = fix_at 0.5" to hold the timing-error scale.')
        return FixedTimeError(arg)
    raise PybnfError(f'time_error: unsupported sigma_t source "{verb}"',
                     'A timing-error scale is either estimated ("sigma_t = fit <param__FREE>") '
                     'or fixed ("sigma_t = fix_at <number>"); '
                     f'"{verb}" (a prediction- or data-dependent scale) is not defined for the '
                     'latent measurement time.')


# ============================================================================================
# The objective  -- integrate the datum over the trajectory, datum as a constant
# ============================================================================================

# The σ sources whose value does NOT depend on the (per-τ) prediction, so σ is one scalar over
# the whole integration window (ADR-0112 phase-1 scope). A prediction-dependent σ would make the
# integrand's scale vary with τ -- deferred, and refused at build.
_TAU_INDEPENDENT_SIGMA = ('FreeParameterSigma', 'ConstantSigma', 'DataColumnSigma')


class MarginalizedTimeObjective(SummationObjective):
    """The marginal-time likelihood (ADR-0112): per datum, ``−log ∫ p(ȳ_k|y(τ)) p(τ|t_k) dτ``,
    integrated over the *whole* trajectory rather than scored at one matched row.

    Reuses the surrounding surface: it is a ``SummationObjective`` (so constraints, the base
    ``evaluate_multiple`` that sets ``_pset_values`` and applies the measurement layer, and the
    per-observable noise ``overrides`` all apply), and its integrand's observation-likelihood
    factor is ``NoiseModel.log_density`` -- the same density LOO/WAIC consume (ADR-0056). What
    differs from :class:`~pybnf.objective.LikelihoodObjective` is the per-datum loop: ``ȳ_k`` is a
    constant of the integrand and ``y(τ)`` is the full column, so ``_sim_row_for`` is not used.
    """

    #: The marginal per-observation contribution ``log z_k`` *is* a genuine normalized
    #: per-observation log-likelihood, so LOO/WAIC and information_criteria.txt work unchanged
    #: (ADR-0056, ADR-0112): ``evaluate_pointwise`` records ``log z_k`` per datum, and ``k``
    #: already counts an estimated ``σ_t`` (it is a declared free parameter, like a fitted σ).
    supports_pointwise_log_likelihood = True

    def __init__(self, ind_var_rounding=0, overrides=None, noise=None, sigma_source=None,
                 sigma_sources=None, time_prior=None, sigma_t_source=None, support=None):
        super().__init__(ind_var_rounding)
        # Reuse the LikelihoodObjective spec plumbing verbatim for the *noise* half.
        self.overrides = dict(overrides) if overrides else {}
        self.noise = noise
        self.sigma_source = sigma_source
        self._default_source_map = dict(sigma_sources) if sigma_sources is not None else None
        # The time half (this module's addition).
        self.time_prior = time_prior              # TimeErrorPrior
        self.sigma_t_source = sigma_t_source      # TimeErrorSource
        self.support = support                    # (t_0, t_max); None -> infer from the trajectory

    def evaluate(self, sim_data, exp_data, show_warnings=True, data_key=None):
        """Score one (sim, exp) experiment by the marginal-time likelihood.

        Mirrors :meth:`SummationObjective.evaluate`'s outer shape -- iterate exp rows, read the
        family/source per column -- but replaces the single-row prediction with the quadrature
        integral over the trajectory's time axis. Returns ``None`` (an unscoreable point) on a
        NaN/inf prediction or an empty integral, matching the base objective's contract.
        """
        indvar = min(exp_data.cols, key=exp_data.cols.get)          # column 0 = independent var (time)
        comparable = set(sim_data.cols) | set(self._per_measurement_models)
        compare_cols = set(exp_data.cols).intersection(comparable)
        if show_warnings:
            self._check_columns(exp_data.cols, compare_cols)
        compare_cols.discard(indvar)

        tau_grid = np.asarray(sim_data[indvar], dtype=float)        # quadrature nodes = the stored grid
        t0, tmax = self._resolve_support(tau_grid)
        pset_values = getattr(self, '_pset_values', {})
        sigma_t = self.sigma_t_source.value(pset_values)

        func_value = 0.0
        for rownum in range(exp_data.data.shape[0]):
            t_k = exp_data.data[rownum, 0]
            for col_name in compare_cols:
                y_bar = exp_data.data[rownum, exp_data.cols[col_name]]
                if np.isnan(y_bar):
                    continue
                noise_model, sources = self._spec_for(col_name)
                sigma, extra = self._resolve_noise_scalar(sources, noise_model, exp_data, rownum, col_name)
                y_traj = np.asarray(sim_data[col_name], dtype=float)    # the whole prediction column
                if np.any(np.isnan(y_traj)) or np.any(np.isinf(y_traj)):
                    return None
                log_z_k = self._log_marginal_contribution(
                    noise_model, y_traj, y_bar, sigma, extra, tau_grid, t_k, sigma_t, t0, tmax)
                if log_z_k is None or not np.isfinite(log_z_k):
                    return None
                func_value += (-log_z_k) * exp_data.weights[rownum, exp_data.cols[col_name]]
        return func_value

    # -- pointwise log-likelihood: log z_k per datum, for LOO/WAIC/IC (ADR-0056, ADR-0112) -----

    def evaluate_pointwise(self, sim_data_dict, exp_data_dict, pset):
        """The per-observation marginal log-likelihoods ``log z_k`` this fit assigns the data
        under ``pset`` -- the pointwise decomposition LOO/WAIC and information_criteria.txt
        consume (ADR-0056, ADR-0112).

        Returns ``(ids, values)``: ``ids`` the stable per-point labels
        (``model/suffix/observable@indvar=value``, the same format the per-point
        :class:`~pybnf.objective.LikelihoodObjective` emits), ``values`` the matching genuine,
        *unweighted* marginal per-observation log-likelihoods
        ``log ∫ p(ȳ_k | y(τ)) p(τ | t_k) dτ`` -- already normalized (the integrand is
        ``NoiseModel.log_density``), the honest density ``az.loo`` needs. Mirrors ``evaluate``'s
        per-evaluation setup (the ``{name: value}`` map a ``FreeParameterSigma`` /
        ``FreeParameterTimeError`` reads, and the measurement-model layer), so the densities are
        scored against exactly the data ``evaluate`` saw. The emitted point set is fixed by the
        *experimental* data (a NaN observation or one outside the family's observation domain is
        skipped -- both data-only conditions), so every recorded draw yields the same ids in the
        same order: the rectangular ``chain x draw x obs`` array the bridge needs."""
        self._pset_values = {p.name: p.value for p in pset}
        ids, values = [], []
        with np.errstate(all='ignore'):
            if self.measurement:
                self.measurement.apply(sim_data_dict, self._pset_values)
            sigma_t = self.sigma_t_source.value(self._pset_values)
            for model in sim_data_dict:
                for suffix in sim_data_dict[model]:
                    if suffix in exp_data_dict[model]:
                        self._pointwise_suffix(sim_data_dict[model][suffix],
                                               exp_data_dict[model][suffix],
                                               '%s/%s' % (model, suffix), sigma_t, ids, values)
        return ids, np.array(values, dtype=float)

    def _pointwise_suffix(self, sim_data, exp_data, prefix, sigma_t, ids, values):
        """Append ``(id, log z_k)`` for every scored point of one model/suffix -- the pointwise
        twin of :meth:`evaluate`'s loop. Columns are walked in sorted order so the obs axis is
        deterministic across draws; a NaN observation or one outside its family's
        ``observation_in_domain`` (#523) is skipped (both data-only). Unlike ``evaluate``, a
        draw-dependent NaN/inf in the *trajectory* is NOT a skip -- it would change the id set
        between draws -- so the point keeps its id and records whatever (possibly non-finite)
        value the integral yields; the recorder / IC consumer already handle a non-finite row."""
        indvar = min(exp_data.cols, key=exp_data.cols.get)
        comparable = set(sim_data.cols) | set(self._per_measurement_models)
        compare_cols = set(exp_data.cols).intersection(comparable)
        compare_cols.discard(indvar)
        tau_grid = np.asarray(sim_data[indvar], dtype=float)
        t0, tmax = self._resolve_support(tau_grid)
        for rownum in range(exp_data.data.shape[0]):
            t_k = exp_data.data[rownum, exp_data.cols[indvar]]
            for col_name in sorted(compare_cols):
                observation = exp_data.data[rownum, exp_data.cols[col_name]]
                if np.isnan(observation):
                    continue
                noise_model, sources = self._spec_for(col_name)
                if not noise_model.observation_in_domain(observation):
                    continue
                sigma, extra = self._resolve_noise_scalar(sources, noise_model, exp_data, rownum, col_name)
                y_traj = np.asarray(sim_data[col_name], dtype=float)
                log_z_k = self._log_marginal_contribution(
                    noise_model, y_traj, observation, sigma, extra, tau_grid, t_k, sigma_t, t0, tmax)
                values.append(log_z_k)
                ids.append('%s/%s@%s=%g' % (prefix, col_name, indvar, t_k))

    # -- the one-dimensional integral (ADR-0112, phase 1) --------------------------------------

    def _log_marginal_contribution(self, noise_model, y_traj, y_bar, sigma, extra,
                                   tau_grid, t_k, sigma_t, t0, tmax):
        """``log z_k = log ∫ exp(log_density(y(τ), ȳ_k, σ)) · p(τ|t_k) dτ`` over the grid.

        The integrand's observation factor is the family's genuine normalized density
        (:meth:`NoiseModel.log_density`); the time factor is the (un-normalized kernel) ×
        (Python normalizer). Everything is accumulated in log space for underflow safety.
        """
        # log p(ȳ_k | y(τ)) at every node -- reuse the family density, vectorized over τ.
        log_obs = np.array([noise_model.log_density(float(y), y_bar, sigma, extra) for y in y_traj])
        kernel = self.time_prior.unnormalized_density(tau_grid, t_k, sigma_t)
        log_norm = self.time_prior.log_normalizer(t_k, sigma_t, t0, tmax)
        with np.errstate(divide='ignore'):
            log_integrand = log_obs + np.log(kernel) + log_norm     # −inf where kernel == 0
        return self._log_trapezoid(tau_grid, log_integrand)

    @staticmethod
    def _log_trapezoid(tau_grid, log_integrand):
        """``log ∫ exp(log_integrand) dτ`` by the trapezoidal rule, evaluated in log space
        (ADR-0112 phase-1 baseline).

        Per interval ``[τ_i, τ_{i+1}]`` the trapezoid contributes
        ``½ Δτ_i (e^{f_i} + e^{f_{i+1}})``; summed via ``logsumexp`` over the log of each
        interval's two-node contribution, so a node whose integrand underflowed (``−inf``,
        e.g. outside a uniform window) drops out cleanly rather than poisoning the sum. Returns
        ``-inf`` when the whole integrand underflowed (an unscoreable point; ``evaluate`` maps it
        to ``None``).
        """
        from scipy.special import logsumexp
        f = np.asarray(log_integrand, dtype=float)
        tau = np.asarray(tau_grid, dtype=float)
        if f.size < 2:
            return float('-inf')
        dtau = np.diff(tau)
        # log( ½ Δτ_i ) + log( e^{f_i} + e^{f_{i+1}} ), guarding a zero-width interval.
        with np.errstate(divide='ignore'):
            log_half_dt = np.log(0.5 * np.abs(dtau))
        pair = np.logaddexp(f[:-1], f[1:])                          # log(e^{f_i}+e^{f_{i+1}})
        terms = log_half_dt + pair
        finite = np.isfinite(terms)
        if not np.any(finite):
            return float('-inf')
        return float(logsumexp(terms[finite]))

    # -- phase-2 seam (a separate ADR) ---------------------------------------------------------

    def gradient_contribution(self, *args, **kwargs):
        """``d(−log z_k)/dθ`` needs ``∂y(τ)/∂θ`` across the whole window -- the sensitivity the
        augmented-ODE engine produces, not the forward trajectory phase 1 has (ADR-0112 "the
        gradient is an integral, not an envelope"). Phase 1 is gradient-free; ``config.py``
        refuses a gradient ``job_type`` under a ``time_error`` clause before a fit starts."""
        raise NotImplementedError(
            'ADR-0112 phase 2: dz_k/dθ via augmented-ODE sensitivities. Phase 1 is gradient-free; '
            'job_type = trf/lbfgs/gntr/hmc is refused at build (see docs/adr/0112-*.md).')

    # -- noise-scalar resolution: share the existing SigmaSource path --------------------------

    def _spec_for(self, col_name):
        """The ``(NoiseModel, {param: SigmaSource})`` for one observable -- its override or the
        class default (shape mirrors :meth:`LikelihoodObjective._spec_for`)."""
        default_sources = (self._default_source_map
                            or ({self.noise.noise_params[0]: self.sigma_source}
                                if self.sigma_source is not None else {}))
        return self.overrides.get(col_name, (self.noise, default_sources))

    def _resolve_noise_scalar(self, sources, noise_model, exp_data, exp_row, col_name):
        """``(primary σ, {secondary: value})`` for one datum from its noise sources + the PSet,
        reusing each :class:`SigmaSource`'s own ``value`` (ADR-0112: share, do not reimplement).

        A τ-independent source (``fix_at`` / ``fit`` / ``read_exp_file``) yields one scalar over
        the whole window; the base ``evaluate_multiple`` already set ``self._pset_values`` for a
        ``FreeParameterSigma``, and a ``DataColumnSigma`` reads the datum's own ``_SD`` cell. A
        prediction-dependent source is refused at build, so it never reaches here; the guard
        below is the defensive backstop.
        """
        names = noise_model.noise_params
        resolved = {}
        for name in names:
            src = sources[name]
            if type(src).__name__ not in _TAU_INDEPENDENT_SIGMA:
                raise PybnfError(
                    f'time_error: noise scale "{name}" is prediction-dependent',
                    'A time_error marginalization needs a noise scale that does not vary with the '
                    'latent measurement time (fix_at / fit / read_exp_file). A relative / formula / '
                    'prediction_formula σ is deferred to a follow-up (ADR-0112).')
            # sim_data / sim_row are unused by every τ-independent source; pass None.
            resolved[name] = src.value(self, None, None, exp_data, exp_row, col_name)
        primary = resolved[names[0]]
        extra = {n: resolved[n] for n in names[1:]} or None
        return primary, extra

    def _resolve_support(self, tau_grid):
        """``(t_0, t_max)`` -- the declared support, else the trajectory's own span. The base
        model must be simulated over ``[t_0, t_max]`` (ADR-0112 "the grid is explicit")."""
        if self.support is not None:
            return self.support
        return float(np.min(tau_grid)), float(np.max(tau_grid))


# ============================================================================================
# The builders  -- one parsed clause -> the time spec, and the objective swap
# ============================================================================================

def build_time_error_spec(time_error_token, sigma_t_field):
    """``('truncated_normal', ('fit', 'st__FREE'))`` -> ``(TimeErrorPrior, TimeErrorSource)``.

    Parallels :func:`pybnf.objective._build_noise_spec` for the time half; the ploop layer has
    already enforced both-or-neither, so both arguments are present here.
    """
    token = time_error_token.lower()
    if token not in _TIME_ERROR_PRIORS:
        raise PybnfError(
            f'Unknown time_error family "{token}"',
            f'Valid time_error families are: {", ".join(sorted(_TIME_ERROR_PRIORS))}.')
    verb, arg = sigma_t_field
    return _TIME_ERROR_PRIORS[token](), _build_time_error_source(verb, arg)


def build_time_error_objective(base_obj, time_prior, sigma_t_source, support=None):
    """Swap a built per-point :class:`~pybnf.objective.LikelihoodObjective` for a
    :class:`MarginalizedTimeObjective` carrying the same noise spec plus the time prior/source
    (ADR-0112). Returns ``base_obj`` unchanged when ``σ_t`` is a fixed 0 -- the standard
    likelihood is the ``σ_t → 0`` limit, so the two are byte-identical there.

    The caller (``config.py._load_obj_func``) has already checked that ``base_obj`` is a
    per-point likelihood; the time half rides on top of its ``noise`` / ``sigma_source`` /
    ``overrides`` without disturbing them.
    """
    if isinstance(sigma_t_source, FixedTimeError) and sigma_t_source.is_zero:
        return base_obj
    obj = MarginalizedTimeObjective(
        ind_var_rounding=base_obj.rounding,
        overrides=getattr(base_obj, 'overrides', None),
        noise=getattr(base_obj, 'noise', None),
        sigma_source=getattr(base_obj, 'sigma_source', None),
        sigma_sources=getattr(base_obj, '_default_source_map', None),
        time_prior=time_prior,
        sigma_t_source=sigma_t_source,
        support=support)
    # Carry over the observation-layer / structural attributes config sets on any objective
    # (ADR-0036 measurement layer, ADR-0045 per-measurement models, ADR-0051 cumulative).
    for attr in ('measurement', '_per_measurement_models', '_cumulative_cols',
                 '_analytic_scale', 'constraints'):
        if hasattr(base_obj, attr):
            setattr(obj, attr, getattr(base_obj, attr))
    return obj
