"""HMCSampler -- the ``hmc`` fit type: blackjax NUTS on the analytical model's JAX
log-density (issue #425, ADR-0059).

This is an opt-in **reference** sampler: a gradient-based NUTS run that exists to
*evaluate* PyBNF's gradient-free samplers (``am`` / ``dream`` / ``p_dream``) against a
research-grade yardstick on the canonical stress geometries. It runs **only** on the
analytical / bring-your-own-log-density model (``pybnf/analytical_model.py``), never on a
simulator -- a BNGL/SBML posterior provides no cheap gradient, and simulator-path HMC is
rejected outright by the ADR.

Unlike every other sampler, ``hmc`` does **not** dispatch parameter sets through the
dask -> ``execute()`` -> ``score``-column -> ``DirectPassObjective`` loop: the gradient
cannot survive the per-pset dask round-trip. Instead it builds a JAX ``logdensity_fn`` in
process from (a) the model's JAX NLL (``AnalyticalModel.nll_jax``) and (b) the priors' JAX
log-densities (``Prior.logpdf_jax``), hands it to blackjax NUTS with window adaptation, and
writes the draws in the **standard samples format** so the ArviZ bridge (ADR-0055), the
LOO/WAIC sidecar machinery, and the rank-normalized split-R-hat / bulk-tail ESS diagnostics
all work unchanged.

It samples an **unconstrained** ``z in R^d`` (ADR-0059 item 5), mapped to sampling space ``u``
(ADR-0010) by a per-parameter support-aware bijection ``u = b(z)`` (:mod:`pybnf.priors.bijector`),
with target

    log pi(z) = sum_i [ prior_i.logpdf_jax(b_i(z_i)) + log|b_i'(z_i)| ]  +  ( -NLL( scale.inverse(b(z)) ) )

The prior is *defined* in ``u`` (ADR-0010), so there is no ``theta <-> u`` Jacobian -- the only
change-of-variables term is the unconstraining bijection's ``log|b'(z)|``. For an unbounded prior
``b`` is the identity (``z == u``) and this is exactly the density ``am`` samples, now
differentiated; for a positive/bounded/truncated prior ``b`` is a log/logit reparameterization that
lands strictly inside the open support for every finite ``z``, so the ``-inf`` support wall (and the
NUTS divergence it once caused) is unreachable. Diagnostics and the samples file report ``u`` (the
recorded ``Ln_probability`` is un-Jacobianed back to ``log pi(u)``), so HMC stays comparable to the
gradient-free samplers on the *same* posterior coordinate.

It samples the closed-form-truth and stress-geometry menu targets (``gaussian`` /
``rotated_gaussian`` / ``banana`` / ``multimodal``), the **full edition-2 prior catalog** -- every
family supplies a hand-written, scipy-``logpdf``-oracle-checked ``logpdf_jax`` (ADR-0059 item 4),
sampled divergence-free on *any* support via the bijection (item 5) -- and **log-scaled parameters**
(``lognormal_var`` / ``loguniform_var``), whose ``u -> theta`` inverse traces through JAX
(``Scale.inverse_jax``). It samples every built-in menu target (``rotated_quartic`` included now)
**and bring-your-own ``expression`` targets** -- a user's own PEtab-math NLL lambdified to JAX
(``ExpressionModel.nll_jax``, ADR-0059 item 2), so HMC is a reference sampler on arbitrary
closed-form log-densities, not just the canned menu. The only HMC work still deferred is the Python
``callable = module:func`` BYO form (a general callable is not JAX-traceable; ADR-0050).

``jax``/``blackjax`` are the optional ``pybnf[jax]`` extra (ADR-0019): only this module (and
the lazily-imported ``nll_jax`` / ``logpdf_jax``) touches them, and a missing install
surfaces as a pointed :class:`PybnfError` naming the extra, never a bare ``ImportError``.
"""

import logging

import numpy as np

from .base import BayesianAlgorithm, MCMCFamilyConfig
from ...analytical_model import AnalyticalModel, ExpressionModel
from ...printing import print0, print1, print2, PybnfError
from ...priors import bijector_for_support
from ...registry import register_fit_type

# Preserve the shared sampler logging channel.
logger = logging.getLogger('pybnf.algorithms')


def _require_jax():
    """Import ``jax`` / ``jax.numpy`` / ``blackjax`` lazily, or raise a pointed error.

    ``jax``/``blackjax`` are the optional ``pybnf[jax]`` extra (ADR-0059): only the HMC
    path imports them. A missing install surfaces as a :class:`PybnfError` naming the
    extra -- the house pattern, mirroring ``pybnf/petab/formula.py::_require_petab_math``
    -- never a bare ``ImportError`` from deep in the sampler. Returns
    ``(jax, jax.numpy, blackjax)``."""
    try:
        import jax
        import jax.numpy as jnp
        import blackjax
    except ImportError as e:
        raise PybnfError(
            "job_type = hmc needs the gradient-based sampler stack (jax + blackjax), "
            "which is the optional 'jax' extra. Install it with `pip install pybnf[jax]` "
            "(or `uv pip install pybnf[jax]`). HMC is a reference sampler on the "
            "analytical model only (ADR-0059); the gradient-free samplers (am / dream / "
            "p_dream) need no extra."
        ) from e
    return jax, jnp, blackjax


class HMCConfig(MCMCFamilyConfig):
    """Config for the ``hmc`` sampler, co-located with the method (ADR-0006).

    Reuses the global ``population_size`` as the number of independent NUTS chains (so the
    samples-file chain naming ``iter<draw>run<chain>`` and the per-chain diagnostics carry
    over unchanged) and adds the three NUTS knobs on top of the shared MCMC family fields.
    Window adaptation replaces the gradient-free ``burn_in``/``sample_every`` thinning, so
    those inherited keys are unused by ``hmc`` (NUTS draws are near-independent -- every
    post-warmup draw is kept).
    """

    #: Post-warmup draws kept per chain (each becomes one samples.txt row).
    num_samples: int = 1000
    #: Window-adaptation (warmup) steps per chain -- dual-averaging step size + mass matrix.
    num_warmup: int = 1000
    #: NUTS dual-averaging target acceptance probability (Stan-like default).
    target_accept: float = 0.8


@register_fit_type('hmc', family='sampler',
                   display_name='Hamiltonian Monte Carlo (NUTS via blackjax)',
                   schema=HMCConfig)
class HMCSampler(BayesianAlgorithm):
    """blackjax NUTS on the analytical model's JAX log-density (ADR-0059).

    Subclasses :class:`BayesianAlgorithm` to reuse prior loading, the samples-file setup
    and writer (``start_run`` / ``sample_pset``), and the R-hat/ESS diagnostics; overrides
    :meth:`run` to drive NUTS in process instead of the dask score-column loop.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_samples = config.config['num_samples']
        self.num_warmup = config.config['num_warmup']
        self.target_accept = config.config['target_accept']
        #: Post-warmup divergent-transition count per chain (filled in by run()).
        #: A NUTS-specific reliability signal the gradient-free samplers have no
        #: analogue for: divergences flag regions the leapfrog integrator cannot
        #: traverse (a too-sharp curvature for the tuned step size), so a nonzero
        #: count -- like a high split-R-hat -- means HMC's *own* draws are not yet
        #: trustworthy on this geometry (ADR-0059 gates the reference on its own
        #: diagnostics: split-R-hat / ESS / divergences).
        self.divergences = []

    # ------------------------------------------------------------------ #
    # Building the JAX target log-density
    # ------------------------------------------------------------------ #
    def _resolve_analytical_model(self):
        """The single analytical / BYO-expression model this run samples, or a pointed error.

        HMC needs a gradient, and only the JAX-capable log-density models expose one (``nll_jax``):
        the built-in menu :class:`AnalyticalModel` and the bring-your-own :class:`ExpressionModel`
        (ADR-0059 item 2). A simulator model (BNGL/SBML), or more than one model, gets a clear error
        pointing at the ADR rather than a cryptic ``AttributeError`` later."""
        models = list(self.config.models.values())
        analytical = [m for m in models if isinstance(m, (AnalyticalModel, ExpressionModel))]
        if not analytical:
            raise PybnfError(
                "job_type = hmc requires an analytical/expression objective; the model(s) "
                "%s provide no usable gradient (a simulator posterior is differentiated "
                "only through a stiff solve, which the ADR rejects -- see ADR-0059). Use a "
                "gradient-free sampler (am / dream / p_dream) on a simulator model."
                % [getattr(m, 'name', '?') for m in models])
        if len(analytical) > 1:
            raise PybnfError(
                "job_type = hmc supports exactly one analytical target, but %d were "
                "declared (%s). Multi-model HMC is not part of this slice (ADR-0059)."
                % (len(analytical), [m.name for m in analytical]))
        return analytical[0]

    def _coordinate_permutation(self, model):
        """Map ``self.variables`` (declaration) order -> the target's coordinate order
        (bind-by-name, ADR-0034).

        ``nll_jax`` consumes ``theta`` in the target's coordinate order (``mean[0]``,
        ``mean[1]``, ...), while the sampler builds ``u`` in ``self.variables`` order. This
        permutation reorders ``u`` to coordinate order using the **same** by-name rule as the
        numpy score path (:meth:`AnalyticalModel.coordinate_order`), so HMC and the
        gradient-free samplers bind a parameter to the same coordinate -- not by the
        ``p1..pN`` accident of declaration order happening to equal sort order. Identity when
        the parameters are declared in coordinate order (the common case), so no behavior
        change there; correct (not silently wrong) when they are not."""
        var_names = [v.name for v in self.variables]
        coord_names = model.coordinate_order(var_names)
        return [var_names.index(cn) for cn in coord_names]

    def _build_logdensity(self, jnp, nll_fn, coord_perm, bijectors):
        """Compose the JAX target ``log pi(z)`` HMC samples in the unconstrained space
        (ADR-0059 items 3 + 5).

        ``log pi(z) = sum_i [prior_i.logpdf_jax(u_i) + log|b_i'(z_i)|] + (-NLL(scale.inverse(u)))``
        with ``u_i = b_i(z_i)``. Three transforms compose, per parameter:

        * ``u_i = bijectors[i].to_constrained_jax(z_i)`` -- the support-aware unconstraining
          bijection (identity for an unbounded prior; log/logit for a positive/bounded/truncated
          one), so NUTS samples an unbounded ``z`` and the ``-inf`` support wall is unreachable.
          Its Jacobian ``log|b'(z)|`` is the *only* change-of-variables term (the prior is defined
          in ``u``, ADR-0010, so there is no ``theta <-> u`` Jacobian).
        * ``prior_i.logpdf_jax(u_i)`` -- the family log-density, in ``u`` where it is defined.
        * ``theta_i = v.from_sampling_space_jax(u_i)`` -- the ``u -> theta`` scale inverse
          (identity / ``10**u`` / ``exp(u)``), so a log-scaled parameter composes; ``theta`` enters
          only through the likelihood.

        ``coord_perm`` reorders ``theta`` (``self.variables`` order) into the target's coordinate
        order before the NLL, so HMC binds parameters to coordinates by name exactly as the score
        path does (:meth:`_coordinate_permutation`); the prior + Jacobian sums stay in
        ``self.variables`` order (each pairs with ``z_i`` by that order). The per-family JAX prior
        gap is caught by ``Prior.logpdf_jax`` (it raises for an unsupported family), surfaced eagerly
        by the probe in :meth:`run`."""
        perm = jnp.asarray(coord_perm)

        def logdensity_fn(z):
            lp = jnp.asarray(0.0)
            thetas = []
            for i, v in enumerate(self.variables):
                b = bijectors[i]
                u_i = b.to_constrained_jax(z[i])
                lp = lp + b.logdet_jax(z[i]) + v.prior_logpdf_jax(u_i)
                thetas.append(v.from_sampling_space_jax(u_i))
            theta = jnp.stack(thetas)
            return lp - nll_fn(theta[perm])

        return logdensity_fn

    # ------------------------------------------------------------------ #
    # Driving NUTS (bypasses the dask score-column loop)
    # ------------------------------------------------------------------ #
    def run(self, client=None, resume=None, debug=False):
        """Run blackjax NUTS in process and write draws in the standard samples format.

        ``client`` (the dask client the harness passes every sampler) is intentionally
        ignored: a single analytical NUTS chain is a tight numeric loop and the chains run
        as independent blackjax runs, so there is no per-pset dispatch (ADR-0059 item 1)."""
        jax, jnp, blackjax = _require_jax()

        model = self._resolve_analytical_model()
        nll_fn = model.nll_jax()                       # f(theta) -> NLL (pointed error if unsupported target)
        coord_perm = self._coordinate_permutation(model)
        # One support-aware unconstraining bijector per parameter (ADR-0059 item 5): keyed on
        # the prior's u-space support, so a positive/bounded/truncated prior is reparameterized
        # onto the unbounded space NUTS samples and the -inf support wall is unreachable.
        bijectors = [bijector_for_support(*v.prior_support()) for v in self.variables]
        logdensity_fn = self._build_logdensity(jnp, nll_fn, coord_perm, bijectors)

        print2('Running Hamiltonian Monte Carlo (blackjax NUTS) on %i independent chain(s): '
               '%i warmup + %i draws each, target acceptance %.2f.'
               % (self.num_parallel, self.num_warmup, self.num_samples, self.target_accept))

        # start_run sets up Results/ + the samples.txt header and returns one starting pset
        # per chain (latin-hypercube / random, per `initialization`); reuse its u-vector as the
        # seed, mapped through each bijector to the unconstrained space z NUTS actually samples
        # (so HMC and the gradient-free samplers seed from the same start point).
        first_psets = self.start_run(setup_samples=True)
        init_zs = [np.array([bijectors[i].to_unconstrained(u_i) for i, u_i in enumerate(self._param_vec(p))])
                   for p in first_psets]

        # Eager probe at the first chain's start so an unsupported prior family raises its
        # pointed PybnfError here, before the (slower) warmup, rather than from inside
        # blackjax. logpdf_jax is the only place an unsupported family can surface.
        try:
            float(logdensity_fn(jnp.asarray(init_zs[0])))
        except PybnfError:
            raise
        except Exception:
            logger.debug('HMC log-density probe failed for a non-prior reason; '
                         'continuing to the warmup, which will surface it.', exc_info=True)

        self.divergences = []
        for c in range(self.num_parallel):
            positions, logdens, n_divergent = self._run_one_chain(
                jax, jnp, blackjax, logdensity_fn, init_zs[c], c)
            self.divergences.append(n_divergent)
            for d in range(self.num_samples):
                z = np.asarray(positions[d], dtype=float)
                # Map the unconstrained NUTS draw z back to sampling space u (b(z)), and
                # un-Jacobian the recorded log-density to log pi(u) -- the samples file and the
                # diagnostics report u (the gradient-free samplers' coordinate), not z, so the
                # Ln_probability column and R-hat/ESS stay comparable across samplers. (R-hat and
                # bulk/tail ESS are rank-normalized, hence invariant to the monotone z<->u map, so
                # u-space diagnostics also remain the honest measure of NUTS mixing.)
                u = np.array([bijectors[i].to_constrained(z[i]) for i in range(len(self.variables))])
                logdet = sum(bijectors[i].logdet(z[i]) for i in range(len(self.variables)))
                pset = self._pset_from_u(u, name='iter%irun%i' % (d, c))
                self.sample_pset(pset, float(logdens[d]) - logdet, chain_index=c)
                self.chain_history[c].append(u)
            print1('Completed HMC chain %i of %i (%i draws, %i divergent transitions)'
                   % (c + 1, self.num_parallel, self.num_samples, n_divergent))

        # The same diagnostics the gradient-free samplers report, on the NUTS draws.
        self.report_convergence_diagnostics(self.num_samples)
        # Divergences are the one HMC-specific diagnostic the shared report has no slot
        # for; surface the total so a curvature the integrator could not traverse is not
        # silently folded into a clean-looking sample (ADR-0059's reliability gate).
        total_divergent = int(sum(self.divergences))
        if total_divergent:
            print0('HMC saw %i divergent transition(s) across %i chains -- the NUTS draws '
                   'on this geometry are NOT a trustworthy reference (raise num_warmup / '
                   'target_accept, or the target is too sharply curved for this step size).'
                   % (total_divergent, self.num_parallel))
        else:
            print2('HMC saw no divergent transitions.')
        self.update_histograms('_final')
        self._emit_inference_data()
        print0('HMC sampling complete: %i chains x %i draws written to %s'
               % (self.num_parallel, self.num_samples, self.samples_file))

    def _run_one_chain(self, jax, jnp, blackjax, logdensity_fn, init_z, chain_index):
        """Window-adapt then sample one NUTS chain; return ``(positions, logdensities,
        n_divergent)`` in the unconstrained space ``z`` (ADR-0059 item 5).

        The chain's JAX PRNG key is seeded from this chain's own ``np.random.Generator``
        (itself spawned from the run's resolved ``random_seed``), so the whole run is
        reproducible from the config seed -- the same guarantee the gradient-free samplers
        give. Warmup is blackjax window adaptation (dual-averaging step size + diagonal mass
        matrix); the post-warmup draws are collected with ``jax.lax.scan`` over the tuned
        kernel. The scan also carries out each step's ``info.is_divergent`` flag, summed into
        the chain's post-warmup divergent-transition count (the NUTS reliability signal).
        ``run`` maps the returned ``z`` positions back to ``u = b(z)`` before writing."""
        seed = int(self.chain_rngs[chain_index].integers(0, 2 ** 31 - 1))
        warmup_key, sample_key = jax.random.split(jax.random.PRNGKey(seed))
        init_position = jnp.asarray(init_z)

        warmup = blackjax.window_adaptation(
            blackjax.nuts, logdensity_fn, target_acceptance_rate=self.target_accept)
        (last_state, parameters), _ = warmup.run(
            warmup_key, init_position, num_steps=self.num_warmup)

        kernel = blackjax.nuts(logdensity_fn, **parameters)

        def one_step(state, key):
            state, info = kernel.step(key, state)
            return state, (state.position, state.logdensity, info.is_divergent)

        keys = jax.random.split(sample_key, self.num_samples)
        _, (positions, logdens, is_divergent) = jax.lax.scan(one_step, last_state, keys)
        return np.asarray(positions), np.asarray(logdens), int(np.asarray(is_divergent).sum())
