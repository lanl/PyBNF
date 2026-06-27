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

It samples in sampling space ``u`` (ADR-0010), with target

    log pi(u) = sum_i prior_i.logpdf_jax(u_i)  +  ( -NLL( scale.inverse(u) ) )

and NO change-of-variables Jacobian -- the prior is *defined* in ``u``, so this is exactly
the density ``am`` samples, now differentiated w.r.t. ``u`` (keeping HMC and the
gradient-free samplers comparable on the *same* posterior).

It samples the closed-form-truth and stress-geometry menu targets (``gaussian`` /
``rotated_gaussian`` / ``banana`` / ``multimodal``) and the **full edition-2 prior catalog**
-- every family now supplies a hand-written, scipy-``logpdf``-oracle-checked
``logpdf_jax`` (ADR-0059 item 4) -- on the linear scale. The work still deferred to later
slices raises a pointed error rather than sampling a silently-wrong target: the sympy->jax
expression path (BYO ``expression`` targets) and ``rotated_quartic``, the log-scale and
constrained-support unconstraining bijections (item 5 -- the positive/bounded-support and
truncated priors evaluate as correct *densities* now, but HMC quality *at* a hard support
edge awaits the bijection, and the divergence/R-hat gate flags it honestly until then).

``jax``/``blackjax`` are the optional ``pybnf[jax]`` extra (ADR-0019): only this module (and
the lazily-imported ``nll_jax`` / ``logpdf_jax``) touches them, and a missing install
surfaces as a pointed :class:`PybnfError` naming the extra, never a bare ``ImportError``.
"""

import logging

import numpy as np

from .base import BayesianAlgorithm, MCMCFamilyConfig
from ...analytical_model import AnalyticalModel
from ...printing import print0, print1, print2, PybnfError
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
        """The single :class:`AnalyticalModel` this run samples, or a pointed error.

        HMC needs a gradient, and only the analytical / BYO log-density model exposes one
        (``nll_jax``). A simulator model (BNGL/SBML), or more than one model, gets a clear
        error pointing at the ADR rather than a cryptic ``AttributeError`` later."""
        models = list(self.config.models.values())
        analytical = [m for m in models if isinstance(m, AnalyticalModel)]
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

    def _build_logdensity(self, jnp, nll_fn, coord_perm):
        """Compose the JAX target ``log pi(u)`` HMC samples (ADR-0059).

        ``log pi(u) = sum_i prior_i.logpdf_jax(u_i) + (-NLL(scale.inverse(u)))``. The prior
        is defined in ``u`` (ADR-0010), so there is no change-of-variables Jacobian -- this
        is exactly the density the gradient-free samplers target, now differentiable.

        ``coord_perm`` reorders ``u`` (``self.variables`` order) into the target's coordinate
        order before the NLL, so HMC binds parameters to coordinates by name exactly as the
        score path does (:meth:`_coordinate_permutation`). The prior sum stays in
        ``self.variables`` order (each ``prior_i`` already pairs with ``u_i`` by that order).

        This slice supports only the linear scale (``scale.inverse`` is the identity, so
        ``theta = u``); a log-scaled parameter would need a JAX ``10**u`` / ``exp(u)`` and
        its Jacobian, which is deferred, so it raises here. The per-family JAX prior gap is
        caught by ``Prior.logpdf_jax`` (it raises for an unsupported family), surfaced
        eagerly by the probe in :meth:`run`."""
        log_scaled = [v.name for v in self.variables if v.log_space]
        if log_scaled:
            raise PybnfError(
                "job_type = hmc supports only linearly-scaled parameters in this slice "
                "(ADR-0059); parameter(s) %s are log-scaled. A log scale needs a "
                "JAX-traceable 10**u / exp(u) inverse and its Jacobian, which is a later "
                "slice. Declare them with a linear prior (uniform_var / normal_var) or run "
                "a gradient-free sampler." % log_scaled)
        priors = [self.prior.get(v.name) for v in self.variables]
        perm = jnp.asarray(coord_perm)

        def logdensity_fn(u):
            # Linear scale: theta == u. Prior contributions sum in u-space (self.variables
            # order); the model's NLL is evaluated in theta-space -- reordered to the target's
            # coordinate order by `perm` (bind-by-name) -- and negated into the log-density.
            lp = jnp.asarray(0.0)
            for i, var in enumerate(priors):
                if var is not None:
                    lp = lp + var.prior_logpdf_jax(u[i])
            return lp - nll_fn(u[perm])

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
        logdensity_fn = self._build_logdensity(jnp, nll_fn, coord_perm)

        print2('Running Hamiltonian Monte Carlo (blackjax NUTS) on %i independent chain(s): '
               '%i warmup + %i draws each, target acceptance %.2f.'
               % (self.num_parallel, self.num_warmup, self.num_samples, self.target_accept))

        # start_run sets up Results/ + the samples.txt header and returns one starting pset
        # per chain (latin-hypercube / random, per `initialization`); reuse its u-vector as
        # the NUTS initial position so HMC and the gradient-free samplers seed the same way.
        first_psets = self.start_run(setup_samples=True)
        init_us = [self._param_vec(p) for p in first_psets]

        # Eager probe at the first chain's start so an unsupported prior family raises its
        # pointed PybnfError here, before the (slower) warmup, rather than from inside
        # blackjax. logpdf_jax is the only place an unsupported family can surface.
        try:
            float(logdensity_fn(jnp.asarray(init_us[0])))
        except PybnfError:
            raise
        except Exception:
            logger.debug('HMC log-density probe failed for a non-prior reason; '
                         'continuing to the warmup, which will surface it.', exc_info=True)

        self.divergences = []
        for c in range(self.num_parallel):
            positions, logdens, n_divergent = self._run_one_chain(
                jax, jnp, blackjax, logdensity_fn, init_us[c], c)
            self.divergences.append(n_divergent)
            for d in range(self.num_samples):
                u = np.asarray(positions[d], dtype=float)
                pset = self._pset_from_u(u, name='iter%irun%i' % (d, c))
                self.sample_pset(pset, float(logdens[d]), chain_index=c)
                # Diagnostics operate in sampling space u (split-R-hat / ESS); record the
                # raw NUTS draw, not the reflected pset, so they see the true chain.
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

    def _run_one_chain(self, jax, jnp, blackjax, logdensity_fn, init_u, chain_index):
        """Window-adapt then sample one NUTS chain; return ``(positions, logdensities,
        n_divergent)``.

        The chain's JAX PRNG key is seeded from this chain's own ``np.random.Generator``
        (itself spawned from the run's resolved ``random_seed``), so the whole run is
        reproducible from the config seed -- the same guarantee the gradient-free samplers
        give. Warmup is blackjax window adaptation (dual-averaging step size + diagonal mass
        matrix); the post-warmup draws are collected with ``jax.lax.scan`` over the tuned
        kernel. The scan also carries out each step's ``info.is_divergent`` flag, summed into
        the chain's post-warmup divergent-transition count (the NUTS reliability signal)."""
        seed = int(self.chain_rngs[chain_index].integers(0, 2 ** 31 - 1))
        warmup_key, sample_key = jax.random.split(jax.random.PRNGKey(seed))
        init_position = jnp.asarray(init_u)

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
