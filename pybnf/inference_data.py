"""The ArviZ ``InferenceData`` bridge (ADR-0055, issue #438 item 3).

``from_pybnf`` maps a finished PyBNF MCMC run's saved samples onto an
:class:`arviz.InferenceData`, so PyBNF's posterior output becomes first-class in
the ArviZ / bayesplot / loo ecosystem (trace / rank / forest / pair plots,
``az.summary``, ``az.compare``). It is a *format bridge*, not new statistics:
PyBNF already runs the samplers, writes ``Results/samples.txt``, and computes
rank-normalized split-R-hat + bulk/tail ESS (``pybnf.diagnostics``, ADR-0009).

Design (ADR-0055), in brief:

- **Source = the saved sample on disk.** ``Results/samples.txt`` is the *thinned,
  post-burn-in* posterior sample (written every ``sample_every`` iterations after
  burn-in) -- the same draws ``credible*.txt`` and the histograms are built from.
  It is **not** the dense in-memory chain ``diagnostics.txt`` is computed on, so
  ArviZ recomputing diagnostics here gives valid numbers on *fewer* draws: R-hat
  is comparable, ``az.ess`` reads lower than ``diagnostics.txt`` by design. A user
  wanting denser ArviZ diagnostics lowers ``sample_every``. PyBNF's own final
  R-hat/ESS are copied into the object's ``attrs`` so nothing is lost.
- **Posterior in sampling space.** A log-scaled parameter is emitted in its
  sampling space (``log10`` / ``ln``), the space the sampler moved in and the space
  ``diagnostics.py`` already uses -- so ArviZ's diagnostics share PyBNF's
  parameterization and Vehtari method. The variable is named ``<scale>_<name>``
  (e.g. ``log10_k``) so the space is explicit. Linear parameters are unchanged.
  Recovering each parameter's scale needs the run's free parameters: the auto-emit
  path passes the live ``variables``; the standalone path auto-discovers the
  ``.conf`` copied into ``Results/`` and falls back to natural-space-with-a-warning
  if it cannot.
- **Groups: ``posterior`` + ``sample_stats`` (``lp``).** ``log_likelihood`` (which
  unlocks ``az.loo`` / ``az.waic``, #438 item 4), ``prior``, and ``observed_data``
  are deferred to the follow-on that rides on this bridge.

``arviz`` is an optional extra (``pip install pybnf[arviz]``), imported lazily so
core stays dependency-free (ADR-0019). Both arviz major lines are supported -- the
classic 0.x ``InferenceData`` and the 1.x xarray-``DataTree`` rewrite -- since they
differ only in the one ``from_dict`` construction call (see ``_build_idata``); the
extra is uncapped so installing the bridge never downgrades a user's arviz.
"""

import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger('pybnf.inference_data')

# Parsed out of a PSet name of the form ``iter<draw>run<chain>`` (the population
# samplers' per-chain naming). ``run`` is the chain; ``iter`` is the iteration the
# draw was recorded at -- a stride of ``sample_every``, mapped to a contiguous draw
# index by ordering within the chain.
_RUN_RE = re.compile(r'(?<=run)\d+')
_ITER_RE = re.compile(r'(?<=iter)\d+')


def _require_arviz():
    """Import arviz lazily, with a clear install hint (never a hard import)."""
    try:
        import arviz as az
    except ImportError as e:
        raise ImportError(
            "The ArviZ InferenceData bridge needs the optional 'arviz' extra. "
            "Install it with:  pip install pybnf[arviz]"
        ) from e
    return az


def _resolve_samples_path(source):
    """Resolve ``source`` to ``(samples_path, results_dir)``.

    ``source`` may be a ``samples.txt`` file, a ``Results/`` directory, or an
    output directory containing ``Results/samples.txt``.
    """
    p = Path(source)
    if p.is_file():
        return p, p.parent
    if p.is_dir():
        if (p / 'samples.txt').is_file():
            return p / 'samples.txt', p
        if (p / 'Results' / 'samples.txt').is_file():
            return p / 'Results' / 'samples.txt', p / 'Results'
    raise FileNotFoundError(
        "Could not find a samples.txt under %r. Pass a Results/ directory, an "
        "output directory, or a samples.txt file from a finished MCMC run." % str(source))


def _parse_samples(samples_path):
    """Parse ``samples.txt`` into ``(param_names, chains)``.

    ``param_names`` is the header parameter order; ``chains`` maps a chain index to
    a list of ``(iter, lp, values)`` rows (values aligned to ``param_names``).
    """
    with open(samples_path) as f:
        header = f.readline()
        if not header.startswith('#'):
            raise ValueError("Malformed samples file %s: missing '# Name ...' header." % samples_path)
        # '# Name\tLn_probability\t<param names...>'
        cols = header.lstrip('#').strip().split('\t')
        if len(cols) < 3:
            raise ValueError("samples file %s has no parameter columns." % samples_path)
        param_names = cols[2:]

        chains = {}
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            fields = line.split('\t')
            if len(fields) != len(param_names) + 2:
                logger.warning('Skipping malformed samples row (%d fields, expected %d): %r',
                               len(fields), len(param_names) + 2, line[:80])
                continue
            name = fields[0]
            run_m = _RUN_RE.search(name)
            iter_m = _ITER_RE.search(name)
            if run_m is None or iter_m is None:
                logger.warning("Skipping samples row with unparseable name %r "
                               "(expected 'iter<draw>run<chain>').", name)
                continue
            chain = int(run_m.group(0))
            draw_iter = int(iter_m.group(0))
            lp = float(fields[1])
            values = [float(x) for x in fields[2:]]
            chains.setdefault(chain, []).append((draw_iter, lp, values))

    if not chains:
        raise ValueError("No samples found in %s (header only?). A run that wrote no "
                         "post-burn-in samples produces no InferenceData." % samples_path)
    return param_names, chains


def _resolve_variables(results_dir, config, variables):
    """Return a ``name -> FreeParameter`` map for sampling-space transforms, or
    ``None`` (with a warning) when the scale cannot be recovered.

    Precedence: explicit ``variables`` (the auto-emit path, always correct) >
    explicit ``config`` > a ``.conf`` auto-discovered in ``results_dir`` (the
    standalone path; the original conf is copied into ``Results/`` at run start).
    Any failure to reconstruct degrades to natural space rather than raising -- an
    archived run whose model files have moved still produces a usable object.
    """
    if variables is not None:
        return {v.name: v for v in variables}

    cfg = config
    if cfg is None:
        confs = sorted(Path(results_dir).glob('*.conf'))
        if not confs:
            logger.warning(
                'No .conf found in %s to recover parameter scales; log-scaled '
                'parameters will be emitted in NATURAL space (ArviZ diagnostics '
                'then differ from PyBNF\'s log-space diagnostics.txt). Pass '
                'config=... or variables=... for sampling-space output.', results_dir)
            return None
        cfg = confs[0]

    try:
        from .config import Configuration
        from .parse import load_config
        if not isinstance(cfg, Configuration):
            cfg = load_config(str(cfg))
        return {v.name: v for v in cfg.variables}
    except Exception:
        logger.warning(
            'Could not load a config to recover parameter scales (model files '
            'missing or unreadable?); log-scaled parameters will be emitted in '
            'NATURAL space.', exc_info=True)
        return None


def _posterior_var_name(name, var):
    """The InferenceData variable name: ``<scale>_<name>`` for a log parameter
    (e.g. ``log10_k`` / ``ln_k``), the bare name otherwise -- mirroring the
    histogram-edge labelling convention (samplers/base.py)."""
    if var is not None and getattr(var, 'log_space', False):
        return '%s_%s' % (var.scale_name, name)
    return name


def _read_diagnostics_attrs(results_dir):
    """Best-effort summary of PyBNF's own final R-hat/ESS from ``diagnostics.txt``,
    for the InferenceData ``attrs`` -- so the object carries PyBNF's authoritative
    (dense, log-space) convergence numbers alongside whatever ArviZ recomputes on
    the thinned saved sample. Returns ``{}`` when the file is absent/unreadable."""
    diag = Path(results_dir) / 'diagnostics.txt'
    if not diag.is_file():
        return {}
    try:
        with open(diag) as f:
            lines = [ln for ln in f if ln.strip()]
        if len(lines) < 2:
            return {}
        cols = lines[0].lstrip('#').strip().split('\t')
        last = lines[-1].strip().split('\t')
        row = dict(zip(cols, last))
        rhats = [float(v) for k, v in row.items() if k.startswith('rhat_') and v != 'nan']
        bulk = [float(v) for k, v in row.items() if k.startswith('bulk_ess_') and v != 'nan']
        tail = [float(v) for k, v in row.items() if k.startswith('tail_ess_') and v != 'nan']
        attrs = {}
        if rhats:
            attrs['pybnf_max_rhat'] = max(rhats)
        if bulk:
            attrs['pybnf_min_bulk_ess'] = min(bulk)
        if tail:
            attrs['pybnf_min_tail_ess'] = min(tail)
        if 'iteration' in row:
            attrs['pybnf_diagnostics_iteration'] = float(row['iteration'])
        return attrs
    except Exception:
        logger.debug('Could not parse diagnostics.txt for attrs', exc_info=True)
        return {}


def from_pybnf(source, *, config=None, variables=None):
    """Build an :class:`arviz.InferenceData` from a finished PyBNF MCMC run.

    :param source: a ``Results/`` directory, an output directory containing
        ``Results/``, or a ``samples.txt`` file.
    :param config: optional path / :class:`~pybnf.config.Configuration` used to
        recover each parameter's scale for sampling-space output. Ignored when
        ``variables`` is given; auto-discovered from ``source`` when both are None.
    :param variables: optional list of the run's free parameters (the in-process
        auto-emit path passes these directly, so no config reload is needed).
    :returns: an ``InferenceData`` with a ``posterior`` group (one variable per
        parameter, dims ``chain`` x ``draw``, log parameters in sampling space) and
        a ``sample_stats`` group carrying ``lp`` (the recorded log-posterior).
    :raises ImportError: if the optional ``arviz`` extra is not installed.
    """
    az = _require_arviz()

    samples_path, results_dir = _resolve_samples_path(source)
    param_names, chains = _parse_samples(samples_path)
    var_map = _resolve_variables(results_dir, config, variables)

    # Rectangular chain x draw array: chains in index order, draws ordered by the
    # iteration they were recorded at, truncated to the shortest chain (ArviZ needs
    # a rectangular array; a ragged tail from an interrupted run is dropped).
    chain_ids = sorted(chains)
    ordered = {c: sorted(chains[c], key=lambda r: r[0]) for c in chain_ids}
    draw_counts = [len(ordered[c]) for c in chain_ids]
    n_draws = min(draw_counts)
    if n_draws == 0:
        raise ValueError("At least one chain in %s has no samples." % samples_path)
    if len(set(draw_counts)) > 1:
        logger.info('Chains have unequal draw counts %s; truncating all to %d for a '
                    'rectangular InferenceData.', draw_counts, n_draws)

    n_chains = len(chain_ids)
    # values[chain, draw, param]; lp[chain, draw]
    values = np.empty((n_chains, n_draws, len(param_names)))
    lp = np.empty((n_chains, n_draws))
    for ci, c in enumerate(chain_ids):
        for di in range(n_draws):
            draw_iter, draw_lp, vals = ordered[c][di]
            lp[ci, di] = draw_lp
            values[ci, di, :] = vals

    posterior = {}
    natural_fallback = []
    for pi, pname in enumerate(param_names):
        var = var_map.get(pname) if var_map else None
        col = values[:, :, pi]
        if var is not None and getattr(var, 'log_space', False):
            # Natural -> sampling space (log10 / ln), so ArviZ's diagnostics share
            # PyBNF's parameterization. samples.txt stores natural values.
            col = var.to_sampling_space(col)
        elif var_map is not None and var is None:
            natural_fallback.append(pname)
        posterior[_posterior_var_name(pname, var)] = col

    if natural_fallback:
        logger.warning('Parameters %s were in samples.txt but not in the recovered '
                       'config; emitted in natural space.', natural_fallback)

    space_clause = ('log parameters are in sampling space (log10 / ln)'
                    if var_map is not None
                    else 'log parameters are in NATURAL space (scale could not be recovered)')
    attrs = {
        'inference_library': 'pybnf',
        'created_from': str(samples_path),
        'note': ('posterior is the saved (thinned by sample_every, post-burn-in) '
                 'sample; %s. ArviZ recomputes R-hat/ESS on this thinned sample, so '
                 "az.ess reads lower than PyBNF's dense diagnostics.txt by design "
                 '(see pybnf_* attrs).' % space_clause),
    }
    attrs.update(_read_diagnostics_attrs(results_dir))

    idata = _build_idata(az, posterior, lp)
    # Stamp the run metadata on the posterior group, where it sits with the data and
    # survives the per-group netCDF round-trip. (Set here, not via from_dict's attrs=
    # kwarg, which 1.x reads as *per-group* attrs and would ignore a flat dict.) The
    # .posterior / .sample_stats / .attrs / to_netcdf access patterns are identical
    # across the two arviz lines.
    idata.posterior.attrs.update(attrs)
    return idata


def _build_idata(az, posterior, lp):
    """Construct the arviz container, tolerant of both arviz major lines.

    The two lines differ only in ``from_dict``'s calling convention -- 0.x takes
    per-group keywords and returns an ``InferenceData``; 1.x (the xarray-DataTree
    rewrite) takes a single group-keyed mapping and returns a ``DataTree``. Every
    downstream access the bridge and its callers use (``.posterior`` /
    ``.sample_stats`` / ``.attrs`` / ``to_netcdf``) is identical on both, so this
    one branch is the whole compatibility surface. The branch is on the signature,
    not a version string, so it tracks the actual API."""
    import inspect
    sample_stats = {'lp': lp}
    if 'posterior' in inspect.signature(az.from_dict).parameters:
        return az.from_dict(posterior=posterior, sample_stats=sample_stats)
    return az.from_dict({'posterior': posterior, 'sample_stats': sample_stats})
