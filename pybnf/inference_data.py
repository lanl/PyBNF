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
- **Groups: ``posterior`` + ``sample_stats`` (``lp``), plus an optional
  ``log_likelihood``.** When the run recorded the per-observation log-likelihood
  sidecar (``output_inference_data`` set + a per-point likelihood objfunc, ADR-0056,
  #438 item 4), the bridge adds a ``log_likelihood`` group so ``az.loo`` / ``az.waic``
  / ``az.compare`` work directly. Its values are genuine *unweighted* per-point
  log-densities (the complete, normalized family ``log_density``), not ``-score``.
  ``prior`` and ``observed_data`` remain deferred.

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


def _parse_samples(samples_path, loglik_rows=None):
    """Parse ``samples.txt`` into ``(param_names, chains)``.

    ``param_names`` is the header parameter order; ``chains`` maps a chain index to
    a list of ``(iter, lp, values, loglik)`` rows (``values`` aligned to
    ``param_names``; ``loglik`` the matching per-observation log-likelihood vector
    from the sidecar, or ``None``).

    ``loglik_rows`` is the sidecar's data rows in file order (from :func:`_parse_loglik`).
    samples.txt and log_likelihood.txt are each written one data row per saved sample,
    in the same order, so the i-th *data line* of samples.txt aligns with the i-th
    sidecar row -- the position counter advances on every data line (even a malformed,
    skipped one) so a corrupt samples row cannot shear the alignment.
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
        pos = -1  # index into loglik_rows; advances on every data line
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            pos += 1
            ll = loglik_rows[pos] if (loglik_rows is not None and pos < len(loglik_rows)) else None
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
            chains.setdefault(chain, []).append((draw_iter, lp, values, ll))

    if not chains:
        raise ValueError("No samples found in %s (header only?). A run that wrote no "
                         "post-burn-in samples produces no InferenceData." % samples_path)
    return param_names, chains


def _parse_loglik(loglik_path):
    """Parse the per-observation log-likelihood sidecar (ADR-0056, #438 item 4).

    Returns ``(obs_ids, rows)``: ``obs_ids`` the ``# <id>\\t<id>...`` header labelling
    each observation, ``rows`` the per-sample float vectors in file order -- positionally
    aligned with ``samples.txt``'s data rows (both are written one row per saved sample by
    ``sample_pset``). Returns ``(None, [])`` for an absent/empty sidecar, so a run that
    recorded no pointwise log-likelihoods (no ``output_inference_data``, or a non-likelihood
    objfunc) simply yields an InferenceData without the ``log_likelihood`` group.
    """
    obs_ids = None
    rows = []
    with open(loglik_path) as f:
        for line in f:
            s = line.rstrip('\n')
            if not s.strip():
                continue
            if s.startswith('#'):
                if obs_ids is None:
                    obs_ids = s.lstrip('#').strip().split('\t')
                continue
            rows.append([float(x) for x in s.split('\t')])
    return obs_ids, rows


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
        a ``sample_stats`` group carrying ``lp`` (the recorded log-posterior). When a
        ``log_likelihood.txt`` sidecar is present beside the samples (ADR-0056), a
        ``log_likelihood`` group (variable ``y`` over an ``obs_id`` axis) is added so
        ``az.loo`` / ``az.waic`` / ``az.compare`` work directly.
    :raises ImportError: if the optional ``arviz`` extra is not installed.
    """
    az = _require_arviz()

    samples_path, results_dir = _resolve_samples_path(source)
    # The per-observation log-likelihood sidecar (ADR-0056, #438 item 4), if the run
    # wrote one (output_inference_data + a likelihood objfunc). Read first so its rows
    # parse in lockstep with samples.txt -- they are written one row per saved sample,
    # same order, so positional row i corresponds to samples row i.
    loglik_path = Path(results_dir) / 'log_likelihood.txt'
    obs_ids, loglik_rows = _parse_loglik(loglik_path) if loglik_path.is_file() else (None, None)
    param_names, chains = _parse_samples(samples_path, loglik_rows)
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
    # values[chain, draw, param]; lp[chain, draw]; loglik[chain, draw, obs] when a sidecar
    # was found. The log_likelihood group is dropped (with a warning) if any kept draw's
    # sidecar row is missing or the wrong width -- better no group than a misaligned one.
    have_loglik = obs_ids is not None
    n_obs = len(obs_ids) if have_loglik else 0
    values = np.empty((n_chains, n_draws, len(param_names)))
    lp = np.empty((n_chains, n_draws))
    loglik = np.empty((n_chains, n_draws, n_obs)) if have_loglik else None
    for ci, c in enumerate(chain_ids):
        for di in range(n_draws):
            draw_iter, draw_lp, vals, ll = ordered[c][di]
            lp[ci, di] = draw_lp
            values[ci, di, :] = vals
            # Fill the log_likelihood array only while it is still valid: a single bad
            # row disables the group (loglik -> None, the block is skipped thereafter)
            # but values/lp keep filling, so the posterior is never left ragged.
            if have_loglik:
                if ll is None or len(ll) != n_obs:
                    logger.warning('log_likelihood.txt row for chain %d draw %d is %s '
                                   '(expected %d values); omitting the log_likelihood group.',
                                   c, di, 'missing' if ll is None else 'width %d' % len(ll), n_obs)
                    have_loglik, loglik = False, None
                else:
                    loglik[ci, di, :] = ll

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
    loo_clause = (' A log_likelihood group (%d observations) is included, so az.loo / '
                  'az.waic / az.compare work directly; its values are genuine unweighted '
                  'per-point log-densities (not -score).' % n_obs) if have_loglik else ''
    attrs = {
        'inference_library': 'pybnf',
        'created_from': str(samples_path),
        'note': ('posterior is the saved (thinned by sample_every, post-burn-in) '
                 'sample; %s. ArviZ recomputes R-hat/ESS on this thinned sample, so '
                 "az.ess reads lower than PyBNF's dense diagnostics.txt by design "
                 '(see pybnf_* attrs).%s' % (space_clause, loo_clause)),
    }
    attrs.update(_read_diagnostics_attrs(results_dir))

    # The log_likelihood group: one variable 'y' over a named 'obs_id' dimension whose
    # coordinate carries the human-readable point labels (model/suffix/observable@indvar).
    # az.loo / az.waic pool every dim except chain/draw, so the obs axis is found by shape.
    log_likelihood = {'y': loglik} if have_loglik else None
    coords = {'obs_id': obs_ids} if have_loglik else None
    dims = {'y': ['obs_id']} if have_loglik else None

    idata = _build_idata(az, posterior, lp, log_likelihood=log_likelihood, coords=coords, dims=dims)
    # Stamp the run metadata on the posterior group, where it sits with the data and
    # survives the per-group netCDF round-trip. (Set here, not via from_dict's attrs=
    # kwarg, which 1.x reads as *per-group* attrs and would ignore a flat dict.) The
    # .posterior / .sample_stats / .attrs / to_netcdf access patterns are identical
    # across the two arviz lines.
    idata.posterior.attrs.update(attrs)
    return idata


def _build_idata(az, posterior, lp, log_likelihood=None, coords=None, dims=None):
    """Construct the arviz container, tolerant of both arviz major lines.

    The two lines differ only in ``from_dict``'s calling convention -- 0.x takes
    per-group keywords and returns an ``InferenceData``; 1.x (the xarray-DataTree
    rewrite) takes a single group-keyed mapping and returns a ``DataTree``. Every
    downstream access the bridge and its callers use (``.posterior`` /
    ``.sample_stats`` / ``.log_likelihood`` / ``.attrs`` / ``to_netcdf``) is identical
    on both, so this one branch is the whole compatibility surface. The branch is on
    the signature, not a version string, so it tracks the actual API. ``coords`` /
    ``dims`` (used to name and label the log_likelihood group's observation axis) pass
    through to either calling convention unchanged."""
    import inspect
    groups = {'posterior': posterior, 'sample_stats': {'lp': lp}}
    if log_likelihood is not None:
        groups['log_likelihood'] = log_likelihood
    extra = {}
    if coords is not None:
        extra['coords'] = coords
    if dims is not None:
        extra['dims'] = dims
    if 'posterior' in inspect.signature(az.from_dict).parameters:
        return az.from_dict(**groups, **extra)
    return az.from_dict(groups, **extra)
