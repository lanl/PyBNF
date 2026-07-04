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
import warnings
from pathlib import Path

import numpy as np

logger = logging.getLogger('pybnf.inference_data')

# Parsed out of a PSet name of the form ``iter<draw>run<chain>`` (the population
# samplers' per-chain naming). ``run`` is the chain; ``iter`` is the iteration the
# draw was recorded at -- a stride of ``sample_every``, mapped to a contiguous draw
# index by ordering within the chain.
_RUN_RE = re.compile(r'(?<=run)\d+')
_ITER_RE = re.compile(r'(?<=iter)\d+')

# The Adaptive_MCMC (``am``) sampler names its per-chain draw files
# ``A_MCMC/Runs/params_<chain>.txt``; the chain index is the trailing integer.
_AM_PARAMS_RE = re.compile(r'params_(\d+)\.txt$')


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


def _discover_am_chain_files(results_dir):
    """The Adaptive_MCMC (``am``) sampler's per-chain draw files under
    ``results_dir``, ordered by chain index, or ``[]`` if this is not an am run.

    ``am`` (pybnf/algorithms/samplers/adaptive_mcmc.py) records each chain's draws to
    ``A_MCMC/Runs/params_<chain>.txt`` -- one file per chain, one row per recorded
    iteration in order -- which is the authoritative, correctly-ordered per-chain
    posterior. It is preferred over any ``samples.txt`` am also writes, because that
    file reuses the accepted pset's *proposal* name for every recorded draw, so its
    ``iter<draw>run<chain>`` labels collide whenever a proposal is rejected (the draw
    axis cannot be recovered from them); and a run that converges during the adaptive
    phase writes no post-burn-in ``samples.txt`` rows at all. Only ``am`` creates
    ``A_MCMC/Runs``, so its presence is a reliable discriminator.
    """
    runs = Path(results_dir) / 'A_MCMC' / 'Runs'
    if not runs.is_dir():
        return []
    indexed = []
    for f in runs.glob('params_*.txt'):
        m = _AM_PARAMS_RE.search(f.name)
        if m:
            indexed.append((int(m.group(1)), f))
    return [f for _, f in sorted(indexed)]


def _resolve_samples_path(source):
    """Resolve ``source`` to ``(samples, results_dir)``.

    ``source`` may be a ``samples.txt`` file, a ``Results/`` directory, or an output
    directory containing ``Results/``. The returned ``samples`` is either the
    ``samples.txt`` Path (the population samplers -- dream, pdream, basic MCMC) or,
    for an Adaptive_MCMC (``am``) run, the sorted list of its per-chain
    ``A_MCMC/Runs/params_<chain>.txt`` files (see :func:`_discover_am_chain_files`);
    ``from_pybnf`` branches on the returned type. An am run is preferred over any
    ``samples.txt`` it also wrote, whose draw labels are unreliable.
    """
    p = Path(source)
    if p.is_file():
        return p, p.parent
    if p.is_dir():
        for results_dir in (p, p / 'Results'):
            if not results_dir.is_dir():
                continue
            am_files = _discover_am_chain_files(results_dir)
            if am_files:
                return am_files, results_dir
            if (results_dir / 'samples.txt').is_file():
                return results_dir / 'samples.txt', results_dir
    raise FileNotFoundError(
        "Could not find MCMC draws under %r. Pass a Results/ directory, an output "
        "directory, or a samples.txt file from a finished MCMC run -- or an "
        "Adaptive_MCMC run's Results/ with A_MCMC/Runs/params_*.txt." % str(source))


def _parse_am_chains(files):
    """Parse Adaptive_MCMC per-chain ``params_<chain>.txt`` files into
    ``(param_names, chains)`` -- the same shape :func:`_parse_samples` returns, so
    ``from_pybnf``'s rectangular-array builder is shared.

    Each file is a header row of whitespace-separated parameter NAMES (written
    tab-joined by ``Adaptive_MCMC.write_out_params``) followed by whitespace-separated
    numeric draw rows (``np.savetxt`` default), one row per recorded iteration in
    order. There is no Name/Ln_probability column and the chain identity is the file
    (its ``params_<chain>`` index), not an ``iter<draw>run<chain>`` name; the draw
    index is the row's position in the file. ``chains`` maps chain index -> list of
    ``(draw_index, None, values, None)`` rows -- am records neither a per-draw
    log-posterior nor the per-observation log-likelihood sidecar, so lp and loglik are
    None (the built InferenceData omits the sample_stats and log_likelihood groups).
    """
    param_names = None
    chains = {}
    for path in files:
        chain = int(_AM_PARAMS_RE.search(Path(path).name).group(1))
        with open(path) as f:
            header = f.readline()
        names = header.lstrip('#').split()  # names are tab- or space-joined
        if not names:
            raise ValueError("Adaptive_MCMC params file %s has no parameter-name header." % path)
        if param_names is None:
            param_names = names
        elif names != param_names:
            raise ValueError("Adaptive_MCMC chain files disagree on parameter columns "
                             "(%s has %s, expected %s)." % (path, names, param_names))
        with warnings.catch_warnings():
            # A header-only chain (a run that recorded no draws) is a legitimate,
            # explicitly-handled case below, not a corrupt file -- silence numpy's
            # "input contained no data" warning for it.
            warnings.simplefilter('ignore', UserWarning)
            data = np.loadtxt(path, skiprows=1, ndmin=2)
        if data.size == 0:
            continue  # a header-only chain contributes no draws
        if data.shape[1] != len(param_names):
            raise ValueError("Adaptive_MCMC params file %s has %d data columns but %d "
                             "parameter names %s." % (path, data.shape[1], len(param_names), param_names))
        chains[chain] = [(i, None, list(row), None) for i, row in enumerate(data)]

    if not chains:
        raise ValueError("No draws found in the Adaptive_MCMC params files %s (headers "
                         "only?). A run that recorded no draws produces no InferenceData."
                         % [str(f) for f in files])
    return param_names, chains


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
        ``Results/``, or a ``samples.txt`` file. An Adaptive_MCMC (``am``) run's
        ``Results/`` is read from its per-chain ``A_MCMC/Runs/params_<chain>.txt``
        files (it writes no usable ``samples.txt``) -- see :func:`_resolve_samples_path`.
    :param config: optional path / :class:`~pybnf.config.Configuration` used to
        recover each parameter's scale for sampling-space output. Ignored when
        ``variables`` is given; auto-discovered from ``source`` when both are None.
    :param variables: optional list of the run's free parameters (the in-process
        auto-emit path passes these directly, so no config reload is needed).
    :returns: an ``InferenceData`` with a ``posterior`` group (one variable per
        parameter, dims ``chain`` x ``draw``, log parameters in sampling space) and,
        for samplers that record it, a ``sample_stats`` group carrying ``lp`` (the
        recorded log-posterior; omitted for ``am``, which records draws only). When a
        ``log_likelihood.txt`` sidecar is present beside the samples (ADR-0056), a
        ``log_likelihood`` group (variable ``y`` over an ``obs_id`` axis) is added so
        ``az.loo`` / ``az.waic`` / ``az.compare`` work directly.
    :raises ImportError: if the optional ``arviz`` extra is not installed.
    """
    az = _require_arviz()

    samples, results_dir = _resolve_samples_path(source)
    if isinstance(samples, list):
        # Adaptive_MCMC (am): the authoritative draws are its per-chain
        # A_MCMC/Runs/params_<chain>.txt files (correctly ordered), which carry no
        # log-posterior column and no per-observation log-likelihood sidecar.
        param_names, chains = _parse_am_chains(samples)
        obs_ids, loglik_rows = None, None
        have_lp = False
        created_from = ', '.join(str(f) for f in samples)
    else:
        # The per-observation log-likelihood sidecar (ADR-0056, #438 item 4), if the run
        # wrote one (output_inference_data + a likelihood objfunc). Read first so its rows
        # parse in lockstep with samples.txt -- they are written one row per saved sample,
        # same order, so positional row i corresponds to samples row i.
        loglik_path = Path(results_dir) / 'log_likelihood.txt'
        obs_ids, loglik_rows = _parse_loglik(loglik_path) if loglik_path.is_file() else (None, None)
        param_names, chains = _parse_samples(samples, loglik_rows)
        have_lp = True
        created_from = str(samples)
    var_map = _resolve_variables(results_dir, config, variables)

    # Rectangular chain x draw array: chains in index order, draws ordered by the
    # iteration they were recorded at, truncated to the shortest chain (ArviZ needs
    # a rectangular array; a ragged tail from an interrupted run is dropped).
    chain_ids = sorted(chains)
    ordered = {c: sorted(chains[c], key=lambda r: r[0]) for c in chain_ids}
    draw_counts = [len(ordered[c]) for c in chain_ids]
    n_draws = min(draw_counts)
    if n_draws == 0:
        raise ValueError("At least one chain in %s has no samples." % created_from)
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
    # am records no per-draw log-posterior, so lp (and hence the sample_stats group)
    # is omitted for it; every other sampler wires the Ln_probability column here.
    lp = np.empty((n_chains, n_draws)) if have_lp else None
    loglik = np.empty((n_chains, n_draws, n_obs)) if have_loglik else None
    for ci, c in enumerate(chain_ids):
        for di in range(n_draws):
            draw_iter, draw_lp, vals, ll = ordered[c][di]
            if have_lp:
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
    source_clause = ('posterior is the Adaptive_MCMC per-chain recorded draws '
                     '(A_MCMC/Runs/params_*.txt); no sample_stats.lp group is emitted '
                     '(am records draws only, not the per-draw log-posterior)'
                     if not have_lp else
                     'posterior is the saved (thinned by sample_every, post-burn-in) sample')
    attrs = {
        'inference_library': 'pybnf',
        'created_from': created_from,
        'note': ('%s; %s. ArviZ recomputes R-hat/ESS on these draws, so az.ess reads '
                 "lower than PyBNF's dense diagnostics.txt by design (see pybnf_* "
                 'attrs).%s' % (source_clause, space_clause, loo_clause)),
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
    through to either calling convention unchanged. ``lp`` is ``None`` for a sampler
    that records no per-draw log-posterior (am), in which case the sample_stats group
    is omitted -- the posterior group alone is a valid InferenceData."""
    import inspect
    groups = {'posterior': posterior}
    if lp is not None:
        groups['sample_stats'] = {'lp': lp}
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
