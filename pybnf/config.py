"""Classes and methods for configuring the fitting run"""


from .data import Data, DuplicateColumnError
from . import objective  # noqa: F401 -- imported for its side effect: running the module fires the @register_objfunc decorators, populating OBJFUNC_REGISTRY before _load_obj_func dispatches.
from . import algorithms  # noqa: F401 -- imported for its side effect: running the leaves fires the @register_fit_type decorators, populating FIT_TYPE_REGISTRY (incl. each method's config schema) before _build_config dispatches. No cycle: nothing in algorithms/ imports config.
from .registry import OBJFUNC_REGISTRY, FIT_TYPE_REGISTRY
from .priors import PRIOR_KEYWORD_MAP
from . import config_schema
from . import edition

from pydantic import ValidationError

from .pset import BNGLModel, ModelError, SbmlModel, SbmlModelNoTimeout, FreeParameter, TimeCourse, ParamScan, \
    Mutation, MutationSet
from .bngsim_sbml_model import (
    BNGSIM_HAS_SBML,
    BNGSIM_SBML_ERROR,
    BngsimSbmlModelNoTimeout,
)
from .bngsim_antimony_model import (
    BNGSIM_HAS_ANTIMONY,
    BNGSIM_ANTIMONY_ERROR,
    BngsimAntimonyModelNoTimeout,
)
from .printing import verbosity, print1, PybnfError
from .constraint import ConstraintSet

import numpy as np
import os
import re
import logging
import subprocess
from pathlib import Path
import roadrunner


logger = logging.getLogger(__name__)

# A PEtab per-measurement placeholder symbol surviving in a measurement-model formula
# (``observableParameter1_obs_y``): its presence marks a row-varying scale/offset bound per data
# point (a ``PerMeasurementModel``, ADR-0045), not a constant pre-materialized column.
_MEASUREMENT_PLACEHOLDER = re.compile(r'(?:observable|noise)Parameter\d+_\w+')

# The fit types that do NOT drive the shared simulation run loop, and so cannot honor the
# total wall-clock budget (``wall_time_fit``, ADR-0093): ``hmc`` runs blackjax NUTS in
# process over an analytical target, with no per-simulation point for the budget to stop
# at. Naming the budget for one of these is refused, not silently ignored (#529).
_NO_WALL_TIME_FIT = frozenset({'hmc'})


def init_logging(file_prefix, debug=False, log_level_name='info'):

    file_name = f'{file_prefix}.log'

    # Parse log level
    if log_level_name == 'debug' or log_level_name == 'd':
        log_level = logging.DEBUG
    elif log_level_name == 'info' or log_level_name == 'i':
        log_level = logging.INFO
    elif log_level_name == 'warning' or log_level_name == 'w':
        log_level = logging.WARNING
    elif log_level_name == 'error' or log_level_name == 'e':
        log_level = logging.ERROR
    elif log_level_name == 'critical' or log_level_name == 'c':
        log_level = logging.CRITICAL
    elif log_level_name == 'none' or log_level_name == 'n':
        log_level = logging.CRITICAL
        file_name = os.devnull
    else:
        # Should not get here because ArgumentParser catches invalid input
        raise ValueError(f'Invalid --log_level setting "{log_level_name}"')


    fmt = logging.Formatter(fmt='%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s')

    fh = logging.FileHandler(file_name, mode='a')
    fh.setLevel(log_level)
    fh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(10)
    root.addHandler(fh)

    # Route Python warnings (numpy RuntimeWarning, YAMLLoadWarning, etc.) through
    # the logging system so they go to the log file instead of spamming the terminal.
    logging.captureWarnings(True)

    dlog = logging.getLogger('distributed')
    dlog.handlers[:] = []  # remove any existing handlers
    dlog.setLevel(max(logging.WARNING, log_level))
    dlog.addHandler(fh)

    tlog = logging.getLogger('tornado')
    tlog.handlers[:] = []  # remove any existing handlers
    tlog.setLevel(logging.CRITICAL)
    tlog.addHandler(fh)

    talog = logging.getLogger('tornado.application')
    talog.handlers[:] = []
    talog.setLevel(logging.CRITICAL)
    talog.addHandler(fh)

    asynclog = logging.getLogger('asyncio')
    asynclog.setLevel(999)  # Higher than critical -> silent

    if debug:
        dfh = logging.FileHandler(f'{file_prefix}_debug.log', mode='a')
        dfh.setLevel(logging.DEBUG)
        dfh.setFormatter(fmt)

        root.addHandler(dfh)
        dlog.addHandler(dfh)
        tlog.addHandler(dfh)
        talog.addHandler(dfh)

def reinit_logging(file_prefix, debug=False, log_level_name='info'):
    """
    Shut down logging, then restart it.
    Used when some module (e.g. distributed v1.22.0) breaks the logging.
    """
    if logging.root:
        del logging.root.handlers[:]
    init_logging(file_prefix, debug, log_level_name)


# The one cross-fit_type config reach (ADR-0013, generalized in ADR-0015): when
# refine == 1, the optimizer named by ``refine_method`` (a start-point refiner --
# sim / powell / cmaes) polishes a *non*-self fit's best fit, so that fit's
# effective config must carry the whole chosen-refiner schema as a coherent group.
# ``Configuration._refiner_schema`` (a registry-keyed lookup off ``refine_method``)
# plus ``_refine_pulls_in`` are the single source of this fact, shared by
# ``_build_config`` (which overlays the schema) and ``check_unused_keys`` (which
# exempts the refiner's keys from the unused-key warning) -- so narrowing does not
# duplicate the refine fact, and adding a refiner is one ``refiner=True`` registry
# flag, not a config-py edit. Defaults to ``sim`` for backward compatibility.


# Non-schema config keys that are valid for *every* fit_type, so they never count as
# unused (#401, ADR-0014). They are the keys not modeled by any Pydantic schema yet
# still legitimate: the run selector ``fit_type``; the structural keys ``parse.py``
# synthesizes (``models``/``exp_data``) or the user supplies (``mutant``); the two
# required user keys (``population_size``/``max_iterations``); and the two run-level
# keys consumed outside the typed schema -- ``verbosity`` (read before config is
# built) and ``postprocess`` (loaded by ``_load_postprocessing``). Model-path keys
# (matched by the ``*.bngl/xml/ant`` regex) and tuple free-parameter keys are
# structural too, recognized positionally in ``_is_unused_key`` rather than listed.
# This set is the schema-free remainder of what the old model-checking ``used``
# whitelist hand-listed (which mis-spelled ``postprocess`` as ``postprocessing`` and
# omitted the global-schema keys -- both fixed by deriving from the schema instead).
STRUCTURAL_PASSTHROUGH = frozenset({
    'fit_type', 'models', 'exp_data', 'mutant',
    'population_size', 'max_iterations', 'verbosity', 'postprocess',
    # The bring-your-own objective expression (ADR-0050): consumed by
    # _add_inline_expression_target (which synthesizes the ExpressionModel), not the typed
    # schema, so it is a legitimate non-schema key like 'postprocess' -- never an unused-key
    # warning.
    'expression',
    # The bring-your-own objective callable (ADR-0050, the expression form's sibling): consumed
    # by _add_inline_callable_target (which synthesizes the CallableModel), not the typed schema
    # -- a legitimate non-schema key like 'expression', never an unused-key warning.
    'callable',
    # Experimental data bound to a callable objective (ADR-0050 data follow-up): the ``data =
    # f.exp, ...`` files _add_inline_callable_target loads onto the CallableModel. A legitimate
    # non-schema key consumed only on the callable path; never an unused-key warning.
    'data',
})


def _implicit_median_neg_bin_scopes(config, obj, explicit_global):
    """The scopes (the whole-fit default and/or named observables) where a ``neg_bin``
    noise model resolves to the modern median default (ADR-0031) without an explicit
    location -- the warn-worthy set, since ``neg_bin``'s legacy centering was the mean.
    The location-scale families are silent (byte-identical at the median). Takes the raw
    config so it works through the ``_load_obj_func`` test idiom (no ``self``)."""
    scopes = []
    # The class default (objective = neg_bin, or a whole-fit noise_model line): warn only
    # when median was reached implicitly -- not via an explicit global noise_location,
    # nor an explicit location field on the whole-fit line.
    whole_fit = config.get(('noise_model', None))
    line_set_location = whole_fit is not None and whole_fit[2] is not None
    if (obj.noise is not None and isinstance(obj.noise, objective.NegBinomial)
            and obj.noise.location is objective.MEDIAN
            and not explicit_global and not line_set_location):
        scopes.append('the whole fit')
    # Per-observable noise_model overrides carry their own location; a neg_bin override
    # with no location field resolves to the median default implicitly.
    for k, v in config.items():
        if isinstance(k, tuple) and k[0] == 'noise_model' and k[1] is not None:
            family, _fields, loc = v
            if family.lower() == 'neg_bin' and loc is None:
                scopes.append(f"observable '{k[1]}'")
    return scopes


class Configuration:
    def __init__(self, d=None):
        """
        Instantiates a Configuration object using a dictionary generated
        by the configuration file parser.  Default key, value pairs are used
        when possible for pairs not present in the provided dictionary.
        :param d: The result from parsing a configuration file
        :type d: dict
        """
        if d is None:
            d = dict()
            
        # An inline analytical objective -- a named target (ADR-0059: ``objective = banana,
        # ...``) or a bring-your-own expression / callable (ADR-0050: ``objective = expression``
        # + ``expression = ...``, or ``objective = callable`` + ``callable = mod:func``) --
        # synthesizes its own model from the config line, so it needs no ``model`` / ``model:``
        # declaration: that is the whole point of the no-sidecar surface.
        _obj = d.get('objective')
        # An inline analytical/expression/callable objective synthesizes its own model. Orphan
        # ``mode:`` lines (``('objective_modes', None)`` with no objective target) count here too,
        # so the model-presence guard does not pre-empt the pointed "you wrote mode: lines but no
        # 'objective = multimodal'" error _add_inline_analytical_target gives.
        has_inline_target = (('objective_target', None) in d
                             or ('objective_modes', None) in d
                             or (isinstance(_obj, str) and _obj.lower() in ('expression', 'callable')))
        if (('models' not in d or len(d['models']) == 0) and not has_inline_target):
            raise UnspecifiedConfigurationKeyError("'model' must be specified in the configuration file.")
        # Edition-gate the new-era `model:` declaration syntax (ADR-0028) before the
        # run selector, so a legacy conf that reaches for it gets the model-syntax
        # error rather than an incidental fit_type default warning.
        self._resolve_model_declarations(d)
        # Normalize the run selector across editions into the internal 'fit_type'
        # slot (ADR-0028): the modern edition names the run with 'job_type', legacy
        # with 'fit_type'. Surface-only -- downstream reads and the registry are
        # untouched. Must precede _build_config (line below), which dispatches on
        # d['fit_type'].
        self._resolve_run_selector(d)
        # Whether the user named the legacy ``objfunc`` key (raw presence, before the
        # schema injects its 'chi_sq' default): the modern objective surface forbids it
        # (ADR-0031), and _load_obj_func reads this to tell "user wrote objfunc" from
        # "schema defaulted it". A modern conf names an objective through the new keys,
        # so the legacy "defaulting to chi_sq" warning would mislead -- suppress it
        # there (the edition int is already parse-coerced; full validation is later in
        # _check_edition).
        self._user_objfunc = 'objfunc' in d
        _ed = d.get('edition')
        _modern_hint = isinstance(_ed, int) and not isinstance(_ed, bool) and _ed >= 2
        if not self._user_objfunc and not _modern_hint:
            print1('Warning: objfunc was not specified. Defaulting to chi_sq.')
        if not self._req_user_params() <= d.keys() and d['fit_type'] != 'check':
            unspecified_keys = []
            for k in self._req_user_params():
                if k not in d.keys():
                    unspecified_keys.append(k)
            raise UnspecifiedConfigurationKeyError(
                "The following configuration keys must be specified:\n\t"+",".join(unspecified_keys))

        if d['fit_type'] == 'check':
            # Model checking cannot run refine or bootstrap; strip them so they do
            # not crash downstream (always, regardless of verbosity). The unused-key
            # warning below is the same schema-derived derivation every fit_type uses
            # -- check just has no method schema, so its valid set is global +
            # structural (#401, ADR-0014).
            self._strip_uncheckable_keys(d)
        self._check_refine_method(d)
        if verbosity >= 1:
            self.check_unused_keys(d)
        # The MCMC-family beta-ladder preprocessing now runs inside _build_config
        # as the method schema's postprocess() hook (ADR-0006 #3), dispatched
        # uniformly there; non-MCMC methods inherit a no-op.
        self.config = self._build_config(d)
        self._check_random_seed()
        self._check_wall_time_fit()
        self._check_wall_time_refine_frac()
        self._check_edition()

        self._data_map = dict()  # Internal structure to help get both regular and mutant data to the right place
        self.models = self._load_models()
        logger.debug('Loaded models')
        self._load_actions()
        logger.debug('Loaded actions')
        self._load_simulators()
        logger.debug('Loaded simulators')
        self._load_mutants()
        logger.debug('Loaded mutants')
        self._load_conditions()
        logger.debug('Loaded conditions')
        self.mapping = self._check_actions()  # dict of model prefix -> set of experimental data prefixes
        logger.debug('Loaded model:exp mapping')
        self.exp_data, self.constraints = self._load_exp_data()
        logger.debug('Loaded data')
        # New-era experiment: front-end (ADR-0028, Chunk 3). Runs after exp_data and
        # mapping exist so it can extend them directly: it synthesizes each experiment's
        # action (suffix = experiment name, output points derived from the data), attaches
        # it to the resolved model, and adds the stacked-replicate Data + mapping entry --
        # keyed by the experiment name (the link is stated, so _check_actions' suffix-match
        # wart is bypassed: a new-era model carries no data on its model line).
        self._load_experiments()
        logger.debug('Loaded experiments')
        # Deferred until here (post-_load_experiments) so the 'smoothing' misuse check sees the
        # edition-2 models' synthesized simulate/parameter_scan actions -- and thus their
        # `stochastic` flag (#471). Legacy models set the flag at parse time, so the check is
        # order-independent for them.
        self._check_smoothing_misuse()
        # New-era observable: column-header overrides (ADR-0028, Chunk 4). Runs after every
        # experimental Data exists (so it can rename columns across all of them) and before
        # the objective is built (so the objective's by-name column match + per-observable
        # noise see the final names): renames each data column <header> -> <entity> (and
        # <header>_SD -> <entity>_SD) so a differently-named data column matches its model
        # observable. Edition-gated (>= 2); a legacy/same-named conf is untouched.
        self._load_observables()
        logger.debug('Loaded observable overrides')
        self.obj = self._load_obj_func()
        logger.debug('Loaded objective function')
        self.variables = self._load_variables()
        # New-era measurement-model observation layer (ADR-0036). Runs after the objective
        # (which it attaches to) and the variables (whose free-parameter names it excludes
        # from the constant snapshot): compiles each `observable: <id>, formula: <expr>`
        # line into a MeasurementModel evaluated post-simulation. No-op when none declared.
        # Built *before* the free-parameter orphan check below so an observation-layer nuisance
        # (an observableParameters scale referenced only by a measurement model, ADR-0044) is
        # recognized as used, not mis-flagged as a typo.
        self._load_measurement_models()
        logger.debug('Loaded measurement models')
        # Prediction-dependent noise (a combined additive+proportional error model whose σ
        # scales with the simulated output -- ADR-0075). Classifies each PredictionFormulaSigma's
        # symbols against the model namespace (which are free-parameter coefficients vs simulated
        # columns), so its estimated nuisances surface through required_free_noise_params to both
        # the declared check below and the orphan check. After the measurement layer (whose
        # namespace it shares) and before the orphan check; a no-op when none is declared.
        self._load_prediction_noise()
        logger.debug('Loaded prediction-dependent noise')
        if self.config['fit_type'] != 'check':
            self._check_variable_correspondence()
        logger.debug('Loaded variables')
        self._postprocess_normalization()
        # Analytic per-series scaling (ADR-0066) is resolved in _postprocess_normalization, so it
        # attaches to the (already-built) objective here, after the normalization grid is compiled.
        self._attach_analytic_scale()
        # Analytic noise profiling (ADR-0108, #562). Runs last of the objective-facing
        # steps: it needs the fully built objective (whose per-observable noise specs it
        # reads) and the loaded free parameters (which it partitions), and it must follow
        # _check_variable_correspondence, which is a statement about the DECLARED variables
        # -- profiling removes a declared parameter from the search, not from the .conf.
        # A no-op (leaves self.variables untouched) unless noise_profiling = 1.
        self._apply_noise_profiling()
        self._load_postprocessing()
        self.config['time_length'] = self._load_t_length()
        # Off-diagonal cross-product pruning (#484, ADR-0069). Runs last, when exp_data,
        # constraints, and postprocessing are all populated: records per model the full
        # output suffixes any consumer reads, so the backend execute() can skip the
        # unscored {action} x {condition} cross-product it would otherwise compute and
        # discard. A no-op (leaves emit_suffixes empty) for every non-edition-2 or
        # not-cleanly-separable model, so legacy jobs are byte-identical.
        self._compute_emit_suffixes()
        logger.debug('Completed configuration')

    @staticmethod
    def default_config():
        """Default configuration values.

        The defaults now live in the typed Pydantic schema (``config_schema``,
        ADR-0002); this is a thin compat shim over it so existing callers keep
        working. Returns a fresh plain dict each call.
        """
        return config_schema.default_config_dict()

    @staticmethod
    def _build_config(d):
        """Build the effective config dict from a raw parsed config ``d``.

        The flow is ``raw dict -> Pydantic (validate / coerce / default) ->
        effective dict`` (ADR-0002). The effective dict is **narrowed** to the keys
        the selected fit_type actually reads (ADR-0013, M2.1 Stage c) -- so a ``de``
        fit no longer carries ``cognitive`` / ``simplex_*`` / the MCMC defaults --
        assembled in (up to) four overlays:

        1. the validated *global* keys (``GlobalConfig``) -- its full defaults
           overlaid by the global keys the user set;
        2. the selected fit_type's *method* schema -- its validated keys plus its
           own defaults (absent for ``check``, which has no co-located schema);
        3. the **refine->simplex overlay** -- the one cross-fit_type reach: when
           ``refine == 1`` on a non-``sim`` fit, the whole Simplex schema as a
           coherent group (``_REFINER_SCHEMA`` / :meth:`_refine_pulls_in`), so
           ``_refine_best_fit`` never meets a half-populated state;
        4. the *extras* -- required user keys (``population_size`` /
           ``max_iterations``), keys of *other* methods the user set, and the
           structural model-path / free-parameter (tuple) / ``models`` /
           ``exp_data`` keys -- carried through unchanged (already parse-coerced).

        The raw dict is partitioned by key ownership: a string key owned by
        ``GlobalConfig`` is a global key, one owned by the selected method's
        schema is a method key, everything else (including the keys of *other*
        methods) is an extra. The three buckets are disjoint, so an extra never
        clobbers a validated default. Narrowing drops only the *unset defaults* of
        other methods: a user-set foreign key (e.g. ``cognitive`` on a ``de`` fit)
        rides through as an extra unchanged, reported by ``check_unused_keys``
        exactly as before. A Pydantic ``ValidationError`` becomes a ``PybnfError``.

        The result is a plain dict so the existing ``config.config['x']`` reaches
        and writes stay untouched (dict-compat per ADR-0002); typed access
        migrates opportunistically in Stage (c).
        """
        entry = FIT_TYPE_REGISTRY.get(d.get('fit_type'))
        method_schema = entry.schema if entry is not None else None
        if method_schema is not None:
            # Method-owned preprocessing on the RAW dict, before defaults merge
            # (raw-presence semantics like ``'beta' not in d`` mean "user did not
            # set it"): the MCMC family's beta-ladder mutates d in place, DREAM's
            # postprocess also pins adaptive_step_size off when step_size is set;
            # every other model inherits the no-op postprocess (ADR-0006 #3).
            method_schema.postprocess(d, d.get('fit_type'))
        global_keys = config_schema.SCHEMA_KEYS
        method_keys = method_schema.owned_keys() if method_schema is not None else frozenset()

        global_input = {k: v for k, v in d.items()
                        if isinstance(k, str) and k in global_keys}
        method_input = {k: v for k, v in d.items()
                        if isinstance(k, str) and k in method_keys}
        extras = {k: v for k, v in d.items()
                  if not (isinstance(k, str) and (k in global_keys or k in method_keys))}
        try:
            effective = config_schema.build_effective_global(global_input)
            if method_schema is not None:
                effective.update(
                    config_schema.build_effective_method(method_schema, method_input))
            # The one cross-fit_type reach (ADR-0013/0015): when refine pulls in
            # the chosen refiner (refine_method) on a fit that is not itself that
            # refiner, overlay the whole refiner schema as a coherent group (its
            # defaults overlaid by any of its keys the user set). A fit that *is*
            # the refiner already carries the group via its own method schema above.
            refiner_schema = Configuration._refiner_schema(d)
            if (Configuration._refine_pulls_in(d) and refiner_schema is not None
                    and d.get('fit_type') != d.get('refine_method', 'sim')):
                refiner_input = {k: v for k, v in d.items()
                                 if isinstance(k, str) and k in refiner_schema.owned_keys()}
                effective.update(
                    config_schema.build_effective_method(refiner_schema, refiner_input))
        except ValidationError as e:
            raise PybnfError('Invalid configuration', f'Invalid configuration:\n{e}')
        effective.update(extras)
        return effective

    @staticmethod
    def _refine_pulls_in(conf_dict):
        """True when ``refine`` will run a refiner over this fit's config, so the
        whole chosen-refiner schema must be present (ADR-0013/0015). The single
        predicate behind both the :meth:`_build_config` overlay and
        ``check_unused_keys``'s refiner-key exemption -- the refine fact in one
        place."""
        return conf_dict.get('refine') == 1

    @staticmethod
    def _refiner_schema(conf_dict):
        """The config schema of the refiner selected by ``refine_method`` (default
        ``sim``), or ``None`` when ``refine_method`` is not a registered refiner.

        Registry-keyed (ADR-0005/0015): adding a refiner is a ``refiner=True`` flag
        on its ``register_fit_type``, with no edit here. An invalid ``refine_method``
        is reported with a friendly error by :meth:`_check_refine_method`; this
        helper degrades to ``None`` so the build does not also raise."""
        method = conf_dict.get('refine_method', 'sim')
        entry = FIT_TYPE_REGISTRY.get(method)
        if entry is not None and entry.refiner:
            return entry.schema
        return None

    @staticmethod
    def _check_refine_method(conf_dict):
        """Validate ``refine_method`` when ``refine == 1``: it must name a
        registered refiner (``refiner=True``). Raises ``PybnfError`` otherwise.
        A no-op when refine is off (or stripped, e.g. for ``check``)."""
        if conf_dict.get('refine') != 1:
            return
        method = conf_dict.get('refine_method', 'sim')
        entry = FIT_TYPE_REGISTRY.get(method)
        if entry is None or not entry.refiner:
            valid = ', '.join(sorted(c for c, e in FIT_TYPE_REGISTRY.items() if e.refiner))
            raise PybnfError(f'Invalid refine_method {method}',
                             f"Invalid refine_method '{method}'. Options are: {valid}.")

    def _check_random_seed(self):
        """Validate the optional random seed before NumPy consumes it."""
        seed = self.config['random_seed']
        if seed is None:
            return
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0 or seed >= 2**32:
            raise PybnfError(f'Invalid random_seed {seed}',
                             "Config key 'random_seed' must be an integer from 0 to %i." % (2**32 - 1))
        self.config['random_seed'] = int(seed)

    def _check_wall_time_fit(self):
        """Validate the fit's total wall-clock budget (``wall_time_fit``, ADR-0093).

        A non-negative integer number of seconds; 0 (the default) means unbounded.
        The budget is honored by every fit that drives the shared run loop -- it is
        the loop that stops launching work and finalizes -- so a fit type that runs
        its own loop instead is **refused** here rather than silently ignoring the
        key: ``hmc`` samples in process through blackjax, with no per-simulation
        dispatch point for the budget to stop at.
        """
        limit = self.config.get('wall_time_fit', 0)
        if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, np.integer)) or limit < 0:
            raise PybnfError(f'Invalid wall_time_fit {limit}',
                             "Config key 'wall_time_fit' must be a non-negative integer number of seconds "
                             "(0, the default, means no total time limit).")
        self.config['wall_time_fit'] = int(limit)
        if limit and self.config['fit_type'] in _NO_WALL_TIME_FIT:
            raise PybnfError(
                f"wall_time_fit is not honored by fit_type {self.config['fit_type']}",
                f"Config key 'wall_time_fit' is not honored by job_type = "
                f"{self.config['fit_type']}: that job runs its own in-process sampling loop "
                f"instead of the simulation run loop the budget stops, so a deadline set here "
                f"would never fire.",
                hint='Bound this job with its own budget keys (num_warmup / num_samples / '
                     'num_parallel), or remove wall_time_fit.')

    def _check_wall_time_refine_frac(self):
        """Validate the refine's share of the wall-clock budget (``wall_time_refine_frac``,
        ADR-0107).

        A fraction in [0, 1): the tail of ``wall_time_fit`` the search may not spend, so
        that the refine ``refine = 1`` asked for actually runs (#564). 1 is excluded
        because it would leave the search nothing at all -- a fit that never searches is
        not what any conf means -- and a negative reserve is meaningless. The key is inert
        without both ``wall_time_fit`` and ``refine = 1``, so it is validated whether or
        not it is in play: a typo should be caught where it is written, not where it
        happens to bite.
        """
        frac = self.config.get('wall_time_refine_frac', 0.0)
        if isinstance(frac, (bool, np.bool_)) or not isinstance(frac, (int, float, np.integer, np.floating)) \
                or not 0.0 <= float(frac) < 1.0:
            raise PybnfError(f'Invalid wall_time_refine_frac {frac}',
                             "Config key 'wall_time_refine_frac' must be a number in [0, 1): the share of "
                             "wall_time_fit held back from the search so the refine can run "
                             "(0.1, the default, reserves a tenth of the budget; 0 reserves nothing "
                             "and lets the search spend it all).")
        self.config['wall_time_refine_frac'] = float(frac)

    @staticmethod
    def _resolve_model_declarations(d):
        """Edition-gate the new-era ``model:`` declaration syntax (ADR-0028, Chunk 1).

        The parser folds each ``model:`` file into the *same* structures a legacy
        ``model = file : none`` line produces -- the ``models`` set plus an empty exp
        list -- so the downstream model loader is untouched (a different front-end,
        the same internal objects). It also accumulates the declared files in the
        structural ``'model'`` marker. This gate requires ``edition >= 2`` for that
        syntax (the parser accepts it regardless, so the error is an explanatory
        ``require_edition`` rather than a bare parse failure), then consumes the marker
        so it never reaches the schema or the unused-key warning.

        The legacy ``model = file : exp`` form is **unchanged at every edition** -- a
        modern conf still uses it to bind data until the ``experiment:`` / ``data:``
        surface lands (Chunk 3); the "refuse legacy everything" pass is Chunk 5.
        Mutates ``d``.
        """
        declared = d.pop('model', None)
        if declared:
            ed = edition.resolve_edition(d.get('edition'))
            edition.require_edition(ed, 2, "the 'model:' declaration syntax")

    @staticmethod
    def _resolve_run_selector(d):
        """Normalize the run selector into the internal ``fit_type`` slot, honoring
        the edition-gated ``fit_type`` -> ``job_type`` rename (ADR-0028 addendum).

        ``fit_type`` is a misnomer: the key selects across optimizers, samplers, and
        the model checker -- not just *fitting* -- so the modern era renames it to
        ``job_type`` (the value names the procedure; the key names the *kind of job*).
        This is a **surface-only** rename: whichever key the edition allows is read
        here and written into ``d['fit_type']``, so ``FIT_TYPE_REGISTRY`` and the
        downstream ``config['fit_type']`` reads are untouched. The gate mirrors
        :meth:`_load_obj_func`'s ``objfunc`` -> ``objective`` gating exactly:

        * **Modern** (``edition >= 2``): ``job_type`` names the run; the legacy
          ``fit_type`` key is rejected, and -- as with the modern objective surface --
          there is **no implicit default**, so ``job_type`` must be named.
        * **Legacy** (no ``edition`` / implicit edition 1): ``fit_type`` names the run
          (defaulting to ``de``); a modern ``job_type`` is rejected with the edition it
          needs (``require_edition``). The historical ``bmc`` -> ``mh`` alias is kept
          (legacy-only).

        Mutates ``d`` so the run selector lives in ``d['fit_type']``. (The edition is
        already parse-coerced to an int; a malformed value raises here via
        ``resolve_edition``, the same validation :meth:`_check_edition` repeats later.)
        """
        ed = edition.resolve_edition(d.get('edition'))
        if edition.is_modern(ed):
            if 'fit_type' in d:
                raise PybnfError(
                    'fit_type is legacy syntax',
                    f"Config key 'fit_type' is legacy (edition 1) syntax and is not "
                    f"available under edition {ed}. Name the run with 'job_type' instead "
                    f"(e.g. 'job_type = de').")
            if 'job_type' not in d:
                raise UnspecifiedConfigurationKeyError(
                    f"Under edition {ed} the run must be named explicitly with "
                    f"'job_type = <name>' (an optimizer, a sampler, or 'check'); there is "
                    f"no implicit default.")
            d['fit_type'] = d['job_type']
        else:
            if 'job_type' in d:
                edition.require_edition(ed, 2, "the 'job_type' key")
            if 'fit_type' not in d:
                d['fit_type'] = 'de'
                # Legacy branch only: 'job_type' is rejected at this edition, so name the
                # key this reader actually sets rather than the modern one (#525 follow-up).
                print1('Warning: No job type was specified (legacy key: fit_type). '
                       'Defaulting to de (Differential Evolution).')
            if d['fit_type'] == 'bmc':
                d['fit_type'] = 'mh'  # 'bmc' option was renamed to 'mh'. Preserve backwards compatibility.

    def _check_edition(self):
        """Validate the optional ``edition`` marker (ADR-0031) before any
        edition-gated logic reads it. Absent (``None``) is legacy and always valid;
        an explicit value must be a supported integer edition. Translation to the
        legacy edition is left to each read site via ``edition.resolve_edition`` so
        the effective config keeps the raw value."""
        value = self.config['edition']
        if value is not None:
            edition.validate_edition(value)

    @staticmethod
    def _valid_config_keys(conf_dict):
        """The config keys the chosen fit_type legitimately reads (#401, ADR-0014).

        Derived entirely from the registry's schemas -- the single source of truth
        that replaced the hand-maintained ``alg_specific`` dict and the model-checking
        ``used`` whitelist. It is the union of:

        * the global schema (``GlobalConfig``, run-level keys read for any fit_type);
        * the selected fit_type's own method schema -- its owned fields *plus* the
          runtime-defaulted keys it reads but does not model (``schema.valid_keys()``);
          absent for ``check``, which has no co-located schema, so its valid set is
          just global + structural;
        * the chosen-refiner group, when ``refine`` pulls in the ``refine_method``
          refiner on a fit that is not itself that refiner (the one cross-fit_type
          reach, ``_refiner_schema``, ADR-0013/0015);
        * :data:`STRUCTURAL_PASSTHROUGH` -- the schema-free always-valid keys.

        This is exactly the ownership ``_build_config`` partitions a raw dict by, so
        "what narrowing keeps" and "what the warning accepts" cannot drift.
        """
        keys = set(config_schema.SCHEMA_KEYS) | set(STRUCTURAL_PASSTHROUGH)
        entry = FIT_TYPE_REGISTRY.get(conf_dict.get('fit_type'))
        if entry is not None and entry.schema is not None:
            keys |= entry.schema.valid_keys()
        refiner_schema = Configuration._refiner_schema(conf_dict)
        if (Configuration._refine_pulls_in(conf_dict) and refiner_schema is not None
                and conf_dict.get('fit_type') != conf_dict.get('refine_method', 'sim')):
            keys |= refiner_schema.valid_keys()
        return keys

    @staticmethod
    def _is_unused_key(k, valid_keys):
        """True when raw config key ``k`` is unused (ignored) for the fit_type whose
        ``valid_keys`` were passed: a string key the method does not read and that is
        not a structural model-path key. A non-string key is a free-parameter tuple
        (e.g. ``('uniform_var', 'p1')``), which is structural and never unused -- the
        ``isinstance(k, str)`` guard also keeps ``re.search`` off a tuple (CFG-CHECK-1).
        """
        if not isinstance(k, str):
            return False
        if k in valid_keys:
            return False
        # A model-path key (e.g. 'parabola.bngl', 'gaussian.target') is structural,
        # not a config knob. The extensions MUST match the model_file grammar in
        # parse.py (``.*?\.(bngl|xml|ant|target)``) -- a missing ``target`` here
        # spuriously warned on every .target model under the broad policy (#401).
        return re.search(r'\.(bngl|xml|ant|target)', k) is None

    @staticmethod
    def check_unused_keys(conf_dict):
        """Warn for each config key the chosen fit_type will ignore.

        A key is unused precisely when it is a non-structural *extra* -- owned by
        neither the global schema, the fit_type's own method schema (including its
        runtime keys), nor the refine->simplex group, and not a structural
        model-path / free-parameter / ``models`` / ``exp_data`` / required key
        (#401, ADR-0014). This one derivation, off :meth:`_valid_config_keys`,
        replaced three hand-maintained per-fit_type ownership encodings -- the
        ``alg_specific`` dict, the model-checking ``used`` whitelist, and the
        warn-only branches of ``MCMCFamilyConfig.postprocess`` -- so they can no
        longer drift from what ``_build_config`` narrows to. Runs on the raw config
        dict before ``_build_config`` (the same dict narrowing partitions), and so
        covers every fit_type uniformly, ``check`` included.
        """
        valid = Configuration._valid_config_keys(conf_dict)
        for k in conf_dict:
            if Configuration._is_unused_key(k, valid):
                # % (k,) (not % k): only string keys reach here, but keep the
                # single-arg tuple form so the message can never spread a value.
                print1('Warning: Configuration key {} is not used in job_type {}, so I am ignoring it'.format(k, conf_dict['fit_type']))
                logger.warning('Ignoring unused key {} for fitting algorithm {}'.format(k, conf_dict['fit_type']))

    @staticmethod
    def _strip_uncheckable_keys(conf_dict):
        """Remove the keys ``job_type = check`` cannot honor so they do not crash
        downstream: ``refine`` and ``bootstrap`` (model checking runs neither), and
        ``wall_time_fit`` (a check is one evaluation of given parameters, not a
        search, so there is no run to budget). The unused-key *warnings* for check now
        come from the unified :meth:`check_unused_keys`; this keeps only the
        crash-prevention deletion the old ``check_unused_keys_model_checking`` also
        did. Mutates and returns ``conf_dict``.
        """
        for k in ('refine', 'bootstrap', 'wall_time_fit'):
            conf_dict.pop(k, None)
        return conf_dict

    @staticmethod
    def _req_user_params():
        """Configuration keys that the user must specify"""
        return {'models', 'population_size', 'max_iterations'}

    @staticmethod
    def _absolute(directory):
        """
        Convert relative paths to absolute paths
        """
        home_dir = os.getcwd()
        if os.name == 'nt':  # Windows
            if directory == '':
                return ''
            # Check for both unix-like and windows-like paths starting from root
            if directory[0] == '/' or re.match(r'[A-Z]:', directory):
                return directory
            else:
                return os.path.join(home_dir, directory)
        return '' if directory == '' else directory if directory[0] == '/' else str(Path(home_dir) / directory)

    def _load_t_length(self):
        # New-era BNGL models carry no ``__FREE`` markers (ADR-0034); suppress the
        # legacy "no __FREE -> error" guard here too (as ``check`` already does), so
        # measuring output lengths never trips it.
        modern = edition.is_modern(edition.resolve_edition(self.config.get('edition')))
        timeDict = {}
        for mf in self.config['models']:
            if re.search(r'\.bngl$', mf):
                time = BNGLModel(mf, suppress_free_param_error=(self.config['fit_type']=='check' or modern)).find_t_length()
                for i,v in time.items():
                    timeDict[i] = v
            elif re.search(r'\.(xml|ant)$', mf):
                for tc in self.config['time_course']:
                    suffix = tc['suffix']
                    try:
                        step = float(tc['step'])
                    except KeyError:
                        step = 1.
                    end_time = float(tc['time'])
                    timeDict[suffix] = int(np.round(end_time / step))
        # New-era experiments (ADR-0028) synthesize their actions outside both the model
        # file and the legacy time_course list, so find_t_length / the xml branch above
        # never see them. _load_experiments recorded each one's output length (n_points-1),
        # keyed by the bare experiment suffix; merge it in.
        timeDict.update(getattr(self, '_experiment_time_length', {}))
        return timeDict
        
                

    def _load_models(self):
        """
        Loads models specified in configuration file in a dictionary keyed on
        Model.name
        """

        allowed_sbml_backends = ('roadrunner', 'bngsim')
        if self.config['sbml_backend'] not in allowed_sbml_backends:
            raise PybnfError('Invalid sbml_backend {}. Options are: {}.'.format(self.config['sbml_backend'], ', '.join(allowed_sbml_backends)))
        allowed_bngl_backends = ('auto', 'bionetgen', 'bngsim')
        bngl_backend = self.config.get('bngl_backend', 'auto')
        self.config['bngl_backend'] = bngl_backend
        if bngl_backend not in allowed_bngl_backends:
            raise PybnfError('Invalid bngl_backend {}. Options are: {}.'.format(bngl_backend, ', '.join(allowed_bngl_backends)))

        allowed_stochastic_seed = ('auto', 'auto_honorbngl', 'random', 'random_honorbngl')
        stochastic_seed = self.config.get('stochastic_seed', 'auto')
        self.config['stochastic_seed'] = stochastic_seed
        if stochastic_seed not in allowed_stochastic_seed:
            raise PybnfError('Invalid stochastic_seed {}. Options are: {}.'.format(stochastic_seed, ', '.join(allowed_stochastic_seed)))

        # If needed, choose the default timeout, which depends on what simulators the models use.
        if self.config['wall_time_sim'] is None:
            self.config['wall_time_sim'] = 0
            for mf in self.config['models']:
                if re.search(r'\.bngl$', mf):
                    self.config['wall_time_sim'] = 3600
                    break

        # New-era (edition >= 2) BNGL models bind free parameters by id and carry no
        # ``__FREE`` markers (ADR-0034), so the "no __FREE -> error" guard is legacy-only;
        # the model checker (``fit_type == 'check'``) suppresses it at every edition.
        modern = edition.is_modern(edition.resolve_edition(self.config.get('edition')))

        md = {}
        for mf in self.config['models']:
            # Initialize model type based on extension
            try:
                if re.search(r'\.bngl$', mf):
                    # #473: the model-scoped generate_network conf option rides into the
                    # BNGLModel so its edition-2-synthesized generate_network line carries the
                    # cap (max_stoich / max_agg / max_iter). None when unset -> the bare default.
                    model = BNGLModel(mf, suppress_free_param_error=(self.config['fit_type']=='check' or modern),
                                      generate_network_options=self.config.get('generate_network'))
                    model.bng_command = self._absolute(self.config['bng_command'])
                    logger.debug(f'Set model {mf} command to {model.bng_command}')
                elif re.search(r'\.xml$', mf):
                    save_flag = (self.config['delete_old_files'] == 0)
                    if self.config['sbml_backend'] == 'bngsim':
                        if not BNGSIM_HAS_SBML:
                            raise PybnfError(
                                f'sbml_backend = bngsim was requested, but {BNGSIM_SBML_ERROR}.'
                            )
                        strict_ssa = bool(self.config.get('sbml_ssa_strict', 1))
                        # bngsim now enforces wall_time_sim in-process via
                        # SimulationTimeout, so the subprocess wrapper is no
                        # longer needed for either zero or positive timeouts.
                        model = BngsimSbmlModelNoTimeout(
                            mf,
                            self._absolute(mf),
                            save_files=save_flag,
                            integrator=self.config['sbml_integrator'],
                            strict_ssa=strict_ssa,
                            rtol=self.config.get('sbml_rtol'),
                            atol=self.config.get('sbml_atol'),
                        )
                    elif self.config['wall_time_sim'] == 0:
                        model = SbmlModelNoTimeout(
                            mf,
                            self._absolute(mf),
                            save_files=save_flag,
                            integrator=self.config['sbml_integrator'],
                        )
                    else:
                        model = SbmlModel(
                            mf,
                            self._absolute(mf),
                                save_files=save_flag,
                                integrator=self.config['sbml_integrator'],
                            )
                elif re.search(r'\.ant$', mf):
                    save_flag = (self.config['delete_old_files'] == 0)
                    if not BNGSIM_HAS_ANTIMONY:
                        raise PybnfError(
                            f'Antimony model support was requested, but {BNGSIM_ANTIMONY_ERROR}.'
                        )
                    strict_ssa = bool(self.config.get('sbml_ssa_strict', 1))
                    # bngsim now enforces wall_time_sim in-process via
                    # SimulationTimeout, so the subprocess wrapper is no
                    # longer needed for either zero or positive timeouts.
                    model = BngsimAntimonyModelNoTimeout(
                        mf,
                        self._absolute(mf),
                        save_files=save_flag,
                        integrator=self.config['sbml_integrator'],
                        strict_ssa=strict_ssa,
                        rtol=self.config.get('sbml_rtol'),
                        atol=self.config.get('sbml_atol'),
                    )
                elif re.search(r'\.target$', mf):
                    from .analytical_model import AnalyticalModel
                    model = AnalyticalModel(mf)
                else:
                    # Should not get here - should be caught in parsing
                    raise ValueError(f'Unrecognized model suffix in {mf}')
            except FileNotFoundError:
                raise PybnfError(f'Model file {mf} was not found.')
            except ModelError as e:
                raise PybnfError(f'In model file {mf}: {e.message}')
            if model.name in md:
                raise PybnfError(f'Multiple models with the name "{model.name}". Please give all your models different names. ')
            md[model.name] = model
            self._data_map[model.name] = self.config[mf]  # List of exp files associated with this model

        self._validate_analytical_data_key()
        self._add_inline_analytical_target(md)
        self._add_inline_expression_target(md)
        self._add_inline_callable_target(md)

        for model in md.values():
            if isinstance(model, BNGLModel) and not model.has_observables:
                print1(f'Warning: Model {model.file_path} has no observables defined. Fitting will not work without observables.')
                logger.warning(f'Model {model.file_path} has no observables defined')

        # The 'smoothing' misuse check (stochastic-method + explicit-seed) is deferred to
        # _check_smoothing_misuse(), called from __init__ AFTER _load_experiments(): an
        # edition-2 model's simulate/parameter_scan is synthesized there from the experiment
        # line (BNGLModel.add_action), so its `stochastic` flag is not yet set at this point
        # (#471). Checking here would false-alarm on every edition-2 ssa/nf fit.

        # Warn once per model when explicit BNGL seeds will be overridden by the
        # current policy. Saves the user from "this fit gives different numbers
        # than I expected and I forgot model.bngl had seed=>42 in it" debugging.
        if self.config['stochastic_seed'] in ('auto', 'random'):
            for m in md.values():
                if isinstance(m, BNGLModel) and m.seeded:
                    print1('Warning: model {} contains an explicit "seed" argument; it will be '
                           'overridden by stochastic_seed={}. Use stochastic_seed={}_honorbngl to '
                           'honor the BNGL seed.'.format(m.name, self.config['stochastic_seed'], self.config['stochastic_seed']))

        if self.config['parallelize_models'] > len(md):
            raise PybnfError('Job contains %i models, so "parallelize_models" should be at most %i' % (len(md), len(md)))

        return md

    def _check_smoothing_misuse(self):
        """Warn/raise on ``smoothing > 1`` used with non-stochastic models or honored BNGL seeds.

        Called from ``__init__`` AFTER ``_load_experiments()`` so it sees the fully-synthesized
        actions: an edition-2 model's ``simulate``/``parameter_scan`` is built there from the
        experiment line (``BNGLModel.add_action``), which is where its ``stochastic`` flag gets
        set (#471). Running this inside ``_load_models`` (before experiments are loaded) would
        flag every edition-2 ``method: ssa``/``nf`` fit as non-stochastic and false-alarm.
        """
        if self.config['smoothing'] <= 1:
            return
        md = self.models
        stochastic = np.any([m.stochastic for m in md.values()])
        if not stochastic:
            print1('Warning: You specified smoothing=%i, but it looks like none of your models use a stochastic '
                   'method. All of your smoothing replicates will come out identical.' % self.config['smoothing'])
        seeded_models = [m for m in md.values() if isinstance(m, BNGLModel) and m.seeded]
        if seeded_models and self.config['stochastic_seed'].endswith('_honorbngl'):
            raise PybnfError(
                'You specified smoothing=%i with stochastic_seed=%s, and one of your simulation '
                'commands contains an explicit "seed" argument. Under the "_honorbngl" policies, '
                'that seed is honored verbatim, which would cause all of your smoothing replicates '
                'to come out the same. Switch to stochastic_seed=auto (default) or stochastic_seed=random '
                'so PyBNF overrides the BNGL seed per replicate.'
                % (self.config['smoothing'], self.config['stochastic_seed'])
            )

    def _add_inline_analytical_target(self, md):
        """Synthesize the AnalyticalModel for an inline named objective (ADR-0059 item 6).

        The whole off-the-shelf analytical menu is declarable inline in an edition-2 config now --
        no ``.target`` JSON sidecar for any of them::

            objective = gaussian,        mean = 0 0,  variance = 1 1
            objective = rotated_gaussian, mean = 0 0, variances = 2 0.5, angle = 0.5236
            objective = rotated_quartic, mean = 0 0,  angle = 0.5236, coeff = 0.01 1
            objective = banana,          a = 1, b = 100
            objective = multimodal                       # components on separate mode: lines
            mode: weight = 0.5, mean = -4 -4, variance = 0.5 0.5
            mode: weight = 0.5, mean =  4  4, variance = 0.5 0.5

        The parser leaves a structural ``('objective_target', None)`` = ``(name, {field: value})``
        (and, for multimodal, a ``('objective_modes', None)`` list of component dicts); this method
        validates them against the target's field schema, builds the same ``target_def`` dict a
        ``.target`` file would (deriving ``rotated_gaussian``'s covariance from variances + angle),
        and injects an in-memory :class:`AnalyticalModel` into the model dict ``md`` -- so the run
        executes it like any other model and ``DirectPassObjective`` reads its ``score``; no
        experimental data is needed.

        Edition-2 gated. Fields default to the documented values and are **echoed** at run start so
        the geometry is never silently assumed (the #425 discoverability footgun); an unknown name,
        an unknown/duplicate/missing field, or a scalar-vs-vector mismatch errors clearly."""
        spec = self.config.get(('objective_target', None))
        modes_raw = self.config.get(('objective_modes', None))
        if spec is None:
            if modes_raw:
                raise PybnfError(
                    "'mode:' line(s) were given without an inline 'objective = multimodal' target. "
                    "A mode: record declares one mixture component of a multimodal objective.")
            return
        from .analytical_model import (AnalyticalModel, INLINE_TARGET_SCHEMAS,
                                       MULTIMODAL_MODE_SCHEMA, build_rotated_covariance)
        ed = edition.resolve_edition(self.config.get('edition'))
        edition.require_edition(
            ed, 2, "the inline 'objective = <target>, <field> = <value>' named-target syntax")
        target_name, user_fields = spec
        schema = INLINE_TARGET_SCHEMAS[target_name]   # parser only emits known names
        consts = self._coerce_inline_target_fields(target_name, schema, user_fields,
                                                   where=f"objective '{target_name}'")

        if target_name == 'multimodal':
            if not modes_raw:
                raise PybnfError(
                    "'objective = multimodal' needs at least one mixture component. Add a "
                    "'mode: weight = .., mean = .. .., variance = .. ..' line per mode.")
            consts['modes'] = [self._coerce_inline_target_fields(
                                   'multimodal', MULTIMODAL_MODE_SCHEMA, m, where='a mode: record')
                               for m in modes_raw]
        elif modes_raw:
            raise PybnfError(
                f"'mode:' line(s) are only valid with 'objective = multimodal', not "
                f"'objective = {target_name}'.")

        if target_name == 'rotated_gaussian':
            # Sugar: derive the covariance matrix the model consumes from the conf-friendly
            # principal-variances + angle form (2-D), then drop the friendly fields.
            variances, angle = consts.pop('variances'), consts.pop('angle')
            if len(variances) != 2:
                raise PybnfError(
                    "Inline 'objective = rotated_gaussian' is 2-D (a single angle defines a planar "
                    f"rotation); 'variances' needs exactly 2 values, got {variances}.")
            consts['covariance'] = build_rotated_covariance(variances, angle)

        model = AnalyticalModel(target_def={'type': target_name, **consts}, name=target_name)
        if model.name in md:
            raise PybnfError(
                f'The inline objective target "{model.name}" collides with a declared model '
                f'of the same name. Rename the model file.')
        md[model.name] = model
        self._data_map[model.name] = []   # analytical target: no experimental data
        # _check_actions reads the per-model exp list at self.config[model.file_path];
        # mirror the .target path's empty list (there is no file, so file_path == name).
        self.config[model.file_path] = []
        # Eager bind-by-name validation (ADR-0034): an inline objective line is unambiguously an
        # analytical fit whose declared free parameters ARE the target's coordinates, so validate
        # them here -- a pointed config-load error (the #425 footgun example), not a swallowed
        # failed-simulation at run. (The file ``.target`` path stays lazy: there a param-agnostic
        # ``AnalyticalModel`` may be a throwaway vehicle for an unrelated config feature.) Declared
        # ids come from the config keys -- the same source _load_variables reads, which has not run
        # yet (it follows _load_models).
        declared = sorted(k[1] for k in self.config.keys() if self._is_free_param_key(k))
        if declared:
            model.coordinate_order(declared)   # raises a pointed PybnfError on a bad name set
        print1(f'Objective: analytical {target_name} target ({self._echo_inline_target(consts)}).')
        logger.info(f'Inline analytical objective: {target_name} with {consts}')

    @staticmethod
    def _coerce_inline_target_fields(target_name, schema, user_fields, *, where):
        """Validate the parsed inline fields against a target/mode field ``schema`` and coerce each
        to its declared kind (ADR-0059 item 6 completion).

        ``schema`` maps ``field -> (kind, default)`` (``kind`` ``'scalar'``/``'vector'``, ``default``
        ``None`` = required). Unknown, duplicate (already caught by the parser), or missing-required
        fields, and a scalar-where-vector (or vice-versa) value, all raise a pointed ``PybnfError``
        naming ``where``. A 1-D vector parsed as a bare scalar (``mean = 0``) is wrapped back into a
        one-element list; a scalar field keeps its single number. Returns the coerced ``{field:
        value}`` dict (vectors as ``list``, scalars as ``float``)."""
        unknown = sorted(set(user_fields) - set(schema))
        if unknown:
            raise PybnfError(
                f"Unknown field(s) {unknown} for {where}.",
                f"It takes field(s) {sorted(schema)}; got unexpected {unknown}.")
        coerced = {}
        for field, (kind, default) in schema.items():
            if field not in user_fields:
                if default is None:
                    raise PybnfError(
                        f"{where[0].upper() + where[1:]} is missing the required field '{field}' "
                        f"({kind}). Required fields: "
                        f"{sorted(f for f, (_, d) in schema.items() if d is None)}.")
                coerced[field] = default
                continue
            value = user_fields[field]
            is_list = isinstance(value, list)
            if kind == 'scalar':
                if is_list:
                    raise PybnfError(
                        f"In {where}, field '{field}' takes a single number, got {value}.")
                coerced[field] = float(value)
            else:   # vector: accept a 1-element bare scalar as a 1-D vector
                coerced[field] = [float(x) for x in (value if is_list else [value])]
        return coerced

    @staticmethod
    def _echo_inline_target(consts):
        """A compact, deterministic echo of an inline target's resolved fields for the run-start
        banner (the #425 'never silently assume the geometry' principle). Modes are summarized by
        count; scalars/vectors print as given."""
        parts = []
        for k in sorted(consts):
            if k == 'modes':
                parts.append(f'{len(consts[k])} modes')
            else:
                parts.append(f'{k} = {consts[k]}')
        return ', '.join(parts)

    def _add_inline_expression_target(self, md):
        """Synthesize the :class:`ExpressionModel` for a bring-your-own objective (ADR-0050).

        ``objective = expression`` + ``expression = 0.5*((1 - x1)^2 + 100*(x2 - x1^2)^2)``
        declares the user's closed-form negative log-likelihood directly on the config line,
        with no ``.bngl`` / ``.target`` model file and no ``.exp``. The expression is compiled
        to a numpy callable over the **declared free parameters** (bind-by-name, ADR-0050 §4)
        and the resulting model is injected straight into the model dict ``md`` -- so the run
        executes it like any other model and ``DirectPassObjective`` reads its ``score`` cell
        (the same fileless synthesis / injection / score path as the named-target sibling
        :meth:`_add_inline_analytical_target`).

        **Data binding (ADR-0050 follow-up).** With a ``data = curve.exp`` key the expression is a
        *per-observation* NLL over the parameters and the bound data columns: the data files are
        loaded, their column headers join the parameters in the compile namespace (so the
        expression may reference them), and the synthesized model evaluates the expression per row
        and sums it (:meth:`ExpressionModel.execute`). A column header that collides with a declared
        parameter name is a pointed error (the symbol would be ambiguous). With no ``data`` key the
        expression is the original pure-parameter form.

        Edition-2 gated. Every free symbol must be a declared free parameter (or a bound data
        column); an undeclared symbol (a typo, or a parameter the user forgot to declare) errors
        clearly, naming it. The expression is echoed at run start. No-op unless
        ``objective = expression``."""
        if str(self.config.get('objective', '')).lower() != 'expression':
            return
        ed = edition.resolve_edition(self.config.get('edition'))
        edition.require_edition(ed, 2, "the 'objective = expression' bring-your-own objective")
        formula = self.config.get('expression')
        if not formula or not str(formula).strip():
            raise PybnfError(
                "objective = expression requires an 'expression' key",
                "You set 'objective = expression' but did not supply the expression itself. Add a "
                "companion line, e.g. 'expression = 0.5*((1 - x1)^2 + 100*(x2 - x1^2)^2)', a "
                "PEtab-math negative log-likelihood over your declared free parameters (ADR-0050).")
        formula = str(formula).strip()
        # The declared free-parameter names -- the same source _load_variables reads (legacy
        # ``(*_var, id)`` tuples + new-era ``('parameter', id)`` records). _load_models runs
        # before _load_variables, so derive the names from the config keys directly rather than
        # from self.variables (not yet built).
        declared = {k[1] for k in self.config.keys() if self._is_free_param_key(k)}
        # Load any bound experimental data and collect its column headers -- the data-binding
        # follow-up. The headers join the parameters in the compile namespace so the expression may
        # reference them; a header that shadows a declared parameter is rejected (ambiguous symbol).
        model_data = self._load_analytical_data(self.config.get('data'))
        data_columns = self._analytical_data_columns(model_data)
        clash = sorted(data_columns & declared)
        if clash:
            raise PybnfError(
                f"Data column(s) {clash} collide with declared free parameter name(s) of the same "
                f"name.",
                "An objective expression binds a symbol to either a free parameter or a data "
                "column, so the two namespaces must be disjoint; rename the parameter or the data "
                "column.")
        from .analytical_model import ExpressionModel
        from .petab.formula import compile_objective_expression
        # Compile + validate now (at config load) so an unparseable expression or an undeclared
        # symbol surfaces immediately with a pointed error -- not mid-run on a dask worker.
        _func, ordered_names = compile_objective_expression(formula, declared,
                                                            data_columns=data_columns)
        referenced_columns = [n for n in ordered_names if n in data_columns]
        if model_data and not referenced_columns:
            print1("Warning: 'data' was bound but the objective expression references no data "
                   "column; the data is unused (the expression is a pure function of the "
                   "parameters).")
        # Fail fast: every bound experiment must carry every column the expression references, so a
        # per-observation evaluation never hits a missing column mid-run on a worker.
        for exp_name, d in (model_data or {}).items():
            missing = [c for c in referenced_columns if c not in d.cols]
            if missing:
                raise PybnfError(
                    f"Experiment '{exp_name}' is missing data column(s) {sorted(missing)} that the "
                    f"objective expression references.",
                    f"Its columns are {sorted(d.cols)}. Every bound experiment must carry every "
                    f"data column the expression references (ADR-0050 data binding).")
        model = ExpressionModel(formula, ordered_names, name='expression',
                                data=model_data, data_columns=referenced_columns)
        if model.name in md:
            raise PybnfError(
                f'The inline objective expression model "{model.name}" collides with a declared '
                f'model of the same name. Rename the model file.')
        md[model.name] = model
        self._data_map[model.name] = []   # bring-your-own analytical target: no suffix-matched data
        # _check_actions reads the per-model exp list at self.config[model.file_path]; mirror the
        # named-target path's empty list (there is no file, so file_path == name).
        self.config[model.file_path] = []
        bound_params = ', '.join(model._param_names) if model._param_names else '(no free parameters)'
        bound = bound_params + (f'; data columns {sorted(data_columns)}' if referenced_columns else '')
        print1(f'Objective: bring-your-own expression NLL = {formula} [binds {bound}].')
        logger.info(f'Inline expression objective: {formula!r} binding {ordered_names}')

    def _add_inline_callable_target(self, md):
        """Synthesize the :class:`CallableModel` for a bring-your-own callable objective (ADR-0050).

        ``objective = callable`` + ``callable = mymodule:negative_log_likelihood`` points at a
        Python entry point computing the user's negative log-likelihood -- the escape hatch for
        densities the inline ``expression`` grammar cannot express (logsumexp mixtures, loops,
        ``scipy.stats``). The entry point is resolved + validated **at config load** (fail fast --
        a missing module / bad attribute / non-callable errors here, not mid-run on a dask worker)
        and the synthesized model is injected straight into the model dict ``md`` -- the same
        fileless synthesis / injection / score path as the sibling
        :meth:`_add_inline_expression_target`, swapping "compile an expression" for "import a
        function".

        Experimental data is optional and bound by the ``data = f1.exp, f2.exp`` key (ADR-0050
        data follow-up): each ``.exp`` file is loaded into a :class:`~pybnf.data.Data` and handed
        to the callable as a name->Data mapping keyed by file stem (``f(params, data)``), one entry
        per experiment -- so multi-experiment data is presented by name, mirroring ``params``. With
        no ``data`` key the callable is invoked ``data=None`` (the pure-analytical case). The
        ``data`` key's objective scoping is enforced once in :meth:`_validate_analytical_data_key`.

        Edition-2 gated. Gradient-free (a general callable is not JAX-traceable), so
        ``job_type = hmc`` rejects the resulting model with a pointed error
        (:meth:`HMCSampler._resolve_analytical_model`). No-op unless ``objective = callable``."""
        if str(self.config.get('objective', '')).lower() != 'callable':
            return
        ed = edition.resolve_edition(self.config.get('edition'))
        edition.require_edition(ed, 2, "the 'objective = callable' bring-your-own objective")
        entry_point = self.config.get('callable')
        if not entry_point or not str(entry_point).strip():
            raise PybnfError(
                "objective = callable requires a 'callable' key",
                "You set 'objective = callable' but did not supply the entry point itself. Add a "
                "companion line, e.g. 'callable = mymodule:negative_log_likelihood', a "
                "'module:func' (or 'path/to/file.py:func') reference to a Python callable "
                "f(params, data=None) -> float returning your negative log-likelihood (ADR-0050).")
        entry_point = str(entry_point).strip()
        from .analytical_model import CallableModel, resolve_callable_entry_point
        # Resolve + validate now (at config load) so a bad reference surfaces immediately with a
        # pointed error -- not mid-run on a dask worker. (The resolved function is re-imported
        # lazily on the worker; here we only fail-fast on a broken reference.)
        resolve_callable_entry_point(entry_point)
        # Load the bound experimental data (if any) into a name->Data map keyed by file stem -- one
        # entry per experiment, the multi-experiment presentation handed to the callable. Loaded
        # eagerly here (fail fast on a missing/unreadable file) and carried on the model to the
        # workers, NOT routed through self.exp_data: a callable scores its own NLL, so its data is
        # never suffix-matched to a model action (DirectPassObjective ignores exp_data entirely).
        model_data = self._load_analytical_data(self.config.get('data'))
        model = CallableModel(entry_point, name='callable', data=model_data)
        if model.name in md:
            raise PybnfError(
                f'The inline objective callable model "{model.name}" collides with a declared '
                f'model of the same name. Rename the model file.')
        md[model.name] = model
        self._data_map[model.name] = []   # bring-your-own analytical target: no experimental data
        # _check_actions reads the per-model exp list at self.config[model.file_path]; mirror the
        # named-target/expression path's empty list (there is no file, so file_path == name). This
        # also repurposes the 'callable' config slot (which held the entry-point string, already
        # captured on the model) as that empty exp list.
        self.config[model.file_path] = []
        if model_data:
            bound = ', '.join(sorted(model_data))
            print1(f'Objective: bring-your-own callable NLL = {entry_point} '
                   f'[data: {bound}].')
        else:
            print1(f'Objective: bring-your-own callable NLL = {entry_point} '
                   f'(params-only; no data bound).')
        logger.info(f'Inline callable objective: {entry_point!r} '
                    f'with data {sorted(model_data) if model_data else None}')

    def _validate_analytical_data_key(self):
        """The ``data = f.exp, ...`` key binds measurements to a bring-your-own *analytical*
        objective (``expression`` or ``callable``); any other objective binds data through a model /
        experiment, so a stray ``data`` key there is a misconfiguration. Enforce that scoping once,
        with a pointed error, before either synthesis path consumes the key (ADR-0050 data
        follow-up)."""
        data_files = self.config.get('data')
        if not data_files:
            return
        objective = str(self.config.get('objective', '')).lower()
        if objective not in ('expression', 'callable'):
            raise PybnfError(
                "the 'data' key is only valid with 'objective = expression' or "
                "'objective = callable'",
                "You declared 'data = %s' but the objective is %r. A bring-your-own analytical "
                "objective scores its own measurements (ADR-0050); other objectives bind "
                "experimental data through a model / experiment, not the top-level 'data' key."
                % (', '.join(map(str, data_files)), self.config.get('objective')))

    def _load_analytical_data(self, data_files):
        """Load the ``data = f1.exp, ...`` files for a bring-your-own analytical objective
        (``expression`` or ``callable``) into a name->Data map keyed by file stem (ADR-0050 data
        follow-up), or ``None`` when no data is declared.

        Each ``.exp`` file is one experiment; the stem (``curve1.exp`` -> ``curve1``) is its name,
        so the objective indexes its measurements by experiment exactly as it indexes parameters by
        name. Loaded eagerly (a missing/unreadable file is a pointed config-load error, not a
        mid-run worker failure); two files sharing a stem collide with a pointed error (the name
        would be ambiguous). Returns ``None`` (not an empty dict) for the no-data case so the
        ``data=None`` default and the pure-analytical contract are preserved exactly."""
        if not data_files:
            return None
        model_data = {}
        for ef in data_files:
            key = self._file_prefix(ef)
            if key in model_data:
                raise PybnfError(
                    f"Two analytical-objective data files map to the same experiment name '{key}'.",
                    "An analytical objective's data files are keyed by their stem (the name before "
                    "'.exp'), so each must have a distinct stem; rename one of the colliding files.")
            try:
                model_data[key] = Data(file_name=self._absolute(ef))
            except FileNotFoundError:
                raise PybnfError(f"Analytical objective data file '{ef}' was not found.")
        return model_data

    @staticmethod
    def _analytical_data_columns(model_data):
        """The set of data-column headers across all bound experiments (the symbols an
        ``objective = expression`` may reference in addition to the free parameters). Empty when no
        data is bound."""
        if not model_data:
            return set()
        columns = set()
        for d in model_data.values():
            columns.update(d.cols)
        return columns

    def _load_mutants(self):

        if 'mutant' not in self.config:
            return

        for base, name, mutations, exps in self.config['mutant']:
            base = self._file_prefix(base, '(bngl|xml|ant)')
            if base not in self.models:
                raise PybnfError(f'Mutant {name} declared corresponding to model {base}, but that model was not found')
            mut_objects = [Mutation(var, op, float(val)) for var, op, val in mutations]
            mut_set = MutationSet(mut_objects, name)
            self.models[base].add_mutant(mut_set)
            # Check that the exp files will have simulation outputs
            for ex in exps:
                ename = self._file_prefix(ex, '(exp|con|prop)')
                base_suffix = re.match(f'.*(?={name})', ename)
                suffix_choices = [x[1] for x in self.models[base].suffixes]
                if len(suffix_choices) == 0:
                    raise PybnfError(f"Model {base} has no action suffixes, so I can't have mutant model {name} with "
                                     f"data file {ex} based on that model")
                if not base_suffix or base_suffix.group(0) not in suffix_choices:
                    raise PybnfError(f'Experimental file name {ex} in mutant model {name}. This file name should consist of '
                                     f'the model suffix it corresponds to, followed by the mutant name (e.g. {suffix_choices[0]}{name}.exp)')
            # Stages these exp files to get loaded along with regular model ones
            self._data_map[base] += exps

    def _load_conditions(self):
        """Map new-era ``condition:`` lines to MutationSets on the base model (ADR-0028).

        A ``condition:`` is a named set of parameter perturbations -- a PyBNF Mutant = a
        PEtab Condition -- i.e. the *perturbation half* of a legacy ``mutant``, with **no
        data binding** (data is introduced only by an experiment's ``data:``, Chunk 3).
        So this reuses ``_load_mutants``' asset (``Mutation`` / ``MutationSet`` /
        ``add_mutant``) but skips the legacy ``: exps`` suffix-matching and ``_data_map``
        staging that couple a mutant to its data.

        Edition-gated (``>= 2``): the parser accepts ``condition:`` regardless, so the
        error is an explanatory ``require_edition`` rather than a parse failure. The base
        model is the single declared model when ``model:`` is omitted, or the named model
        (resolved by filename stem) otherwise; under multiple models a ``model:`` ref is
        required (the multi-model end-to-end path is exercised once the multi-model
        exporter lands -- ADR-0027/0028).
        """
        # Free parameters a condition perturbation *references by value* (a per-condition
        # estimated initial condition, ADR-0076): they bind no model entity of their own, so
        # the new-era typo check (_check_variable_correspondence_modern) treats them as
        # nuisances -- legitimate, condition-wired free parameters, not typos.
        self._condition_free_params = set()
        conditions = [(k[1], v) for k, v in self.config.items()
                      if isinstance(k, tuple) and k[0] == 'condition']
        if not conditions:
            return
        ed = edition.resolve_edition(self.config.get('edition'))
        edition.require_edition(ed, 2, "the 'condition:' syntax")
        for name, (model_ref, perts) in conditions:
            if model_ref is not None:
                base = self._file_prefix(model_ref, '(bngl|xml|ant|target)')
                if base not in self.models:
                    raise PybnfError(
                        f"Condition '{name}' references model '{model_ref}', but no model "
                        f"with id '{base}' was declared.")
            elif len(self.models) == 1:
                base = next(iter(self.models))
            else:
                raise PybnfError(
                    f"Condition '{name}' does not name a model, but the job declares "
                    f"{len(self.models)} models. Add 'model: <file>' to the condition to "
                    f"say which model it perturbs.")
            mut_objects = [self._build_condition_mutation(name, var, op, val)
                           for var, op, val in perts]
            self._condition_free_params.update(
                m.value for m in mut_objects if getattr(m, 'is_param_ref', False))
            self.models[base].add_mutant(MutationSet(mut_objects, name))
            logger.debug(f"Condition '{name}' applied to model '{base}' "
                         f"({len(mut_objects)} perturbation(s))")

    @staticmethod
    def _build_condition_mutation(cond_name, var, op, val):
        """Build a :class:`~pybnf.pset.Mutation` for one condition perturbation, routing a
        BNGL **species pattern** target (contains ``(``) to a ``setConcentration`` species
        perturbation and a bare identifier to a ``setParameter`` parameter perturbation (#474).

        A species perturbation keeps its value UNEVALUATED (a number or a param-expression
        string like ``IGF1_cold_conc*(NA*Vecf)``) for inline emission; only ``=`` (an absolute
        setConcentration) is meaningful for a species amount. A parameter perturbation is the
        pre-#474 behaviour: ``op`` in ``= * / + -`` with a float value -- OR, when the value is
        a bare identifier rather than a number, a **parameter reference**: the condition sets
        ``var`` to the current fit value of the named free parameter (a per-condition estimated
        initial condition, PEtab's parameter-valued condition ``targetValue``, ADR-0076),
        resolved from the PSet at apply time (:meth:`~pybnf.pset.Mutation.amount`)."""
        if '(' in var:
            if op != '=':
                raise PybnfError(
                    f"Condition '{cond_name}': species perturbation \"{var}\" {op} {val} uses a "
                    f"relative operator '{op}', but a species amount (setConcentration) supports "
                    "only '=' (an absolute set -- a wash to 0, or a bolus/amount). Use '='.")
            return Mutation(var, op, val, is_species=True)
        try:
            return Mutation(var, op, float(val))
        except (TypeError, ValueError):
            return Mutation(var, op, val, is_param_ref=True)

    def _load_experiments(self):
        """Map new-era ``experiment:`` lines to synthesized actions + exp_data (ADR-0028).

        An ``experiment:`` is a named simulation bound to its measurement files -- a PEtab
        v2 Experiment. The experiment NAME replaces the legacy BNGL Suffix as the
        simulation's identity: it is the synthesized action's suffix AND the ``exp_data``
        key. Unlike the legacy filename->suffix convention, the data<->simulation link is
        *stated* here, so the suffix-match wart in ``_check_actions`` does not apply (a
        new-era model carries no data on its model line, so ``_check_actions`` already
        passed it with an empty mapping, which this method then extends).

        For each experiment this:

        * resolves the base model (the single declared model when ``model:`` is omitted, or
          the named model by filename stem otherwise);
        * reads the ``data:`` files and **stacks replicates** -- multiple files become one
          ``Data`` whose rows are concatenated, NOT averaged (averaging is *smoothing*, a
          different axis; the objective sums over every row, so replicate rows simply
          contribute more measurement terms -- the thing the legacy surface cannot do);
        * infers the simulation type from the data's independent variable (``time`` =>
          time_course; anything else names a swept parameter => parameter_scan) unless
          ``type:`` states it;
        * synthesizes the action with the data's independent-variable points as explicit
          output points (Chunk 3a) -- a ``TimeCourse`` over the data's time grid, or a
          ``ParamScan`` over the data's dose grid (ADR-0046) -- so the simulation lands on
          exactly the data and the objective's by-indvar match always succeeds, with no
          hand-tuned uniform grid;
        * attaches the action to the model and registers the stacked ``Data`` under the
          experiment's data key (the experiment name, or name+condition when a condition
          is applied -- the conditioned simulation output's suffix) in ``self.exp_data``
          and ``self.mapping``.

        Edition-gated (``>= 2``): the parser accepts ``experiment:`` regardless, so the
        error is an explanatory ``require_edition`` rather than a parse failure.

        A **parameter_scan** (dose-response) experiment runs each dose to steady state by
        default (no ``t_end:``), mapping to PEtab ``time=inf`` (ADR-0046); an optional
        ``t_end:`` fixes a finite endpoint. ``.con``/``.prop`` (BPSL) files in an
        experiment's ``data:`` are split off by :meth:`_partition_experiment_data` and loaded
        as constraints bound to the experiment's simulation (:meth:`_load_experiment_constraints`);
        they are PyBNF-native, so the PEtab exporter refuses an experiment carrying them.
        """
        # Parameter ids referenced only as a row-varying per-measurement noise token (the
        # ADR-0045 binding table) -- recognized as used by the free-parameter orphan check
        # (_check_variable_correspondence_modern), like the measurement-layer nuisances. Always
        # defined (even with no experiments) so the check can union it unconditionally.
        self._per_measurement_free_params = set()
        # experiment name -> (base model, data_key) for the new-era per-experiment
        # normalization override (`normalization <experiment>.<observable> = <type>`,
        # ADR-0053): users author by experiment NAME, but the exp_data/sim suffix is the
        # data_key (name, or name+condition for a conditioned experiment), so
        # _postprocess_normalization resolves the qualifier through this map. Always
        # defined (even with no experiments) so the resolver can read it unconditionally.
        self._experiment_data_keys = {}
        experiments = [(k[1], v) for k, v in self.config.items()
                       if isinstance(k, tuple) and k[0] == 'experiment']
        if not experiments:
            return
        ed = edition.resolve_edition(self.config.get('edition'))
        edition.require_edition(ed, 2, "the 'experiment:' syntax")

        # Output-row counts (suffix -> n_points - 1) for the synthesized actions, merged
        # into time_length by _load_t_length (the actions live neither in the model file
        # nor the legacy time_course list, so find_t_length / the xml branch miss them).
        self._experiment_time_length = {}

        # Conditions a pre-equilibration experiment CONSUMES (applies inline as setParameter,
        # ADR-0052): base model -> set of condition suffixes. Removed from each model's mutant
        # list after the loop so they do not also run as redundant separate simulations.
        consumed_conditions = {}
        # Conditions used the ordinary way (a regular conditioned experiment's mutant suffix),
        # so a consumed one cannot also be a live mutant: base -> set of condition suffixes.
        regular_conditions = {}

        for name, fields in experiments:
            base = self._resolve_experiment_model(name, fields.get('model'))
            model = self.models[base]
            exp_files, constraint_files = self._partition_experiment_data(name, fields['data'])
            preequilibrate = fields.get('preequilibrate')
            data_key = self._resolve_experiment_data_key(
                name, model, base, fields.get('condition'), preequilibrate)
            self._experiment_data_keys[name] = (base, data_key)
            method = fields.get('method', 'ode')
            if preequilibrate is None and fields.get('condition') is not None:
                regular_conditions.setdefault(base, set()).add(fields['condition'])

            if exp_files:
                # The common case: quantitative .exp data drives the simulation grid (and the
                # objective). A constraint, if any, rides the same simulation (below).
                stacked, replicate_lengths = self._load_experiment_data(name, exp_files)
                if fields.get('measurement_params'):
                    self._attach_measurement_params(
                        name, stacked, fields['measurement_params'], replicate_lengths)
                action_type = self._infer_experiment_type(name, stacked, fields.get('type'))
                points = sorted({float(x) for x in stacked[stacked.indvar]})
                if preequilibrate is not None:
                    # New-era pre-equilibration (ADR-0052, #440): equilibrate UNDER the named
                    # condition (unmeasured, to steady state), perturb to the measurement
                    # condition, then measure -- one simulation, two phases, state carried over.
                    # The measured phase is a time course OR (#474) a parameter_scan (a
                    # preincubate->wash->dose-scan protocol); the scan sweeps the data's
                    # independent-variable column (stacked.indvar).
                    action = self._build_preequilibration_action(
                        name, model, base, method, points, action_type, fields,
                        consumed_conditions, indvar=stacked.indvar)
                elif action_type == 'steady_state':
                    # Steady-state measurement (ADR-0086, #521): the data's only time is
                    # ``inf``, so there is no output grid -- the run relaxes to equilibrium
                    # (BNGL ``steady_state=>1`` / bngsim's early-stop on ||dx/dt|| / the
                    # RoadRunner steady-state solve) and the objective scores that final
                    # state. ``t_end:`` bounds the relaxation (default 1e6); it is a
                    # max-time bound here, not a readout time.
                    action = self._steady_state_action(name, method, fields)
                elif action_type == 'time_course':
                    action = TimeCourse({'suffix': name, 'method': method}, explicit_points=points)
                    self._attach_nf_options(action, fields, method)
                else:
                    # parameter_scan / dose-response (ADR-0046): the data's independent-variable
                    # column (col 0) is the swept parameter -- its values are the doses, fed to the
                    # scan as explicit_points (par_scan_vals). A `t_end:` field fixes the endpoint
                    # (a finite PEtab measurement time); with none the scan runs each dose to
                    # STEADY STATE (the new-era default, mapping to PEtab time=inf). steady_state=1
                    # requests bngsim's KINSOL solve with the parity fallback (ss_method=>"newton",
                    # not user-exposed); that execution and its warn-and-score-last-value policy on
                    # non-convergence already live in bngsim_model -- the only new wire is requesting it.
                    scan = {'suffix': name, 'method': method, 'param': stacked.indvar}
                    t_end = fields.get('t_end')
                    if t_end is not None:
                        scan['time'] = t_end
                    else:
                        scan['steady_state'] = 1
                    action = ParamScan(scan, explicit_points=points)
                    self._attach_nf_options(action, fields, method)
                model.add_action(action)
                self.exp_data.setdefault(base, {})[data_key] = stacked
                self.mapping.setdefault(base, set()).add(data_key)
                logger.debug(f"Experiment '{name}' ({action_type}) on model '{base}': "
                             f"{len(points)} point(s), data key '{data_key}', "
                             f"{len(exp_files)} replicate file(s)")
            else:
                # Constraint-only experiment (data: is .con/.prop only -- ADR-0028 addendum).
                # There is no measurement grid to derive, so the experiment carries its own
                # timing on the experiment line (`t_end:` + optional `n_steps:`) -- the new-era
                # home for the simulation the legacy model kept in its `begin actions` block.
                # A uniform-grid TimeCourse is synthesized and run; no exp_data/mapping entry is
                # registered (there is no quantitative data to score), but the action still runs
                # (the engine simulates every model action), producing the output the
                # constraints below read.
                if preequilibrate is not None:
                    raise PybnfError(
                        f"Experiment '{name}' uses pre-equilibration (preequilibrate:) but has "
                        "no .exp measurement data. Pre-equilibration measures a time course over "
                        "the data grid after equilibrating; give it an .exp file (ADR-0052).")
                action = self._constraint_only_action(name, method, fields)
                model.add_action(action)
                logger.debug(f"Experiment '{name}' (constraint-only time_course) on model "
                             f"'{base}': t_end={action.time}, n_steps={action.stepnumber}, "
                             f"data key '{data_key}', {len(constraint_files)} constraint file(s)")

            # time_length is keyed by the bare action suffix (the experiment name); the
            # condition/mutant combinations are formed downstream (adaptive_mcmc). The output is
            # one row per explicit point for a data-driven action (a time course: one per sample
            # time including the forced t=0; a scan: one per dose), or n_steps+... for a uniform
            # grid -- ``Action.output_length`` gives the row count either way.
            self._experiment_time_length[name] = action.output_length() - 1

            # BPSL constraints (.con/.prop) ride along with this experiment's simulation
            # (ADR-0028 addendum): each binds to the experiment's own simulation output
            # (base_model = the experiment's model, base_suffix = its data_key), so a bare
            # observable in the constraint resolves to this experiment's sim columns --
            # inheriting the experiment's condition and model -- and the objective adds the
            # penalty alongside the .exp terms. They are PyBNF-native (no PEtab v2 shape), so
            # the exporter refuses an experiment carrying them.
            for cf in constraint_files:
                self._load_experiment_constraints(name, base, data_key, cf)

        # Remove the conditions that pre-equilibration experiments consumed (applied inline as
        # setParameter, ADR-0052) from each model's mutant list, so they do not also run as
        # redundant separate simulations. A condition cannot be both consumed and a live mutant:
        # if one is also used as a regular conditioned experiment's measurement condition, that
        # is ambiguous -- raise.
        for base, conds in consumed_conditions.items():
            clash = conds & regular_conditions.get(base, set())
            if clash:
                raise PybnfError(
                    f"Condition(s) {sorted(clash)} are used both for pre-equilibration "
                    "(applied inline as setParameter) and as a regular experiment's "
                    "'condition:' (a separate mutant simulation) on the same model. A "
                    "condition cannot be both; use distinct conditions (ADR-0052).")
            model = self.models[base]
            model.mutants = [m for m in model.mutants if m.suffix not in conds]

    def _preequilibration_perturbations(self, exp_name, model, condition_name):
        """Absolute ``(kind, name, value)`` perturbations for a pre-equilibration phase, read
        from a named condition's ``MutationSet`` (ADR-0052, #474).

        A pre-equilibration condition is applied INLINE (a mid-protocol change), so each of its
        mutations must resolve to an absolute (``=``) value; a relative op (``* / + -``) needs
        the nominal value and is deferred, so it raises a clear message here. ``kind`` is
        ``'param'`` for a parameter target (emitted as ``setParameter``, what receptor's
        ``Ligand_isPresent = 1`` uses) or ``'species'`` for a BNGL species pattern target
        (emitted as ``setConcentration`` -- a wash / bolus, #474); a species ``value`` may be a
        number or a param-expression string (``IGF1_cold_conc*(NA*Vecf)``), kept unevaluated."""
        mut_set = next((m for m in model.mutants if m.suffix == condition_name), None)
        if mut_set is None:
            raise PybnfError(
                f"Experiment '{exp_name}' references condition '{condition_name}', but no "
                f"condition with that name is defined. Define it with a 'condition:' line.")
        perts = []
        for mut in mut_set.mutations:
            if getattr(mut, 'is_param_ref', False):
                raise PybnfError(
                    f"Experiment '{exp_name}': pre-equilibration condition '{condition_name}' "
                    f"sets '{mut.name}' to the value of free parameter '{mut.value}'. A "
                    f"parameter-valued condition perturbation (a per-condition estimated initial "
                    f"condition, ADR-0076) is not yet supported inline in a pre-equilibration "
                    f"phase; use it in the measurement 'condition:' of an ordinary experiment.")
            if mut.operation != '=':
                raise PybnfError(
                    f"Experiment '{exp_name}': pre-equilibration condition '{condition_name}' "
                    f"uses a relative perturbation ('{mut.name} {mut.operation} {mut.value}'). "
                    "Pre-equilibration currently supports only absolute ('=') perturbations; "
                    "use '=' (ADR-0052).")
            kind = 'species' if getattr(mut, 'is_species', False) else 'param'
            perts.append((kind, mut.name, mut.value))
        return perts

    @staticmethod
    def _steady_state_action(name, method, fields):
        """The measured ``TimeCourse`` for a steady-state experiment (ADR-0086, #521): a
        relaxation to equilibrium whose final row is scored against the data's ``time =
        inf`` row(s).

        ``t_end:`` is the max-time BOUND on the relaxation (as it is for a steady-state
        parameter_scan, ADR-0046), not a readout time; omitted, it defaults to bngsim's own
        ``steady_state(max_time=1e6)`` bound.

        Deterministic (``ode``) only, the same boundary the steady-state parameter_scan
        draws: a stochastic trajectory has a stationary *distribution*, not a fixed point
        that ``||dx/dt|| -> 0`` detects, and NFsim has no steady-state solve at all
        (ADR-0052). Refuse rather than silently integrate to the bound."""
        if method != 'ode':
            raise PybnfError(
                f"Experiment '{name}' measures a steady state (its data's time is inf), but "
                f"its method is '{method}'. A steady state is a deterministic fixed point "
                "(the relaxation early-stops on ||dx/dt||), so it needs 'method: ode' -- a "
                "stochastic run has a stationary distribution, not a fixed point, and NFsim "
                "has no steady-state solve. Use 'method: ode', or measure a finite time "
                "course instead.")
        tc = {'suffix': name, 'method': method, 'steady_state': 1}
        if fields.get('t_end') is not None:
            tc['time'] = fields['t_end']
        return TimeCourse(tc)

    @staticmethod
    def _attach_nf_options(action, fields, method):
        """Carry the optional NFsim options (``gml:``, ``complex:``) from an experiment's fields
        onto its synthesized action, for a ``method: nf`` experiment. No-op off the NF path, so
        an ODE/SSA action is unchanged. :meth:`pybnf.pset.BNGLModel` emits them into the NFsim
        simulate/parameter_scan."""
        if method != 'nf':
            return
        if fields.get('gml') is not None:
            action.gml = fields['gml']
        if fields.get('complex') is not None:
            action.complex = fields['complex']

    def _build_preequilibration_action(self, name, model, base, method, points, action_type,
                                       fields, consumed_conditions, indvar=None):
        """Build the measured action for a pre-equilibration experiment (ADR-0052, #474) and
        record the conditions it consumes.

        Reads the ``preequilibrate:`` condition (the unmeasured equilibration state) and the
        measurement ``condition:`` (optional -- omitted measures at the model default) as
        absolute inline perturbations (``setParameter`` for a parameter target, ``setConcentration``
        for a species target -- a wash / bolus, #474), attaches them to the measured action, and
        marks both conditions consumed (removed from the model's mutants after the loop).

        The measured phase is a ``TimeCourse`` over the data's time grid (ADR-0052) OR, when the
        data's independent variable is a swept parameter (``type: parameter_scan``, #474), a
        ``ParamScan`` over the data's dose grid -- the preincubate->wash->dose-scan protocol. A
        pre-equilibrated scan saves the post-intervention state and resets each dose to it
        (``saveConcentrations`` + ``reset_conc=>1``), which bngsim honors natively
        (lanl/bngsim#11); ``t_end:`` fixes the scan's measurement time (e.g. a dissociation read
        at 20/60 min), or with none each dose runs to steady state (ADR-0046)."""
        if action_type not in ('time_course', 'parameter_scan', 'steady_state'):
            raise PybnfError(
                f"Experiment '{name}' uses pre-equilibration (preequilibrate:) with unsupported "
                f"type '{action_type}'. The measured phase must be a time_course, a "
                "steady_state or a parameter_scan.")
        preequil_cond = fields['preequilibrate']
        equil_perts = self._preequilibration_perturbations(name, model, preequil_cond)
        consumed_conditions.setdefault(base, set()).add(preequil_cond)
        meas_cond = fields.get('condition')
        if meas_cond is not None:
            measure_perts = self._preequilibration_perturbations(name, model, meas_cond)
            consumed_conditions[base].add(meas_cond)
        else:
            measure_perts = []
        # NFsim has no steady-state solve, so an NF pre-equilibration cannot equilibrate to
        # steady state; it must run its equilibration phase for a FIXED time given by
        # ``equil_t_end:`` (any method may use it; ODE/SSA default to a steady-state solve).
        equil_t_end = fields.get('equil_t_end')
        if method == 'nf' and equil_t_end is None:
            raise PybnfError(
                f"Experiment '{name}' is a network-free (method: nf) pre-equilibration, but NFsim "
                "has no steady-state solve. Give the equilibration a fixed duration with "
                "'equil_t_end: <time>' (the interval to run before the measured phase).")
        if action_type == 'parameter_scan':
            # The measured phase is a dose-response scan (#474). Its endpoint is the scan's
            # measurement time (``t_end:``, e.g. the 20/60-min dissociation read); with none each
            # dose runs to steady state (ADR-0046 default). The scan resets each dose to the
            # carried post-intervention state (reset_conc=>1) -- the pre-equilibrated dissociation.
            scan = {'suffix': name, 'method': method, 'param': indvar}
            t_end = fields.get('t_end')
            if t_end is not None:
                scan['time'] = t_end
            else:
                scan['steady_state'] = 1
            action = ParamScan(scan, explicit_points=points)
        elif action_type == 'steady_state':
            # The measured phase is itself a steady state (ADR-0086, #521): equilibrate,
            # intervene, then relax to the NEW equilibrium and score that. ``t_end:`` bounds
            # the measured relaxation (the equilibration phase keeps its own bound).
            action = self._steady_state_action(name, method, fields)
        else:
            action = TimeCourse({'suffix': name, 'method': method}, explicit_points=points)
        action.set_preequilibration(equil_perts, measure_perts, equil_fixed_time=equil_t_end)
        self._attach_nf_options(action, fields, method)
        return action

    def _resolve_experiment_model(self, name, model_ref):
        """Resolve an experiment's base model: the single declared model when ``model:`` is
        omitted, or the named model (by filename stem) otherwise. Mirrors the condition
        resolution idiom (ADR-0028)."""
        if model_ref is not None:
            base = self._file_prefix(model_ref, '(bngl|xml|ant|target)')
            if base not in self.models:
                raise PybnfError(
                    f"Experiment '{name}' references model '{model_ref}', but no model "
                    f"with id '{base}' was declared.")
            return base
        if len(self.models) == 1:
            return next(iter(self.models))
        raise PybnfError(
            f"Experiment '{name}' does not name a model, but the job declares "
            f"{len(self.models)} models. Add 'model: <file>' to the experiment to say "
            "which model it simulates.")

    def _resolve_experiment_data_key(self, name, model, base, condition, preequilibrate=None):
        """The exp_data/sim-output key for an experiment: the experiment name for a
        wildtype experiment, or name+condition for a conditioned one (the suffix the
        conditioned simulation output carries -- action suffix + the condition's
        MutationSet suffix). Validates that the named condition exists on the model.

        Under pre-equilibration (``preequilibrate:`` set -- ADR-0052), BOTH the
        pre-equilibration condition and the measurement ``condition:`` are applied INLINE as
        ``setParameter`` actions on the single measured (base) simulation, not as a separate
        mutant. So the data key is just the experiment name (the conditions are validated and
        consumed in :meth:`_load_experiments`, not turned into a mutant suffix here)."""
        if preequilibrate is not None:
            return name
        if condition is None:
            return name
        cond_suffixes = {m.suffix for m in model.mutants}
        if condition not in cond_suffixes:
            raise PybnfError(
                f"Experiment '{name}' references condition '{condition}', but no condition "
                f"with that name is defined on model '{base}'. Define it with a "
                "'condition:' line.")
        return name + condition

    def _compute_emit_suffixes(self):
        """Record, per model, the full output suffixes any consumer actually reads --
        the emit-set (#484, ADR-0069). Populates ``self.emit_suffixes[model_name]`` only
        for edition-2 Mechanism-A models whose actions are cleanly separable; every other
        model is left out (pruning off -> byte-identical).

        Under edition-2 Mechanism A (ONE model + ``condition:`` perturbations) the single
        model runs every synthesized action under every condition mutant, but only the
        diagonal ``(action, its own condition)`` pairs are consumed. The emit-set is the
        union over three consumer channels:

        * the scored objective -- the ``exp_data`` data-keys (the diagonal);
        * constraints -- each ``ConstraintSet``'s home ``base_suffix`` (which covers a
          constraint-only experiment whose data-key is not in ``exp_data``) plus any
          producible ``suffix.observable`` cross-reference (:meth:`_constraint_ref_suffixes`);
        * postprocessing scripts -- their ``(model, suffix)`` targets (a hard direct index,
          ``Result.postprocess_data``).

        Enabled per model only when ALL hold: edition >= 2; the model received
        experiment-synthesized actions; and every action suffix equals an experiment name
        (**separability** -- no hand-written ``begin actions`` block is mixed in, so each
        synthesized action is its own reset-independent block that can be skipped
        individually). Anything else fails safe (the model is absent from
        ``self.emit_suffixes`` -> the backend guard is a no-op).

        Raises a :class:`PybnfError` if a constraint/postprocessing consumer references a
        suffix no ``action x condition`` pair produces -- turning a would-be silent drop
        into an actionable load-time error.
        """
        self.emit_suffixes = {}
        ed = edition.resolve_edition(self.config.get('edition'))
        if ed < 2:
            return
        # Experiment names synthesized as actions onto each model (= the action suffixes).
        exp_names_by_model = {}
        for name, (base, _dk) in self._experiment_data_keys.items():
            exp_names_by_model.setdefault(base, set()).add(name)
        for base, model in self.models.items():
            if base not in exp_names_by_model:
                continue  # no experiment-synthesized actions -> nothing to prune
            action_suffixes = {s[1] for s in model.suffixes}
            if action_suffixes != exp_names_by_model[base]:
                continue  # a begin-actions block is mixed in -> not separable, fail safe
            producible = set(model.get_suffixes())
            emit = set(self.exp_data.get(base, {}).keys())
            for cs in self.constraints:
                if cs.base_model != base:
                    continue
                emit.add(cs.base_suffix)
                for c in cs.constraints:
                    emit |= self._constraint_ref_suffixes(c, producible)
            emit |= {suff for (m, suff) in self.postprocessing if m == base}
            missing = emit - producible
            if missing:
                raise PybnfError(
                    f"Model '{base}': a constraint or postprocessing script references output "
                    f"suffix(es) {sorted(missing)} that no experiment x condition pair produces. "
                    f"Add the missing experiment (e.g. \"experiment: <name>, condition: <cond>\") "
                    f"so the pair is simulated (lanl/PyBNF#484).")
            self.emit_suffixes[base] = emit

    @staticmethod
    def _constraint_ref_suffixes(constraint, producible):
        """The producible output suffixes one constraint references through a dotted
        ``suffix.observable`` quantity (its two sides and any ``altpenalty`` sides). A
        bare observable (no ``.``) resolves to the constraint's own ``base_suffix`` (already
        unioned by the caller), so only dotted cross-references matter here. Restricting to
        ``producible`` suffixes keeps a stray token (e.g. a numeric bound that slipped
        through) from polluting the emit-set; a genuine dotted ref to an off-diagonal cell
        is kept alive (defensive) rather than silently pruned (#484)."""
        refs = set()
        for q in (constraint.quant1, constraint.quant2, constraint.alt1, constraint.alt2):
            if isinstance(q, str) and '.' in q:
                prefix = q.split('.', 1)[0]
                if prefix in producible:
                    refs.add(prefix)
        return refs

    @staticmethod
    def _partition_experiment_data(name, data_files):
        """Split an experiment's ``data:`` files by kind (ADR-0028): ``.exp`` measurement
        files (quantitative data, drive the simulation grid + the objective) vs ``.con``/
        ``.prop`` BPSL constraint files (qualitative data, scored as penalties). Any other
        extension is an error. Returns ``(exp_files, constraint_files)`` preserving order."""
        exp_files, constraint_files = [], []
        for ef in data_files:
            if re.search(r'\.exp$', ef):
                exp_files.append(ef)
            elif re.search(r'\.(con|prop)$', ef):
                constraint_files.append(ef)
            else:
                raise PybnfError(
                    f"Experiment '{name}' data file '{ef}': unsupported extension. A "
                    "'data:' file is either an .exp measurement file or a .con/.prop "
                    "constraint file.")
        return exp_files, constraint_files

    def _constraint_only_action(self, name, method, fields):
        """Synthesize the time-course action for a **constraint-only** experiment -- one whose
        ``data:`` is BPSL ``.con``/``.prop`` files only, with no ``.exp`` measurements
        (ADR-0028 addendum).

        With no measurement data there is no independent-variable grid to derive, and a
        constraint's times are often variable conditions (``at A=5.5`` -- "when column A
        reaches 5.5") that cannot be resolved before the trajectory exists. So the experiment
        states its own simulation timing -- the new-era home for what a legacy ``.prop`` job
        kept in the model's ``begin actions`` block: ``t_end:`` (the integration endpoint,
        required here), an optional ``t_start:`` (the integration start; default 0, like every
        config-action time course), and an optional ``n_steps:`` (the uniform output
        resolution over ``[t_start, t_end]``; default = the :class:`~pybnf.pset.TimeCourse`
        step of 1). A uniform-grid ``TimeCourse`` is returned (no ``explicit_points``); the
        constraints (loaded separately) score against its output. Neither
        ``type: parameter_scan`` (a scan's swept axis comes from data) nor
        ``type: steady_state`` (whose ``time = inf`` grid likewise comes from data, ADR-0086)
        has a constraint-only form; both are refused."""
        action_type = (fields.get('type') or 'time_course').lower()
        if action_type not in ('time_course', 'timecourse'):
            raise PybnfError(
                f"Experiment '{name}' has only constraint data (.con/.prop) and type "
                f"'{fields['type']}'. A constraint-only experiment runs a time course to a "
                "fixed endpoint; a parameter_scan's swept axis and a steady_state's "
                "'time = inf' grid can only come from .exp data.")
        t_end = fields.get('t_end')
        if t_end is None:
            raise PybnfError(
                f"Experiment '{name}' has only constraint data (.con/.prop) and no .exp to "
                "derive its simulation grid. Give it an explicit endpoint with 't_end: "
                "<time>' (and optionally 't_start: <t0>' / 'n_steps: <N>') -- the new-era home "
                "for the timing a legacy job kept in the model's 'begin actions' block.")
        d = {'suffix': name, 'method': method, 'time': t_end}
        t_start = fields.get('t_start')
        if t_start is not None:
            if float(t_start) >= float(t_end):
                raise PybnfError(
                    f"Experiment '{name}': t_start ({t_start}) must be less than t_end ({t_end}).")
            d['t_start'] = t_start
        span = float(t_end) - float(t_start if t_start is not None else 0.0)
        n_steps = fields.get('n_steps')
        if n_steps is not None:
            if float(n_steps) < 1:
                raise PybnfError(
                    f"Experiment '{name}': 'n_steps' must be a positive integer (got {n_steps}).")
            d['step'] = span / float(n_steps)
        return TimeCourse(d)

    def _load_experiment_constraints(self, name, base, data_key, constraint_file):
        """Load a BPSL constraint file (``.con``/``.prop``) bound to a new-era experiment
        (ADR-0028 addendum).

        The constraint reads the experiment's *own* simulation output: ``base_model`` is the
        experiment's resolved model and ``base_suffix`` is the experiment's ``data_key`` (the
        experiment name, or name+condition when a condition is applied -- the suffix the
        conditioned simulation output carries). So a bare observable in the constraint file
        resolves to ``sim_data_dict[base][data_key]`` -- this experiment's simulation --
        inheriting the experiment's condition and model with no extra binding syntax. The
        resulting :class:`~pybnf.constraint.ConstraintSet` joins ``self.constraints``, so the
        objective adds its penalty alongside the experiment's quantitative ``.exp`` terms
        (:meth:`pybnf.objective.ObjectiveFunction.evaluate_multiple`). BPSL has no core-PEtab
        representation, so the PEtab v2 exporter refuses an experiment carrying it."""
        cs = ConstraintSet(base, data_key)
        try:
            cs.load_constraint_file(constraint_file, scale=self.config['constraint_scale'],
                                    qualitative_loss=self.config['qualitative_loss'],
                                    qualitative_scale=self._qualitative_scale_param())
        except FileNotFoundError:
            raise PybnfError(
                f"Constraint file {constraint_file} for experiment '{name}' was not found.")
        self.constraints.add(cs)

    def _load_experiment_data(self, name, data_files):
        """Read an experiment's ``.exp`` ``data:`` files and stack their replicates.

        Returns ``(stacked_data, replicate_lengths)``; the row counts retain the block boundary
        needed to align a replicate-aware ``measurement_params`` sidecar (ADR-0083). A single
        file's Data is returned as-is. Multiple files are **replicates**: their rows are
        vertically concatenated into one ``Data`` (NOT averaged -- the objective sums over
        all rows, so duplicate-indvar rows from replicates add measurement terms).
        *Ragged* replicates -- replicate files that measure different subsets of observables
        (issue #494) -- are stacked onto the **union** of their columns via
        :meth:`_stack_replicates`, so ``_SD`` noise columns still ride through intact
        (ADR-0021). ``data_files`` is the ``.exp`` subset (constraint ``.con``/``.prop``
        files are split off by :meth:`_partition_experiment_data` and loaded separately); a
        stray non-``.exp`` here is an internal-consistency error."""
        datas = []
        for ef in data_files:
            if not re.search(r'\.exp$', ef):
                raise PybnfError(
                    f"Experiment '{name}' data file '{ef}': expected an .exp measurement "
                    "file (constraint .con/.prop files are routed separately).")
            try:
                datas.append(Data(file_name=ef))
            except FileNotFoundError:
                raise PybnfError(f"Experimental data file {ef} for experiment '{name}' was not found.")
            except DuplicateColumnError as err:
                raise PybnfError(f"Parsing data file {ef} for experiment '{name}'. {err.args[0]}")
        replicate_lengths = [d.data.shape[0] for d in datas]
        if len(datas) == 1:
            return datas[0], replicate_lengths
        return self._stack_replicates(datas), replicate_lengths

    @staticmethod
    def _stack_replicates(datas):
        """Vertically stack replicate ``.exp`` grids into one ``Data``, padding ragged
        replicates (issue #494).

        Replicates that measure **different** subsets of observables -- e.g. a PEtab
        measurement table with ragged ``replicateId`` coverage, which reconstructs to
        per-replicate ``.exp`` files with different column sets (ADR-0039) -- are stacked
        onto the **union** of every file's columns, ``NaN``-filling the cells a replicate
        does not measure. The objective skips ``NaN`` exp cells (#479), so a padded column
        scores over its measured points only and the fit is unchanged (PyBNF sums residuals
        over all rows regardless of which file each lived in). Columns follow first-appearance
        order across the files, so the independent variable (column 0 of every ``.exp``)
        stays first; a **homogeneous** replicate set (every file the same columns in the same
        order) stacks byte-identically to a plain ``vstack``, matching by column *name* so a
        reordered-but-equal-columns set stacks correctly too.
        """
        union = []
        for d in datas:
            for j in range(d.data.shape[1]):
                header = d.headers[j]
                if header not in union:
                    union.append(header)
        col_index = {header: i for i, header in enumerate(union)}
        blocks = []
        for d in datas:
            block = np.full((d.data.shape[0], len(union)), np.nan)
            for j in range(d.data.shape[1]):
                block[:, col_index[d.headers[j]]] = d.data[:, j]
            blocks.append(block)
        stacked = Data()
        stacked.cols = dict(col_index)
        stacked.headers = {i: header for header, i in col_index.items()}
        stacked.indvar = datas[0].indvar
        stacked.data = np.vstack(blocks)
        return stacked

    def _attach_measurement_params(self, name, stacked, mp_file, replicate_lengths):
        """Attach an experiment's per-measurement binding table to its exp ``Data`` (ADR-0045).

        Reads the ``measurement_params: <file>.tsv`` sidecar -- ``{column: {placeholder:
        {key: token}}}`` -- and rebuilds it onto ``stacked.measurement_params`` as
        ``{column: {placeholder: [token_per_row]}}`` aligned to ``stacked``'s row order (so a
        :class:`~pybnf.noise.PerMeasurementFormulaSigma` can read the row's token at
        ``exp_row``). In the ADR-0083 five-column format ``key`` is ``(replicate, time)`` and
        ``replicate_lengths`` maps each stacked row back to the owning ``data:`` file. In a
        legacy ADR-0045 four-column sidecar ``key`` is a bare time and its token is shared by all
        replicates. A token that is a parameter id (not a number) is collected into
        ``_per_measurement_free_params`` so the free-parameter orphan check recognizes it as used.
        A ragged replicate's NaN-padded cell gets a ``None`` token without requiring a sidecar
        entry because the objective does not score that absent measurement.
        """
        from .petab._measurement_params import read_measurement_params
        raw = read_measurement_params(self._absolute(mp_file))
        row_times = list(stacked[stacked.indvar])
        if sum(replicate_lengths) != len(row_times):
            raise PybnfError(
                f"Experiment '{name}': internal replicate row counts do not match the stacked "
                "experimental data while loading measurement_params (ADR-0083).")
        row_replicates = [replicate for replicate, length in enumerate(replicate_lengths)
                          for _ in range(length)]
        binding = {}
        for column, by_placeholder in raw.items():
            binding[column] = {}
            for placeholder, by_time in by_placeholder.items():
                tokens = []
                for exp_row, (replicate, t) in enumerate(zip(row_replicates, row_times)):
                    # _stack_replicates NaN-pads a column absent from a ragged replicate. It is
                    # not a measured point and therefore needs no token for that replicate.
                    if column in stacked.cols and np.isnan(stacked.data[exp_row, stacked.cols[column]]):
                        tokens.append(None)
                    else:
                        tokens.append(self._token_at_time(
                            by_time, t, name, column, placeholder, replicate))
                binding[column][placeholder] = tokens
                for tok in tokens:
                    if tok is not None and not self._is_numeric_token(tok):
                        self._per_measurement_free_params.add(tok)
        stacked.measurement_params = binding

    @staticmethod
    def _is_numeric_token(token):
        """Whether a per-measurement binding token is a numeric literal (inlined at eval) vs a
        parameter id (resolved from the PSet -- a free-parameter nuisance)."""
        try:
            float(token)
            return True
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _token_at_time(by_time, t, name, column, placeholder, replicate=None):
        """The sidecar token for ``(replicate, time)`` with legacy bare-time fallback.

        Exact hits precede a tight-tolerance match (the .exp and sidecar times share a source,
        so tolerance only absorbs float-repr noise). Raises if no row matches.
        """
        replicate_key = (replicate, t)
        if replicate is not None and replicate_key in by_time:
            return by_time[replicate_key]
        if replicate is not None:
            for key, token in by_time.items():
                if (isinstance(key, tuple) and len(key) == 2 and key[0] == replicate
                        and np.isclose(key[1], t, rtol=0, atol=1e-9)):
                    return token
        # A four-column ADR-0045 sidecar intentionally shares a time binding across every
        # replicate. Bare-time rows in a mixed table provide the same explicit fallback.
        if t in by_time:
            return by_time[t]
        for key, token in by_time.items():
            if not isinstance(key, tuple) and np.isclose(key, t, rtol=0, atol=1e-9):
                return token
        replicate_hint = '' if replicate is None else f' in replicate {replicate + 1}'
        raise PybnfError(
            f"Experiment '{name}': the measurement_params sidecar has no token for "
            f"placeholder '{placeholder}' of column '{column}' at time={t}{replicate_hint} "
            "(ADR-0083).")

    def _infer_experiment_type(self, name, data, explicit_type):
        """Infer an experiment's simulation type from the data's independent variable
        (``time`` => time_course, or steady_state when every time is ``inf``; otherwise
        the indvar names a swept parameter => parameter_scan), unless ``type:`` states it.
        Returns ``'time_course'``, ``'steady_state'`` or ``'parameter_scan'`` (a scan runs
        each dose to steady state by default -- ADR-0046).

        A ``time`` column of ``inf`` is PEtab's steady-state measurement (ADR-0086, #521):
        the datum is the t -> infinity limit, so the experiment relaxes to equilibrium
        instead of integrating to an endpoint the (infinite) grid cannot supply."""
        n_infinite, n_points = self._infinite_indvar_count(data)
        if explicit_type is not None:
            t = explicit_type.lower()
            if t in ('time_course', 'timecourse'):
                return 'time_course'
            if t in ('parameter_scan', 'param_scan', 'parameterscan'):
                return 'parameter_scan'
            if t in ('steady_state', 'steadystate'):
                if n_infinite != n_points:
                    # A relaxation outputs no grid a finite measurement time could name, so
                    # the two statements contradict: say so here rather than at scoring time.
                    raise PybnfError(
                        f"Experiment '{name}' declares 'type: steady_state', but its data is "
                        f"not measured at steady state ({n_points - n_infinite} of "
                        f"{n_points} row(s) carry a finite time). A steady-state measurement "
                        "is written with 'time = inf' in the .exp.")
                return 'steady_state'
            raise PybnfError(
                f"Experiment '{name}' has type '{explicit_type}', which is not recognized. "
                "Use 'time_course', 'steady_state' or 'parameter_scan' (bifurcate is not "
                "supported).")
        if data.indvar is not None and data.indvar.lower() == 'time':
            if n_infinite and n_infinite == n_points:
                return 'steady_state'
            if n_infinite:
                # A steady state and a time course are two different simulations, so one
                # experiment cannot carry both grids; split them into two experiments.
                raise PybnfError(
                    f"Experiment '{name}' mixes steady-state measurements (time = inf) with "
                    f"{n_points - n_infinite} finite time(s). A steady state is a separate "
                    "simulation from a time course, so give each its own 'experiment:' (and "
                    "its own .exp).")
            return 'time_course'
        return 'parameter_scan'

    @staticmethod
    def _infinite_indvar_count(data):
        """``(n_infinite, n_points)`` over the data's independent-variable column -- how many
        rows are measured at ``+inf`` (a steady state, ADR-0086) out of how many."""
        if data.indvar is None:
            return 0, 0
        values = np.asarray([float(x) for x in data[data.indvar]])
        return int(np.count_nonzero(np.isposinf(values))), int(values.size)

    def _load_observables(self):
        """Apply new-era ``observable:`` column-header overrides (ADR-0028, Chunk 4).

        By default a ``.exp`` column header IS the model observable/function name, and the
        objective matches an experimental column to a simulation column **by name** (so the
        default needs no override). An ``observable: <entity>, column: <header>`` line is
        the opt-in override for the common case where the measured data column is named
        something other than the model entity: it renames the ``<header>`` column to
        ``<entity>`` -- and its ``<header>_SD`` per-point noise companion (ADR-0021) to
        ``<entity>_SD`` -- in every experimental ``Data``, so the by-name match succeeds and
        the fit scores the column. Without it a differently-named data column has no
        matching simulation column and the objective **raises** at eval time
        (``_check_columns``); the rename is therefore load-bearing, not cosmetic.

        The override is global -- a top-level line, not per-experiment -- so it applies
        across all experimental data; a data file that does not contain ``<header>`` is
        left unchanged (an experiment that simply does not measure that observable). A
        ``<header>`` present in **no** data file is almost always a typo and errors,
        listing the columns actually present. The independent-variable column cannot be
        remapped and an ``<entity>`` that would clobber an existing column errors -- both
        enforced by ``Data.rename_column``.

        Edition-gated (``>= 2``): the parser accepts ``observable:`` regardless, so the
        error is an explanatory ``require_edition`` rather than a parse failure. Runs after
        ``_load_experiments`` (so every experimental ``Data`` -- experiment-sourced or
        legacy ``model = X : Y.exp`` -- already exists in ``self.exp_data``) and before
        ``_load_obj_func`` (so the objective's per-observable noise / column logic sees the
        final names).
        """
        overrides = [(k[1], v) for k, v in self.config.items()
                     if isinstance(k, tuple) and k[0] == 'observable']
        if not overrides:
            return
        ed = edition.resolve_edition(self.config.get('edition'))
        edition.require_edition(ed, 2, "the 'observable:' syntax")
        all_data = [d for model_data in self.exp_data.values() for d in model_data.values()]
        for entity, header in overrides:
            found = False
            for d in all_data:
                # The observable column and its _SD noise companion are renamed
                # independently: each is present iff the data file measures (the noise of)
                # this observable, and finding either marks the override applied so a
                # genuine typo (present nowhere) still errors below.
                if header in d.cols:
                    d.rename_column(header, entity)
                    found = True
                sd = f'{header}_SD'
                if sd in d.cols:
                    d.rename_column(sd, f'{entity}_SD')
                    found = True
            if not found:
                present = sorted({c for d in all_data for c in d.cols})
                raise PybnfError(
                    f"Observable override 'observable: {entity}, column: {header}' names "
                    f"data column '{header}', but no experimental data file contains a "
                    f"column with that name (columns present: {present}). Check for a typo "
                    "in the column name.")
            logger.debug(f"Observable override: data column '{header}' -> model entity "
                         f"'{entity}' (with its _SD companion, where present)")

    def _load_measurement_models(self):
        """Build the measurement-model observation layer (ADR-0036) and attach it to the
        objective.

        A new-era ``observable: <id>, formula: <expr>`` line declares a *measurement model*:
        a PEtab ``observableFormula`` evaluated as a post-simulation transform over the output
        trajectory + the PSet (the observation layer), never by editing the model file. Each
        formula is validated against the model's expression namespace (the BNGL ``ParamList``
        via ``_bngl``; SBML species u parameters via ``_sbml`` -- ADR-0026/0036) and carried
        as a :class:`~pybnf.measurement.MeasurementModel`; fixed model constants are
        snapshotted (free parameters resolve from the PSet at eval time). The layer attaches
        to ``self.obj`` and is applied at the objective's ``evaluate_multiple`` seam before
        the by-name column match. Edition-gated (>= 2); a job with no formula line leaves
        ``self.obj.measurement`` as the no-op default.
        """
        # The free parameters any measurement model references but no model entity provides
        # (an observation-layer nuisance, e.g. an observableParameters scale -- ADR-0044);
        # surfaced to the orphan check so it is not mis-flagged as a typo. Always defined.
        self._measurement_free_params = set()
        specs = [(k[1], v) for k, v in self.config.items()
                 if isinstance(k, tuple) and k[0] == 'measurement']
        if not specs:
            return
        ed = edition.resolve_edition(self.config.get('edition'))
        edition.require_edition(
            ed, 2, "the 'observable: <id>, formula: <expr>' measurement-model syntax")

        from .measurement import MeasurementLayer, MeasurementModel, PerMeasurementModel
        from .petab.formula import compile_petab_formula, inline_assignment_rules

        namespace, constants, assignment_rules = self._model_expression_namespace()
        free_names = {v.name for v in self.variables}
        # Free parameters resolve from the PSet at eval time, not from the constant snapshot.
        constants = {n: val for n, val in constants.items() if n not in free_names}
        # A measurement-model formula may reference any model entity OR any declared free
        # parameter (resolved from the PSet at eval time, ADR-0036 §4). The model-id case was
        # always in ``namespace``; widening it to ``free_names`` admits an observation-layer
        # nuisance -- a free parameter that is *not* a model entity, e.g. an observableParameters
        # scale/offset substituted into the observableFormula (ADR-0044).
        allowed = namespace | free_names

        models = []                 # constant measurement models -> pre-materialized layer
        per_measurement = {}        # row-varying ones -> bound per data point in the objective
        for obs_id, formula in specs:
            # An SBML assignment-rule variable is declared in the model (so it would pass the
            # namespace check if it were not excluded) but is algebraically computed -- never a
            # simulation-output column and value-less -- so it cannot be resolved as a symbol at
            # fit time. Inline it instead: substitute the rule's RHS down to the species the rule
            # is defined over (recursively), so `observable: Epo_cells, formula: Epo_cells` just
            # works -- the D2D convenience observable IS an assignment rule (#465, the option-2
            # successor to #464's reconstruct-from-species rejection). A formula naming no rule
            # variable is returned verbatim; an unresolvable rule (untranslatable MathML / a
            # cyclic dependency) raises a pointed error here at load, not late in materialize.
            if assignment_rules:
                formula = inline_assignment_rules(
                    formula, assignment_rules, observable_id=obs_id)
            # A surviving per-measurement placeholder marks a row-varying scale/offset (ADR-0045):
            # its token differs per data row, so it is bound per data point, NOT pre-materialized.
            # Admit the placeholder symbol(s) to the allowed set so the rest validates fail-fast.
            placeholders = set(_MEASUREMENT_PLACEHOLDER.findall(formula))
            allowed_here = allowed | placeholders
            # Fail fast at load: parse + validate the formula's free symbols against the allowed
            # set (a pointed PybnfError on an unknown symbol / missing petab extra). The callable
            # is rebuilt lazily at eval time (dropped across pickling), so this is a validation
            # pass, not the runtime compile -- but its free-symbol list records which free
            # parameters the model reads from the PSet (the nuisances above).
            _func, names = compile_petab_formula(
                formula, allowed_here,
                detail=(f"Measurement model '{obs_id}': allowed symbols are the model's "
                        f"species/parameters/observables/functions, the fit's free "
                        f"parameters {sorted(allowed)}, and any row-varying placeholder."))
            # The placeholder tokens are per-row nuisances (collected from the sidecar, ADR-0045),
            # not always-present PSet names; only the non-placeholder free symbols are nuisances.
            self._measurement_free_params |= (set(names) - placeholders) & free_names
            if placeholders:
                per_measurement[obs_id] = PerMeasurementModel(
                    obs_id, formula, allowed_here, constants)
            else:
                models.append(MeasurementModel(obs_id, formula, allowed, constants))
        self.obj.measurement = MeasurementLayer(models)
        if per_measurement:
            self._attach_per_measurement_models(per_measurement)
        logger.debug("Built measurement-model layer with %d model(s) + %d per-measurement: %s",
                     len(models), len(per_measurement),
                     [m.observable_id for m in models] + sorted(per_measurement))

    def _attach_per_measurement_models(self, per_measurement):
        """Register the row-varying measurement models on the objective (ADR-0045).

        A :class:`~pybnf.measurement.PerMeasurementModel` is bound per data point in the
        objective's prediction step (``_prediction``), which only the per-point objfuncs (the
        :class:`~pybnf.objective.SummationObjective` family -- least-squares + the likelihoods)
        have. The column-joint ``kl`` / ``wasserstein`` (a
        :class:`~pybnf.objective.ColumnSummationObjective`) score a whole column's shape at once
        and have no per-point seam, so a row-varying observable scale has no meaning there --
        raise rather than silently drop it.
        """
        from .objective import SummationObjective
        if not isinstance(self.obj, SummationObjective):
            raise NotImplementedError(
                f"A row-varying observableParameters scale/offset (a per-measurement measurement "
                f"model, ADR-0045) is bound per data point in the objective's prediction step, "
                f"which the column-joint objective '{type(self.obj).__name__}' (kl / wasserstein) "
                f"does not have. Use a per-point objective (chi_sq, sos, laplace, ...).")
        self.obj._per_measurement_models = per_measurement

    def _load_prediction_noise(self):
        """Classify each prediction-dependent noise source and validate it (ADR-0075).

        A ``noise_model <obs> = <family>, <param> = prediction_formula <expr>`` line builds a
        :class:`~pybnf.noise.PredictionFormulaSigma`: an expression whose σ scales with the
        simulated output (a combined additive+proportional error model ``σ_abs + σ_rel * y``).
        Which of its symbols are **free-parameter coefficients** (the estimated nuisances,
        resolved from the PSet) vs **model entities** (columns read from the current
        simulation) is model-namespace dependent, so it is settled here at load -- after the
        objective and variables are built and the model namespace is available (the same source
        the measurement layer uses), before the free-parameter orphan check.

        Each formula is validated against the model namespace u the declared free parameters
        (``compile_petab_formula`` -- a pointed error on an unknown symbol / missing petab
        extra, so an undeclared coefficient is caught cleanly here rather than at eval), then
        its free-parameter symbols are recorded on the source (``set_param_names``) so they
        surface through ``required_free_noise_params`` to the orphan check as legitimate
        nuisances. Edition-gated (>= 2); a no-op when no prediction_formula source is declared.
        """
        sources = self._prediction_noise_sources()
        if not sources:
            return
        ed = edition.resolve_edition(self.config.get('edition'))
        edition.require_edition(
            ed, 2, "a prediction-dependent 'sigma = prediction_formula <expr>' noise source")
        from .petab.formula import compile_petab_formula
        namespace, _constants, _rules = self._model_expression_namespace()
        free_names = {v.name for v in self.variables}
        allowed = namespace | free_names
        for label, src in sources:
            # Validate every symbol is a model entity or a declared free parameter, and adopt the
            # compiler's canonical ordering (the callable is rebuilt lazily worker-side, dropped
            # across pickling -- this is the validation pass, ADR-0036 §5).
            _func, names = compile_petab_formula(
                src.formula, allowed,
                detail=(f"Prediction-dependent noise for {label}: allowed symbols are the "
                        f"model's species/parameters/observables/functions and the fit's free "
                        f"parameters {sorted(allowed)}."))
            param_names = set(names) & free_names       # the estimated coefficient nuisances
            model_entities = set(names) - free_names    # ⊆ namespace (compile validated it)
            if not model_entities:
                raise PybnfError(
                    f"Prediction-dependent noise for {label} references no model output "
                    f"({src.formula!r}): its σ does not depend on the simulation, so use the "
                    f"'formula' source (an expression over free parameters), not "
                    f"'prediction_formula' (ADR-0075).")
            src.set_param_names(param_names)
            src.names = names
        logger.debug("Classified %d prediction-dependent noise source(s)", len(sources))

    def _prediction_noise_sources(self):
        """The ``(label, PredictionFormulaSigma)`` pairs on the objective -- the whole-fit
        default sources and every per-observable ``noise_model`` override (ADR-0075). Empty
        for a job with no prediction_formula source (the common case)."""
        from .noise import PredictionFormulaSigma
        obj = self.obj
        found = []
        if hasattr(obj, '_default_sources'):
            for src in obj._default_sources().values():
                if isinstance(src, PredictionFormulaSigma):
                    found.append(('the whole-fit noise model', src))
        for col, (_family, srcs) in getattr(obj, 'overrides', {}).items():
            for src in srcs.values():
                if isinstance(src, PredictionFormulaSigma):
                    found.append((f"observable '{col}'", src))
        return found

    def _model_expression_namespace(self):
        """The union expression namespace + fixed-constant snapshot across the job's models,
        read from the model files directly (stdlib, simulator-free -- the same source the
        importer uses): the BNGL ``ParamList`` (parameters u observables u functions) for a
        ``.bngl`` model, species u parameters u compartments for a ``.xml`` SBML model
        (ADR-0026/0036).

        An Antimony (``.ant``) model is converted to SBML at load (the same
        ``antimony.getSBMLString`` the model loader uses) and shares the SBML branch, so its
        species/parameter namespace is available to a measurement formula exactly as for a
        ``.xml`` model -- ``.ant`` and ``.xml`` are interchangeable everywhere else, and a
        formula over a ``.ant`` model's species must not be rejected as "not a known model
        entity" just because the namespace was built by parsing the file as BNGL (#463).

        Returns ``(namespace_symbols, constants, assignment_rules)``, where ``assignment_rules``
        maps an SBML assignment-rule variable (declared as a parameter, but algebraically
        computed -- never a simulation-output column and value-less, so it cannot be resolved as
        a symbol at ``materialize``) to its rule's RHS as a PEtab-math infix string (``None`` if
        the rule's MathML was not translatable). Such a variable is **excluded** from
        ``namespace_symbols``; ``assignment_rules`` lets the loader **inline** a formula that
        references one down to the species the rule is defined over (#465)."""
        from .petab._bngl import parse_model as parse_bngl
        from .petab._sbml import parse_model as parse_sbml
        namespace = set()
        constants = {}
        assignment_rules = {}
        for mf in self.config['models']:
            text = Path(self._absolute(mf)).read_text(encoding='utf-8', errors='replace')
            if mf.endswith('.xml') or mf.endswith('.ant'):
                if mf.endswith('.ant'):
                    # Convert Antimony -> SBML text the same way the model loader does, so the
                    # SBML scanner sees the model's species/parameters (#463). A .ant model
                    # already needs the antimony package to run at all, so requiring it to
                    # validate a .ant measurement formula is consistent.
                    from .bngsim_antimony_model import _antimony_text_to_sbml_text
                    text = _antimony_text_to_sbml_text(text, self._absolute(mf))
                ent = parse_sbml(text)
                namespace |= ent.namespace_symbols
                constants.update(ent.constants)
                assignment_rules.update(ent.assignment_rules)
            else:  # .bngl -- the BNGL ParamList (parameters u observables u functions)
                ent = parse_bngl(text)
                namespace |= (set(ent.parameters) | set(ent.observable_names)
                              | set(ent.function_names))
                for name, rhs in ent.parameters.items():
                    try:
                        constants[name] = float(rhs)
                    except (TypeError, ValueError):
                        pass  # an expression-valued parameter is not a numeric constant
        return namespace, constants, assignment_rules

    def _load_simulators(self):

        model_types = set([type(m) for m in self.models.values()])

        # For each model type that exists in the run, check that the simulator is available, and pass the simulator
        # path to the appropriate Model subclass
        if BNGLModel in model_types:
            if self.config['bng_command'] == '':
                raise PybnfError('The location of the BioNetGen simulator (BNG2.pl) is not specified. Please set the '
                                 '"bng_command" configuration key to the location of the file BNG2.pl, or set the '
                                 'BNGPATH environmental variable to the folder containing BNG2.pl.\n'
                                 'If BioNetGen is not yet installed, please refer to installation instructions at '
                                 'https://lanl.github.io/PyBNF/installation.html#bionetgen')
            elif re.search(r'BNG2.pl', self.config['bng_command']) is None:
                raise PybnfError('The specified "bng_command" parameter in the configuration file must include the script '
                                 'name at the end of the path (e.g. /path/to/BNG2.pl)')
            else:  # check to make sure BNG2.pl is available
                try:
                    logger.info('Checking to make sure bng_command is appropriately set')
                    cmd = [self.config['bng_command'], '-v']
                    if os.name == 'nt':  # Windows
                        cmd = ['perl'] + cmd
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError:
                    #  Occurs on Windows if BNG2.pl is nonexistent, or on Mac/Linux if BNG2.pl exists but crashed
                    raise PybnfError('BioNetGen failed to execute. Please check that "bng_command" parameter in the '
                                     'configuration file points to the BNG2.pl script, or that the BNGPATH environmental '
                                     'variable is set to the folder containing BNG2.pl.\n'
                                     'For help, refer to '
                                     'https://lanl.github.io/PyBNF/installation.html#bionetgen')
                except FileNotFoundError:
                    #  Occurs on Mac/Linux if BNG2.pl is nonexistent.
                    raise PybnfError('The BioNetGen simulator (BNG2.pl) was not found at the specified location. Please set the '
                                     '"bng_command" configuration key to the location of the file BNG2.pl, or set the '
                                     'BNGPATH environmental variable to the folder containing BNG2.pl.\n'
                                     'If BioNetGen is not yet installed, please refer to installation instructions at '
                                     'https://lanl.github.io/PyBNF/installation.html#bionetgen')
        # The CVODE tolerances are positive, and both are bngsim-only (#546): the
        # RoadRunner backend has its own integrator settings, and the BNGL path takes
        # atol/rtol from the actions block, so accepting either key elsewhere would
        # silently do nothing.
        for tol_key in ('sbml_rtol', 'sbml_atol'):
            tol = self.config.get(tol_key)
            if tol is None:
                continue
            if self.config['sbml_backend'] != 'bngsim':
                raise PybnfError(
                    'Config option "{}" is only supported when sbml_backend = bngsim. '
                    'Current sbml_backend is "{}".'.format(tol_key, self.config['sbml_backend'])
                )
            if tol <= 0.:
                raise PybnfError(f'Config option "{tol_key}" must be a positive number; got {tol}.')
        # Check that the integrator is valid
        if self.config['sbml_backend'] == 'bngsim':
            bngsim_integrators = ('cvode', 'gillespie')
            if self.config['sbml_integrator'] not in bngsim_integrators:
                raise PybnfError(
                    'Config option "sbml_backend = bngsim" supports sbml_integrator in {}; got {}.'.format(', '.join(bngsim_integrators), self.config['sbml_integrator'])
                )
        else:
            integrators = ('cvode', 'euler', 'rk4', 'gillespie')
            if self.config['sbml_integrator'] not in integrators:
                raise PybnfError('Invalid sbml_integrator {}. Options are: {}.'.format(self.config['sbml_integrator'],
                                                                                   ', '.join(integrators)))
            if self.config['sbml_integrator'] == 'euler':
                if roadrunner.__version__ < '1.5.1':
                    raise PybnfError('Config option "sbml_integrator = euler" requires Roadrunner version 1.5.1 or higher. You '
                                     f'have version {roadrunner.__version__}')
                print1('Warning: "sbml_integrator = euler" can be numerically unstable. Confirm that your model is '
                       'producing reasonable output.')
            if self.config.get('sbml_ssa_strict', 1) != 1:
                raise PybnfError(
                    'Config option "sbml_ssa_strict" is only supported when sbml_backend = bngsim. '
                    'Current sbml_backend is "{}".'.format(self.config['sbml_backend'])
                )

    def _load_actions(self):

        for (key, ActionType) in (('time_course', TimeCourse), ('param_scan', ParamScan)):
            # Iterate through all time courses and param scans included in the config dict, create the corresponding
            # Action objects, and add them to the appropriate model(s).
            for action_dict in self.config[key]:
                if 'subdivisions' in action_dict and self.config['sbml_integrator'] != 'euler':
                    print1('Warning: Ignoring "subdivisions" setting because that is only used with sbml_integrator = '
                           'euler')
                if 'model' in action_dict:
                    action = ActionType(action_dict)
                    try:
                        # Model lookup - should work if model name included the extension or not.
                        model_key = self._file_prefix(action_dict['model'], '(bngl|xml|ant)')
                        self.models[model_key].add_action(action)
                    except KeyError:
                        raise PybnfError('{} declared for model {}, but that model was not found.'.format(key, action_dict['model']))
                else:
                    # Apply to all models (hopefully just 1)
                    if len(self.models) > 1:
                        print1(f'Warning: Applying the same {key} action to all models in this fitting run.')
                    for m in self.models:
                        self.models[m].add_action(ActionType(action_dict))

    @staticmethod
    def _file_prefix(ef, ext="exp"):
        return re.sub(r"\."+ext, "", re.split('/', ef)[-1])

    def _load_exp_data(self):
        """
        Loads experimental data files in a nested dictionary keyed on model name, then data prefix
        Also loads constraint files (which at this point are stored in the same structures as the exp files) and stores
        them in a set.
        """
        ed = {}
        csets = set()
        for m in self._data_map:
            ed[m] = {}
            for ef in self._data_map[m]:
                if re.search("exp$", ef):
                    try:
                        d = Data(file_name=ef)
                    except FileNotFoundError:
                        raise PybnfError(f'Experimental data file {ef} was not found.')
                    except DuplicateColumnError as err:
                        raise PybnfError(f'Parsing data file {ef}. {err.args[0]}')
                    ed[m][self._file_prefix(ef)] = d
                else:
                    cs = ConstraintSet(self._file_prefix(m, '(bngl|xml|ant)'), self._file_prefix(ef, '(con|prop)'))
                    try:
                        cs.load_constraint_file(ef, scale=self.config['constraint_scale'],
                                                qualitative_loss=self.config['qualitative_loss'],
                                                qualitative_scale=self._qualitative_scale_param())
                    except FileNotFoundError:
                        raise PybnfError(f'Constraint file {ef} was not found')
                    csets.add(cs)
        return ed, csets

    def _check_actions(self):
        mapping = dict()
        for model in self.models.values():
            suffs = set(model.get_suffixes())
            efs_per_m = {self._file_prefix(ef) for ef in self.config[model.file_path] if re.search(r"\.exp$", ef)}
            if not efs_per_m <= suffs:
                for ef in efs_per_m:
                    if ef not in suffs:
                        raise UnmatchedExperimentalDataError(f"Action not specified for '{ef}.exp'",
                              f"You specified that model {model.name} corresponds to data file {ef}.exp, but I can't find the "
                              f"corresponding action in the model file or config file. One of the actions in {model.file_path} "
                              f"needs to include the argument 'suffix=>\"{ef}\" ', or your config file needs to include "
                              f"an action with the suffix {ef}.")
            logger.debug(f'Model {model.name} was mapped to {efs_per_m}')
            mapping[model.name] = efs_per_m
        return mapping

    def _load_obj_func(self):
        """Build the objective function, honoring the edition-gated objective surface
        (ADR-0031).

        * **Legacy edition** (no ``edition`` / implicit edition 1): the historical
          ``objfunc`` key selects a registered objective, byte-identical to before; a
          modern key, if named, errors with the edition it needs (``require_edition``).
        * **Modern edition** (``edition >= 2``): ``objfunc`` is rejected as legacy
          syntax, and the objective is named through exactly one of the three modern
          keys -- ``objective`` (a per-point named token desugaring to a noise model, or
          the bare ``score`` passthrough), a whole-fit ``noise_model`` line, or
          ``profile_objective`` (column-joint) -- with **no implicit default**.

        A shared tail applies the whole-fit ``noise_location`` default and warns when a
        ``neg_bin`` resolves to the modern median centering implicitly (median is the
        universal default baked into every family's constructor; ADR-0031).
        """
        ed = edition.resolve_edition(self.config.get('edition'))
        user_objfunc = getattr(self, '_user_objfunc', 'objfunc' in self.config)
        has_objective = self.config.get('objective') is not None
        has_profile = self.config.get('profile_objective') is not None
        has_whole_noise = ('noise_model', None) in self.config
        has_per_obs_noise = any(isinstance(k, tuple) and k[0] == 'noise_model' and k[1] is not None
                                for k in self.config)

        if edition.is_modern(ed):
            if user_objfunc:
                raise UnknownObjectiveFunctionError(
                    'objfunc is legacy syntax',
                    f"Config key 'objfunc' is legacy (edition 1) syntax and is not available under "
                    f"edition {ed}. Name the objective with the modern surface instead: "
                    f"'objective = <name>' (a per-point noise model, or 'score'), "
                    f"'noise_model = <family>, ...' (whole-fit per-point), or "
                    f"'profile_objective = <name>' (column-joint).")
            selected = [name for name, present in
                        (('objective', has_objective), ('noise_model', has_whole_noise),
                         ('profile_objective', has_profile)) if present]
            if not selected:
                raise UnknownObjectiveFunctionError(
                    'No objective specified',
                    f"Under edition {ed} the objective must be named explicitly (there is no "
                    f"implicit default). Set exactly one of: 'objective = <name>', "
                    f"'noise_model = <family>, ...', or 'profile_objective = <name>'.")
            if len(selected) > 1:
                raise UnknownObjectiveFunctionError(
                    'Multiple objective keys specified',
                    f"Specify exactly one global objective; got {', '.join(selected)}. "
                    f"(Per-observable 'noise_model <obs> = ...' overrides are separate and may "
                    f"accompany 'objective' or a whole-fit 'noise_model'.)")
            if has_profile:
                if has_per_obs_noise:
                    raise UnknownObjectiveFunctionError(
                        'profile_objective cannot take per-observable noise_model overrides',
                        f"profile_objective = {self.config['profile_objective']} is a column-joint "
                        f"objective; per-observable 'noise_model <obs> = ...' overrides apply only to "
                        f"a per-point objective ('objective' or a whole-fit 'noise_model').")
                obj = objective.build_profile_objective(self.config, self.config['profile_objective'])
            elif has_objective:
                obj = objective.build_named_objective(self.config, self.config['objective'])
            else:
                obj = objective.build_whole_fit_noise_objective(self.config)
        else:
            # Legacy edition: a modern key, if named, must opt into the edition first
            # (require_edition raises at edition 1, naming the key and the fix).
            if has_objective:
                edition.require_edition(ed, 2, "the 'objective' key")
            if has_profile:
                edition.require_edition(ed, 2, "the 'profile_objective' key")
            if has_whole_noise:
                edition.require_edition(ed, 2, "a whole-fit 'noise_model = <family>, ...' line")
            # The historical objfunc path. Cross-config requirement check stays in
            # config (not the registry, which holds only the construction recipe):
            # neg_bin cannot be built without its r parameter.
            objfunc = self.config['objfunc']
            if objfunc == 'neg_bin' and 'neg_bin_r' not in self.config:
                raise UnknownObjectiveFunctionError("Objective function neg_bin cannot be defined "
                                                    "without configuration neg_bin_r defined")
            entry = OBJFUNC_REGISTRY.get(objfunc)
            if entry is None:
                raise UnknownObjectiveFunctionError(f"Objective function {objfunc} not defined",
                      f"Objective function {objfunc} is not defined. Valid objective function choices "
                      "are: chi_sq, chi_sq_dynamic, lognormal, laplace, sos, sod, norm_sos, "
                      "ave_norm_sos, neg_bin, neg_bin_dynamic, kl, direct_pass")
            # Uniform construction (ADR-0011): every objective builds itself from the
            # config via its from_config classmethod -- no per-objfunc recipe.
            obj = entry.cls.from_config(self.config)

        # Whole-fit default location (ADR-0024): the mean/median interpretation of the
        # prediction, applied to a per-point objective's default noise model
        # (per-observable noise_model location fields override it).
        location = self.config.get('noise_location')
        if location is not None:
            if location not in objective._NOISE_LOCATIONS:
                raise PybnfError(
                    f"noise_location must be 'mean' or 'median', not {location!r}.")
            if not isinstance(obj, objective.LikelihoodObjective):
                raise UnknownObjectiveFunctionError(
                    "noise_location is only meaningful for a likelihood (per-point noise-model) "
                    "objective (normal/lognormal/laplace/neg_bin/...); the selected objective has "
                    "no noise model whose location can be set.")
            obj.set_default_location(location)
        if edition.is_modern(ed) and isinstance(obj, objective.LikelihoodObjective):
            # The universal default centering is the median (ADR-0031), baked into every
            # family's constructor -- so no flip is needed here. The location-scale
            # families are byte-identical at the median; the one family whose legacy
            # centering was the mean is neg_bin, so a neg_bin that resolves to median
            # *implicitly* (no explicit location) is a number change from legacy and
            # almost always a forgotten 'location = mean'. Warn for it (explicit
            # mean/median is silent).
            scopes = _implicit_median_neg_bin_scopes(self.config, obj, explicit_global=location is not None)
            if scopes:
                logger.warning(
                    f"neg_bin is defaulting to median centering under edition {ed} for "
                    f"{', '.join(scopes)} (ADR-0031); its legacy default was the mean. Set "
                    f"'location = mean' (or 'noise_location = mean' for the whole fit) to keep "
                    f"mean centering, or '= median' to silence this warning.")

        # The explicit per-observable cumulative->incident prediction transform (ADR-0051,
        # #418): differences a declared column's cumulative counts to per-interval increments
        # in the objective's prediction step. Family-independent -- it attaches to any per-point
        # objective (the SummationObjective family) regardless of the noise family, the
        # generalization of neg_bin_dynamic's legacy `_Cum` substring. Bound per data point in
        # `_prediction`, which the column-joint kl / wasserstein lack, so raise rather than
        # silently drop it (mirrors `_attach_per_measurement_models`).
        cumulative_cols = objective._build_cumulative_cols(self.config)
        if cumulative_cols:
            if not isinstance(obj, objective.SummationObjective):
                raise UnknownObjectiveFunctionError(
                    'cumulative is not available for a column-joint objective',
                    f"The cumulative->incident prediction transform (#418) is applied per data "
                    f"point in the objective's prediction step, which the column-joint objective "
                    f"'{type(obj).__name__}' (kl / wasserstein) does not have. Declare 'cumulative' "
                    f"only with a per-point objective (chi_sq, sos, laplace, neg_bin, ...).")
            obj._cumulative_cols = cumulative_cols
        return obj

    def _attach_analytic_scale(self):
        """Attach the resolved analytic per-series scaling to the objective (ADR-0066, #479).

        Runs *after* ``_postprocess_normalization`` (unlike ``cumulative``, whose set is read
        straight from the raw parse keys, ``scale`` needs the resolved experiment x observable
        grid), so it keys off the compiled ``config['analytic_scale']`` (``{data_key:
        frozenset(cols)}``). The objective profiles out each declared series' optimal
        multiplicative scale at scoring time (family-appropriate -- geomean for a log family,
        least-squares otherwise), so an overall scale difference between model and arbitrary-unit
        data is not penalized. Like ``cumulative`` it rides the per-point prediction seam, so it
        needs a per-point (``SummationObjective``) objective; a column-joint kl / wasserstein has
        no such seam and is refused."""
        analytic_scale = self.config.get('analytic_scale')
        if not analytic_scale:
            return
        if not isinstance(self.obj, objective.SummationObjective):
            raise UnknownObjectiveFunctionError(
                'analytic scale is not available for a column-joint objective',
                f"Analytic per-series scaling ('normalization <obs> = scale', #479) is profiled "
                f"per data point at scoring time, which the column-joint objective "
                f"'{type(self.obj).__name__}' (kl / wasserstein) does not have. Declare 'scale' "
                f"only with a per-point objective (lognormal, chi_sq, sos, ...).")
        self.obj._analytic_scale = analytic_scale

    def _apply_noise_profiling(self):
        """Resolve ``noise_profiling = 1`` and partition the free parameters (ADR-0108, #562).

        Profiling replaces every estimated noise scale that IS a free parameter with its
        closed-form MLE at each evaluation, so those parameters are no longer *searched*: they
        are removed from :attr:`variables` (the list every algorithm builds its box, population
        and PSets from) into :attr:`profiled_variables`, and their names are handed to the
        objective, which solves for them per evaluation.

        Three things deliberately do **not** change:

        * the parameters stay **declared**. ``noise_profiling`` is a switch on an otherwise
          valid .conf, so the same file runs with and without it (and the declaration is still
          what ``required_free_noise_params`` validates against). Their bounds and prior become
          inert -- which is the point: a profiled scale has no box to run into;
        * they stay **estimated quantities**, so they keep counting in ``k`` for AIC/BIC/AICc
          (``Algorithm._compute_information_criteria``). Only the *search* drops them;
        * their fitted values are still reported -- as ``Results/profiled_noise.txt``, since a
          value the fit solves for rather than proposes is not a coordinate of the best PSet.

        Refusals (all pointed, all before the run starts): a non-per-point objective, an
        estimated scale with no closed-form profile (each reason listed by
        :meth:`~pybnf.objective.LikelihoodObjective.noise_profiling_plan`), a fit with nothing
        to profile, and a **Bayesian sampler** -- profiling maximizes the nuisance out where a
        posterior integrates it out, so a profiled sampler's draws would not be posterior draws.
        """
        self.profiled_noise_params = []
        self.profiled_variables = []
        if not self.config.get('noise_profiling'):
            return
        if not isinstance(self.obj, objective.LikelihoodObjective):
            raise UnknownObjectiveFunctionError(
                'noise_profiling needs a likelihood objective',
                f"noise_profiling = 1 profiles out an estimated noise scale, but the objective "
                f"'{type(self.obj).__name__}' has no per-point noise model to estimate one "
                f"with. Use a likelihood objective (normal / lognormal / lnnormal / laplace) "
                f"whose scale is declared 'fit <parameter>'.")
        entry = FIT_TYPE_REGISTRY.get(self.config.get('fit_type'))
        if entry is not None and entry.family == 'sampler':
            raise PybnfError(
                'noise_profiling is not available for a Bayesian sampler',
                f"noise_profiling = 1 replaces each estimated noise scale by its maximum-"
                f"likelihood value, which is a PROFILE, not a marginal: it maximizes the "
                f"nuisance out where a posterior integrates it over its prior. Draws from "
                f"'{self.config['fit_type']}' ({entry.display_name}) would therefore not be "
                f"posterior draws, and the model parameters' credible intervals would be too "
                f"narrow. Drop noise_profiling for this fit and sample the scale as a free "
                f"parameter, or use it with an optimizer.")
        names, refusals = self.obj.noise_profiling_plan()
        if refusals:
            listed = '\n  * '.join(refusals)
            raise PybnfError(
                'noise_profiling cannot profile every estimated noise scale in this fit',
                f"noise_profiling = 1 applies to EVERY estimated noise scale in the fit -- "
                f"profiling some while searching others would silently change what the searched "
                f"ones mean -- and this fit has one it cannot profile:\n  * {listed}\n"
                f"Declare those scales as 'fit <parameter>' free parameters (the only form with "
                f"a closed-form profile), or drop noise_profiling and search them.")
        if not names:
            raise PybnfError(
                'noise_profiling has nothing to profile in this fit',
                "noise_profiling = 1 was set, but no observable in this fit estimates its noise "
                "scale as a free parameter -- every scale is fixed (read from a data column, a "
                "constant, or the measurement). There is no search dimension to remove. Drop "
                "noise_profiling, or declare a scale with 'noise_model <obs> = <family>, "
                "<scale> = fit <parameter>'.")
        profiled = set(names)
        declared = {v.name for v in self.variables}
        missing = sorted(profiled - declared)
        if missing:
            # An invariant guard rather than a user-facing path: _load_variables already
            # refuses an undeclared estimated noise parameter, and it and self.variables read
            # the same config keys. Kept because the alternative failure -- a profiled name
            # that no evaluation can solve for -- surfaces as a KeyError deep inside scoring.
            raise PybnfError(
                f"noise_profiling: noise parameter(s) {', '.join(missing)} are not free parameters",
                f"noise_profiling = 1 profiles the estimated noise scale(s) "
                f"{', '.join(missing)}, but they are not declared as free parameters. A profiled "
                f"scale is still a declared, estimated parameter -- the switch removes it from "
                f"the SEARCH, not from the .conf -- so declare each one, e.g. "
                f"'loguniform_var = {missing[0]} 1e-3 1e3'.")
        self.profiled_variables = [v for v in self.variables if v.name in profiled]
        self.variables = [v for v in self.variables if v.name not in profiled]
        self.profiled_noise_params = sorted(profiled)
        self.obj._profiled_noise_params = frozenset(profiled)
        print1('Analytic noise profiling: %d estimated noise scale(s) (%s) are profiled out '
               'of the search (%d searched parameter(s) remain).'
               % (len(self.profiled_noise_params), ', '.join(self.profiled_noise_params),
                  len(self.variables)))
        logger.info('noise_profiling: profiling %s; searching %s'
                    % (self.profiled_noise_params, [v.name for v in self.variables]))

    def _qualitative_scale_param(self):
        """The free-parameter name that ``qualitative_scale = fit <param>`` ties every qualitative
        constraint's scale (logit ``s`` / probit ``sigma``) to, or ``None`` when
        the scales are fixed as authored. Validates the only supported form, ``fit <param>``."""
        spec = self.config.get('qualitative_scale')
        if not spec:
            return None
        tokens = list(spec)
        if len(tokens) != 2 or str(tokens[0]).lower() != 'fit':
            raise PybnfError(
                f"Invalid qualitative_scale = {' '.join(map(str, tokens))}",
                "The only supported form is 'qualitative_scale = fit <parameter>', naming a declared "
                "free parameter to estimate the qualitative-constraint scale jointly with the fit.")
        return tokens[1]

    def _load_variables(self):
        """
        Loads the variable names from the config dict into FreeParameter instances.
        :return: a list of FreeParameter instances
        """
        # Every free-parameter noise source the objective estimates -- the objfunc's
        # own default (chi_sq_dynamic's sigma__FREE, neg_bin_dynamic's r__FREE,
        # laplace's b__FREE) plus any per-observable noise_model 'fit' source -- must
        # be declared as a free parameter. One general check derived from the
        # objective's sources (ADR-0021) replacing the old per-objfunc magic-name
        # special cases; self.obj is already built (it precedes _load_variables).
        declared_params = {k[1] for k in self.config.keys() if self._is_free_param_key(k)}
        missing = self.obj.required_free_noise_params() - declared_params
        if missing:
            names = ', '.join(sorted(missing))
            raise PybnfError(f'Noise free parameter(s) {names} not declared',
                             f'Objective function {self.config["objfunc"]} (or a per-observable noise_model) '
                             f'estimates the noise parameter(s) {names}, but they are not declared as free '
                             f'parameters in the .conf file (and the model file). Declare each as a variable, '
                             f'e.g. "uniform_var = {sorted(missing)[0]} <lower> <upper>".')
        # An estimated qualitative-constraint scale (qualitative_scale = fit <param>) likewise names
        # a free parameter that must be declared.
        qscale = self._qualitative_scale_param()
        if qscale is not None and qscale not in declared_params:
            raise PybnfError(f'qualitative_scale free parameter {qscale} not declared',
                             f'qualitative_scale = fit {qscale} ties every qualitative-constraint scale to a '
                             f'free parameter, but {qscale} is not declared. Declare it as a positive '
                             f'(log-scaled) variable, e.g. "loguniform_var = {qscale} <lower> <upper>".')
        fit_type = self.config['fit_type']
        self._check_variable_keyword_combination(fit_type)
        variables = []
        initialization_distribution = self.config.get('initialization_distribution', 'prior')
        for k in self.config.keys():
            if isinstance(k, tuple) and re.search('var$', k[0]):
                if k[0] in ('var', 'logvar'):
                    # 2nd number (step size) may be absent, must fill in appropriately
                    if len(self.config[k]) >= 2:
                        stepsize = self.config[k][1] # easy, it was right there
                    else:
                        stepsize = None  # Will sort out within SimplexAlgorithm
                    free_param = FreeParameter(k[1], k[0], self.config[k][0], stepsize)
                else:
                    if len(self.config[k]) == 3:
                        free_param = FreeParameter(k[1], k[0], self.config[k][0], self.config[k][1],
                                                       bounded=self.config[k][2],
                                                       initialization_distribution=initialization_distribution)
                    else:
                        # Two numbers (a location/scale family) or one (a one-parameter
                        # unbounded family: exponential/chisquare/rayleigh -- ADR-0010/#417);
                        # p2 is absent for the latter.
                        p2 = self.config[k][1] if len(self.config[k]) >= 2 else None
                        free_param = FreeParameter(k[1], k[0], self.config[k][0], p2,
                                                   initialization_distribution=initialization_distribution)
            elif isinstance(k, tuple) and k[0] == 'parameter':
                # New-era ``parameter:`` record (ADR-0043) -- a fully-labeled free-parameter
                # declaration. Edition-gated (>= 2); the legacy positional ``*_var`` lines above
                # are unchanged at every edition.
                ed = edition.resolve_edition(self.config.get('edition'))
                edition.require_edition(ed, 2, "the 'parameter:' declaration syntax")
                free_param = self._free_parameter_from_record(k[1], self.config[k],
                                                              initialization_distribution)
            else:
                continue

            logger.debug(f'Adding parameter {free_param.name} with bounds [{free_param.lower_bound}, {free_param.upper_bound}]')
            variables.append(free_param)
        logger.info('Loaded variables')
        return variables

    @staticmethod
    def _is_free_param_key(k):
        """Whether a config key declares a free parameter -- a legacy ``(*_var, id)`` tuple
        or a new-era ``('parameter', id)`` record (ADR-0043)."""
        return isinstance(k, tuple) and (bool(re.search('var$', k[0])) or k[0] == 'parameter')

    def _free_parameter_from_record(self, pid, raw_fields, initialization_distribution):
        """Build a :class:`FreeParameter` from a new-era ``parameter:`` record (ADR-0043).

        ``raw_fields`` is the parsed ``{field: str}`` map -- every part of the line is named:
        ``prior`` (the family), ``space`` (``linear``/``log10``, the sampling-space transform),
        the family's own distribution fields (``mean``/``sd``, ``location``/``scale``, ...),
        ``lower``/``upper`` (the bounds that truncate the prior -- #417/ADR-0020), and
        ``initial_value`` (the start point). No positional numbers; the family names its fields
        via ``Prior.field_names``. The truncation/box capability is unchanged -- this only maps
        named fields onto the existing ``FreeParameter`` constructor.
        """
        fields = dict(raw_fields)

        def _num(name):
            v = fields.pop(name)
            try:
                return float(v)
            except (TypeError, ValueError):
                raise PybnfError(f"Parameter '{pid}': field '{name}' must be a number, got {v!r}.")

        prior_name = fields.pop('prior', None)
        # The sampling-space transform. PyBNF samples in linear, log10, or natural log; each
        # base is named explicitly so it is never ambiguous (ADR-0022/0043). ``lin`` is
        # accepted as PEtab's spelling of ``linear``; ``log`` is rejected as ambiguous (PEtab
        # means natural by it, PyBNF historically means log10) -- write ``ln`` or ``log10``.
        # The base prefixes the family keyword (``log{f}_var`` / ``ln{f}_var`` / ``var``).
        pscale = str(fields.pop('parameter_scale', 'linear')).lower()
        scale_prefix = {'lin': '', 'linear': '', 'log10': 'log', 'ln': 'ln'}
        if pscale == 'log':
            raise PybnfError(
                f"Parameter '{pid}': parameter_scale 'log' is ambiguous -- write 'log10' "
                f"(base 10) or 'ln' (natural log) explicitly (ADR-0022).")
        if pscale not in scale_prefix:
            raise PybnfError(f"Parameter '{pid}': parameter_scale must be 'linear', 'log10', or "
                             f"'ln', got '{pscale}'.")
        prefix = scale_prefix[pscale]

        lower = _num('lower') if 'lower' in fields else None
        upper = _num('upper') if 'upper' in fields else None
        # Bounds come as a pair: an open side is an explicit +-inf, never a blank
        # (ADR-0047 -- no specification by absence). Omitting *both* is the untruncated
        # shorthand. One-sided truncation IS supported now -- spell the open side with
        # an infinity. The graded floor rule (positivity, support floor) is applied per
        # path below: finite for a uniform box, the family floor for a truncated prior.
        if (lower is None) != (upper is None):
            present, absent = ('lower', 'upper') if upper is None else ('upper', 'lower')
            raise PybnfError(
                f"Parameter '{pid}': bounds come as a pair -- '{present}' is set but "
                f"'{absent}' is missing. For an open {absent} side write an explicit "
                f"infinity ('{absent}: inf' or '{absent}: -inf'), not a blank (ADR-0047).")
        initial_value = _num('initial_value') if 'initial_value' in fields else None
        is_log_scale = prefix in ('log', 'ln')

        if prior_name is None:
            if lower is not None:
                # No prior but bounds -> uniform over the bounds (PEtab's default for an
                # estimated parameter without an explicit prior; the importer does the same).
                self._require_finite_box(pid, lower, upper, is_log_scale, "a uniform box")
                keyword = f'{prefix}uniform_var'
                self._reject_extra_fields(pid, fields, keyword)
                return FreeParameter(pid, keyword, lower, upper, value=initial_value, bounded=True,
                                     initialization_distribution=initialization_distribution)
            # No prior and no bounds -> the no-prior start point (legacy var/logvar/lnvar). Its
            # start value is carried in the FreeParameter's first slot *in sampling space*
            # (Simplex reads it via from_sampling_space(p1)), so map the theta-space
            # initial_value through the scale -- making initial_value the real value (theta) for
            # a log start point too, consistent with the prior-param case.
            if initial_value is None:
                raise PybnfError(f"Parameter '{pid}': declares no prior, no bounds, and no "
                                 f"initial_value -- nothing to fit. Give it a 'prior:', a "
                                 f"'lower:'/'upper:' box, or an 'initial_value:'.")
            if is_log_scale and initial_value <= 0.0:
                raise PybnfError(f"Parameter '{pid}': a {pscale} start point needs "
                                 f"initial_value > 0, got {initial_value}.")
            self._reject_extra_fields(pid, fields, 'a no-prior start point')
            _, start_scale = PRIOR_KEYWORD_MAP[f'{prefix}var']
            return FreeParameter(pid, f'{prefix}var', float(start_scale.forward(initial_value)), None,
                                 initialization_distribution=initialization_distribution)

        prior_name = str(prior_name).lower()
        if prior_name == 'uniform':
            # Uniform: lower/upper ARE the support (and the bounds); no separate family fields.
            if lower is None:
                raise PybnfError(f"Parameter '{pid}': a uniform prior needs 'lower' and 'upper'.")
            self._require_finite_box(pid, lower, upper, is_log_scale, "a uniform prior")
            keyword = f'{prefix}uniform_var'
            self._reject_extra_fields(pid, fields, keyword)
            return FreeParameter(pid, keyword, lower, upper, value=initial_value, bounded=True,
                                 initialization_distribution=initialization_distribution)

        keyword = f'{prefix}{prior_name}_var'
        if keyword not in PRIOR_KEYWORD_MAP:
            raise PybnfError(f"Parameter '{pid}': unknown prior family '{prior_name}'.")
        fam, _scale = PRIOR_KEYWORD_MAP[keyword]
        params = []
        for fname in fam.field_names:
            if fname not in fields:
                raise PybnfError(f"Parameter '{pid}': prior '{prior_name}' needs field '{fname}'.")
            params.append(_num(fname))
        self._reject_extra_fields(pid, fields, f"prior '{prior_name}'")
        p1 = params[0]
        p2 = params[1] if len(fam.field_names) >= 2 else None
        # A three-parameter family (student_t, ADR-0057) carries its third value in p3;
        # field_names ordered it last (df/location/scale -> p1/p2/p3). The carrier and
        # build_prior pass it through; it is None for the one- and two-parameter families.
        p3 = params[2] if len(fam.field_names) >= 3 else None
        # lower/upper truncate an unbounded-support family to a reflecting box: two finite
        # walls (two-sided, ADR-0020) or one finite wall + an infinity (half-bounded,
        # ADR-0047). The graded floor rule warns/errors on a sub-floor lower bound first.
        lower, upper = self._graded_truncation_bounds(pid, lower, upper, fam, _scale)
        return FreeParameter(pid, keyword, p1, p2, lb=lower, ub=upper, value=initial_value,
                             initialization_distribution=initialization_distribution, p3=p3)

    @staticmethod
    def _require_finite_box(pid, lower, upper, is_log, where):
        """A Uniform family's bounds ARE its support, so they must be finite -- an
        infinite bound describes an unbounded prior's open tail, not a box. A log
        scale additionally needs a strictly positive lower bound (ADR-0047)."""
        for label, v in (('lower', lower), ('upper', upper)):
            if v is None or not np.isfinite(v):
                raise PybnfError(
                    f"Parameter '{pid}': {where} needs a finite '{label}' bound "
                    f"(got {v}); an infinite bound describes an unbounded prior's open "
                    f"tail, not a uniform box.")
        if is_log and lower <= 0.0:
            raise PybnfError(
                f"Parameter '{pid}': {where} on a log scale needs 'lower' > 0 "
                f"(log of <= 0 is -inf), got lower={lower}.")

    @staticmethod
    def _graded_truncation_bounds(pid, lower, upper, fam, scale):
        """Apply the ADR-0047 graded sentinel/floor rule to a truncated family's bounds.

        ``lower``/``upper`` are in theta, already validated to be both-set or both-None
        (the pairing rule). Omit-both passes through as the untruncated shorthand. On a
        positive-support family -- whose theta floor, derived from the family's natural
        support and the scale, is finite (0 for the linear half-bounded families;
        0 for any log form; the doubly-unbounded families floor at -inf and are exempt) --
        a sloppy-but-lossless ``lower: -inf`` is warned and canonicalized to the floor,
        and a *finite* ``lower`` below the floor (a wall in the zero-density region, a
        likely wrong family/scale) is an error. These families are all unbounded above,
        so the upper side needs no floor. Returns the (possibly canonicalized) bounds."""
        if lower is None:
            return lower, upper
        floor = scale.inverse(fam.support_lo_u)   # theta-space support floor
        if np.isfinite(floor):
            if lower == -np.inf:
                logger.warning(
                    f"Parameter '{pid}': 'lower: -inf' on a prior whose support floor "
                    f"is {floor:g} -- interpreting as open below at the floor. Write "
                    f"'lower: {floor:g}' to silence this (ADR-0047).")
                lower = floor
            elif lower < floor:
                raise PybnfError(
                    f"Parameter '{pid}': 'lower: {lower:g}' is below the prior's support "
                    f"floor {floor:g} -- a finite wall in the zero-density region (likely "
                    f"a wrong family or scale). Use 'lower: {floor:g}' for an open lower "
                    f"side, or a value >= {floor:g} (ADR-0047).")
        return lower, upper

    @staticmethod
    def _reject_extra_fields(pid, leftover, where):
        """Raise a clear error if a ``parameter:`` record carries fields unknown to ``where``
        (a typo or a field from a different family) -- naming every part means an unrecognised
        name is an error, not a silently-ignored token."""
        if leftover:
            unknown = ', '.join(sorted(leftover))
            raise PybnfError(f"Parameter '{pid}': unknown field(s) for {where}: {unknown}.")

    def _check_variable_keyword_combination(self, fit_type):
        """Validate that the fit's free-parameter keywords match what the fit_type
        accepts -- the var/logvar-vs-prior rule, generalized for the box optimizers.

        Three categories of fit_type, derived from two registry flags (ADR-0005):

        * **point-only start optimizer** (``refiner`` and not ``start_from_box`` --
          Simplex, Powell): begins from a single value per parameter, so it takes
          only the no-prior ``var`` / ``logvar`` keywords.
        * **box-capable start optimizer** (``refiner`` and ``start_from_box`` --
          CMA-ES, #404/ADR-0017): runs either from a single ``var`` / ``logvar``
          point *or* over a bounded-prior box (``uniform_var`` / ``loguniform_var``),
          but not a mix, and not an unbounded prior (which has no box to span).
        * **everything else** (samplers, population optimizers): draws every
          variable from a prior, so it never takes ``var`` / ``logvar``.

        ADR-0015 derived the var/logvar rule from ``refiner`` alone because "is a
        refiner" and "takes a var/logvar point" then coincided; box mode is exactly
        the divergence it flagged, so the capability splits onto ``start_from_box``.
        """
        start_point_types = {code for code, e in FIT_TYPE_REGISTRY.items() if e.refiner}
        box_types = {code for code, e in FIT_TYPE_REGISTRY.items() if e.start_from_box}
        bounded_prior_kws = {kw for kw, (fam, _scale) in PRIOR_KEYWORD_MAP.items()
                             if fam.has_bounded_support}

        used = {k[0] for k in self.config.keys()
                if isinstance(k, tuple) and re.search('var$', k[0])}
        point_kws = used & {'var', 'logvar'}
        prior_kws = used - {'var', 'logvar'}
        unbounded_prior_kws = prior_kws - bounded_prior_kws

        if fit_type not in start_point_types:
            if point_kws:
                names = ' / '.join(sorted(start_point_types))
                raise PybnfError(
                    'Tried to use start-point variable type {} in another algorithm.'.format(' / '.join(sorted(point_kws))),
                    "You've used the {} keyword, but var / logvar are only for the "
                    "start-point optimizers (job_type = {}).\nValid keywords for other "
                    "algorithms are: uniform_var, normal_var, lognormal_var, "
                    "loguniform_var.".format(' / '.join(sorted(point_kws)), names))
            return

        # A start-point optimizer (Simplex / Powell / CMA-ES) from here on.
        if not prior_kws:
            return  # classic single-point start (var / logvar only, or no vars)

        names = ' / '.join(sorted(start_point_types))
        if fit_type not in box_types:
            raise PybnfError(
                'Invalid start-point variable type {}'.format(' / '.join(sorted(prior_kws))),
                "You've specified a start-point optimizer (job_type = {}; one of {}), "
                "but defined a variable with the {} keyword.\nFor these optimizers, "
                "you must instead define a single initial value for each variable\n"
                "using the var or logvar keyword (e.g. var = p1 42 ).".format(fit_type, names, ' / '.join(sorted(prior_kws))))

        # Box-capable optimizer given priors: must be a clean bounded-prior box.
        if point_kws:
            raise PybnfError(
                'Mixed start-point and box variable types',
                "job_type = {} uses both a single-value start point (var / logvar) and "
                "a prior-based variable ({}).\nUse one consistent style: var / logvar "
                "for a point start, or uniform_var / loguniform_var for a global box "
                "search.".format(fit_type, ' / '.join(sorted(prior_kws))))
        if unbounded_prior_kws:
            raise PybnfError(
                'Box-mode optimizer requires a bounded prior',
                "job_type = {} runs a global box search when given priors, which needs a "
                "bounded box, but variable type {} is unbounded.\nUse uniform_var / "
                "loguniform_var for box mode, or var / logvar for a single-point start.".format(fit_type, ' / '.join(sorted(unbounded_prior_kws))))

    def _check_variable_correspondence(self):
        """Verify the config's free parameters and the models' parameters line up.

        PyBNF's load-time guard against a mistyped or orphaned free parameter. It
        runs once at config load (every fit_type except 'check') and raises rather
        than letting a fit proceed with a silently-wrong parameterization. Two
        directions:

        * config -> model: every .conf variable must appear in *at least one*
          model; a variable in no model is almost always a typo. This is what makes
          the per-model appliers' silent skip safe -- BngsimModel.execute and
          SbmlModelNoTimeout._modify_params deliberately skip a parameter not in
          *that* model (it may belong to another model in a multi-model fit), and
          this check catches a genuine typo here, before any simulation runs.
        * model -> config: every ``__FREE`` declared in a model file must be in the
          .conf, so you can't forget to fit one.

        Parameter names are unioned across all models, so multi-model and
        mixed-type fits work (a variable valid in any one model passes). The models
        spell parameters differently -- BNGL uses ``k__FREE``, SBML/bngsim use the
        bare ``kcat`` -- and the union accommodates both in one PSet.

        Param-agnostic models are skipped: a model exposing no enumerable parameter
        set takes its parameters from the .conf, so nothing can be proven a typo
        against it. AnalyticalModel and ExpressionModel are the examples (empty
        ``param_names`` by design -- the analytical menu and the bring-your-own
        expression); the ``hasattr`` guard also covers any future model type that never
        sets ``param_names``. This is the single config-level correspondence guard;
        do not add a duplicate elsewhere, and keep its regression tests
        (test_config_class) in sync.

        Under a new-era edition (>= 2) this delegates to
        :meth:`_check_variable_correspondence_modern`: BNGL free parameters bind by
        id with no ``__FREE`` marker (ADR-0034), so the model -> config "must-fit"
        direction goes away and the config -> model direction resolves against each
        model's full parameter namespace. The legacy body below is unchanged.
        """
        from .analytical_model import AnalyticalModel, CallableModel, ExpressionModel

        if edition.is_modern(edition.resolve_edition(self.config.get('edition'))):
            self._check_variable_correspondence_modern()
            return

        # Skip if any model is param-agnostic (no enumerable parameter set): its
        # parameters come from the .conf, so nothing here is provably a typo. The
        # hasattr clause future-proofs against a model type that never sets
        # param_names (which would otherwise AttributeError in the union below).
        for m in self.models.values():
            if isinstance(m, (AnalyticalModel, ExpressionModel, CallableModel)) or not hasattr(m, 'param_names'):
                return

        model_vars = set()
        for m in self.models.values():
            model_vars.update(getattr(m, 'param_names', set()))

        variables_names = {v.name for v in self.variables}
        extra_in_conf = variables_names.difference(model_vars)
        # An estimated qualitative-constraint scale (qualitative_scale = fit) is a
        # nuisance bound to no model parameter, not a typo.
        qscale = self._qualitative_scale_param()
        if qscale is not None:
            extra_in_conf.discard(qscale)
        extra_in_model = set(model_vars).difference(variables_names)
        # Only __FREE-suffixed names are "must-fit" model parameters; ignore other
        # model params (e.g. SBML species/globals) that legitimately aren't fit.
        # (Was `p[-8:] == '__FREE'`, comparing an 8-char slice to a 6-char string --
        # always False, so this whole direction silently never fired.)
        extra_in_model = {p for p in extra_in_model if p.endswith('__FREE')}

        if len(extra_in_conf) > 0:
            raise PybnfError('The following variables are declared in the .conf file, but were not found in any model '
                             f'file: {extra_in_conf}')
        if len(extra_in_model) > 0:
            raise PybnfError('The following free parameters are in your model files, but are not declared in your '
                             f'.conf file: {extra_in_model}')

    def _check_variable_correspondence_modern(self):
        """New-era (edition >= 2) free-parameter typo check (ADR-0034).

        Bind-by-id: a config free parameter whose name matches a model parameter id
        is bound to that parameter (``set_param``) -- the same contract the SBML and
        bngsim backends use -- with no ``__FREE`` marker and the model file carried
        verbatim. The model-file marker used to double as a wiring check; this
        replaces that with a typo check on the config free parameters:

        * a free parameter matching a model parameter id -> bound (the common case);
        * a free parameter matching no id but referenced by the objective /
          ``noise_model`` surface (an intended nuisance, e.g. a free sigma the model
          never sees) -> fine -- the same source :meth:`_load_variables` validates;
        * a free parameter matching no id **and** referenced by no such surface ->
          almost certainly a typo -> error (listing the models' parameter names).

        There is no model -> config direction in the new era: a verbatim model
        carries no ``__FREE`` markers, so every model parameter is an optional knob,
        fit only when declared (unlike legacy's must-fit ``__FREE``). Ids are unioned
        across models, so multi-model fits work (a variable valid in any one model
        passes), mirroring the legacy union.
        """
        from .analytical_model import AnalyticalModel, CallableModel, ExpressionModel

        # Param-agnostic models (the analytical menu's AnalyticalModel, the bring-your-own
        # ExpressionModel / CallableModel) take their parameters from the .conf -- and the
        # expression/callable binds them by name itself -- so nothing can be proven a typo against
        # them: skip the whole check, exactly as the legacy branch does.
        for m in self.models.values():
            if isinstance(m, (AnalyticalModel, ExpressionModel, CallableModel)) or not hasattr(m, 'param_names'):
                return

        model_ids = set()
        for m in self.models.values():
            model_ids.update(self._bindable_param_ids(m))

        # Free parameters the objective / per-observable noise_model estimates but no
        # model ever sees (e.g. chi_sq_dynamic's free sigma): legitimate nuisances,
        # bound to no model id. Same source _load_variables checks (ADR-0021). The
        # measurement layer's free symbols join them -- an observableParameters scale
        # referenced only by a measurement model is an observation-layer nuisance (ADR-0044).
        nuisance = set(self.obj.required_free_noise_params())
        nuisance |= getattr(self, '_measurement_free_params', set())
        # A parameter id used only as a row-varying per-measurement noise token (the ADR-0045
        # binding table) is a legitimate nuisance, bound to no model id.
        nuisance |= getattr(self, '_per_measurement_free_params', set())
        # A free parameter a condition perturbation references by value (a per-condition
        # estimated initial condition, ADR-0076) binds no model entity of its own -- it feeds
        # the condition, not the model directly -- so it is a legitimate nuisance too.
        nuisance |= getattr(self, '_condition_free_params', set())
        # An estimated qualitative-constraint scale (qualitative_scale = fit) is a
        # nuisance too -- it feeds the constraint penalty, not the model.
        qscale = self._qualitative_scale_param()
        if qscale is not None:
            nuisance.add(qscale)

        orphans = sorted(v.name for v in self.variables
                         if v.name not in model_ids and v.name not in nuisance)
        if orphans:
            listed = ', '.join(sorted(model_ids)) if model_ids else '(none)'
            raise PybnfError(
                'Free parameter(s) match no model parameter: ' + ', '.join(orphans),
                f"The free parameter(s) {', '.join(orphans)} are declared in the .conf "
                f"file but match no parameter id in any model file, and are not "
                f"referenced by the objective or a noise_model as a nuisance parameter. "
                f"Under edition >= 2 a free parameter binds to a model parameter by id "
                f"(there is no '__FREE' marker), so this is almost certainly a typo.\n"
                f"The model parameter ids are: {listed}")

    @staticmethod
    def _bindable_param_ids(model):
        """The model parameter ids a new-era config free parameter may bind to.

        For a BNGL model this is the full ``begin parameters`` namespace
        (``model_param_names``, ADR-0034), not the legacy ``__FREE`` tokens
        (``param_names``); for the SBML / bngsim backends it is ``param_names``
        (species + globals), already the bind-by-id namespace. Returns an empty set
        for a model exposing neither (the caller has already excluded the
        param-agnostic models that legitimately take their parameters from the conf).
        """
        if isinstance(model, BNGLModel):
            return set(getattr(model, 'model_param_names', ()))
        return set(getattr(model, 'param_names', ()))

    def _postprocess_normalization(self):
        """
        Postprocessing on the 'normalization' key
        :return:
        """
        seedoc = "\nSee the documentation for the syntax options for the 'normalization' key"
        # peak/init/zero/unit rescale the SIMULATED column before scoring; floor (an additive
        # measurement-noise floor x' = x + rho*max(x)) and scale (analytic per-series optimal
        # scaling) are the ADR-0066 primitives, applied to the data too. floor/scale are
        # edition-2-only new-era features that can chain, so the legacy per-file form keeps only
        # the original four.
        valid = ('init', 'peak', 'zero', 'unit', 'floor', 'scale')
        legacy_valid = ('init', 'peak', 'zero', 'unit')

        # New-era per-observable rules (ADR-0053, #444; chains ADR-0066, #479): structural
        # ('normalization', target) tuple keys, where target is an observable name or
        # '<experiment>.<observable>' and the value is a *chain* (a bare string, or a list of
        # transforms each a string or a (name, arg) tuple). Normalization is a per-observable
        # *prediction* transform (a sibling of the per-observable noise_model / cumulative
        # surface, ADR-0021/0051), so the new era keys it by observable (and optionally an
        # experiment), never a filename. Collected + validated here, then resolved against the
        # (experiment x column) grid below; the layers form a total specificity order (whole-fit
        # default < per-observable < per-(experiment, observable)).
        ed = edition.resolve_edition(self.config.get('edition'))
        per_obs, per_exp_obs = {}, {}
        norm_tuple_keys = [k for k in self.config
                           if isinstance(k, tuple) and k[0] == 'normalization']
        for k in norm_tuple_keys:
            target = k[1]
            chain = self._canonical_norm_chain(self.config[k], valid, f" for '{target}'", seedoc)
            if '.' in target:
                exp_name, obs = target.split('.', 1)
                per_exp_obs[(exp_name, obs)] = chain
            else:
                per_obs[target] = chain
        if norm_tuple_keys:
            edition.require_edition(
                ed, 2, "per-observable normalization ('normalization <observable> = <type>')")

        base = self.config['normalization']

        if type(base) == dict:
            # Legacy per-FILE normalization (keyed by .exp filename). It is incompatible
            # with the new-era surface, which keys data by experiment name, not filename
            # (ADR-0028): the filename stem is no longer the data key, so a per-file rule
            # would silently fail to resolve (issue #444). Under a modern edition redirect
            # to the per-observable form rather than mis-resolve; legacy edition keeps the
            # historical filename behaviour byte-identical.
            if ed >= 2:
                raise PybnfError(
                    "Per-file normalization (normalization = <type>: <file.exp>) is a legacy "
                    "form not supported under edition >= 2",
                    "Under edition >= 2 a config keys its data by experiment name, not by .exp "
                    "filename, so per-file normalization does not apply. Use the per-observable "
                    "form 'normalization <observable> = <type>' (every experiment) or "
                    "'normalization <experiment>.<observable> = <type>' (one experiment)."
                    + seedoc)
            self._postprocess_legacy_normalization_dict(legacy_valid, seedoc)
            return

        whole_fit = (self._canonical_norm_chain(base, valid, "", seedoc)
                     if isinstance(base, (str, list)) else None)
        if whole_fit is None and not norm_tuple_keys:
            return  # nothing declared (base is None and no per-observable rules)

        result, analytic_scale, exp_ops = self._resolve_normalization_grid(
            whole_fit, per_obs, per_exp_obs, seedoc)
        # {data_key: [(transform, [cols])]} of the SIM-side (peak/init/zero/unit/floor) transforms.
        self.config['normalization'] = result or None
        # {data_key: frozenset(cols)} the objective analytically scales at scoring time, or None.
        if analytic_scale:
            self.config['analytic_scale'] = analytic_scale
        # Apply the SYMMETRIC transforms (floor) to the experimental data in place, once: a floor
        # is only meaningful applied identically to sim and data (ADR-0066). The sim side is
        # floored per-evaluation from result[data_key]; here the data is floored from its own max.
        for m, dk, ops in exp_ops:
            self.exp_data[m][dk].normalize(ops)

    #: Normalization transforms applied *symmetrically* to sim and data (ADR-0066): a floor
    #: (``x' = x + rho*max(x)``) is only meaningful applied identically to both. peak/init/
    #: zero/unit remain sim-only (the data is pre-normalized by the user). ``scale`` is applied
    #: to neither column directly -- it is profiled jointly from both at scoring time.
    _SYMMETRIC_NORMALIZATIONS = frozenset({'floor'})

    @staticmethod
    def _canonical_norm_chain(value, valid, where, seedoc):
        """Canonicalize a normalization value (a bare string or a chain list; parse.py, ADR-0066)
        into a list of transforms, validating each transform name against ``valid``. Each
        transform stays a bare string (argument-less) or a ``(name, arg)`` tuple (``floor``'s
        rho). ``where`` is a context suffix for the error (`` for 'x'``)."""
        chain = list(value) if isinstance(value, list) else [value]
        for t in chain:
            name = t[0] if isinstance(t, tuple) else t
            if name not in valid:
                raise PybnfError(
                    f"Invalid normalization type '{name}'{where}",
                    f"Invalid normalization type '{name}'. Options are: init, peak, zero, unit, "
                    f"floor, scale." + seedoc)
        return chain

    def _resolve_normalization_grid(self, whole_fit, per_obs, per_exp_obs, seedoc):
        """Resolve the per-(experiment, observable) normalization grid from the layered
        new-era rules (ADR-0053; chains + scale ADR-0066): for each measured observable column
        of each experiment, the most specific rule wins -- a per-(experiment, observable)
        override (``<exp>.<obs>``), else a per-observable rule (``<obs>``), else the whole-fit
        default (``normalization = <chain>``). Each rule is a *chain* of transforms.

        Returns ``(result, analytic_scale, exp_ops)``:

        * ``result`` -- ``{data_key: [(transform, [columns])]}`` of the SIM-side transforms
          (peak/init/zero/unit/floor), applied in order, the form ``Result.normalize`` /
          ``Data.normalize`` already consume (a ``transform`` is a bare string or a ``(name,
          arg)`` tuple).
        * ``analytic_scale`` -- ``{data_key: frozenset(columns)}`` the objective profiles a
          per-series optimal ``scale`` for at scoring time (routed to the objective, not a
          ``Data`` transform).
        * ``exp_ops`` -- ``[(model, data_key, [(transform, [columns])])]`` of the SYMMETRIC
          transforms (floor) the caller applies to the *experimental* data in place.

        Validates that every declared target matched a real measured observable (a typo
        otherwise), mirroring the new-era ``observable:`` override's unknown-header check.
        """
        # data_key (model, suffix) -> experiment name, so a ``<exp>.<obs>`` override matches
        # by the experiment NAME the user wrote even when the data_key is name+condition.
        dk_to_name = {(base, dk): name
                      for name, (base, dk) in self._experiment_data_keys.items()}
        known_exp_names = set(self._experiment_data_keys)

        matched_obs, matched_exp_obs, all_observables = set(), set(), set()
        result, analytic_scale, exp_ops = {}, {}, []
        for m in self.exp_data:
            for dk in self.exp_data[m]:
                d = self.exp_data[m][dk]
                cols = [c for c in d.cols if d.cols[c] != 0 and not c.endswith('_SD')]
                all_observables.update(cols)
                exp_name = dk_to_name.get((m, dk), dk)
                # Resolve each column's chain (most specific rule wins), preserving column order.
                chain_of_col = {}
                for c in cols:
                    if (exp_name, c) in per_exp_obs:
                        chain_of_col[c] = per_exp_obs[(exp_name, c)]
                        matched_exp_obs.add((exp_name, c))
                    elif c in per_obs:
                        chain_of_col[c] = per_obs[c]
                        matched_obs.add(c)
                    elif whole_fit is not None:
                        chain_of_col[c] = whole_fit
                    # else: covered by no rule -> left un-normalized
                # Group columns sharing an identical chain (in first-appearance order), then emit
                # each group's transforms in chain order, splitting them by seam: peak/init/zero/
                # unit/floor are Data-level SIM transforms (floor also lands on the data via
                # exp_ops); scale is an analytic scoring-time transform routed to the objective.
                by_chain, order = {}, []
                for c in cols:
                    if c not in chain_of_col:
                        continue
                    key = tuple(chain_of_col[c])
                    if key not in by_chain:
                        by_chain[key] = (chain_of_col[c], [])
                        order.append(key)
                    by_chain[key][1].append(c)
                sim_groups, scale_cols, sym_ops = [], [], []
                for key in order:
                    chain, gcols = by_chain[key]
                    for t in chain:
                        name = t[0] if isinstance(t, tuple) else t
                        if name == 'scale':
                            scale_cols.extend(gcols)
                        else:
                            sim_groups.append((t, list(gcols)))
                            if name in self._SYMMETRIC_NORMALIZATIONS:
                                sym_ops.append((t, list(gcols)))
                if sim_groups:
                    result[dk] = sim_groups
                if scale_cols:
                    analytic_scale[dk] = frozenset(scale_cols)
                if sym_ops:
                    exp_ops.append((m, dk, sym_ops))

        # Typo guards: a declared observable / experiment.observable that matched nothing.
        unmatched_obs = sorted(set(per_obs) - matched_obs)
        if unmatched_obs:
            raise PybnfError(
                f"normalization references unknown observable(s) {unmatched_obs}",
                f"normalization was specified for {unmatched_obs}, but no experiment measures "
                f"an observable by that name. Measured observables are "
                f"{sorted(all_observables)}." + seedoc)
        for (exp_name, obs) in per_exp_obs:
            if exp_name not in known_exp_names:
                raise PybnfError(
                    f"normalization references unknown experiment '{exp_name}'",
                    f"normalization '{exp_name}.{obs}' names experiment '{exp_name}', which is "
                    f"not defined. Experiments are {sorted(known_exp_names)}." + seedoc)
            if (exp_name, obs) not in matched_exp_obs:
                raise PybnfError(
                    f"normalization references unknown observable '{obs}' in experiment "
                    f"'{exp_name}'",
                    f"normalization '{exp_name}.{obs}' names observable '{obs}', which experiment "
                    f"'{exp_name}' does not measure." + seedoc)
        return result, analytic_scale, exp_ops

    def _postprocess_legacy_normalization_dict(self, valid, seedoc):
        """Legacy per-FILE normalization (``normalization = <type>: <file.exp>``), keyed by
        ``.exp`` filename -- the historical surface, kept byte-identical for legacy-edition
        jobs (the new-era per-observable form lives in :meth:`_resolve_normalization_grid`).
        Re-keys the filename dict to the data suffix and expands each entry to the
        ``{suffix: [(type, [columns])]}`` form the normalizer consumes."""
        newdict = dict()
        for ef in self.config['normalization']:
            if not isinstance(ef, str):
                continue  # skip the modern ('normalization', target) tuple keys
            if ef not in self.config['exp_data']:
                raise PybnfError(f"Invalid exp file {ef} under the normalization key",
                                 f"The exp file {ef} given under the 'normalization' keyword is not associated with "
                                 "any model." + seedoc)
            val = self.config['normalization'][ef]

            # Figure out how to get to the right data object (it's in a dict keyed on model name, then suffix)
            m = None
            for modelpath in self.config['models']:
                if ef in self.config[modelpath]:
                    m = self._file_prefix(modelpath, '(bngl|xml|ant)')
                    break
            suff = self._file_prefix(ef)

            def checkval(v):
                if v not in valid:
                    raise PybnfError("Invalid normalization type '{}'".format(self.config['normalization'][ef]),
                                     "Invalid normalization type '{}'. Options are: init, peak, zero, unit".format(self.config['normalization'][ef]) + seedoc)
            if type(val) == str:
                # This exp file has a single normalization type for all columns.
                # Convert to column-specific form using only the columns present in the .exp file,
                # so that simulation columns used only by .prop constraints are not normalized.
                checkval(val)
                exp_cols = [c for c in self.exp_data[m][suff].cols
                            if self.exp_data[m][suff].cols[c] != 0 and not c.endswith('_SD')]
                if not exp_cols:
                    continue
                val = [(val, exp_cols)]
            else:
                # This exp file has a list of one or more pairs specifying (normalization_type, [columns])
                for (i, (ntype, cols)) in enumerate(val):
                    checkval(ntype)
                    new_cols = []
                    if type(cols[0]) == int:
                        # Need to convert to string labels, because the indices into the sim data will be different
                        to_convert = cols
                        for label in self.exp_data[m][suff].cols:
                            ci = self.exp_data[m][suff].cols[label]
                            if ci in to_convert:
                                to_convert.remove(ci)
                                new_cols.append(label)
                        if len(to_convert) > 0:
                            raise PybnfError(f"Invalid normalization column {to_convert[0]} for file {ef}",
                                             "Specified normalization for column %i in file %s, but that file "
                                             "contains only %i columns." % (
                                             to_convert[0], ef, self.exp_data[m][suff].data.shape[1]) + seedoc)
                    else:
                        new_cols = cols
                    # Iterate over a copy: the _SD branch below removes from
                    # new_cols, and aliasing the iterator to the same list
                    # would skip the element after each removal (so a second
                    # consecutive _SD column was silently kept).
                    new_cols_iter = list(new_cols)
                    for c in new_cols_iter:
                        if c not in self.exp_data[m][suff].cols:
                            raise PybnfError(f"Invalid normalization column {c} for file {ef}",
                                             f"Specified normalization for column {c} in file {ef}, but that file does "
                                             "not contain that column name." + seedoc)
                        if c[-3:] == '_SD':
                            logger.info(f'Removing {c} from the normalization list')
                            print1(f"Warning: You specified a normalization for {c}, but I can't normalize a "
                                   "standard deviation separately, because it's not an output of the simulation. "
                                   f"I'm ignoring your {c} setting and assuming it's on the same scale as its data "
                                   "column.")
                            new_cols.remove(c)
                    # Update with the postprocessed normalization info
                    val[i] = (ntype, new_cols)

            newdict[suff] = val
        self.config['normalization'].update(newdict)

    def _load_postprocessing(self):
        """
        Loads config info for user-specified Python scripts for postprocessing data
        :return:
        """
        self.postprocessing = dict()
        if 'postprocess' not in self.config:
            return

        for spec in self.config['postprocess']:
            script = self._absolute(spec[0])
            suffixes = spec[1:]

            # Check for simple errors in the script here, before we start running anything.
            try:
                # This incantation loads the module as postproc
                import importlib.util
                logger.info(f'Prepare to load the script {script}')
                spec = importlib.util.spec_from_file_location("postprocessor", script)
                if not spec:
                    raise PybnfError(f'Could not load the postprocessing script {script}. Make sure this is a Python '
                                     'file (.py)')
                postproc = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(postproc)
                # Now postproc is the user-defined Python module
            except OSError:
                raise PybnfError(f'Could not load the postprocessing script {script}')
            try:
                func = postproc.postprocess
            except NameError:
                raise PybnfError(f'The postprocessing script {script} should contain a definition of the function '
                                 'postprocess(data). This function was not found.')

            for suff in suffixes:

                # Need to backsolve the model name based on the suffix.
                model_choices = []
                for modelname in self.models:
                    if suff in self.models[modelname].get_suffixes():
                        model_choices.append(modelname)
                if len(model_choices) == 0:
                    raise PybnfError(f'Suffix {suff} was specified for a postprocessing script, but that suffix was not '
                                     'found in any model')
                if len(model_choices) > 1:
                    raise PybnfError(f'Suffix {suff} was specified for a postprocessing script, but was found in multiple '
                                     'models. Please rename suffixes to avoid this ambiguity.')
                self.postprocessing[(model_choices[0], suff)] = script


class UnknownObjectiveFunctionError(PybnfError):
    pass


class UnspecifiedConfigurationKeyError(PybnfError):
    pass


class UnmatchedExperimentalDataError(PybnfError):
    pass
