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
            
        if 'models' not in d or len(d['models']) == 0:
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
        self.obj = self._load_obj_func()
        logger.debug('Loaded objective function')
        self.variables = self._load_variables()
        if self.config['fit_type'] != 'check':
            self._check_variable_correspondence()
        logger.debug('Loaded variables')
        self._postprocess_normalization()
        self._load_postprocessing()
        self.config['time_length'] = self._load_t_length()
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
                print1('Warning: fit_type was not specified. Defaulting to de (Differential Evolution).')
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
                print1('Warning: Configuration key {} is not used in fit_type {}, so I am ignoring it'.format(k, conf_dict['fit_type']))
                logger.warning('Ignoring unused key {} for fitting algorithm {}'.format(k, conf_dict['fit_type']))

    @staticmethod
    def _strip_uncheckable_keys(conf_dict):
        """Remove the keys ``fit_type = check`` cannot honor so they do not crash
        downstream: ``refine`` and ``bootstrap`` (model checking runs neither). The
        unused-key *warnings* for check now come from the unified
        :meth:`check_unused_keys`; this keeps only the crash-prevention deletion the
        old ``check_unused_keys_model_checking`` also did. Mutates and returns
        ``conf_dict``.
        """
        for k in ('refine', 'bootstrap'):
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
        timeDict = {}
        for mf in self.config['models']:
            if re.search(r'\.bngl$', mf):
                time = BNGLModel(mf, suppress_free_param_error=self.config['fit_type']=='check').find_t_length()
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

        md = {}
        for mf in self.config['models']:
            # Initialize model type based on extension
            try:
                if re.search(r'\.bngl$', mf):
                    model = BNGLModel(mf, suppress_free_param_error=self.config['fit_type']=='check')
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

        for model in md.values():
            if isinstance(model, BNGLModel) and not model.has_observables:
                print1(f'Warning: Model {model.file_path} has no observables defined. Fitting will not work without observables.')
                logger.warning(f'Model {model.file_path} has no observables defined')

        if self.config['smoothing'] > 1:
            # Check for misuse of 'smoothing' feature
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
            mut_objects = [Mutation(var, op, float(val)) for var, op, val in perts]
            self.models[base].add_mutant(MutationSet(mut_objects, name))
            logger.debug(f"Condition '{name}' applied to model '{base}' "
                         f"({len(mut_objects)} perturbation(s))")

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
                                 'https://pybnf.readthedocs.io/en/latest/installation.html#bionetgen')
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
                                     'https://pybnf.readthedocs.io/en/latest/installation.html#bionetgen')
                except FileNotFoundError:
                    #  Occurs on Mac/Linux if BNG2.pl is nonexistent.
                    raise PybnfError('The BioNetGen simulator (BNG2.pl) was not found at the specified location. Please set the '
                                     '"bng_command" configuration key to the location of the file BNG2.pl, or set the '
                                     'BNGPATH environmental variable to the folder containing BNG2.pl.\n'
                                     'If BioNetGen is not yet installed, please refer to installation instructions at '
                                     'https://pybnf.readthedocs.io/en/latest/installation.html#bionetgen')
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
                        cs.load_constraint_file(ef, scale=self.config['constraint_scale'])
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
        return obj

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
        declared_params = {k[1] for k in self.config.keys()
                           if isinstance(k, tuple) and re.search('var$', k[0])}
        missing = self.obj.required_free_noise_params() - declared_params
        if missing:
            names = ', '.join(sorted(missing))
            raise PybnfError(f'Noise free parameter(s) {names} not declared',
                             f'Objective function {self.config["objfunc"]} (or a per-observable noise_model) '
                             f'estimates the noise parameter(s) {names}, but they are not declared as free '
                             f'parameters in the .conf file (and the model file). Declare each as a variable, '
                             f'e.g. "uniform_var = {sorted(missing)[0]} <lower> <upper>".')
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
                        free_param = FreeParameter(k[1], k[0], self.config[k][0], self.config[k][1],
                                                   initialization_distribution=initialization_distribution)

                logger.debug(f'Adding parameter {free_param.name} with bounds [{free_param.lower_bound}, {free_param.upper_bound}]')
                variables.append(free_param)
        logger.info('Loaded variables')
        return variables

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
                    "start-point optimizers (fit_type = {}).\nValid keywords for other "
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
                "You've specified a start-point optimizer (fit_type = {}; one of {}), "
                "but defined a variable with the {} keyword.\nFor these optimizers, "
                "you must instead define a single initial value for each variable\n"
                "using the var or logvar keyword (e.g. var = p1 42 ).".format(fit_type, names, ' / '.join(sorted(prior_kws))))

        # Box-capable optimizer given priors: must be a clean bounded-prior box.
        if point_kws:
            raise PybnfError(
                'Mixed start-point and box variable types',
                "fit_type = {} uses both a single-value start point (var / logvar) and "
                "a prior-based variable ({}).\nUse one consistent style: var / logvar "
                "for a point start, or uniform_var / loguniform_var for a global box "
                "search.".format(fit_type, ' / '.join(sorted(prior_kws))))
        if unbounded_prior_kws:
            raise PybnfError(
                'Box-mode optimizer requires a bounded prior',
                "fit_type = {} runs a global box search when given priors, which needs a "
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
        against it. AnalyticalModel is the current example (empty ``param_names`` by
        design); the ``hasattr`` guard also covers any future model type that never
        sets ``param_names``. This is the single config-level correspondence guard;
        do not add a duplicate elsewhere, and keep its regression tests
        (test_config_class) in sync.
        """
        from .analytical_model import AnalyticalModel

        # Skip if any model is param-agnostic (no enumerable parameter set): its
        # parameters come from the .conf, so nothing here is provably a typo. The
        # hasattr clause future-proofs against a model type that never sets
        # param_names (which would otherwise AttributeError in the union below).
        for m in self.models.values():
            if isinstance(m, AnalyticalModel) or not hasattr(m, 'param_names'):
                return

        model_vars = set()
        for m in self.models.values():
            model_vars.update(getattr(m, 'param_names', set()))

        variables_names = {v.name for v in self.variables}
        extra_in_conf = variables_names.difference(model_vars)
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

    def _postprocess_normalization(self):
        """
        Postprocessing on the 'normalization' key
        :return:
        """
        seedoc = "\nSee the documentation for the syntax options for the 'normalization' key"
        valid = ('init', 'peak', 'zero', 'unit')
        if type(self.config['normalization']) == dict:
            # Iterate through the keys, which should be .exp file names. Check that these are actual exp files that
            # are used in the fitting, then add to the dictionary just the suffix, for easier lookup later
            newdict = dict()
            for ef in self.config['normalization']:
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
        elif type(self.config['normalization']) == str:
            if self.config['normalization'] not in valid:
                raise PybnfError("Invalid normalization type '{}'".format(self.config['normalization']),
                                 "Invalid normalization type '{}'. Options are: init, peak, zero, unit".format(self.config['normalization']) + seedoc)
            # Convert global normalization to column-specific form for each exp file,
            # so that simulation columns used only by .prop constraints are not normalized.
            ntype = self.config['normalization']
            newdict = dict()
            for m in self.exp_data:
                for suff in self.exp_data[m]:
                    exp_cols = [c for c in self.exp_data[m][suff].cols
                                if self.exp_data[m][suff].cols[c] != 0 and not c.endswith('_SD')]
                    if exp_cols:
                        newdict[suff] = [(ntype, exp_cols)]
            self.config['normalization'] = newdict

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
