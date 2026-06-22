"""Pydantic config schema — the typed substrate behind ``Configuration`` (ADR-0002).

M2.1 turns PyBNF's configuration knowledge from scattered, untyped sources (the
``parse.py`` key-type lists, the hand-rolled ``default_config()`` dict, the
per-fit_type preprocessing) into a typed Pydantic model that owns the defaults,
types, and coercion in one place. This module holds the **global** schema and the
shared :class:`PyBNFConfigModel` base; the per-method (family) schemas subclass
that base and co-locate with their algorithm classes (M2.1 Stage b, ADR-0006),
reached by ``Configuration._build_config`` through the registry
(``FitTypeEntry.schema``).

Each fit_type's effective config is **narrowed** to the keys that method reads
(ADR-0013, M2.1 Stage c): ``global defaults + its own method schema (defaults +
overrides) + extras``, so a ``de`` fit no longer carries ``cognitive`` /
``particle_weight`` / the MCMC defaults. :func:`default_union` (global defaults
unioned with every method schema's defaults) is **no longer** the
``_build_config`` baseline; it survives only behind the ``default_config()``
compat shim as the "every possible default" view. The one cross-fit_type reach --
``refine`` pulling in the whole Simplex schema -- is an explicit overlay in
``Configuration._build_config`` (the ``_REFINER_SCHEMA`` seam).

The flow is ``pyparsing (structural) -> raw dict -> Pydantic (validate / coerce /
default) -> effective dict``. ``Configuration`` keeps exposing a plain dict
(``config.config``) so the ~294 existing ``config.config['x']`` reaches stay
untouched (dict-compat per ADR-0002); typed access migrates opportunistically in
Stage (c). A Pydantic ``ValidationError`` is translated to ``PybnfError`` at the
call site in ``config.py``.

Faithfulness notes (the golden-config net, ``test_config_golden.py``, is the
oracle):

* Pydantic v2 does **not** validate field defaults, so an int-literal default on
  a float-typed field (e.g. ``adaptive_n_max = 30``) is stored and dumped as the
  int ``30`` -- byte-identical to the old ``default_config()``. A *user-supplied*
  value still coerces through the field type.
* For a key with a typed scalar field, ``parse.ploop`` has already coerced the
  value (the ``numkeys_int`` / ``numkeys_float`` / ``mult*`` lists), so per-field
  coercion here is redundant -- both produce the same type. It is **not** a no-op
  for every token-list key, though: ~15 have no typed field that coerces, so for
  those ``parse.py`` is the *sole* coercion source -- ``credible_intervals`` is a
  bare ``list``; ``exchange_every`` / ``adaptive_step_size`` / ``starting_params``
  / ``calculate_covari`` are ``Any``; the required keys ``population_size`` /
  ``max_iterations`` / ``verbosity`` and the ``RUNTIME_KEYS`` ride through as
  extras. Making the schema own *all* coercion would mean modeling those 15 --
  decided not worth it (lanl/PyBNF#402, closed): the redundancy is benign and it
  would fight the deliberate extras / ``RUNTIME_KEYS`` / ``Any`` escape hatches.
* ``exchange_every`` is typed ``Any`` (now on ``MCMCFamilyConfig``): ``parse``
  coerces it to ``int``, but the MCMC family's ``postprocess`` hook overwrites it
  with ``np.inf`` for the non-PT methods, so the field must hold both an int and an
  infinity without re-coercion.
* ``lambda`` is a Python keyword, so its field is ``lambda_`` with
  ``alias='lambda'``; the model is dumped ``by_alias=True`` to emit ``'lambda'``.

Keys a user may legitimately set that are *not* in this schema (the required
``population_size`` / ``max_iterations``, method-only keys like ``beta_range`` /
``init_size``, and the structural model/exp/free-parameter keys) are not modeled
here: ``Configuration`` carries them through as "extras" alongside the dumped
schema. They gain typed validation as method schemas land in Stage (b).
"""

from typing import Any, ClassVar, Literal, Optional

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .registry import FIT_TYPE_REGISTRY


def _default_bng_command():
    """Reproduce ``default_config()``'s BNGPATH-derived default exactly."""
    try:
        return str(Path(os.environ['BNGPATH']) / 'BNG2.pl')
    except KeyError:
        return ''


class PyBNFConfigModel(BaseModel):
    """Shared base for every PyBNF config model -- the global schema and each
    per-method (family) schema (M2.1 Stage b, ADR-0006).

    Carries the common pydantic settings, the :meth:`owned_keys` helper
    ``Configuration._build_config`` uses to partition a raw config dict into
    global / method / pass-through keys, and a no-op :meth:`postprocess` hook the
    MCMC family overrides with the beta-ladder logic (all other models inherit
    the no-op).
    """

    model_config = ConfigDict(
        populate_by_name=True,     # accept either 'lambda' (alias) or 'lambda_'
        extra='forbid',            # only this model's keys are passed in; catch leaks
        arbitrary_types_allowed=True,
    )

    # Config keys this method's algorithm reads at runtime but does NOT model as a
    # schema field, because they default at runtime from other state (e.g. scatter
    # search's ``init_size`` -> ``10*len(variables)``, simplex's
    # ``simplex_max_iterations`` -> ``max_iterations``) rather than from a fixed
    # literal -- the schema docstrings explain each exclusion. They are nonetheless
    # *valid* keys for their fit_type, so ``owned_keys()`` alone would wrongly flag
    # them as unused; :meth:`valid_keys` unions them in. Each method schema overrides
    # this with its own set; they merge across the class hierarchy in
    # :meth:`runtime_keys` (#401, ADR-0014). Declared ``ClassVar`` so pydantic treats
    # it as a plain class attribute, not a model field (subclasses may override with a
    # bare assignment -- pydantic keeps the ``ClassVar`` semantics).
    RUNTIME_KEYS: ClassVar[frozenset] = frozenset()

    @classmethod
    def owned_keys(cls):
        """The config-key names (aliases where defined) this model owns."""
        return frozenset(
            (f.alias or name) for name, f in cls.model_fields.items()
        )

    @classmethod
    def runtime_keys(cls):
        """The runtime-defaulted config keys this method reads (:data:`RUNTIME_KEYS`),
        merged across the class hierarchy so a leaf schema's set unions its bases'
        (e.g. ``BasicMCMCConfig`` adds ``reps_per_beta`` to the MCMC family's
        ``beta_range``). Reads each class's *own* ``__dict__`` entry so inheritance
        does not double-count."""
        return frozenset().union(
            *(klass.__dict__.get('RUNTIME_KEYS', frozenset()) for klass in cls.__mro__))

    @classmethod
    def valid_keys(cls):
        """Every config key this method legitimately reads: its owned schema fields
        plus its runtime-defaulted keys. The per-fit_type validity unit that
        ``Configuration.check_unused_keys`` checks a raw key against, replacing the
        hand-maintained ``alg_specific`` dict (#401, ADR-0014)."""
        return cls.owned_keys() | cls.runtime_keys()

    @classmethod
    def postprocess(cls, conf_dict, fit_type):
        """No-op preprocessing hook (ADR-0006). ``MCMCFamilyConfig`` overrides this
        with the beta ladder (``algorithms/samplers/base.py``); every other model
        inherits this no-op. Mutates and returns the raw config dict."""
        return conf_dict


class GlobalConfig(PyBNFConfigModel):
    """The global (run-level) configuration defaults, typed.

    Field order and grouping mirror the old ``Configuration.default_config()``
    so the two stay easy to diff. The per-method groups (PSO, DE, ... ) lift out
    into per-method schemas as Stage (b) migrates each method; what remains here
    is the truly run-level surface plus not-yet-migrated method blocks.
    """

    # --- edition (ADR-0031): the select-and-freeze conventions marker ---
    # An optional integer opting the .conf into a frozen set of modernized
    # conventions (``pybnf/edition.py``). ``None`` == absent == legacy (implicit
    # edition 1, byte-identical to historical behavior). Validated and resolved in
    # ``Configuration._check_edition``; the modern-syntax guard and the neg_bin
    # median-centering default key off it. Kept raw (None / int) in the effective
    # config -- the resolution to the legacy edition happens at each read site.
    edition: Optional[int] = None

    # --- global / run-level ---
    # job_type is the MODERN (edition >= 2) name for the run selector -- the key that
    # chooses across optimizers / samplers / the model checker (ADR-0028 addendum).
    # 'fit_type' was a misnomer (the registry's `family` spans more than fitting), so
    # the modern era renames it. This is a surface-only rename: Configuration normalizes
    # whichever key the edition allows into the internal STRUCTURAL_PASSTHROUGH
    # 'fit_type' slot (see _resolve_run_selector), so the registry lookup and every
    # downstream config['fit_type'] read are untouched. None == the key was not named
    # (legacy confs name the run with 'fit_type', which never reaches this field).
    job_type: Optional[str] = None
    # objfunc is the LEGACY (edition-1) objective key; the modern surface (ADR-0031)
    # is the three keys below. Under a modern edition (`edition >= 2`) naming `objfunc`
    # is an error; in the legacy edition it works as it always has. Its 'chi_sq'
    # default is only consulted on the legacy path (the modern surface has no implicit
    # default -- an objective must be named).
    objfunc: str = 'chi_sq'
    # The modern objective surface (ADR-0031), each gated to `edition >= 2`:
    #   * objective -- the named per-point catch-all: a legacy token that desugars to a
    #     noise model (sos / chi_sq / laplace / ...), or the bare `score` passthrough.
    #   * profile_objective -- a column-joint (shape-comparison) objective: kl or
    #     wasserstein.
    # The whole-fit per-point `noise_model = <family>, ...` line is the third key; it
    # rides the structural ('noise_model', None) tuple, not a typed field. None == the
    # key was not named (each defaults None so presence == user intent at the read site
    # in _load_obj_func, which enforces "exactly one objective key").
    objective: Optional[str] = None
    profile_objective: Optional[str] = None
    output_dir: str = 'pybnf_output'
    delete_old_files: int = 1
    num_to_output: int = 5000
    output_every: int = 20
    initialization: str = 'lh'
    initialization_distribution: Literal['prior', 'bounds'] = 'prior'
    refine: int = 0
    # Which local optimizer polishes the best fit when refine == 1 (#403). One of
    # the start-point optimizer fit_type codes -- 'sim' (Nelder-Mead Simplex,
    # default + backward-compatible), 'powell' (conjugate-direction), or 'cmaes'
    # (CMA-ES). The chosen refiner's whole method schema is pulled into a non-self
    # fit's effective config as a coherent group (the refiner seam, ADR-0013/0015);
    # validated against the registry in Configuration. Run-level (not method-owned)
    # because it selects across methods.
    refine_method: str = 'sim'
    bng_command: str = Field(default_factory=_default_bng_command)
    smoothing: int = 1
    backup_every: int = 1
    time_course: Any = ()
    param_scan: Any = ()
    min_objective: float = float('-inf')
    bootstrap: int = 0
    bootstrap_max_obj: Optional[float] = None
    ind_var_rounding: int = 0
    local_objective_eval: int = 0
    constraint_scale: float = 1.0
    sbml_integrator: str = 'cvode'
    sbml_backend: str = 'roadrunner'
    bngl_backend: str = 'auto'
    sbml_ssa_strict: int = 1
    stochastic_seed: str = 'auto'
    parallel_count: Optional[int] = None
    save_best_data: int = 0
    # Opt-in (new-era only): also embed each time-indexed observable's experimental
    # data into the end-of-run Results/<model>_bestfit.bngl artifact as sidecar
    # .tfun reference functions, so the model self-contains its comparison curves
    # (ADR-0048). A no-op in legacy (the artifact itself is edition >= 2 only) and
    # when unset.
    embed_best_fit_data: int = 0
    simulation_dir: Optional[str] = None
    parallelize_models: int = 1
    starting_params: Any = None
    random_seed: Optional[int] = None

    # --- differential evolution ---
    # Migrated to DEFamilyConfig / DifferentialEvolutionConfig in
    # algorithms/optimizers/differential_evolution.py (Stage b); under narrowing
    # (ADR-0013) present only in a de/ade fit's effective config.

    # --- particle swarm ---
    # Migrated to PSOConfig in algorithms/optimizers/particle_swarm.py (Stage b);
    # under narrowing (ADR-0013) present only in a pso fit's effective config.

    # --- scatter search ---
    # Migrated to ScatterSearchConfig in algorithms/optimizers/scatter_search.py
    # (Stage b); under narrowing (ADR-0013) present only in an ss fit's config.

    # --- MCMC samplers ---
    # Migrated to MCMCFamilyConfig in algorithms/samplers/base.py (Stage b); under
    # narrowing (ADR-0013) present only in each MCMC fit_type's own config.
    # neg_bin_r stays here: it is an objfunc/noise param (read when objfunc=neg_bin
    # regardless of fit_type), NOT MCMC-family knowledge.
    neg_bin_r: float = 24.0
    # noise_location is likewise an objfunc/noise param: the whole-fit default
    # interpretation of the model prediction -- 'mean' or 'median' -- for the global
    # objfunc's noise model (ADR-0024). None -> each family's own default (median for
    # the location-scale families). Per-observable ``noise_model ... location =``
    # fields override it. Validated + applied in Configuration._load_obj_func.
    noise_location: Optional[str] = None

    # --- simplex ---
    # Migrated to SimplexConfig in algorithms/optimizers/simplex.py (Stage b);
    # under narrowing (ADR-0013) present in a sim fit's config, or in any fit with
    # refine == 1 via the refine->simplex overlay (the _REFINER_SCHEMA seam).

    # --- failure / wall-time / normalization ---
    max_failed_simulations: int = 100
    wall_time_gen: int = 3600
    wall_time_sim: Optional[int] = None   # chosen when loading models
    normalization: Any = None

    # --- cluster ---
    cluster_type: Optional[str] = None
    scheduler_node: Optional[str] = None
    scheduler_file: Optional[str] = None
    worker_nodes: Any = None
    output_trajectory: Any = None
    output_noise_trajectory: Any = None

    # --- DREAM family ---
    # Migrated to MCMCFamilyConfig in algorithms/samplers/base.py (Stage b); under
    # narrowing (ADR-0013) present only in a dream/p_dream fit's own config.


# The config keys the global schema is the source of truth for. Used by
# Configuration to split a raw config dict into global keys vs the selected
# method's keys vs pass-through extras (tuple free-parameter keys, model-path
# keys, required user keys, not-yet-migrated method keys, ...). Method-owned keys
# (e.g. PSO's) are NOT in this set; they live on the per-method schema.
SCHEMA_KEYS = GlobalConfig.owned_keys()


def _registered_schemas():
    """The distinct, non-None per-method schemas in the fit_type registry,
    deduplicated by identity (several codes may share one schema)."""
    seen = []
    for entry in FIT_TYPE_REGISTRY.values():
        schema = entry.schema
        if schema is not None and schema not in seen:
            seen.append(schema)
    return seen


def default_union():
    """The fully-defaulted config as a plain dict: the global defaults unioned
    with every registered method schema's defaults (ADR-0006).

    Historically the ``_build_config`` baseline (every fit_type saw every method's
    defaults); ADR-0013 narrowing removed it from that role, so this is now only
    the "every possible default" view behind the ``default_config()`` compat shim,
    not any fit's effective config. Requires the algorithm leaves to be imported
    so the registry is populated (``config.py`` does a side-effect ``from . import
    algorithms``); with an empty registry it degrades to the global defaults alone.
    """
    eff = GlobalConfig().model_dump(by_alias=True)
    for schema in _registered_schemas():
        eff.update(schema().model_dump(by_alias=True))
    return eff


def default_config_dict():
    """The fully-defaulted config as a plain dict (replaces the old hand-rolled
    ``default_config()`` literal; single source of truth). Returns the full union
    of global + per-method defaults so callers reading method keys off it keep
    working."""
    return default_union()


def build_effective_global(schema_subset):
    """Validate/coerce/default the global portion of a raw config and return it
    as a plain dict (keys by alias). ``schema_subset`` must contain only keys in
    :data:`SCHEMA_KEYS`. Raises ``pydantic.ValidationError`` on invalid input;
    the caller (``config.py``) translates that to ``PybnfError``.
    """
    return build_effective_method(GlobalConfig, schema_subset)


def build_effective_method(schema, schema_subset):
    """Validate/coerce/default the portion of a raw config owned by a per-method
    ``schema`` (a :class:`PyBNFConfigModel` subclass) and return it as a plain
    dict (keys by alias). ``schema_subset`` must contain only keys in
    ``schema.owned_keys()``. Raises ``pydantic.ValidationError`` on invalid
    input; the caller translates it to ``PybnfError``.
    """
    return schema(**schema_subset).model_dump(by_alias=True)
