"""Pydantic config schema — the typed substrate behind ``Configuration`` (ADR-0002).

M2.1 turns PyBNF's configuration knowledge from scattered, untyped sources (the
``parse.py`` key-type lists, the hand-rolled ``default_config()`` dict, the
per-fit_type preprocessing) into a typed Pydantic model that owns the defaults,
types, and coercion in one place. This module holds the **global** schema and the
shared :class:`PyBNFConfigModel` base; the per-method (family) schemas subclass
that base and co-locate with their algorithm classes (M2.1 Stage b, ADR-0006),
reached by ``Configuration._build_config`` through the registry
(``FitTypeEntry.schema``).

The effective config stays the **full union** of every method's defaults for
every fit_type (ADR-0006): :func:`default_union` collects the global defaults
plus every registered method schema's defaults, so a ``de`` fit still carries
``cognitive`` / ``particle_weight`` / ... exactly as before. Per-method narrowing
is deferred to Stage (c).

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
* Values reaching the model have already been coerced by ``parse.ploop`` (the
  ``numkeys_int`` / ``numkeys_float`` / ``mult*`` lists), so per-field coercion
  here is effectively a no-op today; it becomes the sole owner once the
  ``parse.py`` token lists migrate into the schema.
* ``exchange_every`` is typed ``Any``: ``parse`` coerces it to ``int``, but
  ``postprocess_mcmc_keys`` overwrites it with ``np.inf`` for the non-PT methods,
  so the field must hold both an int and an infinity without re-coercion.
* ``lambda`` is a Python keyword, so its field is ``lambda_`` with
  ``alias='lambda'``; the model is dumped ``by_alias=True`` to emit ``'lambda'``.

Keys a user may legitimately set that are *not* in this schema (the required
``population_size`` / ``max_iterations``, method-only keys like ``beta_range`` /
``init_size``, and the structural model/exp/free-parameter keys) are not modeled
here: ``Configuration`` carries them through as "extras" alongside the dumped
schema. They gain typed validation as method schemas land in Stage (b).
"""

from typing import Any, Optional

import os

from pydantic import BaseModel, ConfigDict, Field

from .registry import FIT_TYPE_REGISTRY


def _default_bng_command():
    """Reproduce ``default_config()``'s BNGPATH-derived default exactly."""
    try:
        return os.environ['BNGPATH'] + '/BNG2.pl'
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

    @classmethod
    def owned_keys(cls):
        """The config-key names (aliases where defined) this model owns."""
        return frozenset(
            (f.alias or name) for name, f in cls.model_fields.items()
        )

    @classmethod
    def postprocess(cls, conf_dict, fit_type):
        """No-op preprocessing hook (ADR-0006). The MCMC-family schema overrides
        this with ``postprocess_mcmc_keys`` (the beta ladder); every other model
        inherits this no-op. Mutates and returns the raw config dict."""
        return conf_dict


class GlobalConfig(PyBNFConfigModel):
    """The global (run-level) configuration defaults, typed.

    Field order and grouping mirror the old ``Configuration.default_config()``
    so the two stay easy to diff. The per-method groups (PSO, DE, ... ) lift out
    into per-method schemas as Stage (b) migrates each method; what remains here
    is the truly run-level surface plus not-yet-migrated method blocks.
    """

    # --- global / run-level ---
    objfunc: str = 'chi_sq'
    output_dir: str = 'pybnf_output'
    delete_old_files: int = 1
    num_to_output: int = 5000
    output_every: int = 20
    initialization: str = 'lh'
    refine: int = 0
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
    simulation_dir: Optional[str] = None
    parallelize_models: int = 1
    starting_params: Any = None
    random_seed: Optional[int] = None

    # --- differential evolution ---
    # Migrated to DEFamilyConfig / DifferentialEvolutionConfig in
    # algorithms/optimizers/differential_evolution.py (Stage b); still present in
    # the effective config for every fit_type via default_union().

    # --- particle swarm ---
    # Migrated to PSOConfig in algorithms/optimizers/particle_swarm.py (Stage b);
    # still present in the effective config for every fit_type via default_union().

    # --- scatter search ---
    # Migrated to ScatterSearchConfig in algorithms/optimizers/scatter_search.py
    # (Stage b); still present in the effective config via default_union().

    # --- MCMC samplers ---
    step_size: float = 0.2
    burn_in: int = 10000
    sample_every: int = 100
    output_hist_every: int = 100
    hist_bins: int = 10
    adaptive: int = 10000
    credible_intervals: list = Field(default_factory=lambda: [68.0, 95.0])
    beta: list = Field(default_factory=lambda: [1.0])
    exchange_every: Any = 20         # int from parse, but np.inf after postprocess
    beta_max: float = float('inf')
    cooling: float = 0.01
    continue_run: int = 0
    neg_bin_r: float = 24.0
    stablizingCov: float = 0.001
    calculate_covari: Any = None

    # --- simplex ---
    # Migrated to SimplexConfig in algorithms/optimizers/simplex.py (Stage b);
    # still present in the effective config for every fit_type via default_union()
    # (the refine->simplex cross-method reach relies on it).

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
    gamma_prob: float = 0.1
    zeta: float = 1e-6
    lambda_: float = Field(0.1, alias='lambda')
    crossover_number: int = 3
    adaptive_step_size: Any = True   # bool default; user may pass int 0/1
    archive_size: Optional[int] = None
    archive_thin_rate: int = 10
    snooker_prob: float = 0.1
    delta: int = 1
    outlier_method: str = 'iqr'
    rhat_threshold: float = 0.0
    diagnostics_every: int = 0
    precondition_adapt: Any = None


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

    This preserves the historical full-union effective config -- every fit_type
    sees every method's defaults -- even though each method now owns its own
    fields. Requires the algorithm leaves to be imported so the registry is
    populated (``config.py`` does a side-effect ``from . import algorithms``);
    with an empty registry it degrades to the global defaults alone.
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
