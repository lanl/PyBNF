"""Unit tests for the Pydantic config substrate (config_schema, ADR-0002).

These target the substrate's load-bearing details directly -- the things the
end-to-end golden net (test_config_golden.py) would catch only indirectly:
default parity with the old hand-rolled dict, the faithfulness tricks that keep
the effective config byte-identical (the ``lambda`` keyword alias,
``exchange_every`` holding ``inf``, ``adaptive_step_size`` staying an int), the
schema-vs-extras split in ``Configuration._build_config``, and the
``ValidationError -> PybnfError`` translation that is M2.1's new validation seam.
"""

import math

import pytest

from .context import config, config_schema, printing


class TestGlobalSchemaDefaults:
    def test_default_dict_is_fresh_plain_dict(self):
        a = config_schema.default_config_dict()
        b = config_schema.default_config_dict()
        assert type(a) is dict and a == b and a is not b
        # mutating one must not touch the other (no shared mutable defaults)
        a['credible_intervals'].append(99.0)
        assert b['credible_intervals'] == [68.0, 95.0]

    def test_schema_keys_are_global_only_union_adds_method_keys(self):
        # Stage (b): SCHEMA_KEYS is the GlobalConfig-owned set only; the full
        # effective union (default_config_dict) additionally carries every
        # registered method schema's keys (ADR-0006). PSO's keys left
        # GlobalConfig for PSOConfig but remain in the union for every fit_type.
        union = set(config_schema.default_config_dict())
        assert config_schema.SCHEMA_KEYS == config_schema.GlobalConfig.owned_keys()
        assert config_schema.SCHEMA_KEYS < union
        assert 'cognitive' not in config_schema.SCHEMA_KEYS  # PSO key, migrated out
        assert 'cognitive' in union                          # still in the union
        assert 'objfunc' in config_schema.SCHEMA_KEYS        # truly global, stays

    def test_representative_defaults_and_types(self):
        d = config_schema.default_config_dict()
        # int-literal default on a float field stays int (pydantic does not
        # validate defaults) -- byte-identical to the old default_config().
        assert d['adaptive_n_max'] == 30 and type(d['adaptive_n_max']) is int
        assert d['min_objective'] == float('-inf')
        assert d['adaptive_n_stop'] == float('inf')
        assert d['beta_max'] == float('inf')
        # bool default preserved as bool (not coerced to int)
        assert d['adaptive_step_size'] is True
        assert d['credible_intervals'] == [68.0, 95.0]
        assert d['beta'] == [1.0]
        assert d['exchange_every'] == 20
        assert d['random_seed'] is None
        assert d['bngl_backend'] == 'auto'
        assert d['initialization_distribution'] == 'prior'

    def test_lambda_keyword_is_emitted_by_alias(self):
        d = config_schema.default_config_dict()
        assert d['lambda'] == 0.1          # config key is 'lambda', not 'lambda_'
        assert 'lambda_' not in d


class TestBuildEffectiveGlobal:
    def test_user_value_overrides_default(self):
        eff = config_schema.build_effective_global({'backup_every': 5})
        assert eff['backup_every'] == 5
        assert eff['delete_old_files'] == 1   # untouched default still present

    def test_rejects_keys_owned_by_a_method_schema(self):
        # extra='forbid': a key that migrated out of GlobalConfig to a method schema
        # (e.g. MCMC's 'cooling') is not accepted here -- _build_config routes such
        # keys to build_effective_method instead.
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            config_schema.build_effective_global({'cooling': 0.5})


class TestBuildEffectiveMethod:
    """The faithfulness tricks live on whichever schema owns the field. After the
    Step-6 split they sit on the MCMC leaf models: the lambda alias and
    adaptive_step_size-int on DreamConfig, exchange_every-holds-inf on
    BasicMCMCConfig, and string coercion on any of them (here MCMCFamilyConfig's
    own burn_in). build_effective_method validates a raw subset against a schema."""

    def _eff(self, schema, subset):
        return config_schema.build_effective_method(schema, subset)

    def test_lambda_alias_roundtrips(self):
        from pybnf.algorithms.samplers.dream import DreamConfig
        eff = self._eff(DreamConfig, {'lambda': 0.25})
        assert eff['lambda'] == 0.25 and 'lambda_' not in eff

    def test_exchange_every_holds_infinity(self):
        # the MCMC family's postprocess assigns np.inf to exchange_every for non-PT
        # methods; the Any-typed field must accept it without re-coercion.
        from pybnf.algorithms.samplers.basic_mcmc import BasicMCMCConfig
        eff = self._eff(BasicMCMCConfig, {'exchange_every': float('inf')})
        assert math.isinf(eff['exchange_every'])

    def test_exchange_every_keeps_int_for_pt(self):
        from pybnf.algorithms.samplers.basic_mcmc import BasicMCMCConfig
        eff = self._eff(BasicMCMCConfig, {'exchange_every': 5})
        assert eff['exchange_every'] == 5 and type(eff['exchange_every']) is int

    def test_adaptive_step_size_preserves_int(self):
        # Any-typed so a user 0/1 stays an int (truthy), matching old behavior --
        # typing it bool would silently turn int 0/1 into False/True.
        from pybnf.algorithms.samplers.dream import DreamConfig
        eff = self._eff(DreamConfig, {'adaptive_step_size': 0})
        assert eff['adaptive_step_size'] == 0 and type(eff['adaptive_step_size']) is int

    def test_schema_coerces_string_numbers(self):
        # Forward-looking: once parse.py stops coercing, the schema owns it.
        from pybnf.algorithms.samplers.base import MCMCFamilyConfig
        eff = self._eff(MCMCFamilyConfig, {'burn_in': '250'})
        assert eff['burn_in'] == 250 and type(eff['burn_in']) is int


class TestConfigurationBuildConfig:
    """``Configuration._build_config`` splits schema keys from pass-through
    extras and translates pydantic errors."""

    def test_split_defaults_overrides_and_extras(self):
        raw = {
            'fit_type': 'sa',                     # sa owns cooling/beta_max
            'cooling': 0.3,                       # method key -> validated override
            'population_size': 10,                # required user key -> extra
            'beta_range': [0.1, 1.0],             # other method's key -> extra
            ('uniform_var', 'p1'): [-10.0, 10.0],  # free-param tuple key -> extra
            'model.bngl': ['a.exp'],              # model-path key -> extra
        }
        eff = config.Configuration._build_config(raw)
        # validated method override + an untouched same-schema default
        assert eff['cooling'] == 0.3
        assert eff['beta_max'] == float('inf')
        # every extra carried through unchanged
        assert eff['population_size'] == 10
        assert eff['beta_range'] == [0.1, 1.0]
        assert eff[('uniform_var', 'p1')] == [-10.0, 10.0]
        assert eff['model.bngl'] == ['a.exp']
        # narrowing dropped the foreign defaults: an unset MCMC default like burn_in
        # is no longer present on an sa fit (only sa's own keys + global + extras)
        assert 'burn_in' not in eff

    def test_validation_error_becomes_pybnferror(self):
        with pytest.raises(printing.PybnfError):
            config.Configuration._build_config({'random_seed': 'not-an-int'})

    def test_invalid_initialization_distribution_raises(self):
        with pytest.raises(printing.PybnfError):
            config.Configuration._build_config({'initialization_distribution': 'posterior'})

    def test_extras_never_clobber_validated_schema(self):
        # The split is disjoint, so a schema key is only ever validated, never
        # overwritten by an unvalidated extra of the same name.
        eff = config.Configuration._build_config({'cooling': 0.7})
        assert eff['cooling'] == 0.7

    def test_method_schema_validates_its_own_keys(self):
        # A migrated method's keys leave the extras bucket and are validated by
        # its co-located schema: a pso fit's PSO keys round-trip through PSOConfig
        # (here, string coercion proves it is the schema, not a raw passthrough).
        eff = config.Configuration._build_config({'fit_type': 'pso', 'cognitive': '2.5'})
        assert eff['cognitive'] == 2.5 and type(eff['cognitive']) is float
        # and the rest of the PSO default block comes from PSOConfig (its own
        # schema), not the dropped union baseline
        assert eff['social'] == 1.5

    def test_other_methods_keys_pass_through_as_extras(self):
        # A *different* method's keys are owned neither by GlobalConfig (migrated
        # out) nor by the selected fit's schema -> they ride through as extras,
        # unchanged, exactly as before Stage (b). Here: PSO's 'cognitive' on a de
        # fit.
        eff = config.Configuration._build_config({'fit_type': 'de', 'cognitive': 2.0})
        assert eff['cognitive'] == 2.0

    def test_de_island_keys_validated_for_de_pass_through_for_ade(self):
        # The DE shared-base split: de owns the island/migration fields (validated
        # by DifferentialEvolutionConfig); ade's own AsyncDEConfig does not own them.
        # Under narrowing (ADR-0013) an ade fit carries islands only when the user sets
        # it (then as an unchanged extra), not as a union default.
        de_eff = config.Configuration._build_config({'fit_type': 'de', 'islands': '4'})
        assert de_eff['islands'] == 4 and type(de_eff['islands']) is int  # schema-coerced
        assert config.Configuration._build_config({'fit_type': 'de'})['islands'] == 1  # de default
        ade_set = config.Configuration._build_config({'fit_type': 'ade', 'islands': 4})
        assert ade_set['islands'] == 4  # user-set extra rides through unchanged
        # ade opts into n_starts (its own AsyncDEConfig, #501) but not islands, so an ade
        # fit carries n_starts as a schema default while islands stays narrowed away.
        ade_default = config.Configuration._build_config({'fit_type': 'ade'})
        assert ade_default['n_starts'] == 1 and 'islands' not in ade_default


class TestRegistrySchemaSeam:
    """``_build_config`` reaches each method's schema through
    ``FitTypeEntry.schema`` (ADR-0006). Migrated methods carry a schema; the rest
    are still ``None`` and have their keys pass through as extras."""

    def test_pso_entry_carries_its_schema(self):
        from pybnf.algorithms.optimizers.particle_swarm import PSOConfig
        from pybnf.registry import FIT_TYPE_REGISTRY
        assert FIT_TYPE_REGISTRY['pso'].schema is PSOConfig

    def test_de_family_shares_a_base_schema(self):
        # Shared-base pattern (ADR-0006): both de and ade extend the key-minimal family
        # base with their own subclass -- de adds the island/migration fields, ade adds
        # nothing but the shared n_starts field. The base itself stays key-minimal.
        from pybnf.algorithms.optimizers.differential_evolution import (
            AsyncDEConfig, DEFamilyConfig, DifferentialEvolutionConfig)
        from pybnf.registry import FIT_TYPE_REGISTRY
        assert FIT_TYPE_REGISTRY['ade'].schema is AsyncDEConfig
        assert FIT_TYPE_REGISTRY['de'].schema is DifferentialEvolutionConfig
        assert issubclass(DifferentialEvolutionConfig, DEFamilyConfig)
        assert issubclass(AsyncDEConfig, DEFamilyConfig)
        assert 'mutation_rate' in DEFamilyConfig.owned_keys()
        assert 'islands' in DifferentialEvolutionConfig.owned_keys()
        assert 'islands' not in DEFamilyConfig.owned_keys()
        assert 'islands' not in AsyncDEConfig.owned_keys()    # async DE has no islands
        # Multi-start (#498): the n_starts field rides each method's own subclass, not the
        # shared base -- so the ADR-0006 "ade adds no keys to the family base" seam stays
        # intact while both de (ADR-0071) and ade (#501) opt in.
        assert 'n_starts' in DifferentialEvolutionConfig.owned_keys()
        assert 'n_starts' in AsyncDEConfig.owned_keys()
        assert 'n_starts' not in DEFamilyConfig.owned_keys()
        # ade adds exactly n_starts beyond the shared family base (nothing else).
        assert AsyncDEConfig.owned_keys() - DEFamilyConfig.owned_keys() == {'n_starts'}

    def test_ss_owns_local_min_limit_and_n_starts(self):
        # ss's defaulted keys are local_min_limit and the shared n_starts multi-start
        # field (#498/ADR-0071); init_size/reserve_size are runtime-defaulted, so they
        # are NOT owned by the schema (stay extras).
        from pybnf.algorithms.optimizers.scatter_search import ScatterSearchConfig
        from pybnf.registry import FIT_TYPE_REGISTRY
        assert FIT_TYPE_REGISTRY['ss'].schema is ScatterSearchConfig
        assert ScatterSearchConfig.owned_keys() == {'local_min_limit', 'n_starts'}

    def test_sim_owns_only_defaulted_simplex_keys(self):
        # Simplex owns the six unconditionally-read simplex_* knobs plus the shared
        # n_starts multi-start field (MultiStartConfig, #498/ADR-0072-- sim runs n_starts
        # concurrent starts in box mode); simplex_log_step / simplex_max_iterations are
        # runtime-guarded and simplex_start_point is internal, so none of those are owned.
        from pybnf.algorithms.optimizers.simplex import SimplexConfig
        from pybnf.registry import FIT_TYPE_REGISTRY
        assert FIT_TYPE_REGISTRY['sim'].schema is SimplexConfig
        assert SimplexConfig.owned_keys() == {
            'simplex_step', 'simplex_reflection', 'simplex_expansion',
            'simplex_contraction', 'simplex_shrink', 'simplex_stop_tol', 'n_starts'}

    def test_powell_owns_n_starts_multistart_field(self):
        # Powell owns its three defaulted powell_* knobs plus the shared n_starts field
        # (MultiStartConfig, #498/ADR-0072): powell runs n_starts concurrent starts in box
        # mode. powell_max_iterations is runtime-guarded; powell_start_point is internal.
        from pybnf.algorithms.optimizers.powell import PowellConfig
        from pybnf.registry import FIT_TYPE_REGISTRY
        assert FIT_TYPE_REGISTRY['powell'].schema is PowellConfig
        assert PowellConfig.owned_keys() == {
            'powell_step', 'powell_line_tol', 'powell_stop_tol', 'n_starts'}

    def test_simplex_group_present_only_when_refine_pulls_it_in(self):
        # ADR-0013 narrowing: a non-sim fit no longer carries simplex_* by default
        # (foreign defaults are dropped). refine == 1 pulls in the WHOLE Simplex
        # schema as a coherent all-or-nothing group, so _refine_best_fit never meets
        # a half-populated state -- the one cross-fit_type reach, the _REFINER_SCHEMA
        # seam in config.py.
        plain = config.Configuration._build_config({'fit_type': 'de'})
        assert 'simplex_step' not in plain and 'simplex_stop_tol' not in plain
        refined = config.Configuration._build_config({'fit_type': 'de', 'refine': 1})
        # all six knobs appear together at their defaults (coherent group)
        assert refined['simplex_step'] == 1.0 and refined['simplex_stop_tol'] == 0.0
        assert refined['simplex_reflection'] == 1.0 and refined['simplex_shrink'] == 0.5
        # a user-set simplex key overrides within the group; the siblings still default
        over = config.Configuration._build_config(
            {'fit_type': 'de', 'refine': 1, 'simplex_step': 0.3})
        assert over['simplex_step'] == 0.3 and over['simplex_contraction'] == 0.5
        # sim carries the group via its own schema -- no refine needed
        sim = config.Configuration._build_config({'fit_type': 'sim'})
        assert sim['simplex_step'] == 1.0

    def test_mcmc_leaves_subclass_the_family_and_inherit_the_beta_ladder(self):
        # Step 6: each MCMC code maps to its own leaf model, all subclassing
        # MCMCFamilyConfig. mh/pt/am inherit the β-ladder postprocess verbatim;
        # dream/p_dream OVERRIDE it to add the step_size->adaptive_step_size coupling
        # (ADR-0013), but still run the β-ladder via super().postprocess().
        from pybnf.algorithms.samplers.base import MCMCFamilyConfig
        from pybnf.algorithms.samplers.basic_mcmc import BasicMCMCConfig
        from pybnf.algorithms.samplers.adaptive_mcmc import AdaptiveMCMCConfig
        from pybnf.algorithms.samplers.dream import DreamConfig
        from pybnf.algorithms.samplers.pdream import PDreamConfig
        from pybnf.registry import FIT_TYPE_REGISTRY
        # sa is NOT here: M2.2 (ADR-0008) moved it out of the MCMC family into a
        # standalone SimulatedAnnealing optimizer (see test_sa_schema_is_standalone).
        inherit = {'mh': BasicMCMCConfig, 'pt': BasicMCMCConfig, 'am': AdaptiveMCMCConfig}
        override = {'dream': DreamConfig, 'p_dream': PDreamConfig}
        for code, leaf in {**inherit, **override}.items():
            assert FIT_TYPE_REGISTRY[code].schema is leaf, code
            assert issubclass(leaf, MCMCFamilyConfig)
        for leaf in inherit.values():  # β-ladder inherited verbatim
            assert leaf.postprocess.__func__ is MCMCFamilyConfig.postprocess.__func__
        for leaf in override.values():  # DREAM overrides, but not with the no-op base
            assert leaf.postprocess.__func__ is not MCMCFamilyConfig.postprocess.__func__
            assert leaf.postprocess.__func__ is not \
                config_schema.PyBNFConfigModel.postprocess.__func__
        # p_dream inherits dream's override (does not re-override)
        assert PDreamConfig.postprocess.__func__ is DreamConfig.postprocess.__func__
        assert PDreamConfig.__mro__[1] is DreamConfig  # p_dream extends dream
        # the family base carries the hook; PyBNFConfigModel's is the no-op
        assert MCMCFamilyConfig.postprocess.__func__ is not \
            config_schema.PyBNFConfigModel.postprocess.__func__

    def test_sa_schema_is_standalone_not_mcmc_family(self):
        # M2.2 (ADR-0008): sa is a true optimizer, so its config is a standalone
        # PyBNFConfigModel (NOT MCMCFamilyConfig) and it inherits the no-op
        # postprocess -- the β-ladder no longer runs for sa.
        from pybnf.algorithms.samplers.base import MCMCFamilyConfig
        from pybnf.algorithms.optimizers.simulated_annealing import SimulatedAnnealingConfig
        from pybnf.registry import FIT_TYPE_REGISTRY
        assert FIT_TYPE_REGISTRY['sa'].schema is SimulatedAnnealingConfig
        assert FIT_TYPE_REGISTRY['sa'].family == 'optimizer'
        assert issubclass(SimulatedAnnealingConfig, config_schema.PyBNFConfigModel)
        assert not issubclass(SimulatedAnnealingConfig, MCMCFamilyConfig)
        assert SimulatedAnnealingConfig.postprocess.__func__ is \
            config_schema.PyBNFConfigModel.postprocess.__func__
        assert SimulatedAnnealingConfig.owned_keys() == {'step_size', 'beta', 'cooling', 'beta_max'}

    def test_mcmc_key_ownership_partition(self):
        # Each MCMC key is owned by exactly the leaf whose algorithm reads it;
        # neg_bin_r stays global (an objfunc param), never on the family.
        from pybnf.algorithms.samplers.base import MCMCFamilyConfig
        from pybnf.algorithms.samplers.basic_mcmc import BasicMCMCConfig
        from pybnf.algorithms.samplers.dream import DreamConfig
        assert {'step_size', 'beta', 'burn_in'} <= MCMCFamilyConfig.owned_keys()
        assert 'exchange_every' in BasicMCMCConfig.owned_keys()
        assert 'exchange_every' not in MCMCFamilyConfig.owned_keys()
        assert {'gamma_prob', 'lambda'} <= DreamConfig.owned_keys()
        assert 'gamma_prob' not in MCMCFamilyConfig.owned_keys()
        assert 'neg_bin_r' not in DreamConfig.owned_keys()
        assert 'neg_bin_r' in config_schema.SCHEMA_KEYS  # stays global

    def test_beta_ladder_runs_through_build_config(self):
        # The beta-ladder hook fires inside _build_config on the raw dict: a non-pt
        # MCMC fit gets exchange_every -> inf and a beta_list built from beta.
        eff = config.Configuration._build_config(
            {'fit_type': 'am', 'population_size': 4, 'beta': [1.0]})
        assert math.isinf(eff['exchange_every'])      # postprocess set inf (non-pt)
        assert eff['reps_per_beta'] == 1
        assert eff['beta_list'] == [1.0, 1.0, 1.0, 1.0]  # one beta x subpop_size 4
        # a non-MCMC fit's no-op postprocess leaves no beta_list
        de_eff = config.Configuration._build_config({'fit_type': 'de'})
        assert 'beta_list' not in de_eff

    def test_migrated_methods_so_far(self):
        # Stage (b) migrates one method/family per step: pso (Step 1), de+ade
        # (Step 2), ss (Step 3), sim (Step 4), the whole MCMC family (Step 5).
        # powell + cmaes land co-located with their own schemas (#403/ADR-0015).
        # The gradient-based optimizers (#386/#481) land with their own schemas: trf
        # (trust-region least-squares) with TRFConfig, lbfgs (L-BFGS-B) with LBFGSConfig,
        # and gntr (general-objective Fisher/Gauss-Newton trust region) with GNTRConfig.
        # profile_likelihood (#446/#466) lands with ProfileLikelihoodConfig.
        # Only 'check' remains unmigrated. Each step extends this set -- a ratchet.
        from pybnf.registry import FIT_TYPE_REGISTRY
        migrated = {c for c, e in FIT_TYPE_REGISTRY.items() if e.schema is not None}
        assert migrated == {'pso', 'de', 'ade', 'ss', 'sim', 'powell', 'cmaes',
                            'mh', 'pt', 'sa', 'am', 'dream', 'p_dream', 'hmc', 'trf', 'lbfgs',
                            'gntr', 'profile_likelihood'}
        assert FIT_TYPE_REGISTRY['check'].schema is None


class TestParserSchemaNumericInvariant:
    """The two hand-maintained sources of PyBNF's config-key knowledge must agree:
    ``parse.py``'s ``numkeys_int`` / ``numkeys_float`` lists (the *structural* parser --
    a key absent there is rejected as "not a valid configuration key" before the schema
    is ever consulted) and the Pydantic method schemas (the *type* layer). Every numeric
    field a registered method schema declares therefore has to appear in the matching
    parser list, or the key can only ever be defaulted -- never set from a ``.conf``.

    This invariant guards that seam. It is the net that was missing when the ``trf_*`` /
    ``lbfgs_*`` tunables (and ``powell_line_tol``) were added to their schemas but not to
    ``parse.py`` -- schema-known yet unparseable (#386 follow-up)."""

    @staticmethod
    def _numeric_kind(annotation):
        """``'int'`` / ``'float'`` for an int/float (or ``Optional`` thereof) field, else
        ``None`` -- ``bool`` (an int subclass but not a numeric knob), ``str``, ``list``,
        ``Any``, ``Literal``, and a union of several real types are all skipped."""
        import types
        import typing
        origin = typing.get_origin(annotation)
        if origin is typing.Union or origin is getattr(types, 'UnionType', None):
            args = [a for a in typing.get_args(annotation) if a is not type(None)]
            if len(args) != 1:
                return None
            annotation = args[0]
        if annotation is bool:
            return None
        if annotation is int:
            return 'int'
        if annotation is float:
            return 'float'
        return None

    def test_every_schema_numeric_field_is_in_the_matching_parser_list(self):
        # Each int-typed field (incl. Optional[int]) must be in parse.numkeys_int and
        # each float-typed field in parse.numkeys_float -- keyed by the field's config
        # alias (e.g. lambda_ -> 'lambda'), which is what a .conf actually writes.
        from pybnf import parse
        missing = []
        for schema in config_schema._registered_schemas():
            for name, field in schema.model_fields.items():
                key = field.alias or name
                kind = self._numeric_kind(field.annotation)
                if kind == 'int' and key not in parse.numkeys_int:
                    missing.append((schema.__name__, key, 'numkeys_int'))
                elif kind == 'float' and key not in parse.numkeys_float:
                    missing.append((schema.__name__, key, 'numkeys_float'))
        assert not missing, (
            'schema numeric fields absent from the parse.py key lists (structurally '
            'unparseable from a .conf, only defaultable): %s' % missing)

    def test_every_max_iterations_runtime_budget_is_int_parseable(self):
        # The *_max_iterations cycle budgets are RUNTIME_KEYS (runtime-defaulted to the
        # global max_iterations, so not schema fields), but they are still user-settable
        # int keys -- so they too must live in numkeys_int. The other runtime keys are
        # internal *_start_point injections / non-numeric inputs, exempt by construction.
        from pybnf import parse
        missing = []
        for schema in config_schema._registered_schemas():
            for key in schema.runtime_keys():
                if key.endswith('_max_iterations') and key not in parse.numkeys_int:
                    missing.append((schema.__name__, key))
        assert not missing, (
            '*_max_iterations runtime budgets absent from parse.numkeys_int: %s' % missing)
