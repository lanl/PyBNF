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

    def test_lambda_keyword_is_emitted_by_alias(self):
        d = config_schema.default_config_dict()
        assert d['lambda'] == 0.1          # config key is 'lambda', not 'lambda_'
        assert 'lambda_' not in d


class TestBuildEffectiveGlobal:
    def test_user_value_overrides_default(self):
        eff = config_schema.build_effective_global({'cooling': 0.5})
        assert eff['cooling'] == 0.5
        assert eff['delete_old_files'] == 1   # untouched default still present

    def test_lambda_alias_roundtrips(self):
        eff = config_schema.build_effective_global({'lambda': 0.25})
        assert eff['lambda'] == 0.25 and 'lambda_' not in eff

    def test_exchange_every_holds_infinity(self):
        # postprocess_mcmc_keys assigns np.inf to exchange_every for non-PT
        # methods; the field must accept it without re-coercion.
        eff = config_schema.build_effective_global({'exchange_every': float('inf')})
        assert math.isinf(eff['exchange_every'])

    def test_exchange_every_keeps_int_for_pt(self):
        eff = config_schema.build_effective_global({'exchange_every': 5})
        assert eff['exchange_every'] == 5 and type(eff['exchange_every']) is int

    def test_adaptive_step_size_preserves_int(self):
        # Any-typed so a user 0/1 stays an int (truthy), matching old behavior --
        # typing it bool would silently turn int 0/1 into False/True.
        eff = config_schema.build_effective_global({'adaptive_step_size': 0})
        assert eff['adaptive_step_size'] == 0 and type(eff['adaptive_step_size']) is int

    def test_schema_coerces_string_numbers(self):
        # Forward-looking: once parse.py stops coercing, the schema owns it.
        eff = config_schema.build_effective_global({'burn_in': '250'})
        assert eff['burn_in'] == 250 and type(eff['burn_in']) is int


class TestConfigurationBuildConfig:
    """``Configuration._build_config`` splits schema keys from pass-through
    extras and translates pydantic errors."""

    def test_split_defaults_overrides_and_extras(self):
        raw = {
            'cooling': 0.3,                       # schema key -> validated override
            'population_size': 10,                # required user key -> extra
            'beta_range': [0.1, 1.0],             # method-only key -> extra
            ('uniform_var', 'p1'): [-10.0, 10.0],  # free-param tuple key -> extra
            'model.bngl': ['a.exp'],              # model-path key -> extra
        }
        eff = config.Configuration._build_config(raw)
        # validated schema override + an untouched default
        assert eff['cooling'] == 0.3
        assert eff['burn_in'] == 10000
        # every extra carried through unchanged
        assert eff['population_size'] == 10
        assert eff['beta_range'] == [0.1, 1.0]
        assert eff[('uniform_var', 'p1')] == [-10.0, 10.0]
        assert eff['model.bngl'] == ['a.exp']

    def test_validation_error_becomes_pybnferror(self):
        with pytest.raises(printing.PybnfError):
            config.Configuration._build_config({'random_seed': 'not-an-int'})

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
        # and the full PSO default block is still present (union baseline)
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
        # by DifferentialEvolutionConfig); ade uses only the family base, so for
        # an ade fit those keys ride through as extras (still present via union).
        de_eff = config.Configuration._build_config({'fit_type': 'de', 'islands': '4'})
        assert de_eff['islands'] == 4 and type(de_eff['islands']) is int  # schema-coerced
        ade_eff = config.Configuration._build_config({'fit_type': 'ade', 'islands': 4})
        assert ade_eff['islands'] == 4  # union default (1) overridden by the extra


class TestRegistrySchemaSeam:
    """``_build_config`` reaches each method's schema through
    ``FitTypeEntry.schema`` (ADR-0006). Migrated methods carry a schema; the rest
    are still ``None`` and have their keys pass through as extras."""

    def test_pso_entry_carries_its_schema(self):
        from pybnf.algorithms.optimizers.particle_swarm import PSOConfig
        from pybnf.registry import FIT_TYPE_REGISTRY
        assert FIT_TYPE_REGISTRY['pso'].schema is PSOConfig

    def test_de_family_shares_a_base_schema(self):
        # Shared-base pattern (ADR-0006): ade registers against the family base
        # directly (adds no keys); de extends it with the island/migration fields.
        from pybnf.algorithms.optimizers.differential_evolution import (
            DEFamilyConfig, DifferentialEvolutionConfig)
        from pybnf.registry import FIT_TYPE_REGISTRY
        assert FIT_TYPE_REGISTRY['ade'].schema is DEFamilyConfig
        assert FIT_TYPE_REGISTRY['de'].schema is DifferentialEvolutionConfig
        assert issubclass(DifferentialEvolutionConfig, DEFamilyConfig)
        assert 'mutation_rate' in DEFamilyConfig.owned_keys()
        assert 'islands' in DifferentialEvolutionConfig.owned_keys()
        assert 'islands' not in DEFamilyConfig.owned_keys()

    def test_migrated_methods_so_far(self):
        # Stage (b) migrates one method/family per step: pso (Step 1), de+ade
        # (Step 2). The rest still pass their keys through. Each later step adds
        # to this set -- a deliberate per-step ratchet.
        from pybnf.registry import FIT_TYPE_REGISTRY
        migrated = {c for c, e in FIT_TYPE_REGISTRY.items() if e.schema is not None}
        assert migrated == {'pso', 'de', 'ade'}
        assert FIT_TYPE_REGISTRY['ss'].schema is None
        assert FIT_TYPE_REGISTRY['check'].schema is None
