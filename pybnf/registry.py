"""Self-registering dispatch tables mapping ``fit_type`` / ``objfunc`` codes to
the classes that implement them.

This module is intentionally **dependency-free** -- it imports nothing else from
PyBNF. The ``algorithms`` package imports ``objective`` (``base.py`` ->
``ObjectiveCalculator``, ``model_check.py`` -> ``ConstraintCounter``), so a
registry living under ``algorithms/`` would cycle the moment ``objective.py``
tried to import its decorator. Keeping the registry at the top level (and empty
of PyBNF imports) lets every method/objective module pull in its decorator
freely: leaves do ``from ...registry import register_fit_type``, ``objective.py``
does ``from .registry import register_objfunc``. See ADR-0005.

Population is the importer's job: importing ``pybnf.algorithms`` runs every leaf
and fires the ``@register_fit_type`` decorators; importing ``pybnf.objective``
fires the ``@register_objfunc`` ones. Both happen before either dispatcher runs.
"""

from dataclasses import dataclass, field


# --- fit_type registry (consumed by pybnf.pybnf._create_algorithm) -----------

@dataclass(frozen=True)
class FitTypeEntry:
    """One row of the ``fit_type`` table.

    ``cls`` is constructed as ``cls(config, **kwargs)``; ``kwargs`` carries
    variant flags (e.g. ``sa`` -> ``{'sa': True}``). ``family`` is one of
    ``optimizer`` | ``sampler`` | ``checker`` (the benchmark harness filters on
    it). ``deprecated`` drives a user-facing warning at dispatch -- the method
    still runs.
    """

    cls: type
    kwargs: dict = field(default_factory=dict)
    family: str = ''
    display_name: str = ''
    deprecated: bool = False


FIT_TYPE_REGISTRY: dict = {}


def register_fit_type(*codes, family, display_name, kwargs=None, deprecated=False):
    """Class decorator registering ``codes`` -> the decorated Algorithm class.

    May be stacked to bind several codes with differing metadata to one class
    (e.g. ``pt`` / ``mh`` / ``sa`` all map to ``BasicBayesMCMCAlgorithm`` but
    differ in ``deprecated`` / ``kwargs``).
    """
    def deco(cls):
        entry = FitTypeEntry(cls=cls, kwargs=dict(kwargs or {}), family=family,
                             display_name=display_name, deprecated=deprecated)
        for code in codes:
            FIT_TYPE_REGISTRY[code] = entry
        return cls
    return deco


# --- objfunc registry (consumed by config.Configuration._load_obj_func) ------

@dataclass(frozen=True)
class ObjFuncEntry:
    """One row of the ``objfunc`` table.

    ``config_args`` lists the config keys pulled positionally into ``cls(*args)``
    -- the per-objective construction recipe, needed because objective
    constructors are non-uniform: most take ``ind_var_rounding``, ``neg_bin``
    also takes ``neg_bin_r``, ``direct_pass`` takes nothing. This recipe
    disappears in M2.4, when NoiseModels take ``config`` uniformly. Cross-config
    validation (``neg_bin`` requires ``neg_bin_r``) stays in ``config.py``, not
    here.
    """

    cls: type
    config_args: tuple = ()


OBJFUNC_REGISTRY: dict = {}


def register_objfunc(*codes, config_args=()):
    """Class decorator registering ``codes`` -> the decorated objective class,
    recording the config keys its constructor consumes (pulled positionally)."""
    def deco(cls):
        entry = ObjFuncEntry(cls=cls, config_args=tuple(config_args))
        for code in codes:
            OBJFUNC_REGISTRY[code] = entry
        return cls
    return deco
