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
    still runs. ``schema`` is the method's co-located Pydantic config model
    (``PyBNFConfigModel`` subclass) that ``config._build_config`` validates this
    fit_type's keys against (M2.1 Stage b, ADR-0006); ``None`` until that method
    has been migrated, in which case its keys stay pass-through extras.
    ``refiner`` marks a start-point local optimizer that ``refine = 1`` may run as
    a post-fit polish (``refine_method``, #403, ADR-0015) -- ``sim`` / ``powell`` /
    ``cmaes``; the chosen refiner's schema is the coherent group the refiner seam
    in ``config.py`` pulls into a non-self fit's effective config.
    """

    cls: type
    kwargs: dict = field(default_factory=dict)
    family: str = ''
    display_name: str = ''
    deprecated: bool = False
    schema: type = None
    refiner: bool = False


FIT_TYPE_REGISTRY: dict = {}


def register_fit_type(*codes, family, display_name, kwargs=None, deprecated=False,
                      schema=None, refiner=False):
    """Class decorator registering ``codes`` -> the decorated Algorithm class.

    May be stacked to bind several codes with differing metadata to one class
    (e.g. ``pt`` / ``mh`` / ``sa`` all map to ``BasicBayesMCMCAlgorithm`` but
    differ in ``deprecated`` / ``kwargs``). ``schema`` is the method's
    co-located Pydantic config model (M2.1 Stage b, ADR-0006). ``refiner`` marks a
    fit_type usable as a ``refine_method`` (#403, ADR-0015).
    """
    def deco(cls):
        entry = FitTypeEntry(cls=cls, kwargs=dict(kwargs or {}), family=family,
                             display_name=display_name, deprecated=deprecated,
                             schema=schema, refiner=refiner)
        for code in codes:
            FIT_TYPE_REGISTRY[code] = entry
        return cls
    return deco


# --- objfunc registry (consumed by config.Configuration._load_obj_func) ------

@dataclass(frozen=True)
class ObjFuncEntry:
    """One row of the ``objfunc`` table: a code -> its objective class.

    Construction is uniform: ``config.Configuration._load_obj_func`` calls
    ``entry.cls.from_config(config)`` for every code (ADR-0011, M2.4). The old
    per-objfunc positional ``config_args`` recipe is gone -- each class reads what
    it needs from config in its own ``from_config`` classmethod. Cross-config
    validation (``neg_bin`` requires ``neg_bin_r``) stays in ``config.py``, not
    here.
    """

    cls: type


OBJFUNC_REGISTRY: dict = {}


def register_objfunc(*codes):
    """Class decorator registering ``codes`` -> the decorated objective class.
    The class builds itself from the config via its ``from_config`` classmethod."""
    def deco(cls):
        entry = ObjFuncEntry(cls=cls)
        for code in codes:
            OBJFUNC_REGISTRY[code] = entry
        return cls
    return deco


# --- prior-family registry (consumed by pybnf.priors + parse.py + config) -----

@dataclass(frozen=True)
class PriorFamilyEntry:
    """One row of the prior-family table (ADR-0010, M2.3).

    ``cls`` is a ``pybnf.priors.Prior`` subclass (a distribution family in the
    sampling space ``u``); ``base_name`` is the family's config stem, from which
    the regular keyword pair ``{base}_var`` (linear) and ``log{base}_var``
    (log10) is generated; ``has_bounded_support`` (mirrored off the class) says
    whether the family has finite support -- which both drives the ``parse.py``
    grammar partition (bounded keywords take the optional ``b``/``u`` flag) and
    decides reflecting-bounds eligibility on ``FreeParameter``.
    """

    cls: type
    base_name: str
    has_bounded_support: bool


PRIOR_FAMILY_REGISTRY: dict = {}


def register_prior_family(base_name):
    """Class decorator registering a ``Prior`` family under ``base_name``.

    ``has_bounded_support`` is read off the decorated class so it is declared in
    exactly one place. Population is the importer's job: importing
    ``pybnf.priors`` runs every family leaf and fires these decorators before
    ``PRIOR_KEYWORD_MAP`` / the grammar are built. See ADR-0010, ADR-0005."""
    def deco(cls):
        PRIOR_FAMILY_REGISTRY[base_name] = PriorFamilyEntry(
            cls=cls, base_name=base_name, has_bounded_support=cls.has_bounded_support)
        return cls
    return deco
