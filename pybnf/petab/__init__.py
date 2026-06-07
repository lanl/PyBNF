"""PEtab v2 problem importer (issue #407) -- the two-adapter proof.

A PEtab v2 problem (``problem.yaml`` + TSV tables + SBML model) and a native
``.conf`` should produce the *same* internal ``FreeParameter`` / ``Prior`` /
``NoiseModel`` objects (ADR-0004). This package is that thin adapter, built one
self-contained chunk at a time.

**Step 1 (here): the ``parameters`` table -> ``FreeParameter`` / ``Prior``.**
Dependency-free and simulator-free, so it runs in the bngsim-less CI tier. The
mapping is driven by the prior-family registry (ADR-0010) rather than a parallel
table, and PEtab/PyBNF boundaries are surfaced as explicit ``NotImplementedError``
(see :mod:`pybnf.petab.parameters`). Later chunks (observables -> NoiseModel,
measurements/conditions -> exp-data, the ``observableFormula`` sympy layer,
``problem.yaml`` + SBML wiring) adopt the ``petab`` library as an optional extra
where it pays for itself; Step 1 deliberately does not.
"""

from .parameters import (
    PetabParameterRow,
    free_parameter_from_row,
    free_parameters_from_file,
    free_parameters_from_table,
    read_parameter_table,
)

__all__ = [
    'PetabParameterRow',
    'free_parameter_from_row',
    'free_parameters_from_table',
    'free_parameters_from_file',
    'read_parameter_table',
]
