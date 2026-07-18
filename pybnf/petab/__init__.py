"""PEtab v2 problem importer (issue #407) -- the two-adapter proof.

A PEtab v2 problem (``problem.yaml`` + TSV tables + SBML model) and a native
``.conf`` should produce the *same* internal ``FreeParameter`` / ``Prior`` /
``NoiseModel`` objects (ADR-0004). This package is that thin adapter, built one
self-contained chunk at a time.

**Step 1: the ``parameters`` table -> ``FreeParameter`` / ``Prior``** (ADR-0019,
see :mod:`pybnf.petab.parameters`). **Step 2: the ``observables`` table noise half
-> ``(NoiseModel, SigmaSource)``** (ADR-0023, see :mod:`pybnf.petab.observables`),
on the decoupled ``(family x sigma-source)`` engine #410 built (ADR-0021). Both are
dependency-free and simulator-free, so they run in the bngsim-less CI tier; the
mapping is registry/constructor-driven rather than a parallel table, and
PEtab/PyBNF boundaries are surfaced as explicit ``NotImplementedError``. Later
chunks (the ``observableFormula`` sympy layer, measurements/conditions -> exp-data,
``problem.yaml`` + SBML wiring) adopt the ``petab`` library as an optional extra
where it pays for itself; these first chunks deliberately do not.
"""

from .conditions import (
    PetabConditionRow,
    PetabExperimentRow,
    condition_name_from_id,
    conditions_from_rows,
    read_condition_table,
    read_experiment_table,
)
from .convert import petab1to2_preserve_scale
from .export import clean_model_for_petab, export_job, write_problem_yaml
from .import_ import import_job, read_problem_yaml
from .measurements import (
    PetabMeasurementRow,
    data_from_measurement_rows,
    measurement_rows_from_data,
    read_measurement_table,
    write_measurement_table,
)
from .observables import (
    PetabObservableRow,
    noise_model_from_row,
    noise_models_from_file,
    noise_models_from_table,
    petab_observable_row,
    read_observable_table,
    write_observable_table,
)
from .parameters import (
    PetabParameterRow,
    free_parameter_from_row,
    free_parameters_from_file,
    free_parameters_from_table,
    petab_parameter_row,
    read_parameter_table,
    write_parameter_table,
)

__all__ = [
    # parameters (import + export)
    'PetabParameterRow',
    'free_parameter_from_row',
    'free_parameters_from_table',
    'free_parameters_from_file',
    'read_parameter_table',
    'petab_parameter_row',
    'write_parameter_table',
    # observables (import + export)
    'PetabObservableRow',
    'noise_model_from_row',
    'noise_models_from_table',
    'noise_models_from_file',
    'read_observable_table',
    'petab_observable_row',
    'write_observable_table',
    # measurements (import + export)
    'PetabMeasurementRow',
    'measurement_rows_from_data',
    'data_from_measurement_rows',
    'read_measurement_table',
    'write_measurement_table',
    # conditions / experiments (import + export)
    'PetabConditionRow',
    'PetabExperimentRow',
    'read_condition_table',
    'read_experiment_table',
    'conditions_from_rows',
    'condition_name_from_id',
    # job exporter
    'export_job',
    'clean_model_for_petab',
    'write_problem_yaml',
    # job importer
    'import_job',
    'read_problem_yaml',
    # scale-preserving v1 -> v2 converter
    'petab1to2_preserve_scale',
]
