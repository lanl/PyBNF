#!/usr/bin/env python
"""Regenerate the broken (and clean) PEtab v2 fixtures for the lint clinic.

This is a **developer tool**, not part of the test run. It materializes one tiny,
self-contained BNGL-native PEtab v2 problem per subdirectory: a ``clean/``
baseline that lints without complaint, and a gallery of ``*/`` variants each
carrying exactly ONE defect that a specific ``petab.v2.lint`` task must flag.

The point is dogfooding: PyBNF registers a BNGL model loader into ``petab``
(``pybnf.petab.bngl_model.register_bngl``), so the standard petab validator can
load and check a ``language: bngl`` problem. This clinic proves that with that
loader in place, petab's own lint tasks correctly catch the mistakes a
BNGL-native problem can make -- exactly the confidence we want before proposing
the loader upstream to libpetab-python (issue #420).

The *expected outcome* of each fixture (which lint task flags it, or that it
raises at load) is recorded test-side in ``examples/tutorial/_manifest.py``
(``LINT_CASES``); this script owns only the recipe that PRODUCES each defect, so
the two never silently drift -- every non-clean case here must have a matching
manifest entry, and vice versa (asserted below).

Usage (no simulation backend needed -- lint is static):

    python examples/tutorial/13_petab_lint_clinic/regenerate_fixtures.py
"""
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))          # so `_manifest` (one dir up) imports

from _manifest import LINT_CASES                # noqa: E402


# --------------------------------------------------------------------------- #
# The clean baseline problem: a one-species exponential decay A --k--> 0.
# Every broken variant is this problem with a single targeted mutation.
# --------------------------------------------------------------------------- #
_MODEL = """\
# One-species exponential decay, A --k--> 0.  The whole lint clinic is built on
# this three-line model; each broken fixture perturbs the PEtab *tables* around
# it (or, for `malformed_bngl`, the model itself).
begin model
  begin parameters
    k   0.5
    A0  10
  end parameters
  begin molecule types
    A()
  end molecule types
  begin seed species
    A()  A0
  end seed species
  begin observables
    Molecules  Obs_A  A()
  end observables
  begin reaction rules
    decay: A() -> 0  k
  end reaction rules
end model
"""

_YAML = """\
format_version: 2.0.0
parameter_files:
  - parameters.tsv
observable_files:
  - observables.tsv
measurement_files:
  - measurements.tsv
model_files:
  decay:
    location: decay.bngl
    language: bngl
"""

_PARAMETERS = (
    "parameterId\testimate\tlowerBound\tupperBound\n"
    "k\ttrue\t0.01\t5\n"
)

_OBSERVABLES = (
    "observableId\tobservableFormula\tnoiseFormula\tnoiseDistribution\n"
    "obs_A\tObs_A\t1\tnormal\n"
)

# A(t) = A0 * exp(-k t) with A0 = 10, k = 0.5 (values are illustrative -- lint
# never simulates, so exactness does not matter here).
_MEASUREMENTS = (
    "observableId\texperimentId\ttime\tmeasurement\n"
    "obs_A\t\t0\t10\n"
    "obs_A\t\t1\t6.0653\n"
    "obs_A\t\t2\t3.6788\n"
    "obs_A\t\t3\t2.2313\n"
    "obs_A\t\t4\t1.3534\n"
)


def _base():
    """A fresh dict of {relative filename: text} for the clean problem."""
    return {
        'problem.yaml': _YAML,
        'decay.bngl': _MODEL,
        'parameters.tsv': _PARAMETERS,
        'observables.tsv': _OBSERVABLES,
        'measurements.tsv': _MEASUREMENTS,
    }


# --------------------------------------------------------------------------- #
# The mutations. Each takes the base file dict and edits it IN PLACE to inject
# exactly one defect. Keyed by the subdirectory name (== LINT_CASES[i].folder).
# --------------------------------------------------------------------------- #
def _clean(files):
    pass


def _undefined_observable(files):
    # A measurement points at an observable id that the observable table never
    # defines -> CheckMeasuredObservablesDefined.
    files['measurements.tsv'] = files['measurements.tsv'].replace(
        'obs_A\t\t0\t10', 'obs_TYPO\t\t0\t10', 1)


def _observable_shadows_entity(files):
    # The observable id collides with the model's own observable `Obs_A`
    # -> CheckObservablesDoNotShadowModelEntities.
    files['observables.tsv'] = files['observables.tsv'].replace('obs_A', 'Obs_A')
    files['measurements.tsv'] = files['measurements.tsv'].replace('obs_A', 'Obs_A')


def _missing_parameter(files):
    # The observable formula multiplies by a symbol that is declared nowhere --
    # not in the model, not in parameters.tsv
    # -> CheckAllParametersPresentInParameterTable.
    files['observables.tsv'] = (
        "observableId\tobservableFormula\tnoiseFormula\tnoiseDistribution\n"
        "obs_A\tscale_undeclared * Obs_A\t1\tnormal\n")


def _override_placeholder_mismatch(files):
    # The formula declares an observableParameter placeholder, but no measurement
    # supplies the override to substitute for it
    # -> CheckOverridesMatchPlaceholders.
    files['observables.tsv'] = (
        "observableId\tobservableFormula\tnoiseFormula\tobservablePlaceholders\tnoiseDistribution\n"
        "obs_A\tobservableParameter1_obs_A * Obs_A\t1\tobservableParameter1_obs_A\tnormal\n")


def _bad_condition_target(files):
    # A condition perturbs a symbol that is not a settable model entity
    # -> CheckValidConditionTargets.
    files['conditions.tsv'] = (
        "conditionId\ttargetId\ttargetValue\n"
        "c_bad\tNOT_A_MODEL_SYMBOL\t1.0\n")
    files['problem.yaml'] = files['problem.yaml'].replace(
        'parameter_files:',
        'condition_files:\n  - conditions.tsv\nparameter_files:')


def _bad_prior(files):
    # `normal` is a valid prior distribution, but it needs TWO parameters
    # (mean, sd); here only one is given -> CheckPriorDistribution.
    files['parameters.tsv'] = (
        "parameterId\testimate\tlowerBound\tupperBound\tpriorDistribution\tpriorParameters\n"
        "k\ttrue\t0.01\t5\tnormal\t0\n")


def _unknown_prior_distribution(files):
    # `bogus_dist` is not a recognized distribution name. petab's parameter table
    # is a typed (pydantic) model, so this is rejected the moment the problem is
    # LOADED (Problem.from_yaml raises) -- before lint even runs. That early,
    # structural rejection is itself the validator doing its job.
    files['parameters.tsv'] = (
        "parameterId\testimate\tlowerBound\tupperBound\tpriorDistribution\tpriorParameters\n"
        "k\ttrue\t0.01\t5\tbogus_dist\t0;1\n")


def _malformed_bngl(files):
    # A BNGL syntax error (a mistyped block terminator). CheckModel shells out to
    # `BNG2.pl --check`, so this fixture is only flagged where a BioNetGen is
    # available (needs_bng2 in the manifest); without one the loader degrades to
    # "valid" and the defect is invisible.
    files['decay.bngl'] = files['decay.bngl'].replace(
        'end parameters', 'end paramters')


# --------------------------------------------------------------------------- #
# Wave 2 (backlog #1). First the checks that need no new tables ...
# --------------------------------------------------------------------------- #
def _pos_log_measurement(files):
    # The observable is scored on a log scale (`log-normal` noise), which is only
    # defined for positive data -- but a measurement value is <= 0
    # -> CheckPosLogMeasurements.
    files['observables.tsv'] = files['observables.tsv'].replace(
        'normal', 'log-normal')
    files['measurements.tsv'] = files['measurements.tsv'].replace(
        'obs_A\t\t4\t1.3534', 'obs_A\t\t4\t-1', 1)


def _duplicate_observable_id(files):
    # Two observable-table rows share the primary key `obs_A`
    # -> CheckUniquePrimaryKeys (petab loads both rows, then flags the dupe).
    files['observables.tsv'] = files['observables.tsv'] + 'obs_A\tObs_A\t1\tnormal\n'


def _model_entity_as_parameter(files):
    # The model's observable `Obs_A` is listed in the parameter table, where only
    # free model *parameters* may go -> CheckValidParameterInConditionOrParameter-
    # Table. (A model entity is never a valid parameter row, so this necessarily
    # also trips CheckAllParametersPresentInParameterTable; the test asserts the
    # target task is AMONG those that flagged.)
    files['parameters.tsv'] = files['parameters.tsv'] + 'Obs_A\ttrue\t0.01\t5\n'


def _measurement_bad_model_id(files):
    # Every measurement names modelId `ghost`, but the only model_files entry is
    # `decay` -> CheckMeasurementModelId. (Each row must carry a modelId; an empty
    # cell would be read as NaN and rejected at load, not by lint.)
    files['measurements.tsv'] = (
        'observableId\texperimentId\ttime\tmeasurement\tmodelId\n'
        'obs_A\t\t0\t10\tghost\n'
        'obs_A\t\t1\t6.0653\tghost\n'
        'obs_A\t\t2\t3.6788\tghost\n')


def _missing_config_file(files):
    # The problem omits the required `parameter_files` section. Like an unknown
    # prior NAME, this is a structural defect the problem-config schema rejects
    # the instant the YAML loads (Problem.from_yaml raises) -- one layer before
    # the CheckProblemConfig lint task, which guards the very same requirement,
    # would run. The early rejection is itself the validator doing its job.
    files['problem.yaml'] = files['problem.yaml'].replace(
        'parameter_files:\n  - parameters.tsv\n', '')
    del files['parameters.tsv']


# --------------------------------------------------------------------------- #
# ... then the experiment/condition-table batch. These introduce two tables the
# base problem never needed: experiments.tsv (experimentId, time, conditionId --
# at `time`, apply `conditionId`) and conditions.tsv (conditionId, targetId,
# targetValue). Both are wired in through new problem.yaml keys (experiment_files
# / condition_files). Condition targets use `A0` -- a FIXED model parameter, a
# valid perturbation target that is NOT the estimated parameter `k`, so no
# spurious parameter-table errors mask the defect under test.
# --------------------------------------------------------------------------- #
def _add_experiment_tables(files, *, conditions=None, experiments):
    """Wire an experiments.tsv (and optional conditions.tsv) into the problem and
    point every measurement at experiment `e1`."""
    keys = ''
    if conditions is not None:
        files['conditions.tsv'] = conditions
        keys += 'condition_files:\n  - conditions.tsv\n'
    files['experiments.tsv'] = experiments
    keys += 'experiment_files:\n  - experiments.tsv\n'
    files['problem.yaml'] = files['problem.yaml'].replace(
        'parameter_files:', keys + 'parameter_files:')
    files['measurements.tsv'] = files['measurements.tsv'].replace(
        'obs_A\t\t', 'obs_A\te1\t')


def _missing_experiment_condition(files):
    # Experiment e1's second period (t=1) applies condition `c_missing`, which the
    # (absent) condition table never defines -> CheckExperimentConditionsExist.
    # The first period uses the model as-is (empty conditionId), so
    # CheckInitialChangeSymbols has nothing to complain about.
    _add_experiment_tables(
        files,
        experiments=('experimentId\ttime\tconditionId\n'
                     'e1\t0\t\n'
                     'e1\t1\tc_missing\n'))


def _undefined_experiment(files):
    # A measurement points at experiment `e_ghost`, but there is no experiment
    # table at all -> CheckUndefinedExperiments (a WARNING, not an error).
    files['measurements.tsv'] = files['measurements.tsv'].replace(
        'obs_A\t\t', 'obs_A\te_ghost\t')


def _unused_experiment(files):
    # The experiment table defines `e_unused`, which no measurement references
    # -> CheckUnusedExperiments (a WARNING). e1 is the used experiment; both
    # conditions are applied, so no condition goes unused.
    _add_experiment_tables(
        files,
        conditions=('conditionId\ttargetId\ttargetValue\n'
                    'c_a0\tA0\t5\n'
                    'c_alt\tA0\t7\n'),
        experiments=('experimentId\ttime\tconditionId\n'
                     'e1\t0\tc_a0\n'
                     'e_unused\t0\tc_alt\n'))


def _unused_condition(files):
    # The condition table defines `c_orphan`, which no experiment period applies
    # -> CheckUnusedConditions (a WARNING). c_a0 is the applied condition.
    _add_experiment_tables(
        files,
        conditions=('conditionId\ttargetId\ttargetValue\n'
                    'c_a0\tA0\t5\n'
                    'c_orphan\tA0\t7\n'),
        experiments='experimentId\ttime\tconditionId\ne1\t0\tc_a0\n')


def _initial_change_symbol(files):
    # Condition c_a0 is applied at the START of experiment e1 (t=0) and sets A0
    # from `undefined_symbol * 2`. An initial change may only reference parameter-
    # table symbols (or `time`) -> CheckInitialChangeSymbols. (`undefined_symbol`
    # is declared nowhere, so this also trips CheckAllParametersPresentInParameter-
    # Table; the test asserts the target task is AMONG those that flagged.)
    _add_experiment_tables(
        files,
        conditions=('conditionId\ttargetId\ttargetValue\n'
                    'c_a0\tA0\tundefined_symbol * 2\n'),
        experiments='experimentId\ttime\tconditionId\ne1\t0\tc_a0\n')


_MUTATIONS = {
    'clean': _clean,
    'undefined_observable': _undefined_observable,
    'observable_shadows_entity': _observable_shadows_entity,
    'missing_parameter': _missing_parameter,
    'override_placeholder_mismatch': _override_placeholder_mismatch,
    'bad_condition_target': _bad_condition_target,
    'bad_prior': _bad_prior,
    'unknown_prior_distribution': _unknown_prior_distribution,
    'malformed_bngl': _malformed_bngl,
    # wave 2 -- no new tables
    'pos_log_measurement': _pos_log_measurement,
    'duplicate_observable_id': _duplicate_observable_id,
    'model_entity_as_parameter': _model_entity_as_parameter,
    'measurement_bad_model_id': _measurement_bad_model_id,
    'missing_config_file': _missing_config_file,
    # wave 2 -- experiment/condition tables
    'missing_experiment_condition': _missing_experiment_condition,
    'undefined_experiment': _undefined_experiment,
    'unused_experiment': _unused_experiment,
    'unused_condition': _unused_condition,
    'initial_change_symbol': _initial_change_symbol,
}


def main():
    manifest_folders = {c.folder for c in LINT_CASES}
    if manifest_folders != set(_MUTATIONS):
        missing = manifest_folders - set(_MUTATIONS)
        extra = set(_MUTATIONS) - manifest_folders
        raise SystemExit(
            f'LINT_CASES / _MUTATIONS out of sync: '
            f'no recipe for {sorted(missing)}; no manifest entry for {sorted(extra)}')

    for folder, mutate in _MUTATIONS.items():
        files = _base()
        mutate(files)
        dest = _HERE / folder
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for name, text in files.items():
            (dest / name).write_text(text)
        print(f'wrote {folder}/ ({len(files)} files)')


if __name__ == '__main__':
    main()
