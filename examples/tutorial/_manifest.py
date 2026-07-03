"""Test-side manifest for the edition-2 tutorial examples.

This module is the single source of truth the *tests* read (never the tutorial
confs themselves -- those stay clean and realistic). It records, per example:

* how to (re)generate each committed ``.exp`` -- the model, observable columns,
  time grid, and any synthetic noise (used by ``regenerate_data.py``); and
* what each committed ``.conf`` should recover -- the true parameter values, the
  recovery tolerance, and the pytest marker tier (default / slow / jax-gated),
  used by ``tests/test_tutorial_examples.py``.

Keeping the truth + tolerances here (not in the confs) is what lets one artifact
set serve both the tutorial (clean, copy-pasteable) and the test suite
(asserting, gated) without the two colliding.
"""
import math
from dataclasses import dataclass, field
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parent

_INV_LN10 = 1.0 / math.log(10.0)   # natural-log location/scale -> log10 (PyBNF's log parameterization)


@dataclass(frozen=True)
class Dataset:
    """One committed ``.exp`` to generate from a model at its true parameters."""
    exp: str                 # filename, relative to the example folder
    obs: tuple               # observable columns to write (independent var is 'time')
    t_end: float
    n_points: int
    noise_sd: float = 0.0     # gaussian noise (absolute sd) added + written as _SD; 0 = clean
    noise_seed: int = 0
    sd: float = None          # emit a constant _SD column at this value (independent of the
                              # gaussian noise above -- for a clean curve scored under chi_sq/laplace)
    outliers: tuple = ()      # ((row_index, obs_name, replacement_value), ...): gross outliers
                              # spliced into the named observable column (deterministic contamination)
    scan: str = None          # if set, an independent-variable column name (dose-response)
    doses: tuple = ()         # explicit scan values for `scan` -- a steady-state parameter_scan
                              # dataset (the .exp rows ARE the doses; no time grid, ADR-0046)
    condition: tuple = None       # (name, "var op val"): a measurement condition (perturbation)
    preequilibrate: tuple = None  # (name, "var op val"): equilibrate here to steady state first
                                  # (two-phase protocol; carry-over into the measurement, ADR-0052)
    measurements: tuple = ()  # derived measurement-model columns (ADR-0036); when set, the
                              # .exp holds these columns, not the raw `obs` (which are only
                              # simulated as inputs to the measurement formulas)
    model: str = None         # generate this dataset from a DIFFERENT model than the example's
                              # (a multi-model joint fit, ADR-0041: each experiment its own model
                              # file, all sharing the example's free parameters by bare id)


@dataclass(frozen=True)
class Measurement:
    """A derived measurement-model column (ADR-0036 ``observableFormula``) written
    into a ``.exp`` instead of a raw model observable.

    The data generator simulates the model's *raw* observables (the dataset's
    ``obs``), then materializes ``formula`` over them -- with any observation-layer
    ``nuisance`` at its true value -- to produce the column the fit actually scores.
    Because the same measurement-layer code generates the data and evaluates the
    fit, a correct fit recovers both the model rate and the nuisance exactly.
    """
    obs_id: str                  # derived column name (written to the .exp, scored)
    formula: str                 # observableFormula over raw obs + nuisance symbols
    nuisance: dict = field(default_factory=dict)  # {name: true_value}, absent from the model


@dataclass(frozen=True)
class ConfCheck:
    """One committed ``.conf`` and how the test checks it.

    Exactly one *mode* applies:
      * ``refused=True``  -- a gradient fit that must be refused, not run;
      * ``profile={...}`` -- a profile-likelihood run; the dict maps each
        parameter to its expected identifiability class (and ``recover`` holds
        the true values that an *identifiable* CI must bracket);
      * otherwise         -- a plain fit that must recover ``recover`` within ``tol``.
    """
    conf: str                 # filename, relative to the example folder
    recover: dict             # {param: true_value} (recovery target / CI-bracket target)
    tol: float = 0.03          # fractional recovery tolerance
    marker: str = 'default'    # 'default' (bngsim+newera) | 'slow' | 'jax'
    refused: bool = False      # True => a gradient fit that must be REFUSED, not run
    profile: dict = None       # {param: expected identifiability class} for profile_likelihood
    max_obj: float = None      # constraint fit: assert best objective <= this (0 => all satisfied)
    dragged_min_err: float = None  # cautionary conf: at least one `recover` param must be off by
                                   # >= this fraction (a non-robust objective broken by outliers)
    note: str = ''


@dataclass(frozen=True)
class LintCase:
    """One PEtab v2 fixture in the ``13_petab_lint_clinic`` gallery and how the
    linter should react to it.

    ``outcome`` selects the expectation:
      * ``'clean'``  -- ``lint_problem`` reports no errors;
      * ``'error'``  -- ``lint_problem`` reports errors, and ``task`` (a
        ``petab.v2.lint`` Check class name) is among the flagging tasks;
      * ``'raises'`` -- the defect is structural, so ``Problem.from_yaml`` itself
        rejects the problem before lint runs.
    """
    folder: str                  # subdir under 13_petab_lint_clinic/
    outcome: str                 # 'clean' | 'error' | 'raises'
    task: str = ''               # expected Check class in the report (outcome='error')
    needs_bng2: bool = False     # CheckModel shells out to `BNG2.pl --check`
    blurb: str = ''              # one-line description (README + test id)


# The negative-lint gallery: each fixture carries exactly one defect (or none),
# and the test asserts the petab.v2 validator -- running through PyBNF's BNGL
# loader -- reacts as recorded here. See the folder's regenerate_fixtures.py for
# the recipe that produces each defect.
LINT_CASES = (
    LintCase('clean', 'clean',
             blurb='a valid BNGL-native PEtab v2 problem (baseline)'),
    LintCase('undefined_observable', 'error', task='CheckMeasuredObservablesDefined',
             blurb='a measurement references an observable the table never defines'),
    LintCase('observable_shadows_entity', 'error',
             task='CheckObservablesDoNotShadowModelEntities',
             blurb='an observable id collides with a model species/observable name'),
    LintCase('missing_parameter', 'error', task='CheckAllParametersPresentInParameterTable',
             blurb='an observable formula uses a symbol declared nowhere'),
    LintCase('override_placeholder_mismatch', 'error', task='CheckOverridesMatchPlaceholders',
             blurb='a formula placeholder has no matching measurement override'),
    LintCase('bad_condition_target', 'error', task='CheckValidConditionTargets',
             blurb='a condition perturbs a symbol that is not a model entity'),
    LintCase('bad_prior', 'error', task='CheckPriorDistribution',
             blurb='a normal prior is given the wrong number of parameters'),
    LintCase('unknown_prior_distribution', 'raises',
             blurb='an unrecognized prior distribution name (rejected at load)'),
    LintCase('malformed_bngl', 'error', task='CheckModel', needs_bng2=True,
             blurb='a BNGL syntax error (caught by BNG2.pl --check)'),
)


@dataclass(frozen=True)
class PriorCase:
    """One estimated parameter in the ``15_petab_priors`` fixture: the PEtab v2
    ``priorDistribution`` / ``priorParameters`` a modeler wrote, and the PyBNF
    :class:`~pybnf.pset.FreeParameter` it must import to.

    ``regenerate_fixtures.py`` writes the left half (the PEtab ``parameters.tsv``
    row) from these fields; ``tests/test_tutorial_priors.py`` reads that committed
    table back and asserts ``free_parameter_from_row`` produces the right half
    (``exp_type`` / ``exp_p1`` / ``exp_p2`` / ``exp_bounded``). So the one manifest
    row pins both the fixture *and* the expected import -- they can never drift.
    """
    param: str                    # parameterId (a real model parameter, bound by bare id)
    distribution: str             # PEtab priorDistribution ('' => none: PEtab defaults to uniform/bounds)
    prior_params: str             # PEtab priorParameters cell ('' => none)
    lower: float                  # lowerBound
    upper: float                  # upperBound
    exp_type: str                 # expected FreeParameter.type keyword after import
    exp_p1: float                 # expected FreeParameter.p1
    exp_p2: float = None          # expected FreeParameter.p2 (None for a one-parameter family)
    exp_bounded: bool = True      # expected FreeParameter.bounded
    blurb: str = ''               # the prior's modelling story (README + test id)


# A receptor--ligand binding model (L + R <-> C), four estimated parameters, each
# with the prior family a modeler would naturally reach for -- a positive gallery
# that complements lesson 13's `bad_prior`. A rate spanning orders of magnitude
# takes a LOG-normal; a positive rate a half-bounded GAMMA; a measured amount a
# NORMAL; a dose you set yourself only a UNIFORM range. Each imports (through the
# real PEtab loader) to the matching *_var FreeParameter, priorParameters and all.
PRIOR_CASES = (
    PriorCase('kon', 'log-normal', '0;1', 0.001, 10,
              exp_type='lognormal_var', exp_p1=0.0, exp_p2=_INV_LN10,
              blurb='an association rate spanning orders of magnitude -> log-normal belief'),
    PriorCase('koff', 'gamma', '2;0.5', 0.001, 10,
              exp_type='gamma_var', exp_p1=2.0, exp_p2=0.5,
              blurb='a positive dissociation rate -> gamma (shape, scale) on (0, inf)'),
    PriorCase('R0', 'normal', '30;5', 1, 100,
              exp_type='normal_var', exp_p1=30.0, exp_p2=5.0,
              blurb='a receptor amount measured with error -> normal (mean, sd)'),
    PriorCase('L0', '', '', 1, 100,
              exp_type='uniform_var', exp_p1=1.0, exp_p2=100.0,
              blurb='a dose you set yourself, known only to a range -> PEtab default uniform'),
)


@dataclass(frozen=True)
class Example:
    folder: str
    model: str
    truth: dict                          # ground-truth parameter values
    build_free: dict                     # {param: (var_type, lo, hi)} to construct data-gen fit
    datasets: tuple = ()
    confs: tuple = ()

    @property
    def path(self):
        return TUTORIAL_DIR / self.folder


EXAMPLES = (
    Example(
        folder='01_logistic_growth',
        model='logistic_growth.bngl',
        truth={'r': 1.2, 'K': 100.0},
        build_free={'r': ('uniform_var', 0.1, 5.0), 'K': ('uniform_var', 20.0, 300.0)},
        datasets=(
            Dataset('logistic_growth.exp', obs=('Obs_N',), t_end=8, n_points=17),
        ),
        confs=(
            ConfCheck('logistic_growth_trf.conf', recover={'r': 1.2, 'K': 100.0}, tol=0.02),
            # Fit to qualitative (BPSL .prop) data only: a satisfying parameter set
            # drives the constraint-penalty objective to ~0.
            ConfCheck('logistic_growth_constraints.conf', recover={}, max_obj=1e-6),
        ),
    ),
    Example(
        folder='02_bateman_chain',
        model='bateman_chain.bngl',
        truth={'k1': 0.8, 'k2': 0.25},
        build_free={'k1': ('uniform_var', 0.05, 3.0), 'k2': ('uniform_var', 0.02, 2.0)},
        datasets=(
            Dataset('bateman_chain.exp', obs=('Obs_A', 'Obs_B', 'Obs_C'), t_end=20, n_points=21),
            # Lesson 4: observe ONLY A -> k2 is structurally non-identifiable.
            Dataset('bateman_A_only.exp', obs=('Obs_A',), t_end=20, n_points=21),
        ),
        confs=(
            ConfCheck('bateman_chain_de.conf', recover={'k1': 0.8, 'k2': 0.25}, tol=0.03),
            # Profile likelihood: all three species observed -> both rates identifiable.
            ConfCheck('bateman_chain_profile_likelihood.conf',
                      recover={'k1': 0.8, 'k2': 0.25},
                      profile={'k1': 'identifiable', 'k2': 'identifiable'}),
            # Observe only A: A(t) doesn't depend on k2 -> k2 structurally non-identifiable.
            ConfCheck('bateman_A_only_profile_likelihood.conf',
                      recover={'k1': 0.8},
                      profile={'k1': 'identifiable',
                               'k2': 'structurally non-identifiable'}),
        ),
    ),
    Example(
        folder='05_noisy_decay',
        model='noisy_decay.bngl',
        truth={'k': 0.5, 'A0': 100.0},
        build_free={'k': ('uniform_var', 0.05, 3.0), 'A0': ('uniform_var', 20.0, 400.0)},
        datasets=(
            # A noisy dataset (gaussian sd=3, written as _SD columns) for the
            # uncertainty lessons -- bootstrapping resamples these residuals.
            Dataset('noisy_decay.exp', obs=('Obs_A',), t_end=10, n_points=21,
                    noise_sd=3.0, noise_seed=7),
        ),
        # The bootstrap conf is CLI-orchestrated, so it is verified by its own
        # subprocess test (tests/test_tutorial_bootstrap.py), not the inline
        # harness verifier -- no ConfCheck here.
        confs=(),
    ),
    Example(
        folder='03_gompertz_growth',
        model='gompertz_growth.bngl',
        truth={'r': 0.4, 'K': 100.0},
        build_free={'r': ('uniform_var', 0.05, 3.0), 'K': ('uniform_var', 20.0, 400.0)},
        datasets=(
            Dataset('gompertz_growth.exp', obs=('Obs_X',), t_end=20, n_points=21),
        ),
        confs=(
            ConfCheck('gompertz_growth_pso.conf', recover={'r': 0.4, 'K': 100.0}, tol=0.05),
        ),
    ),
    Example(
        folder='11_interop',
        model='decay.bngl',
        truth={'k': 0.5},
        build_free={'k': ('uniform_var', 0.05, 3.0)},
        datasets=(
            # Integer time grid (0..8): an SBML/Antimony model simulates on bngsim's
            # default output grid, which lands on integers, so the data times match.
            Dataset('decay.exp', obs=('Obs_A',), t_end=8, n_points=9),
        ),
        # The SAME A->B dynamics as BNGL / Antimony / SBML, all fit through bngsim,
        # all recovering the same k -- a backend/format interop regression.
        confs=(
            ConfCheck('fit_bngl.conf', recover={'k': 0.5}, tol=0.03),
            ConfCheck('fit_antimony.conf', recover={'k': 0.5}, tol=0.03),
            ConfCheck('fit_sbml.conf', recover={'k': 0.5}, tol=0.03),
        ),
    ),
    Example(
        folder='10_per_observable_noise',
        model='two_reporter.bngl',
        truth={'k1': 0.8, 'k2': 0.25},
        build_free={'k1': ('uniform_var', 0.05, 3.0), 'k2': ('uniform_var', 0.02, 2.0)},
        datasets=(
            # Two readouts: Obs_A (clean, pins k1) and Obs_C (three gross outliers,
            # the only handle on k2). Both carry a constant _SD column.
            Dataset('reporters.exp', obs=('Obs_A', 'Obs_C'), t_end=20, n_points=21,
                    sd=3.0, outliers=((3, 'Obs_C', 5.0), (7, 'Obs_C', 75.0),
                                      (11, 'Obs_C', 15.0))),
        ),
        confs=(
            # Per-observable noise: Obs_A Gaussian, Obs_C Laplace -> robust exactly
            # where the outliers are; both rates recovered.
            ConfCheck('per_observable.conf', recover={'k1': 0.8, 'k2': 0.25}, tol=0.03),
            # One Gaussian model for both observables: Obs_C's outliers drag k2
            # (clean Obs_A still pins k1) -- at least 10% off.
            ConfCheck('both_gaussian.conf', recover={'k1': 0.8, 'k2': 0.25},
                      dragged_min_err=0.10),
        ),
    ),
    Example(
        folder='09_experiment_design',
        model='inducible_gene.bngl',
        truth={'k_deg': 2.0},
        build_free={'k_deg': ('uniform_var', 0.1, 10.0)},
        datasets=(
            # Dose-response: sweep the stimulus strength k_prod and read the STEADY-STATE
            # level A_ss = k_prod/k_deg at each dose (a parameter_scan, no time grid).
            Dataset('dose_response.exp', obs=('A_tot',), t_end=0, n_points=0,
                    scan='k_prod', doses=(1.0, 2.0, 4.0, 8.0, 16.0)),
            # Washout: equilibrate with the stimulus ON (A -> k_prod/k_deg), then switch it
            # OFF and watch A relax -- a two-phase pre-equilibration protocol.
            Dataset('washout.exp', obs=('A_tot',), t_end=3, n_points=7,
                    condition=('stim_off', 'Stimulus_isOn = 0'),
                    preequilibrate=('stim_on', 'Stimulus_isOn = 1')),
        ),
        # Two experiment designs, one rate constant: both recover k_deg.
        confs=(
            ConfCheck('dose_response.conf', recover={'k_deg': 2.0}, tol=0.03),
            ConfCheck('washout.conf', recover={'k_deg': 2.0}, tol=0.03),
        ),
    ),
    Example(
        folder='08_robust_objectives',
        model='contaminated_decay.bngl',
        truth={'k': 0.5, 'A0': 100.0},
        build_free={'k': ('uniform_var', 0.05, 3.0), 'A0': ('uniform_var', 20.0, 400.0)},
        datasets=(
            # A clean decay curve with three gross outliers spliced in, plus a
            # constant _SD column (assumed measurement error) for chi_sq/laplace.
            Dataset('contaminated_decay.exp', obs=('Obs_A',), t_end=10, n_points=21,
                    sd=3.0, outliers=((4, 'Obs_A', 90.0), (12, 'Obs_A', 55.0),
                                      (16, 'Obs_A', 40.0))),
        ),
        confs=(
            # A heavy-tailed Laplace noise model shrugs the outliers off and
            # recovers the truth (while estimating the noise scale `noise_scale`).
            ConfCheck('decay_laplace.conf', recover={'k': 0.5, 'A0': 100.0}, tol=0.03),
            # A Gaussian noise model is pulled off the truth by the same outliers:
            # its k must be wrong by at least 10% -- the whole point of the lesson.
            ConfCheck('decay_gaussian.conf', recover={'k': 0.5, 'A0': 100.0},
                      dragged_min_err=0.10),
        ),
    ),
    Example(
        folder='07_algorithm_bakeoff',
        model='oscillator.bngl',
        truth={'alpha': 1.2, 'gamma': 0.8},
        build_free={'alpha': ('uniform_var', 0.1, 5.0),
                    'gamma': ('uniform_var', 0.1, 5.0)},
        datasets=(
            Dataset('oscillator.exp', obs=('Obs_P', 'Obs_Q'), t_end=12, n_points=25),
        ),
        # Six metaheuristics, one oscillatory landscape, same (alpha, gamma) truth.
        # Every optimizer in the population/metaheuristic family must recover it --
        # a broad regression gate across the whole non-gradient algorithm surface.
        confs=(
            ConfCheck('oscillator_de.conf',    recover={'alpha': 1.2, 'gamma': 0.8}, tol=0.03),
            ConfCheck('oscillator_ade.conf',   recover={'alpha': 1.2, 'gamma': 0.8}, tol=0.03),
            ConfCheck('oscillator_pso.conf',   recover={'alpha': 1.2, 'gamma': 0.8}, tol=0.03),
            ConfCheck('oscillator_cmaes.conf', recover={'alpha': 1.2, 'gamma': 0.8}, tol=0.03),
            ConfCheck('oscillator_sa.conf',    recover={'alpha': 1.2, 'gamma': 0.8}, tol=0.03),
            ConfCheck('oscillator_ss.conf',    recover={'alpha': 1.2, 'gamma': 0.8}, tol=0.03),
        ),
    ),
    Example(
        folder='14_observable_layer',
        model='conversion.bngl',
        truth={'k': 0.7},
        build_free={'k': ('uniform_var', 0.05, 3.0)},
        datasets=(
            # A scaled readout: the data is scale*Obs_B with a true scale of 2.0.
            Dataset('conversion_scaling.exp', obs=('Obs_B',), t_end=8, n_points=17,
                    measurements=(Measurement('obs_scaled', 'scale * Obs_B',
                                              nuisance={'scale': 2.0}),)),
            # A fraction/ratio readout (dimensionless, self-normalizing).
            Dataset('conversion_ratio.exp', obs=('Obs_A', 'Obs_B'), t_end=8, n_points=17,
                    measurements=(Measurement('obs_fracB', 'Obs_B / (Obs_A + Obs_B)'),)),
            # A log-transformed readout (natural log; emphasizes the small-value tail).
            Dataset('conversion_log.exp', obs=('Obs_A',), t_end=8, n_points=17,
                    measurements=(Measurement('obs_logA', 'log(Obs_A)'),)),
        ),
        confs=(
            # Scaling factor as an observation-layer nuisance (PEtab observableParameters):
            # `scale` is declared only in the fit, never in the model, yet is recovered.
            ConfCheck('conversion_scaling.conf', recover={'k': 0.7, 'scale': 2.0}, tol=0.03),
            # A derived fraction the model never emits as a column -- built post-simulation.
            ConfCheck('conversion_ratio.conf', recover={'k': 0.7}, tol=0.03),
            # Fitting a transformed readout (log space).
            ConfCheck('conversion_log.conf', recover={'k': 0.7}, tol=0.03),
        ),
    ),
    Example(
        folder='16_joint_fit',
        model='central_bolus.bngl',
        truth={'k12': 0.8, 'k21': 0.4, 'ke': 0.3},
        build_free={'k12': ('uniform_var', 0.05, 3.0),
                    'k21': ('uniform_var', 0.05, 3.0),
                    'ke':  ('uniform_var', 0.05, 3.0)},
        datasets=(
            # Two dosing routes of the SAME drug, each its own model file but sharing
            # {k12, k21, ke} (a multi-model joint fit, ADR-0041). Central bolus: the
            # plasma curve is the classic biexponential decay.
            Dataset('central_dose.exp', obs=('Obs_Central',), t_end=20, n_points=21,
                    model='central_bolus.bngl'),
            # Peripheral bolus: the plasma starts at ZERO and rises (fed by k21) before
            # it falls -- a shape whose rise pins k21, complementing the central bolus.
            Dataset('peripheral_dose.exp', obs=('Obs_Central',), t_end=20, n_points=21,
                    model='peripheral_bolus.bngl'),
        ),
        confs=(
            # One shared-parameter fit to BOTH experiments (two models): recover all
            # three rates that no single route pins as tightly on its own.
            ConfCheck('joint_fit.conf', recover={'k12': 0.8, 'k21': 0.4, 'ke': 0.3},
                      tol=0.05),
        ),
    ),
    Example(
        folder='06_step_input',
        model='step_input.bngl',
        truth={'k': 0.6, 'J_base': 1.2, 'J_step': 2.4},
        build_free={'k': ('uniform_var', 0.1, 3.0),
                    'J_base': ('uniform_var', 0.1, 5.0),
                    'J_step': ('uniform_var', 0.1, 6.0)},
        datasets=(
            Dataset('step_input.exp', obs=('Obs_X',), t_end=12, n_points=25),
        ),
        confs=(
            ConfCheck('step_input_de.conf',
                      recover={'k': 0.6, 'J_base': 1.2, 'J_step': 2.4}, tol=0.05),
            ConfCheck('step_input_trf_refused.conf', recover={}, refused=True,
                      note='piecewise input -> gradient optimizer must refuse'),
        ),
    ),
    Example(
        folder='06_step_input',
        model='step_input_smooth.bngl',
        truth={'k': 0.6, 'J_base': 1.2, 'J_step': 2.4, 'tau': 4.0},
        build_free={'k': ('uniform_var', 0.1, 3.0),
                    'J_base': ('uniform_var', 0.1, 5.0),
                    'J_step': ('uniform_var', 0.1, 6.0),
                    'tau': ('uniform_var', 1.0, 8.0)},
        datasets=(
            Dataset('step_input_smooth.exp', obs=('Obs_X',), t_end=12, n_points=25),
        ),
        confs=(
            # The smooth sigmoid step IS differentiable -> trf fits it, and even
            # recovers the transition time tau (impossible for the hard if()).
            ConfCheck('step_input_smooth_trf.conf',
                      recover={'k': 0.6, 'J_base': 1.2, 'J_step': 2.4, 'tau': 4.0},
                      tol=0.03),
        ),
    ),
)


def example_by_folder(folder):
    for ex in EXAMPLES:
        if ex.folder == folder:
            return ex
    raise KeyError(folder)
