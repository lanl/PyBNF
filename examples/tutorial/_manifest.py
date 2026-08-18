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
    noise_cv: float = None    # MULTIPLICATIVE gaussian noise: each point *= (1 + cv*N(0,1)),
                              # seeded by `noise_cv_seed`. No _SD column -- the scale-free objectives
                              # derive their own weighting (sos: sigma=1; norm_sos: sigma proportional
                              # to the value). Data spanning orders of magnitude with constant-relative
                              # noise -- the scale-free objective lesson (35): sos over-weights the
                              # large points, norm_sos weights every point equally.
    noise_cv_seed: int = 0
    noise_lognormal_sigma: float = None  # MULTIPLICATIVE lognormal noise on a log10 scale:
                              # each point *= 10**(sigma*N(0,1) - sigma**2*ln10/2), seeded by
                              # `noise_lognormal_seed`. The -sigma^2*ln10/2 term is the mean
                              # correction, so E[obs] = the model value (mean-aligned) -- matching
                              # a `noise_model = lognormal, ..., location = mean` fit. Always
                              # positive (the lognormal support). No _SD column (the lognormal
                              # confs set sigma with `fix_at`): data spanning orders of magnitude
                              # with constant RELATIVE scatter -- the lognormal lesson (42).
    noise_lognormal_seed: int = 0
    noise_combined_abs: float = None  # COMBINED additive+proportional gaussian noise: each point
                              # += N(0, sigma) with a per-point sigma = noise_combined_abs +
                              # noise_combined_rel * y_true (an additive floor PLUS a term that
                              # scales with the model prediction), seeded by `noise_combined_seed`.
                              # No _SD column -- the two coefficients are ESTIMATED jointly with the
                              # rate via a `sigma = prediction_formula sd_abs + sd_rel*<obs>` fit:
                              # sigma is a function of the PREDICTED state, so it cannot be a fixed
                              # data column (the state-dependent noise lesson, 48). Both fields
                              # (abs + rel) must be set together.
    noise_combined_rel: float = None
    noise_combined_seed: int = 0
    sd: float = None          # emit a constant _SD column at this value (independent of the
                              # gaussian noise above -- for a clean curve scored under chi_sq/laplace)
    sd_by_obs: tuple = ()     # ((obs_name, sd_value), ...): a DIFFERENT constant _SD per
                              # observable (channels measured with differing precision),
                              # overriding the single `sd`. A tight _SD on one channel pins
                              # its parameters; a loose _SD on another leaves its parameters
                              # weakly identified -- the priors lesson's weak-vs-strong setup (27)
    count_dispersion: float = None  # if set, resample each observable as OVER-DISPERSED integer
                              # COUNTS: negative-binomial draws with mean = the model value and this
                              # dispersion r (variance = mean + mean^2/r), seeded by `count_seed`.
                              # No _SD column is written -- a count likelihood (neg_bin) is
                              # self-normalizing (lesson 18).
    count_seed: int = 0
    count_replicates: int = 1  # write this many INDEPENDENT count observations per time point
                              # (replicate measurements): K stacked blocks, each an independent
                              # negative-binomial draw of the same model means (seeded by
                              # count_seed + k). Replicates are how over-dispersion is actually
                              # pinned -- one time course barely constrains it -- so the
                              # estimate-the-dispersion lesson (41) needs them (neg_bin_dynamic).
    cumulative_obs: tuple = ()  # observable columns that are CUMULATIVE counts in the model and
                              # must be differenced to per-interval INCIDENT counts before count
                              # sampling (row i - row i-1; row 0 kept raw) -- matching the conf's
                              # per-observable `cumulative` flag on those columns (ADR-0051, #418).
                              # So the committed .exp is incident counts, scored against the model's
                              # cumulative prediction differenced the same way (lesson 28).
    scale: float = None       # multiply every observable column by this constant -- data in
                              # ARBITRARY units (e.g. detector AU), for a scale-free shape
                              # objective (profile_objective = kl/wasserstein; lesson 19)
    outliers: tuple = ()      # ((row_index, obs_name, replacement_value), ...): gross outliers
                              # spliced into the named observable column (deterministic contamination)
    normalize: tuple = ()     # ((obs_name, type), ...): write the named column PRE-normalized
                              # (init/peak/zero, ADR-0053) -- data as an experimentalist reports it
                              # (e.g. dF/F0, % of max), which the conf's `normalization` key then
                              # reproduces on the simulation so the two are comparable (lesson 22)
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
    marker: str = 'default'    # 'default' (bngsim+newera) | 'slow' | 'jax' | 'antimony'
                               # ('jax'/'antimony' gate on an optional dependency: the conf
                               #  SKIPS where it is absent rather than failing)
    refused: bool = False      # True => a gradient fit that must be REFUSED, not run
    profile: dict = None       # {param: expected identifiability class} for profile_likelihood
    max_obj: float = None      # constraint fit: assert best objective <= this (0 => all satisfied)
    dragged_min_err: float = None  # cautionary conf: at least one `recover` param must be off by
                                   # >= this fraction (a non-robust objective broken by outliers)
    hazard: bool = False       # True => a fit whose model has a numerical hazard (finite-time
                               # blowup): some simulations MUST fail, yet the fit completes and
                               # recovers `recover` -- the verifier asserts fail_count > 0 too
    aborts: bool = False       # True => every candidate hits the hazard (all sims fail): the
                               # max_failed_simulations guard must abort the run with a PybnfError
    note: str = ''


@dataclass(frozen=True)
class LintCase:
    """One PEtab v2 fixture in the ``13_petab_lint_clinic`` gallery and how the
    linter should react to it.

    ``outcome`` selects the expectation:
      * ``'clean'``   -- ``lint_problem`` reports no errors;
      * ``'error'``   -- ``lint_problem`` reports errors, and ``task`` (a
        ``petab.v2.lint`` Check class name) is among the flagging tasks;
      * ``'warning'`` -- ``lint_problem`` reports NO errors, but ``task`` is among
        the WARNING-level items (some petab checks advise rather than reject);
      * ``'raises'``  -- the defect is structural, so ``Problem.from_yaml`` itself
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

    # --- wave 2 (backlog #1): the measurement/observable/parameter checks that
    #     need no new tables -----------------------------------------------------
    LintCase('pos_log_measurement', 'error', task='CheckPosLogMeasurements',
             blurb='a log-normal observable given a non-positive measurement value'),
    LintCase('duplicate_observable_id', 'error', task='CheckUniquePrimaryKeys',
             blurb='the observable table repeats a primary key (two obs_A rows)'),
    LintCase('model_entity_as_parameter', 'error',
             task='CheckValidParameterInConditionOrParameterTable',
             blurb='a model entity (the observable Obs_A) placed in the parameter table'),
    LintCase('measurement_bad_model_id', 'error', task='CheckMeasurementModelId',
             blurb='a measurement names a modelId no model_files entry defines'),
    LintCase('missing_config_file', 'raises',
             blurb='the problem omits the required parameter_files (rejected at load)'),

    # --- wave 2 (backlog #1): the experiment/condition-table batch --------------
    LintCase('missing_experiment_condition', 'error',
             task='CheckExperimentConditionsExist',
             blurb='an experiment period applies a conditionId the condition table lacks'),
    LintCase('undefined_experiment', 'warning', task='CheckUndefinedExperiments',
             blurb='a measurement references an experimentId no experiment defines (warns)'),
    LintCase('unused_experiment', 'warning', task='CheckUnusedExperiments',
             blurb='the experiment table defines an experiment no measurement uses (warns)'),
    LintCase('unused_condition', 'warning', task='CheckUnusedConditions',
             blurb='the condition table defines a condition no experiment applies (warns)'),
    LintCase('initial_change_symbol', 'error', task='CheckInitialChangeSymbols',
             blurb='a t=0 condition sets a target from a symbol outside the parameter table'),
)

# Two petab.v2.lint tasks are in the default task set but cannot be provoked
# through a file-based PEtab problem in petab 0.8.2, so the clinic documents them
# rather than shipping a fixture that fakes a trigger (see the clinic README):
#   * CheckExperimentTable (duplicate timepoints) -- ExperimentTable.from_df
#     groups rows by experimentId and iterates df[TIME].unique(), so two rows at
#     the same time collapse into ONE period; a duplicate timepoint is
#     unreachable without constructing Experiment objects by hand.
#   * CheckMeasuredExperimentsDefined -- absent from petab 0.8.2's
#     default_validation_tasks; the warning-level CheckUndefinedExperiments (the
#     `undefined_experiment` fixture) supersedes it.
LINT_UNCOVERED = ('CheckExperimentTable', 'CheckMeasuredExperimentsDefined')


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
class ChannelCase:
    """One readout CHANNEL in the ``20_petab_observable_parameters`` fixture.

    A PEtab v2 observable measured with its OWN estimated gain
    (``observableParameters``, a per-observable scale substituted into the
    ``observableFormula``) and its OWN estimated noise (``noiseParameters``, a
    per-observable sigma id -- the Boehm ``sd_*`` pattern, ADR-0037/0044). This
    records both the PEtab side a modeler writes and the native-conf lines
    ``import_job`` must produce, so the one manifest row pins the fixture *and* the
    expected import -- they cannot drift.
    """
    obs: str              # PEtab observableId
    raw_obs: str          # the model observable the gain scales (obs = gain * raw_obs)
    scale_param: str      # estimated observableParameters id (this channel's gain)
    sigma_param: str      # estimated noiseParameters id (this channel's noise level)
    petab_noise: str      # PEtab noiseDistribution ('normal' | 'laplace')
    conf_family: str      # noise_model family it imports to ('gaussian' | 'laplace')
    conf_sigma_key: str   # the imported sigma field name ('sigma' for gaussian, 'scale' for laplace)
    blurb: str = ''       # the channel's story (README + test id)


# Two readout channels of one A->B->C model, each with its own estimated detector
# GAIN (observableParameters) and its own estimated NOISE level (noiseParameters) --
# the standard PEtab mechanism for per-observable nuisances (the Boehm sd_* pattern).
# A Gaussian channel and a Laplace channel show the noise FAMILY is per-observable
# too. Each imports (through the real PEtab loader) to a native `observable:` formula
# carrying the gain and a `noise_model <obs> =` line carrying the estimated sigma.
OBS_PARAM_CASES = (
    ChannelCase('obs_B', 'Obs_B', 'scale_B', 'sd_B', 'normal', 'gaussian', 'sigma',
                blurb='a Gaussian channel with its own gain (scale_B) and sigma (sd_B)'),
    ChannelCase('obs_C', 'Obs_C', 'scale_C', 'sd_C', 'laplace', 'laplace', 'scale',
                blurb='a Laplace channel with its own gain (scale_C) and scale (sd_C)'),
)

# The model rates are estimated too (bound by bare id), so the imported job carries
# them alongside the six per-channel nuisances.
OBS_PARAM_MODEL_RATES = ('k1', 'k2')


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
            # The same fit by the other edition-2 gradient method (L-BFGS-B, a
            # quasi-Newton optimizer) -- lands on the same (r, K).
            ConfCheck('logistic_growth_lbfgs.conf', recover={'r': 1.2, 'K': 100.0}, tol=0.02),
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
            # Non-integer time grid (step 0.5, 0..4): the SBML/Antimony path outputs at
            # exactly the experiment's measurement times (bngsim sample_times), so the
            # half-integer points score just like the native BNGL path -- the #469/#470
            # regression fixture (previously the SBML grid dropped off-integer times).
            Dataset('decay.exp', obs=('Obs_A',), t_end=4, n_points=9),
        ),
        # The SAME A->B dynamics as BNGL / Antimony / SBML, all fit through bngsim,
        # all recovering the same k -- a backend/format interop regression.
        confs=(
            ConfCheck('fit_bngl.conf', recover={'k': 0.5}, tol=0.03),
            # `antimony` is an optional extra (pybnf[antimony] -> bngsim[antimony]), so
            # this conf is unrunnable on a stock install -- mark it so it SKIPS there
            # instead of failing on PyBNF's "requires optional dependency 'antimony'".
            # Its BNGL and SBML twins below need no marker: python-libsbml is a hard
            # dependency of bngsim, so the SBML path is present wherever bngsim is.
            ConfCheck('fit_antimony.conf', recover={'k': 0.5}, tol=0.03,
                      marker='antimony'),
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
            # A Student-t noise model (df=4) is robust too, via a tunable
            # tail-heaviness dial -- recovers the truth despite the same outliers.
            ConfCheck('decay_student_t.conf', recover={'k': 0.5, 'A0': 100.0}, tol=0.03),
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
            # Powell addendum -- the LOCAL contrast (a start-point conjugate-direction
            # optimizer). From a good start (var = alpha 1.3, gamma 0.9) it descends
            # straight to the truth; from a bad start it TRAPS at an aliased frequency.
            ConfCheck('oscillator_powell.conf', recover={'alpha': 1.2, 'gamma': 0.8}, tol=0.03),
            ConfCheck('oscillator_powell_trapped.conf', recover={'alpha': 1.2, 'gamma': 0.8},
                      dragged_min_err=0.10,
                      note='local optimizer from a wrong-basin start traps at an aliased frequency'),
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
        folder='17_bayesian_uncertainty',
        model='bateman_chain.bngl',
        truth={'k1': 0.8, 'k2': 0.25},
        build_free={'k1': ('uniform_var', 0.05, 3.0), 'k2': ('uniform_var', 0.02, 2.0)},
        datasets=(
            # A clean bateman curve carrying a constant _SD column: the data sits
            # exactly at the truth, and the _SD sets the likelihood width, so the
            # posterior is centred on the truth with a scale the sampler can resolve.
            # Both rates are O(1), so the DREAM chains mix them together well.
            Dataset('bateman_chain.exp', obs=('Obs_A', 'Obs_B', 'Obs_C'),
                    t_end=20, n_points=21, sd=3.0),
        ),
        # The Bayesian conf samples a whole posterior (a slow DREAM run that writes
        # credible intervals), asserted differently from a point-recovery fit, so it
        # has its own slow-tier verifier (tests/test_tutorial_bayesian.py) rather than
        # a ConfCheck here.
        confs=(),
    ),
    Example(
        folder='18_count_likelihood',
        model='mrna_decay.bngl',
        truth={'k_deg': 0.3, 'N0': 200.0},
        build_free={'k_deg': ('uniform_var', 0.02, 2.0),
                    'N0': ('uniform_var', 20.0, 1000.0)},
        datasets=(
            # Molecule COUNTS, not a smooth curve: each point is an over-dispersed
            # negative-binomial draw (dispersion r=40) around the model mean A(t).
            # No _SD column -- a count likelihood is self-normalizing.
            Dataset('mrna_decay.exp', obs=('Obs_A',), t_end=15, n_points=31,
                    count_dispersion=40.0, count_seed=3),
        ),
        confs=(
            # The count likelihood (negative binomial), estimating the dispersion r
            # as a nuisance: recovers the decay rate and starting count from the
            # over-dispersed counts.
            ConfCheck('mrna_decay_neg_bin.conf',
                      recover={'k_deg': 0.3, 'N0': 200.0}, tol=0.10),
        ),
    ),
    Example(
        folder='19_shape_objectives',
        model='pulse.bngl',
        truth={'k1': 0.8, 'k2': 0.25},
        build_free={'k1': ('uniform_var', 0.3, 3.0),
                    'k2': ('uniform_var', 0.05, 0.3)},
        datasets=(
            # The pulse B(t) in ARBITRARY units (scaled x1000): a shape objective
            # normalizes it away, so only the pulse shape -- and thus k1, k2 -- is fit.
            Dataset('pulse_shape.exp', obs=('Obs_B',), t_end=20, n_points=41,
                    scale=1000.0),
        ),
        confs=(
            # Two members of the profile_objective family recover the same rates from
            # the scale-free pulse shape: KL cross-entropy and Wasserstein distance.
            # (KL's un-normalized objective is larger-magnitude and converges a touch
            # looser than the fully-normalized Wasserstein -- hence the wider tol.)
            ConfCheck('pulse_kl.conf', recover={'k1': 0.8, 'k2': 0.25}, tol=0.05),
            ConfCheck('pulse_wasserstein.conf', recover={'k1': 0.8, 'k2': 0.25}, tol=0.03),
        ),
    ),
    Example(
        folder='21_numerical_hazards',
        model='quadratic_finite_time_growth.bngl',
        truth={'k': 0.5},
        # k > 0.625 pulls the finite-time pole 1/(k*X0) inside the window (X0=0.2,
        # t_end=8), so the upper part of this range blows up and those sims fail.
        build_free={'k': ('uniform_var', 0.1, 2.0)},
        datasets=(
            # A clean sub-blowup trajectory at the truth (pole at t=10, past t_end=8).
            Dataset('quadratic_finite_time_growth.exp', obs=('Obs_X',),
                    t_end=8, n_points=17),
        ),
        confs=(
            # The fit survives the blowups (some sims fail, scored +inf) and still
            # recovers k, guarded by wall_time_sim + max_failed_simulations.
            ConfCheck('blowup_survives.conf', recover={'k': 0.5}, tol=0.03, hazard=True),
            # Bounds placed entirely inside the hazardous region: every sim fails,
            # so max_failed_simulations aborts the run instead of spinning forever.
            ConfCheck('blowup_aborts.conf', recover={}, aborts=True),
        ),
    ),
    Example(
        folder='22_normalization',
        model='conversion.bngl',
        truth={'k': 0.5},
        build_free={'k': ('uniform_var', 0.05, 3.0)},
        datasets=(
            # A -> B in arbitrary units, then normalized as an experimentalist would
            # report it: the reactant Obs_A to its initial reading, the product Obs_B
            # to its peak. The absolute scale (detector gain, A0) is gone.
            Dataset('conversion.exp', obs=('Obs_A', 'Obs_B'), t_end=10, n_points=21,
                    normalize=(('Obs_A', 'init'), ('Obs_B', 'peak'))),
        ),
        confs=(
            # Normalize the simulation the same way -> shape matched, k recovered
            # without knowing A0 or the gain.
            ConfCheck('normalized_shape.conf', recover={'k': 0.5}, tol=0.03),
            # Forget the `normalization` lines: the raw-scale simulation can't match
            # the normalized data, and k is dragged far off (>= 10%).
            ConfCheck('no_normalization.conf', recover={'k': 0.5}, dragged_min_err=0.10),
        ),
    ),
    Example(
        folder='23_resume',
        model='decay.bngl',
        truth={'k': 0.5, 'A0': 100.0},
        build_free={'k': ('uniform_var', 0.05, 3.0), 'A0': ('uniform_var', 20.0, 400.0)},
        datasets=(
            Dataset('decay.exp', obs=('Obs_A',), t_end=10, n_points=21),
        ),
        # Checkpoint/resume is orchestrated by the CLI (`pybnf ... -r`), not by an
        # algorithm's run() alone, so it is verified by its own subprocess test
        # (tests/test_tutorial_resume.py), not the inline harness -- no ConfCheck.
        confs=(),
    ),
    Example(
        folder='24_moment_equations',
        model='moments.bngl',
        truth={'s': 1.0, 'birth_rate': 0.4, 'death_rate': 0.8},
        build_free={'s': ('uniform_var', 0.1, 5.0),
                    'birth_rate': ('uniform_var', 0.05, 2.0),
                    'death_rate': ('uniform_var', 0.05, 3.0)},
        datasets=(
            # Both closed moment trajectories of the birth-death-immigration process:
            # the mean rises to -s/(b-d)=2.5, the variance to 5.0.
            Dataset('moments.exp', obs=('Obs_Mean', 'Obs_Variance'), t_end=15, n_points=16),
        ),
        confs=(
            # Fitting mean AND variance jointly recovers all three process rates --
            # the variance is what separates birth from death (the mean gives only
            # s and the net b-d).
            ConfCheck('moment_fit.conf',
                      recover={'s': 1.0, 'birth_rate': 0.4, 'death_rate': 0.8}, tol=0.03),
        ),
    ),
    Example(
        folder='25_island_de',
        model='transit_pk.bngl',
        truth={'k_transit': 12.76, 'k_abs': 9.11, 'k_elim': 0.96},
        build_free={'k_transit': ('uniform_var', 1.0, 30.0),
                    'k_abs': ('uniform_var', 1.0, 30.0),
                    'k_elim': ('uniform_var', 0.1, 5.0)},
        datasets=(
            # Only the central (plasma) compartment is observed; all three rates must
            # be inferred from that one curve.
            Dataset('transit_pk.exp', obs=('Obs_Central',), t_end=8, n_points=33),
        ),
        confs=(
            # Island DE (job_type=de + islands/migrate_every/num_to_migrate) recovers
            # all three PK rates from the plasma curve alone.
            ConfCheck('island_de.conf',
                      recover={'k_transit': 12.76, 'k_abs': 9.11, 'k_elim': 0.96}, tol=0.03),
        ),
    ),
    Example(
        folder='26_mcmc_samplers',
        model='bateman_chain.bngl',
        truth={'k1': 0.8, 'k2': 0.25},
        build_free={'k1': ('uniform_var', 0.05, 3.0), 'k2': ('uniform_var', 0.02, 2.0)},
        datasets=(
            # Same clean-at-truth bateman posterior as lesson 17 (constant _SD sets the
            # likelihood width), sampled here by the mh and pt MCMC samplers.
            Dataset('bateman_chain.exp', obs=('Obs_A', 'Obs_B', 'Obs_C'),
                    t_end=20, n_points=21, sd=3.0),
        ),
        # The mh/pt posterior samplers write credible intervals (unlike am), asserted
        # by their own slow-tier verifier (tests/test_tutorial_mcmc.py) -- no ConfCheck.
        confs=(),
    ),
    Example(
        folder='27_priors',
        model='bateman_chain.bngl',
        truth={'k1': 0.8, 'k2': 0.25},
        # The flat-prior conf's parameters (the informative conf swaps k2's line for a
        # gamma_var; the test builds both confs itself). build_free is what regenerate_data
        # runs the truth simulation through -- the prior family here is irrelevant to the data.
        build_free={'k1': ('uniform_var', 0.05, 3.0), 'k2': ('uniform_var', 0.02, 2.0)},
        datasets=(
            # Two channels measured with DIFFERENT precision: Obs_A tight (_SD 3, pins k1),
            # Obs_C loose (_SD 25, so k2 is only weakly identified). Data sits exactly at the
            # truth; the per-observable _SD sets each channel's likelihood width. Obs_B is
            # NOT measured -- k2's only handle is the noisy Obs_C.
            Dataset('bateman_chain.exp', obs=('Obs_A', 'Obs_C'), t_end=20, n_points=21,
                    sd_by_obs=(('Obs_A', 3.0), ('Obs_C', 25.0))),
        ),
        # Two Bayesian confs (flat vs informative prior on the weak k2) sampled by DREAM;
        # the payoff is a credible-interval WIDTH comparison, asserted by a dedicated slow-tier
        # verifier (tests/test_tutorial_bayesian_priors.py) rather than a ConfCheck.
        confs=(),
    ),
    Example(
        folder='28_cumulative_counts',
        model='linearized_seir.bngl',
        truth={'beta_eff': 0.8, 'gamma': 0.3},   # sigma held fixed at the model default (0.5)
        build_free={'beta_eff': ('uniform_var', 0.2, 3.0),
                    'gamma': ('uniform_var', 0.1, 2.0)},
        datasets=(
            # Two epidemic COUNT channels (negative-binomial, dispersion r=50): Obs_I is the
            # current infectious count (a prevalence, sampled directly), and Obs_R is INCIDENT
            # recoveries per day -- the model's CUMULATIVE Obs_R differenced to per-interval
            # increments (cumulative_obs), matching the conf's `cumulative` flag. No _SD column
            # (a count likelihood is self-normalizing). Daily grid: t_end=12, 13 points.
            Dataset('seir_counts.exp', obs=('Obs_I', 'Obs_R'), t_end=12, n_points=13,
                    count_dispersion=50.0, count_seed=2, cumulative_obs=('Obs_R',)),
        ),
        confs=(
            # Fit beta_eff + gamma from the two count channels: a neg_bin base scores the
            # prevalence Obs_I, and a per-observable neg_bin+`cumulative` override scores the
            # incident recoveries. Count-data recovery, so a looser tol like lesson 18.
            ConfCheck('incidence_fit.conf', recover={'beta_eff': 0.8, 'gamma': 0.3}, tol=0.10),
        ),
    ),
    Example(
        folder='30_data_fusion',
        model='reversible_conversion.bngl',
        truth={'kf': 0.7, 'kr': 0.2},
        build_free={'kf': ('uniform_var', 0.05, 3.0), 'kr': ('uniform_var', 0.02, 2.0)},
        datasets=(
            # A relaxation TIME COURSE (Obs_B rising to equilibrium at the default total
            # A0=80): its relaxation rate is kf+kr.
            Dataset('relaxation.exp', obs=('Obs_B',), t_end=8, n_points=17, sd=1.0),
            # A steady-state TITRATION: the independent variable is the total amount A0
            # (not time), so each row is run to STEADY STATE and reads the equilibrium
            # Obs_B. This sees only the ratio kf/kr (the equilibrium constant).
            Dataset('titration.exp', obs=('Obs_B',), t_end=0, n_points=0,
                    scan='A0', doses=(20.0, 40.0, 80.0, 160.0, 320.0), sd=1.0),
            # (qualitative.prop is committed by hand, like lesson 1 -- BPSL facts, not a
            # model-generated .exp, so it is not regenerated here.)
        ),
        confs=(
            # One fit to all three data types at once. The titration alone is degenerate
            # (ratio only); fused with the kinetics + qualitative facts it recovers both
            # rates. de + refine.
            ConfCheck('data_fusion.conf', recover={'kf': 0.7, 'kr': 0.2}, tol=0.03),
        ),
    ),
    Example(
        folder='31_bngl_sbml_fit',
        model='binding_low.bngl',   # the primary (BNGL) model; the high dataset names its own
        truth={'kf': 0.002, 'kr': 0.3},
        build_free={'kf': ('uniform_var', 0.0002, 0.05), 'kr': ('uniform_var', 0.02, 3.0)},
        datasets=(
            # Two conditions of A + B <-> C, one modeled in each language, sharing kf, kr.
            # LOW ligand (B0=75) from the BNGL model -- native Obs_C column.
            Dataset('bind_low.exp', obs=('Obs_C',), t_end=8, n_points=9),
            # HIGH ligand (B0=150) from the SBML model -- the bngsim SBML path reports the
            # raw species, so this dataset's obs is the species id `C` (not `Obs_C`), and
            # regenerate_data drives it through the sbml_backend (model= override, ADR-0041).
            Dataset('bind_high.exp', obs=('C',), t_end=8, n_points=9, model='binding_high.xml'),
        ),
        confs=(
            # A joint fit whose two experiments' models are in DIFFERENT languages (BNGL +
            # SBML), both on the bngsim backend, sharing the rate constants. de + refine.
            ConfCheck('bngl_sbml_fit.conf', recover={'kf': 0.002, 'kr': 0.3}, tol=0.03),
        ),
    ),
    Example(
        folder='06_step_input',
        model='step_input.bngl',
        truth={'k': 0.6, 'J_base': 1.2, 'J_step': 2.4, 'tau': 4.0},
        build_free={'k': ('uniform_var', 0.1, 3.0),
                    'J_base': ('uniform_var', 0.1, 5.0),
                    'J_step': ('uniform_var', 0.1, 6.0),
                    'tau': ('uniform_var', 2.0, 7.0)},
        datasets=(
            Dataset('step_input.exp', obs=('Obs_X',), t_end=12, n_points=25),
        ),
        confs=(
            ConfCheck('step_input_de.conf',
                      recover={'k': 0.6, 'J_base': 1.2, 'J_step': 2.4}, tol=0.05),
            # The hard if() step is differentiable as of bngsim 0.12.2, INCLUDING the
            # crossing term that makes the switch time tau estimable -- so this conf
            # (formerly step_input_trf_refused.conf) recovers rather than refuses.
            ConfCheck('step_input_trf.conf',
                      recover={'k': 0.6, 'J_base': 1.2, 'J_step': 2.4, 'tau': 4.0},
                      tol=0.03,
                      note='hard if() step on the gradient path, switch time included'),
            # The refusal the lesson still teaches, and a durable one: an SSA
            # trajectory has no derivative w.r.t. the rate parameters to carry, so
            # forward sensitivities exist only for the ODE backend.
            ConfCheck('step_input_ssa_refused.conf', recover={}, refused=True,
                      note='stochastic (ssa) scored action -> gradient optimizer must refuse'),
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
            # The smooth sigmoid twin. It fits the same parameters to the same
            # accuracy as the hard step above -- the two differ in what they cost
            # the integrator, not in whether a gradient exists.
            ConfCheck('step_input_smooth_trf.conf',
                      recover={'k': 0.6, 'J_base': 1.2, 'J_step': 2.4, 'tau': 4.0},
                      tol=0.03),
        ),
    ),
    Example(
        folder='32_prior_gallery',
        model='bateman_chain.bngl',
        truth={'k1': 0.8, 'k2': 0.25},
        # The gallery swaps k2's prior across the whole family catalog; the prior is
        # irrelevant to the DATA, which regenerate_data simulates through the flat
        # build_free at the truth (identical setup to Lesson 27's weak-k2 bateman).
        build_free={'k1': ('uniform_var', 0.05, 3.0), 'k2': ('uniform_var', 0.02, 2.0)},
        datasets=(
            # Same two-quality-channel data as Lesson 27: Obs_A tight (_SD 3, pins k1),
            # Obs_C loose (_SD 25, so k2 is only weakly identified). Obs_B unmeasured.
            Dataset('bateman_chain.exp', obs=('Obs_A', 'Obs_C'), t_end=20, n_points=21,
                    sd_by_obs=(('Obs_A', 3.0), ('Obs_C', 25.0))),
        ),
        # A flat prior plus one conf per family (normal/laplace/gamma/beta/half_normal
        # positional + student_t via the parameter: record). The payoff -- each
        # informative prior narrows the weak k2 -- is a credible-interval WIDTH
        # comparison asserted by tests/test_tutorial_prior_gallery.py (structural build
        # check on all families + a slow-tier sampler run on a representative trio),
        # not a ConfCheck.
        confs=(),
    ),
    Example(
        folder='35_scale_free_objectives',
        model='decay.bngl',
        truth={'k': 0.4},
        build_free={'k': ('uniform_var', 0.02, 3.0)},
        datasets=(
            # One decay spanning three orders of magnitude (A0=1000 -> ~1) with 15%
            # MULTIPLICATIVE noise: huge absolute scatter on the early large points,
            # tiny on the informative small tail. The objective choice decides whether
            # the tail is used (norm_sos) or drowned out (sos).
            Dataset('decay.exp', obs=('Obs_A',), t_end=18, n_points=19,
                    noise_cv=0.15, noise_cv_seed=3),
        ),
        confs=(
            # sos (absolute error) is dominated by the large early residuals and drags k
            # off; ave_norm_sos normalizes by the COLUMN mean, which on a single column is
            # just a constant -> it reduces to sos and is dragged the same way (the teaching
            # point: column normalization is the MULTI-observable tool).
            ConfCheck('sos.conf', recover={'k': 0.4}, dragged_min_err=0.10),
            ConfCheck('ave_norm_sos.conf', recover={'k': 0.4}, dragged_min_err=0.10),
            # norm_sos (per-point RELATIVE error) weights every point equally and recovers k;
            # sod (L1) is a linear penalty, less dominated by the large points, and recovers too.
            ConfCheck('norm_sos.conf', recover={'k': 0.4}, tol=0.05),
            ConfCheck('sod.conf', recover={'k': 0.4}, tol=0.07),
        ),
    ),
    Example(
        folder='36_estimate_noise',
        model='decay.bngl',
        truth={'k': 0.5},
        build_free={'k': ('uniform_var', 0.05, 3.0)},
        datasets=(
            # A decay with CONSTANT additive noise (sd 4, seeded) and an _SD column
            # reporting that true noise level. The `read_exp_file _SD` conf reads it;
            # the `fit noise_level` conf ignores it and estimates a single constant
            # sigma jointly with k. A fine 41-point grid so the noise is well estimated.
            Dataset('decay.exp', obs=('Obs_A',), t_end=20, n_points=41,
                    noise_sd=4.0, noise_seed=1),
        ),
        # Three new-era `noise_model = normal` confs differing only in the sigma SOURCE
        # (fix_at 1 / read_exp_file _SD / fit noise_level) all recover k; the payoff --
        # the `fit` conf ALSO recovers the noise level ~ 4 -- is asserted by
        # tests/test_tutorial_estimate_noise.py (which needs a k-tight, sigma-loose split
        # a single-tolerance ConfCheck can't express), not here.
        confs=(),
    ),
    Example(
        folder='39_adaptive_mcmc',
        model='two_species_oscillator.bngl',
        truth={'k4': 41.77, 'k6': 92.2},
        # Fit the two constant fluxes that set the oscillation's DC baseline; k2/kd
        # (hence the frequency) are held fixed. Both fluxes move the offsets, so the
        # data pins their COMBINATION tighter than either alone -> a correlated posterior.
        build_free={'k4': ('uniform_var', 30.0, 55.0), 'k6': ('uniform_var', 75.0, 110.0)},
        datasets=(
            # Clean-at-truth 2SHO oscillation over ~3 periods; the constant _SD sets the
            # (deliberately loose) likelihood width so the tilted posterior is wide enough
            # for Adaptive Metropolis to sample and diagnose (R-hat/ESS via ArviZ).
            Dataset('two_species_oscillator.exp', obs=('Obs_S1', 'Obs_S2'),
                    t_end=3.0, n_points=61, sd=1.0),
        ),
        # am writes raw per-chain draws (Results/A_MCMC/Runs/params_*.txt) and NO
        # credible intervals; the ArviZ bridge (from_pybnf) reads those files directly, so
        # its slow-tier verifier (tests/test_tutorial_am_diagnostics.py) loads the run via
        # from_pybnf and asserts R-hat/recovery.
        confs=(),
    ),
    Example(
        folder='40_preconditioned_dream',
        model='oscillator.bngl',
        truth={'alpha': 1.2, 'gamma': 0.8},
        build_free={'alpha': ('uniform_var', 0.1, 5.0),
                    'gamma': ('uniform_var', 0.1, 5.0)},
        datasets=(
            # The linearized Lotka-Volterra oscillator (lesson 07's model). The
            # frequency omega = sqrt(alpha*gamma) is pinned tightly by the period, but
            # the individual rates trade off along alpha*gamma = const -> a long, thin,
            # strongly anti-correlated (alpha, gamma) posterior (corr ~ -0.99): exactly
            # the tilted geometry Preconditioned DREAM whitens its proposals to. A small
            # constant _SD (0.01) sets the likelihood width; zero-noise-at-truth data
            # centres the posterior on the truth so the 95% credible interval brackets it.
            Dataset('oscillator.exp', obs=('Obs_P', 'Obs_Q'),
                    t_end=12, n_points=25, sd=0.01),
        ),
        # p_dream writes credible intervals (it inherits the base sampler's histogram
        # step, unlike am); its slow-tier verifier (tests/test_tutorial_pdream.py) asserts
        # the 95% credible interval brackets truth, like the mh/pt verifier (lesson 26).
        confs=(),
    ),
    Example(
        folder='41_estimate_dispersion',
        model='sis_epidemic.bngl',
        truth={'beta': 1.2},   # gamma held fixed at the model default (0.4, a known recovery rate)
        build_free={'beta': ('uniform_var', 0.3, 3.0)},
        datasets=(
            # An SIS outbreak's infected COUNT, measured with REPLICATES: 6 independent
            # negative-binomial count observations (dispersion r=25) at each of 25 time
            # points. The population scale (S0/I0/N) is KNOWN and held fixed, and the
            # recovery rate gamma is known too -- so the transmission rate beta AND the
            # over-dispersion r are what the fit estimates. Replicates are what make the
            # dispersion identifiable (one time course barely constrains it). No _SD
            # column: a count likelihood is self-normalizing.
            Dataset('sis_counts.exp', obs=('Obs_I',), t_end=12, n_points=25,
                    count_dispersion=25.0, count_seed=6, count_replicates=6),
        ),
        # The fit estimates BOTH beta and the dispersion r_disp; beta comes back tight
        # but r_disp is inherently noisy (a variance-of-variance estimate), so the two
        # need different tolerances -- a single-tol ConfCheck can't express that. Its
        # dedicated verifier (tests/test_tutorial_neg_bin_dynamic.py) asserts beta tight
        # and r_disp within a generous window.
        confs=(),
    ),
    Example(
        folder='42_lognormal_error',
        model='infusion_washout.bngl',
        truth={'kel': 0.35},
        build_free={'kel': ('uniform_var', 0.05, 3.0)},
        datasets=(
            # An infusion+washout PK curve whose amount spans nearly three orders of
            # magnitude (a tall plateau ~30 down to a tiny washout tail ~0.05), with
            # MULTIPLICATIVE lognormal noise (log10 sigma 0.2, mean-aligned). The big
            # early/plateau points carry huge absolute scatter, the informative tail
            # tiny absolute scatter -- so a Gaussian (constant-sigma) fit over-weights
            # the plateau and under-uses the tail, while a lognormal fit weights every
            # point equally in log space. No _SD column (the confs set sigma with fix_at).
            Dataset('infusion_washout.exp', obs=('Obs_Drug',), t_end=24, n_points=41,
                    noise_lognormal_sigma=0.2, noise_lognormal_seed=8),
        ),
        confs=(
            # The right likelihood for multiplicative error: recovers kel from the whole
            # curve, tail included.
            ConfCheck('lognormal.conf', recover={'kel': 0.35}, tol=0.05),
            # A Gaussian (constant-sigma) fit is dragged off by over-weighting the
            # large-magnitude plateau points and ignoring the informative low tail.
            ConfCheck('gaussian_dragged.conf', recover={'kel': 0.35}, dragged_min_err=0.10),
        ),
    ),
    Example(
        folder='44_initialization',
        model='oscillator.bngl',
        truth={'alpha': 1.2, 'gamma': 0.8},
        build_free={'alpha': ('uniform_var', 0.1, 5.0),
                    'gamma': ('uniform_var', 0.1, 5.0)},
        datasets=(
            # The same linearized Lotka-Volterra oscillator as lesson 07 -- a hard,
            # multimodal (alpha, gamma) landscape. On a deliberately tiny budget it is
            # WHERE the initial population starts that decides whether the fit finds the
            # basin, which is what the initialization keys control.
            Dataset('oscillator.exp', obs=('Obs_P', 'Obs_Q'), t_end=12, n_points=25),
        ),
        confs=(
            # Informative normal priors near the truth, drawn from by
            # initialization_distribution = prior (the default): the initial population
            # is seeded where the answer is, so a tiny 8-iteration budget recovers.
            ConfCheck('prior_seeded.conf', recover={'alpha': 1.2, 'gamma': 0.8}, tol=0.05),
            # Flat uniform priors + the same tiny budget: no informative seeding, so the
            # search can't climb out of a wrong-frequency local minimum -- dragged far off.
            ConfCheck('uninformed.conf', recover={'alpha': 1.2, 'gamma': 0.8},
                      dragged_min_err=0.10,
                      note='no informative prior to seed from -> tiny budget cannot find the basin'),
        ),
    ),
    Example(
        folder='45_model_selection',
        model='richards.bngl',   # the TRUE model the data is generated from
        truth={'r': 0.8, 'K': 100.0, 'b': 3.0},
        build_free={'r': ('uniform_var', 0.1, 3.0), 'K': ('uniform_var', 50.0, 200.0),
                    'b': ('uniform_var', 0.5, 6.0)},
        datasets=(
            # An asymmetric Richards growth curve (shape exponent b=3, a sharp approach
            # to K) with gaussian noise + a constant _SD -- the data four competing
            # growth laws are fit to. All candidate models observe Obs_N, so they share
            # this one column.
            Dataset('growth.exp', obs=('Obs_N',), t_end=20, n_points=31,
                    noise_sd=2.0, noise_seed=5),
        ),
        # Four candidate confs (logistic / gompertz / richards / von_bertalanffy) all
        # fit this Richards data; only Richards recovers the truth, and the payoff is
        # the AIC RANKING across models -- asserted by the dedicated verifier
        # tests/test_tutorial_model_selection.py, not a per-conf ConfCheck.
        confs=(),
    ),
    Example(
        folder='46_model_checking',
        model='signaling_pulse.bngl',   # the healthy circuit (the impaired sibling is checked too)
        # A `check` lesson: no fitting and no .exp data -- the model is scored AS
        # WRITTEN against a BPSL .prop spec, so there is no recovery truth and no
        # free parameters. `truth` records the two models' identifying rate (the
        # healthy clearance vs the knocked-down lesion) for documentation only;
        # `datasets`/`confs` are empty (a check reads the .prop, not a generated
        # .exp, and the two check confs are asserted by the dedicated verifier
        # tests/test_tutorial_examples.py::test_tutorial_model_check_discriminates,
        # not the recover/constraint ConfCheck machinery).
        truth={'kclr_healthy': 0.5, 'kclr_impaired': 0.03},
        build_free={},
        datasets=(),
        confs=(),
    ),
    Example(
        folder='47_condition_perturbations',
        model='reversible_conversion.bngl',
        # One model + a condition: perturbation (edition-2 Mechanism A): a wildtype and a
        # reverse-reaction knockout of A <-> B, measured as two time courses, fit with ONE
        # model file. This is the first tutorial with regular (non-pre-equilibration)
        # conditions across >1 experiment, so it is the genuine {action} x {condition}
        # cross-product -- of which only the scored diagonal is simulated (#484, ADR-0069).
        truth={'kf': 0.7, 'kr': 0.2},
        build_free={'kf': ('uniform_var', 0.05, 3.0), 'kr': ('uniform_var', 0.02, 2.0)},
        datasets=(
            # WILDTYPE: the reaction as written -- Obs_B relaxes to B_eq = kf/(kf+kr)*A0 at
            # rate kf+kr, so one time course sees BOTH rates.
            Dataset('wildtype.exp', obs=('Obs_B',), t_end=8, n_points=17, sd=1.0),
            # KNOCKOUT: the ko condition sets kr=0, so Obs_B runs irreversibly to the full
            # total A0 at rate kf -- a clean second look at kf. regenerate_data applies the
            # condition, so this .exp is the knockout curve (not the wildtype).
            Dataset('knockout.exp', obs=('Obs_B',), t_end=8, n_points=17, sd=1.0,
                    condition=('ko', 'kr = 0')),
        ),
        confs=(
            # One fit to both conditions at once on ONE model + a condition: perturbation.
            # Only the scored (wildtype, knockout-under-ko) diagonal is simulated; the
            # off-diagonal cross-product is pruned (#484). de + refine.
            ConfCheck('condition_perturbations.conf', recover={'kf': 0.7, 'kr': 0.2}, tol=0.05),
        ),
    ),
    Example(
        folder='48_state_dependent_noise',
        model='decay.bngl',
        # A decay measured with COMBINED additive+proportional error: sigma = sd_abs + sd_rel*y,
        # an additive floor plus a term that scales with the predicted state. The fit estimates
        # both coefficients jointly with the rate via `sigma = prediction_formula sd_abs +
        # sd_rel*Obs_A` -- the first lesson whose sigma is a function of the PREDICTION, not the
        # data (35 scales with the data, 36 estimates a CONSTANT sigma, 42 is lognormal). ADR-0075.
        truth={'k': 0.4},
        build_free={'k': ('uniform_var', 0.05, 3.0)},
        datasets=(
            # sigma_i = 5.0 + 0.1*Obs_A(t_i): over A0=1000 -> ~0 the proportional term (sd_rel*A)
            # dominates the early points (at A=1000, sigma~=105) and the additive floor 5.0 the late
            # ones (crossover A=50) -- so the many high-A points constrain sd_rel and the long low-A
            # tail constrains sd_abs. No _SD column (sigma is a fit of the prediction, not data).
            Dataset('decay.exp', obs=('Obs_A',), t_end=20, n_points=41,
                    noise_combined_abs=5.0, noise_combined_rel=0.1, noise_combined_seed=11),
        ),
        # k recovers tightly; (sd_abs, sd_rel) recover loosely (a combined error model's two
        # coefficients are weakly identified) -- a k-tight / coefficient-loose split a single
        # ConfCheck tolerance cannot express, so tests/test_tutorial_state_dependent_noise.py
        # asserts recovery instead of a manifest ConfCheck (mirroring lesson 36).
        confs=(),
    ),
    Example(
        folder='49_measurement_time_uncertainty',
        model='decay.bngl',
        truth={'k': 1.0},
        build_free={'k': ('uniform_var', 0.2, 3.0)},
        # Data is a decay sampled at times PERTURBED from the reported ones (a timing error),
        # which the shared model-at-truth generator has no mode for -- so it is generated by the
        # lesson's own regenerate_data.py, and there is no Dataset here (ADR-0112, #587).
        datasets=(),
        confs=(
            # Marginalizing the latent time (sigma_t fixed, then estimated) removes the bias:
            # both recover k = 1. The estimate variant also re-discovers a non-zero sigma_t.
            ConfCheck('marginal.conf', recover={'k': 1.0}, tol=0.12, marker='slow',
                      note='time_error = truncated_normal, sigma_t fixed -> recovers k'),
            ConfCheck('estimate_sigma_t.conf', recover={'k': 1.0}, tol=0.12, marker='slow',
                      note='sigma_t estimated jointly -> still recovers k'),
            # The cautionary sibling: trusting the reported times drags k off the truth (~1.36),
            # so the marginalization above is provably not vacuous.
            ConfCheck('standard.conf', recover={'k': 1.0}, dragged_min_err=0.2, marker='slow',
                      note='ignoring the timing uncertainty biases k high'),
        ),
    ),
)


def example_by_folder(folder):
    for ex in EXAMPLES:
        if ex.folder == folder:
            return ex
    raise KeyError(folder)
