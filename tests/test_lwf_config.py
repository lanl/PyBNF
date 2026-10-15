"""What ``job_type = lwf`` refuses when the configuration loads, each refusal naming its cause."""

import pytest

from pybnf.algorithms.filters import liu_west as lw
from pybnf.parse import load_config
from pybnf.printing import PybnfError

from . import lwf_harness as C
from . import recovery_harness as H


def _refusal(tmp_path, **kw):
    """The message of a refusal made before any network is generated."""
    conf = C.write_job(tmp_path, **kw)
    with pytest.raises(PybnfError) as info:
        C.build(conf)
    return info.value.message


def _load_refusal(tmp_path, **kw):
    """A refusal on the loaded configuration, which needs BNG2.pl (skipped without it)."""
    H.require_bng2pl()
    return _refusal(tmp_path, **kw)


# --------------------------------------------------------------------------------------------
# The raw configuration
# --------------------------------------------------------------------------------------------

def test_edition_1_is_refused_by_name(tmp_path):
    message = _refusal(tmp_path, replace={
        'edition': None, 'model:': 'model = %s : %s' % (tmp_path / 'sir.bngl',
                                                         tmp_path / 'cases.exp'),
        'job_type': 'fit_type = lwf', 'experiment:': None, 'noise_model': None,
    }, extra=['objfunc = neg_bin_dynamic'])
    assert 'requires the edition-2 configuration surface' in message


def test_a_cluster_key_is_refused_by_name(tmp_path):
    message = _refusal(tmp_path, extra=['cluster_type = slurm'])
    assert 'does not accept cluster_type = slurm' in message and 'local cluster' in message


@pytest.mark.parametrize('line', ['max_iterations = 5', 'smoothing = 3', 'save_best_data = 1',
                                  'initialization_distribution = bounds', 'step_size = 0.1'])
def test_a_key_the_filter_does_not_read_is_reported_unused_once(tmp_path, capsys, line):
    H.require_bng2pl()
    load_config(C.write_job(tmp_path, replace={'verbosity': None}, extra=[line, 'verbosity = 1']))
    key = line.split(' ')[0]
    assert capsys.readouterr().out.count('key %s is not used in job_type lwf' % key) == 1


@pytest.mark.parametrize('line', ['normalization = floor 0.03', 'normalization = scale'])
def test_a_normalization_the_loader_would_apply_is_refused_by_name(tmp_path, line):
    message = _refusal(tmp_path, extra=[line])
    assert 'does not accept normalization' in message and 'before any row is weighed' in message


def test_a_postprocessing_script_is_refused_by_name(tmp_path):
    script = tmp_path / 'pp.py'
    script.write_text('def postprocess(data):\n    return data\n')
    message = _refusal(tmp_path, extra=['postprocess = %s epi' % script])
    assert 'does not accept postprocess' in message and 'no postprocessing script' in message


def test_an_analytical_objective_expression_is_refused_by_name(tmp_path):
    """An expression line was accepted and ignored: no schema owns it, so neither check saw it."""
    message = _refusal(tmp_path, extra=['expression = (x - 1)^2'])
    assert 'does not accept expression' in message and 'analytical objective' in message


def test_ind_var_rounding_is_read_by_the_objective_and_not_reported_unused(tmp_path, capsys):
    H.require_bng2pl()
    load_config(C.write_job(tmp_path, replace={'verbosity': None},
                            extra=['ind_var_rounding = 1', 'verbosity = 1']))
    assert 'ind_var_rounding' not in capsys.readouterr().out


@pytest.mark.parametrize('key, accepted, refused', [('refine', 0, 1), ('bootstrap', 0, 2)])
def test_a_key_is_accepted_at_the_value_that_asks_for_nothing_and_refused_at_any_other(
        tmp_path, key, accepted, refused):
    H.require_bng2pl()
    load_config(C.write_job(tmp_path / 'ok', extra=['%s = %s' % (key, accepted)]))
    message = _refusal(tmp_path / 'no', extra=['%s = %s' % (key, refused)])
    assert 'job_type = lwf accepts %s = %s only' % (key, accepted) in message


def test_bngl_backend_bionetgen_is_refused(tmp_path):
    message = _refusal(tmp_path, replace={'bngl_backend': 'bngl_backend = bionetgen'})
    assert 'does not accept bngl_backend = bionetgen' in message


@pytest.mark.parametrize('line, phrase', [
    ('start_point = beta 0.5', 'start_point declaration for beta'),
    ('observable: doubled, formula: 2*cases', 'measurement declaration for doubled'),
])
def test_a_declaration_the_filter_does_not_honour_is_refused(tmp_path, line, phrase):
    assert 'job_type = lwf does not read a %s' % phrase in _refusal(tmp_path, extra=[line])


def test_several_models_are_refused(tmp_path):
    (tmp_path / 'other.bngl').write_text(C.MODEL)
    assert '2 models' in _refusal(tmp_path, extra=['model: %s' % (tmp_path / 'other.bngl')])


def test_an_sbml_model_is_refused(tmp_path):
    message = _refusal(tmp_path, replace={'model:': 'model: %s' % (tmp_path / 'm.xml')})
    assert 'the model %s' % (tmp_path / 'm.xml') in message and 'one BNGL model' in message


@pytest.mark.parametrize('replace, extra, phrase', [
    ({}, ['experiment: again, data: {data}'], '2 experiment: lines'),
    ({'experiment:': 'experiment: epi, preequilibrate: c0, data: {data}'}, [], "'preequilibrate:'"),
    ({'experiment:': 'experiment: epi, type: parameter_scan, data: {data}'}, [],
     'type: parameter_scan'),
    ({'experiment:': 'experiment: epi, method: ssa, data: {data}'}, [], 'method: ssa'),
    ({'experiment:': 'experiment: epi, data: {data}, {data2}'}, [], '2 data files'),
    ({'experiment:': 'experiment: epi, data: {data}, {prop}'}, [], 'constraint file(s)'),
], ids=['two', 'preequilibrate', 'scan', 'ssa', 'replicates', 'constraints'])
def test_an_experiment_the_filter_cannot_assimilate_is_refused(tmp_path, replace, extra, phrase):
    names = dict(data=tmp_path / 'cases.exp', data2=tmp_path / 'more.exp',
                 prop=tmp_path / 'c.prop')
    C.write_data(tmp_path / 'more.exp')
    (tmp_path / 'c.prop').write_text('cases > 0 at time=1\n')
    replace = {k: v.format(**names) for k, v in replace.items()}
    assert phrase in _refusal(tmp_path, replace=replace, extra=[e.format(**names) for e in extra])


def test_parallel_count_below_1_is_refused(tmp_path):
    message = _refusal(tmp_path, extra=['lwf_independent_runs = 2', 'parallel_count = 0'])
    assert 'does not accept parallel_count = 0' in message and 'at least one worker' in message


def test_population_size_and_parallel_count_with_one_run_are_refused(tmp_path):
    message = _refusal(tmp_path / 'size', extra=['population_size = 2'])
    assert 'population_size = 2' in message and 'lwf_independent_runs = 2' in message
    message = _refusal(tmp_path / 'workers', extra=['parallel_count = 4'])
    assert 'parallel_count with one independent run' in message


@pytest.mark.parametrize('line', [
    'lwf_particles = 1', 'lwf_jitter = 1', 'lwf_resample_threshold = 0',
    'lwf_forecast_intervals = -1', 'lwf_independent_runs = 0', 'lwf_bounds = clamp',
    'lwf_continue = 2', 'lwf_parameter_jitter = beta 0'])
def test_an_lwf_setting_out_of_range_is_refused_by_name(tmp_path, line):
    key = line.split(' ', 1)[0]
    replace = {key: None} if any(b.startswith(key) for b in C.BASE) else {}
    assert key in _refusal(tmp_path, replace=replace, extra=[line])


def test_a_parameter_jitter_line_that_names_no_free_parameter_once_is_refused(tmp_path):
    message = _load_refusal(tmp_path / 'unknown', extra=['lwf_parameter_jitter = kappa 0.2'])
    assert 'lwf_parameter_jitter names kappa, which is not a free parameter' in message
    message = _refusal(tmp_path / 'twice', extra=['lwf_parameter_jitter = beta 0.2',
                                                  'lwf_parameter_jitter = beta 0.3'])
    assert "lwf_parameter_jitter for 'beta' twice" in message
    message = _refusal(tmp_path / 'no_h', extra=['lwf_parameter_jitter = beta'])
    assert "'lwf_parameter_jitter = v h'" in message


@pytest.mark.parametrize('key', ['lwf_particles = 100', 'lwf_parameter_jitter = beta 0.2'])
def test_a_key_of_the_filter_is_reported_unused_under_another_job_type(tmp_path, capsys, key):
    H.require_bng2pl()
    conf = C.write_job(tmp_path, replace={'job_type': 'job_type = de', 'lwf_particles': None,
                                          'lwf_forecast_intervals': None, 'verbosity': None},
                       extra=[key, 'population_size = 4', 'max_iterations = 2', 'verbosity = 1'])
    load_config(conf)
    assert '%s is not used in job_type de' % key.split(' ')[0] in capsys.readouterr().out


# --------------------------------------------------------------------------------------------
# The loaded configuration
# --------------------------------------------------------------------------------------------

def test_an_initial_value_is_a_start_point_and_is_refused(tmp_path):
    message = _load_refusal(tmp_path, replace={'loguniform_var = beta': (
        'parameter: beta, prior: uniform, lower: 0.2, upper: 2, initial_value: 0.6')})
    assert 'start point for beta' in message


def test_data_bound_on_the_model_line_is_refused(tmp_path):
    message = _load_refusal(tmp_path, replace={
        'model:': 'model = %s : %s' % (tmp_path / 'sir.bngl', tmp_path / 'cases.exp')})
    assert 'cases.exp' in message


def test_a_constraint_bound_on_the_model_line_is_refused(tmp_path):
    (tmp_path / 'epi.con').write_text('cases < 100000 at time=5\n')
    message = _load_refusal(tmp_path, replace={
        'model:': 'model = %s : %s' % (tmp_path / 'sir.bngl', tmp_path / 'epi.con')})
    assert 'does not accept the constraints epi on the model sir' in message


@pytest.mark.parametrize('replace, extra, phrase', [
    ({'noise_model cases': None}, [], 'the column cases without the cumulative flag'),
    ({'noise_model cases': 'noise_model cases = neg_bin, dispersion = fit r, location = median, '
                           'cumulative'}, [], 'the median location on cases'),
    ({'noise_model cases': 'noise_model cases = normal, sigma = fit r, cumulative'}, [],
     'the Gaussian noise model on cases'),
    ({}, ['noise_model Inf = neg_bin, dispersion = fit r, location = mean'],
     'noise_model lines for Inf'),
], ids=['no_cumulative', 'median', 'gaussian', 'other_observable'])
def test_an_observation_model_the_filter_cannot_forecast_from_is_refused(tmp_path, replace, extra,
                                                                         phrase):
    assert phrase in _load_refusal(tmp_path, replace=replace, extra=extra)


@pytest.mark.parametrize('replace, extra, phrase', [
    ({'noise_model = neg_bin': 'noise_model = neg_bin, dispersion = fit r, location = median'},
     [], 'the whole-fit noise_model line (the median location)'),
    ({'noise_model = neg_bin': 'noise_model = neg_bin, dispersion = fit r2, location = mean'},
     ['loguniform_var = r2 1 100'],
     "the whole-fit noise_model line (the dispersion r2, where the cases line's is r)"),
    ({}, ['noise_location = median'],
     'the whole-fit noise_model line (the median location)'),
    ({'noise_model = neg_bin': 'noise_model = neg_bin, dispersion = fix_at 30, location = mean',
      'noise_model cases': 'noise_model cases = neg_bin, dispersion = fix_at 20, location = mean, '
                           'cumulative', 'loguniform_var = r': None}, [],
     "the whole-fit noise_model line (the dispersion fix_at 30, where the cases line's is fix_at 20)"),
    ({'noise_model cases': 'noise_model cases = neg_bin, dispersion = fix_at 20, location = mean, '
                           'cumulative'}, [],
     'the whole-fit noise_model line (the dispersion source FreeParameterSigma)'),
], ids=['median', 'other_dispersion', 'noise_location_median', 'fixed_differ', 'fixed_vs_fit'])
def test_a_whole_fit_objective_that_differs_from_the_columns_line_is_refused(tmp_path, replace,
                                                                             extra, phrase):
    """The weights never read the whole-fit objective, so one that differs would be ignored. The
    hint spells out the two lines with the column's dispersion."""
    message = _load_refusal(tmp_path, replace=replace, extra=extra)
    assert 'job_type = lwf does not accept ' + phrase in message
    dispersion = 'fix_at 20' if 'noise_model cases' in replace else 'fit r'
    assert "'noise_model = neg_bin, dispersion = %s, location = mean'" % dispersion in message


def test_a_fixed_dispersion_is_quoted_exactly_in_the_refusal_and_its_hint(tmp_path):
    """%g quoted 20.000001 and 20.000004 both as 20, so the refusal named two equal values and its
    hint told the user to write a value other than the one they wrote."""
    message = _load_refusal(tmp_path, replace={
        'noise_model = neg_bin': 'noise_model = neg_bin, dispersion = fix_at 20.000001, '
                                 'location = mean',
        'noise_model cases': 'noise_model cases = neg_bin, dispersion = fix_at 20.000004, '
                             'location = mean, cumulative',
        'loguniform_var = r': None})
    assert "fix_at 20.000001, where the cases line's is fix_at 20.000004" in message
    assert "'noise_model = neg_bin, dispersion = fix_at 20.000004, location = mean'" in message


def _fixed_dispersion(value):
    return {'noise_model = neg_bin': 'noise_model = neg_bin, dispersion = fix_at %s, '
                                     'location = mean' % value,
            'noise_model cases': 'noise_model cases = neg_bin, dispersion = fix_at %s, '
                                 'location = mean, cumulative' % value,
            'loguniform_var = r': None}


@pytest.mark.parametrize('value', ['0', '-5', 'inf', 'nan', '1e7', '1e12', '1e308', '1e-7',
                                   '1e-300', '5e-324'])
def test_a_fixed_dispersion_outside_the_range_the_filter_honours_is_refused(tmp_path, value):
    """0, -5 and inf loaded and every particle was then counted as failing to integrate; nan was
    refused as differing from itself; 1e12 was scored as the clip's floor, not the particles'
    means (lanl/PyBNF#987); 1e-300 crashed the forecast draw after every row was assimilated."""
    message = _load_refusal(tmp_path, replace=_fixed_dispersion(value))
    assert 'the dispersion fix_at %s on cases' % lw._exact(float(value)) in message
    assert 'between 1e-06 and 1e+06' in message


@pytest.mark.parametrize('value', ['1e-6', '1e6'])
def test_a_fixed_dispersion_at_either_end_of_the_range_loads(tmp_path, value):
    H.require_bng2pl()
    alg = C.build(C.write_job(tmp_path, replace=_fixed_dispersion(value)))
    assert alg._dispersion_value == float(value)


@pytest.mark.parametrize('line, phrase', [
    ('uniform_var = r -10 100', 'whose prior reaches below 0 (lower bound -10)'),
    ('normal_var = r 20 30', 'whose prior reaches below 0 (lower bound -inf)'),
    ('loguniform_var = r 1 1e7', 'whose prior reaches above 1e+06 (upper bound 10000000)'),
    ('lognormal_var = r 1 1', 'whose prior reaches above 1e+06 (upper bound inf)'),
], ids=['below_0', 'unbounded_normal', 'above_1e6', 'unbounded_lognormal'])
def test_a_fitted_dispersion_whose_prior_leaves_the_range_is_refused(tmp_path, line, phrase):
    """Every particle drawn or moved below 0 was counted as failing to integrate, and one above
    1e6 risks a score for the wrong mean (lanl/PyBNF#987)."""
    message = _load_refusal(tmp_path, replace={'loguniform_var = r': line})
    assert 'the dispersion r, ' + phrase in message


def test_a_fitted_dispersion_with_a_prior_bounded_at_0_loads(tmp_path):
    H.require_bng2pl()
    alg = C.build(C.write_job(tmp_path, replace={'loguniform_var = r': 'uniform_var = r 0 100'}))
    assert alg._dispersion_name == 'r'


def test_an_objective_token_that_stands_for_the_columns_line_loads(tmp_path):
    """``objective = neg_bin_dynamic`` with ``noise_location = mean`` is the whole-fit line
    ``noise_model = neg_bin, dispersion = fit r__FREE, location = mean``."""
    H.require_bng2pl()
    config = load_config(C.write_job(tmp_path, replace={
        'noise_model = neg_bin': 'objective = neg_bin_dynamic',
        'noise_model cases': 'noise_model cases = neg_bin, dispersion = fit r__FREE, '
                             'location = mean, cumulative',
        'loguniform_var = r': 'loguniform_var = r__FREE 1 100'}, extra=['noise_location = mean']))
    assert lw._resolve_noise_model(config.obj, 'cases', config.config).name == 'r__FREE'


def test_two_data_columns_are_refused(tmp_path):
    H.require_bng2pl()
    conf = C.write_job(tmp_path)
    (tmp_path / 'cases.exp').write_text('# time cases Inf\n1 4 10\n2 6 12\n')
    with pytest.raises(PybnfError, match=r'2 data columns \(cases, Inf\)'):
        C.build(conf)


def test_data_whose_first_column_is_not_time_is_refused(tmp_path):
    (tmp_path / 'scan.exp').write_text('# beta cases\n0.5 4\n1.0 6\n')
    message = _load_refusal(tmp_path, replace={
        'experiment:': 'experiment: epi, data: %s' % (tmp_path / 'scan.exp')})
    assert "independent variable is 'beta'" in message


@pytest.mark.parametrize('times, phrase', [
    ((0, 1, 2), 'have the time 0. Every particle starts at t = 0'),
    ((1, 3, 2), 'do not increase at row 3 (t = 2)'),
    ((1, 2, 4), 'Row 3 of the experiment epi (t = 4) is 2 after the row before it'),
], ids=['not_after_0', 'not_increasing', 'uneven'])
def test_a_grid_that_is_not_one_step_per_row_from_the_origin_is_refused(tmp_path, times, phrase):
    assert phrase in _load_refusal(tmp_path, counts=(4, 6, 5), times=times)


@pytest.mark.parametrize('count, shown', [(-3, '-3'), ('-inf', '-inf'), ('inf', 'inf')],
                         ids=['negative', 'minus_inf', 'inf'])
def test_a_count_below_0_or_not_finite_is_refused(tmp_path, count, shown):
    message = _load_refusal(tmp_path, counts=(4, 6, count, 10))
    assert "Row(s) 3 of the experiment epi have the count %s in 'cases'" % shown in message
    assert 'row holding nan' in message


def test_a_finite_support_without_reflecting_bounds_is_refused(tmp_path):
    message = _load_refusal(tmp_path, replace={
        'uniform_var = gamma': 'uniform_var = gamma 0.1 0.5 u'})
    assert 'the parameter gamma, whose prior has finite support but no reflecting bounds' in message


def test_an_unknown_initialization_is_refused(tmp_path):
    assert 'initialization = sobol is not a way to draw' in _load_refusal(
        tmp_path, extra=['initialization = sobol'])


def test_the_key_lists_name_global_keys_only():
    from pybnf import config
    from pybnf.config_schema import GlobalConfig

    assert (lw._READ_GLOBAL_KEYS | set(lw._REFUSED_VALUES) | lw._CLUSTER_KEYS
            <= set(GlobalConfig.owned_keys()))
    assert lw._STRUCTURAL_KEYS <= config.STRUCTURAL_PASSTHROUGH


def test_every_passthrough_key_is_structural_reported_unused_or_refused():
    """No schema owns a passthrough key, so the shared check never reports one; one the filter
    neither reads nor refuses would be ignored without a word, as postprocess was."""
    from pybnf import config

    assert config.STRUCTURAL_PASSTHROUGH <= (lw._STRUCTURAL_KEYS | {'max_iterations'}
                                             | set(lw._REFUSED_GLOBAL_KEYS))


def test_a_petab_export_of_an_lwf_job_is_refused(tmp_path):
    from pybnf.petab.export import export_job
    with pytest.raises((PybnfError, NotImplementedError)) as info:
        export_job(C.write_job(tmp_path), tmp_path / 'petab')
    assert 'neg_bin' in str(info.value) or 'cumulative' in str(info.value)
