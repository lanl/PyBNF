"""bngsim's analytic ``∂f/∂p`` decline, surfaced as something a fit's user sees (#606).

``CVodeSensInit1`` takes ONE sensitivity-RHS callback for every column, so a single
rate law bngsim cannot differentiate declines the analytic ``∂f/∂p`` for the *whole*
model and CVODES' internal difference quotient carries every column instead. That
substitution preserves correctness and multiplies cost -- one extra RHS evaluation per
column per step -- and on a fit measured in hours the cost is what ends runs: on
``Smith_BMCSystBiol2013`` all 25 columns fell back, every start timed out to ``inf``,
and thirteen hours produced nothing (#558). PyBNF said nothing at all; the only signal
was a bngsim log line on a worker.

These tests pin the two halves of ADR-0121's answer:

* **the verdict comes off the codegen artifact**, not off the log line. The log line is
  emitted while *generating* codegen source, which a warm structural cache skips
  entirely (lanl/bngsim#174), so it is present on the first build of a model and absent
  on every later one -- while the model is on the fallback all the same.
  ``test_a_declined_model_is_reported_identically_cold_and_warm`` is that measurement,
  against the real backend: same model, two probes, one verdict, reason only the first
  time.
* **the reason is prose only.** It is captured and reported when heard, and nothing --
  no policy, no refusal -- may key off its absence.
"""

import logging
import types

import pytest

from pybnf import _bngsim_caps
from pybnf.algorithms.optimizers.gradient_base import GradientOptimizer
from pybnf.printing import PybnfError


# --- route ladder: what answers "is this run on the analytic path?" ----------- #

class _FakeSim:
    """The Simulator surface :func:`analytic_sens_rhs_probe` reads, and nothing else."""

    def __init__(self, *, source='', so_path='', owned=None, published=None):
        self._codegen_c_source = source
        self._codegen_so_path = so_path
        if owned is not None:
            self._codegen_provides_sens_rhs = lambda: owned
        if published is not None:
            self.has_analytic_sens_rhs = published


def test_a_published_per_run_attribute_outranks_every_other_route():
    """Route 1 -- the key bngsim does not publish yet, honoured in BOTH directions.

    It fires on no build that exists, which is the point: naming it costs nothing and
    means PyBNF reads the real answer on the first build that grows one, without
    another PyBNF release. Same shape as ADR-0119's route 1."""
    sim = _FakeSim(published=False, owned=True,
                   source='bngsim_codegen_sens_rhs(void) {}')
    state, route = _bngsim_caps.analytic_sens_rhs_probe(sim)
    assert state is False
    assert 'has_analytic_sens_rhs' in route


def test_bngsims_own_method_outranks_reading_the_artifact_directly():
    """Route 2 -- upstream's answer to upstream's question, wherever it exists.

    Route 3 replicates it for the sub-0.14.0 builds PyBNF's floor still admits; a build
    carrying the method keeps answering correctly even if the symbol ever moves."""
    sim = _FakeSim(owned=True, source='no symbol here')
    state, route = _bngsim_caps.analytic_sens_rhs_probe(sim)
    assert state is True
    assert '_codegen_provides_sens_rhs' in route


def test_a_method_that_raises_falls_through_to_the_artifact():
    sim = _FakeSim(source='bngsim_codegen_sens_rhs(void) {}')
    sim._codegen_provides_sens_rhs = lambda: 1 / 0
    state, route = _bngsim_caps.analytic_sens_rhs_probe(sim)
    assert state is True
    assert 'C source' in route


@pytest.mark.parametrize('source,expected', [
    ('static void bngsim_codegen_sens_rhs(void) {}', True),
    ('static void bngsim_codegen_rhs(void) {}', False),
])
def test_a_jit_run_is_read_off_its_generated_source(source, expected):
    """Route 3a. The JIT backends compile no ``.so`` and keep the source on the
    Simulator; the source names the symbol only where it also defines the function,
    which is what makes the substring test equal to the ``.so`` symbol test."""
    state, route = _bngsim_caps.analytic_sens_rhs_probe(_FakeSim(source=source))
    assert state is expected
    assert 'C source' in route


def test_an_artifact_that_cannot_be_opened_reports_no_opinion():
    state, route = _bngsim_caps.analytic_sens_rhs_probe(
        _FakeSim(so_path='/nonexistent/rhs_deadbeef.so'))
    assert state is None
    assert 'could not be opened' in route


def test_a_run_with_no_codegen_artifact_reports_no_opinion():
    """``codegen=False`` integrates the interpreted RHS: there is nothing to read, and
    guessing in either direction is wrong. A false *present* hides the cost this whole
    probe exists to surface; a false *absent* warns about a fit that is fine."""
    state, route = _bngsim_caps.analytic_sens_rhs_probe(_FakeSim())
    assert state is None
    assert 'no codegen artifact' in route


def test_no_simulator_at_all_reports_no_opinion():
    assert _bngsim_caps.analytic_sens_rhs_probe(None)[0] is None


# --- the reason channel: prose only, and best-effort by construction ---------- #

_PLAIN_DECLINE = (
    "Forward sensitivity: the Functional rate law for reaction 7 "
    "('per_species_volume_scaling') could not be differentiated, so the analytic "
    "sensitivity RHS is declined for this model and CVODES' internal difference "
    "quotient is used instead (correct, but slower)."
)
_CROSSING_DECLINE = (
    "Forward sensitivity: reaction 1 branches on 'Virus<1'. The analytic sensitivity "
    "RHS is declined for this model, and CVODES' internal difference quotient -- which "
    "is used instead -- does NOT recover the missing term: it integrates the "
    "variational equation smoothly through a crossing whose time moves."
)


def _emit(*messages, level=logging.WARNING):
    for message in messages:
        logging.getLogger('bngsim').log(level, '%s', message)


def test_a_decline_is_captured_with_its_reason_trimmed_out():
    with _bngsim_caps.capture_sens_rhs_declines() as reasons:
        _emit(_PLAIN_DECLINE)
    assert len(reasons) == 1
    reason, fallback_is_wrong = reasons[0]
    assert reason.startswith('the Functional rate law for reaction 7')
    assert 'declined' not in reason          # the clause is the frame, not the reason
    assert fallback_is_wrong is False


def test_the_variant_whose_fallback_is_wrong_is_flagged_as_such():
    """bngsim >= 0.14.0 raises here rather than warning (lanl/bngsim#414/#416), so this
    is reachable only on a sub-0.14.0 build -- which PyBNF's floor still admits, and
    where the returned gradient is wrong rather than merely slow."""
    with _bngsim_caps.capture_sens_rhs_declines() as reasons:
        _emit(_CROSSING_DECLINE)
    assert reasons[0][1] is True


def test_unrelated_bngsim_chatter_is_not_mistaken_for_a_decline():
    with _bngsim_caps.capture_sens_rhs_declines() as reasons:
        _emit('Compiling codegen RHS (-O2): cc ...',
              'wall_time_sim=30 exceeded at 30.001s')
    assert reasons == []


def test_the_same_decline_reported_twice_is_recorded_once():
    with _bngsim_caps.capture_sens_rhs_declines() as reasons:
        _emit(_PLAIN_DECLINE, _PLAIN_DECLINE)
    assert len(reasons) == 1


def test_an_unrecognised_decline_is_kept_whole_rather_than_dropped():
    """The wording is not PyBNF's to depend on: a decline it cannot parse is still a
    decline, and reporting it verbatim beats reporting nothing."""
    with _bngsim_caps.capture_sens_rhs_declines() as reasons:
        _emit('the analytic sensitivity RHS is declined because reasons')
    assert reasons[0][0] == 'the analytic sensitivity RHS is declined because reasons'


def test_the_capture_leaves_the_bngsim_logger_as_it_found_it():
    """PyBNF's own root handlers must keep receiving the decline exactly as today --
    the line still lands in ``<prefix>.log``; this only adds a listener, briefly."""
    bngsim_logger = logging.getLogger('bngsim')
    before = list(bngsim_logger.handlers)
    with _bngsim_caps.capture_sens_rhs_declines():
        assert len(bngsim_logger.handlers) == len(before) + 1
    assert bngsim_logger.handlers == before


def test_a_capture_that_hears_nothing_is_not_a_claim_that_nothing_declined():
    """The whole reason the verdict may not come from this channel. An empty capture is
    indistinguishable between "analytic" and "served from a warm codegen cache", so
    :attr:`SensRhsStatus.declined` reads the verdict and never the reasons."""
    status = _bngsim_caps.SensRhsStatus(False, 'artifact', [], 25)
    assert status.declined is True
    assert status.fallback_is_wrong is False


# --- probe_sens_rhs: never raises, never takes a fit down --------------------- #

def test_a_model_that_cannot_be_prepared_reports_no_opinion():
    """A diagnostic that can end a fit is worse than no diagnostic."""
    def boom():
        raise RuntimeError('no engine model here')

    status = _bngsim_caps.probe_sens_rhs(boom, columns=4)
    assert status.analytic is None
    assert status.columns == 4
    assert 'could not be prepared' in status.route


def test_the_probe_reports_the_verdict_and_whatever_bngsim_said_building_it():
    def build():
        _emit(_PLAIN_DECLINE)
        return _FakeSim(source='rhs only')

    status = _bngsim_caps.probe_sens_rhs(build, columns=25)
    assert status.declined
    assert status.columns == 25
    assert status.reasons[0][0].startswith('the Functional rate law')


# --- what the fit does with the verdict --------------------------------------- #

def _model(name, status):
    return types.SimpleNamespace(name=name, analytic_sens_rhs_status=lambda: status)


def _report(models, policy='warn', n_variables=3):
    """Drive the real reporting code over a stand-in carrying only what it reads.

    ``_report_sensitivity_rhs`` touches ``model_list``, ``variables``, the config and
    ``_fit_type_label`` -- so binding the two real methods onto a namespace exercises
    them exactly, with none of the model building an ``Algorithm`` construction needs.
    """
    opt = types.SimpleNamespace(
        model_list=models,
        variables=[types.SimpleNamespace(name='p%d' % i) for i in range(n_variables)],
        config=types.SimpleNamespace(config={'sensitivity_fallback': policy}),
        _fit_type_label=lambda: 'trf',
    )
    opt._warn_sensitivity_fallback = types.MethodType(
        GradientOptimizer._warn_sensitivity_fallback, opt)
    return GradientOptimizer._report_sensitivity_rhs(opt)


def test_a_declined_model_is_named_on_the_console_with_its_cost(capsys):
    """Console, not log-only. The decline already reaches ``<prefix>.log`` today --
    bngsim's logger propagates to root -- and that is the channel that failed: a shared,
    noisy file written from N worker processes, one line per model, mid-run."""
    status = _bngsim_caps.SensRhsStatus(
        False, 'the compiled codegen artifact this run installs',
        [('reaction 7 could not be differentiated', False)], 25)
    _report([_model('Smith', status)])
    out = capsys.readouterr().out
    assert "model 'Smith'" in out
    assert '25 sensitivity columns' in out
    assert '25x' in out
    assert 'reaction 7 could not be differentiated' in out
    assert 'gradient stays correct' in out
    assert 'the compiled codegen artifact' in out    # how it decided


def test_the_fallback_warning_survives_verbosity_zero(capsys, monkeypatch):
    """A reader who turned the verbosity down is still a reader who would rather not
    spend the next thirteen hours -- the call ``_report_bngsim_build`` already makes for
    a stale compiled core. Only the how-it-decided line is verbosity-gated."""
    from pybnf import printing

    monkeypatch.setattr(printing, 'verbosity', 0)
    _report([_model('Smith', _bngsim_caps.SensRhsStatus(False, 'artifact', [], 25))])
    out = capsys.readouterr().out
    assert 'difference quotient' in out
    assert 'Read from' not in out


def test_a_decline_with_no_reason_says_why_there_is_no_reason(capsys):
    """The warm-cache case. Reporting the verdict with a shrug would read as a PyBNF
    defect; naming the cache tells the reader the verdict is still sound."""
    status = _bngsim_caps.SensRhsStatus(False, 'the compiled codegen artifact', [], 25)
    _report([_model('Smith', status)])
    out = capsys.readouterr().out
    assert 'warm codegen cache' in out


def test_the_wrong_gradient_variant_says_the_columns_are_wrong_not_merely_slow(capsys):
    status = _bngsim_caps.SensRhsStatus(
        False, 'artifact', [('it branches on Virus<1', True)], 4)
    _report([_model('Virus', status)])
    out = capsys.readouterr().out
    assert 'wrong at and after that crossing' in out
    assert '0.14.0' in out


def test_a_model_on_the_analytic_path_says_nothing_to_the_console(capsys):
    _report([_model('fine', _bngsim_caps.SensRhsStatus(True, 'artifact', [], 4))])
    assert capsys.readouterr().out == ''


def test_a_model_with_no_opinion_says_nothing_to_the_console(capsys):
    """No artifact to read is not evidence in either direction, so it is logged and
    never warned about."""
    _report([_model('quiet', _bngsim_caps.SensRhsStatus(None, 'no artifact', [], 4))])
    assert capsys.readouterr().out == ''


def test_a_backend_with_no_opinion_to_give_is_skipped(capsys):
    """A test double or a backend without the hook. The sensitivity-backend gate has
    already refused anything that cannot supply a gradient at all."""
    _report([types.SimpleNamespace(name='stub')])
    assert capsys.readouterr().out == ''


def test_error_refuses_the_fit_and_names_every_declined_model():
    declined = _bngsim_caps.SensRhsStatus(False, 'artifact', [], 25)
    with pytest.raises(PybnfError) as exc:
        _report([_model('a', declined),
                 _model('b', _bngsim_caps.SensRhsStatus(True, 'artifact', [], 25)),
                 _model('c', declined)], policy='error')
    assert "model 'a'" in exc.value.message and "model 'c'" in exc.value.message
    assert "model 'b'" not in exc.value.message
    assert 'sensitivity_fallback' in exc.value.message


def test_error_keys_off_the_verdict_so_a_reasonless_decline_still_refuses():
    """The reason is absent on a warm codegen cache, and a policy that keyed off it
    would refuse a fit on its first run and accept the same fit on its second."""
    with pytest.raises(PybnfError):
        _report([_model('a', _bngsim_caps.SensRhsStatus(False, 'artifact', [], 25))],
                policy='error')


def test_error_does_not_refuse_a_model_that_reported_no_opinion():
    """The knob refuses a KNOWN fallback; it cannot promise to detect one it could not
    read, and refusing an unreadable build would take down e.g. a ``codegen=False`` run
    that is perfectly fine."""
    _report([_model('a', _bngsim_caps.SensRhsStatus(None, 'no artifact', [], 25))],
            policy='error')


def test_ignore_skips_the_check_entirely(capsys):
    """Including the one Simulator construction per model it costs -- which is the
    reason to have an ``ignore`` at all."""
    def explode():
        raise AssertionError('the probe must not run under ignore')

    _report([types.SimpleNamespace(name='m', analytic_sens_rhs_status=explode)],
            policy='ignore')
    assert capsys.readouterr().out == ''


def test_the_warning_is_spoken_once_per_run_not_once_per_bootstrap_refit(capsys):
    """``reset()`` drops the routings so a bootstrap refit rebuilds them, but a model's
    differentiability is not a function of the resampled data -- so ``bootstrap = 100``
    must not print the same warning a hundred times."""
    models = [_model('m', _bngsim_caps.SensRhsStatus(False, 'artifact', [], 25))]
    opt = types.SimpleNamespace(
        model_list=models,
        variables=[types.SimpleNamespace(name='p0')],
        config=types.SimpleNamespace(config={'sensitivity_fallback': 'warn'}),
        _fit_type_label=lambda: 'trf',
        _sens_rhs_reported=False,
    )
    opt._warn_sensitivity_fallback = types.MethodType(
        GradientOptimizer._warn_sensitivity_fallback, opt)
    GradientOptimizer._report_sensitivity_rhs(opt)
    assert 'difference quotient' in capsys.readouterr().out
    GradientOptimizer._report_sensitivity_rhs(opt)
    assert capsys.readouterr().out == ''


def test_a_declined_model_with_no_column_count_falls_back_to_the_fit_width(capsys):
    """A backend that cannot say how wide its request is still gets a cost sentence;
    the fit's own free-parameter count is the right order of magnitude."""
    _report([_model('m', _bngsim_caps.SensRhsStatus(False, 'artifact', [], 0))],
            n_variables=7)
    assert '7 sensitivity columns' in capsys.readouterr().out


# --- the real backend: the measurement ADR-0121 is built on ------------------- #

_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="sens_rhs_fixture">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfSpecies>
      <species id="S" compartment="c" initialConcentration="100" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="k" value="0.3" constant="true"/>
      <parameter id="Km" value="10" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="deg" reversible="false" fast="false">
        <listOfReactants><speciesReference species="S" stoichiometry="1" constant="true"/></listOfReactants>
        <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">%s</math></kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""

# ``k*S`` -- Elementary, differentiated in closed form.
_ANALYTIC_LAW = '<apply><times/><ci>k</ci><ci>S</ci></apply>'
# ``k*S*|S-Km|`` -- Functional, and ``abs()`` is on bngsim's underivable list, so the
# whole model declines. Smooth away from S=Km and cheap to build, which is all this
# fixture needs: nothing here integrates it.
_DECLINED_LAW = ('<apply><times/><ci>k</ci><ci>S</ci>'
                 '<apply><abs/><apply><minus/><ci>S</ci><ci>Km</ci></apply></apply>'
                 '</apply>')


@pytest.fixture
def cold_codegen_cache(tmp_path, monkeypatch):
    """Make this test's first codegen genuinely the first one.

    A test that means to observe the COLD half has to defeat both of the caches that
    otherwise serve an artifact somebody else built: bngsim's on-disk codegen cache
    (moved to an empty directory -- bngsim documents patching the module attribute as
    the supported way), and PyBNF's own process-wide engine-template cache, whose
    templates carry the ``_codegen_so_path`` a previous test compiled. Without both,
    the test silently stops testing what it says it tests.
    """
    from pybnf import bngsim_sbml_model

    bngsim_codegen = pytest.importorskip('bngsim._codegen')
    cache = tmp_path / 'codegen_cache'
    cache.mkdir()
    monkeypatch.setattr(bngsim_codegen, 'CACHE_DIR', cache)
    bngsim_sbml_model._ENGINE_TEMPLATE_CACHE.clear()
    bngsim_sbml_model._ENGINE_TEMPLATE_WARM_ATTEMPTED.clear()
    yield cache
    bngsim_sbml_model._ENGINE_TEMPLATE_CACHE.clear()
    bngsim_sbml_model._ENGINE_TEMPLATE_WARM_ATTEMPTED.clear()


def _sbml_model(tmp_path, law, name):
    from pybnf.bngsim_sbml_model import BngsimSbmlModelNoTimeout
    from pybnf.pset import FreeParameter, PSet, TimeCourse

    path = tmp_path / ('%s.xml' % name)
    path.write_text(_SBML % law)
    pset = PSet([FreeParameter('k', 'uniform_var', 0.0, 1e6, value=0.3),
                 FreeParameter('S', 'uniform_var', 0.0, 1e6, value=100.0)])
    model = BngsimSbmlModelNoTimeout(
        str(path), str(path), pset=pset,
        actions=(TimeCourse({'time': '10', 'step': '1'}),))
    model.enable_output_sensitivities(params=['k'], ic=['S'])
    return model


@pytest.mark.bngsim_sbml
@pytest.mark.skipif(not _bngsim_caps.BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_the_scalar_path_has_no_sensitivity_rhs_to_be_on_either_side_of(tmp_path):
    model = _sbml_model(tmp_path, _ANALYTIC_LAW, 'scalar')
    model._sensitivity_request = None
    status = model.analytic_sens_rhs_status()
    assert status.analytic is None
    assert status.columns == 0


@pytest.mark.bngsim_sbml
@pytest.mark.skipif(not _bngsim_caps.BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_a_differentiable_model_reports_the_analytic_path(tmp_path, cold_codegen_cache):
    status = _sbml_model(tmp_path, _ANALYTIC_LAW, 'analytic').analytic_sens_rhs_status()
    assert status.analytic is True
    assert status.columns == 2               # one parameter axis + one IC axis
    assert status.reasons == []


@pytest.mark.bngsim_sbml
@pytest.mark.skipif(not _bngsim_caps.BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_a_declined_model_is_reported_identically_cold_and_warm(tmp_path,
                                                                cold_codegen_cache):
    """ADR-0121's measurement, and the reason the verdict is not the log line.

    Two probes of the same model. The first generates codegen source and hears bngsim
    decline; the second resolves the artifact straight out of the (now warm) cache,
    generates nothing, and hears nothing at all -- while running on exactly the same
    difference-quotient fallback. A design that took its verdict from the log line
    would report this model as fine on the second run of the fit, which is the run a
    user makes after the first came back empty.
    """
    cold = _sbml_model(tmp_path, _DECLINED_LAW, 'declined').analytic_sens_rhs_status()
    assert cold.analytic is False
    assert cold.columns == 2
    assert cold.reasons, 'a cold codegen cache must surface bngsim\'s own reason'
    assert 'abs' in cold.reasons[0][0]
    assert cold.fallback_is_wrong is False   # smooth: correct, just slower

    from pybnf import bngsim_sbml_model
    # A genuinely separate run of the same fit: drop the process-wide engine template
    # so nothing is inherited in memory, and leave ONLY bngsim's on-disk codegen cache
    # warm. That is the state a user's second `pybnf` invocation is in.
    bngsim_sbml_model._ENGINE_TEMPLATE_CACHE.clear()
    bngsim_sbml_model._ENGINE_TEMPLATE_WARM_ATTEMPTED.clear()
    warm = _sbml_model(tmp_path, _DECLINED_LAW, 'declined2').analytic_sens_rhs_status()
    assert warm.analytic is False, 'the verdict must not depend on the codegen cache'
    assert warm.reasons == [], 'a warm cache generates no source, so it says nothing'


# --- the config surface ------------------------------------------------------- #

def _load(tmp_path, monkeypatch, extra_lines=()):
    """A minimal, simulator-free edition-2 config, plus whatever lines are under test."""
    import json

    from pybnf import config, parse

    (tmp_path / 'gaussian.target').write_text(
        json.dumps({'type': 'gaussian', 'mean': [0.0], 'variance': [1.0]}))
    (tmp_path / 'target.exp').write_text('# index\tscore\n0\t0\n')
    monkeypatch.chdir(tmp_path)
    conf = ('edition = 2\njob_type = de\nobjective = sos\n'
            'model = gaussian.target : target.exp\n'
            'uniform_var = k__FREE 0 10\n'
            'population_size = 5\nmax_iterations = 5\nwall_time_sim = 0\n'
            + ''.join(line + '\n' for line in extra_lines))
    return config.Configuration(parse.ploop(conf.splitlines(keepends=True)))


def test_sensitivity_fallback_defaults_to_warn(tmp_path, monkeypatch):
    """The default must not change any run that works today: it adds a sentence."""
    assert _load(tmp_path, monkeypatch).config['sensitivity_fallback'] == 'warn'


@pytest.mark.parametrize('value', ['warn', 'error', 'ignore'])
def test_sensitivity_fallback_round_trips_through_the_parser(value, tmp_path,
                                                             monkeypatch):
    cfg = _load(tmp_path, monkeypatch, ['sensitivity_fallback = %s' % value])
    assert cfg.config['sensitivity_fallback'] == value


def test_an_unknown_sensitivity_fallback_is_refused_at_config_load(tmp_path,
                                                                   monkeypatch):
    with pytest.raises(PybnfError):
        _load(tmp_path, monkeypatch, ['sensitivity_fallback = maybe'])


# --- the wiring: a real gradient fit's setup reports before it spends anything -- #

def _declined_fit_config(tmp_path, **overrides):
    """A real edition-2 ``trf`` config over the declining SBML fixture.

    The data grid only has to parse -- nothing here simulates the model, and
    ``_setup_gradient_path`` is reached before the first evaluation, which is the whole
    point of checking there.
    """
    from . import recovery_harness as H

    xml = tmp_path / 'declined.xml'
    xml.write_text(_SBML % _DECLINED_LAW)
    exp = tmp_path / 'tc.exp'
    exp.write_text('# time\tS\tS_SD\n'
                   + '\n'.join('%d\t%g\t1' % (t, 100.0 - t) for t in range(4)) + '\n')
    return H.make_newera_config(
        tmp_path, str(xml), str(exp),
        {'k': ('uniform_var', 1e-2, 3.0), 'S': ('uniform_var', 10.0, 300.0)},
        'tc', 'trf', objective='chi_sq', random_seed=1234, population_size=1,
        max_iterations=1, sbml_backend='bngsim', **overrides)


@pytest.mark.bngsim_sbml
@pytest.mark.skipif(not _bngsim_caps.BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_a_real_gradient_setup_warns_before_the_fit_evaluates_anything(
        tmp_path, capsys, cold_codegen_cache):
    """The end the issue is about: a real ``trf`` fit over a model bngsim cannot
    differentiate says so at setup, on the console, on the head node -- not as a log
    line on a worker after the first evaluation, and not never."""
    from . import recovery_harness as H

    alg = H.build(_declined_fit_config(tmp_path), 'trf')
    capsys.readouterr()                      # discard construction chatter
    alg._setup_gradient_path()

    out = capsys.readouterr().out
    assert 'difference quotient' in out
    assert 'sensitivity columns' in out
    assert 'abs' in out                      # bngsim's own reason, cold cache


@pytest.mark.bngsim_sbml
@pytest.mark.skipif(not _bngsim_caps.BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_sensitivity_fallback_error_refuses_the_real_fit_at_setup(tmp_path):
    from . import recovery_harness as H

    alg = H.build(_declined_fit_config(tmp_path, sensitivity_fallback='error'), 'trf')
    with pytest.raises(PybnfError, match='sensitivity_fallback'):
        alg._setup_gradient_path()


@pytest.mark.bngsim_sbml
@pytest.mark.skipif(not _bngsim_caps.BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_a_differentiable_real_fit_setup_says_nothing(tmp_path, capsys,
                                                      cold_codegen_cache):
    """No new noise for the fits that were already fine -- the default has to be a
    sentence for the models that need it and silence for everything else."""
    from . import recovery_harness as H

    xml = tmp_path / 'fine.xml'
    xml.write_text(_SBML % _ANALYTIC_LAW)
    exp = tmp_path / 'tc.exp'
    exp.write_text('# time\tS\tS_SD\n'
                   + '\n'.join('%d\t%g\t1' % (t, 100.0 - t) for t in range(4)) + '\n')
    conf = H.make_newera_config(
        tmp_path, str(xml), str(exp),
        {'k': ('uniform_var', 1e-2, 3.0), 'S': ('uniform_var', 10.0, 300.0)},
        'tc', 'trf', objective='chi_sq', random_seed=1234, population_size=1,
        max_iterations=1, sbml_backend='bngsim')
    alg = H.build(conf, 'trf')
    capsys.readouterr()
    alg._setup_gradient_path()
    assert 'difference quotient' not in capsys.readouterr().out


@pytest.mark.bngsim
@pytest.mark.skipif(not _bngsim_caps.BNGSIM_HAS_OUTPUT_SENS,
                    reason='needs a bngsim build with the output_sensitivities feature')
def test_the_net_backend_answers_through_its_own_simulator_construction():
    """The BNGL/``.net`` twin of the SBML hook -- a different Simulator construction
    (``_codegen_kwargs`` plus the request), so it is worth exercising rather than
    assuming the two share a path. ``e2e_ode_decay.net`` is Elementary throughout, so
    the answer is the analytic one."""
    from pathlib import Path

    from pybnf import bngsim_model, pset

    net_path = Path(__file__).resolve().parent / 'bngl_files' / 'e2e_ode_decay.net'
    model = bngsim_model.BngsimModel(
        net_path.stem,
        ['simulate({method=>"ode",t_start=>0,t_end=>10,n_steps=>20,suffix=>"tc"})'],
        [('simulate', 'tc')], [], nf=str(net_path))
    model.param_set = pset.PSet([])

    assert model.analytic_sens_rhs_status().analytic is None   # scalar path
    model.enable_output_sensitivities(params=['k'], ic=[])
    status = model.analytic_sens_rhs_status()
    assert status.analytic is True
    assert status.columns == 1
