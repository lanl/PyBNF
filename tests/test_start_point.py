"""A fit's start point: the supported surface, its validation, and its record (#583, #559).

Both issues are about the same absence. There was no way to say "start this fit at exactly
this point, inside the declared box", every workaround failed *silently*, and no artifact
recorded where a run actually began -- so a fit displaced from its intended start reported a
plausible number and looked exactly like a correct one. ADR-0116 closes that with one fact,
two spellings, and one record:

* ``start_point = <parameter> <value>`` -- edition-agnostic, one line per parameter, the only
  spelling a legacy ``*_var`` conf can use (the configuration #559 was filed against);
* ``parameter: <id>, ..., initial_value: <v>`` -- the edition-2 record field (ADR-0043),
  which until now was honored by the twelve population/sampler fit_types and silently
  discarded by the seven start-point optimizers, i.e. by exactly the ones both issues name;
* ``Results/start_point.txt`` -- the resolved start, written before anything is scored.

The tests below are grouped by the claim they pin: resolution, refusal, the record, and the
two silent failures that motivated the work.
"""
import numpy as np
import pytest

from . import integration_harness as H
from .context import config
from pybnf.algorithms.optimizers.cmaes import CMAESAlgorithm
from pybnf.algorithms.optimizers.local_base import StartPointOptimizer
from pybnf.printing import PybnfError
from pybnf.pset import FreeParameter


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _conf(tmp_path, extra=None, fit_type='cmaes', bounds=(-10.0, 10.0), n=2):
    """A real ``Configuration`` over an analytical 2-D Gaussian with ``uniform_var``
    parameters ``p1..pN`` -- the legacy positional declaration style, on purpose: the
    ``start_point`` key has to work for a conf that cannot express ``initial_value``."""
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec([2.0, -1.0], [1.0, 1.0]))
    lo, hi = bounds
    base = {
        'output_dir': str(tmp_path) + '/out',
        'models': {tgt}, tgt: [exp], 'exp_data': {exp},
        'objfunc': 'direct_pass', 'fit_type': fit_type, 'initialization': 'lh',
        'delete_old_files': 1, 'verbosity': 0, 'wall_time_sim': 0, 'random_seed': 1234,
        'population_size': 4, 'max_iterations': 2,
    }
    base.update({('uniform_var', 'p%d' % (i + 1)): [lo, hi] for i in range(n)})
    base.update(extra or {})
    return config.Configuration(base)


def _probe(variables, start_point=None):
    """A bare :class:`StartPointOptimizer` over hand-built parameters -- the resolver
    without a model, a backend, or a scheduler. The gradient optimizers cannot be
    constructed at all without a sensitivity-capable bngsim build (their ``_after_init``
    gates run before start resolution), so this is the only way to cover the shared
    resolver in the default test tier."""
    class Probe(StartPointOptimizer):
        START_POINT_KEY = 'probe_start_point'

        def __init__(self):
            self.variables = variables
            self.config = type('C', (), {})()
            self.config.config = {}
            self.config.start_point = dict(start_point or {})

        def start_run(self):
            raise NotImplementedError

        def got_result(self, res):
            raise NotImplementedError

    return Probe()


def _values(pset):
    return {p.name: p.value for p in pset}


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
class TestStartPointResolution:

    def test_declared_start_point_is_the_start(self, tmp_path):
        """The headline. A declared start point is where the fit begins -- for cmaes, and
        through the same shared resolver for powell / sim / gntr / lbfgs / trf / ms."""
        conf = _conf(tmp_path, {('start_point', 'p1'): 3.5, ('start_point', 'p2'): -4.25})
        assert conf.start_point == {'p1': 3.5, 'p2': -4.25}
        assert _values(CMAESAlgorithm(conf).start_pset) == {'p1': 3.5, 'p2': -4.25}

    def test_no_declared_start_point_is_byte_identical(self, tmp_path):
        """The common case must not move: with nothing declared the start is the box
        center it has always been."""
        assert _values(CMAESAlgorithm(_conf(tmp_path)).start_pset) == {'p1': 0.0, 'p2': 0.0}

    def test_partial_start_point_pins_only_what_is_declared(self, tmp_path):
        """A start point is partial by design -- naming one parameter must not force the
        user to restate every other one."""
        conf = _conf(tmp_path, {('start_point', 'p1'): 3.5})
        assert _values(CMAESAlgorithm(conf).start_pset) == {'p1': 3.5, 'p2': 0.0}

    def test_initial_value_reaches_a_start_point_optimizer(self, tmp_path):
        """#583/#559's core defect: ``initial_value`` was carried on FreeParameter.value,
        honored by de/pso/ss/sa and every sampler, and read by ``_resolve_start_pset`` on
        none of its branches -- so the seven optimizers the issues are about started at the
        box center instead. Measured before the fix: k -> 1.505, S0 -> 89.44."""
        vs = [FreeParameter('k', 'uniform_var', 1e-2, 3.0, bounded=True),
              FreeParameter('S0', 'loguniform_var', 20.0, 400.0, bounded=True)]
        got = _values(_probe(vs, {'k': 0.3, 'S0': 100.0})._resolve_start_pset())
        assert got == pytest.approx({'k': 0.3, 'S0': 100.0})

    def test_injected_refiner_start_still_wins(self, tmp_path):
        """A refine begins from what the search FOUND. A declared start point governs the
        fit; it must not hijack the polish phase that follows it."""
        conf = _conf(tmp_path, {('start_point', 'p1'): 3.5, ('start_point', 'p2'): 3.5})
        alg = CMAESAlgorithm(conf)
        injected = alg.start_pset.__class__([v.set_value(7.0) for v in alg.variables])
        conf.config['cmaes_start_point'] = injected
        assert _values(CMAESAlgorithm(conf).start_pset) == {'p1': 7.0, 'p2': 7.0}

    def test_mixed_bounded_and_point_parameters_no_longer_start_at_a_lower_bound(self):
        """A silent failure neither issue reported. ``_is_box_start`` was all-or-nothing, so
        one unbounded parameter sent EVERY parameter down the ``p1`` branch -- and ``p1`` for
        a bounded parameter is its LOWER BOUND, read as if it were a sampling-space start
        value. A ``loguniform_var`` over [1e-3, 1e3] started at 10**1e-3 = 1.0023, its lower
        corner, with nothing logged at any level. Resolving per parameter fixes it: the
        bounded parameter gets its box center, the point parameter keeps its point."""
        vs = [FreeParameter('boxed', 'loguniform_var', 1e-3, 1e3, bounded=True),
              FreeParameter('pt', 'var', 5.0, None)]
        got = _values(_probe(vs)._resolve_start_pset())
        assert got['boxed'] == pytest.approx(1.0)     # the box center, not 1.0023
        assert got['pt'] == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# Refusal -- the point of the surface is that it does not fail quietly
# --------------------------------------------------------------------------- #
class TestStartPointRefusals:

    def test_out_of_box_start_point_is_refused_not_folded(self, tmp_path):
        """#583 item 2, corrected. An out-of-box value was never *clamped to the bound*:
        ``FreeParameter.set_value`` folds it back with a periodic triangle wave, so it lands
        on an arbitrary interior point -- 250 into [1, 100] became 50, not 100 -- and says so
        only at DEBUG. A start point is refused instead."""
        with pytest.raises(PybnfError, match='out of bounds'):
            _conf(tmp_path, {('start_point', 'p1'): 999.0})

    def test_unknown_parameter_name_is_refused(self, tmp_path):
        with pytest.raises(PybnfError, match='unknown parameter'):
            _conf(tmp_path, {('start_point', 'nope'): 1.0})

    def test_non_finite_start_point_is_refused(self, tmp_path):
        with pytest.raises(PybnfError, match='non-finite'):
            _conf(tmp_path, {('start_point', 'p1'): float('inf')})

    def test_start_point_alongside_starting_params_is_refused(self, tmp_path):
        """Two ways to say the same thing, matched differently -- ``starting_params`` is
        positional by declaration order, a start point is by name."""
        with pytest.raises(PybnfError, match='starting_params'):
            _conf(tmp_path, {('start_point', 'p1'): 1.0, 'starting_params': [1.0, 2.0]})

    def test_start_point_on_job_type_check_is_refused(self, tmp_path):
        """``check`` builds an empty PSet and simulates the model's own values, so a start
        point for it is a statement about nothing."""
        with pytest.raises(PybnfError, match='check'):
            _conf(tmp_path, {('start_point', 'p1'): 1.0}, fit_type='check')

    def test_a_log_scaled_parameter_refuses_a_non_positive_start(self, tmp_path):
        """The likely user error is writing the log10 value, which the legacy ``logvar``
        convention invites. The hint says which number to write instead."""
        conf_keys = {('loguniform_var', 'p1'): [1e-3, 1e3], ('start_point', 'p1'): -3.0}
        with pytest.raises(PybnfError, match='log scale'):
            _conf(tmp_path, conf_keys, n=1)

    def test_contradictory_spellings_are_refused(self, tmp_path):
        """Both spellings for one parameter is fine when they agree; when they disagree
        there is no defensible winner, and silently picking one is the failure class this
        work exists to remove."""
        cfg = _conf(tmp_path)
        cfg.config[('parameter', 'p1')] = {'initial_value': '1.0'}
        cfg.config[('start_point', 'p1')] = 2.0
        with pytest.raises(PybnfError, match='contradictory start point'):
            cfg._load_start_point()


class TestStartingParamsIsLoud:
    """#559 §1: ``starting_params`` is read at exactly one site -- the Bayesian sampler base
    -- and was accepted-then-discarded by the other fourteen fit_types. A ``gntr`` job seeded
    with it produced bit-identical output to the same job with the line deleted."""

    @pytest.mark.parametrize('fit_type', ['de', 'cmaes', 'pso', 'ss'])
    def test_refused_on_a_fit_type_that_ignores_it(self, tmp_path, fit_type):
        with pytest.raises(PybnfError, match='starting_params'):
            _conf(tmp_path, {'starting_params': [1.0, 2.0]}, fit_type=fit_type)

    @pytest.mark.parametrize('fit_type', ['am', 'mh'])
    def test_still_accepted_by_the_samplers_that_read_it(self, tmp_path, fit_type):
        """All 338 shipped confs that set it are samplers, so this is the whole installed
        base and it must keep working."""
        conf = _conf(tmp_path, {'starting_params': [1.0, 2.0]}, fit_type=fit_type)
        assert conf.config['starting_params'] == [1.0, 2.0]


# --------------------------------------------------------------------------- #
# The record -- #583's second ask, useful regardless of the first
# --------------------------------------------------------------------------- #
class TestStartPointRecord:

    def _run(self, tmp_path, monkeypatch, extra=None):
        H.install(monkeypatch)
        conf = _conf(tmp_path, extra)
        alg = CMAESAlgorithm(conf)
        H.drive(alg)
        from pathlib import Path
        return (Path(alg.res_dir) / 'start_point.txt').read_text()

    def test_record_names_the_declared_start_and_its_source(self, tmp_path, monkeypatch):
        text = self._run(tmp_path, monkeypatch, {('start_point', 'p1'): 3.5})
        rows = {ln.split('\t')[0]: ln.split('\t') for ln in text.splitlines()
                if ln and not ln.startswith('#')}
        assert float(rows['p1'][1]) == pytest.approx(3.5)
        assert rows['p1'][2] == 'start_point'
        assert float(rows['p2'][1]) == pytest.approx(0.0)
        assert rows['p2'][2] == 'box_center'

    def test_record_states_the_declared_box(self, tmp_path, monkeypatch):
        text = self._run(tmp_path, monkeypatch, {('start_point', 'p1'): 3.5})
        row = [ln.split('\t') for ln in text.splitlines() if ln.startswith('p1')][0]
        assert float(row[3]) == pytest.approx(-10.0)
        assert float(row[4]) == pytest.approx(10.0)

    def test_record_is_written_even_without_a_declared_start(self, tmp_path, monkeypatch):
        """The provenance is the point: a run that declared nothing still has to say where
        it began, because that is the run whose start nobody can otherwise reconstruct."""
        text = self._run(tmp_path, monkeypatch)
        assert 'box_center' in text and 'starts\t' in text


# --------------------------------------------------------------------------- #
# The displacement that motivated #583
# --------------------------------------------------------------------------- #
def test_a_bounded_prior_pins_the_median_and_a_start_point_fixes_it():
    """#583 item 1, reproduced exactly and then closed.

    The documented trick for pinning a box-mode optimizer's start was a narrow prior at the
    desired value, because ``_resolve_start_pset`` returns ``value_from_quantile(0.5)``. But
    that is the prior's MEDIAN, and for a normal truncated to [lower, upper] the median is
    not the mean whenever the truncation is asymmetric. On the Borghans ``init_Z_state``
    (mean 0.0879205, sd 0.2, bounds [0, 1]) the left tail is cut at -0.44 sd and the fit
    starts at 0.17318 -- a factor of 1.97 out, with no symptom of its own.

    A ``TruncatedPrior`` reports ``has_bounded_support``, so this is not an edge case of the
    box branch: the box branch is what fires for every bounded prior, normal included."""
    v = FreeParameter('init_Z_state', 'normal_var', 0.0879205, 0.2, lb=0.0, ub=1.0)
    assert v.has_bounded_support
    displaced = _values(_probe([v])._resolve_start_pset())['init_Z_state']
    assert displaced == pytest.approx(0.17318, rel=1e-4)
    assert displaced / 0.0879205 == pytest.approx(1.97, rel=1e-2)

    pinned = _values(_probe([v], {'init_Z_state': 0.0879205})._resolve_start_pset())
    assert pinned['init_Z_state'] == pytest.approx(0.0879205)


def test_an_out_of_box_value_reflects_rather_than_clamping():
    """Both issues describe an out-of-box value as 'silently clamped to the bound'. It is
    not: ``set_value`` applies a periodic triangle-wave fold, so the value lands at an
    arbitrary interior point rather than on the nearest wall. Pinned here because it is the
    reason a start point is refused rather than corrected -- 'we moved it to the bound' would
    at least be predictable, and this is not."""
    v = FreeParameter('k', 'loguniform_var', 1e-5, 1e3, bounded=True)
    assert v.set_value(1e9).value == pytest.approx(1e-3)
    assert v.set_value(1e9).value != pytest.approx(1e3)
