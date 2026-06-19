"""Tests for ADR-0028 Chunk 3a: explicit simulation output points.

A new-era ``experiment:`` derives its simulation grid from the data's
independent-variable column, so the simulation outputs at exactly the data's
points instead of a uniform grid. This file covers the backend plumbing in
``pybnf/pset.py`` (no config syntax yet):

* ``TimeCourse`` / ``ParamScan`` carry an ``explicit_points`` field.
* ``BNGLModel.add_action`` emits ``sample_times`` / ``par_scan_vals`` (covers
  BNG2.pl and bngsim, which both read the bracket list out of the action text).
* ``SbmlModelNoTimeout.execute`` uses ``simulate(times=...)`` / an explicit-value
  loop, verified against a real RoadRunner run at irregular points.
"""

import os

import numpy as np
import pytest

from .context import pset, printing, data, raises


class TestExplicitPointsActions:
    """The action value objects: explicit_points sorting, dedup, and the t=0 rule."""

    def test_time_course_sorts_dedupes_and_forces_zero(self):
        tc = pset.TimeCourse({'suffix': 'e'}, explicit_points=[300, 10, 10, 100])
        # Sorted, deduplicated, with t=0 forced in for a correct integration start.
        assert tc.explicit_points == [0.0, 10.0, 100.0, 300.0]
        # time (t_end) defaults to the last point when not given explicitly.
        assert tc.time == 300.0

    def test_time_course_zero_already_present_not_duplicated(self):
        tc = pset.TimeCourse({'suffix': 'e'}, explicit_points=[0, 5, 10])
        assert tc.explicit_points == [0.0, 5.0, 10.0]

    def test_time_course_legacy_path_has_no_explicit_points(self):
        tc = pset.TimeCourse({'time': '10', 'step': '2'})
        assert tc.explicit_points is None
        assert tc.stepnumber == 5

    @raises(printing.PybnfError)
    def test_time_course_needs_a_positive_point(self):
        # Only t=0 would remain after forcing it in -- not a usable grid.
        pset.TimeCourse({'suffix': 'e'}, explicit_points=[0])

    def test_param_scan_sorts_dedupes_and_keeps_no_zero(self):
        ps = pset.ParamScan({'param': 'kAB', 'time': '500'},
                            explicit_points=[0.02, 0.005, 0.005, 0.0125])
        # Sorted + deduped, and -- unlike a time course -- no spurious 0 dose.
        assert ps.explicit_points == [0.005, 0.0125, 0.02]
        # min/max derived from the values so the legacy required-key check passes.
        assert ps.min == 0.005
        assert ps.max == 0.02

    def test_param_scan_legacy_path_has_no_explicit_points(self):
        ps = pset.ParamScan({'param': 'k', 'min': '0', 'max': '10', 'step': '2', 'time': '5'})
        assert ps.explicit_points is None
        assert ps.stepnumber == 5

    @raises(printing.PybnfError)
    def test_param_scan_needs_a_value(self):
        pset.ParamScan({'param': 'kAB', 'time': '500'}, explicit_points=[])


class TestBnglActionText:
    """BNGLModel.add_action emits sample_times / par_scan_vals into the action text."""

    def _model(self):
        return pset.BNGLModel('bngl_files/Simple.bngl')

    def test_time_course_emits_sample_times(self):
        m = self._model()
        m.add_action(pset.TimeCourse({'suffix': 'egf'}, explicit_points=[10, 20, 50, 100]))
        line = m.actions[-1]
        assert 'sample_times=>[0.0,10.0,20.0,50.0,100.0]' in line
        assert 'n_steps' not in line  # explicit points replace the uniform grid
        assert 'suffix=>"egf"' in line
        assert ('simulate', 'egf') in m.suffixes

    def test_time_course_sample_times_round_trips_under_find_t_length(self):
        # The emitted sample_times must parse back the way find_t_length counts output
        # rows (one row per sample time), so output-array sizing stays consistent.
        m = self._model()
        m.add_action(pset.TimeCourse({'suffix': 'egf'}, explicit_points=[10, 20, 50, 100]))
        m.bngl_file_text = m.actions[-1]
        assert m.find_t_length()['egf'] == 4  # 5 sample times (incl. t=0) -> 5 rows -> length 4

    def test_param_scan_emits_par_scan_vals_and_omits_min_max(self):
        m = self._model()
        m.add_action(pset.ParamScan({'param': 'kAB', 'time': '500', 'suffix': 'dose'},
                                   explicit_points=[0.005, 0.0065, 0.02]))
        line = m.actions[-1]
        assert 'par_scan_vals=>[0.005,0.0065,0.02]' in line
        # par_min/par_max/n_scan_pts must be absent: BNG ignores par_scan_vals when they
        # are present ("defined min/max takes precedence").
        assert 'par_min' not in line and 'par_max' not in line and 'n_scan_pts' not in line
        assert 'parameter=>"kAB"' in line

    def test_legacy_time_course_still_emits_n_steps(self):
        m = self._model()
        m.add_action(pset.TimeCourse({'time': '100', 'step': '10', 'suffix': 'leg'}))
        line = m.actions[-1]
        assert 'n_steps=>10' in line and 't_end=>100' in line
        assert 'sample_times' not in line

    @raises(printing.PybnfError)
    def test_time_course_too_few_points_for_bngl_raises(self):
        # BNG's sample_times needs >= 3 points; [0, 5] is only 2 after forcing t=0.
        m = self._model()
        m.add_action(pset.TimeCourse({'suffix': 'e'}, explicit_points=[5]))


@pytest.mark.roadrunner
class TestSbmlExplicitPoints:
    """Real RoadRunner runs at exactly the data's irregular points.

    abc.xml's embedded parameters are the ones that generated abc_data.exp and
    abc_scan.exp, so reproducing those data sets simultaneously proves (a) output
    lands on exactly the requested irregular points and (b) integration starts at
    the model baseline (t=0), not at the first requested time.
    """

    def _abc_model(self):
        f = 'bngl_files/abc.xml'
        ps = pset.PSet([
            pset.FreeParameter('kAB', 'uniform_var', 0, 1, value=0.01),
            pset.FreeParameter('kBA', 'uniform_var', 0, 1, value=0.01),
            pset.FreeParameter('kBC', 'uniform_var', 0, 1, value=0.1),
            pset.FreeParameter('kCB', 'uniform_var', 0, 1, value=0.1),
        ])
        return f, ps

    def test_time_course_outputs_at_exactly_the_data_points(self):
        f, ps = self._abc_model()
        exp = data.Data(file_name='bngl_files/abc/abc_data.exp')
        points = sorted(set(exp['time']))  # irregular: 10,20,30,40,50,100,200,300,400,500
        action = pset.TimeCourse({'suffix': 'abc_data'}, explicit_points=points)
        m = pset.SbmlModelNoTimeout(f, os.getcwd() + '/' + f, pset=ps, actions=(action,))
        result = m.execute(os.getcwd(), 'abc_tc_explicit', 1000)
        sim = result['abc_data']

        # Output time column is exactly the requested points with t=0 prepended.
        assert list(sim['time']) == [0.0] + points
        # Values reproduce the synthetic data -> correct integration from the baseline.
        # (RoadRunner integrating from times[0]=10 would give A=20, the IC, at t=10.)
        for row in range(exp.data.shape[0]):
            t = exp.data[row, 0]
            sim_row = np.argmax(np.isclose(sim['time'], t, atol=0.))
            assert np.isclose(sim['A'][sim_row], exp['A'][row], atol=0.05)
            assert np.isclose(sim['B'][sim_row], exp['B'][row], atol=0.05)
            assert np.isclose(sim['C'][sim_row], exp['C'][row], atol=0.05)

    def test_param_scan_sweeps_exactly_the_data_values(self):
        f, ps = self._abc_model()
        exp = data.Data(file_name='bngl_files/abc/abc_scan.exp')
        values = sorted(set(exp['kAB']))  # irregular dose grid
        action = pset.ParamScan({'param': 'kAB', 'time': '500', 'suffix': 'abc_scan'},
                               explicit_points=values)
        m = pset.SbmlModelNoTimeout(f, os.getcwd() + '/' + f, pset=ps, actions=(action,))
        result = m.execute(os.getcwd(), 'abc_scan_explicit', 1000)
        sim = result['abc_scan']

        assert sim.indvar == 'kAB'
        assert list(sim['kAB']) == values
        for row in range(exp.data.shape[0]):
            kab = exp.data[row, 0]
            sim_row = np.argmax(np.isclose(sim['kAB'], kab, atol=0.))
            assert np.isclose(sim['A'][sim_row], exp['A'][row], atol=0.05)
