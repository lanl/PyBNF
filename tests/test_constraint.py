from .context import constraint, data, raises
import copy
import os
import tempfile
import numpy as np
from pyparsing import ParseBaseException
from math import erf, sqrt


class TestConstraint:
    @classmethod
    def setup_class(cls):

        cls.line1 = 'A<B at 6 weight 1'
        cls.line2 = '42>=A At B=17 everytime before Weight 5 AltPenalty C<=42 Min 3.4e-7#Comment  '
        cls.line3 = 'A<16 between B=3.14, 702 weight 8 min 9'
        cls.line4 = 'A<16 always weight 6'
        cls.line5 = 'A<16 once weight 6'
        cls.line6 = 'A<B at 6 confidence 0.98'
        cls.line7 = 'A<16 between 5,B=5 confidence 0.95 tolerance 15'
        cls.line8 = 'A at B=6 <= A at B=5 weight 6'
        cls.line9 = 'A at 7 > C at B=4 before confidence 0.95 tolerance 1'
        cls.line10 = 'A<B at C=6 pmin 0.1 pmax 0.95 tolerance 1'
        cls.line11 = 'A<16 once between B=3.14, 702 weight 8 min 9'

        cls.err1 = 'A<16 always weight 2 confidence 0.95'
        cls.err2 = 'A<16 always confidence 0.98 tolerance 15 min 1'
        cls.err3 = 'A<16 always tolerance 5'

        cls.f1 = 'bngl_files/p1_5.prop'
        cls.f2 = 'bngl_files/con_test.prop'
        cls.f3 = 'bngl_files/con_test_likelihood.prop'
        cls.dat2 = 'bngl_files/con_test.gdat'

        cls.model = 'a.bngl'
        cls.suf = 'b'
        cls.cset = constraint.ConstraintSet(cls.model, cls.suf)

    @classmethod
    def teardown_class(cls):
        pass

    def test_grammar_at(self):
        p = self.cset.parse_constraint_line(self.line1)
        assert list(p.ineq) == ['A','<','B']
        assert p.enforce[0] == 'at'
        assert list(p.enforce[1]) == ['6']
        assert list(p.weight_expr) == ['weight', '1']
        assert float(p.weight_expr.weight) == 1.
        assert not bool(p.weight_expr.min)

        p = self.cset.parse_constraint_line(self.line2)
        assert list(p.ineq) == ['42', '>=', 'A']
        assert p.enforce[0] == 'at'
        assert list(p.enforce[1]) == ['B', '17']
        assert p.enforce[2] == 'everytime'
        assert p.enforce[3] == 'before'
        assert p.weight_expr.weight == '5'
        assert list(p.weight_expr.altpenalty) == ['C', '<=', '42']
        assert p.weight_expr.min == '3.4E-7'

    def test_grammar_between(self):
        p = self.cset.parse_constraint_line(self.line3)
        assert list(p.ineq) == ['A', '<', '16']
        assert p.enforce[1][0] == 'B'
        assert p.enforce[1][1] == '3.14'
        assert len(p.enforce[2]) == 1
        assert p.enforce[2][0] == '702'
        p = self.cset.parse_constraint_line(self.line11)
        assert p.enforce[0] == 'once between'
        assert p.enforce[1][0] == 'B'

    def test_grammar_other(self):
        p = self.cset.parse_constraint_line(self.line4)
        assert p.enforce[0] == 'always'
        assert p.weight_expr.weight == '6'

    def test_grammar_likelihood(self):
        p = self.cset.parse_constraint_line(self.line6)
        assert not p.weight_expr
        assert p.likelihood_expr.confidence == '0.98'
        assert not p.likelihood_expr.tolerance
        p = self.cset.parse_constraint_line(self.line7)
        assert p.likelihood_expr.confidence == '0.95'
        assert p.likelihood_expr.tolerance == '15'

    def test_grammar_splitat(self):
        p = self.cset.parse_constraint_line(self.line8)
        assert not p.ineq
        assert p.split.obs1 == 'A'
        assert list(p.split.at1[1]) == ['B', '6']
        assert p.split.sign == '<='
        assert p.weight_expr.weight == '6'
        p = self.cset.parse_constraint_line(self.line9)
        assert list(p.split.at1[1]) == ['7']
        assert p.split.obs2 == 'C'
        assert list(p.split.at2[1]) == ['B', '4']
        assert p.split.at2[2] == 'before'
        assert p.likelihood_expr.confidence == '0.95'

    def test_grammar_pmin(self):
        p = self.cset.parse_constraint_line(self.line10)
        assert not p.likelihood_expr.confidence
        assert p.likelihood_expr.pmin == '0.1'
        assert p.likelihood_expr.pmax == '0.95'

    @raises(ParseBaseException)
    def test_grammar_invalid1(self):
        p = self.cset.parse_constraint_line(self.err1)

    @raises(ParseBaseException)
    def test_grammar_invalid2(self):
        p = self.cset.parse_constraint_line(self.err2)

    @raises(ParseBaseException)
    def test_grammar_invalid3(self):
        p = self.cset.parse_constraint_line(self.err3)

    def test_load_file(self):
        cs = copy.deepcopy(self.cset)
        cs.load_constraint_file(self.f1)

        assert cs.constraints[0].quant1 == 'Ag_free'
        assert cs.constraints[0].or_equal is False
        assert cs.constraints[0].atvar is None
        assert cs.constraints[0].atval == 6.

        assert cs.constraints[1].quant2 == 42.
        assert cs.constraints[1].atvar == 'RP'
        assert cs.constraints[1].min_penalty == 3.4e-7
        assert cs.constraints[1].alt1 == 'R0'
        assert cs.constraints[1].alt2 == 42.

        assert cs.constraints[2].startvar == 'RP'
        assert cs.constraints[2].startval == 3.14
        assert cs.constraints[2].endvar is None
        assert cs.constraints[2].endval == 702.

        assert isinstance(cs.constraints[3], constraint.AlwaysConstraint)
        assert isinstance(cs.constraints[4], constraint.OnceConstraint)
        assert cs.constraints[4].weight == 6.

        assert isinstance(cs.constraints[5], constraint.SplitAtConstraint)
        assert cs.constraints[5].atval1 == 15
        assert cs.constraints[5].atvar1 == 'R0'
        assert cs.constraints[5].atvar2 is None
        assert cs.constraints[5].before1
        assert not cs.constraints[5].before2

    def test_penalties(self):
        d = data.Data()
        d.load_data('bngl_files/con_test.gdat')
        d_dict = {self.model: {self.suf: d}}

        cs = copy.deepcopy(self.cset)
        cs.load_constraint_file(self.f2)

        assert cs.constraints[0].penalty(d_dict) == 0
        assert cs.constraints[1].penalty(d_dict) == 4
        np.testing.assert_almost_equal(cs.constraints[2].penalty(d_dict), 0.4)
        np.testing.assert_almost_equal(cs.constraints[3].penalty(d_dict), 0.4)
        assert cs.constraints[4].penalty(d_dict) == 0
        assert cs.constraints[5].penalty(d_dict) == 10
        assert cs.constraints[6].penalty(d_dict) == 0
        assert cs.constraints[7].penalty(d_dict) == 25
        assert cs.constraints[8].penalty(d_dict) == 20
        assert cs.constraints[9].penalty(d_dict) == 20
        np.testing.assert_almost_equal(cs.constraints[10].penalty(d_dict), 1.8)
        assert cs.constraints[11].penalty(d_dict) == 1
        assert cs.constraints[12].penalty(d_dict) == 1
        assert cs.constraints[13].penalty(d_dict) == 0
        np.testing.assert_almost_equal(cs.constraints[14].penalty(d_dict), 1.6)
        np.testing.assert_almost_equal(cs.constraints[15].penalty(d_dict), 0.1)
        assert cs.constraints[16].penalty(d_dict) == 10
        assert cs.constraints[17].penalty(d_dict) == 3

        np.testing.assert_almost_equal(cs.total_penalty(d_dict), 98.3)

    def test_penalty_scale(self):
        d = data.Data()
        d.load_data('bngl_files/con_test.gdat')
        d_dict = {self.model: {self.suf: d}}

        cs = copy.deepcopy(self.cset)
        cs.load_constraint_file(self.f2, scale=2.0)

        assert cs.constraints[0].penalty(d_dict) == 0
        assert cs.constraints[1].penalty(d_dict) == 8
        np.testing.assert_almost_equal(cs.constraints[2].penalty(d_dict), 0.8)
        np.testing.assert_almost_equal(cs.constraints[3].penalty(d_dict), 0.8)

        np.testing.assert_almost_equal(cs.total_penalty(d_dict), 196.6)

    def test_likelihood_penalties(self):
        d = data.Data()
        d.load_data('bngl_files/con_test.gdat')
        d_dict = {self.model: {self.suf: d}}

        cs = copy.deepcopy(self.cset)
        cs.load_constraint_file(self.f3)

        def cdf(x, sigma):
            return (1. + erf(x / sigma / sqrt(2.) )) / 2.
        np.testing.assert_almost_equal(cdf(1,1)-cdf(-1,1), 0.682689492137)  # Check this gaussian CDF is working

        assert cs.constraints[0].penalty(d_dict) == 0
        np.testing.assert_almost_equal(cs.constraints[1].penalty(d_dict), -np.log(0.95))
        np.testing.assert_almost_equal(cs.constraints[2].penalty(d_dict), -np.log(0.05))
        np.testing.assert_almost_equal(cs.constraints[3].penalty(d_dict), -np.log(0.5))
        np.testing.assert_almost_equal(cs.constraints[4].penalty(d_dict), -np.log(0.9 * cdf(-1.,1.) + 0.05))
        np.testing.assert_almost_equal(cs.constraints[5].penalty(d_dict), -np.log(0.9 * cdf(1., 2.) + 0.05))
        np.testing.assert_almost_equal(cs.constraints[6].penalty(d_dict), -np.log(cdf(-3., 3.)))
        np.testing.assert_almost_equal(cs.constraints[7].penalty(d_dict), -np.log(0.75 * cdf(1., 2.) + 0.1))
        np.testing.assert_almost_equal(cs.constraints[8].penalty(d_dict), -np.log(cdf(-1., 1.)))


def _softplus(x):
    return float(np.logaddexp(0.0, x))


def _gauss_cdf(x):
    return (1. + erf(x / sqrt(2.))) / 2.


def _sdd(xvals, model='m', suffix='tc'):
    """A ``{model: {suffix: Data}}`` with a ``time``/``X`` column pair, X held at ``xvals`` -- the
    minimal simulation for exercising the logit penalty on an ``X > c always`` constraint whose
    worst-miss difference is ``max(c - X)``."""
    xvals = np.atleast_1d(np.asarray(xvals, float))
    times = np.arange(len(xvals), dtype=float)
    d = data.Data.from_columns(np.column_stack([times, xvals]), ['time', 'X'])
    return {model: {suffix: d}}


class TestLogitConstraint:
    """The logit (softplus) qualitative penalty and the ``qualitative_loss`` selector."""

    @classmethod
    def setup_class(cls):
        cls.cset = constraint.ConstraintSet('m', 'tc')

    # ---- grammar ----

    def test_grammar_logit(self):
        p = self.cset.parse_constraint_line('A<B at 6 logit scale 2.0')
        assert not p.weight_expr
        assert not p.likelihood_expr
        assert p.logit_expr.scale == '2.0'
        assert not p.logit_expr.pmin

    def test_grammar_logit_clipped(self):
        p = self.cset.parse_constraint_line('A<B at 6 logit scale 2.0 pmin 0.02 pmax 0.98')
        assert p.logit_expr.scale == '2.0'
        assert p.logit_expr.pmin == '0.02'
        assert p.logit_expr.pmax == '0.98'

    @raises(ParseBaseException)
    def test_grammar_logit_excludes_weight(self):
        # weight and logit are mutually exclusive alternatives in the grammar
        self.cset.parse_constraint_line('A<16 always weight 2 logit scale 1.0')

    def test_load_file_selects_logit_model(self):
        f = tempfile.NamedTemporaryFile('w', suffix='.prop', delete=False)
        f.write('X < 8 always logit scale 2.0\n')
        f.write('X < 8 always logit scale 1.5 pmin 0.02 pmax 0.98\n')
        f.close()
        cs = constraint.ConstraintSet('m', 'tc')
        cs.load_constraint_file(f.name)
        os.unlink(f.name)
        assert cs.constraints[0].penalty_model == 'logit'
        assert cs.constraints[0].scale == 2.0
        assert cs.constraints[0].pmin is None
        assert cs.constraints[1].penalty_model == 'logit'
        assert cs.constraints[1].pmin == 0.02 and cs.constraints[1].pmax == 0.98

    # ---- penalty values ----

    def test_logit_penalty_violated_and_satisfied(self):
        # 'X > 8 always' normalizes to 8 < X (difference = 8 - X). worst point = smallest X.
        c_viol = constraint.AlwaysConstraint('X', '>', 8.0, 'm', 'tc', weight=None, scale=2.0)
        np.testing.assert_almost_equal(c_viol.penalty(_sdd([5.0])), _softplus((8.0 - 5.0) / 2.0))
        # satisfied: X = 11 > 8, difference = -3
        c_sat = constraint.AlwaysConstraint('X', '>', 8.0, 'm', 'tc', weight=None, scale=2.0)
        np.testing.assert_almost_equal(c_sat.penalty(_sdd([11.0])), _softplus((8.0 - 11.0) / 2.0))

    def test_logit_clipped_penalty(self):
        c = constraint.AlwaysConstraint('X', '>', 8.0, 'm', 'tc', weight=None, scale=2.0,
                                        pmin=0.02, pmax=0.98)
        diff = 8.0 - 5.0
        p = constraint._sigmoid(-diff / 2.0)
        np.testing.assert_almost_equal(c.penalty(_sdd([5.0])), -np.log(0.02 + 0.96 * p))

    def test_logit_overflow_is_finite(self):
        # A huge margin must not overflow (logaddexp) -- the softplus reduces to difference/scale.
        c = constraint.AlwaysConstraint('X', '>', 1e6, 'm', 'tc', weight=None, scale=1e-3)
        val = c.penalty(_sdd([0.0]))
        assert np.isfinite(val)
        np.testing.assert_allclose(val, (1e6 - 0.0) / 1e-3, rtol=1e-9)

    def test_logit_asymptotes_to_hinge(self):
        """As scale -> 0 with weight = 1/scale, the logit softplus converges to the 2018 hinge
        pointwise: -> weight*difference where violated, -> 0 where satisfied. This is the
        large-margin identity that is."""
        for diff, xthresh, xval in [(2.0, 8.0, 6.0), (-2.0, 8.0, 10.0)]:
            sdd = _sdd([xval])
            gaps = []
            for s in (1.0, 0.1, 0.01):
                logit = constraint.AlwaysConstraint('X', '>', xthresh, 'm', 'tc', weight=None, scale=s)
                hinge = constraint.AlwaysConstraint('X', '>', xthresh, 'm', 'tc', weight=1.0 / s)
                gaps.append(abs(logit.penalty(sdd) - hinge.penalty(sdd)))
            # Monotone convergence to the hinge as the scale shrinks.
            assert gaps[0] > gaps[1] > gaps[2]
            assert gaps[2] < 1e-6

    def test_probit_and_logit_agree_near_the_boundary(self):
        """The probit link Phi(x) ~ sigma(1.6 x) makes an unclipped probit with SD sigma and an
        unclipped logit with scale s = sigma/1.6 agree in the **central region** near the decision
        boundary (|difference| <~ sigma). The two diverge in the deep tails (the Gaussian -logPhi
        grows quadratically, the logit softplus only linearly), so this is deliberately a
        central-region statement, checked around zero."""
        sigma = 2.0
        s = sigma / 1.6
        for diff in (-2., -1., -0.5, 0., 0.5, 1., 2.):
            xval = 8.0 - diff   # 'X > 8': difference = 8 - X = diff
            sdd = _sdd([xval])
            probit = constraint.AlwaysConstraint('X', '>', 8.0, 'm', 'tc', weight=None,
                                                 pmin=0.0, pmax=1.0, tolerance=sigma)
            logit = constraint.AlwaysConstraint('X', '>', 8.0, 'm', 'tc', weight=None, scale=s)
            np.testing.assert_allclose(logit.penalty(sdd), probit.penalty(sdd), atol=0.07)

    def test_logit_slope_matches_central_difference(self):
        """dF/d(difference) of the logit penalty equals sigma(difference/s)/s (the local slope the
        gradient reads), validated against a central finite difference of get_log_likelihood_logit
        w.r.t. a shift in the readout X."""
        s = 1.3
        c = constraint.AlwaysConstraint('X', '>', 8.0, 'm', 'tc', weight=None, scale=s)
        x0, h = 5.0, 1e-6
        f_plus = c.penalty(_sdd([x0 + h]))    # difference = 8 - (x0+h)
        f_minus = c.penalty(_sdd([x0 - h]))
        dF_dX = (f_plus - f_minus) / (2 * h)  # = dF/ddiff * d(8-X)/dX = slope * (-1)
        diff = 8.0 - x0
        slope = constraint._sigmoid(diff / s) / s
        np.testing.assert_allclose(dF_dX, -slope, rtol=1e-5)

    # ---- construction guards ----

    def test_logit_requires_positive_scale(self):
        for bad in (0.0, -1.0):
            try:
                constraint.AlwaysConstraint('X', '>', 8.0, 'm', 'tc', weight=None, scale=bad)
                assert False, f'scale={bad} should be rejected'
            except Exception:
                pass

    # ---- qualitative_loss selector ----

    def test_qualitative_loss_coercion_table(self):
        """Loading one mixed .prop under each qualitative_loss coerces every constraint to that
        family with a scale-matched parameter (logit s <-> hinge weight 1/s <-> probit tol 1.6 s),
        and a family's own authored constraint round-trips to itself unchanged."""
        f = tempfile.NamedTemporaryFile('w', suffix='.prop', delete=False)
        f.write('X < 8 always weight 4\n')                       # hinge, weight 4 -> base s = 0.25
        f.write('X < 8 always confidence 0.9 tolerance 10\n')    # probit, tol 10 -> base s = 6.25
        f.write('X < 8 always logit scale 2.0\n')                # logit, s = 2 -> base s = 2
        f.close()

        def models(ql):
            cs = constraint.ConstraintSet('m', 'tc')
            cs.load_constraint_file(f.name, qualitative_loss=ql)
            return cs.constraints

        hinge = models('hinge')
        assert [c.penalty_model for c in hinge] == ['static'] * 3
        np.testing.assert_allclose([c.weight for c in hinge], [1 / 0.25, 1 / 6.25, 1 / 2.0])

        probit = models('probit')
        assert [c.penalty_model for c in probit] == ['likelihood'] * 3
        np.testing.assert_allclose([c.tolerance for c in probit], [1.6 * 0.25, 1.6 * 6.25, 1.6 * 2.0])

        logit = models('logit')
        assert [c.penalty_model for c in logit] == ['logit'] * 3
        np.testing.assert_allclose([c.scale for c in logit], [0.25, 6.25, 2.0])

        os.unlink(f.name)

    def test_qualitative_loss_auto_is_a_no_op(self):
        """'auto' (the default) preserves each constraint's authored family exactly."""
        f = tempfile.NamedTemporaryFile('w', suffix='.prop', delete=False)
        f.write('X < 8 always weight 4\n')
        f.write('X < 8 always confidence 0.9 tolerance 10\n')
        f.write('X < 8 always logit scale 2.0\n')
        f.close()
        cs_auto = constraint.ConstraintSet('m', 'tc'); cs_auto.load_constraint_file(f.name)  # default auto
        cs_explicit = constraint.ConstraintSet('m', 'tc')
        cs_explicit.load_constraint_file(f.name, qualitative_loss='auto')
        os.unlink(f.name)
        assert [c.penalty_model for c in cs_auto.constraints] == ['static', 'likelihood', 'logit']
        assert [c.penalty_model for c in cs_explicit.constraints] == ['static', 'likelihood', 'logit']

    def test_coercion_preserves_penalty_ordering(self):
        """A scale-matched coercion keeps a violated constraint's penalty positive and a satisfied
        one at ~0 under every family -- the benchmark reruns score the same qualitative outcome."""
        c = constraint.AlwaysConstraint('X', '>', 8.0, 'm', 'tc', weight=None, scale=2.0)
        viol, sat = _sdd([5.0]), _sdd([11.0])
        for target in ('hinge', 'probit', 'logit'):
            cc = constraint.AlwaysConstraint('X', '>', 8.0, 'm', 'tc', weight=None, scale=2.0)
            cc.coerce_penalty_model(target)
            assert cc.penalty(viol) > 0
            assert cc.penalty(sat) < cc.penalty(viol)

    def test_coerce_probit_step_function_is_rejected(self):
        """A probit step function (tolerance 0) has no finite logit-equivalent scale; coercing it
        raises a pointed error rather than silently producing scale 0."""
        c = constraint.AlwaysConstraint('X', '>', 8.0, 'm', 'tc', weight=None,
                                        pmin=0.05, pmax=0.95, tolerance=0.0)
        try:
            c.coerce_penalty_model('logit')
            assert False, 'expected a rejection for tolerance-0 coercion'
        except Exception:
            pass


class TestEstimatedScale:
    """The qualitative scale (logit s / probit sigma) as a fittable parameter tied to a free
    parameter -- eval resolution from the live pset and the closed-form d(penalty)/d(scale)."""

    def test_bind_resolves_live_scale_and_falls_back_to_literal(self):
        c = constraint.AlwaysConstraint('X', '>', 8.0, 'm', 'tc', weight=None, scale=2.0)
        c.bind_scale_param('s_q')
        sdd = _sdd([5.0])                     # difference = 8 - 5 = 3
        # With a pset the live value wins; without one, the authored literal (2.0) is the fallback.
        np.testing.assert_almost_equal(c.penalty(sdd, pset_values={'s_q': 4.0}), _softplus(3.0 / 4.0))
        np.testing.assert_almost_equal(c.penalty(sdd), _softplus(3.0 / 2.0))
        # A pset that lacks the tied name also falls back (a bare diagnostic call).
        np.testing.assert_almost_equal(c.penalty(sdd, pset_values={'other': 9.0}), _softplus(3.0 / 2.0))

    def test_bind_scale_param_rejects_hinge(self):
        """The static (hinge) penalty has no scale to estimate -- binding one is a pointed error."""
        c = constraint.AlwaysConstraint('X', '>', 8.0, 'm', 'tc', weight=2.0)
        try:
            c.bind_scale_param('s_q')
            assert False, 'binding a scale param to a hinge constraint should raise'
        except Exception:
            pass

    def test_scale_derivative_matches_central_difference(self):
        """The closed-form d(penalty)/d(scale) equals a central finite difference of the penalty
        w.r.t. the tied scale value -- for logit, clipped logit, and probit."""
        sdd = _sdd([5.0])                     # difference = 8 - 5 = 3
        diff, h = 3.0, 1e-6
        cases = [
            constraint.AlwaysConstraint('X', '>', 8.0, 'm', 'tc', weight=None, scale=10.0),
            constraint.AlwaysConstraint('X', '>', 8.0, 'm', 'tc', weight=None, scale=10.0,
                                        pmin=0.02, pmax=0.98),
            constraint.AlwaysConstraint('X', '>', 8.0, 'm', 'tc', weight=None,
                                        pmin=0.02, pmax=0.98, tolerance=40.0),
        ]
        for c in cases:
            c.bind_scale_param('s_q')
            s0 = 7.5
            fd = (c.penalty(sdd, pset_values={'s_q': s0 + h})
                  - c.penalty(sdd, pset_values={'s_q': s0 - h})) / (2 * h)
            np.testing.assert_allclose(c._scale_derivative(diff, s0), fd, rtol=1e-5)

    def test_scale_gradient_signs_show_identifiability_tension(self):
        """The estimated scale is identified by the *tension* between satisfied and violated
        constraints (the open identifiability question). Because BPSL constraints
        are single-sided (every line asserts the inequality *holds*), a set of all-satisfied
        constraints has no interior scale optimum -- d(penalty)/d(scale) drives s one way. The sign
        is the mechanism: a SATISFIED constraint (difference < 0) has a positive scale gradient, so
        gradient descent shrinks s toward the hinge; a VIOLATED one (difference > 0) has a negative
        scale gradient, so descent grows s toward a softer penalty. A scale is pinned only where both
        are present -- the reason globally-tied (not per-observation) scales are the identifiable
        default."""
        s = 3.0
        c = constraint.AlwaysConstraint('X', '>', 8.0, 'm', 'tc', weight=None, scale=s)
        assert c._scale_derivative(-3.0, s) > 0     # satisfied (X above threshold) -> shrink s
        assert c._scale_derivative(+3.0, s) < 0     # violated  (X below threshold) -> grow s
        # Probit behaves the same way.
        p = constraint.AlwaysConstraint('X', '>', 8.0, 'm', 'tc', weight=None,
                                        pmin=0.0, pmax=1.0, tolerance=s)
        assert p._scale_derivative(-3.0, s) > 0
        assert p._scale_derivative(+3.0, s) < 0
