from .context import constraint, data
import copy
import numpy as np
from nose.tools import raises
from pyparsing import ParseBaseException


class TestConstraint:
    def __init__(self):
        pass

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

        np.testing.assert_almost_equal(cs.total_penalty(d_dict), 95.3)

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

        np.testing.assert_almost_equal(cs.total_penalty(d_dict), 190.6)

    def test_likelihood_penalties(self):
        d = data.Data()
        d.load_data('bngl_files/con_test.gdat')
        d_dict = {self.model: {self.suf: d}}

        cs = copy.deepcopy(self.cset)
        cs.load_constraint_file(self.f3)

        assert cs.constraints[0].penalty(d_dict) == 0
        np.testing.assert_almost_equal(cs.constraints[1].penalty(d_dict), -np.log(0.95))
        np.testing.assert_almost_equal(cs.constraints[2].penalty(d_dict), -np.log(0.05))
        np.testing.assert_almost_equal(cs.constraints[3].penalty(d_dict), -np.log(0.5))
        np.testing.assert_almost_equal(cs.constraints[4].penalty(d_dict), -np.log(0.9/(1+np.exp(1)) + 0.05))
        np.testing.assert_almost_equal(cs.constraints[5].penalty(d_dict), -np.log(0.9 / (1 + np.exp(-0.5)) + 0.05))
        np.testing.assert_almost_equal(cs.constraints[6].penalty(d_dict), -np.log(1 / (1 + np.exp(1))))
