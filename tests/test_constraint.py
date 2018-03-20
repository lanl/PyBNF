from .context import constraint, data
import copy
import numpy as np


class TestConstraint:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):

        cls.line1 = 'A<B at 6 weight 1'
        cls.line2 = '42>=A At B=17 everytime Weight 5 AltPenalty C<=42 Min 3.4e-7#Comment  '
        cls.line3 = 'A<16 between B=3.14, 702 weight 8 min 9'
        cls.line4 = 'A<16 always weight 6'
        cls.line5 = 'A<16 once weight 6'

        cls.f1 = 'bngl_files/p1_5.con'
        cls.f2 = 'bngl_files/con_test.con'
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

    def test_load_file(self):
        cs = copy.deepcopy(self.cset)
        cs._load_constraint_file(self.f1)

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

    def test_penalties(self):
        d = data.Data()
        d.load_data('bngl_files/con_test.gdat')
        d_dict = {self.model: {self.suf: d}}

        cs = copy.deepcopy(self.cset)
        cs._load_constraint_file(self.f2)

        print('start?')
        assert cs.constraints[0].penalty(d_dict) == 0
        print(':D 1')
        assert cs.constraints[1].penalty(d_dict) == 4
        print(':D 2')
        np.testing.assert_almost_equal(cs.constraints[2].penalty(d_dict), 0.4)
        print(':D 3')
        np.testing.assert_almost_equal(cs.constraints[3].penalty(d_dict), 0.4)
        print(':D 4')
        assert cs.constraints[4].penalty(d_dict) == 0
        print(':D 5')
        assert cs.constraints[5].penalty(d_dict) == 10
        print(':D 6')
        assert cs.constraints[6].penalty(d_dict) == 0
        print(':D 7')
        assert cs.constraints[7].penalty(d_dict) == 25
        print(':D 8')
        assert cs.constraints[8].penalty(d_dict) == 20
        print(':D 9')
        print(cs.constraints[9].penalty(d_dict))
        assert cs.constraints[9].penalty(d_dict) == 20
        print(':D 10')

        np.testing.assert_almost_equal(cs.total_penalty(d_dict), 79.8)
