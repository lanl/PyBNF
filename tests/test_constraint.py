from .context import constraint


class TestConstraint:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        cls.cset = constraint.ConstraintSet()

        cls.line1 = 'A<B at 6 weight 1'
        cls.line2 = '42>=A At B=17 Weight 5 AltPenalty C<=42 Min 3.4e-7#Comment  '
        cls.line3 = 'A<16 between B=3.14, 702 weight 8 min 9'
        cls.line4 = 'A<16 always weight 6'
        cls.line5 = 'A<16 once weight 6'

    @classmethod
    def teardown_class(cls):
        pass

    def test_grammar_at(self):
        p = self.cset.parse_constraint_line(self.line1)
        assert list(p.ineq) == ['A','<','B']
        assert list(p.enforce) == ['at', '6']
        assert list(p.weight_expr) == ['weight', '1']
        assert float(p.weight_expr.weight) == 1.
        assert not bool(p.weight_expr.min)

        p = self.cset.parse_constraint_line(self.line2)
        assert list(p.ineq) == ['42', '>=', 'A']
        assert list(p.enforce) == ['at', 'B', '17']
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