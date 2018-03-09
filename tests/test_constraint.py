from .context import constraint


class TestConstraint:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        cls.cset = constraint.ConstraintSet()

    @classmethod
    def teardown_class(cls):
        pass

    def test_grammar(self):
        p = self.cset.parse_constraint_line('A<B at 6 weight 1')
        assert list(p.ineq) == ['A','<','B']
        assert list(p.enforce) == ['at', '6']
        assert list(p.weight) == ['weight', '1']