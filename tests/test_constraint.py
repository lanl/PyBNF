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
        print(self.cset.parse_constraint_line('A<B at 6 weight 1'))
