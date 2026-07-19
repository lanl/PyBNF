import pytest

from .context import pset
from .context import printing
from .context import raises


class TestMutationParamRef:
    """A parameter-reference condition perturbation (a per-condition estimated initial
    condition, ADR-0076): the value is a free-parameter id resolved from the PSet at apply
    time, not a fixed number."""

    def test_amount_resolves_free_parameter_reference(self):
        m = pset.Mutation('I0_', '=', 'I0_CA', is_param_ref=True)
        assert m.amount({'I0_CA': 232.9}) == 232.9

    def test_amount_missing_reference_raises(self):
        m = pset.Mutation('I0_', '=', 'I0_CA', is_param_ref=True)
        with pytest.raises(printing.PybnfError, match='I0_CA'):
            m.amount({'other': 1.0})

    def test_mutate_absolute_returns_referenced_value(self):
        m = pset.Mutation('I0_', '=', 'I0_CA', is_param_ref=True)
        assert m.mutate(999.0, {'I0_CA': 232.9}) == 232.9

    def test_mutate_relative_uses_referenced_value(self):
        # A relative op composes with the referenced value (k *= scale), not a fixed number.
        m = pset.Mutation('k', '*', 'scale', is_param_ref=True)
        assert m.mutate(4.0, {'scale': 2.5}) == 10.0

    def test_numeric_mutation_ignores_param_values(self):
        m = pset.Mutation('k', '=', 5.0)
        assert m.amount() == 5.0
        assert m.mutate(1.0) == 5.0


class TestPSet:
    @classmethod
    def setup_class(cls):
        """Define constants to be used in tests"""
        cls.p0 = pset.FreeParameter('var0__FREE', 'normal_var', 0, 1, value=1.0)
        cls.p1 = pset.FreeParameter('var1__FREE', 'lognormal_var', 1, 2, value=0.1)
        cls.p2 = pset.FreeParameter('var2__FREE', 'loguniform_var', 0.01, 100, value=99.0)
        cls.p3 = pset.FreeParameter('var2__FREE', 'uniform_var', 0, 10, value=5.0)
        cls.p4 = pset.FreeParameter('var2__FREE', 'uniform_var', 0, 10, bounded=False)

        cls.fps0 = [cls.p0, cls.p1, cls.p2]
        cls.fps1 = [cls.p3, cls.p4, cls.p2]
        cls.fps2 = [cls.p3, cls.p0, cls.p1]

    def test_initialization(self):
        ps1 = pset.PSet(self.fps0)
        assert ps1['var0__FREE'] == 1.0
        assert ps1['var1__FREE'] == 0.1
        assert ps1['var2__FREE'] == 99.0

    def test_iteration(self):
        ps1 = pset.PSet(self.fps0)
        for p in ps1:
            assert p in ps1.fps

    def test_iteration_reentrant(self):
        # CQ-7: __iter__ must return a fresh iterator, not store a cursor on
        # self -- nested iteration over one PSet must not clobber the outer
        # loop. The old self.idx implementation produced the wrong cross-product.
        ps1 = pset.PSet(self.fps0)
        pairs = [(a, b) for a in ps1 for b in ps1]
        assert len(pairs) == len(self.fps0) ** 2
        # First element of the outer loop must still be paired with every inner
        # element (the buggy shared-cursor version exhausts on the first pass).
        first_outer = self.fps0[0]
        assert [b for (a, b) in pairs if a is first_outer] == self.fps0

    def test_get_freeparameter(self):
        p1 = pset.PSet(self.fps0)
        assert p1.get_param('var0__FREE') == self.p0

    @raises(printing.PybnfError)
    def test_faulty_initialization(self):
        ps2 = pset.PSet(self.fps1)

    def test_keys_to_string(self):
        ps1 = pset.PSet(self.fps0)
        assert ps1.keys_to_string() == 'var0__FREE\tvar1__FREE\tvar2__FREE'

    def test_values_to_string(self):
        ps1 = pset.PSet(self.fps0)
        assert ps1.values_to_string() == '1.0\t0.1\t99.0'

    def test_get_id(self):
        ps1a = pset.PSet(self.fps0)
        ps1b = pset.PSet(self.fps0)
        ps2 = pset.PSet(self.fps2)
        assert ps1a.get_id() == ps1b.get_id()
        assert ps1a.get_id() != ps2.get_id()
        assert ps2.get_id() != ps1b.get_id()

    @raises(TypeError)
    def test_immutable(self):
        ps1 = pset.PSet(self.fps0)
        ps1['var0__FREE'] = 1.5


class TestActionStepGuard:
    """step=0 must raise a clean PybnfError, not an unguarded ZeroDivisionError
    from the `stepnumber` divide (time/step and (max-min)/step)."""

    @raises(printing.PybnfError)
    def test_time_course_zero_step_raises_pybnf_error(self):
        pset.TimeCourse({'time': '10', 'step': '0'})

    @raises(printing.PybnfError)
    def test_param_scan_zero_step_raises_pybnf_error(self):
        pset.ParamScan({'min': '0', 'max': '10', 'step': '0', 'time': '10', 'param': 'k'})

    def test_nonzero_step_still_works(self):
        # Guard doesn't disturb the normal path.
        assert pset.TimeCourse({'time': '10', 'step': '2'}).stepnumber == 5
        assert pset.ParamScan({'min': '0', 'max': '10', 'step': '2',
                               'time': '10', 'param': 'k'}).stepnumber == 5
