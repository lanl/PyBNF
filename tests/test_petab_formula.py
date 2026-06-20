"""Tests for the PEtab ``observableFormula`` expression layer (#407, ADR-0035).

The expression layer is graded by three oracles, weakest to strongest:

1. **Translator unit tests** -- the reversible ``formula.py`` pair on crafted
   expressions: a BNGL body -> PEtab math -> BNGL body round trip (sympy-normalized),
   the ``func()`` reference convention, the ``^``/``ln``/``log10``/``sqrt`` spellings,
   and free-symbol validation.
2. **Syntactic round trip (fast tier).** A crafted BNGL model whose measurement model
   is a multi-operator function (a quotient of sums like Boehm's) exports *with
   inlining* -> imports (synthesizing the function back) -> re-exports with inlining;
   the ``observableFormula`` and the synthesized body are graded sympy-equal. This is
   the exporter-first oracle ADR-0035 unlocks (the exporter emits its own expression).
3. **Semantic round trip (``-m recovery``, bngsim).** The import-synthesized model is
   simulated through the real bngsim backend and the synthesized measurement function
   reproduces the original one's trace -- catching a self-consistent-but-wrong
   translator pair a purely syntactic oracle would miss.

``petab``/``sympy`` is the optional ``pybnf[petab]`` extra; the expression-path tests
``importorskip('petab')``. The petab-absent contract (a pointed "install pybnf[petab]"
error, not an ``ImportError``) is tested *without* skipping -- it is the bare-name path's
dependency-free guarantee.
"""

import builtins
import sys
from pathlib import Path

import pytest

from pybnf.petab import export_job, import_job
from pybnf.petab._bngl import parse_model
from pybnf.petab.formula import (
    bngl_body_to_petab_math,
    petab_math_to_bngl_body,
)
from pybnf.printing import PybnfError

# A crafted BNGL model whose measurement model is a Boehm-style quotient of sums over
# observables (obsA/obsB) and parameters (kA/kB/kC) -- a multi-operator expression with
# no SBML required. Actionless (new-era, ADR-0028): export reads the experiment surface
# and the recovery sim synthesizes the action from the data's time column.
CRAFTED_MODEL = """\
begin model
  begin parameters
    kA 2
    kB 3
    kC 0.5
  end parameters
  begin molecule types
    A()
    B()
  end molecule types
  begin seed species
    A() 10
    B() 4
  end seed species
  begin observables
    Molecules obsA A()
    Molecules obsB B()
  end observables
  begin functions
    pRel() = (100*obsA + 200*obsB*kA)/(obsB + kB*obsA + 2*kC*obsB)
  end functions
  begin reaction rules
    A() -> B() kA
  end reaction rules
end model
"""

CRAFTED_EXP = '# time\tpRel\n0\t5\n1\t6\n2\t7\n'

CRAFTED_CONF = (
    'edition = 2\njob_type = de\nobjective = sos\nmodel: crafted.bngl\n'
    'experiment: meas, data: meas.exp\n'
    'uniform_var = kA 0 10\nuniform_var = kB 0 10\nuniform_var = kC 0 10\n')


def _entities(model_text=CRAFTED_MODEL):
    return parse_model(model_text)


def _sympy_equal(petab_expr_a, petab_expr_b):
    """True iff two PEtab math strings denote the same function.

    By numeric sampling at distinct positive points, not symbolic ``simplify``: petab
    floatifies literals (a ``sqrt`` parses back with a ``1.0/2.0`` Float exponent, not an
    exact ``Rational(1/2)``), and sympy treats Float-vs-exact powers as unequal under
    ``simplify`` -- so a symbolic test false-rejects a correct ``sqrt`` translation.
    Positive points keep ``sqrt``/``log`` real; multiple points rule out coincidental
    agreement (the corrupt ``z/2`` and ``sqrt(z)`` collide only at ``z=4``).
    """
    import sympy as sp
    from petab.v2.math import sympify_petab
    ea = sympify_petab(petab_expr_a, evaluate=False)
    eb = sympify_petab(petab_expr_b, evaluate=False)
    syms = sorted(ea.free_symbols | eb.free_symbols, key=str)
    for k in range(1, 6):
        subs = {s: sp.Rational(3 + 2 * k + 5 * i, 7) for i, s in enumerate(syms)}
        va, vb = float(sp.N(ea.subs(subs))), float(sp.N(eb.subs(subs)))
        if abs(va - vb) > 1e-7 * max(1.0, abs(vb)):
            return False
    return True


# ---------------------------------------------------------------------------
# 1. Translator unit tests (the reversible pair)
# ---------------------------------------------------------------------------

class TestTranslatorPair:

    @pytest.mark.parametrize('body', [
        'kA*obsA + kB',
        '(100*obsA + 200*obsB*kA)/(obsB + kB*obsA + 2*kC*obsB)',   # quotient of sums
        'obsA^2 + 2*obsA - kC',
        'kA*(obsA - obsB)/(kB + kC)',
        'sqrt(obsA)',                                              # the petab ^1/2 defect
        'kA*sqrt(obsA) + kB',
        '(obsA + obsB)/sqrt(kC)',
        'sqrt(kA*obsA + kB)',
    ])
    def test_bngl_body_round_trips_through_petab_math(self, body):
        pytest.importorskip('petab')
        ent = _entities()
        petab_math = bngl_body_to_petab_math(body, ent)
        back = petab_math_to_bngl_body(petab_math, ent)
        # The pair is mutually inverse up to sympy normalization (not bytes): re-translate
        # the round-tripped body and assert all three denote the same function.
        again = bngl_body_to_petab_math(back, ent)
        assert _sympy_equal(petab_math, again)

    def test_sqrt_serializes_precedence_safe_not_the_petab_defect(self):
        # Guards the petab 0.8.x petab_math_str defect (ADR-0035): a sqrt must NOT export as
        # the unparenthesized `x ^ 1/2` (which re-parses as x/2 and silently corrupts the
        # measurement model). Our printer parenthesizes the half-power; the emitted string
        # must denote sqrt, not x/2, and be valid PEtab math the validator re-parses.
        pytest.importorskip('petab')
        out = bngl_body_to_petab_math('sqrt(obsA)', _entities())
        assert ' ^ 1/2' not in out                    # not the defective form
        assert _sympy_equal(out, 'sqrt(obsA)')         # means sqrt...
        assert not _sympy_equal(out, 'obsA/2')         # ...not the corruption

    def test_round_trip_guard_refuses_a_corrupt_serialization(self, monkeypatch):
        # The standing tripwire: if the PEtab serializer ever emits a string that does not
        # parse back to the same function, refuse it loudly rather than corrupt silently.
        pytest.importorskip('petab')
        import pybnf.petab.formula as F

        class _Defective:                              # emits petab's buggy `x ^ 1/2`
            def doprint(self, expr):
                return 'obsA ^ 1/2'

        monkeypatch.setattr(F, '_petab_printer_cls', lambda: (lambda: _Defective()))
        with pytest.raises(PybnfError, match='silently corrupt'):
            F.bngl_body_to_petab_math('sqrt(obsA)', _entities())

    def test_function_reference_uses_bngl_paren_convention(self):
        # A body referencing another global function writes it `f()`; the PEtab side is a
        # bare symbol; the BNGL side re-appends the parens. (pRel is a function here.)
        pytest.importorskip('petab')
        ent = parse_model(
            'begin observables\n Molecules obsA A()\nend observables\n'
            'begin functions\n g() = obsA*2\n h() = g()^2 + obsA\nend functions\n')
        petab_math = bngl_body_to_petab_math('g()^2 + obsA', ent)
        assert 'g(' not in petab_math            # PEtab has no zero-arg user function
        back = petab_math_to_bngl_body(petab_math, ent)
        assert 'g()' in back                     # BNGL references the function with parens

    @pytest.mark.parametrize('petab_in,needle', [
        ('sqrt(obsA)', 'sqrt(obsA)'),
        ('ln(obsA)', 'ln(obsA)'),
        ('log10(obsA)', 'log10(obsA)'),
        ('obsA^2', 'obsA^2'),
    ])
    def test_bngl_spellings(self, petab_in, needle):
        pytest.importorskip('petab')
        assert needle in petab_math_to_bngl_body(petab_in, _entities())

    def test_unknown_symbol_is_an_error_not_a_free_parameter(self):
        pytest.importorskip('petab')
        with pytest.raises(PybnfError, match='not a parameter, observable, or function'):
            petab_math_to_bngl_body('obsA + nope', _entities())

    def test_per_measurement_placeholder_is_deferred(self):
        pytest.importorskip('petab')
        with pytest.raises(NotImplementedError, match='placeholder'):
            petab_math_to_bngl_body('obsA * observableParameter1_x', _entities())

    def test_expression_without_petab_raises_pointed_error(self, monkeypatch):
        # The dependency-free guarantee: with petab absent the expression path raises a
        # pointed PybnfError naming the extra, NOT a bare ImportError from the call stack.
        # (No importorskip -- this is exactly the petab-absent contract.)
        real_import = builtins.__import__

        def _block(name, *args, **kwargs):
            if name == 'petab' or name.startswith('petab.'):
                raise ImportError('petab blocked for this test')
            return real_import(name, *args, **kwargs)

        for mod in [m for m in sys.modules if m.startswith('petab')]:
            monkeypatch.delitem(sys.modules, mod, raising=False)
        monkeypatch.setattr(builtins, '__import__', _block)
        with pytest.raises(PybnfError, match=r'pybnf\[petab\]'):
            petab_math_to_bngl_body('obsA + 1', _entities())


# ---------------------------------------------------------------------------
# 2. Syntactic round trip (fast tier) -- the exporter-first oracle (ADR-0035)
# ---------------------------------------------------------------------------

def _write_crafted_src(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'crafted.bngl').write_text(CRAFTED_MODEL)
    (src / 'meas.exp').write_text(CRAFTED_EXP)
    (src / 'job.conf').write_text(CRAFTED_CONF)
    return src


def _func_observable_formula(petab_dir):
    """The synthesized/inlined function row's observableFormula (the single func_ row)."""
    import csv
    with open(petab_dir / 'observables.tsv') as fh:
        rows = [r for r in csv.DictReader(fh, delimiter='\t')
                if r['observableId'].startswith('func_')]
    assert len(rows) == 1, rows
    return rows[0]['observableFormula']


class TestSyntacticRoundTrip:

    def test_expression_formula_round_trips_export_import_reexport(self, tmp_path):
        pytest.importorskip('petab')
        src = _write_crafted_src(tmp_path)
        p1, imported, p2 = tmp_path / 'p1', tmp_path / 'imp', tmp_path / 'p2'
        export_job(src / 'job.conf', p1, inline_functions=True)
        import_job(p1 / 'problem.yaml', imported)
        export_job(imported / 'imported.conf', p2, inline_functions=True)

        # The observableFormula survives export -> import(synthesize) -> re-export, equal
        # up to sympy normalization (the translators are mutually inverse, not byte-exact).
        assert _sympy_equal(_func_observable_formula(p1), _func_observable_formula(p2))

    def test_synthesized_body_matches_the_original_function(self, tmp_path):
        pytest.importorskip('petab')
        src = _write_crafted_src(tmp_path)
        p1, imported = tmp_path / 'p1', tmp_path / 'imp'
        export_job(src / 'job.conf', p1, inline_functions=True)
        import_job(p1 / 'problem.yaml', imported)

        # The imported model carries the original `pRel` and the synthesized `func_pRel`;
        # both denote the same measurement model (sympy-normalized over the petab grammar).
        ent = parse_model((imported / 'crafted.bngl').read_text())
        assert 'pRel' in ent.function_bodies and 'func_pRel' in ent.function_bodies
        orig = bngl_body_to_petab_math(ent.function_bodies['pRel'], ent)
        synth = bngl_body_to_petab_math(ent.function_bodies['func_pRel'], ent)
        assert _sympy_equal(orig, synth)

    def test_imported_expression_problem_passes_petab_validation(self, tmp_path):
        # The external oracle: the synthesized model + inlined formula load through petab's
        # BnglModel and pass every default validation task (so the emitted problem is
        # genuinely valid PEtab, not merely self-consistent).
        pytest.importorskip('petab.v2')
        from petab.v2 import Problem
        from petab.v2.lint import ValidationIssueSeverity, default_validation_tasks

        from pybnf.petab.bngl_model import register_bngl
        register_bngl()
        src = _write_crafted_src(tmp_path)
        p1, imported, p2 = tmp_path / 'p1', tmp_path / 'imp', tmp_path / 'p2'
        export_job(src / 'job.conf', p1, inline_functions=True)
        import_job(p1 / 'problem.yaml', imported)
        export_job(imported / 'imported.conf', p2, inline_functions=True)

        problem = Problem.from_yaml(str(p2 / 'problem.yaml'))
        assert type(problem.model).__name__ == 'BnglModel'
        errors = [(type(t).__name__, t.run(problem).message)
                  for t in default_validation_tasks
                  if t.run(problem) is not None
                  and getattr(t.run(problem), 'level', None) ==
                  ValidationIssueSeverity.ERROR]
        assert errors == []


# ---------------------------------------------------------------------------
# 3. Semantic round trip (-m recovery, bngsim) -- the strongest oracle
# ---------------------------------------------------------------------------

@pytest.mark.recovery
@pytest.mark.bngsim
class TestSemanticRoundTrip:

    def test_synthesized_function_reproduces_the_original_trace(self, tmp_path):
        """Simulate the import-synthesized model: the synthesized `func_pRel` must match
        the original `pRel` trace cell-for-cell. A self-consistent-but-wrong translator
        pair (the failure a syntactic oracle misses) would diverge here."""
        pytest.importorskip('petab')
        import numpy as np

        from .recovery_harness import build, make_newera_config, require_bng2pl
        from pybnf.pset import PSet
        require_bng2pl()

        # Build the import-synthesized model (original pRel + synthesized func_pRel).
        src = _write_crafted_src(tmp_path)
        p1, imported = tmp_path / 'p1', tmp_path / 'imp'
        export_job(src / 'job.conf', p1, inline_functions=True)
        import_job(p1 / 'problem.yaml', imported)
        model_path = imported / 'crafted.bngl'
        exp_path = next(imported.glob('*.exp'))

        # Simulate it once (network gen via BNG2.pl, then in-process bngsim ODE) at the
        # model's nominal kA; print_functions is forced, so both functions are output.
        conf = make_newera_config(
            tmp_path / 'sim', str(model_path), str(exp_path),
            {'kA': ('uniform_var', 0, 10)}, 'meas', 'de',
            population_size=4, max_iterations=1)
        alg = build(conf, 'de')
        pset = PSet([v.set_value(2.0) for v in alg.variables])   # kA at its nominal
        model = alg.model_list[0].copy_with_param_set(pset)
        folder = str(tmp_path / 'run')
        Path(folder).mkdir(parents=True, exist_ok=True)
        home = Path.cwd()
        try:
            ds = model.execute(folder, 'meas', 0)
        finally:
            import os
            os.chdir(home)
        data = ds[next(iter(ds))]

        assert 'pRel' in data.cols and 'func_pRel' in data.cols
        np.testing.assert_allclose(data['pRel'], data['func_pRel'], rtol=1e-9, atol=1e-12)
