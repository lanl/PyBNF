"""Tests for the PEtab ``observableFormula`` expression layer (#407, ADR-0035/0036).

ADR-0036 makes a PEtab ``observableFormula`` a **measurement model** -- evaluated as a
post-simulation transform over the output trajectory (the observation layer in
:mod:`pybnf.measurement`), never by editing the model file. This supersedes ADR-0035's
``begin functions`` *synthesis into the model*; the reversible translator's surviving
production directions are:

* :func:`~pybnf.petab.formula.bngl_body_to_petab_math` -- the exporter's inlining mode (a BNGL
  function body -> a PEtab math ``observableFormula``), which still generates the round-trip
  oracle; its precedence/spelling logic is graded here.
* :func:`~pybnf.petab.formula.compile_petab_formula` -- the layer's compiler (PEtab math -> a
  numpy callable), graded by numeric hand-computation in ``tests/test_measurement_layer.py``.

The oracles, weakest to strongest:

1. **Export-inline translator unit tests** -- ``bngl_body_to_petab_math`` on crafted bodies:
   the ``^``/``ln``/``log10``/``sqrt`` spellings, the ``sqrt`` precedence defect, the ``func()``
   reference convention, the standing round-trip self-check, and free-symbol validation.
2. **Syntactic round trip (fast tier).** A crafted BNGL model whose measurement model is a
   multi-operator function exports *with inlining* -> imports (to a conf measurement-model
   line, carrying the model verbatim) -> re-exports; the ``observableFormula`` is graded
   sympy-equal across the hop, and the imported job carries a measurement model (not a
   synthesized function).
3. **Semantic round trip (``-m recovery``, bngsim).** The imported job is simulated through the
   real bngsim backend and the measurement layer's computed column reproduces the original
   model function's trace -- catching a self-consistent-but-wrong translation a syntactic
   oracle would miss.

``petab``/``sympy`` is the optional ``pybnf[petab]`` extra; the expression-path tests
``importorskip('petab')``. The petab-absent contract (a pointed "install pybnf[petab]" error,
not an ``ImportError``) is tested *without* skipping -- the bare-name path's dependency-free
guarantee.
"""

import builtins
import sys

import pytest

from pybnf.petab import export_job, import_job
from pybnf.petab._bngl import parse_model
from pybnf.petab.formula import bngl_body_to_petab_math
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
# 1. Export-inline translator (BNGL function body -> PEtab math observableFormula)
# ---------------------------------------------------------------------------

class TestExportInlineTranslator:

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
    def test_inlined_formula_is_equivalent_to_the_body(self, body):
        # The emitted observableFormula denotes the same function as the BNGL body. The
        # forward translator's own _assert_round_trips guards this internally (it re-parses
        # its output); here we confirm it externally against the body too. All these bodies
        # are already valid PEtab math, so _sympy_equal can parse both sides.
        pytest.importorskip('petab')
        out = bngl_body_to_petab_math(body, _entities())
        assert _sympy_equal(out, body)

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

    def test_function_reference_strips_parens_on_the_petab_side(self):
        # A body referencing another global function writes it `f()`; PEtab math has no
        # zero-arg user function, so the inlined formula references it as a bare symbol.
        pytest.importorskip('petab')
        ent = parse_model(
            'begin observables\n Molecules obsA A()\nend observables\n'
            'begin functions\n g() = obsA*2\n h() = g()^2 + obsA\nend functions\n')
        petab_math = bngl_body_to_petab_math('g()^2 + obsA', ent)
        assert 'g(' not in petab_math            # PEtab has no zero-arg user function
        assert _sympy_equal(petab_math, 'g^2 + obsA')

    def test_unknown_symbol_is_an_error_not_a_free_parameter(self):
        pytest.importorskip('petab')
        with pytest.raises(PybnfError, match='not a parameter, observable, or function'):
            bngl_body_to_petab_math('obsA + nope', _entities())

    def test_per_measurement_placeholder_is_deferred(self):
        pytest.importorskip('petab')
        with pytest.raises(NotImplementedError, match='placeholder'):
            bngl_body_to_petab_math('obsA * observableParameter1_x', _entities())

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
            bngl_body_to_petab_math('obsA + 1', _entities())


# ---------------------------------------------------------------------------
# 2. Syntactic round trip (fast tier) -- export -> import -> re-export (ADR-0036)
# ---------------------------------------------------------------------------

def _write_crafted_src(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'crafted.bngl').write_text(CRAFTED_MODEL)
    (src / 'meas.exp').write_text(CRAFTED_EXP)
    (src / 'job.conf').write_text(CRAFTED_CONF)
    return src


def _func_observable_formula(petab_dir):
    """The inlined/measurement function row's observableFormula (the single func_ row)."""
    import csv
    with open(petab_dir / 'observables.tsv') as fh:
        rows = [r for r in csv.DictReader(fh, delimiter='\t')
                if r['observableId'].startswith('func_')]
    assert len(rows) == 1, rows
    return rows[0]['observableFormula']


def _imported_measurement_formula(imported_dir):
    """The ``observable: <id>, formula: <expr>`` measurement-model formula in the conf."""
    from pybnf.parse import ploop
    conf = ploop((imported_dir / 'imported.conf').read_text().splitlines(keepends=True))
    meas = {k[1]: v for k, v in conf.items()
            if isinstance(k, tuple) and k[0] == 'measurement'}
    assert len(meas) == 1, meas
    return next(iter(meas.values()))


class TestSyntacticRoundTrip:

    def test_expression_formula_round_trips_export_import_reexport(self, tmp_path):
        pytest.importorskip('petab')
        src = _write_crafted_src(tmp_path)
        p1, imported, p2 = tmp_path / 'p1', tmp_path / 'imp', tmp_path / 'p2'
        export_job(src / 'job.conf', p1, inline_functions=True)
        import_job(p1 / 'problem.yaml', imported)
        export_job(imported / 'imported.conf', p2, inline_functions=True)

        # The observableFormula survives export -> import(measurement model) -> re-export,
        # equal up to sympy normalization (the importer carries it verbatim; the exporter
        # re-emits it, so this is byte-stable, but graded structurally to be safe).
        assert _sympy_equal(_func_observable_formula(p1), _func_observable_formula(p2))

    def test_import_carries_a_measurement_model_not_a_synthesized_function(self, tmp_path):
        # ADR-0036: no begin-functions synthesis. The imported model is carried verbatim
        # (original `pRel`, no synthesized `func_pRel`), and the measurement model lives in
        # the conf as an `observable: func_pRel, formula:` line whose formula denotes the
        # same function as the original pRel inlined.
        pytest.importorskip('petab')
        src = _write_crafted_src(tmp_path)
        p1, imported = tmp_path / 'p1', tmp_path / 'imp'
        export_job(src / 'job.conf', p1, inline_functions=True)
        import_job(p1 / 'problem.yaml', imported)

        ent = parse_model((imported / 'crafted.bngl').read_text())
        assert 'pRel' in ent.function_bodies          # original model carried verbatim
        assert 'func_pRel' not in ent.function_bodies  # NO synthesis into the model
        orig = bngl_body_to_petab_math(ent.function_bodies['pRel'], ent)
        assert _sympy_equal(orig, _imported_measurement_formula(imported))

    def test_imported_expression_problem_passes_petab_validation(self, tmp_path):
        # The external oracle: the re-exported problem (verbatim model + the measurement
        # model's formula) loads through petab's BnglModel and passes every default
        # validation task (so the emitted problem is genuinely valid PEtab, not merely
        # self-consistent).
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

    def test_measurement_layer_reproduces_the_original_function_trace(self, tmp_path):
        """Simulate the imported job through the real bngsim backend and apply the
        measurement layer: the layer's computed ``func_pRel`` column must match the original
        BNGL function ``pRel`` the verbatim model still carries, cell-for-cell. A
        self-consistent-but-wrong translation (the failure a syntactic oracle misses) would
        diverge here -- and this exercises the real config -> layer wiring, not a stub."""
        pytest.importorskip('petab')
        import os

        import numpy as np

        from pybnf import config as config_mod
        from pybnf.parse import ploop
        from pybnf.pset import PSet
        from .recovery_harness import build, require_bng2pl
        require_bng2pl()

        src = _write_crafted_src(tmp_path)
        p1, imported = tmp_path / 'p1', tmp_path / 'imp'
        export_job(src / 'job.conf', p1, inline_functions=True)
        import_job(p1 / 'problem.yaml', imported)

        # Build a real bngsim config from the imported conf (its `observable: func_pRel,
        # formula:` line builds the measurement layer) and run the verbatim model.
        conf_text = (imported / 'imported.conf').read_text() + '\nbngl_backend = bngsim\n'
        home = os.getcwd()
        os.chdir(imported)
        try:
            conf = config_mod.Configuration(ploop(conf_text.splitlines(keepends=True)))
            assert conf.obj.measurement and len(conf.obj.measurement) == 1
            alg = build(conf, 'de')
            values = {v.name: 2.0 for v in alg.variables}    # nominal kA/kB/kC
            pset = PSet([v.set_value(values[v.name]) for v in alg.variables])
            model = alg.model_list[0].copy_with_param_set(pset)
            os.makedirs(alg.sim_dir, exist_ok=True)
            ds = model.execute(alg.sim_dir, 'meas', 0)
        finally:
            os.chdir(home)

        # Apply the measurement layer post-simulation, exactly as the objective does.
        conf.obj.measurement.apply({model.name: ds}, values)
        data = ds[next(iter(ds))]
        assert 'pRel' in data.cols and 'func_pRel' in data.cols
        np.testing.assert_allclose(data['pRel'], data['func_pRel'], rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# Placeholder substitution (ADR-0044): the constant-per-observable reduction primitive
# ---------------------------------------------------------------------------

class TestPlaceholderSubstitution:

    def test_id_token_becomes_a_free_symbol(self):
        # A constant-per-observable observableParameter is substituted by its parameter id,
        # which stays a free symbol (it resolves from the PSet at eval time, ADR-0044).
        pytest.importorskip('petab')
        from pybnf.petab.formula import substitute_placeholders
        out = substitute_placeholders('observableParameter1_obs * x',
                                      {'observableParameter1_obs': 'scaling'})
        assert 'observableParameter' not in out
        assert _sympy_equal(out, 'scaling * x')

    def test_numeric_token_inlines_as_a_constant(self):
        pytest.importorskip('petab')
        from pybnf.petab.formula import substitute_placeholders
        out = substitute_placeholders('0.1 + 0.05*noiseParameter1_obs',
                                      {'noiseParameter1_obs': '4'})
        assert 'noiseParameter' not in out
        assert _sympy_equal(out, '0.1 + 0.05*4')        # == 0.3

    def test_no_substitution_returns_verbatim_without_petab(self, monkeypatch):
        # An empty substitution map is the bare-name / no-placeholder common case: the formula
        # is returned byte-verbatim and petab is never imported (the dependency-free guarantee).
        import builtins
        from pybnf.petab.formula import substitute_placeholders
        real_import = builtins.__import__

        def no_petab(name, *a, **k):
            if name.startswith('petab'):
                raise AssertionError('petab must not be imported for an empty substitution')
            return real_import(name, *a, **k)
        monkeypatch.setattr(builtins, '__import__', no_petab)
        assert substitute_placeholders('x', {}) == 'x'

    def test_unmatched_placeholder_is_left_in_place(self):
        # A placeholder not in the map is left untouched (the caller validates it downstream:
        # the importer raises the deferred-frontier error on a surviving placeholder).
        pytest.importorskip('petab')
        from pybnf.petab.formula import substitute_placeholders
        out = substitute_placeholders('observableParameter1_obs * observableParameter2_obs',
                                      {'observableParameter1_obs': 'scaling'})
        assert 'observableParameter2_obs' in out

    def test_formula_free_symbols_lists_sorted_names(self):
        pytest.importorskip('petab')
        from pybnf.petab.formula import formula_free_symbols
        assert formula_free_symbols('0.1 + 0.05*slope + base') == ['base', 'slope']
        assert formula_free_symbols('0.5') == []        # a pure constant has no free symbols
