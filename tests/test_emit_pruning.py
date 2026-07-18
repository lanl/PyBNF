"""Off-diagonal ``{action} × {condition}`` cross-product pruning (lanl/PyBNF#484, ADR-0069).

Under edition-2 Mechanism A (ONE model + ``condition:`` perturbations) a model runs every
synthesized action under every condition mutant, but only the scored ``(action, its own
condition)`` diagonal is consumed. PyBNF records that diagonal (plus constraint/postprocessing
consumers) as the model's ``emit_suffixes`` and the backend ``execute`` skips every other pair,
so N experiments × M conditions cost N simulations instead of N×(M+1).

Two layers:

* pure-logic + config tests (no simulator) pin the emit-set computation, its enable gate, and
  the ``Model._emit_skip`` decision -- including the pre-equilibration-phase exemption;
* a bngsim execute oracle proves the cross-product is actually pruned (a simulate-count and the
  emitted keys), that the scored objective is invariant to pruning, and that ``emit_suffixes =
  None`` is byte-identical (the full cross-product still runs).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .context import config, parse
from pybnf.pset import Model
from pybnf.printing import PybnfError
from . import recovery_harness as H


# --------------------------------------------------------------------------- #
# A tiny irreversible model whose one rate a condition can knock out, and a
# two-experiment (wildtype + conditioned) config over it -- the minimal genuine
# {action} × {condition} cross-product. No `begin actions` block (edition-2).
# --------------------------------------------------------------------------- #
_MODEL_BNGL = """\
begin model
begin parameters
  k  1.0
end parameters
begin molecule types
  A()
  B()
end molecule types
begin seed species
  A()  10
  B()  0
end seed species
begin observables
  Molecules  Obs_B  B()
end observables
begin reaction rules
  R: A() -> B()  k
end reaction rules
end model
"""

_EXP = "# time\tObs_B\n0\t0\n1\t3\n2\t5\n"


def _write_fixture(tmp_path):
    """Write the model + two .exp files into ``tmp_path`` and return the model path."""
    model = tmp_path / 'convert.bngl'
    model.write_text(_MODEL_BNGL)
    (tmp_path / 'wt.exp').write_text(_EXP)
    (tmp_path / 'mut.exp').write_text(_EXP)
    return model


def _conf(tmp_path, body_lines):
    """Build a real edition-2 ``Configuration`` from the shared scalars + ``body_lines``
    (the model/condition/experiment/var lines), through the real parser."""
    scalars = [
        'edition = 2',
        'bngl_backend = bngsim',
        'job_type = de',
        'objective = sos',
        f'output_dir = {tmp_path / "out"}',
        'population_size = 4',
        'max_iterations = 1',
        'verbosity = 0',
        'wall_time_sim = 0',
        'random_seed = 1234',
    ]
    text = '\n'.join(scalars + body_lines) + '\n'
    return config.Configuration(parse.ploop(text.splitlines(keepends=True)))


def _two_experiment_conf(tmp_path, *, extra=()):
    """A wildtype experiment (no condition) + a knockout experiment under condition ``c``
    (``k = 0``), plus any ``extra`` config lines. Diagonal = {'wt', 'mutc'}."""
    model = _write_fixture(tmp_path)
    body = [
        f'model: {model}',
        'condition: c, perturbations: k = 0',
        f'experiment: wt, data: {tmp_path / "wt.exp"}',
        f'experiment: mut, condition: c, data: {tmp_path / "mut.exp"}',
        'uniform_var = k 0.1 10',
        *extra,
    ]
    return _conf(tmp_path, body)


# --------------------------------------------------------------------------- #
# Model._emit_skip -- the per-(action, condition) gate
# --------------------------------------------------------------------------- #
class TestEmitSkip:
    """``_emit_skip(action_suffix)`` decides whether one ``(action, condition)`` pair is
    off every consumer's emit-set. The condition context comes from
    ``_emit_context_suffix`` ('' for the base wildtype run, the mutant's suffix on its
    copy), folded onto the action's own suffix."""

    @staticmethod
    def _model(emit, *, suffixes=(('ode', 'A'), ('ode', 'B')), ctx=''):
        m = Model()
        m.emit_suffixes = emit
        m.suffixes = list(suffixes)
        m._emit_context_suffix = ctx
        return m

    def test_none_never_skips(self):
        """The default (pruning off) never skips -- byte-identical legacy behavior."""
        assert self._model(None)._emit_skip('A') is False
        assert self._model(None)._emit_skip('B') is False

    @pytest.mark.parametrize('ctx, suffix, expected', [
        ('', 'A', False),   # base run, A is the scored wildtype diagonal -> keep
        ('', 'B', True),    # base run, B belongs to condition c -> off-diagonal, skip
        ('c', 'B', False),  # c mutant, B->Bc is the scored diagonal -> keep
        ('c', 'A', True),   # c mutant, A->Ac is off-diagonal -> skip
    ])
    def test_diagonal_only_kept(self, ctx, suffix, expected):
        m = self._model({'A', 'Bc'}, ctx=ctx)
        assert m._emit_skip(suffix) is expected

    def test_unregistered_preequilibration_phase_never_skipped(self):
        """An intermediate pre-equilibration phase emits its own simulate under an
        unregistered ``<name>_preequil`` suffix; it carries state into the measured phase
        and must always run, even though it is not in the emit-set."""
        m = self._model({'A'}, suffixes=(('ode', 'A'),))
        assert m._emit_skip('A_preequil') is False  # not a registered suffix -> keep
        assert m._emit_skip('A') is False            # registered + in emit -> keep


# --------------------------------------------------------------------------- #
# Configuration._compute_emit_suffixes -- what the emit-set contains and its gate
# --------------------------------------------------------------------------- #
class TestComputeEmitSuffixes:

    def test_emit_set_is_the_scored_diagonal(self, tmp_path):
        """Two experiments (wildtype + conditioned) over one model + one condition:
        the emit-set is exactly the scored diagonal, NOT the full cross-product."""
        cfg = _two_experiment_conf(tmp_path)
        model = cfg.models['convert']
        assert set(model.get_suffixes()) == {'wt', 'wtc', 'mut', 'mutc'}   # full cross-product
        assert cfg.emit_suffixes['convert'] == {'wt', 'mutc'}              # scored diagonal only

    def test_constraint_home_suffix_is_kept(self, tmp_path):
        """A constraint-only experiment's data-key is not in exp_data but IS a constraint
        home; it must be in the emit-set (its simulation feeds the constraint)."""
        model = _write_fixture(tmp_path)
        (tmp_path / 'facts.prop').write_text('Obs_B at time=2 > Obs_B at time=1 weight 1\n')
        cfg = _conf(tmp_path, [
            f'model: {model}',
            f'experiment: wt, data: {tmp_path / "wt.exp"}',
            f'experiment: check, data: {tmp_path / "facts.prop"}, t_end: 2',
            'uniform_var = k 0.1 10',
        ])
        assert cfg.emit_suffixes['convert'] == {'wt', 'check'}

    def test_pre_edition2_disables_pruning(self, tmp_path):
        """The emit-set gate requires edition >= 2 (``condition:`` is edition-2 only). Below
        that, pruning is off entirely -- byte-identical legacy behavior."""
        cfg = _two_experiment_conf(tmp_path)
        assert cfg.emit_suffixes  # edition-2 -> populated
        cfg.config['edition'] = 1
        cfg._compute_emit_suffixes()
        assert cfg.emit_suffixes == {}

    def test_separability_gate_disables_a_mixed_actions_model(self, tmp_path):
        """A model whose action suffixes are not exactly its experiment names (e.g. a
        hand-written begin-actions block mixed in) is not cleanly separable, so pruning
        stays off for it (fail-safe)."""
        cfg = _two_experiment_conf(tmp_path)
        model = cfg.models['convert']
        # Simulate a stray non-experiment action suffix and recompute: the gate must bail.
        model.suffixes.append(('ode', 'handwritten'))
        cfg._compute_emit_suffixes()
        assert 'convert' not in cfg.emit_suffixes

    def test_nonproducible_consumer_ref_raises(self, tmp_path):
        """A consumer (here a postprocessing target) that references a suffix no
        action × condition pair produces is a load-time error, not a silent drop."""
        cfg = _two_experiment_conf(tmp_path)
        cfg.postprocessing[('convert', 'ghost')] = 'noop.py'
        with pytest.raises(PybnfError, match='ghost'):
            cfg._compute_emit_suffixes()


# --------------------------------------------------------------------------- #
# Execute-level oracle (real bngsim): the cross-product is actually pruned
# --------------------------------------------------------------------------- #
@pytest.mark.bngsim
class TestExecutePruning:
    """Build the runtime bngsim model from the two-experiment config and execute it,
    spying the net backend's simulate entry point."""

    @staticmethod
    def _build_model(tmp_path):
        cfg = _two_experiment_conf(tmp_path)
        alg = H.build(cfg, 'de')
        return cfg, alg, alg.model_list[0]

    @staticmethod
    def _spy_simulate(monkeypatch):
        """Record ``<action suffix><condition suffix>`` for every net simulate that actually
        runs. ``_prepare_simulate_run`` is called once per simulate *after* the emit-set skip
        guard (a skipped action ``continue``s before it), and it carries both the parsed
        suffix and the mutant's ``_emit_context_suffix`` -- so its call list is exactly the
        simulations performed."""
        import pybnf.bngsim_model.net_model as nm
        seen = []
        orig = nm.BngsimModel._prepare_simulate_run

        def spy(self, state, sim_params, *args, **kw):
            seen.append(sim_params.get('suffix', 'time_course') + self._emit_context_suffix)
            return orig(self, state, sim_params, *args, **kw)

        monkeypatch.setattr(nm.BngsimModel, '_prepare_simulate_run', spy)
        return seen

    def test_only_the_diagonal_is_simulated(self, tmp_path, monkeypatch):
        _cfg, alg, model = self._build_model(tmp_path)
        seen = self._spy_simulate(monkeypatch)
        from pybnf.pset import PSet
        ps = PSet([v.set_value(1.0) for v in alg.variables])
        out = str(tmp_path / 'run')
        Path(out).mkdir(exist_ok=True)
        home = Path.cwd()
        try:
            ds = model.copy_with_param_set(ps).execute(out, 'run', 60)
        finally:
            import os
            os.chdir(home)
        assert sorted(seen) == ['mutc', 'wt']            # exactly the 2 scored pairs
        assert set(ds.keys()) == {'wt', 'mutc'}          # only the diagonal emitted

    def test_pruning_off_runs_the_full_cross_product(self, tmp_path, monkeypatch):
        """With ``emit_suffixes = None`` (the legacy default) the full N×(M+1) cross-product
        still runs -- pruning is a strict, opt-in reduction, byte-identical when off."""
        _cfg, alg, model = self._build_model(tmp_path)
        model.emit_suffixes = None
        seen = self._spy_simulate(monkeypatch)
        from pybnf.pset import PSet
        ps = PSet([v.set_value(1.0) for v in alg.variables])
        out = str(tmp_path / 'run_full')
        Path(out).mkdir(exist_ok=True)
        home = Path.cwd()
        try:
            ds = model.copy_with_param_set(ps).execute(out, 'run', 60)
        finally:
            import os
            os.chdir(home)
        assert sorted(seen) == ['mut', 'mutc', 'wt', 'wtc']       # all 4 pairs run
        assert set(ds.keys()) == {'wt', 'wtc', 'mut', 'mutc'}     # full cross-product

    def test_scored_objective_is_invariant_to_pruning(self, tmp_path):
        """Pruning removes only unscored pairs, so the objective is identical whether the
        full cross-product or just the diagonal is simulated."""
        cfg, alg, model = self._build_model(tmp_path)
        from pybnf.pset import PSet
        ps = PSet([v.set_value(0.7) for v in alg.variables])
        out = str(tmp_path / 'obj')
        Path(out).mkdir(exist_ok=True)
        home = Path.cwd()
        try:
            pruned = model.copy_with_param_set(ps).execute(out, 'pruned', 60)
            full_model = model.copy_with_param_set(ps)
            full_model.emit_suffixes = None
            full = full_model.execute(out, 'full', 60)
        finally:
            import os
            os.chdir(home)
        score_pruned = cfg.obj.evaluate_multiple({model.name: pruned}, cfg.exp_data, ps)
        score_full = cfg.obj.evaluate_multiple({model.name: full}, cfg.exp_data, ps)
        assert score_pruned == pytest.approx(score_full)
