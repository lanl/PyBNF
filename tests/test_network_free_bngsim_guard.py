"""Config-build guard: a network-free method (NFsim/RuleMonkey) on a new-era
``experiment:`` under ``bngl_backend = bngsim`` is rejected loudly (#434).

The bngsim network-free bridge has no explicit-output-time support (verified absent in
bngsim 0.9.50), so a data-driven ``experiment:`` -- which outputs at the data's
independent-variable points -- would have those points silently dropped and be integrated
on a uniform 0..100 grid, mis-scoring the objective. BNG2.pl honors the points. So the
unsupported combination is rejected at config build with an actionable error instead of
producing a silently-wrong fit. The real fix (native bngsim support + the bridge passing
the points through) is tracked in #427.

These tests need no simulator: only the config-build path (``_load_experiments``) runs.
"""

import os

import pytest

from pybnf.config import Configuration
from pybnf.parse import ploop
from pybnf.printing import PybnfError

_MODEL = """\
begin model
begin parameters
  kA 2
  kB 3
end parameters
begin molecule types
  A()
  B()
end molecule types
begin seed species
  A() 10
  B() 0
end seed species
begin observables
  Molecules obsA A()
end observables
begin reaction rules
  A() -> B() kA
end reaction rules
end model
"""

# A time course (independent variable "time") and a dose-response (independent variable
# "kA", a model parameter -> parameter_scan). Both are data-driven, so both synthesize an
# action with the data's points as explicit output points -- the thing bngsim NF drops.
_TC = "# time\tobsA\n0\t10\n1\t6\n2\t4\n"
_DOSE = "# kA\tobsA\n0.1\t1\n0.2\t2\n0.5\t3\n"

_BASE = [
    "edition = 2", "job_type = de", "objective = sos", "model: m.bngl",
    "uniform_var = kA 0 10", "uniform_var = kB 0 10",
    "population_size = 4", "max_iterations = 1", "verbosity = 0",
]


def _build(tmp_path, *, conf_lines):
    """Write fixtures and build a Configuration from ``conf_lines`` (run from ``tmp_path``
    so the relative paths in the conf resolve)."""
    (tmp_path / "m.bngl").write_text(_MODEL)
    (tmp_path / "tc.exp").write_text(_TC)
    (tmp_path / "dose.exp").write_text(_DOSE)
    conf_text = "\n".join(conf_lines) + "\n"
    home = os.getcwd()
    os.chdir(tmp_path)
    try:
        return Configuration(ploop(conf_text.splitlines(keepends=True)))
    finally:
        os.chdir(home)


class TestGuardRaises:
    """The unsupported combination -- network-free method + explicit bngsim -- is rejected
    at config build with an actionable message."""

    @pytest.mark.parametrize("method", ["nf", "rm", "rulemonkey", "nfsim", "nf_exact"])
    def test_network_free_time_course_on_bngsim_raises(self, tmp_path, method):
        with pytest.raises(PybnfError) as exc:
            _build(tmp_path, conf_lines=_BASE + [
                "bngl_backend = bngsim",
                f"experiment: tc, method: {method}, data: tc.exp",
            ])
        msg = str(exc.value)
        # Names the experiment, the BNG2.pl escape hatch, and the upstream tracker.
        assert "tc" in msg
        assert "bionetgen" in msg
        assert "#427" in msg

    def test_network_free_parameter_scan_on_bngsim_raises(self, tmp_path):
        # The guard is simulation-type agnostic: a dose-response (parameter_scan) is data-
        # driven too (its doses are the explicit output points bngsim NF cannot honor).
        with pytest.raises(PybnfError, match="#427"):
            _build(tmp_path, conf_lines=_BASE + [
                "bngl_backend = bngsim",
                "experiment: dose, method: nf, data: dose.exp",
            ])

    def test_guard_precedes_the_generic_invalid_method_error(self, tmp_path):
        # 'rm' is not an accepted new-era TimeCourse method, so without the guard it would
        # raise the generic "Invalid time course method" from pset. The guard runs first,
        # so the actionable bngsim message wins.
        with pytest.raises(PybnfError, match="bngsim"):
            _build(tmp_path, conf_lines=_BASE + [
                "bngl_backend = bngsim",
                "experiment: tc, method: rm, data: tc.exp",
            ])


class TestGuardDoesNotFire:
    """Everything outside the one unsupported combination builds cleanly."""

    @pytest.mark.parametrize("method", ["ode", "ssa"])
    def test_non_network_free_method_on_bngsim_is_unaffected(self, tmp_path, method):
        conf = _build(tmp_path, conf_lines=_BASE + [
            "bngl_backend = bngsim",
            f"experiment: tc, method: {method}, data: tc.exp",
        ])
        # Config built: the experiment's Data is registered and its action carries the method.
        assert "tc" in conf.exp_data["m"]
        line = next(a for a in conf.models["m"].actions if 'suffix=>"tc"' in a)
        assert f'method=>"{method}"' in line

    def test_default_method_ode_on_bngsim_is_unaffected(self, tmp_path):
        # No method: -> defaults to ode (not network-free), so bngsim is fine.
        conf = _build(tmp_path, conf_lines=_BASE + [
            "bngl_backend = bngsim",
            "experiment: tc, data: tc.exp",
        ])
        assert "tc" in conf.exp_data["m"]

    def test_network_free_under_bng2pl_is_unaffected(self, tmp_path):
        # NFsim under BNG2.pl honors the explicit output points -> no guard.
        conf = _build(tmp_path, conf_lines=_BASE + [
            "bngl_backend = bionetgen",
            "experiment: tc, method: nf, data: tc.exp",
        ])
        assert "tc" in conf.exp_data["m"]
        line = next(a for a in conf.models["m"].actions if 'suffix=>"tc"' in a)
        assert 'method=>"nf"' in line


def _patch_bngsim(monkeypatch, *, available, has_nfsim=False, has_rulemonkey=False):
    """Pin the two resolution seams the auto path consults, so these tests are deterministic
    regardless of whether bngsim (and which NF backend) is actually installed here:
    ``algorithms.base._bngsim_runtime_available`` (is bngsim importable + enabled) and the
    ``bngsim_model._runtime`` capability flags (which NF backend is built, ADR-0018 seam)."""
    monkeypatch.setattr("pybnf.algorithms.base._bngsim_runtime_available", lambda: available)
    monkeypatch.setattr("pybnf.bngsim_model._runtime.BNGSIM_HAS_NFSIM", has_nfsim)
    monkeypatch.setattr("pybnf.bngsim_model._runtime.BNGSIM_HAS_RULEMONKEY", has_rulemonkey)


class TestAutoResolution:
    """The default ``bngl_backend = auto`` is guarded exactly when it would resolve to the
    bngsim NF bridge: bngsim importable AND the *specific* NF backend the method needs is
    built. Otherwise auto falls back to BNG2.pl (which honors the points) and is not guarded.
    """

    def test_auto_builds_when_bngsim_unavailable(self, tmp_path, monkeypatch):
        # bngsim not installed -> auto uses BNG2.pl -> points honored -> no guard.
        _patch_bngsim(monkeypatch, available=False, has_nfsim=True, has_rulemonkey=True)
        conf = _build(tmp_path, conf_lines=_BASE + ["experiment: tc, method: nf, data: tc.exp"])
        assert "tc" in conf.exp_data["m"]

    def test_auto_raises_when_bngsim_and_nfsim_available(self, tmp_path, monkeypatch):
        # bngsim + NFsim built -> auto lands on the bngsim NF bridge -> guard fires.
        _patch_bngsim(monkeypatch, available=True, has_nfsim=True)
        with pytest.raises(PybnfError) as exc:
            _build(tmp_path, conf_lines=_BASE + ["experiment: tc, method: nf, data: tc.exp"])
        msg = str(exc.value)
        assert "auto" in msg and "NFsim" in msg and "bionetgen" in msg and "#427" in msg

    def test_auto_builds_when_nfsim_not_built(self, tmp_path, monkeypatch):
        # bngsim installed but NFsim NOT built -> algorithms.base falls back to BNG2.pl, so
        # the points are honored -> the tightened guard must NOT false-positive here.
        _patch_bngsim(monkeypatch, available=True, has_nfsim=False, has_rulemonkey=True)
        conf = _build(tmp_path, conf_lines=_BASE + ["experiment: tc, method: nf, data: tc.exp"])
        assert "tc" in conf.exp_data["m"]

    def test_auto_rulemonkey_raises_when_rulemonkey_built(self, tmp_path, monkeypatch):
        # A RuleMonkey token needs the RuleMonkey backend specifically.
        _patch_bngsim(monkeypatch, available=True, has_rulemonkey=True)
        with pytest.raises(PybnfError, match="RuleMonkey"):
            _build(tmp_path, conf_lines=_BASE + ["experiment: tc, method: rm, data: tc.exp"])

    def test_auto_rulemonkey_passes_guard_when_only_nfsim_built(self, tmp_path, monkeypatch):
        # NFsim being present does not satisfy a RuleMonkey method, so the guard does not
        # fire (auto would fall back to BNG2.pl). 'rm' is not an accepted new-era TimeCourse
        # method, so the downstream generic error surfaces instead -- proving the guard
        # passed through rather than raising its own bngsim/#427 message.
        _patch_bngsim(monkeypatch, available=True, has_nfsim=True, has_rulemonkey=False)
        with pytest.raises(PybnfError) as exc:
            _build(tmp_path, conf_lines=_BASE + ["experiment: tc, method: rm, data: tc.exp"])
        msg = str(exc.value)
        assert "Invalid time course method" in msg
        assert "#427" not in msg
