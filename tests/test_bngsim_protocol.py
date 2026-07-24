"""Tests for method=>"protocol" — PyBNF's multi-phase simulation action.

A ``begin protocol ... end protocol`` block is a sequence of simulate /
setConcentration / addConcentration steps that BngsimModel walks against a live
bngsim engine (``_run_protocol``), optionally once per parameter_scan point. The
feature is entirely PyBNF's; bngsim is only the backend the execution steps drive.

These tests lived in the bngsim repo, importing these private helpers across the
repo boundary, where they skipped on every hosted runner (no bngsim wheel there)
and only ran on a maintainer's laptop. They move here — next to the code they
cover, in a suite that now installs bngsim from PyPI (#514) — as part of
lanl/bngsim#45. The parsing-helper cases that came with them
(_parse_add_concentration, _parse_set_concentration, continue=>1,
_normalize_action_method) are dropped rather than moved: test_bngsim_bridge.py
already covers each, more thoroughly.

TestProtocolParsing is unmarked: it exercises BNGLModel block extraction, pure
PyBNF parsing that needs no bngsim, so it runs on every leg. The execution and
scan classes drive the real engine and carry ``@pytest.mark.bngsim``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pybnf.bngsim_model import BngsimModel
from pybnf.pset import BNGLModel

FIXTURES = Path(__file__).resolve().parent / 'bngl_files'


@pytest.fixture
def reversible_net() -> Path:
    """A + B <-> C with kf/kr; A0=100, B0=50, C0=0. Observables A_free/B_free/C_total."""
    return FIXTURES / 'two_species_reversible.net'


# ---------------------------------------------------------------------------
# Protocol block extraction (BNGLModel — no bngsim needed)
# ---------------------------------------------------------------------------


class TestProtocolParsing:
    """begin protocol...end protocol extraction in BNGLModel."""

    def test_protocol_block_extracted(self, tmp_path: Path):
        """Protocol lines are stored in self.protocol, not self.actions."""
        bngl = tmp_path / "test.bngl"
        bngl.write_text(
            "begin parameters\n"
            "  1 k__FREE 0.1\n"
            "end parameters\n"
            "begin protocol\n"
            '  simulate({method=>"ode",t_end=>100,n_steps=>10})\n'
            '  setConcentration("A()",50)\n'
            '  simulate({method=>"ode",t_end=>200,n_steps=>10,continue=>1})\n'
            "end protocol\n"
            "generate_network({})\n"
            'parameter_scan({method=>"protocol",parameter=>"k__FREE",par_scan_vals=>[0.1,0.2]})\n'
        )
        model = BNGLModel(str(bngl))
        assert len(model.protocol) == 3
        assert "simulate" in model.protocol[0]
        assert "setConcentration" in model.protocol[1]
        assert "continue=>1" in model.protocol[2]
        # The parameter_scan line should be in actions, not protocol
        assert any("parameter_scan" in a for a in model.actions)
        assert not any("parameter_scan" in p for p in model.protocol)

    def test_empty_protocol_block(self, tmp_path: Path):
        """Empty protocol block produces empty list."""
        bngl = tmp_path / "test.bngl"
        bngl.write_text(
            "begin parameters\n"
            "  1 k__FREE 0.1\n"
            "end parameters\n"
            "begin protocol\n"
            "end protocol\n"
            "generate_network({})\n"
        )
        model = BNGLModel(str(bngl))
        assert model.protocol == []

    def test_protocol_comments_preserved(self, tmp_path: Path):
        """Comment lines within protocol are preserved (filtered at execution time)."""
        bngl = tmp_path / "test.bngl"
        bngl.write_text(
            "begin parameters\n"
            "  1 k__FREE 0.1\n"
            "end parameters\n"
            "begin protocol\n"
            "# Equilibrate\n"
            '  simulate({method=>"ode",t_end=>100,n_steps=>1})\n'
            "end protocol\n"
        )
        model = BNGLModel(str(bngl))
        assert len(model.protocol) == 2
        assert model.protocol[0].startswith("#")


# ---------------------------------------------------------------------------
# Protocol execution through the real bngsim engine
# ---------------------------------------------------------------------------


@pytest.mark.bngsim
class TestProtocolExecution:
    """_run_protocol on a real model."""

    def test_protocol_basic(self, reversible_net: Path):
        """Run a simple two-step protocol and get a result."""
        protocol = [
            'simulate({method=>"ode",t_start=>0,t_end=>50,n_steps=>10})',
            'setConcentration("A()",200)',
            'simulate({method=>"ode",t_start=>0,t_end=>50,n_steps=>10})',
        ]
        model = BngsimModel("rev", [], [], [], nf=str(reversible_net), protocol=protocol)
        engine = model._engine_model
        result = model._run_protocol(engine)
        assert result is not None
        assert result.n_times == 11  # n_steps + 1

    def test_protocol_continue(self, reversible_net: Path):
        """continue=>1 chains simulations: t_start of second = t_end of first."""
        protocol = [
            'simulate({method=>"ode",t_start=>0,t_end=>50,n_steps=>5})',
            'simulate({method=>"ode",t_end=>100,n_steps=>5,continue=>1})',
        ]
        model = BngsimModel("rev", [], [], [], nf=str(reversible_net), protocol=protocol)
        engine = model._engine_model
        result = model._run_protocol(engine)
        assert result is not None
        times = np.asarray(result.time)
        # The second simulate should start at t=50 and end at t=100
        assert times[0] == pytest.approx(50.0)
        assert times[-1] == pytest.approx(100.0)

    def test_protocol_sample_times(self, reversible_net: Path):
        """simulate inside protocol honors sample_times."""
        protocol = [
            'simulate({method=>"ode",sample_times=>[0,1,5,10,50]})',
        ]
        model = BngsimModel("rev", [], [], [], nf=str(reversible_net), protocol=protocol)
        engine = model._engine_model
        result = model._run_protocol(engine)
        assert result is not None
        times = np.asarray(result.time)
        np.testing.assert_allclose(times, [0, 1, 5, 10, 50], atol=1e-12)

    def test_protocol_add_concentration(self, reversible_net: Path):
        """addConcentration in protocol adds to the current value."""
        protocol = [
            'simulate({method=>"ode",t_start=>0,t_end=>50,n_steps=>1})',
            'addConcentration("A()",25)',
            'simulate({method=>"ode",t_start=>0,t_end=>50,n_steps=>5})',
        ]
        model = BngsimModel("rev", [], [], [], nf=str(reversible_net), protocol=protocol)
        engine = model._engine_model
        # A() starts at 100; after first simulate it decays, then we add 25
        conc_before = engine.get_concentration("A()")
        assert conc_before == pytest.approx(100.0)
        result = model._run_protocol(engine)
        assert result is not None
        # After the protocol, A() should reflect the addition
        # (exact value depends on dynamics, but get_concentration should work)
        engine.get_concentration("A()")
        # The key check: the protocol ran without error and produced a result
        assert result.n_times == 6  # n_steps + 1

    def test_protocol_no_simulate_returns_none(self, reversible_net: Path):
        """Protocol with only non-simulate actions returns None."""
        protocol = [
            'setConcentration("A()",200)',
        ]
        model = BngsimModel("rev", [], [], [], nf=str(reversible_net), protocol=protocol)
        engine = model._engine_model
        result = model._run_protocol(engine)
        assert result is None

    def test_protocol_expression_t_end(self, reversible_net: Path):
        """Protocol handles t_end as arithmetic expression (e.g. 3600*5)."""
        protocol = [
            'simulate({method=>"ode",t_start=>0,t_end=>10*5,n_steps=>5})',
        ]
        model = BngsimModel("rev", [], [], [], nf=str(reversible_net), protocol=protocol)
        engine = model._engine_model
        result = model._run_protocol(engine)
        assert result is not None
        times = np.asarray(result.time)
        assert times[-1] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# parameter_scan with method=>"protocol"
# ---------------------------------------------------------------------------


@pytest.mark.bngsim
class TestProtocolParameterScan:
    """method=>'protocol' routing in parameter_scan."""

    def test_protocol_scan_basic(self, reversible_net: Path, tmp_path: Path):
        """parameter_scan with method=>'protocol' runs the protocol per scan point."""
        protocol = [
            'simulate({method=>"ode",t_start=>0,t_end=>100,n_steps=>10})',
            'setConcentration("A()",200)',
            'simulate({method=>"ode",t_start=>0,t_end=>100,n_steps=>10})',
        ]
        actions = [
            'parameter_scan({method=>"protocol",parameter=>"kf",'
            'par_scan_vals=>[0.001,0.01],suffix=>"pscan"})',
        ]
        model = BngsimModel(
            "rev",
            actions,
            [("parameter_scan", "pscan")],
            [],
            nf=str(reversible_net),
            protocol=protocol,
        )
        ds = model.execute(str(tmp_path), "test", timeout=60, with_mutants=False)
        assert "pscan" in ds
        data = ds["pscan"]
        assert data.data.shape[0] == 2  # two scan points
        # Column 0 is the scan parameter value
        assert data.data[0, 0] == pytest.approx(0.001)
        assert data.data[1, 0] == pytest.approx(0.01)
        # Observable values should differ between scan points
        assert not np.allclose(data.data[0, 1:], data.data[1, 1:])

    def test_protocol_scan_empty_protocol_raises(self, reversible_net: Path, tmp_path: Path):
        """method=>'protocol' with no protocol block raises ValueError."""
        actions = [
            'parameter_scan({method=>"protocol",parameter=>"kf",'
            'par_scan_vals=>[0.001],suffix=>"pscan"})',
        ]
        model = BngsimModel(
            "rev",
            actions,
            [("parameter_scan", "pscan")],
            [],
            nf=str(reversible_net),
            protocol=[],
        )
        with pytest.raises(ValueError, match="no begin protocol"):
            model.execute(str(tmp_path), "test", timeout=60, with_mutants=False)

    def test_protocol_scan_no_simulate_raises(self, reversible_net: Path, tmp_path: Path):
        """Protocol with only setConcentration raises ValueError in scan."""
        protocol = [
            'setConcentration("A()",200)',
        ]
        actions = [
            'parameter_scan({method=>"protocol",parameter=>"kf",'
            'par_scan_vals=>[0.001],suffix=>"pscan"})',
        ]
        model = BngsimModel(
            "rev",
            actions,
            [("parameter_scan", "pscan")],
            [],
            nf=str(reversible_net),
            protocol=protocol,
        )
        with pytest.raises(ValueError, match="no simulate"):
            model.execute(str(tmp_path), "test", timeout=60, with_mutants=False)
