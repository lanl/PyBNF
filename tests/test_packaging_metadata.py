import fnmatch
import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path


def test_project_metadata_declares_python_floor_and_bngsim_dependency():
    pyproject_path = Path(__file__).resolve().parents[1] / 'pyproject.toml'
    metadata = tomllib.loads(pyproject_path.read_text())
    project = metadata['project']

    assert project['requires-python'] == '>=3.11'
    # bngsim >= 0.11.35: observable/expression-level steady-state forward sensitivities on
    # SteadyStateResult.output_sensitivities (lanl/bngsim#12), which make a scored KINSOL/Newton
    # dose-response scan differentiable (#478). (0.11.34 added the native carried-state
    # parameter_scan/bifurcate + named saved states of lanl/bngsim#11 for #474.)
    assert 'bngsim>=0.11.35,<1' in project['dependencies']


def test_every_pybnf_subpackage_is_shipped():
    """Every importable subpackage under ``pybnf/`` must be matched by the
    setuptools package-discovery config, so a newly added subpackage cannot
    silently vanish from the built wheel.

    Regression guard: a hand-maintained ``[tool.setuptools] packages`` list
    omitted ``pybnf.priors`` and ``pybnf.noise`` (added in later refactors).
    The source tree imports both at module load, so the gap broke an installed
    wheel while a source checkout kept working. We now use glob discovery
    (``[tool.setuptools.packages.find] include = ["pybnf*"]``); this test
    re-derives the on-disk packages and asserts the include patterns cover
    them, using ``fnmatch`` to mirror setuptools' own glob semantics (and so
    avoid importing setuptools, which the test venv may not provide).
    """
    repo_root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((repo_root / 'pyproject.toml').read_text())

    # On-disk subpackages: every directory under pybnf/ holding an __init__.py.
    pkg_root = repo_root / 'pybnf'
    ondisk = sorted(
        '.'.join(init.parent.relative_to(repo_root).parts)
        for init in pkg_root.rglob('__init__.py')
    )
    # The two subpackages the old hand-maintained list dropped -- anchor the
    # regression so the test is meaningful even if discovery is reworked again.
    assert 'pybnf.priors' in ondisk
    assert 'pybnf.noise' in ondisk

    find_cfg = metadata['tool']['setuptools']['packages']['find']
    includes = find_cfg.get('include', ['*'])
    excludes = find_cfg.get('exclude', [])
    for pkg in ondisk:
        assert any(fnmatch.fnmatch(pkg, pat) for pat in includes), (
            f'{pkg} is on disk but not matched by packages.find include={includes}'
        )
        assert not any(fnmatch.fnmatch(pkg, pat) for pat in excludes), (
            f'{pkg} is on disk but excluded by packages.find exclude={excludes}'
        )


def test_subprocess_no_jax_import_outside_hmc():
    """No non-HMC path may import ``jax``: importing PyBNF, registering every job type,
    and running a **log-scaled gradient assembly** must leave ``sys.modules`` jax-free.

    ``jax``/``jaxlib``/``blackjax`` are the optional ``pybnf[jax]`` extra, adopted for
    exactly one thing (ADR-0059): autodiff of the analytical model's log-density for the
    ``hmc`` reference sampler. That surface -- each prior family's ``logpdf_jax``, the
    bijector's ``*_jax`` peers, ``Scale.inverse_jax``, the PEtab formula's ``jax``
    lambdify backend -- hangs off general-purpose objects, so any layer can reach for a
    jax peer and quietly make a second subsystem depend on the extra. That is exactly
    what happened (#524): gradient assembly obtained a log-scaled parameter's
    ``d theta/d u`` with ``jax.grad(inverse_jax)``, so most gradient fits -- anything
    declaring a rate constant ``loguniform_var`` -- hard-required the extra. ADR-0087
    replaced it with the scale's analytic ``d_inverse``; this test keeps the next such
    reuse from going unnoticed.

    Runs in a subprocess because the pytest session itself imports jax (``test_hmc.py``,
    and the ADR-0087 parity tests), so the parent's ``sys.modules`` says nothing.

    Both halves matter, and which one fires depends on the environment. **Without** the
    extra installed, a module-scope ``import jax`` anywhere in the imported surface makes
    the child die with ``ImportError`` (a nonzero exit, asserted below). **With** it
    installed -- the ``pytest-jax`` CI leg, and a dev machine -- the ``sys.modules`` check
    additionally catches a leak inside a function body, which is the shape #524 took.
    """
    script = textwrap.dedent('''
        import sys

        import numpy as np

        # Every job type registers here, including hmc: its module must import no jax at
        # import time (ADR-0059), or `pip install pybnf` could not even list the samplers.
        import pybnf.algorithms
        import pybnf.config
        import pybnf.objective
        import pybnf.pset
        import pybnf.priors          # 18 families, each with a logpdf_jax peer
        import pybnf.gradient
        try:
            import pybnf.petab       # its formula module has a 'jax' lambdify backend
        except ImportError:
            pass                     # the optional pybnf[petab] extra is absent

        from pybnf.data import Data, OutputSensitivities
        from pybnf.gradient import (
            assemble_gaussian_gradient, ExperimentRouting, ParamRoute, PARAM)
        from pybnf.objective import ChiSquareObjective
        from pybnf.pset import FreeParameter

        # A log-scaled free parameter through the prior/scale surface (sampling-space
        # transform, prior density, a draw) -- the priors package's own jax peers.
        k = FreeParameter('k', 'loguniform_var', 0.01, 100.0, value=2.0)
        u = k.to_sampling_space(k.value)
        assert np.isclose(k.from_sampling_space(u), k.value)
        k.prior_logpdf(k.value)
        k.sample_value(np.random.default_rng(0))

        # ...and through a full gradient assembly, the #524 regression path: the
        # native->sampling Jacobian factor of a log-scaled parameter.
        times = np.array([0.0, 1.0, 2.0])
        sim = Data.from_columns(
            np.column_stack([times, [100.0, 74.0, 55.0]]), ['time', 'Stot'])
        sim.output_sensitivities = OutputSensitivities(
            selectors=['observable:Stot'], param_names=['k'], ic_species=[],
            d_param=np.array([0.0, -80.0, -106.0]).reshape(3, 1, 1))
        exp = Data.from_columns(
            np.column_stack([times, [100.0, 70.0, 60.0], [5.0] * 3]),
            ['time', 'Stot', 'Stot_SD'])
        routing = ExperimentRouting(routes={'k': ParamRoute.single('k', PARAM, 'k', 1.0)})
        result = assemble_gaussian_gradient(
            ChiSquareObjective(), [(sim, exp, routing)], [k])
        # d theta/du = ln(10)*theta scales the one column (ADR-0087), analytically.
        assert np.isclose(result.jacobian[1, 0], (-80.0 / 5.0) * np.log(10.0) * 2.0)

        # Report the top-level roots only -- importing jax pulls in ~250 submodules,
        # and the name that matters is which extra leaked, not its module list.
        roots = {m.split('.')[0] for m in sys.modules}
        leaked = sorted(roots & {'jax', 'jaxlib', 'blackjax'})
        assert not leaked, (
            'imported outside the hmc path: %r -- the pybnf[jax] extra is for '
            'ADR-0059 autodiff of the analytical log-density only (see #524).' % (leaked,))
        print('OK')
    ''')
    result = subprocess.run(
        [sys.executable, '-c', script], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        'stdout=%r stderr=%r' % (result.stdout, result.stderr)
    )
    assert 'OK' in result.stdout


def test_subprocess_pybnf_no_bngsim_package_root_import_smoke():
    """Importing pybnf and its bngsim-aware submodules must not raise under
    PYBNF_NO_BNGSIM=1.

    Belt-and-suspenders for
    ``test_bngsim_capabilities.py::test_subprocess_pybnf_no_bngsim_disables``,
    which only checks ``pybnf._bngsim_caps``. This test exercises the full
    set of submodules that lazily import bngsim at top level, so a future
    accidental ``import bngsim`` at module scope would surface here.
    """
    script = textwrap.dedent('''
        import pybnf
        import pybnf.algorithms
        import pybnf.bngsim_model
        import pybnf.bngsim_sbml_model
        import pybnf.bngsim_antimony_model
        from pybnf._bngsim_caps import BNGSIM_AVAILABLE
        assert BNGSIM_AVAILABLE is False, BNGSIM_AVAILABLE
        print('OK')
    ''')
    env = os.environ.copy()
    env['PYBNF_NO_BNGSIM'] = '1'
    result = subprocess.run(
        [sys.executable, '-c', script],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        'stdout=%r stderr=%r' % (result.stdout, result.stderr)
    )
    assert 'OK' in result.stdout
