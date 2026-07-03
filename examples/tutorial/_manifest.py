"""Test-side manifest for the edition-2 tutorial examples.

This module is the single source of truth the *tests* read (never the tutorial
confs themselves -- those stay clean and realistic). It records, per example:

* how to (re)generate each committed ``.exp`` -- the model, observable columns,
  time grid, and any synthetic noise (used by ``regenerate_data.py``); and
* what each committed ``.conf`` should recover -- the true parameter values, the
  recovery tolerance, and the pytest marker tier (default / slow / jax-gated),
  used by ``tests/test_tutorial_examples.py``.

Keeping the truth + tolerances here (not in the confs) is what lets one artifact
set serve both the tutorial (clean, copy-pasteable) and the test suite
(asserting, gated) without the two colliding.
"""
from dataclasses import dataclass, field
from pathlib import Path

TUTORIAL_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Dataset:
    """One committed ``.exp`` to generate from a model at its true parameters."""
    exp: str                 # filename, relative to the example folder
    obs: tuple               # observable columns to write (independent var is 'time')
    t_end: float
    n_points: int
    noise_sd: float = 0.0     # gaussian noise (absolute sd) added + written as _SD; 0 = clean
    noise_seed: int = 0
    scan: str = None          # if set, an independent-variable column name (dose-response)


@dataclass(frozen=True)
class ConfCheck:
    """One committed ``.conf`` and what a fit of it must recover."""
    conf: str                 # filename, relative to the example folder
    recover: dict             # {param: true_value} the fit must return (within tol)
    tol: float = 0.03          # fractional recovery tolerance
    marker: str = 'default'    # 'default' (bngsim+newera) | 'slow' | 'jax'
    refused: bool = False      # True => a gradient fit that must be REFUSED, not run
    note: str = ''


@dataclass(frozen=True)
class Example:
    folder: str
    model: str
    truth: dict                          # ground-truth parameter values
    build_free: dict                     # {param: (var_type, lo, hi)} to construct data-gen fit
    datasets: tuple = ()
    confs: tuple = ()

    @property
    def path(self):
        return TUTORIAL_DIR / self.folder


EXAMPLES = (
    Example(
        folder='01_logistic_growth',
        model='logistic_growth.bngl',
        truth={'r': 1.2, 'K': 100.0},
        build_free={'r': ('uniform_var', 0.1, 5.0), 'K': ('uniform_var', 20.0, 300.0)},
        datasets=(
            Dataset('logistic_growth.exp', obs=('Obs_N',), t_end=8, n_points=17),
        ),
        confs=(
            ConfCheck('logistic_growth_trf.conf', recover={'r': 1.2, 'K': 100.0}, tol=0.02),
        ),
    ),
    Example(
        folder='02_bateman_chain',
        model='bateman_chain.bngl',
        truth={'k1': 0.8, 'k2': 0.25},
        build_free={'k1': ('uniform_var', 0.05, 3.0), 'k2': ('uniform_var', 0.02, 2.0)},
        datasets=(
            Dataset('bateman_chain.exp', obs=('Obs_A', 'Obs_B', 'Obs_C'), t_end=20, n_points=21),
            # Lesson 4: observe ONLY A -> k2 is structurally non-identifiable.
            Dataset('bateman_A_only.exp', obs=('Obs_A',), t_end=20, n_points=21),
        ),
        confs=(
            ConfCheck('bateman_chain_de.conf', recover={'k1': 0.8, 'k2': 0.25}, tol=0.03),
        ),
    ),
    Example(
        folder='03_gompertz_growth',
        model='gompertz_growth.bngl',
        truth={'r': 0.4, 'K': 100.0},
        build_free={'r': ('uniform_var', 0.05, 3.0), 'K': ('uniform_var', 20.0, 400.0)},
        datasets=(
            Dataset('gompertz_growth.exp', obs=('Obs_X',), t_end=20, n_points=21),
        ),
        confs=(
            ConfCheck('gompertz_growth_pso.conf', recover={'r': 0.4, 'K': 100.0}, tol=0.05),
        ),
    ),
    Example(
        folder='06_step_input',
        model='step_input.bngl',
        truth={'k': 0.6, 'J_base': 1.2, 'J_step': 2.4},
        build_free={'k': ('uniform_var', 0.1, 3.0),
                    'J_base': ('uniform_var', 0.1, 5.0),
                    'J_step': ('uniform_var', 0.1, 6.0)},
        datasets=(
            Dataset('step_input.exp', obs=('Obs_X',), t_end=12, n_points=25),
        ),
        confs=(
            ConfCheck('step_input_de.conf',
                      recover={'k': 0.6, 'J_base': 1.2, 'J_step': 2.4}, tol=0.05),
            ConfCheck('step_input_trf_refused.conf', recover={}, refused=True,
                      note='piecewise input -> gradient optimizer must refuse'),
        ),
    ),
    Example(
        folder='06_step_input',
        model='step_input_smooth.bngl',
        truth={'k': 0.6, 'J_base': 1.2, 'J_step': 2.4, 'tau': 4.0},
        build_free={'k': ('uniform_var', 0.1, 3.0),
                    'J_base': ('uniform_var', 0.1, 5.0),
                    'J_step': ('uniform_var', 0.1, 6.0),
                    'tau': ('uniform_var', 1.0, 8.0)},
        datasets=(
            Dataset('step_input_smooth.exp', obs=('Obs_X',), t_end=12, n_points=25),
        ),
        confs=(
            # The smooth sigmoid step IS differentiable -> trf fits it, and even
            # recovers the transition time tau (impossible for the hard if()).
            ConfCheck('step_input_smooth_trf.conf',
                      recover={'k': 0.6, 'J_base': 1.2, 'J_step': 2.4, 'tau': 4.0},
                      tol=0.03),
        ),
    ),
)


def example_by_folder(folder):
    for ex in EXAMPLES:
        if ex.folder == folder:
            return ex
    raise KeyError(folder)
