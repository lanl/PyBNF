"""Ground truth for the ``examples/real-world/`` corpus (published-model PyBNF jobs
organized as ``Author-Year/job_slug``, on the edition-2 surface -- see README.md).

Like ``examples/tutorial/_manifest.py``, this keeps the *test* knowledge (which conf,
which simulator, what a run should produce) OUT of the committed ``.conf`` files, so
those stay clean, runnable, and faithful to the paper. ``tests/test_real_world_examples.py``
imports ``EXAMPLES`` and drives each one through the real bngsim backend.

Each entry records:

* ``folder`` / ``conf``  -- the committed example to run (paths inside the conf are
  relative to ``folder``);
* ``simulator``          -- ``ode`` | ``ssa`` | ``nf``: the BioNetGen method the
  edition-2 ``experiment:`` synthesizes (deterministic ODE, Gillespie SSA, or
  network-free NFsim). This is the axis #380 cares about: representative
  deterministic / stochastic / network-free examples validated through bngsim;
* ``observables``        -- the model observables/functions the data bind to (the
  columns a correct simulation must produce; a finite objective proves they mapped);
* ``stochastic``         -- True for ssa/nf (seed-dependent output -> no determinism
  assertion, wider tolerances);
* ``heavy``              -- True when one build/fit is expensive (large network
  generation or many network-free sims) -> the test adds ``@pytest.mark.slow`` so it
  is opt-in even within the recovery tier;
* ``constraint_only``    -- True for a BPSL-only job with qualitative ``.prop`` inputs
  and no quantitative ``.exp`` data;
* ``recover`` / ``tol``  -- optional {param: truth} for the synthetic-data examples
  where a bounded fit can be expected to move a parameter toward a known value; left
  empty for the experimental-data fits (there the check is only that the end-to-end
  simulate -> score -> propose loop runs and yields a finite, improving objective).
"""
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RealWorldExample:
    folder: str
    conf: str
    simulator: str                 # 'ode' | 'ssa' | 'nf'
    observables: tuple             # data-bound model observable/function names
    system: str                    # one-line description (biology + paper mapping)
    stochastic: bool = False
    heavy: bool = False            # cluster-scale build/fit -> excluded from the executable tier
    blocked: str = ''              # non-empty => cannot yet complete through bngsim (with reason)
    constraint_only: bool = False  # qualitative BPSL job: constraints, no quantitative data
    recover: dict = field(default_factory=dict)
    tol: float = 0.5

    @property
    def path(self) -> Path:
        return _ROOT / self.folder


EXAMPLES = [
    # ---- deterministic ODE ------------------------------------------------- #
    RealWorldExample(
        folder='Kozer-2013/egfr_ode', conf='egfr_ode.conf', simulator='ode', heavy=True,
        observables=('pre1_dose', 'pre2_time', 'pre3_dose', 'pre4_time'),
        system='EGFR activation & clustering (Kozer 2013); ODE, time course + dose-response'),
    RealWorldExample(
        folder='Mitra-2019/receptor', conf='receptor.conf', simulator='ode',
        observables=('RLbonds', 'pR'),
        system='Ligand/receptor binding + phosphorylation (BioNetFit 1 ex 5); ODE, pre-equilibration'),
    RealWorldExample(
        folder='Erickson-2019/igf1r', conf='igf1r.conf', simulator='ode',
        observables=('IGF1_hot_bound',),
        system='IGF1/IGF1R harmonic-oscillator binding (Erickson 2019; Kiselyov 2009); '
               'ODE, 7-rate/3-dataset preincubate-wash-dose-scan fit'),
    RealWorldExample(
        folder='Rijal-2025/lacud5_ode', conf='lacud5_ode.conf', simulator='ode',
        observables=('Mean_mRNA', 'mRNA_SD'),
        system='lacUV5/lacUD5 promoter noise (Rijal & Mehta 2025; Jones et al. 2014); exact moment ODE twin'),
    RealWorldExample(
        folder='Rijal-2025/five_dl1_ode', conf='five_dl1_ode.conf', simulator='ode',
        observables=('Mean_mRNA', 'mRNA_SD'),
        system='5DL1 promoter noise (Rijal & Mehta 2025; Jones et al. 2014); exact moment ODE twin'),
    RealWorldExample(
        folder='Salazar-Cavazos-2019/egfr_simpull', conf='egfr_simpull.conf', simulator='ode',
        observables=('pY1068_percent', 'pY1173_percent', 'phosR_per'),
        system='Multisite EGFR phosphorylation (Salazar-Cavazos 2020); ODE, SiMPull '
               'dose-response + time course, authors\' PyBioNetFit fit'),
    RealWorldExample(
        folder='Kirsch-2020/phosphoswitch_bpsl', conf='phosphoswitch_bpsl.conf',
        simulator='ode', observables=('p38ATF2all',), constraint_only=True,
        system='JNK/p38/ATF2 S90 phosphoswitch (Kirsch 2020); ODE, BPSL-only '
               'cross-condition qualitative constraints'),

    # ---- stochastic SSA ---------------------------------------------------- #
    RealWorldExample(
        folder='Rijal-2025/lacud5_ssa', conf='lacud5_ssa.conf', simulator='ssa',
        stochastic=True,
        observables=('Mean_mRNA', 'mRNA_SD'),
        system='lacUV5/lacUD5 promoter noise (Rijal & Mehta 2025; Jones et al. 2014); exact SSA ensemble moments'),
    RealWorldExample(
        folder='Rijal-2025/five_dl1_ssa', conf='five_dl1_ssa.conf', simulator='ssa',
        stochastic=True,
        observables=('Mean_mRNA', 'mRNA_SD'),
        system='5DL1 promoter noise (Rijal & Mehta 2025; Jones et al. 2014); exact SSA ensemble moments'),
    RealWorldExample(
        folder='Gupta-2018/fceri_gamma', conf='fceri_gamma.conf', simulator='ssa',
        stochastic=True, heavy=True,
        observables=('LynFree', 'RecMon', 'RecPbeta', 'RecPgamma', 'RecSyk', 'RecSykPS'),
        system='FceRI gamma-chain signaling (Gupta & Mendes 2018); Gillespie SSA, synthetic data'),

    # ---- network-free NFsim ------------------------------------------------ #
    RealWorldExample(
        folder='Mitra-2019/receptor_nf', conf='receptor_nf.conf', simulator='nf', stochastic=True,
        heavy=True,
        observables=('RLbonds', 'pR'),
        # Fixed-time NF pre-equilibration (equil_t_end: 600) makes this run through bngsim
        # (previously it hung: NFsim has no steady-state solve, so the default steady-state
        # equilibration integrated to t=1e6). It is HEAVY -- NFsim on ~1000 molecules is
        # cluster-scale, exceeding wall_time_sim for some parameter sets -- so it stays in the
        # backend-free tier only.
        system='Ligand/receptor binding (BioNetFit 1 ex 6); network-free NFsim, pre-equilibration'),
    RealWorldExample(
        folder='Monine-2010/tlbr', conf='tlbr.conf', simulator='nf', stochastic=True,
        observables=('FL',),
        system='Trivalent-ligand/bivalent-receptor aggregation (BioNetFit 1 ex 3); network-free NFsim, dose-response'),
    RealWorldExample(
        folder='Kozer-2013/egfr_nf', conf='egfr_nf.conf', simulator='nf',
        stochastic=True, heavy=True,
        observables=('pre1_dose', 'pre2_time', 'pre3_dose', 'pre4_time'),
        system='EGFR activation & clustering (Kozer 2013; BioNetFit 1 ex 2); network-free NFsim, time course + dose-response'),
]


def example_by_folder(folder: str) -> RealWorldExample:
    for ex in EXAMPLES:
        if ex.folder == folder:
            return ex
    raise KeyError(folder)
