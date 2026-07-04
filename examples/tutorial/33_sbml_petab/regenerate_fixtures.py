#!/usr/bin/env python
"""Regenerate lesson 33's SBML model and PEtab v2 measurement table.

This is a **developer tool**, not part of the test run. It (1) converts the
Antimony source ``cycle.ant`` to the committed SBML ``cycle.xml``, and (2)
simulates that SBML at the true rate ``k = 1.0`` through the bngsim backend to
fill ``measurements.tsv`` -- so the PEtab measurements are the model's own output
at the truth, and a fit recovers ``k`` exactly. The other PEtab tables
(parameters/observables/conditions/experiments) and ``problem.yaml`` are static
and rewritten here too, so this script is the single source of the whole problem.

Only INTEGER measurement times are used: the new-era ``experiment:`` surface does
not yet thread a non-integer data grid into an SBML simulation (lanl/PyBNF#470),
so the problem is authored on ``t = 0..8``.

Usage (needs antimony + bngsim + BNG2.pl; set BNGPATH):

    python examples/tutorial/33_sbml_petab/regenerate_fixtures.py
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import tests.recovery_harness as H          # noqa: E402
from pybnf.pset import PSet                 # noqa: E402

K_TRUE = 1.0
GRID = list(range(0, 9))                    # integer grid t = 0..8 (see #470)
OBSERVABLES = ('X1', 'X2')                  # bare species observed directly

_YAML = """\
format_version: 2.0.0
parameter_files:
  - parameters.tsv
observable_files:
  - observables.tsv
measurement_files:
  - measurements.tsv
condition_files:
  - conditions.tsv
experiment_files:
  - experiments.tsv
model_files:
  cycle:
    location: cycle.xml
    language: sbml
"""


def _write_sbml_from_antimony():
    import antimony as a
    rc = a.loadAntimonyFile(str(_HERE / 'cycle.ant'))
    if rc < 0:
        raise RuntimeError(f'antimony failed to load cycle.ant: {a.getLastError()}')
    (_HERE / 'cycle.xml').write_text(a.getSBMLString(a.getMainModuleName()))
    a.clearPreviousLoads()


def _simulate_truth():
    """Return {species: {t: value}} from bngsim at k = K_TRUE on the integer grid."""
    H.require_bng2pl()
    scratch = Path(tempfile.mkdtemp(prefix='sbml_petab_gen_'))
    try:
        ph = scratch / 'ph.exp'
        ph.write_text('#\ttime\t' + '\t'.join(OBSERVABLES) + '\n' +
                      '\n'.join(f'{t}\t' + '\t'.join('0' for _ in OBSERVABLES) for t in GRID) + '\n')
        conf = H.make_newera_config(scratch, str(_HERE / 'cycle.xml'), ph,
                                    {'k': ('uniform_var', 0.1, 5.0)}, 'timecourse', 'de',
                                    sbml_backend='bngsim', population_size=4, max_iterations=1)
        alg = H.build(conf, 'de')
        truth = PSet([v.set_value(K_TRUE) for v in alg.variables])
        model = alg.model_list[0].copy_with_param_set(truth)
        out = str(scratch / 'truth')
        os.makedirs(out, exist_ok=True)
        home = os.getcwd()
        try:
            ds = model.execute(out, 'truth', 0)
        finally:
            os.chdir(home)
        data = ds['timecourse'] if 'timecourse' in ds else ds[next(iter(ds))]
        arr = np.asarray(data.data)
        idx = {n: i for n, i in data.cols.items()}
        return {s: {row[idx['time']]: row[idx[s]] for row in arr} for s in OBSERVABLES}
    finally:
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)


def main():
    _write_sbml_from_antimony()
    truth = _simulate_truth()

    (_HERE / 'parameters.tsv').write_text(
        'parameterId\testimate\tlowerBound\tupperBound\n'
        'k\ttrue\t0.1\t5\n')
    (_HERE / 'observables.tsv').write_text(
        'observableId\tobservableFormula\tnoiseFormula\tnoiseDistribution\n'
        + ''.join(f'obs_{s}\t{s}\t1\tnormal\n' for s in OBSERVABLES))
    rows = ''.join(f'obs_{s}\texp1\t{t}\t{truth[s][t]:.10g}\n'
                   for t in GRID for s in OBSERVABLES)
    (_HERE / 'measurements.tsv').write_text(
        'observableId\texperimentId\ttime\tmeasurement\n' + rows)
    # No experimental perturbations: a single time course under the model's own
    # initial state, so conditions/experiments carry only their headers (Lessons 9
    # and 29 show these tables filled in for dose-response and washout protocols).
    (_HERE / 'conditions.tsv').write_text('conditionId\n')
    (_HERE / 'experiments.tsv').write_text('experimentId\ttime\tconditionId\n')
    (_HERE / 'problem.yaml').write_text(_YAML)
    print(f'wrote cycle.xml + PEtab v2 problem [{len(GRID)} pts, obs {OBSERVABLES}]')


if __name__ == '__main__':
    main()
