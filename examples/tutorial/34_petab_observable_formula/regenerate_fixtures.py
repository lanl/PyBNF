#!/usr/bin/env python
"""Regenerate lesson 34's PEtab v2 problem: arithmetic observableFormula in a table.

This is a **developer tool**, not part of the test run. It simulates the raw
species of ``chain.bngl`` at the true rates through bngsim, materializes each
ARITHMETIC observable formula (a ratio, a natural log, a scale by a model
parameter) with the very measurement-layer code the fit uses, and writes the
committed PEtab v2 tables -- so ``measurements.tsv`` is the model's own observed
output at the truth and a fit recovers ``k1``/``k2`` exactly.

The point of the lesson is that these formulas live in a standard PEtab
``observables.tsv`` and round-trip through import/export; the model file is never
edited (ADR-0036).

Usage (needs bngsim + BNG2.pl; set BNGPATH):

    python examples/tutorial/34_petab_observable_formula/regenerate_fixtures.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import tests.recovery_harness as H          # noqa: E402
from pybnf.pset import PSet                 # noqa: E402
from pybnf.data import Data                 # noqa: E402
from pybnf.measurement import MeasurementModel   # noqa: E402

TRUTH = {'k1': 0.7, 'k2': 0.35}
VD = 10.0                                   # fixed model parameter used by a formula
GRID = [0, 1, 2, 3, 4, 6, 8]
RAW = ('Obs_A', 'Obs_B', 'Obs_C')

# observableId -> arithmetic observableFormula (PEtab math == native measurement math here)
OBSERVABLES = {
    'frac_C': 'Obs_C / (Obs_A + Obs_B + Obs_C)',   # RATIO: the fraction that has reached C
    'log_A':  'ln(Obs_A)',                          # LOG: linearizes A's decay (pins k1)
    'conc_B': 'Obs_B / Vd',                         # SCALE: a concentration = amount / volume
}

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
  chain:
    location: chain.bngl
    language: bngl
"""


def _simulate_raw():
    """Raw Obs_A/Obs_B/Obs_C at the true rates on the integer grid, as a Data table."""
    H.require_bng2pl()
    scratch = Path(tempfile.mkdtemp(prefix='obsformula_gen_'))
    try:
        ph = scratch / 'ph.exp'
        ph.write_text('#\ttime\t' + '\t'.join(RAW) + '\n' +
                      '\n'.join(f'{t}\t' + '\t'.join('0' for _ in RAW) for t in GRID) + '\n')
        conf = H.make_newera_config(scratch, str(_HERE / 'chain.bngl'), ph,
                                    {'k1': ('uniform_var', 0.05, 5.0),
                                     'k2': ('uniform_var', 0.05, 5.0)},
                                    'timecourse', 'de', population_size=4, max_iterations=1)
        alg = H.build(conf, 'de')
        truth = PSet([v.set_value(TRUTH[v.name]) for v in alg.variables])
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
        names = [n for n, _ in sorted(data.cols.items(), key=lambda kv: kv[1])]
        return Data.from_columns(arr, names)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main():
    raw = _simulate_raw()
    consts = {'Vd': VD}
    materialized = {}
    for oid, formula in OBSERVABLES.items():
        allowed = set(RAW) | set(consts)
        materialized[oid] = np.asarray(
            MeasurementModel(oid, formula, allowed).materialize(raw, consts), dtype=float)

    (_HERE / 'parameters.tsv').write_text(
        'parameterId\testimate\tlowerBound\tupperBound\n'
        'k1\ttrue\t0.05\t5\n'
        'k2\ttrue\t0.05\t5\n')
    (_HERE / 'observables.tsv').write_text(
        'observableId\tobservableFormula\tnoiseFormula\tnoiseDistribution\n'
        + ''.join(f'{oid}\t{f}\t1\tnormal\n' for oid, f in OBSERVABLES.items()))
    rows = ''.join(f'{oid}\texp1\t{t}\t{materialized[oid][i]:.10g}\n'
                   for i, t in enumerate(GRID) for oid in OBSERVABLES)
    (_HERE / 'measurements.tsv').write_text(
        'observableId\texperimentId\ttime\tmeasurement\n' + rows)
    (_HERE / 'conditions.tsv').write_text('conditionId\n')
    (_HERE / 'experiments.tsv').write_text('experimentId\ttime\tconditionId\n')
    (_HERE / 'problem.yaml').write_text(_YAML)
    print(f'wrote PEtab v2 problem [{len(GRID)} pts, obs {tuple(OBSERVABLES)}]')


if __name__ == '__main__':
    main()
