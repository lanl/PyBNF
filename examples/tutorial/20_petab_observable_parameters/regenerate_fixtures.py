#!/usr/bin/env python
"""Regenerate the PEtab v2 tables for lesson 20 from the manifest's channel cases.

This is a **developer tool**, not part of the test run. It writes the committed
PEtab v2 problem (``problem.yaml`` + the three tables) for a two-channel readout
in which each observable carries its own estimated GAIN
(``observableParameters``) and its own estimated NOISE (``noiseParameters``). The
model (``twochannel.bngl``) is hand-written and never edited here.

The expected *import* of each channel (which native ``noise_model`` /
``observable:`` lines it must produce) lives test-side in ``_manifest.py``
(``OBS_PARAM_CASES``); this script owns only the recipe that PRODUCES the PEtab
tables from those same cases, so the two never silently drift.

Usage (no simulation backend needed -- import is static):

    python examples/tutorial/20_petab_observable_parameters/regenerate_fixtures.py
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))          # so `_manifest` (one dir up) imports

from _manifest import OBS_PARAM_CASES, OBS_PARAM_MODEL_RATES   # noqa: E402

_YAML = """\
format_version: 2.0.0
parameter_files:
  - parameters.tsv
observable_files:
  - observables.tsv
measurement_files:
  - measurements.tsv
model_files:
  twochannel:
    location: twochannel.bngl
    language: bngl
"""

# A couple of illustrative time points per channel -- import reconstructs the .exp
# from these, but the values are not what the lesson is about (the gains/noise are).
_POINTS = ((0, 0.0), (4, 30.0), (8, 45.0))


def _observables_tsv():
    rows = ['observableId\tobservableFormula\tobservablePlaceholders\t'
            'noiseFormula\tnoisePlaceholders\tnoiseDistribution']
    for c in OBS_PARAM_CASES:
        op = f'observableParameter1_{c.obs}'      # the gain placeholder in the formula
        npl = f'noiseParameter1_{c.obs}'          # the noise placeholder in the noiseFormula
        rows.append(f'{c.obs}\t{op} * {c.raw_obs}\t{op}\t{npl}\t{npl}\t{c.petab_noise}')
    return '\n'.join(rows) + '\n'


def _measurements_tsv():
    rows = ['observableId\texperimentId\ttime\tmeasurement\t'
            'observableParameters\tnoiseParameters']
    for c in OBS_PARAM_CASES:
        for t, v in _POINTS:
            # A constant-per-observable override: every row of this channel names the
            # same estimated gain and noise id (ADR-0037/0044).
            rows.append(f'{c.obs}\t\t{t}\t{v}\t{c.scale_param}\t{c.sigma_param}')
    return '\n'.join(rows) + '\n'


def _parameters_tsv():
    rows = ['parameterId\testimate\tlowerBound\tupperBound']
    for p in OBS_PARAM_MODEL_RATES:               # the model rates
        rows.append(f'{p}\ttrue\t0.01\t5')
    for c in OBS_PARAM_CASES:                     # each channel's gain + noise level
        rows.append(f'{c.scale_param}\ttrue\t0.1\t10')
        rows.append(f'{c.sigma_param}\ttrue\t0.001\t100')
    return '\n'.join(rows) + '\n'


def main():
    (_HERE / 'problem.yaml').write_text(_YAML)
    (_HERE / 'observables.tsv').write_text(_observables_tsv())
    (_HERE / 'measurements.tsv').write_text(_measurements_tsv())
    (_HERE / 'parameters.tsv').write_text(_parameters_tsv())
    print(f'wrote problem.yaml + 3 tables for {len(OBS_PARAM_CASES)} channels')


if __name__ == '__main__':
    main()
