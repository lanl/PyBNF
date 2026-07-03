#!/usr/bin/env python
"""Regenerate the PEtab v2 *priors* fixture for lesson 15.

A **developer tool**, not part of the test run. It materializes one small,
self-contained BNGL-native PEtab v2 problem whose ``parameters.tsv`` carries a
gallery of ``priorDistribution`` / ``priorParameters`` -- the *positive*
counterpart to lesson 13's ``bad_prior``, where each prior is well-formed and
imports cleanly into a PyBNF fit.

The parameters table (the star of this lesson) is written straight from
``_manifest.PRIOR_CASES`` so the committed fixture and the test's expected import
can never drift: the same manifest rows drive both the TSV here and the
``free_parameter_from_row`` assertions in ``tests/test_tutorial_priors.py``.

Usage (no simulation backend needed -- lint + import are static for a BNGL model):

    python examples/tutorial/15_petab_priors/regenerate_fixtures.py
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))          # so `_manifest` (one dir up) imports

from _manifest import PRIOR_CASES               # noqa: E402


# --------------------------------------------------------------------------- #
# The model: mass-action receptor--ligand binding, L + R <-> C. Its four
# parameters (kon, koff, R0, L0) are exactly the ones the priors table estimates,
# bound by bare id (new-era, ADR-0034). No `begin actions` block: a PEtab problem
# carries the simulation recipe in its own tables, not in the model file.
# --------------------------------------------------------------------------- #
_MODEL = """\
# Mass-action receptor--ligand binding, L + R <-> C.  The four rate/amount
# parameters below are what the PEtab parameters table puts priors on; each is
# fit by bare id (new-era binds a free parameter to the model parameter it names).
begin model
  begin parameters
    kon   0.5    # association rate    (L + R -> C)      -- log-normal prior
    koff  0.2    # dissociation rate   (C -> L + R)      -- gamma prior
    R0    30     # total receptor                        -- normal prior
    L0    50     # ligand dose (input)                   -- uniform (range only)
  end parameters
  begin molecule types
    L()
    R()
    C()
  end molecule types
  begin seed species
    L()  L0
    R()  R0
    C()  0
  end seed species
  begin observables
    Molecules  Obs_C  C()
  end observables
  begin reaction rules
    bind:   L() + R() -> C()  kon
    unbind: C() -> L() + R()  koff
  end reaction rules
end model
"""

_YAML = """\
format_version: 2.0.0
parameter_files:
  - parameters.tsv
observable_files:
  - observables.tsv
measurement_files:
  - measurements.tsv
model_files:
  binding:
    location: binding.bngl
    language: bngl
"""

_OBSERVABLES = (
    "observableId\tobservableFormula\tnoiseFormula\tnoiseDistribution\n"
    "obs_C\tObs_C\t1\tnormal\n"
)

# Illustrative complex-formation values (lint + import never simulate, so exactness
# does not matter here -- the priors table is the point of this fixture).
_MEASUREMENTS = (
    "observableId\texperimentId\ttime\tmeasurement\n"
    "obs_C\t\t0\t0\n"
    "obs_C\t\t2\t8.1\n"
    "obs_C\t\t5\t11.4\n"
    "obs_C\t\t10\t12.0\n"
)


def _parameters_tsv():
    """The parameters table, written from ``PRIOR_CASES`` (the manifest is truth).

    Every PEtab v2 parameters table has parameterId / estimate / lowerBound /
    upperBound; this one adds the optional priorDistribution / priorParameters
    columns. A blank prior cell (the ``L0`` dose) is PEtab's way of saying "no
    explicit prior" -- the validator defaults it to a uniform over the bounds.
    """
    header = ("parameterId\testimate\tlowerBound\tupperBound"
              "\tpriorDistribution\tpriorParameters\n")
    rows = [header]
    for c in PRIOR_CASES:
        rows.append(f"{c.param}\ttrue\t{c.lower:g}\t{c.upper:g}"
                    f"\t{c.distribution}\t{c.prior_params}\n")
    return ''.join(rows)


def main():
    files = {
        'problem.yaml': _YAML,
        'binding.bngl': _MODEL,
        'parameters.tsv': _parameters_tsv(),
        'observables.tsv': _OBSERVABLES,
        'measurements.tsv': _MEASUREMENTS,
    }
    for name, text in files.items():
        (_HERE / name).write_text(text)
    print(f'wrote {_HERE.name}/ ({len(files)} files, '
          f'{len(PRIOR_CASES)} priored parameters)')


if __name__ == '__main__':
    main()
