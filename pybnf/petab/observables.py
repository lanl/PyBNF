"""PEtab v2 ``observables`` table (noise half) -> ``(NoiseModel, SigmaSource)``
(issue #407; depends on #410, ADR-0021; scales per ADR-0022).

The observables chunk of the PEtab v2 importer. It mirrors
:mod:`pybnf.petab.parameters` exactly -- the same two-adapter proof (ADR-0019): a
native ``noise_model`` ``.conf`` line and a PEtab observables row land on the
*same* internal ``(NoiseModel, SigmaSource)`` pair. Where they coincide the
abstraction is right; the PEtab table *is* the decoupled ``(family x sigma-source)``
model #410 chose (ADR-0021's Option (A)), so the mapping is a direct, column-for-
column translation rather than reverse-engineering a bundled code.

Two deliberately separated layers (the "neutral seam", as in ``parameters.py``):

* **The TSV reader** (``read_observable_table``) -- the *disposable* half: a
  dependency-free ``csv`` parse of ``observables.tsv`` into
  :class:`PetabObservableRow` records. When the later ``observableFormula`` chunk
  pulls in the ``petab`` library, this is swapped for ``petab``'s ``observable_df``
  reader with no change below.
* **The mapping** (``noise_model_from_row``) -- the *asset*: a
  :class:`PetabObservableRow` -> ``(NoiseModel, SigmaSource)``, built through the
  ordinary ``Gaussian`` / ``Laplace`` constructors so the importer lands on a pair
  bit-identical to the native ``noise_model`` surface (``objective._build_noise_spec``).

**Scope: the noise half only.** An observables row has two orthogonal noise
columns that line up one-to-one with the two axes #410 decoupled:

* ``noiseDistribution`` (``normal`` / ``laplace``) -> the distribution **family**.
* ``observableTransformation`` (``lin`` / ``log`` / ``log10``) -> the scale the
  noise is **additive on** (``additive_on``). PEtab's ``log`` is the **natural**
  log, so it maps to ``LN``, not ``LOG10`` (ADR-0022, translated at this seam).
* ``noiseFormula`` -> the **sigma-source**: a number -> ``ConstantSigma``, a bare
  noise-parameter id -> ``FreeParameterSigma``.

The deterministic prediction is taken as the distribution's **median** (PEtab v2
hardcodes this; the location axis, ADR-0011) -- trivial on ``LINEAR`` (offset 0),
and on the log scales exactly the native ``lognormal`` interpretation.

The noise-half vocabulary is **fully covered**: PEtab's ``noiseDistribution`` enum
is exactly ``{normal, laplace}`` and PyBNF maps both (the Laplace kernel landed in
#410); its ``observableTransformation`` enum is exactly ``{lin, log, log10}`` and
PyBNF maps all three (ADR-0022's ``LN`` closed the natural-log gap). So unlike the
parameters chunk (five catalog-parity prior families PyBNF lacks), there is **no**
unsupported-family boundary here. The only deferred capability is a non-trivial
``noiseFormula`` *expression* -- the sympy layer where the ``petab`` library earns
its keep -- surfaced as an explicit ``NotImplementedError`` so the boundary is in
code, not silent. Malformed rows (unknown ``noiseDistribution`` /
``observableTransformation`` spelling -- e.g. a future PEtab version's new value,
a missing ``noiseFormula``, a blank ``observableId``) raise ``PybnfError``.

The ``observableFormula`` (the model-output expression) is the **deferred sibling
half** -- a separate, later chunk that adopts the ``petab`` sympy layer. It is
recorded on :class:`PetabObservableRow` so that chunk reuses this reader, but the
noise asset neither reads nor validates it (real ``observableFormula``s are
non-trivial expressions, so coupling that boundary here would make the noise asset
raise on nearly every real PEtab problem).
"""

import csv
import re
from dataclasses import dataclass

from ..noise import (LINEAR, LN, LOG10, MEDIAN, ConstantSigma, FreeParameterSigma,
                     Gaussian, Laplace)
from ..printing import PybnfError

# PEtab v2 noiseDistribution -> the PyBNF NoiseModel family class. Both are
# supported (the Laplace kernel landed in #410, ADR-0021), so there is no
# "known-PEtab-but-PyBNF-lacks" set the way parameters.py has for prior families.
_PETAB_DISTRIBUTION_TO_FAMILY = {
    'normal': Gaussian,
    'laplace': Laplace,
}

# PEtab v2 observableTransformation -> the additive-noise scale (ADR-0022). PEtab's
# ``log`` is the NATURAL log (-> LN); ``log10`` is base-10 (-> LOG10). This is the
# vocabulary translation at the seam, exactly like parameters.py converting PEtab's
# natural-log priors to PyBNF's log10.
_PETAB_TRANSFORMATION_TO_SCALE = {
    'lin': LINEAR,
    'log': LN,
    'log10': LOG10,
}

# A single bare identifier (a noise-parameter id) -- anything else with operators,
# calls, or whitespace is an expression for the deferred sympy layer.
_IDENTIFIER = re.compile(r'[A-Za-z_]\w*\Z')


@dataclass(frozen=True)
class PetabObservableRow:
    """One row of a PEtab v2 observables table, in PyBNF's neutral vocabulary.

    The dependency-free seam between the (disposable) TSV reader and the (asset)
    mapping: the mapping never depends on *how* the row was read, so a future
    ``petab``-library adoption feeds it by constructing these from
    ``Problem.observable_df`` records.

    ``observable_transformation`` / ``noise_distribution`` are ``None`` when the
    column is absent or blank; the mapping applies the PEtab v2 defaults (``lin`` /
    ``normal``). ``observable_formula`` -- the model-output expression -- is
    recorded for the deferred ``observableFormula`` chunk but is **not** consumed by
    the noise mapping (this is the noise half only).
    """

    observable_id: str
    observable_formula: str | None = None
    observable_transformation: str | None = None
    noise_formula: str | None = None
    noise_distribution: str | None = None


# ---------------------------------------------------------------------------
# Mapping: PetabObservableRow -> (NoiseModel, SigmaSource) (the asset)
# ---------------------------------------------------------------------------

def noise_model_from_row(row):
    """Map one PEtab v2 observables row's **noise half** to an
    ``(NoiseModel, SigmaSource)`` pair, bit-identical to the native ``noise_model``
    ``.conf`` surface (ADR-0021, ADR-0023).

    ``noiseDistribution`` selects the family (``normal`` -> ``Gaussian``,
    ``laplace`` -> ``Laplace``); ``observableTransformation`` selects the scale the
    noise is additive on (``lin`` -> ``LINEAR``, ``log`` -> ``LN``, ``log10`` ->
    ``LOG10``); the prediction is the median (PEtab default). ``noiseFormula``
    becomes the sigma-source: a number -> ``ConstantSigma``, a bare noise-parameter
    id -> ``FreeParameterSigma``.

    Raises ``NotImplementedError`` for a non-trivial ``noiseFormula`` expression
    (the deferred sympy layer) and ``PybnfError`` for a malformed row (unknown
    ``noiseDistribution`` / ``observableTransformation`` spelling, missing
    ``noiseFormula``).
    """
    dist = row.noise_distribution or 'normal'
    transform = row.observable_transformation or 'lin'

    if dist not in _PETAB_DISTRIBUTION_TO_FAMILY:
        raise PybnfError(
            f"Observable '{row.observable_id}': unknown PEtab noiseDistribution "
            f"{dist!r} (expected one of {sorted(_PETAB_DISTRIBUTION_TO_FAMILY)}).")
    if transform not in _PETAB_TRANSFORMATION_TO_SCALE:
        raise PybnfError(
            f"Observable '{row.observable_id}': unknown PEtab observableTransformation "
            f"{transform!r} (expected one of {sorted(_PETAB_TRANSFORMATION_TO_SCALE)}).")

    family_cls = _PETAB_DISTRIBUTION_TO_FAMILY[dist]
    scale = _PETAB_TRANSFORMATION_TO_SCALE[transform]
    noise = family_cls(additive_on=scale, location=MEDIAN)
    source = _sigma_source_from_noise_formula(row.noise_formula, row.observable_id)
    return noise, source


def _sigma_source_from_noise_formula(formula, observable_id):
    """Map a PEtab ``noiseFormula`` to its ``SigmaSource``.

    A numeric literal -> ``ConstantSigma`` (the native ``fix_at`` source); a single
    bare identifier (the noise-parameter id) -> ``FreeParameterSigma`` (the native
    ``fit`` source; the id is passed through verbatim as the free-parameter name,
    bound to a declared ``FreeParameter`` by the later measurements/conditions
    chunk). PEtab never expresses a per-point data column, so this does not produce
    ``DataColumnSigma`` -- that ``_SD``-column source is PyBNF-native (ADR-0023).
    A non-trivial expression is the deferred sympy layer (NotImplementedError).
    """
    if formula is None or formula.strip() == '':
        raise PybnfError(
            f"Observable '{observable_id}' is missing a noiseFormula: the noise "
            f"half needs a noise parameter (a number or a noise-parameter id).")
    formula = formula.strip()

    try:
        return ConstantSigma(float(formula))
    except ValueError:
        pass

    if _IDENTIFIER.match(formula):
        return FreeParameterSigma(formula)

    raise NotImplementedError(
        f"Observable '{observable_id}': noiseFormula {formula!r} is an expression, "
        f"not a number or a bare noise-parameter id. Evaluating PEtab "
        f"noise/observable formulae needs the sympy layer (the deferred "
        f"observableFormula chunk, #407), which adopts the petab library.")


# ---------------------------------------------------------------------------
# Table-level helpers
# ---------------------------------------------------------------------------

def noise_models_from_table(rows):
    """Map an observables table to a ``{observable_id: (NoiseModel, SigmaSource)}``
    override map -- exactly the ``LikelihoodObjective(overrides=...)`` map the native
    ``noise_model`` lines produce (ADR-0021), the two-adapter proof at table level.
    """
    return {row.observable_id: noise_model_from_row(row) for row in rows}


def noise_models_from_file(path):
    """Read ``observables.tsv`` at ``path`` and map it to the noise override map."""
    return noise_models_from_table(read_observable_table(path))


# ---------------------------------------------------------------------------
# TSV reader (the disposable half of the seam)
# ---------------------------------------------------------------------------

def read_observable_table(path):
    """Read a PEtab v2 ``observables.tsv`` into :class:`PetabObservableRow` records.

    Dependency-free (stdlib ``csv``). Unknown extra columns (e.g.
    ``observableName``) are tolerated and ignored.
    """
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        return [_row_from_record(rec) for rec in reader]


def _row_from_record(rec):
    oid = rec.get('observableId')
    if oid is None or oid.strip() == '':
        raise PybnfError("PEtab observables row is missing an observableId.")
    return PetabObservableRow(
        observable_id=oid.strip(),
        observable_formula=_parse_str(rec.get('observableFormula')),
        observable_transformation=_parse_str(rec.get('observableTransformation')),
        noise_formula=_parse_str(rec.get('noiseFormula')),
        noise_distribution=_parse_str(rec.get('noiseDistribution')),
    )


def _parse_str(s):
    if s is None:
        return None
    s = s.strip()
    return s or None
