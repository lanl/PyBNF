"""PEtab v2 ``observables`` table (noise half) -> ``(NoiseModel, SigmaSource)``
(issue #407; depends on #410, ADR-0021; scales per ADR-0022).

The observables chunk of the PEtab v2 importer. It mirrors
:mod:`pybnf.petab.parameters` exactly -- the same two-adapter proof (ADR-0019): a
native ``noise_model`` ``.conf`` line and a PEtab observables row land on the
*same* internal ``(NoiseModel, SigmaSource)`` pair where the native surface can
express it. PyBNF's noise engine (ADR-0021) is richer than PEtab v2's noise
vocabulary, so PEtab maps onto a subset of it.

Two deliberately separated layers (the "neutral seam", as in ``parameters.py``):

* **The TSV reader** (``read_observable_table``) -- the *disposable* half: a
  dependency-free ``csv`` parse of ``observables.tsv`` into
  :class:`PetabObservableRow` records. When the later ``observableFormula`` chunk
  pulls in the ``petab`` library, this is swapped for ``petab``'s ``observable_df``
  reader with no change below.
* **The mapping** (``noise_model_from_row``) -- the *asset*: a
  :class:`PetabObservableRow` -> ``(NoiseModel, SigmaSource)``, built through the
  ordinary ``Gaussian`` / ``Laplace`` constructors.

**Scope: the noise half only.** The PEtab v2 spec (verified against the current
spec, not the v1 shape -- see the note below) gives the noise model through two
columns:

* ``noiseDistribution`` -- a single column carrying **both** the distribution
  family **and** the scale its noise is additive on. PEtab v2 allows exactly
  ``normal`` / ``log-normal`` / ``laplace`` / ``log-laplace`` (default ``normal``).
  PEtab's log is the **natural** log, so the log forms map to ``LN`` (ADR-0022),
  not ``LOG10``.
* ``noiseFormula`` -- the **sigma-source** (the noise distribution's scale
  parameter): a number -> ``ConstantSigma``, a bare noise-parameter id ->
  ``FreeParameterSigma``.

The mapping (all with the prediction taken as the distribution's **median** --
PEtab v2 specifies this for every noise distribution; the location axis, ADR-0011):

=================  ========================================
noiseDistribution  PyBNF (NoiseModel)
=================  ========================================
``normal``         ``Gaussian(LINEAR, MEDIAN)``
``log-normal``     ``Gaussian(LN, MEDIAN)``   (natural log)
``laplace``        ``Laplace(LINEAR, MEDIAN)``
``log-laplace``    ``Laplace(LN, MEDIAN)``    (natural log)
=================  ========================================

**Spec note (PEtab v2 vs v1).** PEtab v2 *removed* the separate
``observableTransformation`` column (``lin`` / ``log`` / ``log10``) and folded it
into ``noiseDistribution`` as the ``log-`` prefixes above; and v2's log is natural
(base e), with no ``log10`` form. (An earlier draft of this adapter encoded the v1
shape; this is the corrected v2 mapping.)

**The two-adapter equivalence is exact for the linear families and ``laplace``,
structural for the natural-log families.** ``laplace`` matches the native
``laplace`` token exactly (``Laplace(LINEAR, MEDIAN)``), and ``normal`` matches the
native ``normal`` token exactly too (``Gaussian(LINEAR, MEDIAN)``): native ``normal``
now also defaults to ``MEDIAN`` (ADR-0031). The natural-log families have
**no native ``.conf`` token** -- the native ``lognormal`` token is log10, and the
native surface has no natural-log family -- so ``log-normal`` / ``log-laplace`` are
validated structurally and against the kernels' analytic NLL rather than a native
config line. That PyBNF can represent natural-log noise the native surface does not
expose is fine: the engine is the superset, the native grammar a convenience.

The noise mapping is **complete** for PEtab v2: every one of the four
``noiseDistribution`` values maps with no gaps (the Laplace kernel landed in #410,
the ``LN`` scale in ADR-0022). The only deferred capability is a non-trivial
``noiseFormula`` *expression* -- the sympy layer where the ``petab`` library earns
its keep -- surfaced as an explicit ``NotImplementedError``. Malformed rows (an
unknown ``noiseDistribution`` spelling -- e.g. a future PEtab value -- a missing
``noiseFormula``, a blank ``observableId``) raise ``PybnfError``.

The ``observableFormula`` (the model-output expression) and the
``observablePlaceholders`` / ``noisePlaceholders`` columns are the **deferred
sibling half** -- a separate, later chunk that adopts the ``petab`` sympy layer.
``observable_formula`` is recorded on :class:`PetabObservableRow` so that chunk
reuses this reader, but the noise asset neither reads nor validates it (real
``observableFormula`` values are non-trivial expressions, so coupling that boundary
here would make the noise asset raise on nearly every real PEtab problem).
"""

import csv
import re
from dataclasses import dataclass

from ..noise import (LINEAR, LN, MEDIAN, ConstantSigma, FreeParameterSigma,
                     Gaussian, Laplace)
from ..printing import PybnfError
from ._tsv import num, write_tsv

# A per-measurement placeholder symbol (``noiseParameter1_obs_y`` /
# ``observableParameter2_obs_y``): the full token a row-varying noiseFormula references,
# extracted to declare the noisePlaceholders slot the measurements' ``noiseParameters``
# column binds per row (ADR-0045). Mirrors ``import_._PLACEHOLDER_SYMBOL``.
_PLACEHOLDER_SYMBOL = re.compile(r'(?:observable|noise)Parameter\d+_\w+')

# The same placeholder split into its ``<kind>Parameter<n>`` head and ``_<observableId>``
# suffix, so the suffix can be **retargeted** to the regenerated observableId: PEtab requires a
# placeholder to be ``noiseParameter${n}_${observableId}``, but a conf carries the source
# problem's suffix (a model function ``y`` is named ``obs_y`` there, ``func_y`` on re-export),
# so the noiseFormula's placeholder is rewritten to match the row's observableId (ADR-0045).
_PLACEHOLDER_HEAD = re.compile(r'((?:observable|noise)Parameter\d+)_\w+')

# PEtab v2 noiseDistribution -> (PyBNF NoiseModel family class, additive-noise
# scale). The single PEtab column carries both axes: the family (Gaussian/Laplace)
# and whether the noise is additive on the linear or the natural-log scale. PEtab's
# log is natural (base e) -> LN (ADR-0022), never log10. Both families are supported
# (the Laplace kernel landed in #410), so every value maps -- there is no
# "known-PEtab-but-PyBNF-lacks" gap the way parameters.py has for prior families.
_PETAB_NOISE_DISTRIBUTION = {
    'normal':      (Gaussian, LINEAR),
    'log-normal':  (Gaussian, LN),
    'laplace':     (Laplace,  LINEAR),
    'log-laplace': (Laplace,  LN),
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

    ``noise_distribution`` is ``None`` when the column is absent or blank; the
    mapping applies the PEtab v2 default (``normal``). ``observable_formula`` -- the
    model-output expression -- is recorded for the deferred ``observableFormula``
    chunk but is **not** consumed by the noise mapping (this is the noise half only).
    """

    observable_id: str
    observable_formula: str | None = None
    noise_formula: str | None = None
    noise_distribution: str | None = None
    # The semicolon-delimited placeholder ids (PEtab v2 ``observablePlaceholders`` /
    # ``noisePlaceholders``). The importer does not yet consume them (the deferred
    # formula half, ADR-0023); the exporter sets ``noise_placeholders`` to declare the
    # per-point ``_SD`` noise slot a ``noiseParameters`` override binds to (ADR-0025).
    observable_placeholders: str | None = None
    noise_placeholders: str | None = None


# ---------------------------------------------------------------------------
# Mapping: PetabObservableRow -> (NoiseModel, SigmaSource) (the asset)
# ---------------------------------------------------------------------------

def noise_model_from_row(row):
    """Map one PEtab v2 observables row's **noise half** to an
    ``(NoiseModel, SigmaSource)`` pair (ADR-0021, ADR-0023).

    ``noiseDistribution`` selects the family and the additive scale together
    (``normal`` -> ``Gaussian(LINEAR)``, ``log-normal`` -> ``Gaussian(LN)``,
    ``laplace`` -> ``Laplace(LINEAR)``, ``log-laplace`` -> ``Laplace(LN)``); the
    prediction is the median (PEtab default). ``noiseFormula`` becomes the
    sigma-source: a number -> ``ConstantSigma``, a bare noise-parameter id ->
    ``FreeParameterSigma``.

    Raises ``NotImplementedError`` for a non-trivial ``noiseFormula`` expression
    (the deferred sympy layer) and ``PybnfError`` for a malformed row (unknown
    ``noiseDistribution`` spelling, missing ``noiseFormula``).
    """
    dist = row.noise_distribution or 'normal'
    if dist not in _PETAB_NOISE_DISTRIBUTION:
        raise PybnfError(
            f"Observable '{row.observable_id}': unknown PEtab noiseDistribution "
            f"{dist!r} (expected one of {sorted(_PETAB_NOISE_DISTRIBUTION)}).")

    family_cls, scale = _PETAB_NOISE_DISTRIBUTION[dist]
    noise = family_cls(additive_on=scale, location=MEDIAN)
    source = _sigma_source_from_noise_formula(row.noise_formula, row.observable_id)
    # The per-observable spec is (family, {param: source}) -- one entry under the
    # family's primary parameter name (ADR-0058). PEtab's two families are both
    # single-parameter (gaussian/sigma, laplace/scale).
    return noise, {noise.noise_params[0]: source}


def _sigma_source_from_noise_formula(formula, observable_id):
    """Map a PEtab ``noiseFormula`` (the noise distribution's scale parameter) to its
    ``SigmaSource``.

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
# Export: a fitted model column -> PetabObservableRow (the reverse asset; ADR-0025)
# ---------------------------------------------------------------------------

# A fitted ``.exp`` column is a BNGL observable or a BNGL function; the PEtab
# observableId wraps it with a prefix that keeps the PEtab-id namespace disjoint from
# the model-entity namespace, so ``observableFormula`` can reference the bare model
# name without ever colliding with a PEtab observableId (ADR-0025). A ``measurement``
# column is a conf-declared measurement model (ADR-0036): its id is already a PEtab
# observableId (the conf carries it from an earlier import), so it takes no prefix and
# its ``observableFormula`` is the conf formula verbatim.
_PETAB_OBSERVABLE_PREFIX = {'observable': 'obs_', 'function': 'func_', 'measurement': ''}

_OBSERVABLE_COLUMNS = [
    'observableId', 'observableFormula', 'noiseFormula', 'noiseDistribution',
    'noisePlaceholders']

# Appended after noisePlaceholders only when some row declares a row-varying observable
# placeholder (ADR-0045) -- so a job with none stays byte-identical to the pre-row-varying
# output (the byte-equal export round-trip oracle).
_OBSERVABLE_PLACEHOLDERS_COLUMN = 'observablePlaceholders'


def petab_observable_row(model_name, kind, noise_distribution, noise_source,
                         observable_formula=None):
    """Map one fitted BNGL column to a :class:`PetabObservableRow` (ADR-0025).

    ``model_name`` is the BNGL observable/function name (an ``.exp`` column header);
    ``kind`` is ``'observable'`` or ``'function'``. The ``observableId`` is the
    prefixed name (``obs_<name>`` / ``func_<name>``). ``kind == 'measurement'`` is a
    conf-declared measurement model (ADR-0036): the ``observableId`` is ``model_name``
    verbatim (no prefix -- it is already a PEtab id) and ``observable_formula`` (the conf
    formula) is required.

    ``observableFormula`` is the **bare model name** by default: a BNGL function
    (including a function of a function) is carried verbatim in the model file and
    evaluated there, so the table only references it by name and the export is lossless
    and ``petab``-free. The exporter's opt-in *inlining* mode (ADR-0035) overrides it for
    a **function** column by passing ``observable_formula`` (the function body translated
    to PEtab math), so the emitted problem carries its own measurement model and round-trips
    against the importer's synthesis. An observable column is never inlined (an observable
    is a model species/group, not an algebraic expression), so callers pass
    ``observable_formula`` only for ``kind == 'function'``.

    ``noise_distribution`` is the PEtab family the job's objective maps to (ADR-0023
    reversed: ``gaussian`` -> ``normal``, ``laplace`` -> ``laplace``). ``noise_source``
    is the PEtab representation of the objective's sigma source (ADR-0021 reversed),
    one of:

    * ``('placeholder', None)`` -- a per-point ``_SD`` data column: a declared noise
      **placeholder** (``noiseFormula`` = ``noisePlaceholders`` =
      ``noiseParameter1_<id>``) whose per-point value the measurements'
      ``noiseParameters`` column supplies (``chi_sq``).
    * ``('constant', value)`` -- a fixed sigma written inline as a numeric
      ``noiseFormula`` with no placeholder: a ``fix_at`` constant (``sos`` -> 1,
      ``sod`` -> 1) or an observable's column mean (``ave_norm_sos``).
    * ``('formula', expr)`` -- an expression sigma (``FormulaSigma``, ADR-0044/0045): the
      PEtab-math ``noiseFormula`` over free-parameter ids + constants, emitted verbatim with
      no placeholder. The inverse of the importer's expression-``noiseFormula`` classification,
      so a whole-fit ``FormulaSigma`` round-trips.
    * ``('free_param', id)`` -- a free-parameter (estimated) sigma (``FreeParameterSigma``,
      #439): the bare noise-parameter ``id`` as the ``noiseFormula``, no placeholder, declared
      estimated in the parameter table and an observation-layer nuisance (not a model entity).
      The importer reads a bare-id ``noiseFormula`` back to a ``fit`` source (ADR-0044), so a
      per-observable estimated sigma round-trips.
    * ``('per_measurement', expr)`` -- a **row-varying** placeholder sigma
      (``PerMeasurementFormulaSigma``, ADR-0045): the ``noiseFormula`` expression carries a
      per-measurement placeholder (``noiseParameter1_<id>``) whose value differs row to row,
      supplied by the measurements' ``noiseParameters`` column. The ``noiseFormula`` is emitted
      verbatim and the placeholder(s) it references declared in ``noisePlaceholders`` -- the
      inverse of the importer routing a row-varying ``noiseParameters`` id to the binding table.
    """
    try:
        prefix = _PETAB_OBSERVABLE_PREFIX[kind]
    except KeyError:
        raise PybnfError(
            f"Column '{model_name}': unknown kind {kind!r} (expected 'observable', "
            f"'function', or 'measurement').")
    if kind == 'measurement' and observable_formula is None:
        raise PybnfError(
            f"Measurement-model column '{model_name}' requires an observableFormula "
            f"(the conf measurement-model formula, ADR-0036).")
    observable_id = prefix + model_name

    # A row-varying observable scale (ADR-0045) keeps an observableParameter placeholder in the
    # observableFormula (a measurement-model id carries through verbatim, so it already matches
    # this observableId). Declare it in observablePlaceholders -- the observable-side mirror of
    # noisePlaceholders -- so PEtab v2 binds the measurements' observableParameters column to it
    # (without the declaration petab reads the placeholder as a missing free parameter).
    formula_out = observable_formula if observable_formula is not None else model_name
    obs_placeholders = [p for p in _PLACEHOLDER_SYMBOL.findall(formula_out)
                        if p.startswith('observable')]
    observable_placeholders = ';'.join(obs_placeholders) if obs_placeholders else None

    source_kind, source_value = noise_source
    if source_kind == 'placeholder':
        noise_formula = f'noiseParameter1_{observable_id}'
        noise_placeholders = noise_formula
    elif source_kind == 'constant':
        noise_formula = num(source_value)
        noise_placeholders = None
    elif source_kind == 'formula':
        # An expression sigma (FormulaSigma, ADR-0044/0045): the noiseFormula is the PEtab-math
        # expression verbatim, no placeholder slot (its free symbols are PEtab parameter ids).
        noise_formula = source_value
        noise_placeholders = None
    elif source_kind == 'free_param':
        # A free-parameter (estimated) sigma (FreeParameterSigma, #439): the noiseFormula is the
        # bare noise-parameter id, declared estimated in the parameter table and constant across
        # the observable's measurements (no per-measurement placeholder). The importer reads a
        # bare-id noiseFormula back to a 'fit' source (ADR-0044), so it round-trips.
        noise_formula = source_value
        noise_placeholders = None
    elif source_kind == 'per_measurement':
        # A row-varying placeholder sigma (PerMeasurementFormulaSigma, ADR-0045): the
        # noiseFormula keeps its per-measurement placeholder, retargeted to THIS observableId
        # (PEtab requires noiseParameter${n}_${observableId}; the conf carries the source
        # problem's suffix, which the exporter's prefix regenerates). The declared
        # noisePlaceholders are the retargeted names the measurements' noiseParameters bind to.
        noise_formula = _PLACEHOLDER_HEAD.sub(rf'\1_{observable_id}', source_value)
        placeholders = _PLACEHOLDER_SYMBOL.findall(noise_formula)
        noise_placeholders = ';'.join(placeholders) if placeholders else None
    else:
        raise PybnfError(
            f"Observable '{observable_id}': unknown noise source kind {source_kind!r} "
            f"(expected 'placeholder', 'constant', 'formula', 'free_param', or "
            f"'per_measurement').")

    return PetabObservableRow(
        observable_id=observable_id,
        observable_formula=formula_out,
        noise_formula=noise_formula,
        noise_distribution=noise_distribution,
        observable_placeholders=observable_placeholders,
        noise_placeholders=noise_placeholders,
    )


def write_observable_table(rows, path):
    """Write observable ``rows`` to ``path`` as a PEtab v2 ``observables.tsv``.

    The optional ``observablePlaceholders`` column (a row-varying observable scale's declared
    placeholder, ADR-0045) is appended only when some row carries one, so a job with none stays
    byte-identical to the pre-row-varying output (the byte-equal export round-trip oracle)."""
    include_obs_ph = any(r.observable_placeholders for r in rows)
    columns = _OBSERVABLE_COLUMNS + (
        [_OBSERVABLE_PLACEHOLDERS_COLUMN] if include_obs_ph else [])
    records = []
    for r in rows:
        rec = [r.observable_id, r.observable_formula, r.noise_formula or '',
               r.noise_distribution or '', r.noise_placeholders or '']
        if include_obs_ph:
            rec.append(r.observable_placeholders or '')
        records.append(rec)
    write_tsv(path, columns, records)


# ---------------------------------------------------------------------------
# TSV reader (the disposable half of the seam)
# ---------------------------------------------------------------------------

def read_observable_table(path):
    """Read a PEtab v2 ``observables.tsv`` into :class:`PetabObservableRow` records.

    Dependency-free (stdlib ``csv``). ``noisePlaceholders`` is recorded (it marks a
    named noiseFormula placeholder whose value the measurements' ``noiseParameters``
    column supplies -- a per-observable estimated sigma, ADR-0037). Other unknown extra
    columns (e.g. ``observableName``, ``observablePlaceholders``) are tolerated and ignored.
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
        noise_formula=_parse_str(rec.get('noiseFormula')),
        noise_distribution=_parse_str(rec.get('noiseDistribution')),
        noise_placeholders=_parse_str(rec.get('noisePlaceholders')),
    )


def _parse_str(s):
    if s is None:
        return None
    s = s.strip()
    return s or None
