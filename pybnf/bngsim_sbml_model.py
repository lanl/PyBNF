"""Optional SBML simulation using bngsim."""


import copy
import hashlib
import logging
import os
import secrets
import tempfile
from dataclasses import dataclass

import numpy as np

from .data import Data, OutputSensitivities, stack_scan_sensitivities
from .gradient import derivative
from .gradient.routing import IC, PARAM, SeedTerm
from .printing import PybnfError
from .pset import (
    FailedSimulationError,
    Model,
    ModelError,
    MutationSet,
    ParamScan,
    TimeCourse,
)
from ._seed import resolve_action_seed


_SUPPORTED_INTEGRATORS = ('cvode', 'gillespie')


logger = logging.getLogger(__name__)


# bngsim's CVODE tolerance defaults (``Simulator.__init__``): rtol = atol = 1e-8.
# They are BNG2.pl's, and on the BNGL path that is exactly right -- parity with
# BNG2.pl is what the net backend is measured against, and a BNGL author who needs
# something else writes ``atol=>...`` in the actions block. They are read off the
# constructed Simulator when it exposes them, so these only stand in for a build
# that stops publishing them; see :func:`_effective_tolerances`.
_BNGSIM_DEFAULT_RTOL = 1e-8
_BNGSIM_DEFAULT_ATOL = 1e-8

# The tightest absolute tolerance the scale derivation will reach for on its own
# (#546). CVODE's error test is ``|e_i| <= rtol*|y_i| + atol``, so an atol below
# ~1e-16 is already past double precision's resolution of a state of order one:
# tightening further can only matter for states far below one, where the relative
# term governs anyway, and it buys that at an integration cost with no bound. A
# model that genuinely lives down there says so with ``sbml_atol``, and hears about
# it from the log line in :func:`_derive_atol` rather than finding out from its
# gradient.
_DERIVED_ATOL_FLOOR = 1e-16


# The two non-numeric settings ``sbml_atol`` accepts (#557). Both are opt-ins to the
# derivation rather than replacements for it, which is what separates them from a pinned
# number: a number states the tolerance, these state how far the model's own state is
# allowed to state it.
#
# ``auto``     -- trust the derivation in **both** directions, i.e. lift the upper clamp
#                 that lets it only ever tighten (:func:`_derive_atol`).
# ``tracking`` -- ``auto``'s vector as a *ceiling*, re-evaluated against the state being
#                 integrated rather than against t = 0 (lanl/bngsim#213's
#                 ``CVodeWFtolerances``). Takes an optional depth in decades; bngsim's
#                 own default (12) applies when it is not stated.
_ATOL_AUTO = 'auto'
_ATOL_TRACKING = 'tracking'


def parse_atol_setting(atol):
    """Split an ``sbml_atol`` value into ``(pinned, mode, decades)`` (#557).

    One shape authority for a key that now carries three different kinds of statement,
    so ``config.py`` (which rejects a bad one with a pointed message at config load) and
    the model constructor (which is reachable directly, and must not defer a malformed
    setting to the first ``run``) cannot disagree about what a value means.

    * a **number** -- ``(float, None, None)``. The documented off-switch: it pins the
      scalar and switches every derivation, vector and tracking alike, off.
    * ``auto`` -- ``(None, 'auto', None)``.
    * ``tracking`` / ``tracking <decades>`` -- ``(None, 'tracking', None | float)``.
      ``None`` decades means "bngsim's default depth", which is deliberately not copied
      here: it is a measured property of that mechanism (12, where its own accuracy stops
      improving) and belongs to the library that measured it.
    * ``None`` -- ``(None, None, None)``, the unset default: derive, and only tighten.

    A numeric *string* reads as the number, because a value can reach a model from
    somewhere other than the conf grammar (a hand-built dict, an importer). Raises
    ``ValueError`` with a message naming the value; callers translate it into their own
    error type rather than each re-deciding what the legal shapes are.
    """
    if atol is None:
        return None, None, None
    if isinstance(atol, bool):
        raise ValueError(f'expected a positive number, "auto", or "tracking [decades]"; got {atol!r}')
    if isinstance(atol, (int, float)):
        return _positive_atol(atol), None, None
    tokens = str(atol).strip().split()
    if not tokens:
        raise ValueError('expected a positive number, "auto", or "tracking [decades]"; got an empty value')
    mode = tokens[0].lower()
    # The modes are tried before the number, not because the order could go either way --
    # neither spelling parses as a float -- but so a malformed `tracking <junk>` reports
    # what is wrong with its depth rather than falling through to "not a number".
    if mode == _ATOL_AUTO and len(tokens) == 1:
        return None, _ATOL_AUTO, None
    if mode == _ATOL_TRACKING and len(tokens) <= 2:
        if len(tokens) == 1:
            return None, _ATOL_TRACKING, None
        try:
            decades = float(tokens[1])
        except ValueError:
            raise ValueError(
                f'"tracking" takes an optional depth in decades; got {tokens[1]!r}') from None
        if not np.isfinite(decades) or decades < 0.0:
            raise ValueError(
                f'"tracking" decades must be finite and >= 0; got {decades!r}. A tracking '
                'tolerance with no floor is pure relative error control, which has no '
                'weight to give a species sitting at exactly zero.')
        return None, _ATOL_TRACKING, decades
    if len(tokens) == 1:
        try:
            number = float(tokens[0])
        except ValueError:
            pass
        else:
            return _positive_atol(number), None, None
    raise ValueError(
        f'expected a positive number, "auto", or "tracking [decades]"; got {str(atol)!r}')


def _positive_atol(value):
    """``float(value)``, refusing what CVODE cannot take as an absolute tolerance."""
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f'an absolute tolerance must be a finite positive number; got {value!r}')
    return value


def parse_species_atol_setting(species_atol):
    """Normalize a per-model ``species_atol`` map into ``{name: float}`` (#586).

    The sibling of :func:`parse_atol_setting` for the hand-written per-species vector the
    new-era ``model:`` declaration carries (ADR-0116), and it exists for the same reason:
    ``config.py`` rejects a bad one at config load and the model constructor is reachable
    directly, so one function has to own what a value means or the two will disagree.

    ``None`` (and an empty map) is ``None`` -- nothing stated. Otherwise every entry is
    checked to be a finite **positive** number. Strictly positive rather than bngsim's
    ``>= 0``: zero is a legal entry of a plain ``CVodeSVtolerances`` vector but says that
    species has no absolute error control at all, and it is refused outright as a
    ``TrackingAtol`` ceiling -- so accepting it here would make ``species_atol: X 0``
    parse, integrate, and then fail the moment the same model added ``atol: tracking``.
    One rule that holds under every setting is worth more than the one value it declines.

    Species NAMES are not checked here: they belong to the model, and the check needs the
    species list bngsim will integrate under (:meth:`_check_stated_species_atol`). Raises
    ``ValueError`` naming the offending species; callers translate it into their own
    error type rather than each re-deciding what the legal shapes are.
    """
    if species_atol is None:
        return None
    if not isinstance(species_atol, dict):
        raise ValueError(
            'expected a mapping of species name to absolute tolerance; got '
            f'{type(species_atol).__name__}')
    normalized = {}
    for name, value in species_atol.items():
        # A numeric *string* reads as the number, for parse_atol_setting's reason: a value
        # can reach a model from somewhere other than the conf grammar. ``True`` does not,
        # because ``float(True)`` is 1.0 and would pass silently.
        if isinstance(value, bool):
            raise ValueError(f"species '{name}': expected a positive number; got {value!r}")
        try:
            normalized[str(name)] = _positive_atol(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"species '{name}': {exc}") from None
    return normalized or None


def _derive_atol(scale, rtol, default_atol, *, loosen=False, model_name=None, warned=None):
    """The absolute tolerance a model whose state lives at ``scale`` needs (#546).

    ``rtol * scale``, clamped into ``[_DERIVED_ATOL_FLOOR, default_atol]``. The upper
    clamp is what makes this safe to apply unconditionally: the derivation can only
    *tighten*, never loosen, so a model whose species are of order one (or larger) keeps
    the backend default bit-for-bit and every existing trajectory is unchanged.

    **``loosen`` lifts that upper clamp, and lifts nothing else** (#557,
    ``sbml_atol = auto``). Only-ever-tighten is a no-regression rule, not a statement
    about the model, and the phrase carrying it -- "of order one *or larger*" -- is doing
    a great deal of work: a model whose species all sit far above one has a real,
    computable tolerance need, and the derivation computes it and then discards it.
    ``Weber_BMC2015``'s seven species live at 1.24e+02 .. 4.21e+07, median 4.67e+05, so it
    asks for ``4.665e-03`` and is handed ``1e-8``, 5.7 decades tighter. What that costs is
    the error test read forwards: an ``atol`` far under the state's own magnitude is one
    the absolute term can never reach, so the integrator is held to a *relative* accuracy
    far tighter than the ``rtol`` that is supposed to govern it, and pays in steps for
    resolution nobody asked for. The forward-*sensitivity* solve pays most, since CVODES
    derives its sensitivity tolerances from the state ones.

    The clamp binds on 10 of the 22 subset-I slugs with a readable nominal state, and nine
    of those ten integrate their box samples at the clamped value today -- they are merely
    charged 1.5x-2.6x the integrator steps for it (ADR-0114). That is the whole argument
    for the opt-in: the ten are the *scope* of the problem, not a mandate to move nine
    working results.

    The rule is CVODE's own error test read backwards. It weights each state by
    ``rtol*|y_i| + atol``, so ``atol`` is a declaration that values below it are noise --
    a statement about the *model's units*, and 1e-8 is only true of a model whose states
    are of order one. Giordano_Nature2020 is a population-*fraction* epidemic model whose
    species sit at 1.7e-8..1, median 3.7e-7, so at the default the absolute term buries
    the relative one across the whole early trajectory: no significant digits at all. The
    forward-sensitivity solve carries fewer still -- CVODES scales the state tolerances
    by the parameter magnitude for the sensitivity vectors, which for a fitted rate of
    ~0.1 makes their absolute floor ten times *looser* than the states'. Setting ``atol``
    to ``rtol * scale`` puts the absolute term at the relative one at the model's own
    magnitude, which is the condition for the relative test to be the one that governs.

    ``scale`` of ``None`` (nothing positive to measure) returns ``default_atol``. When
    the floor binds -- the model asks for more resolution than the derivation will reach
    for on its own -- it is logged once per model via the ``warned`` set, because that is
    the case where the tolerance is still not small enough and nothing else would say so.
    The floor is unconditional: ``loosen`` is about the ceiling, and a model asking for
    ``1e-20`` is asking for something ``1e-16`` already declines to reach for.
    """
    if scale is None:
        return default_atol
    ceiling = np.inf if loosen else default_atol
    wanted = rtol * scale
    if wanted < _DERIVED_ATOL_FLOOR and warned is not None and model_name not in warned:
        warned.add(model_name)
        logger.warning(
            'bngsim SBML model %s: its typical species magnitude (%g) puts the absolute '
            'tolerance this model wants (%g) below the %g floor the scale derivation '
            'will reach for on its own; integrating at %g instead. If the fit looks '
            'under-resolved, set sbml_atol explicitly.',
            model_name, scale, wanted, _DERIVED_ATOL_FLOOR, _DERIVED_ATOL_FLOOR,
        )
    return min(max(wanted, _DERIVED_ATOL_FLOOR), ceiling)


def _derive_atol_vector(nominal, scalar_atol, rtol, default_atol, *, loosen=False):
    """One absolute tolerance per species, from the model's nominal state (ADR-0105).

    ``atol_i = clamp(rtol * y_i, scalar_atol, default_atol)``, with ``y_i`` the species'
    own nominal magnitude and ``scalar_atol`` the model-wide number :func:`_derive_atol`
    gives it. Both clamps carry an argument and neither is decoration:

    * **The upper clamp is ADR-0103's, per species.** Nothing is ever loosened past the
      backend default, so a species of order one keeps ``1e-8`` and a model whose species
      are all of order one integrates exactly as before. bngsim's own ``derive_atol`` has
      no upper clamp -- it returns ``1.0e-07`` for ``Brannmark_JBC2010``'s ``X``, looser
      than the default -- so only-ever-tighten is PyBNF's to apply, and is applied here.

      ``loosen`` (#557, ``sbml_atol = auto``) lifts this clamp and only this one; the
      lower one stays exactly where the measurement below put it. What is left is
      ``max(rtol*y_i, scalar_atol)``, i.e. ``rtol * max(y_i, median)`` -- which is
      *bngsim's own* ``derive_atol`` with the model's median supplied as its ``floor``,
      rather than the smallest strictly positive species bngsim would pick by default.
      That equivalence is asserted against the library rather than against a copy of this
      arithmetic, the same way the unclamped default is.

    * **The lower clamp is the model's own scalar, and it is there because the
      alternative was measured and lost.** Read literally, "resolve each species to
      ``rtol`` of its own magnitude" says ``Brannmark``'s ``IRp`` (nominal ``1.76e-9``)
      should be integrated at ``1.76e-17``. Over 100 points sampled from that model's fit
      box, with the fit's own sensitivity request applied, that rule **killed 91
      simulations out of 100 against the scalar's 39** -- ADR-0103's withdrawn
      *minimum* rule reappearing one species at a time. Nothing rescued it: the
      ``1e-16`` floor #549 asks about made no difference at all (91 either way), because
      the damage is done well above it, by the species at ``1.1e-13`` and ``1.6e-12``.

      A tolerance below ``rtol*|y_i|`` only ever binds once ``|y_i|`` has fallen far
      below its *nominal* value, and what it then demands is that a species which has
      decayed into nothing be resolved as if it had not. Initial values cannot tell that
      case apart from a species that genuinely lives down there; that needs the
      trajectory (lanl/bngsim#213).

    So what this vector does is **give back ADR-0103's over-tightening to the species
    that never needed it**, and nothing else. Every entry lands in
    ``[scalar_atol, default_atol]``, so no species is integrated more tightly than PyBNF
    integrates it today and no model that runs today can start failing. On the same 100
    box points that is 39 dead simulations to **33**, in 428 s rather than 576 s.
    ``Brannmark``'s ``IR``/``IRS``/``X`` sit at ~10 and were being held to ``3.3e-10`` --
    ``3.3e-11`` *relative*, three decades tighter than the ``rtol`` that governs them --
    to buy a resolution for ``IRp`` that ADR-0103 had already decided not to chase.

    A species declared at zero falls out of the same expression rather than needing a
    rule: ``rtol * 0`` clamps up to ``scalar_atol``, i.e. a species with no magnitude of
    its own is measured against the model's. That is also the no-regression choice --
    bngsim's ``derive_atol`` substitutes the smallest strictly positive entry instead,
    which on ``Giordano_Nature2020`` would tighten its four zero-seeded species 22x.

    ``nominal`` is ordered; the caller owns the ordering contract against
    ``Model.species_names``.
    """
    ceiling = np.inf if loosen else default_atol
    return np.clip(rtol * np.asarray(nominal, dtype=float), scalar_atol, ceiling)


# Process-level cache of loaded bngsim engine models, keyed by a hash of the
# model's base SBML text. The engine model -- including the analytically
# derived (SymPy) Jacobian -- depends only on the model *structure*, not on
# parameter values, so it is loaded once per worker process and cloned for
# each evaluation rather than re-derived on every objective evaluation. See
# issue #415. The base model is unpickled fresh in each dask worker, so an
# instance attribute would re-derive the Jacobian per evaluation; the cache
# lives at module scope (one per worker process) to amortize across the fit.
_ENGINE_TEMPLATE_CACHE = {}


# Templates a warm was already ATTEMPTED on, keyed by (template key, whether the attempt
# asked for sensitivities) and holding the template the attempt was made on. Only failures
# need remembering -- a successful warm is legible on the template itself
# (_template_is_warm / _template_jacobian_is_warm) -- but a warm that raises leaves no
# trace there, and re-attempting it per action would double the very cost this is meant to
# remove. The shape is part of the key because the two warms are not interchangeable: a
# scalar warm, failed or successful, must not talk a later gradient warm out of running
# (#544). Holding the template rather than a bare flag keeps the memo honest if the cache
# above is cleared and reloaded (the tests do): a fresh template is not the one that
# failed, so it is warmed. See issues #543 and #544.
_ENGINE_TEMPLATE_WARM_ATTEMPTED = {}


def _template_is_warm(template):
    """Does ``template`` already carry a codegen artifact built for sensitivities (#543)?

    bngsim records both on the model: the artifact itself (a compiled ``.so`` path, or
    the C source string under the MIR JIT backend) and ``_want_output_sens``, which it
    documents as doubling for "the attached codegen already has output sens". A plain-RHS
    artifact inherited from a scalar construction has the flag False, and bngsim's own
    ``_prepare_output_sens_codegen`` would then clear and regenerate it at the first
    sensitivity request -- correct, but it saves nothing, so it does not count as warm.
    """
    return bool(
        getattr(template, '_want_output_sens', False)
        and (getattr(template, '_codegen_so_path', '')
             or getattr(template, '_codegen_c_source', ''))
    )


def _template_jacobian_is_warm(template):
    """Has ``template`` already attempted the analytical-Jacobian derivation (#544)?

    bngsim derives the (SymPy) Functional Jacobian at most once per model, guarded by the
    ``_jac_attempted`` sentinel that ``clone()`` copies parent -> child precisely so "a
    derived parent [yields] cheap, already-warm clones". That sentinel is what a clone
    reads, so it -- and not whether the derivation produced a *complete* analytical
    Jacobian -- is what "already warm" has to mean here: a model that fell back to finite
    differences has still paid the derivation, and neither it nor its clones re-pay it.

    Deliberately weaker than :func:`_template_is_warm`, which the sensitivity warm needs:
    a scalar-warmed template answers True here and False there, so a gradient fit whose
    template was scalar-warmed still takes its own (sensitivity-shaped) warm.
    """
    return bool(getattr(template, '_jac_attempted', False))


from ._bngsim_caps import (
    BNGSIM_HAS_OUTPUT_SENS,
    BNGSIM_HAS_PER_SPECIES_ATOL,
    BNGSIM_HAS_SBML,
    BNGSIM_HAS_TRACKING_ATOL,
    BNGSIM_SBML_ERROR,
    bngsim,
    feature_missing_reason,
)

try:
    import libsbml
except ImportError:
    libsbml = None


# libSBML can evaluate initialAssignments in place (resolving assignmentRule
# intermediates), letting us recompute parameter-driven species initials
# without a full model reload. If the transform is unavailable we fall back to
# reloading from SBML text for the (rare) parameter-driven-initial case (#415).
_HAS_EXPAND_INITIAL_ASSIGNMENTS = (
    libsbml is not None
    and hasattr(libsbml, 'SBMLTransforms')
    and hasattr(libsbml.SBMLTransforms, 'expandInitialAssignments')
)


def _require_bngsim_sbml_support():
    if not BNGSIM_HAS_SBML:
        raise RuntimeError(BNGSIM_SBML_ERROR)


def _sbml_doc_from_text(text, source_desc):
    reader = libsbml.SBMLReader()
    doc = reader.readSBMLFromString(text)
    if doc is None:
        raise ModelError(f'Failed to parse SBML from {source_desc}')

    messages = []
    for i in range(doc.getNumErrors()):
        err = doc.getError(i)
        if err.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
            messages.append(err.getMessage())
    if messages:
        raise ModelError(
            'Failed to parse SBML from {}: {}'.format(source_desc, '; '.join(messages[:3]))
        )

    if doc.getModel() is None:
        raise ModelError(f'SBML document from {source_desc} does not contain a model')

    return doc


def _sbml_doc_to_text(doc):
    writer = libsbml.SBMLWriter()
    return writer.writeSBMLToString(doc)


def _is_event_sensitivity_refusal(exc):
    """Is ``exc`` bngsim declining to differentiate this model's discrete events (#536)?

    bngsim classifies each event and refuses the subclasses whose crossing it cannot
    differentiate -- an execution delay, and a trigger that does not reduce to a single
    relational comparison -- with a ``ValueError`` raised from every sensitivity-bearing
    ``run()``. That is a *structural* verdict, identical at every parameter set, and so
    wants a different answer from a candidate point the integrator merely could not get
    through; :meth:`BngsimSbmlModelNoTimeout.execute` turns it into an actionable
    refusal and leaves everything else on the back-off path.

    Recognised by the message, since bngsim raises a plain ``ValueError`` for it. Both
    halves must match, so an unrelated sensitivity error is not swept up. Should bngsim
    ever reword it, this degrades to the old behaviour (one more failed simulation) --
    never to a wrong gradient.
    """
    if not isinstance(exc, ValueError):
        return False
    text = str(exc).lower()
    return 'sensitivities are not supported' in text and 'events' in text


def _mutate_scalar(value, operation, amount):
    if operation == '=':
        return amount
    if operation == '+':
        return value + amount
    if operation == '-':
        return value - amount
    if operation == '*':
        return value * amount
    if operation == '/':
        return value / amount
    raise RuntimeError(f'Invalid mutation operation {operation}')


@dataclass
class _SensitivityRequest:
    """The forward-sensitivity request that activates the SBML/Antimony gradient path.

    The SBML-backend twin of ``net_model._SensitivityRequest`` (#385/#447); a separate
    two-field holder so the two backends stay decoupled. Set by
    :meth:`BngsimSbmlModelNoTimeout.enable_output_sensitivities` (#455); ``None`` on the
    scalar path. ``params`` are native global model parameter ids routed to
    ``Simulator(sensitivity_params=)`` and ``ic`` are species names routed to
    ``Simulator(sensitivity_ic=)`` (the routing lists themselves come from #448). Native
    parameter space throughout -- no transform here.
    """
    params: list   # native global model parameter ids -> sensitivity_params
    ic: list       # species names -> sensitivity_ic


class BngsimSbmlModelNoTimeout(Model):
    # Gradient path (#385/#455): None on the scalar path, a _SensitivityRequest once
    # enable_output_sensitivities() activates forward sensitivities. A class attribute (not
    # only an __init__ assignment) so the scalar path stays intact for instances built via
    # object.__new__ (the _make_simulator test fakes, pickling).
    _sensitivity_request = None

    # Per-action sensitivity gate (#475/#482): the SBML twin of the net backend's
    # gate. On the gradient path only an action whose output is a SCORED gradient
    # target needs forward sensitivities; an incidental/unscored action -- a
    # stochastic (ssa) diagnostic that no data scores -- carries none, so it runs on
    # the ordinary path and neither computes a wasted sensitivity tensor nor aborts
    # the whole fit at a differentiability guard it can never satisfy.
    # ``_scored_suffixes`` is the model's scored full-suffix set (set_scored_suffixes,
    # from exp_data); ``_current_action_suffix`` names the action being prepared. The
    # SBML backend folds the mutant/condition suffix straight into the current suffix
    # (``act.suffix + mut.suffix``, set in execute()) because its mutant loop is inline
    # -- so, unlike net_model.BngsimModel, it needs no per-instance ``_sensitivity_offset``.
    # Both are class attributes so an unset (``None``) scored set falls back to the
    # historical all-actions-bearing behavior, and an object.__new__ instance is safe.
    _scored_suffixes = None
    _current_action_suffix = None

    # The ids the ODE right-hand side reads (ADR-0097), set from the parsed document by
    # _extract_sbml_structure. A class attribute so an instance built via object.__new__ (the
    # _make_simulator test fakes, unpickling) answers "cannot say" rather than raising -- and
    # "cannot say" keeps every parameter axis, which is the safe direction.
    _rhs_symbols = None

    # Explicit CVODE tolerances (``sbml_rtol`` / ``sbml_atol``, #546). ``None`` means
    # "not stated": rtol then takes the backend default and atol is derived from the
    # model's own state scale (:meth:`_effective_tolerances`). Class attributes so an
    # instance built via object.__new__ -- the ``_make_simulator`` test fakes, unpickling
    # -- answers "not stated" rather than raising.
    #
    # ``sbml_atol`` splits three ways (#557, :func:`parse_atol_setting`), and only the
    # first of the three is a tolerance: ``_config_atol`` is a *pinned number*, which
    # switches every derivation off, while ``_atol_mode`` is an opt-in to the derivation
    # -- ``auto`` lifts its only-ever-tighten ceiling, ``tracking`` adds
    # lanl/bngsim#213's trajectory-following weights on top of that. The two are mutually
    # exclusive by construction, so ``_config_atol is not None`` remains exactly what it
    # was: "the user stated the number, take it".
    #
    # ``_config_species_atol`` is the fourth kind of statement and the only one that is not
    # about the whole model: the hand-written ``{species: atol}`` map a new-era ``model:``
    # declaration carries (#586, ADR-0116). It is an OVERRIDE LAYER rather than a
    # replacement -- the species it names take the number stated, verbatim, and every other
    # species keeps whatever the three settings above would have given it. ``None`` means
    # "nothing stated", which is every model that does not use the syntax.
    _config_rtol = None
    _config_atol = None
    _atol_mode = None
    _atol_tracking_decades = None
    _config_species_atol = None

    # The species names bngsim will integrate this model under, captured from the engine
    # load the constructor already pays for (:meth:`_load_engine_model_or_raise`). NOT the
    # same list as ``_species_names``, which is the SBML document's ids: bngsim renames an
    # id that collides with an Antimony reserved word (``NULL`` -> ``_ant_NULL``) and
    # appends a state for every parameter or compartment a rate rule or event assignment
    # drives. Those are the names a ``species_atol`` map is written in and checked against,
    # because they are the ones a per-species vector is ordered by.
    _engine_species_names = None

    # The model's per-species nominal values in bngsim's units, and their typical (median
    # strictly-positive) magnitude, both set from the parsed document by
    # _extract_sbml_structure. ``None`` = nothing to derive an atol from, which keeps the
    # backend default. ``_tolerance_cache`` memoizes the derived *scalar* pair (ADR-0103,
    # still the steady-state cutoff under ADR-0105) and ``_atol_vector_cache`` the derived
    # per-species vector together with the species ordering it was built for.
    # ``_atol_floor_warned`` holds the names the floor warning has already fired for. All
    # class attributes so an instance built via object.__new__ -- the ``_make_simulator``
    # test fakes, unpickling -- answers "cannot say" rather than raising.
    _nominal_species_values = None
    _nominal_state_scale = None
    _tolerance_cache = None
    _atol_vector_cache = None
    _atol_floor_warned = None

    def __init__(self, file, abs_file, pset=None, actions=(), save_files=False, integrator='cvode',
                 strict_ssa=True, rtol=None, atol=None, species_atol=None):
        if integrator not in _SUPPORTED_INTEGRATORS:
            raise ModelError(
                'sbml_backend = bngsim supports sbml_integrator in {}; got {}'.format(', '.join(_SUPPORTED_INTEGRATORS), integrator)
            )

        _require_bngsim_sbml_support()

        self._init_common_attrs(file, abs_file, pset, actions, save_files, integrator, strict_ssa,
                                file_ext='.xml', rtol=rtol, atol=atol, species_atol=species_atol)
        self.stochastic = integrator == 'gillespie' or any(
            getattr(a, 'method', 'ode') == 'ssa' for a in actions
        )

        with open(self.abs_file_path, encoding='utf-8', errors='replace') as fh:
            self._base_sbml_text = fh.read()

        self._extract_sbml_structure()
        self._load_engine_model_or_raise(
            f'Failed to load model {self.name}.xml - There were errors in parsing this SBML file. See log for details.'
        )

        logger.debug('Loaded model %s with bngsim SBML backend', self.name)

    def _init_common_attrs(self, file, abs_file, pset, actions, save_files, integrator, strict_ssa,
                           file_ext, rtol=None, atol=None, species_atol=None):
        """Set the model attributes shared by the SBML and Antimony backends.

        ``file_ext`` is stripped from the file name to form ``self.name`` ('.xml'
        for SBML, '.ant' for Antimony). ``self.stochastic`` is set by the caller
        because the two backends compute it differently. ``rtol``/``atol`` carry the
        config's explicit CVODE tolerances (#546); ``None`` leaves each to the
        backend default / the scale derivation. ``species_atol`` carries this model's
        own hand-written per-species map (#586); its species NAMES are checked once the
        engine model has been loaded (:meth:`_check_stated_species_atol`), because the
        list they are written against is bngsim's rather than the document's.
        """
        self.file_path = file
        self.abs_file_path = abs_file
        self.param_set = pset
        self.name = file[file.rfind('/') + 1:].rsplit(file_ext, 1)[0]
        self.save_files = save_files
        self.actions = list(actions)
        self.integrator = integrator
        self.strict_ssa = bool(strict_ssa)
        self._config_rtol = None if rtol is None else float(rtol)
        try:
            (self._config_atol, self._atol_mode,
             self._atol_tracking_decades) = parse_atol_setting(atol)
        except ValueError as exc:
            raise ModelError(f'Invalid sbml_atol for model {self.name}: {exc}') from exc
        if self._atol_mode == _ATOL_TRACKING and not BNGSIM_HAS_TRACKING_ATOL:
            # Refused rather than degraded. A tolerance mode that silently did not apply
            # produces a plausible trajectory and a wrong conclusion about the model,
            # which is the failure this whole line of work exists to stop making.
            raise ModelError(
                'sbml_atol = tracking needs a bngsim whose Simulator.run takes a '
                'trajectory-following absolute tolerance (lanl/bngsim#213, '
                'bngsim.TrackingAtol); this build does not have one.')
        try:
            self._config_species_atol = parse_species_atol_setting(species_atol)
        except ValueError as exc:
            raise ModelError(f'Invalid species_atol for model {self.name}: {exc}') from exc
        if self._config_species_atol and not BNGSIM_HAS_PER_SPECIES_ATOL:
            # Refused rather than degraded, for the reason the tracking refusal above gives.
            # The DERIVED vector may fall back to the scalar silently (:meth:`_per_species_atol`)
            # because the scalar is a correct tolerance for that model and nobody asked for
            # anything else; a hand-written map is somebody asking, and a stated tolerance
            # that did not apply is a plausible trajectory with a wrong conclusion attached.
            raise ModelError(
                'species_atol needs a bngsim whose Simulator.run takes a per-species '
                'absolute tolerance (lanl/bngsim#196, bngsim.normalize_atol_vector); this '
                'build does not have one.')
        self._tolerance_cache = None
        self._atol_vector_cache = None
        self._atol_floor_warned = set()
        self._engine_species_names = None
        self.suffixes = [(a.bng_codeword, a.suffix) for a in actions]
        self.mutants = [MutationSet()]

    def _extract_sbml_structure(self):
        """Parse ``self._base_sbml_text`` and populate species/parameter names."""
        doc = _sbml_doc_from_text(self._base_sbml_text, self.file_path)
        self._species_names = tuple(
            doc.getModel().getSpecies(i).getId()
            for i in range(doc.getModel().getNumSpecies())
        )
        self._species_name_set = set(self._species_names)
        self._global_param_names = tuple(
            doc.getModel().getParameter(i).getId()
            for i in range(doc.getModel().getNumParameters())
        )
        # The nominal parameter table: the environment a point-dependent seed derivative
        # (#530) is evaluated in, before the fit vector and the condition override it.
        self._nominal_param_values = {
            doc.getModel().getParameter(i).getId():
                float(doc.getModel().getParameter(i).getValue())
            for i in range(doc.getModel().getNumParameters())
        }
        self.global_param_names = self._global_param_names
        self.param_names = self._species_name_set.union(set(self._global_param_names))
        self._initial_dep_names = self._compute_initial_dependency_names(doc.getModel())
        (self._species_unit_factor, self._species_assignment_to_concentration,
         self._unsafe_volume) = self._compute_species_unit_factors(doc.getModel())
        self._ic_seed_map = self._compute_ic_seed_map(doc.getModel())
        self._rhs_symbols = self._compute_rhs_symbols(doc.getModel())
        self._nominal_species_values = self._compute_nominal_species_values(doc.getModel())
        self._nominal_state_scale = self._compute_nominal_state_scale(
            self._nominal_species_values)

    def _compute_nominal_species_values(self, sbml_model):
        """Every species' nominal value, by id, in bngsim's units (ADR-0105).

        The state both tolerance derivations are read off: the **median** of the
        strictly-positive entries is ADR-0103's scalar scale
        (:meth:`_compute_nominal_state_scale`), and the entries themselves are
        ADR-0105's per-species vector (:func:`_derive_atol_vector`). Read from the
        parsed document rather than from a loaded engine model, so it costs nothing on
        top of the parse ``_extract_sbml_structure`` already pays -- and so it is a
        property of the model *text*, not of the fit point or of whether this evaluation
        took the cached-clone or the structural-reload path. ``_species_unit_factor``
        puts an amount-declared species into the concentration units bngsim integrates
        in, exactly as the in-place value paths do.

        A non-positive or non-finite declaration is stored as ``0.0``, meaning "no
        magnitude of its own": the median skips it, and the vector's lower clamp puts it
        at the model's scalar without needing a rule of its own. ``None`` -- meaning
        "nothing to derive from", which leaves the tolerance at the backend default --
        when a non-constant compartment volume makes the unit factors themselves
        parameter-dependent, since then the values here are not the ones that would be
        integrated.
        """
        if self._unsafe_volume:
            return None
        values = {}
        for i in range(sbml_model.getNumSpecies()):
            sp = sbml_model.getSpecies(i)
            factor = self._species_unit_factor.get(sp.getId(), 1.0)
            value = self._species_initial_value(sp) * factor
            values[sp.getId()] = (
                float(value) if np.isfinite(value) and value > 0.0 else 0.0)
        return values

    def _compute_nominal_state_scale(self, nominal_values):
        """The magnitude this model's state lives at, in bngsim's units (#546).

        The scale the *scalar* CVODE absolute tolerance is measured against, answered as
        the **median** strictly-positive nominal species value. Under ADR-0103 that scalar
        was the integration tolerance; under ADR-0105 it is the steady-state convergence
        cutoff whenever the per-species vector is in force (see
        :meth:`_run_tolerance_kwargs`), and still the integration tolerance on every path
        the vector does not reach.

        Median rather than minimum, which is the version of this that had to be
        withdrawn. A biochemical model routinely carries one transient intermediate
        many decades beneath everything else -- ``Brannmark_JBC2010`` seeds ``IRp`` at
        1.8e-9 while its principal species sit at 0.1..10 -- and resolving *that* to a
        relative tolerance drives the absolute one to 1e-17, which the model cannot meet:
        CVODE exhausts ``mxstep`` and the simulation fails outright at fit points it used
        to integrate in milliseconds. The median asks the question the tolerance is
        actually about ("what magnitude is this model written in?") and is unmoved by a
        single outlier at either end, so Brannmark reads 0.033 and keeps a tolerance
        near the default while Giordano reads 3.7e-7 and gets one four decades tighter.

        Zeros are skipped: a species that starts empty says nothing about the model's
        scale (Giordano's ``Threatened``/``Extinct``/``Healed`` all start at exactly 0
        and grow into the same 1e-8..1e-3 band as everything else). ``None`` -- meaning
        "no scale to derive from", which leaves the tolerance at the backend default --
        when nothing positive is declared, or when a non-constant compartment volume
        makes the unit factors themselves parameter-dependent.

        A species whose initial value comes from an ``initialAssignment`` contributes
        only what it *declares*, which is usually 0 and therefore skipped. That is a
        limit on what this can detect, not an error: an undetected small scale leaves the
        tolerance exactly where it is today.
        """
        if nominal_values is None:
            return None
        values = [v for v in nominal_values.values() if v > 0.0]
        return float(np.median(values)) if values else None

    @staticmethod
    def _collect_ast_names(node, out):
        """Recursively collect the symbol names referenced by a libSBML AST."""
        if node is None:
            return
        if node.getType() == libsbml.AST_NAME:
            name = node.getName()
            if name:
                out.add(name)
        for i in range(node.getNumChildren()):
            BngsimSbmlModelNoTimeout._collect_ast_names(node.getChild(i), out)

    def _compute_initial_dependency_names(self, sbml_model):
        """Analyze which initial values depend on which parameters/species.

        Sets ``self._initial_expr_species`` (the species whose initial value is
        fixed at load time by an ``initialAssignment``) and
        ``self._initial_expr_params`` (the *parameters* an ``initialAssignment``
        derives the same way), and returns the set of parameter/species names
        those assignments depend on -- transitively, following
        ``assignmentRule``-defined intermediates. When a changed name is in this
        set, the assignments must be recomputed for the evaluation (see
        :meth:`_recompute_initial_assignments`); an empty set means parameter
        values can never change an initial value, so the fast cached-clone path
        is exact. Returns ``None`` if the model has an ``algebraicRule`` (whose
        effect on initials we do not analyze, so we conservatively force a full
        reload). See issue #415.

        A parameter with an ``initialAssignment`` is a **derived constant** --
        SBML re-evaluates it whenever one of its dependencies changes, exactly
        as it does a species initial (``beta_N = R0_*gamma_/N_`` in
        Bertozzi_PNAS2020). bngsim evaluates it once at load and ``set_param``
        does not propagate to it, so it must be recomputed here too; leaving it
        at its load-time value silently simulated a different model than the
        reload path did (#531). A parameter an ``assignmentRule`` or
        ``rateRule`` governs is excluded -- bngsim recomputes *those*
        dynamically, so an initial override would fight the rule.
        """
        self._initial_expr_species = set()
        self._initial_expr_params = set()
        # symbol -> referenced names, for every expression that *defines* a
        # symbol's value. assignmentRules are included only so the transitive
        # walk can resolve a parameter that an initialAssignment reads through.
        expr_refs = {}
        ia_seeds = {}  # species/parameter symbol -> initialAssignment refs (the seed)
        ia_species = set()
        ia_params = set()
        for i in range(sbml_model.getNumInitialAssignments()):
            ia = sbml_model.getInitialAssignment(i)
            refs = set()
            self._collect_ast_names(ia.getMath(), refs)
            symbol = ia.getSymbol()
            expr_refs.setdefault(symbol, set()).update(refs)
            if symbol in self._species_name_set:
                ia_seeds[symbol] = refs
                ia_species.add(symbol)
            elif symbol in self._global_param_names:
                ia_seeds[symbol] = refs
                ia_params.add(symbol)
        ruled = set()
        for i in range(sbml_model.getNumRules()):
            rule = sbml_model.getRule(i)
            if rule.isAlgebraic():
                # An algebraic rule constrains values implicitly; we do not
                # solve it, so force the correctness-preserving reload path.
                return None
            variable = rule.getVariable() if hasattr(rule, 'getVariable') else None
            if variable:
                ruled.add(variable)
            if rule.isAssignment():
                refs = set()
                self._collect_ast_names(rule.getMath(), refs)
                expr_refs.setdefault(variable, set()).update(refs)
            # Rate rules define d/dt, not the initial value -> ignored here.

        self._initial_expr_species = ia_species
        self._initial_expr_params = ia_params - ruled

        # Seed from every initialAssignment, then expand through any referenced
        # symbol that is itself expression-defined (e.g. a species initial that
        # reads a parameter set by an assignmentRule).
        dep = set()
        worklist = []
        for refs in ia_seeds.values():
            dep.update(refs)
            worklist.extend(refs)
        seen = set(worklist)
        while worklist:
            name = worklist.pop()
            for ref in expr_refs.get(name, ()):
                dep.add(ref)
                if ref not in seen:
                    seen.add(ref)
                    worklist.append(ref)
        return dep

    def _compute_ic_seed_map(self, sbml_model):
        """Map a model parameter to the initial values it seeds, with their derivatives.

        An ``initialAssignment`` makes one entity's starting value a function of others:
        ``I_ = I0_``, ``S_ = N_ - I0_``, ``beta_N = R0_*gamma_/N_``. A free parameter a
        condition assigns to one of those *inputs* (a per-condition estimated initial
        condition, ADR-0076) therefore reaches the trajectory through the assigned entity's own
        sensitivity column, scaled by ``d(entity)/d(param)``. Each parameter maps to the tuple
        of :class:`~pybnf.gradient.routing.SeedTerm`\\ s carrying those columns and derivatives
        (#530); #511 could only express a derivative of exactly ``1``.

        A species term also folds in the unit conversion between the assignment (which sets an
        *amount* for a ``hasOnlySubstanceUnits`` species and a *concentration* otherwise) and
        the PyBNF species value the sensitivity tensor is already expressed in
        (:meth:`_extract_output_sensitivities`) -- the compartment volume, where the two
        disagree.

        A parameter maps to ``None`` -- present but non-routable, so the router refuses rather
        than emit a wrong column -- when any initial value it feeds lies outside the arithmetic
        grammar (:mod:`pybnf.gradient.derivative`), is reached through an ``assignmentRule`` or
        a second ``initialAssignment`` (a chain this does not compose), sits on a species whose
        compartment volume is not a load-time constant, or has a derivative reading something
        other than a global parameter (which the per-point environment cannot supply).
        """
        if self._initial_dep_names is None:
            # An algebraicRule constrains initial values implicitly and is not solved here, so
            # no seed derivative read off the assignments alone can be trusted.
            return {}
        param_set = set(self._global_param_names)
        seeded = {sbml_model.getInitialAssignment(i).getSymbol()
                  for i in range(sbml_model.getNumInitialAssignments())}
        opaque = set(seeded) | self._ruled_symbols(sbml_model)
        terms = {}     # param -> [SeedTerm]
        blocked = set()  # params whose seeding cannot be differentiated
        for i in range(sbml_model.getNumInitialAssignments()):
            ia = sbml_model.getInitialAssignment(i)
            symbol = ia.getSymbol()
            is_species = symbol in self._species_name_set
            if not is_species and symbol not in self._initial_expr_params:
                continue  # a rule-governed or non-parameter target: bngsim owns its value
            refs = set()
            self._collect_ast_names(ia.getMath(), refs)
            inputs = {r for r in refs if r in param_set}
            try:
                tree = derivative.from_sbml_ast(ia.getMath(), libsbml)
                scale = self._ia_species_scale(symbol) if is_species else 1.0
            except (derivative.NotDifferentiable, PybnfError):
                blocked.update(inputs)
                continue
            if refs & opaque:
                # The assignment reads a symbol another assignment or a rule defines: the
                # chain rule through it is not composed here.
                blocked.update(inputs)
                continue
            axis, key = (IC, symbol) if is_species else (PARAM, symbol)
            for param in sorted(inputs):
                try:
                    node = derivative.mul(derivative.num(scale),
                                          derivative.differentiate(tree, param))
                except derivative.NotDifferentiable:
                    blocked.add(param)
                    continue
                if not derivative.symbols(node) <= param_set:
                    # The per-point environment supplies parameter values only.
                    blocked.add(param)
                    continue
                if not derivative.is_constant(node) or node[1] != 0.0:
                    terms.setdefault(param, []).append(SeedTerm(axis, key, node))
        seed_map = {param: tuple(seeds) for param, seeds in terms.items()}
        for param in blocked:
            seed_map[param] = None  # one non-routable use blocks the parameter outright
        return seed_map

    def _compute_rhs_symbols(self, sbml_model):
        """Every symbol the ODE right-hand side can read -- deliberately over-inclusive.

        The gradient router drops a bound id's own ``sensitivity_params`` axis only when the
        right-hand side provably never reads it (ADR-0097). Answering that from the *seeding*
        pattern instead was wrong for any model whose ``initialAssignment``\\ s are a
        steady-state solution over the kinetic constants -- ``Fiedler_BMCSystBiol2016`` seeds all
        six species initials from ``k2 … k10``, which are also the rate constants, so every one
        of them lost its right-hand-side derivative (#535).

        Collected from reaction kinetic laws, every rule (assignment/rate/algebraic) and every
        function definition body, plus each event's trigger/delay/priority and assignments.
        Including *all* rule math regardless of reachability is what makes a chain -- a rate law
        reading a symbol an ``assignmentRule`` derives from a parameter -- resolve without a
        transitive walk. **Not** collected: ``initialAssignment`` math, which seeds initial
        values rather than driving the trajectory, and whose parameters reach the fit through
        :meth:`_compute_ic_seed_map` instead. That exclusion is what keeps a COPASI alias
        (``ModelValue_79 = k_syn_R_M``, ADR-0096) from paying for an axis that really is zero.

        Over-inclusion costs one redundant forward-sensitivity vector; under-inclusion silently
        deletes half a derivative, so every ambiguous case is resolved toward including.
        """
        names = set()
        for i in range(sbml_model.getNumReactions()):
            law = sbml_model.getReaction(i).getKineticLaw()
            if law is not None:
                self._collect_ast_names(law.getMath(), names)
        for i in range(sbml_model.getNumRules()):
            self._collect_ast_names(sbml_model.getRule(i).getMath(), names)
        for i in range(sbml_model.getNumFunctionDefinitions()):
            self._collect_ast_names(sbml_model.getFunctionDefinition(i).getMath(), names)
        for i in range(sbml_model.getNumEvents()):
            event = sbml_model.getEvent(i)
            for part in (event.getTrigger(), event.getDelay(), event.getPriority()):
                if part is not None:
                    self._collect_ast_names(part.getMath(), names)
            for j in range(event.getNumEventAssignments()):
                assignment = event.getEventAssignment(j)
                self._collect_ast_names(assignment.getMath(), names)
                if assignment.getVariable():
                    names.add(assignment.getVariable())
        return frozenset(names)

    def ode_rhs_symbols(self):
        """The ids the ODE right-hand side reads (:meth:`_compute_rhs_symbols`), for the router.

        The gradient router uses this only to *permit* dropping a pure initial-value seed's
        own ``sensitivity_params`` axis: an id absent from this set and seeding nothing but
        species initial conditions has no separate right-hand-side path on that axis, which
        would otherwise duplicate the seeding its ``ic`` axis carries (#537). Absence alone never
        drops an axis (ADR-0097, #535). Known at load
        time -- no simulation.
        """
        return self._rhs_symbols

    @staticmethod
    def _ruled_symbols(sbml_model):
        """Symbols an assignment/rate rule defines -- opaque to the seed differentiation."""
        ruled = set()
        for i in range(sbml_model.getNumRules()):
            rule = sbml_model.getRule(i)
            variable = rule.getVariable() if hasattr(rule, 'getVariable') else None
            if variable:
                ruled.add(variable)
        return ruled

    def _ia_species_scale(self, species):
        """``d(PyBNF species value)/d(initialAssignment value)`` for one species.

        The assignment sets an amount for a ``hasOnlySubstanceUnits`` species and a
        concentration otherwise; the sensitivity tensor is in PyBNF species-value units
        (``_species_unit_factor``). Where the two agree this is ``1``; where they disagree it
        is the compartment volume (or its reciprocal). Raises when the volume is not a
        load-time constant, since the factor would then move with the fit.
        """
        if self._unsafe_volume:
            raise PybnfError('compartment volume is not a load-time constant')
        assignment_to_conc = self._species_assignment_to_concentration[species]
        return assignment_to_conc / self._species_unit_factor[species]

    def _compute_species_unit_factors(self, sbml_model):
        """Per-species factor converting a PyBNF species value to a bngsim
        concentration, the factor converting an ``initialAssignment``'s value to
        that same concentration, and a flag for volumes we cannot safely convert.

        bngsim works in concentrations internally. A concentration-based species
        value is already a concentration (factor 1.0); an amount-based species
        value is an amount, so its concentration is amount / compartment_size
        (factor 1/size). The reload path bakes these units on load, so the
        in-place paths must apply the same conversion to stay numerically
        identical. When an amount-based species sits in a compartment whose size
        is not a load-time constant (non-constant, or set by an
        initialAssignment/rule), the factor would itself be parameter-dependent;
        we flag that and fall back to a full reload. See issue #415.

        The second map answers a different question, for the gradient router (#530): an
        ``initialAssignment`` on a ``hasOnlySubstanceUnits`` species sets an *amount* and on
        any other species a *concentration*, which is not always the unit the PyBNF species
        value carries. Its factor converts the assignment's value to a concentration, so
        composing it with the reciprocal of the first map gives ``d(PyBNF value)/d(assignment
        value)``.
        """
        ruled = set()
        for i in range(sbml_model.getNumInitialAssignments()):
            ruled.add(sbml_model.getInitialAssignment(i).getSymbol())
        for i in range(sbml_model.getNumRules()):
            rule = sbml_model.getRule(i)
            var = rule.getVariable() if hasattr(rule, 'getVariable') else None
            if var:
                ruled.add(var)

        factors = {}
        assignment_factors = {}
        unsafe = False
        for i in range(sbml_model.getNumSpecies()):
            sp = sbml_model.getSpecies(i)
            name = sp.getId()
            comp_id = sp.getCompartment()
            comp = sbml_model.getCompartment(comp_id)
            vol = comp.getSize() if (comp is not None and comp.isSetSize()) else 1.0
            comp_constant = comp.getConstant() if comp is not None else True
            volume_is_constant = (
                comp_constant and comp_id not in ruled
                and vol == vol and vol > 0  # finite and positive
            )
            assignment_factors[name] = (
                1.0 / float(vol) if (sp.getHasOnlySubstanceUnits() and volume_is_constant)
                else 1.0)
            amount_based = sp.isSetInitialAmount() or (
                not sp.isSetInitialConcentration() and sp.getHasOnlySubstanceUnits()
            )
            if not amount_based:
                factors[name] = 1.0
                continue
            if volume_is_constant:
                factors[name] = 1.0 / float(vol)
            else:
                factors[name] = 1.0  # unused: the flag forces a reload
                unsafe = True
        return factors, assignment_factors, unsafe

    def _changed_names(self, mut=None, scan_param=None, phase_overrides=None):
        """The parameter/species names this evaluation changes."""
        changed = set()
        if self.param_set is not None:
            changed.update(self.param_set.keys())
        if mut:
            changed.update(mi.name for mi in mut)
        if scan_param is not None:
            changed.add(scan_param)
        if phase_overrides:
            changed.update(name for name, _ in phase_overrides)
        return changed

    def _changes_touch_initials(self, mut=None, scan_param=None, phase_overrides=None):
        """Whether this evaluation changes a name that a species initial depends
        on, so the baked-in initial concentrations must be recomputed (#415)."""
        if not self._initial_dep_names:
            return False
        return bool(self._changed_names(mut=mut, scan_param=scan_param,
                                        phase_overrides=phase_overrides)
                    & self._initial_dep_names)

    def _needs_structural_reload(self, mut=None, scan_param=None):
        """Whether this evaluation needs a full reload from SBML text.

        True for models with an ``algebraicRule`` (which we do not analyze) or
        with an amount-based species in a non-constant-volume compartment (whose
        unit conversion would be parameter-dependent). Parameter-driven species
        initials no longer force a reload -- they are recomputed in place via
        libSBML, reusing the cached engine model and its derived Jacobian (see
        :meth:`_recompute_species_initials`, issue #415).
        """
        return self._initial_dep_names is None or self._unsafe_volume

    def _load_engine_model_or_raise(self, parse_error_message):
        """Load the bngsim engine model, letting FileNotFoundError propagate
        unwrapped and converting any other failure into a ModelError.

        The loaded model is not kept -- the per-action engine models come from
        :meth:`_get_engine_template`'s text-keyed cache -- but its **species names** are,
        because they are the list a hand-written ``species_atol`` map is written against
        (#586) and this load is already paid for. Checking the map here is what makes an
        unknown species name an error at config load, where the conf author is, rather
        than a silent no-op discovered from a trajectory.
        """
        try:
            engine = self._load_bngsim_model_from_path(self.abs_file_path)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise ModelError(parse_error_message) from exc
        self._engine_species_names = tuple(getattr(engine, 'species_names', None) or ())
        self._check_stated_species_atol()

    def _check_stated_species_atol(self):
        """Refuse a ``species_atol`` naming a species this model does not integrate (#586).

        The list checked against is the **engine** model's, not the SBML document's, and
        the two really do differ: bngsim renames an id that collides with an Antimony
        reserved word (``Smith_BMCSystBiol2013``'s ``NULL``/``null`` load as
        ``_ant_NULL``/``_ant_null``) and promotes to a state every parameter or compartment
        a rate rule or an event assignment drives, which the ``<listOfSpecies>`` does not
        mention at all. Engine names are what a per-species vector is ordered by, so they
        are the only list that can be both checked and applied; validating against document
        ids would accept a name that then silently did nothing, which is the failure this
        whole line of work exists to stop making.

        The message names the species the model does have, because a name that reads wrong
        is usually one of them spelled differently -- and it says so outright for the
        ``_ant_`` case, which is the one a conf author cannot guess from their own file.
        """
        stated = getattr(self, '_config_species_atol', None)
        names = getattr(self, '_engine_species_names', None)
        if not stated or not names:
            return
        known = set(names)
        unknown = [name for name in stated if name not in known]
        if not unknown:
            return
        hints = [f"'{n}' (bngsim renames an id that collides with an Antimony reserved "
                 f"word, so state it as '_ant_{n}')"
                 for n in unknown if f'_ant_{n}' in known]
        shown = list(names[:12]) + (['...'] if len(names) > 12 else [])
        message = (
            f'species_atol for model {self.name} names {len(unknown)} species this model '
            f'does not integrate: {", ".join(sorted(unknown))}. It integrates '
            f'{len(names)}: {", ".join(shown)}.')
        if hints:
            message += ' ' + ' '.join(hints) + '.'
        raise ModelError(message)

    def copy_with_param_set(self, pset):
        newmodel = copy.deepcopy(self)
        newmodel.param_set = pset
        return newmodel

    @property
    def species_names(self):
        return self._species_names

    def _load_bngsim_model_from_path(self, path):
        return bngsim.Model.from_sbml(path)

    def _load_bngsim_model_from_text(self, text):
        model_cls = bngsim.Model
        if hasattr(model_cls, 'from_sbml_string'):
            return model_cls.from_sbml_string(text)

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as tf:
                tf.write(text)
                temp_path = tf.name
            return model_cls.from_sbml(temp_path)
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    @staticmethod
    def _species_initial_value(species):
        if species.isSetInitialAmount():
            return float(species.getInitialAmount())
        if species.isSetInitialConcentration():
            return float(species.getInitialConcentration())
        return 0.0

    @staticmethod
    def _set_species_initial_value(species, value):
        value = float(value)
        if species.isSetInitialAmount() or (
            not species.isSetInitialConcentration() and species.getHasOnlySubstanceUnits()
        ):
            species.setInitialAmount(value)
            if species.isSetInitialConcentration():
                species.unsetInitialConcentration()
        else:
            species.setInitialConcentration(value)
            if species.isSetInitialAmount():
                species.unsetInitialAmount()

    def _set_model_value_if_present(self, sbml_model, name, value):
        if name in self._species_name_set:
            species = sbml_model.getSpecies(name)
            if species is not None:
                self._set_species_initial_value(species, value)
            return

        if name in self._global_param_names:
            param = sbml_model.getParameter(name)
            if param is not None:
                param.setValue(float(value))

    def _get_model_value_if_present(self, sbml_model, name):
        if name in self._species_name_set:
            species = sbml_model.getSpecies(name)
            if species is not None:
                return self._species_initial_value(species)
        elif name in self._global_param_names:
            param = sbml_model.getParameter(name)
            if param is not None:
                return float(param.getValue())
        return None

    def _apply_param_set(self, sbml_model):
        if self.param_set is None:
            return

        for name in self.param_set.keys():
            self._set_model_value_if_present(sbml_model, name, self.param_set[name])

    def _param_set_values(self):
        """The ``{param_id: value}`` snapshot of the current PSet, for resolving a
        parameter-reference condition perturbation (a per-condition estimated initial
        condition, ADR-0076). Empty when no PSet is applied."""
        if self.param_set is None:
            return {}
        return {name: self.param_set[name] for name in self.param_set.keys()}

    def _apply_mutant(self, mut, sbml_model):
        param_values = self._param_set_values()
        for mi in mut:
            current = self._get_model_value_if_present(sbml_model, mi.name)
            if current is None:
                continue
            self._set_model_value_if_present(
                sbml_model,
                mi.name,
                _mutate_scalar(current, mi.operation, mi.amount(param_values)),
            )

    def _build_sbml_doc(self, mut=None, scan_override=None, phase_overrides=None):
        doc = _sbml_doc_from_text(self._base_sbml_text, self.file_path)
        sbml_model = doc.getModel()
        self._apply_param_set(sbml_model)
        if mut:
            self._apply_mutant(mut, sbml_model)
        if scan_override is not None:
            scan_name, scan_value = scan_override
            self._set_model_value_if_present(sbml_model, scan_name, scan_value)
        # The pre-equilibration condition (ADR-0052) is the state the FIRST phase is
        # initialized in, so it is baked into the document like a mutant -- last, so it
        # wins over the fit vector, and before initialAssignment expansion, so a species
        # initial the condition drives is recomputed under it (#547).
        if phase_overrides:
            for name, value in phase_overrides:
                self._set_model_value_if_present(sbml_model, name, value)
        return doc

    def _get_engine_template(self):
        """Return the process-cached base engine model for this SBML text.

        Loads the bngsim model once per worker process (paying the libSBML
        parse) and reuses it for every evaluation, cloning it for each
        per-evaluation parameter application. See issue #415. The template is
        also *warmed* on first use, so the clones inherit the derived analytical
        Jacobian -- and, on the gradient path, the compiled sensitivity RHS --
        instead of each rebuilding it (:meth:`_warm_engine_template`, issues
        #543 and #544).
        """
        key = getattr(self, '_engine_template_key', None)
        if key is None:
            key = hashlib.sha256(self._base_sbml_text.encode('utf-8')).hexdigest()
            self._engine_template_key = key
        template = _ENGINE_TEMPLATE_CACHE.get(key)
        if template is None:
            template = self._load_bngsim_model_from_text(self._base_sbml_text)
            _ENGINE_TEMPLATE_CACHE[key] = template
        self._warm_engine_template(key, template)
        return template

    def _runs_ode_actions(self):
        """Does any action on this model integrate deterministically (#544)?

        The warm below builds artifacts only an ODE solve consumes: bngsim derives the
        analytical Jacobian and runs its large-model codegen under ``dispatch == 'ode'``
        and nowhere else, having deliberately moved both off the model-load path so that
        "non-ODE dispatch ... never pays the SymPy derivation or the codegen compile". A
        gillespie-only model would otherwise warm something no action of its own ever
        reads, which is a cost this backend does not pay today. ``actions`` is set in
        ``__init__``, before any template fetch; an empty list -- a template pulled by a
        structural query on a model that has none yet -- simply defers the warm to the
        first fetch that has one, which is what the warm's laziness is for anyway.
        """
        return any(self._resolve_method(act) == 'ode'
                   for act in getattr(self, 'actions', ()))

    def _warm_engine_template(self, key, template):
        """Build the ODE-solve artifacts on the template itself, once (#543, #544).

        Every ODE action needs a derived analytical Jacobian, and every *gradient* action
        additionally needs a compiled analytical sensitivity RHS. bngsim builds both in
        ``Simulator.__init__`` -- the Jacobian always, the sensitivity RHS whenever the
        constructor is handed ``sensitivity_params``/``sensitivity_ic``. PyBNF builds that
        Simulator on the per-action *clone* (:meth:`_prepare_engine_model` ->
        :meth:`_make_simulator`), so the clone is what derives and records them -- and the
        clone is thrown away at the end of the action. ``clone()`` copies both parent ->
        child but nothing copies them back, so the template stays cold forever and every
        action re-pays: the SymPy Jacobian derivation, and (on the gradient path) the
        regenerated C source and every symbolic derivative behind it, because bngsim keys
        its compiled ``.so`` on a hash of that source. Constructing one throwaway Simulator
        **on the template** writes both where ``clone()`` can find them, and every action
        from then on inherits them. It does not need to run: both happen at construction.

        Measured per action, the Smith rows as reported on #543 and #544 and the yeast row
        through this backend's own ``execute``:

        =========================  ================  =========  =========  ==========
        model                      path              unwarmed   warmed     re-derived
        =========================  ================  =========  =========  ==========
        Smith (133 sp, 16 cols)    sensitivity        2.015 s    0.537 s    4/4 -> 0/4
        Smith                      scalar             0.0401 s   0.0224 s  20/20 -> 0
        yeast_cell_cycle (44 sp)   scalar             0.1542 s   0.0057 s  10/10 -> 0
        =========================  ================  =========  =========  ==========

        Tensor and trajectory are bit-identical either way; only how many times the engine
        re-derives things that depend on model structure alone changes. The ``.net`` backend
        clones from a held ``_engine_model`` in this same shape and does re-attempt the
        derivation per evaluation (4 of 4), but a BNGL network is all-Elementary, so bngsim
        takes its closed-form C++ Jacobian instead of SymPy and the re-attempt costs
        essentially nothing (0.1511 -> 0.1472 s per evaluation on ``egfr_ground.net``, 356
        species). Left unwarmed rather than warmed on a number that does not justify it.

        Two shapes, and the weaker one must not satisfy the stronger. bngsim emits the
        output-sensitivity evaluator only when the model carries ``_want_output_sens`` --
        which the *constructor* sets from its own sensitivity arguments -- and clears a
        plain-RHS artifact to regenerate it at the first sensitivity request. So a
        scalar-warmed template is correct for a gradient fit but saves it nothing, and
        ``_template_is_warm`` (unlike :func:`_template_jacobian_is_warm`) answers False for
        one, leaving the gradient warm to run. Which parameters are requested does not
        enter into it: the emitted source covers the whole RHS, and only that flag varies,
        so the warm is keyed on the template's own state rather than on a shape tuple. The
        reverse direction needs nothing: a sensitivity warm derives the Jacobian on its way
        past.

        A warm that raises is recorded and swallowed, so it costs one attempt per process
        rather than one per action. Everything that can fail here -- a rate law bngsim cannot
        differentiate to closed form, a missing codegen backend -- fails identically when the
        action builds its own Simulator moments later, where :meth:`execute` gives it the
        refusal handling (and the #536 event-sensitivity diagnosis) it already has. An
        optimization must not become a new place for the fit to die.
        """
        if not self._runs_ode_actions():
            return
        kwargs = self._sensitivity_request_kwargs('ode')
        if kwargs:
            if _template_is_warm(template):
                return
        elif _template_jacobian_is_warm(template):
            return
        memo_key = (key, bool(kwargs))
        if _ENGINE_TEMPLATE_WARM_ATTEMPTED.get(memo_key) is template:
            return
        _ENGINE_TEMPLATE_WARM_ATTEMPTED[memo_key] = template
        try:
            bngsim.Simulator(template, method='ode', **kwargs)
        except Exception as exc:
            logger.debug(
                'Could not warm the engine template for model %s (%s: %s); each action '
                'will build its own Jacobian%s.', self.name, type(exc).__name__, exc,
                ' and sensitivity RHS' if kwargs else '')

    def _set_engine_value_if_present(self, engine_model, name, value):
        """Apply a value to the engine model. Returns True iff it is a species.

        Species values are converted from PyBNF units (amount or concentration,
        per the SBML species) to the concentration bngsim works in, matching the
        reload path (see :meth:`_compute_species_unit_factors`)."""
        if name in self._species_name_set:
            engine_model.set_concentration(
                name, float(value) * self._species_unit_factor[name])
            return True
        if name in self._global_param_names:
            engine_model.set_param(name, float(value))
        return False

    def _get_engine_value_if_present(self, engine_model, name):
        if name in self._species_name_set:
            return float(engine_model.get_concentration(name)) / self._species_unit_factor[name]
        if name in self._global_param_names:
            return float(engine_model.get_param(name))
        return None

    def _apply_param_set_engine(self, engine_model):
        """Apply self.param_set to the engine model. Returns True iff a species
        initial value was changed (so the caller must save_concentrations)."""
        touched_species = False
        if self.param_set is None:
            return touched_species
        for name in self.param_set.keys():
            if self._set_engine_value_if_present(engine_model, name, self.param_set[name]):
                touched_species = True
        return touched_species

    def _apply_mutant_engine(self, mut, engine_model):
        touched_species = False
        param_values = self._param_set_values()
        for mi in mut:
            current = self._get_engine_value_if_present(engine_model, mi.name)
            if current is None:
                continue
            new_value = _mutate_scalar(current, mi.operation, mi.amount(param_values))
            if self._set_engine_value_if_present(engine_model, mi.name, new_value):
                touched_species = True
        return touched_species

    def _prepare_engine_model(self, mut=None, scan_override=None, ic_overrides=None,
                              param_overrides=None, phase_overrides=None):
        """Clone the cached engine template and apply per-evaluation values.

        The fast-path analogue of ``_build_sbml_doc`` + reload: param_set,
        mutant deltas, and any scan override are applied in place via
        set_param/set_concentration on a cheap clone of the cached model,
        skipping the libSBML reparse + Jacobian re-derivation. ``ic_overrides``
        (species name -> initial concentration) sets the recomputed initial
        values of species whose initials are parameter-driven, and
        ``param_overrides`` the recomputed values of the derived parameters an
        initialAssignment fixes (#531); both are applied last so they win over a
        direct param_set/scan assignment, exactly as libSBML's own
        initialAssignment expansion does on the reload path. See issue #415.

        ``phase_overrides`` (name -> value pairs) is the pre-equilibration condition
        (ADR-0052, #547): the state the unmeasured first phase is initialized in,
        applied like a mutant so it wins over the fit vector.
        """
        engine_model = self._get_engine_template().clone()
        touched_species = self._apply_param_set_engine(engine_model)
        if mut:
            touched_species |= self._apply_mutant_engine(mut, engine_model)
        if scan_override is not None:
            scan_name, scan_value = scan_override
            if self._set_engine_value_if_present(engine_model, scan_name, scan_value):
                touched_species = True
        if phase_overrides:
            for name, value in phase_overrides:
                if self._set_engine_value_if_present(engine_model, name, value):
                    touched_species = True
        if param_overrides:
            for param_name, value in param_overrides.items():
                engine_model.set_param(param_name, float(value))
        if ic_overrides:
            for species_name, ic in ic_overrides.items():
                self._set_engine_value_if_present(engine_model, species_name, ic)
            touched_species = True
        if touched_species:
            engine_model.save_concentrations()
        engine_model.reset()
        return engine_model

    def _recompute_initial_assignments(self, mut=None, scan_override=None,
                                       phase_overrides=None):
        """Recompute the parameter-driven initial values for this evaluation.

        Builds the SBML doc with this evaluation's parameter/species/scan
        changes applied, evaluates its initialAssignments in place via libSBML
        (which resolves assignmentRule intermediates), and returns
        ``(species_overrides, param_overrides)`` for the species initials and
        the **derived parameters** an initialAssignment fixes at load. This
        reproduces the values a full reload would bake in, without re-deriving
        the Jacobian. See issues #415 and #531.
        """
        doc = self._build_sbml_doc(mut=mut, scan_override=scan_override,
                                   phase_overrides=phase_overrides)
        sbml_model = doc.getModel()
        libsbml.SBMLTransforms.expandInitialAssignments(sbml_model)
        species_overrides = {}
        for species_name in self._initial_expr_species:
            ic = self._get_model_value_if_present(sbml_model, species_name)
            if ic is not None:
                species_overrides[species_name] = ic
        param_overrides = {}
        for param_name in self._initial_expr_params:
            value = self._get_model_value_if_present(sbml_model, param_name)
            if value is not None:
                param_overrides[param_name] = value
        return species_overrides, param_overrides

    def _engine_model_for_action(self, mut=None, scan_override=None, phase_overrides=None):
        """Build the engine model for one action by reusing the cached engine
        template (see issue #415).

        Three paths, cheapest first: (1) clone + in-place set_param when no
        species initial is affected; (2) clone + recomputed initials when a
        parameter-driven initialAssignment is in play; (3) a full reload from
        SBML text only for models with an algebraicRule (which we do not
        analyze). All three avoid re-deriving the analytical Jacobian except
        the last.

        ``phase_overrides`` carries a pre-equilibration experiment's *equilibration*
        condition (ADR-0052, #547) into the build, so the unmeasured first phase starts
        from a model initialized under it -- initialAssignments included.
        """
        scan_param = scan_override[0] if scan_override is not None else None
        if self._needs_structural_reload(mut=mut, scan_param=scan_param):
            doc = self._build_sbml_doc(mut=mut, scan_override=scan_override,
                                       phase_overrides=phase_overrides)
            return self._load_bngsim_model_from_text(_sbml_doc_to_text(doc))
        ic_overrides = param_overrides = None
        if self._changes_touch_initials(mut=mut, scan_param=scan_param,
                                        phase_overrides=phase_overrides):
            if not _HAS_EXPAND_INITIAL_ASSIGNMENTS:
                # No in-place initial evaluation available -> reload to stay correct.
                doc = self._build_sbml_doc(mut=mut, scan_override=scan_override,
                                           phase_overrides=phase_overrides)
                return self._load_bngsim_model_from_text(_sbml_doc_to_text(doc))
            ic_overrides, param_overrides = self._recompute_initial_assignments(
                mut=mut, scan_override=scan_override, phase_overrides=phase_overrides)
        return self._prepare_engine_model(
            mut=mut, scan_override=scan_override, ic_overrides=ic_overrides,
            param_overrides=param_overrides, phase_overrides=phase_overrides)

    def model_text(self, mut=None):
        logger.info('Generating model text for %s', self.name)
        return _sbml_doc_to_text(self._build_sbml_doc(mut=mut))

    def save(self, file_prefix):
        with open(f'{file_prefix}.xml', 'w') as out:
            out.write(self.model_text())

    def save_all(self, file_prefix):
        for mut in self.mutants:
            with open(f'{file_prefix}{mut.suffix}.xml', 'w') as out:
                out.write(self.model_text(mut=mut))

    def add_action(self, action):
        if action.method not in ('ode', 'ssa'):
            raise PybnfError(
                f'time_course or param_scan method {action.method} is not currently supported with '
                'sbml_backend = bngsim. Options are ode or ssa.'
            )
        self.actions.append(action)
        self.suffixes.append((action.bng_codeword, action.suffix))
        if action.method == 'ssa':
            self.stochastic = True

    def get_suffixes(self):
        result = []
        for suffix in self.suffixes:
            for mut in self.mutants:
                result.append(suffix[1] + mut.suffix)
        return result

    @staticmethod
    def _data_with_headers(arr, headers):
        return Data.from_columns(arr, headers)

    def _species_value_factors(self, species_names):
        """Row of divisors taking a bngsim *concentration* to the PyBNF species value (#553).

        The reciprocal of :attr:`_species_unit_factor`, broadcast over a trajectory's species
        axis: the compartment size for an amount-declared (``hasOnlySubstanceUnits``) species
        and ``1.0`` for every other one. bngsim reports every species column as a
        concentration, but an amount-declared species' symbol *means its amount* -- in an SBML
        formula and equally in the PEtab observable formulas PyBNF evaluates against these
        columns -- so the column has to come back out of concentration units before anything
        reads it.

        This mirrors the single-value getter, which already returns
        ``get_concentration(name) / self._species_unit_factor[name]``. Every factor is 1.0
        for a concentration-valued species or a unit compartment, which is why this is a
        no-op on every model that was already correct.
        """
        return np.array(
            [self._species_unit_factor.get(name, 1.0) for name in species_names],
            dtype=float)[np.newaxis, :]

    def _result_to_data(self, result, *, stochastic=False):
        """Convert a bngsim Result to a PyBNF Data object.

        Scalar path: the species value columns alone (``output_sensitivities`` stays
        ``None``). Gradient path (``_sensitivity_request`` active, #385/#455): also attaches
        the native-space forward-sensitivity tensor keyed by the same ``species:<name>``
        selectors as the Data columns. The attachment never perturbs the value columns -- it
        is read off the same Result -- so the scalar path is byte-identical to before."""
        if stochastic:
            arr = result.as_roadrunner()
            return Data(named_arr=arr)
        species = np.asarray(result.species, dtype=float)
        arr = np.zeros((result.n_times, 1 + species.shape[1]))
        arr[:, 0] = result.time
        arr[:, 1:] = species / self._species_value_factors(result.species_names)
        headers = ['time'] + list(result.species_names)
        data = self._data_with_headers(arr, headers)
        if self._sensitivity_request is not None and getattr(
                result, 'has_sensitivities', False) or getattr(
                result, 'has_sensitivities_ic', False):
            data.output_sensitivities = self._extract_output_sensitivities(result)
        return data

    def _initial_state_derivative(self, engine_model, base_state, mut, name):
        """Numerically differentiate the initialized state with respect to ``name``.

        SBML ``initialAssignment`` expressions are evaluated while an engine model is
        prepared, before bngsim's ODE sensitivity system starts. Re-prepare the model at
        two nearby values so a t=0-only experiment retains those assignment derivatives
        (for example ``Rec2(0) = ini_R1 * ini_R2fold`` in Schwen_PONE2014).

        Both sides are converted out of concentration units before differencing (#553), so
        the result is ``d(PyBNF species value)/d(name)`` and matches the units ``base_state``
        arrives in -- which matters for the one-sided branches, where the two are mixed.
        """
        value = self._get_engine_value_if_present(engine_model, name)
        if value is None:
            return np.zeros_like(base_state)
        step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(value))
        value_factors = self._species_value_factors(engine_model.species_names)[0]

        states = {}
        for direction in (-1.0, 1.0):
            try:
                perturbed = self._engine_model_for_action(
                    mut=mut, scan_override=(name, value + direction * step))
                state = np.asarray([
                    perturbed.get_concentration(species)
                    for species in engine_model.species_names
                ], dtype=float) / value_factors
                if np.all(np.isfinite(state)):
                    states[direction] = state
            except Exception:
                # A one-sided difference remains valid when an expression's domain or a
                # model bound makes one perturbation invalid.
                pass
        if -1.0 in states and 1.0 in states:
            return (states[1.0] - states[-1.0]) / (2.0 * step)
        if 1.0 in states:
            return (states[1.0] - base_state) / step
        if -1.0 in states:
            return (base_state - states[-1.0]) / step
        raise PybnfError(
            f"Model {self.name}: could not differentiate the initialized state with "
            f"respect to '{name}' for a t=0-only experiment.")

    def _initial_state_data(self, engine_model, *, method, mut):
        """Return the initialized SBML state as a one-row trajectory at ``t=0``.

        Both bngsim integrators require a positive-duration span, but an experiment
        whose only measurement is at ``t=0`` needs no integration (#510). Construct
        the gradient tensor directly too: ordinary initial conditions contribute an
        identity column, while parameters/species used by an ``initialAssignment`` are
        differentiated through that expression by re-preparing the initialized model.
        """
        species_names = list(engine_model.species_names)
        # Same concentration -> PyBNF-value conversion the integrated path applies (#553).
        value_factors = self._species_value_factors(species_names)[0]
        state = np.asarray([
            engine_model.get_concentration(name) for name in species_names
        ], dtype=float) / value_factors
        data = self._data_with_headers(
            np.concatenate(([0.0], state))[np.newaxis, :], ['time'] + species_names)

        # Preserve the usual method/differentiability gate even though no Simulator is
        # built: a scored SSA action still cannot supply a gradient, while an unscored
        # SSA diagnostic remains sensitivity-free.
        self._sensitivity_request_kwargs(method)
        req = self._sensitivity_request
        if req is None or method != 'ode' or (not req.params and not req.ic):
            return data

        selectors = ['species:%s' % name for name in species_names]
        d_param = None
        if req.params:
            d_param = np.zeros((1, len(species_names), len(req.params)), dtype=float)
            initial_deps = self._initial_dep_names
            for axis, name in enumerate(req.params):
                if initial_deps is None or name in initial_deps:
                    d_param[0, :, axis] = self._initial_state_derivative(
                        engine_model, state, mut, name)
        d_ic = None
        if req.ic:
            d_ic = np.zeros((1, len(species_names), len(req.ic)), dtype=float)
            species_index = {name: i for i, name in enumerate(species_names)}
            initial_deps = self._initial_dep_names
            for axis, name in enumerate(req.ic):
                if (initial_deps is None or name in initial_deps
                        or name in self._initial_expr_species):
                    d_ic[0, :, axis] = self._initial_state_derivative(
                        engine_model, state, mut, name)
                elif name in species_index:
                    # d(PyBNF species value)/d(PyBNF IC value) for the species itself. The
                    # concentration carries the unit factor and the reported value divides it
                    # straight back out (#553), so the identity column is 1.0 in both units.
                    d_ic[0, species_index[name], axis] = 1.0
        data.output_sensitivities = OutputSensitivities(
            selectors=selectors, param_names=list(req.params), ic_species=list(req.ic),
            d_param=d_param, d_ic=d_ic)
        return data

    def _extract_output_sensitivities(self, result):
        """Read the native-space ∂(species)/∂θ tensor off a sensitivity-bearing SBML Result.

        Selectors mirror the Data's species columns (``species:<name>`` -- the kind bngsim
        reports SBML output sensitivities under, GH #205). The ``parameter`` axis is read
        whenever sensitivity params were requested; the ``ic`` axis whenever IC species were.
        The IC axis is rescaled by each species' PyBNF-value -> bngsim-concentration factor, so
        ``d_ic`` is ``∂(species concentration)/∂(PyBNF IC value)`` -- consistent with how a free
        IC parameter is applied through ``set_concentration(name, value * unit_factor)``; the
        factor is 1.0 for a concentration species (the common case), so this is a no-op there.
        """
        species_names = list(result.species_names)
        selectors = ['species:%s' % name for name in species_names]
        param_names = list(result.sensitivity_params)
        ic_species = list(result.sensitivity_ic_species)
        # ∂(concentration)/∂θ -> ∂(PyBNF species value)/∂θ, down the species axis, so the
        # tensor stays in the units _result_to_data hands the value columns over in (#553).
        value_factors = self._species_value_factors(species_names)[0][
            np.newaxis, :, np.newaxis]
        d_param = None
        if param_names:
            d_param = np.asarray(
                result.output_sensitivities(selectors, axis='parameter'), dtype=float)
            d_param = d_param / value_factors
        d_ic = None
        if ic_species:
            d_ic = np.asarray(
                result.output_sensitivities(selectors, axis='ic'), dtype=float)
            # ∂/∂(concentration IC) -> ∂/∂(PyBNF IC value): chain by the unit factor per axis.
            ic_factors = np.array(
                [self._species_unit_factor[s] for s in ic_species], dtype=float)
            d_ic = d_ic * ic_factors[np.newaxis, np.newaxis, :]
            d_ic = d_ic / value_factors
        return OutputSensitivities(
            selectors=selectors, param_names=param_names, ic_species=ic_species,
            d_param=d_param, d_ic=d_ic,
        )

    def _scan_point_sensitivities(self, result):
        """Per-dose-point forward-sensitivity tensor for a reset-to-seed SBML scan point (#476).

        ``None`` on the scalar path (no request) or when the point's ``Result`` carries no
        tensor; otherwise the full per-point :class:`~pybnf.data.OutputSensitivities` (all
        rows), whose final row the scan stacks down the dose axis
        (:func:`pybnf.data.stack_scan_sensitivities`). Each dose point is an independent,
        reset-to-seed ODE run, so ``∂species/∂θ`` at its end-of-run row is well-posed."""
        if self._sensitivity_request is None or not getattr(
                result, 'has_sensitivities', False) or getattr(
                result, 'has_sensitivities_ic', False):
            return None
        return self._extract_output_sensitivities(result)

    def enable_output_sensitivities(self, *, params=None, ic=None):
        """Activate the gradient path: request forward sensitivities ∂g/∂θ (#385/#455).

        Routes ``params`` (native global model parameter ids) to
        ``Simulator(sensitivity_params=)`` and ``ic`` (species names) to
        ``Simulator(sensitivity_ic=)`` at every subsequent ODE run, carrying the resulting
        tensor onto each simulated :class:`Data`. The routing lists themselves come from #448;
        this method only stores them. Gates on the backend capability (#447) exactly as the net
        backend does: a build without forward output sensitivities refuses here with an
        actionable message rather than failing deep in the backend. The version floor is
        unaffected -- scalar (metaheuristic) fits never call this.
        """
        if not BNGSIM_HAS_OUTPUT_SENS:
            reason = feature_missing_reason('output_sensitivities')
            raise PybnfError(
                "Gradient-based fitting needs forward output sensitivities, which this bngsim "
                "build does not provide (%s). Install a bngsim build with the "
                "'output_sensitivities' feature, or run a gradient-free fit."
                % (reason or 'feature unavailable')
            )
        self._sensitivity_request = _SensitivityRequest(
            params=list(params or []), ic=list(ic or []),
        )

    def set_scored_suffixes(self, suffixes):
        """Record which output suffixes are scored gradient targets (#475/#482).

        The SBML twin of :meth:`BngsimModel.set_scored_suffixes`. On the gradient path
        only a SCORED action's output needs forward sensitivities; an incidental/unscored
        action -- a stochastic (ssa) diagnostic that no data scores -- runs sensitivity-free
        so it neither computes a wasted tensor nor aborts the fit at a differentiability
        guard it can never satisfy. ``suffixes`` is the model's ``exp_data`` mapping (or any
        iterable of scored *full* suffixes -- the mutant/condition suffix is already folded
        into each action's key here, since the mutant loop is inline in :meth:`execute`, so
        pass the full set once and every condition's output is keyed by ``act.suffix +
        mut.suffix``). Set by the gradient optimizer's ``_setup_gradient_path`` before the
        model scatter, so it rides the pickle to the workers alongside the sensitivity request.
        """
        self._scored_suffixes = set(suffixes)

    def _action_bears_sensitivities(self):
        """Whether the action currently being prepared is a scored gradient target.

        The per-action half of the #475 gate (SBML twin of
        :meth:`BngsimModel._action_bears_sensitivities`): an action carries forward
        sensitivities only on the gradient path (``_sensitivity_request`` active) AND when
        its full output suffix -- ``act.suffix + mut.suffix``, set into
        :attr:`_current_action_suffix` by :meth:`execute` -- is in the scored set. The
        scalar path returns ``False`` -- no action bears sensitivities there. On the
        gradient path a scored set that is unknown, or a current suffix not yet resolved,
        returns ``True`` (bearing), so any path that activates the request without declaring
        scored suffixes keeps the historical all-actions-bearing behavior.
        """
        if self._sensitivity_request is None:
            return False
        scored = self._scored_suffixes
        if scored is None:
            return True
        suffix = self._current_action_suffix
        if suffix is None:
            return True
        return suffix in scored

    @property
    def has_discrete_events(self):
        """True iff the engine model contains state-jumping discrete events (#461/#536).

        SBML ``event``\\ s reinitialise the integrator state discontinuously, so a
        forward-sensitivity vector carried across one is right only if the solver
        applies the event's own jump at each fire. bngsim originally did not, and
        refused sensitivities on any event-bearing model rather than return stale
        derivatives (bngsim GH #205); the gradient path reads this property as its
        pre-flight differentiability gate
        (:meth:`GradientOptimizer._require_differentiable_dynamics`) so that refusal
        arrives **up front**, with an actionable "use a metaheuristic job_type"
        message, instead of at the first sensitivity-bearing ``simulate()``. bngsim
        now applies the jump and refuses only the subclasses it cannot cross, so on a
        build :data:`~pybnf._bngsim_caps.BNGSIM_HAS_EVENT_SENS` reports (a published
        capability since #558, not a version compare) the gate no longer fires. The net backend's property documents the same
        contract; this is its SBML twin -- and, since a ``.net`` model cannot author
        events, the one that is reachable in practice.

        Only true state-jumping events are counted (the engine core's ``n_events``).
        ``False`` when the engine model or its event count is unavailable (an
        older/stub backend), so the gate never blocks on a missing signal.
        """
        core = getattr(self._get_engine_template(), '_core', None)
        return bool(getattr(core, 'n_events', 0))

    def backend_ic_sensitivity(self):
        """``{species: {param: d(x(0))/d(param)}}`` the backend will seed the run with (#537).

        The reader for lanl/bngsim#155. ``output_sensitivities(axis='parameter')`` is the
        **total** derivative,

            d_param[p] = (right-hand-side path) + sum_k (d(x_k(0))/dp) * d_ic[x_k]

        so any seeding reported here is *already inside* a parameter's own axis, and the router
        must add an initial-condition term only for a ``(species, param)`` pair reported
        **absent**. A present entry whose value is ``0.0`` means seeded with a coefficient that
        vanishes at this state, which is not the same as absent and must not drop a column the
        fit needs at another point.

        Answered from model structure alone -- no simulation -- so the routing can be built once
        at setup, and state-dependent by design, so it is read from the configured model.
        ``None`` when the backend cannot say, which the router refuses rather than guessing at.
        """
        model = self._get_engine_template()
        reader = getattr(model, 'effective_ic_sensitivity', None)
        if reader is None:
            return None
        try:
            seeded = reader()
        except Exception:                                   # pragma: no cover - defensive
            return None
        # Restricted to species PyBNF reads on the SAME scale the backend does. The ``ic`` axis
        # is rescaled by each species' PyBNF-value -> concentration factor when the tensor is
        # read (:meth:`_extract_output_sensitivities`) and the ``parameter`` axis is not, so for
        # a species whose factor is not 1 the two axes are in different units and the parameter
        # axis cannot stand in for an ic term: substituting one for the other on a
        # ``hasOnlySubstanceUnits`` species in a size-2 compartment overstates its column by
        # exactly 2 (measured; ``test_sbml_fd_oracle_amount_species_seed`` is the oracle).
        # Reporting such a species as unseeded keeps the router on the ic route, which the same
        # oracle validates. No model in the PEtab benchmark subset both seeds an initial
        # condition and carries a non-unit factor, so nothing real is given up.
        return {species: row for species, row in seeded.items()
                if self._species_unit_factor.get(species, 1.0) == 1.0}

    def sensitivity_entity_namespace(self):
        """The bind-by-id namespaces the gradient router classifies free parameters against (#448).

        Returns ``(param_values, species_initializers, ic_seed_map)``:

        * ``param_values`` -- the model's global ``parameter`` ids mapped to their nominal
          values: the kinetic ids a free parameter binds to via ``set_param`` and thus routes to
          ``Simulator(sensitivity_params=)``, plus the environment a point-dependent seed
          derivative is evaluated in (#530);
        * ``species_initializers`` -- ``(species, initial-expr)`` pairs in the shape
          :func:`pybnf.gradient.routing.classify_free_param` expects. A free parameter named
          for a species sets that species' initial value (via ``set_concentration``, the
          bind-by-id convention, ADR-0034), so each species' bare initializer expression *is*
          its own name; such a free parameter routes to the initial-condition axis keyed by the
          species (an IC parameter is absent from the ODE RHS, so its parameter axis is zero);
        * ``ic_seed_map`` -- ``{model parameter -> SeedTerms}``, the initial values each
          parameter seeds and the ``d(entity)/d(parameter)`` of each
          (:meth:`_compute_ic_seed_map`), so the router can route a free parameter a condition
          assigns to that parameter onto those columns (a per-condition estimated initial
          condition, ADR-0076, #511/#530); a non-routable seed maps to ``None``.

        This is the only model coupling :mod:`pybnf.gradient.routing` needs, so the routing core
        stays backend-agnostic. No simulation -- all three are known at load time.
        """
        return (dict(self._nominal_param_values),
                [(s, s) for s in self._species_names],
                dict(self._ic_seed_map))

    def _sensitivity_request_kwargs(self, method):
        """Simulator kwargs requesting forward sensitivities on the gradient path.

        Returns ``{}`` on the scalar path (request inactive), so the Simulator construction is
        byte-identical to before this feature existed. On the gradient path it returns
        ``sensitivity_params=``/``sensitivity_ic=`` for an ODE Simulator.

        A non-ODE method (ssa) is non-differentiable -- forward sensitivities are
        deterministic-ODE only (#447). Whether that is an error now depends on whether the
        action's output is a *scored* gradient target (#475/#482): a scored non-ODE action
        still refuses cleanly (a PyBNF-level error, not a backend traceback -- its gradient
        genuinely cannot be supplied), while an incidental/unscored non-ODE action (a
        stochastic diagnostic that no data scores) runs sensitivity-free rather than aborting
        the whole fit. ODE actions are always built sensitivity-bearing (an unscored ODE
        action's unused tensor is harmless), matching the net backend.
        """
        req = self._sensitivity_request
        if req is None:
            return {}
        if method != 'ode':
            if self._action_bears_sensitivities():
                raise PybnfError(
                    "Model %s: gradient-based fitting requires deterministic ODE integration, but a "
                    "scored simulate() action requests method=%r. Forward output sensitivities are "
                    "available only for the ODE backend; run a gradient-free fit for stochastic "
                    "simulation." % (self.name, method)
                )
            # Incidental/unscored non-ODE action (#475): no sensitivities needed, so build a
            # plain Simulator and let it run on the ordinary path.
            return {}
        kwargs = {}
        if req.params:
            kwargs['sensitivity_params'] = list(req.params)
        if req.ic:
            kwargs['sensitivity_ic'] = list(req.ic)
        return kwargs

    @classmethod
    def _scan_point_to_row(cls, result, scan_value, scan_label):
        final_species = np.asarray(result.species[-1, :], dtype=float)
        row = np.concatenate((
            np.array([scan_value, float(result.time[-1])], dtype=float),
            final_species,
        ))
        headers = [scan_label, 'time'] + list(result.species_names)
        return row, headers

    @staticmethod
    def _write_saved_output(path, data):
        headers = [data.headers[i] for i in range(data.data.shape[1])]
        np.savetxt(path, data.data, header=' '.join(headers))

    def _resolve_method(self, action):
        if action.method == 'ssa' or self.integrator == 'gillespie':
            return 'ssa'
        return 'ode'

    def _make_simulator(self, engine_model, method):
        kwargs = {'method': method}
        if method == 'ssa':
            kwargs['strict_ssa'] = getattr(self, 'strict_ssa', True)
        # Gradient path (#385/#455): a no-op on the scalar path (empty kwargs), so the
        # Simulator construction is byte-identical there.
        kwargs.update(self._sensitivity_request_kwargs(method))
        try:
            return bngsim.Simulator(engine_model, **kwargs)
        except bngsim.SsaValidationError as exc:
            raise ModelError(str(exc)) from exc

    def _atol_loosens(self):
        """Whether this model's ``sbml_atol`` opted into loosening (#557).

        True for both non-numeric settings: ``tracking`` builds its ceiling from the same
        derivation ``auto`` produces, so at depth 0 it *is* ``auto`` -- a strict
        extension, which is the property that makes one flag correct for both rather than
        an accident that two modes happen to share.
        """
        return getattr(self, '_atol_mode', None) in (_ATOL_AUTO, _ATOL_TRACKING)

    def _effective_tolerances(self, sim):
        """The ``(rtol, scalar atol)`` pair this model is measured in (#546).

        ``sbml_rtol`` / a **pinned** ``sbml_atol`` win outright when stated. Otherwise
        rtol is the backend's own default (read off the constructed ``Simulator`` so this
        tracks bngsim rather than a copy of its constant) and atol is derived from the
        model's typical species magnitude by :func:`_derive_atol` -- which can only
        tighten, so a model of order-one scale keeps the default pair exactly, unless
        ``sbml_atol = auto`` / ``tracking`` (#557) has lifted that ceiling, in which case
        a model living *above* one gets the looser tolerance its own scale asks for.

        Under ADR-0105 this scalar is no longer always the *integration* tolerance: when
        the per-species vector is in force it becomes the steady-state convergence cutoff
        instead (:meth:`_run_tolerance_kwargs`), which is the one place a single number
        about the model's magnitude is still the right shape. It remains the integration
        tolerance everywhere the vector does not reach -- an older bngsim, an explicit
        ``sbml_atol``, a model with no nominal state to read.

        The scale is the model's **nominal** one (``_compute_nominal_state_scale``, read
        off the document at load), not the state this run is about to integrate, so the
        pair is a constant of the model rather than of the fit point. That matters twice
        over: a tolerance that moved with a fitted initial condition would put a step in
        the objective everywhere the derivation crossed a rounding boundary, and the
        scalar (line-search) evaluations of a gradient fit have to be integrated to the
        same accuracy as the sensitivities they are compared against, or the two disagree
        about what the objective is.

        Memoized per model: it is the same answer for every action of every evaluation,
        and the memo is what the ``sim``-read backend defaults are consulted through.
        """
        cached = getattr(self, '_tolerance_cache', None)
        if cached is not None:
            return cached
        rtol = getattr(self, '_config_rtol', None)
        if rtol is None:
            rtol = float(getattr(sim, '_rtol', _BNGSIM_DEFAULT_RTOL))
        atol = getattr(self, '_config_atol', None)
        if atol is None:
            default_atol = float(getattr(sim, '_atol', _BNGSIM_DEFAULT_ATOL))
            scale = getattr(self, '_nominal_state_scale', None)
            name = getattr(self, 'name', None)
            atol = _derive_atol(scale, rtol, default_atol, loosen=self._atol_loosens(),
                                model_name=name,
                                warned=getattr(self, '_atol_floor_warned', None))
            if atol != default_atol:
                logger.debug(
                    'bngsim SBML model %s: typical species magnitude %g, so the ODE '
                    'absolute tolerance is %g rather than the backend default %g (%s).',
                    name, scale, atol, default_atol,
                    'looser -- sbml_atol opted into the derivation in both directions'
                    if atol > default_atol else 'tighter')
        self._tolerance_cache = (float(rtol), float(atol))
        return self._tolerance_cache

    def _per_species_atol(self, sim, engine_model):
        """The per-species ``atol`` vector for this model, or ``None`` (ADR-0105).

        ``None`` means "there is nothing here a scalar does not already say", and every
        route to it leaves ADR-0103's behaviour untouched:

        * **the backend cannot take one** -- ``BNGSIM_HAS_PER_SPECIES_ATOL`` is a name
          probe rather than a version floor, because the build that first carried
          lanl/bngsim#196 declares the same version string as the wheel 25 commits behind
          it;
        * **``sbml_atol`` pins a number** -- an explicit scalar is the documented
          off-switch for this whole change, and it pins the pre-#196 ``CVodeSStolerances``
          path bit-for-bit, ulp included. ``auto`` / ``tracking`` (#557) are *not* that:
          they state how far the derivation may go, so the vector stays on and gains its
          released upper end;
        * **there is no nominal state to read** (a non-constant compartment volume makes
          the unit factors parameter-dependent), so there is nothing to derive from;
        * **the engine model's species do not line up** with the document's, so the
          vector could not be ordered without guessing -- a mis-ordered vector assigns
          one species' tolerance to another and nothing downstream would say so;
        * **the vector is elementwise the scalar** -- true of 19 of the 23 subset-I
          slugs under the clamped derivation, which is what "ADR-0103 had nothing to give
          back here" looks like: a
          model the scalar derivation left at the backend default has no over-tightening
          to undo. Handing CVODE a constant vector instead of the constant it already has
          buys nothing and costs the ~1 ulp ``cvEwtSetSS``/``cvEwtSetSV`` difference #196
          documents, so those models keep the scalar call they make today, byte for byte.
          Measured: the same 100 box points under a uniform vector and under the scalar
          fail identically, 39 and 39.

        Ordered to the *engine* model's ``species_names`` rather than the document's:
        those are the names bngsim indexes its state by, and are what the unit-factor
        table is already keyed on elsewhere in this class. ``normalize_atol_vector``
        (lanl/bngsim#212) takes the length/position contract here, once at setup, rather
        than at the first ``run()`` -- which on a fit is a long way from where the vector
        was built. Memoized against the ordering it was built for, so a later action with
        a different species list rebuilds rather than silently reusing.

        **A hand-written ``species_atol`` (#586) reaches none of those routes to ``None``.**
        Every one of them says "nothing was stated here that a scalar does not already
        cover", which is exactly what a map on the ``model:`` line is not: the species it
        names take the number stated, and the ones it does not keep whatever the
        derivation, a pinned scalar, or the backend default would have given them. So the
        map is an override *layer* over that base rather than a fourth way to build it,
        and the base is computed here exactly as it is for a model that states nothing --
        except that a pinned ``sbml_atol`` still switches the derivation off, so its base
        is that number broadcast rather than a vector derived around it.
        """
        stated = getattr(self, '_config_species_atol', None)
        pinned = getattr(self, '_config_atol', None)
        if not BNGSIM_HAS_PER_SPECIES_ATOL:
            # Unreachable with a stated map: the constructor refuses that outright rather
            # than degrading to the scalar (:meth:`_init_common_attrs`).
            return None
        if pinned is not None and not stated:
            return None
        names = tuple(getattr(engine_model, 'species_names', None) or ())
        cached = getattr(self, '_atol_vector_cache', None)
        if cached is not None and cached[0] == names:
            return cached[1]
        base = None if pinned is not None else self._derive_per_species_atol(sim, names)
        vector = self._apply_stated_per_species_atol(sim, names, base) if stated else base
        self._atol_vector_cache = (names, vector)
        return vector

    def _apply_stated_per_species_atol(self, sim, names, base):
        """Lay a hand-written ``species_atol`` over the vector this model would have had (#586).

        ``base`` is the derived vector when there is one and ``None`` when there is not --
        a model whose species share one scale, an ordering the derivation could not take,
        or a pinned ``sbml_atol``. ``None`` becomes the model's scalar broadcast over every
        species, which is the tolerance those species are integrated at today, so the map
        moves exactly the species it names and nothing else.

        **A stated entry is used verbatim.** Neither clamp applies to it: not ADR-0103's
        only-ever-tighten ceiling, which ADR-0114 established is a no-regression rule about
        the *derivation* rather than a property of the model, and not ADR-0105's
        model-scalar floor, which is a statement about how far a derivation read off
        *initial values* may reach for on its own. A number a person wrote is neither of
        those; it is the same kind of statement a pinned ``sbml_atol`` is, and it wins for
        the same reason. The one bound kept is ``parse_species_atol_setting``'s: finite and
        strictly positive, which is CVODE's own requirement.

        Nor does the "elementwise the scalar, so send the scalar" off-ramp apply. That
        exists so a model with no over-tightening to give back keeps its ``CVodeSStolerances``
        call byte for byte (19 of 23 subset-I slugs), and it is correct precisely because
        nobody asked for a vector. Here somebody did, so a map whose entries happen to
        agree with the scalar still travels as a vector, and pays #196's ~1 ulp
        ``cvEwtSetSS``/``cvEwtSetSV`` difference for having been stated.
        """
        stated = self._config_species_atol
        if not names:
            raise ModelError(
                f'species_atol was stated for model {self.name}, but bngsim reports no '
                'species to order it against.')
        unknown = sorted(name for name in stated if name not in set(names))
        if unknown:
            # The constructor already checked this against the engine model it loaded, so
            # reaching here means the action's engine model carries a different species
            # list. Raised rather than warned-and-dropped: the config author stated these.
            raise ModelError(
                f'species_atol for model {self.name} names {len(unknown)} species this '
                f'action does not integrate: {", ".join(unknown)}.')
        _, scalar_atol = self._effective_tolerances(sim)
        derived = base is not None
        if not derived:
            base = [scalar_atol] * len(names)
        vector = [stated.get(name, atol) for name, atol in zip(names, base)]
        logger.debug(
            'bngsim SBML model %s: species_atol states %d of %d species by hand (%s); the '
            'rest keep the %s. Range [%g, %g], steady-state cutoff %g.',
            getattr(self, 'name', None), len(stated), len(names),
            ', '.join(f'{n}={stated[n]:g}' for n in sorted(stated)),
            'tolerance the derivation gives them' if derived else 'model-wide scalar',
            min(vector), max(vector), scalar_atol)
        return bngsim.normalize_atol_vector(
            vector, len(names), names, where='the per-model species_atol')

    def _derive_per_species_atol(self, sim, names):
        """Build (and validate, and log) the vector :meth:`_per_species_atol` memoizes.

        The "says nothing a scalar does not" test is taken *before* the ordering one, so
        the models that only ever reach the ordering fallback -- ones whose species are
        all at or above one, where the vector is elementwise the default -- fall back
        silently rather than warning about a difference that would not have existed.
        """
        nominal = getattr(self, '_nominal_species_values', None)
        model_name = getattr(self, 'name', None)
        if not names or not nominal:
            return None
        rtol, scalar_atol = self._effective_tolerances(sim)
        default_atol = float(getattr(sim, '_atol', _BNGSIM_DEFAULT_ATOL))
        by_species = dict(zip(nominal, _derive_atol_vector(
            list(nominal.values()), scalar_atol, rtol, default_atol,
            loosen=self._atol_loosens())))
        if all(atol == scalar_atol for atol in by_species.values()):
            return None
        missing = set(names) - set(by_species)
        if missing:
            # Not a failure worth raising over -- the scalar is a correct tolerance for
            # this model, just a less discriminating one -- but not silent either, because
            # the alternative reading ("the vector is on") would be wrong. bngsim renames a
            # species whose SBML id collides with an Antimony reserved word, which is how
            # Smith_BMCSystBiol2013 gets here: its ``NULL``/``null`` load as
            # ``_ant_NULL``/``_ant_null``.
            logger.warning(
                'bngsim SBML model %s: the engine model integrates %d species the SBML '
                'document does not declare under those names (%s), so the DERIVED '
                'per-species absolute tolerance cannot be ordered against the state '
                'without guessing; %s.',
                model_name, len(missing), ', '.join(sorted(missing)[:4]),
                'the species named by species_atol still take the tolerances stated for '
                'them, and every other species integrates at the scalar'
                if getattr(self, '_config_species_atol', None)
                else 'integrating at the scalar tolerance instead')
            return None
        vector = [by_species[name] for name in names]
        moved = sum(1 for atol in vector if atol != scalar_atol)
        logger.debug(
            'bngsim SBML model %s: per-species ODE absolute tolerance over %d species, '
            '%d of them away from the %g a scalar derivation gives the whole model; '
            'range [%g, %g], steady-state cutoff %g.',
            model_name, len(vector), moved, scalar_atol, min(vector), max(vector),
            scalar_atol)
        return bngsim.normalize_atol_vector(
            vector, len(names), names, where='the derived per-species sbml_atol')

    def _run_tolerance_kwargs(self, sim, engine_model):
        """The tolerance kwargs one deterministic ``Simulator.run`` takes (ADR-0105).

        A vector ``atol`` travels with an explicit ``steady_state_tol``, and that pairing
        is the whole of why this is not two independent decisions. bngsim falls its
        steady-state cutoff back to the *scalar* atol when unset, "also when a per-species
        atol is in force (issue #196): the criterion is a single norm over every species
        and has no per-species reading to take" -- and the scalar it falls back to is the
        Simulator's own ``1e-8``, not anything derived from the vector. Left alone, that
        silently reverts ADR-0103's steady-state fix: on a model whose states are ~1e-8,
        ``||dx/dt|| < 1e-8`` holds *at t = 0*, so every ``time = inf`` measurement
        (ADR-0086) and every pre-equilibration phase (ADR-0052/0104) would return the
        initial state and call it equilibrium -- looking, as it did before, like a
        modelling problem rather than a tolerance one.

        So ADR-0103's median-derived scalar does not retire. It stops being the
        integration tolerance and becomes the convergence criterion, which keeps one
        statement about the model's magnitude governing both and reproduces today's
        steady-state behaviour exactly.

        ``sbml_atol = tracking`` (#557) takes the same pairing for the same reason, one
        step further along: bngsim resolves a ``TrackingAtol`` to *its own* scalar for
        every scalar-shaped consumer, so the cutoff has to be stated there too. What
        travels is the derived vector as the tracking **ceiling** -- a constant of the
        model, read off its nominal state -- with the trajectory-following happening
        strictly *beneath* it. bngsim also ships a bare ``"tracking"`` whose ceiling is
        derived from the **live** state at every ``run``; that is the shape ADR-0105 ruled
        out for the vector and it is ruled out here for the same reason, since a fit that
        moves initial conditions would then put a step in the objective wherever the
        derivation crossed a rounding boundary.

        When the vector declines -- a model whose species share one scale, an ordering it
        cannot take, a bngsim without lanl/bngsim#196 -- tracking still applies, with the
        scalar as a broadcast ceiling. That case is not a degradation: what tracking adds
        is *within*-species and over time, so it is orthogonal to whether the species
        differ from each other, and a model whose species all start at one is exactly
        where a decay into nothing is invisible to any vector read off initial values.
        """
        rtol, scalar_atol = self._effective_tolerances(sim)
        vector = self._per_species_atol(sim, engine_model)
        if getattr(self, '_atol_mode', None) == _ATOL_TRACKING:
            decades = getattr(self, '_atol_tracking_decades', None)
            spec = dict(ceiling=scalar_atol if vector is None else vector)
            if decades is not None:
                spec['decades'] = float(decades)
            return {'rtol': rtol, 'atol': bngsim.TrackingAtol(**spec),
                    'steady_state_tol': scalar_atol}
        if vector is None:
            return {'rtol': rtol, 'atol': scalar_atol}
        return {'rtol': rtol, 'atol': vector, 'steady_state_tol': scalar_atol}

    def _run_simulation(self, engine_model, end_time, n_points, *, method='ode',
                        seed=None, timeout=None, sample_times=None, steady_state=False,
                        suffix=None, sim=None, carry_sensitivities=False):
        """Run one phase on this model.

        ``sim`` reuses an already-constructed :class:`bngsim.Simulator` instead of
        building a fresh one. That is what makes a pre-equilibration protocol possible
        here (ADR-0052, #547): a second ``run()`` on the *same* persistent simulator
        continues from the state the first one left, which is exactly the carry-over the
        measured phase needs. ``carry_sensitivities`` is the gradient-path companion --
        bngsim refuses to seed forward sensitivities on a carried-over state unless it is
        told to seed them from the prior phase's own ``dx/dtheta`` (GH #210).
        """
        if sim is None:
            sim = self._make_simulator(engine_model, method)
        run_kwargs = {}
        if carry_sensitivities:
            run_kwargs['carry_sensitivities'] = True
        if method != 'ssa':
            # CVODE tolerances (#546, ADR-0105). A stochastic run has none to set, so the
            # ssa path is byte-identical to before.
            run_kwargs.update(self._run_tolerance_kwargs(sim, engine_model))
        if steady_state:
            # A steady-state measurement (ADR-0086, #521): relax to equilibrium
            # (early-stop on ||dx/dt||) with ``end_time`` only the max-time bound, rather
            # than integrating to a fixed endpoint. This is bngsim's own parity primitive
            # -- the same one the BNGL ``simulate(steady_state=>1)`` path and the
            # steady-state parameter_scan use -- so all three agree on what "the steady
            # state" means, and it stays forward-sensitivity differentiable.
            run_kwargs['steady_state'] = True
        if timeout is not None:
            try:
                timeout_value = float(timeout)
            except (TypeError, ValueError):
                timeout_value = 0.0
            if timeout_value > 0.0:
                run_kwargs['timeout'] = timeout_value
        if method == 'ssa':
            if seed is None:
                seed = secrets.randbits(31) or 1
            run_kwargs['seed'] = seed
        if sample_times is not None:
            # New-era explicit output points (ADR-0028, #469/#470): output at exactly
            # the experiment's measurement times instead of a uniform n_points grid,
            # mirroring the native BNGL path (net_model._run_prepared_simulate). Without
            # this the SBML/Antimony sim only lands on integer-spaced times, so a data
            # point at a non-grid time (e.g. t=0.5) is never in the output and scoring
            # fails. explicit_points always includes t=0 (TimeCourse), so integration
            # starts from the model baseline.
            pts = [float(p) for p in sample_times]
            return sim.run(
                t_span=(pts[0], pts[-1]),
                n_points=len(pts),
                sample_times=pts,
                **run_kwargs,
            )
        result = sim.run(t_span=(0.0, float(end_time)), n_points=int(n_points), **run_kwargs)
        if steady_state and not int(
                (getattr(result, 'solver_stats', None) or {}).get('steady_state_reached', 0)):
            # Warn-and-score-last-value (ADR-0046): a point that will not equilibrate inside
            # the bound is still scored -- at the furthest relaxation reached -- so the
            # optimizer can walk out of it, but the user hears about it.
            logger.warning(
                'bngsim SBML model %s: action %s did not reach steady state within its '
                't_end=%s bound; scoring the state reached there.',
                self.name, suffix, end_time)
        return result

    def _resolve_action_seed(self, *, explicit_seed, action_index, suffix, method):
        """Apply the stochastic_seed policy to one SBML stochastic action."""
        if method != 'ssa':
            return None
        seed_value, overridden, policy = resolve_action_seed(
            self, explicit_seed=explicit_seed, action_index=action_index,
            suffix=suffix, method=method)
        if overridden:
            logger.debug(
                "BngsimSbmlModel %s action #%d (suffix=%r): overrode explicit "
                "seed=%s under stochastic_seed=%s",
                self.name, action_index, suffix, explicit_seed, policy,
            )
        return seed_value

    def _preequilibration_overrides(self, act, perturbations, phase):
        """One pre-equilibration phase's ``[(name, value)]`` perturbations (ADR-0052, #547).

        The config layer hands each phase's condition over as ``(kind, name, value)``
        triples, where ``kind`` distinguishes a BNGL ``setParameter`` from a
        ``setConcentration``. Here the two are one operation -- ``_set_engine_value_if_present``
        dispatches on whether the name is an SBML species or a global parameter -- so only
        the name and a numeric value survive.

        Both are validated rather than quietly dropped. A target this model does not
        declare would otherwise apply to nothing and the phase would simulate an
        unperturbed model, which is precisely the silent wrong answer #547 was; and a
        param-expression value (the BNGL titrated-competitor idiom, #474) has no SBML
        reading here.
        """
        overrides = []
        for _kind, name, value in perturbations:
            if name not in self.param_names:
                raise PybnfError(
                    f"Experiment '{act.suffix}': the {phase} condition sets '{name}', "
                    f"which model '{self.name}' does not declare as a species or a global "
                    "parameter. A pre-equilibration condition is applied to this model's "
                    "own entities; check the target's spelling (ADR-0052).")
            try:
                overrides.append((name, float(value)))
            except (TypeError, ValueError):
                raise PybnfError(
                    f"Experiment '{act.suffix}': the {phase} condition sets '{name}' to the "
                    f"expression '{value}'. On sbml_backend = bngsim a pre-equilibration "
                    "condition's value must be a number; use a numeric value, or a BNGL model "
                    "for an expression-valued intervention (ADR-0062).")
        return overrides

    def _begin_preequilibration(self, act, mut, *, method, seed, timeout, suffix):
        """Run the unmeasured equilibration phase and apply the intervention (ADR-0052, #547).

        The SBML/Antimony peer of ``BNGLModel._append_preequilibration_actions``: build the
        model under the ``preequilibrate:`` condition, relax it (to steady state, or for a
        fixed ``equil_t_end:``) without measuring, then apply the measurement ``condition:``
        in place. Returns ``(engine_model, sim, carry_sensitivities, equil_result)`` -- and
        ``sim`` is the *same* persistent :class:`bngsim.Simulator` the measured phase must
        run on, since running it again is what carries the equilibrated species state over
        (there is deliberately no reset between the phases). ``equil_result`` is the
        unmeasured phase's own Result: never scored, but it carries the equilibrium's
        ``dx_ss/dtheta``, which is the whole derivative for a measured phase with nothing
        to integrate.

        ``carry_sensitivities`` tells the measured phase to seed its forward sensitivities
        from the equilibration's own ``dx_ss/dtheta`` rather than from zero -- bngsim's
        implicit-function-theorem seeding (GH #210), which it *requires* on a carried-over
        state and refuses without. Mirrors ``net_model._prepare_simulate_run``'s condition
        exactly: the parameter axis is what gates the flag, because an initial-condition
        column cannot survive a stable steady state.
        """
        equil = self._preequilibration_overrides(
            act, act.equil_perturbations, 'pre-equilibration')
        measure = self._preequilibration_overrides(
            act, act.measure_perturbations, 'measurement')
        req = self._sensitivity_request
        carry = bool(req is not None and req.params and method == 'ode')
        if carry and any(name in self._species_name_set for name, _ in measure):
            # A mid-protocol species write retires the carried sensitivity matrix, so the
            # measured phase would need the intervention's own seed row rebuilt on top of it
            # (ADR-0098/0101 -- machinery the net backend has and this one does not). Refuse
            # rather than return a derivative that is wrong across the intervention.
            raise PybnfError(
                f"Experiment '{act.suffix}': its measurement condition writes a species "
                f"amount partway through a pre-equilibration protocol, and sbml_backend = "
                "bngsim cannot carry forward sensitivities across that write, so a "
                "gradient-based fit has no gradient here.",
                hint=["Refit with a gradient-free job_type (e.g. job_type = de), which "
                      "needs no sensitivities.",
                      "Or express the intervention as a parameter the species' initial "
                      "value reads, so it perturbs a parameter rather than the state."])

        engine_model = self._engine_model_for_action(mut=mut, phase_overrides=equil)
        sim = self._make_simulator(engine_model, method)
        if act.equil_fixed_time is not None:
            # A fixed equilibration duration (``equil_t_end:``) -- a timed incubation rather
            # than a relaxation to equilibrium.
            equil_result = self._run_simulation(
                engine_model, act.equil_fixed_time, 2, method=method, seed=seed,
                timeout=timeout, sim=sim, suffix=f'{suffix}_preequil')
        else:
            equil_result = self._run_simulation(
                engine_model, act.equil_max_time, 2, method=method, seed=seed,
                timeout=timeout, steady_state=True, sim=sim, suffix=f'{suffix}_preequil')
        for name, value in measure:
            self._set_engine_value_if_present(engine_model, name, value)
        return engine_model, sim, carry, equil_result

    def _run_preequilibrated_time_course(self, act, mut, *, method, seed, timeout, suffix):
        """The measured time course of a pre-equilibration experiment (ADR-0052, #547).

        Phase two of :meth:`_begin_preequilibration`: the data's own grid, integrated on the
        simulator the equilibration left advanced, so the equilibrated (then perturbed) state
        is this run's initial condition. The clock restarts at 0 -- the data times are
        relative to the intervention -- and only the concentrations carry.

        A ``t = 0``-only experiment (#510) has nothing to integrate: its single row *is* the
        post-intervention state, and its derivative is the equilibration's final
        ``dx_ss/dtheta``, so both are read straight off phase one rather than through the
        fresh-start re-preparation :meth:`_initial_state_data` uses.
        """
        engine_model, sim, carry, equil_result = self._begin_preequilibration(
            act, mut, method=method, seed=seed, timeout=timeout, suffix=suffix)
        if act.initial_state_only:
            return self._preequilibrated_initial_state_data(
                engine_model, equil_result, method=method)
        result = self._run_simulation(
            engine_model, act.time, act.stepnumber + 1, method=method, seed=seed,
            timeout=timeout, sample_times=act.explicit_points,
            steady_state=bool(act.steady_state), suffix=suffix,
            sim=sim, carry_sensitivities=carry)
        return self._result_to_data(result, stochastic=method == 'ssa')

    def _preequilibrated_initial_state_data(self, engine_model, equil_result, *, method):
        """The post-intervention state as a one-row ``t = 0`` trajectory (#510 + ADR-0052).

        The state is read off the engine model, which holds what the equilibration advanced
        it to. On the gradient path the derivative is the equilibration's final
        ``dx_ss/dtheta``: with nothing integrated after it, that row *is* the answer. The
        intervention that precedes this row perturbs only parameters -- a species write,
        which would supersede the row, is refused upstream -- so it leaves the row alone.
        """
        species_names = list(engine_model.species_names)
        state = np.asarray(
            [engine_model.get_concentration(name) for name in species_names], dtype=float)
        data = self._data_with_headers(
            np.concatenate(([0.0], state))[np.newaxis, :], ['time'] + species_names)
        req = self._sensitivity_request
        if req is None or method != 'ode' or not getattr(
                equil_result, 'has_sensitivities', False):
            return data
        sens = self._extract_output_sensitivities(equil_result)
        data.output_sensitivities = OutputSensitivities(
            selectors=sens.selectors, param_names=sens.param_names,
            ic_species=sens.ic_species,
            d_param=None if sens.d_param is None else sens.d_param[-1:],
            d_ic=None if sens.d_ic is None else sens.d_ic[-1:])
        return data

    def _run_preequilibrated_scan(self, act, mut, *, method, seed, timeout, suffix, points):
        """The measured dose-response scan of a pre-equilibration experiment (#474, ADR-0062).

        The preincubate -> wash -> dose-scan protocol: phase one equilibrates and the
        intervention is applied, that state is snapshotted, and every dose starts from the
        snapshot rather than from the model's seed initial conditions. ``save_concentrations``
        + ``reset`` per dose is the engine-level form of BNGL's ``saveConcentrations()`` +
        ``parameter_scan(reset_conc=>1)``, and the reset target being an *advanced* state is
        why each dose still runs with ``carry_sensitivities`` on the gradient path.
        """
        if (act.param in self._species_name_set and self._sensitivity_request is not None
                and self._sensitivity_request.params and method == 'ode'):
            # Same boundary as a species intervention, one phase later: writing the dose as a
            # species amount onto the carried state retires the sensitivity matrix each dose
            # would otherwise continue from, and this backend has no seed-row rebuild for it
            # (ADR-0104). A parameter-valued dose axis -- what a PEtab dose-response sweeps --
            # is unaffected.
            raise PybnfError(
                f"Experiment '{act.suffix}': its pre-equilibrated dose-response sweeps the "
                f"species amount '{act.param}', and sbml_backend = bngsim cannot carry forward "
                "sensitivities across a species write onto the carried state, so a "
                "gradient-based fit has no gradient here.",
                hint=["Refit with a gradient-free job_type (e.g. job_type = de), which needs "
                      "no sensitivities.",
                      "Or sweep a model parameter the species' initial value reads, so the "
                      "dose axis is a parameter rather than the state."])
        engine_model, sim, carry, _ = self._begin_preequilibration(
            act, mut, method=method, seed=seed, timeout=timeout, suffix=suffix)
        engine_model.save_concentrations()
        scan_label = act.param + '_0' if act.param in self._species_name_set else act.param
        rows = []
        headers = None
        sens_slices = []
        for x in points:
            engine_model.reset()
            self._set_engine_value_if_present(engine_model, act.param, x)
            result = self._run_simulation(
                engine_model, act.time, 2, method=method, seed=seed, timeout=timeout,
                sim=sim, carry_sensitivities=carry)
            row, point_headers = self._scan_point_to_row(result, x, scan_label)
            rows.append(row)
            if headers is None:
                headers = point_headers
            sens_slices.append(self._scan_point_sensitivities(result))
        data = self._data_with_headers(np.vstack(rows), headers)
        scan_sens = stack_scan_sensitivities(sens_slices)
        if scan_sens is not None:
            data.output_sensitivities = scan_sens
        return data

    def execute(self, folder, filename, timeout):
        from ._bngsim_caps import BNGSIM_VERSION as _BNGSIM_VERSION
        from ._bngsim_failure import write_failure_report

        backend_name = (
            'bngsim-antimony'
            if type(self).__name__.startswith('BngsimAntimony')
            else 'bngsim-sbml'
        )
        result_dict = {}

        for mut in self.mutants:
            for action_index, act in enumerate(self.actions):
                method = None
                seed_value = None
                suffix_with_mut = act.suffix + mut.suffix
                # Off-diagonal cross-product pruning (#484): under edition-2 one-model +
                # condition: perturbations this single model runs every action under every
                # condition mutant, but only the scored (action, its own condition) diagonal
                # is consumed. Skip any (action, condition) pair not in the emit-set. The
                # wildtype MutationSet has suffix '', so the base run is pruned by the same
                # guard. A no-op when emit_suffixes is unset (legacy/non-edition-2).
                if self.emit_suffixes is not None and suffix_with_mut not in self.emit_suffixes:
                    continue
                # Gate this action's sensitivity request on whether its output is
                # scored (#475/#482): set before any Simulator is built below (each
                # _run_simulation constructs a fresh Simulator through
                # _sensitivity_request_kwargs, which consults _action_bears_sensitivities).
                self._current_action_suffix = suffix_with_mut
                try:
                    method = self._resolve_method(act)
                    seed_value = self._resolve_action_seed(
                        explicit_seed=None,
                        action_index=action_index,
                        suffix=suffix_with_mut,
                        method=method,
                    )
                    if isinstance(act, TimeCourse):
                        if getattr(act, 'preequilibrate', False):
                            # Two phases on one persistent simulator (ADR-0052, #547):
                            # equilibrate under the `preequilibrate:` condition unmeasured,
                            # perturb to the measurement `condition:`, then measure with the
                            # equilibrated state carried over.
                            data = self._run_preequilibrated_time_course(
                                act, mut, method=method, seed=seed_value,
                                timeout=timeout, suffix=suffix_with_mut)
                        else:
                            engine_model = self._engine_model_for_action(mut=mut)
                            if act.initial_state_only:
                                data = self._initial_state_data(
                                    engine_model, method=method, mut=mut)
                            else:
                                result = self._run_simulation(
                                    engine_model, act.time, act.stepnumber + 1,
                                    method=method, seed=seed_value, timeout=timeout,
                                    sample_times=act.explicit_points,
                                    steady_state=bool(act.steady_state),
                                    suffix=suffix_with_mut,
                                )
                                data = self._result_to_data(result, stochastic=method == 'ssa')
                        result_dict[suffix_with_mut] = data
                        if self.save_files:
                            self._write_saved_output(
                                f'{folder}/{filename}_{act.suffix}{mut.suffix}.gdat',
                                data,
                            )
                    elif isinstance(act, ParamScan):
                        if act.param not in self.param_names:
                            raise PybnfError(
                                f'Parameter_scan parameter {act.param} was not found in model {self.name}'
                            )

                        # New-era explicit scan values (ADR-0028, #469/#470): sweep exactly
                        # the data's swept-parameter values instead of a uniform linspace
                        # grid, mirroring the native BNGL path (net_model._scan_independent).
                        # Without this a dose at a non-grid value is never simulated and
                        # scoring fails, exactly as for the time-course grid.
                        if act.explicit_points is not None:
                            points = act.explicit_points
                        else:
                            points = np.linspace(act.min, act.max, act.stepnumber + 1)
                        if getattr(act, 'preequilibrate', False):
                            # The preincubate -> wash -> dose-scan protocol (#474, ADR-0062):
                            # every dose starts from the carried post-intervention state, not
                            # from the model's seed initial conditions.
                            data = self._run_preequilibrated_scan(
                                act, mut, method=method, seed=seed_value, timeout=timeout,
                                suffix=suffix_with_mut, points=points)
                            result_dict[suffix_with_mut] = data
                            if self.save_files:
                                self._write_saved_output(
                                    f'{folder}/{filename}_{act.suffix}{mut.suffix}.scan',
                                    data,
                                )
                            continue
                        scan_label = act.param + '_0' if act.param in self._species_name_set else act.param
                        rows = []
                        headers = None
                        # Gradient path (#476): each independent, reset-to-seed dose
                        # point is a sensitivity-configured ODE run, so collect
                        # ∂species/∂θ at its final row for stacking down the dose axis
                        # (None per point on the scalar path -> no tensor attached).
                        sens_slices = []

                        for x in points:
                            engine_model = self._engine_model_for_action(
                                mut=mut, scan_override=(act.param, x))
                            result = self._run_simulation(
                                engine_model, act.time, 2,
                                method=method, seed=seed_value, timeout=timeout,
                            )
                            row, point_headers = self._scan_point_to_row(result, x, scan_label)
                            rows.append(row)
                            if headers is None:
                                headers = point_headers
                            sens_slices.append(self._scan_point_sensitivities(result))

                        data = self._data_with_headers(np.vstack(rows), headers)
                        scan_sens = stack_scan_sensitivities(sens_slices)
                        if scan_sens is not None:
                            data.output_sensitivities = scan_sens
                        result_dict[suffix_with_mut] = data
                        if self.save_files:
                            self._write_saved_output(
                                f'{folder}/{filename}_{act.suffix}{mut.suffix}.scan',
                                data,
                            )
                    else:
                        raise NotImplementedError('Unknown action type')
                except PybnfError:
                    raise
                except Exception as exc:
                    write_failure_report(
                        folder, filename,
                        backend=backend_name,
                        bngsim_version=_BNGSIM_VERSION,
                        model=self,
                        exception=exc,
                        input_path=getattr(self, 'abs_file_path', None),
                        action_info={
                            'action_index': action_index,
                            'method': method,
                            'suffix': suffix_with_mut,
                            'seed': seed_value,
                            'action_type': type(act).__name__,
                            'mutation_suffix': mut.suffix,
                        },
                    )
                    if isinstance(exc, bngsim.SimulationTimeout):
                        logger.warning(
                            'bngsim SBML model %s: wall_time_sim=%s exceeded at %.3fs',
                            self.name,
                            getattr(exc, 'timeout', timeout),
                            float(getattr(exc, 'elapsed', 0.0) or 0.0),
                        )
                    else:
                        logger.exception('bngsim SBML simulation failed for model %s', self.name)
                    if (self._sensitivity_request is not None
                            and _is_event_sensitivity_refusal(exc)):
                        # bngsim declining to differentiate this model's events is a
                        # *permanent, structural* refusal -- it will refuse identically
                        # at every parameter set -- so it must not be scored as one more
                        # non-integrable trial point the optimizer backs off from (#492)
                        # and eventually gives up on as "all jobs are failing". Reachable
                        # since #536 stopped refusing every event-bearing model up front:
                        # bngsim differentiates the subclasses it can classify and
                        # refuses the rest (an execution delay; a trigger that is not a
                        # single relational comparison), per simulation. Narrow on
                        # purpose -- every other backend failure keeps the
                        # FailedSimulationError back-off, which is right for a candidate
                        # point the integrator cannot get through.
                        raise PybnfError(
                            "Model %s: bngsim cannot supply forward output sensitivities "
                            "for this model's discrete events, so gradient-based fitting "
                            "has no gradient here: %s" % (self.name, exc),
                            hint=["Re-encode the event in a shape bngsim differentiates: "
                                  "a fixed trigger time, a trigger thresholding a fitted "
                                  "constant, or a single relational comparison, all "
                                  "without an execution delay.",
                                  "Or refit with a gradient-free job_type "
                                  "(e.g. job_type = de), which needs no sensitivities."],
                        ) from exc
                    raise FailedSimulationError from exc

        return result_dict


# Retained as an alias for backwards compatibility with code that previously
# imported the subprocess-wrapper subclass. bngsim now enforces the wall-clock
# budget in-process, so the wrapper is unnecessary.
BngsimSbmlModel = BngsimSbmlModelNoTimeout
