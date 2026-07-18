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


# Process-level cache of loaded bngsim engine models, keyed by a hash of the
# model's base SBML text. The engine model -- including the analytically
# derived (SymPy) Jacobian -- depends only on the model *structure*, not on
# parameter values, so it is loaded once per worker process and cloned for
# each evaluation rather than re-derived on every objective evaluation. See
# issue #415. The base model is unpickled fresh in each dask worker, so an
# instance attribute would re-derive the Jacobian per evaluation; the cache
# lives at module scope (one per worker process) to amortize across the fit.
_ENGINE_TEMPLATE_CACHE = {}


from ._bngsim_caps import (
    BNGSIM_HAS_OUTPUT_SENS,
    BNGSIM_HAS_SBML,
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

    def __init__(self, file, abs_file, pset=None, actions=(), save_files=False, integrator='cvode',
                 strict_ssa=True):
        if integrator not in _SUPPORTED_INTEGRATORS:
            raise ModelError(
                'sbml_backend = bngsim supports sbml_integrator in {}; got {}'.format(', '.join(_SUPPORTED_INTEGRATORS), integrator)
            )

        _require_bngsim_sbml_support()

        self._init_common_attrs(file, abs_file, pset, actions, save_files, integrator, strict_ssa,
                                file_ext='.xml')
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
                           file_ext):
        """Set the model attributes shared by the SBML and Antimony backends.

        ``file_ext`` is stripped from the file name to form ``self.name`` ('.xml'
        for SBML, '.ant' for Antimony). ``self.stochastic`` is set by the caller
        because the two backends compute it differently.
        """
        self.file_path = file
        self.abs_file_path = abs_file
        self.param_set = pset
        self.name = file[file.rfind('/') + 1:].rsplit(file_ext, 1)[0]
        self.save_files = save_files
        self.actions = list(actions)
        self.integrator = integrator
        self.strict_ssa = bool(strict_ssa)
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
        self.global_param_names = self._global_param_names
        self.param_names = self._species_name_set.union(set(self._global_param_names))
        self._initial_dep_names = self._compute_initial_dependency_names(doc.getModel())
        self._species_unit_factor, self._unsafe_volume = self._compute_species_unit_factors(
            doc.getModel())

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
        """Analyze which species initials depend on which parameters/species.

        Sets ``self._initial_expr_species`` (the species whose initial value is
        fixed at load time by an ``initialAssignment``) and returns the set of
        parameter/species names that those initials depend on -- transitively,
        following ``assignmentRule``-defined intermediates. When a changed name
        is in this set, those initials must be recomputed for the evaluation
        (see :meth:`_recompute_species_initials`); an empty set means parameter
        values can never change a species initial, so the fast cached-clone
        path is exact. Returns ``None`` if the model has an ``algebraicRule``
        (whose effect on initials we do not analyze, so we conservatively force
        a full reload). See issue #415.

        Only ``initialAssignment``-on-species seed the dependency: a species
        governed by an ``assignmentRule`` is recomputed *dynamically* by bngsim
        under ``set_param`` (verified), so it needs no special handling.
        """
        self._initial_expr_species = set()
        # symbol -> referenced names, for every expression that *defines* a
        # symbol's value. assignmentRules are included only so the transitive
        # walk can resolve a parameter that an initialAssignment reads through.
        expr_refs = {}
        ia_species = {}  # species symbol -> initialAssignment refs (the seed)
        for i in range(sbml_model.getNumInitialAssignments()):
            ia = sbml_model.getInitialAssignment(i)
            refs = set()
            self._collect_ast_names(ia.getMath(), refs)
            symbol = ia.getSymbol()
            expr_refs.setdefault(symbol, set()).update(refs)
            if symbol in self._species_name_set:
                ia_species[symbol] = refs
        for i in range(sbml_model.getNumRules()):
            rule = sbml_model.getRule(i)
            if rule.isAlgebraic():
                # An algebraic rule constrains values implicitly; we do not
                # solve it, so force the correctness-preserving reload path.
                return None
            if rule.isAssignment():
                refs = set()
                self._collect_ast_names(rule.getMath(), refs)
                expr_refs.setdefault(rule.getVariable(), set()).update(refs)
            # Rate rules define d/dt, not the initial value -> ignored here.

        self._initial_expr_species = set(ia_species)

        # Seed from initialAssignment-on-species, then expand through any
        # referenced symbol that is itself expression-defined (e.g. a species
        # initial that reads a parameter set by an assignmentRule).
        dep = set()
        worklist = []
        for refs in ia_species.values():
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

    def _compute_species_unit_factors(self, sbml_model):
        """Per-species factor converting a PyBNF species value to a bngsim
        concentration, plus a flag for volumes we cannot safely convert.

        bngsim works in concentrations internally. A concentration-based species
        value is already a concentration (factor 1.0); an amount-based species
        value is an amount, so its concentration is amount / compartment_size
        (factor 1/size). The reload path bakes these units on load, so the
        in-place paths must apply the same conversion to stay numerically
        identical. When an amount-based species sits in a compartment whose size
        is not a load-time constant (non-constant, or set by an
        initialAssignment/rule), the factor would itself be parameter-dependent;
        we flag that and fall back to a full reload. See issue #415.
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
        unsafe = False
        for i in range(sbml_model.getNumSpecies()):
            sp = sbml_model.getSpecies(i)
            name = sp.getId()
            amount_based = sp.isSetInitialAmount() or (
                not sp.isSetInitialConcentration() and sp.getHasOnlySubstanceUnits()
            )
            if not amount_based:
                factors[name] = 1.0
                continue
            comp_id = sp.getCompartment()
            comp = sbml_model.getCompartment(comp_id)
            vol = comp.getSize() if (comp is not None and comp.isSetSize()) else 1.0
            comp_constant = comp.getConstant() if comp is not None else True
            volume_is_constant = (
                comp_constant and comp_id not in ruled
                and vol == vol and vol > 0  # finite and positive
            )
            if volume_is_constant:
                factors[name] = 1.0 / float(vol)
            else:
                factors[name] = 1.0  # unused: the flag forces a reload
                unsafe = True
        return factors, unsafe

    def _changed_names(self, mut=None, scan_param=None):
        """The parameter/species names this evaluation changes."""
        changed = set()
        if self.param_set is not None:
            changed.update(self.param_set.keys())
        if mut:
            changed.update(mi.name for mi in mut)
        if scan_param is not None:
            changed.add(scan_param)
        return changed

    def _changes_touch_initials(self, mut=None, scan_param=None):
        """Whether this evaluation changes a name that a species initial depends
        on, so the baked-in initial concentrations must be recomputed (#415)."""
        if not self._initial_dep_names:
            return False
        return bool(self._changed_names(mut=mut, scan_param=scan_param)
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
        unwrapped and converting any other failure into a ModelError."""
        try:
            self._load_bngsim_model_from_path(self.abs_file_path)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise ModelError(parse_error_message) from exc

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

    def _apply_mutant(self, mut, sbml_model):
        for mi in mut:
            current = self._get_model_value_if_present(sbml_model, mi.name)
            if current is None:
                continue
            self._set_model_value_if_present(
                sbml_model,
                mi.name,
                _mutate_scalar(current, mi.operation, mi.value),
            )

    def _build_sbml_doc(self, mut=None, scan_override=None):
        doc = _sbml_doc_from_text(self._base_sbml_text, self.file_path)
        sbml_model = doc.getModel()
        self._apply_param_set(sbml_model)
        if mut:
            self._apply_mutant(mut, sbml_model)
        if scan_override is not None:
            scan_name, scan_value = scan_override
            self._set_model_value_if_present(sbml_model, scan_name, scan_value)
        return doc

    def _get_engine_template(self):
        """Return the process-cached base engine model for this SBML text.

        Loads the bngsim model once per worker process (paying the libSBML
        parse + analytical-Jacobian derivation) and reuses it for every
        evaluation, cloning it for each per-evaluation parameter application.
        See issue #415.
        """
        key = getattr(self, '_engine_template_key', None)
        if key is None:
            key = hashlib.sha256(self._base_sbml_text.encode('utf-8')).hexdigest()
            self._engine_template_key = key
        template = _ENGINE_TEMPLATE_CACHE.get(key)
        if template is None:
            template = self._load_bngsim_model_from_text(self._base_sbml_text)
            _ENGINE_TEMPLATE_CACHE[key] = template
        return template

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
        for mi in mut:
            current = self._get_engine_value_if_present(engine_model, mi.name)
            if current is None:
                continue
            new_value = _mutate_scalar(current, mi.operation, mi.value)
            if self._set_engine_value_if_present(engine_model, mi.name, new_value):
                touched_species = True
        return touched_species

    def _prepare_engine_model(self, mut=None, scan_override=None, ic_overrides=None):
        """Clone the cached engine template and apply per-evaluation values.

        The fast-path analogue of ``_build_sbml_doc`` + reload: param_set,
        mutant deltas, and any scan override are applied in place via
        set_param/set_concentration on a cheap clone of the cached model,
        skipping the libSBML reparse + Jacobian re-derivation. ``ic_overrides``
        (species name -> initial concentration) sets the recomputed initial
        values of species whose initials are parameter-driven; it is applied
        last so those values win over a direct param_set/scan assignment, and
        is baked in via save_concentrations. See issue #415.
        """
        engine_model = self._get_engine_template().clone()
        touched_species = self._apply_param_set_engine(engine_model)
        if mut:
            touched_species |= self._apply_mutant_engine(mut, engine_model)
        if scan_override is not None:
            scan_name, scan_value = scan_override
            if self._set_engine_value_if_present(engine_model, scan_name, scan_value):
                touched_species = True
        if ic_overrides:
            for species_name, ic in ic_overrides.items():
                self._set_engine_value_if_present(engine_model, species_name, ic)
            touched_species = True
        if touched_species:
            engine_model.save_concentrations()
        engine_model.reset()
        return engine_model

    def _recompute_species_initials(self, mut=None, scan_override=None):
        """Recompute the parameter-driven species initials for this evaluation.

        Builds the SBML doc with this evaluation's parameter/species/scan
        changes applied, evaluates its initialAssignments in place via libSBML
        (which resolves assignmentRule intermediates), and returns a
        ``{species: initial_concentration}`` mapping for the species whose
        initials are fixed at load by an initialAssignment. This reproduces the
        initials a full reload would bake in, without re-deriving the Jacobian.
        See issue #415.
        """
        doc = self._build_sbml_doc(mut=mut, scan_override=scan_override)
        sbml_model = doc.getModel()
        libsbml.SBMLTransforms.expandInitialAssignments(sbml_model)
        overrides = {}
        for species_name in self._initial_expr_species:
            ic = self._get_model_value_if_present(sbml_model, species_name)
            if ic is not None:
                overrides[species_name] = ic
        return overrides

    def _engine_model_for_action(self, mut=None, scan_override=None):
        """Build the engine model for one action by reusing the cached engine
        template (see issue #415).

        Three paths, cheapest first: (1) clone + in-place set_param when no
        species initial is affected; (2) clone + recomputed initials when a
        parameter-driven initialAssignment is in play; (3) a full reload from
        SBML text only for models with an algebraicRule (which we do not
        analyze). All three avoid re-deriving the analytical Jacobian except
        the last.
        """
        scan_param = scan_override[0] if scan_override is not None else None
        if self._needs_structural_reload(mut=mut, scan_param=scan_param):
            doc = self._build_sbml_doc(mut=mut, scan_override=scan_override)
            return self._load_bngsim_model_from_text(_sbml_doc_to_text(doc))
        ic_overrides = None
        if self._changes_touch_initials(mut=mut, scan_param=scan_param):
            if not _HAS_EXPAND_INITIAL_ASSIGNMENTS:
                # No in-place initial evaluation available -> reload to stay correct.
                doc = self._build_sbml_doc(mut=mut, scan_override=scan_override)
                return self._load_bngsim_model_from_text(_sbml_doc_to_text(doc))
            ic_overrides = self._recompute_species_initials(
                mut=mut, scan_override=scan_override)
        return self._prepare_engine_model(
            mut=mut, scan_override=scan_override, ic_overrides=ic_overrides)

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
        arr[:, 1:] = species
        headers = ['time'] + list(result.species_names)
        data = self._data_with_headers(arr, headers)
        if self._sensitivity_request is not None and getattr(
                result, 'has_sensitivities', False):
            data.output_sensitivities = self._extract_output_sensitivities(result)
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
        d_param = None
        if param_names:
            d_param = np.asarray(
                result.output_sensitivities(selectors, axis='parameter'), dtype=float)
        d_ic = None
        if ic_species:
            d_ic = np.asarray(
                result.output_sensitivities(selectors, axis='ic'), dtype=float)
            # ∂/∂(concentration IC) -> ∂/∂(PyBNF IC value): chain by the unit factor per axis.
            ic_factors = np.array(
                [self._species_unit_factor[s] for s in ic_species], dtype=float)
            d_ic = d_ic * ic_factors[np.newaxis, np.newaxis, :]
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
                result, 'has_sensitivities', False):
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
        """True iff the engine model contains state-jumping discrete events (#461).

        SBML ``event``\\ s reinitialise the integrator state discontinuously, but
        bngsim's CVODES forward-sensitivity vectors are *not* reinitialised across
        the jump, so the sensitivity columns go silently stale at and after an event
        fires -- bngsim therefore refuses forward output sensitivities outright on
        such a model rather than return wrong derivatives (bngsim GH #205). The
        gradient path reads this as its pre-flight differentiability gate
        (:meth:`GradientOptimizer._require_differentiable_dynamics`) to refuse a
        discrete-event model **up front** -- with an actionable "use a metaheuristic
        fit_type" message -- instead of letting the fit start and fail at the first
        sensitivity-bearing ``simulate()``. The net backend's property documents the
        same contract; this is its SBML twin.

        Only true state-jumping events are counted (the engine core's ``n_events``).
        ``False`` when the engine model or its event count is unavailable (an
        older/stub backend), so the gate never blocks on a missing signal.
        """
        core = getattr(self._get_engine_template(), '_core', None)
        return bool(getattr(core, 'n_events', 0))

    def sensitivity_entity_namespace(self):
        """The bind-by-id namespaces the gradient router classifies free parameters against (#448).

        Returns ``(param_ids, species_initializers)``:

        * ``param_ids`` -- the model's global ``parameter`` ids, the kinetic ids a free
          parameter binds to via ``set_param`` and thus routes to
          ``Simulator(sensitivity_params=)``;
        * ``species_initializers`` -- ``(species, initial-expr)`` pairs in the shape
          :func:`pybnf.gradient.routing.classify_free_param` expects. A free parameter named
          for a species sets that species' initial value (via ``set_concentration``, the
          bind-by-id convention, ADR-0034), so each species' bare initializer expression *is*
          its own name; such a free parameter routes to the initial-condition axis keyed by the
          species (an IC parameter is absent from the ODE RHS, so its parameter axis is zero).

        This is the only model coupling :mod:`pybnf.gradient.routing` needs, so the routing core
        stays backend-agnostic. No simulation -- both namespaces are known at load time. (A
        species whose initial is a non-trivial *expression* of a parameter is the deferred
        non-bare-initializer case the router documents, exactly as for the net backend.)
        """
        return list(self._global_param_names), [(s, s) for s in self._species_names]

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

    def _run_simulation(self, engine_model, end_time, n_points, *, method='ode',
                        seed=None, timeout=None, sample_times=None):
        sim = self._make_simulator(engine_model, method)
        run_kwargs = {}
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
        return sim.run(t_span=(0.0, float(end_time)), n_points=int(n_points), **run_kwargs)

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
                        engine_model = self._engine_model_for_action(mut=mut)
                        result = self._run_simulation(
                            engine_model, act.time, act.stepnumber + 1,
                            method=method, seed=seed_value, timeout=timeout,
                            sample_times=act.explicit_points,
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

                        scan_label = act.param + '_0' if act.param in self._species_name_set else act.param
                        # New-era explicit scan values (ADR-0028, #469/#470): sweep exactly
                        # the data's swept-parameter values instead of a uniform linspace
                        # grid, mirroring the native BNGL path (net_model._scan_independent).
                        # Without this a dose at a non-grid value is never simulated and
                        # scoring fails, exactly as for the time-course grid.
                        if act.explicit_points is not None:
                            points = act.explicit_points
                        else:
                            points = np.linspace(act.min, act.max, act.stepnumber + 1)
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
                    raise FailedSimulationError from exc

        return result_dict


# Retained as an alias for backwards compatibility with code that previously
# imported the subprocess-wrapper subclass. bngsim now enforces the wall-clock
# budget in-process, so the wrapper is unnecessary.
BngsimSbmlModel = BngsimSbmlModelNoTimeout
