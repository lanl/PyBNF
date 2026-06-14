"""Optional SBML simulation using bngsim."""


import copy
import hashlib
import logging
import os
import secrets
import tempfile

import numpy as np

from .data import Data
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
    BNGSIM_HAS_SBML,
    BNGSIM_SBML_ERROR,
    bngsim,
)

try:
    import libsbml
except ImportError:
    libsbml = None


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


class BngsimSbmlModelNoTimeout(Model):
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
        """Names whose change requires recomputing a species' initial value.

        Returns the set of parameter/species names that any species' initial
        concentration depends on (transitively through initialAssignments and
        assignmentRules), or ``None`` if the model contains an algebraicRule
        (whose effect on initials we do not analyze, so we conservatively force
        a reload). When this set is empty, fitted parameter values can never
        change a species initial, so the fast cached-clone path is exact. See
        issue #415.
        """
        # symbol -> referenced names, for every expression that *defines* a
        # symbol's value (initialAssignments and assignmentRules). Used both to
        # seed (species-targeted definitions) and to expand transitively.
        expr_refs = {}
        for i in range(sbml_model.getNumInitialAssignments()):
            ia = sbml_model.getInitialAssignment(i)
            refs = set()
            self._collect_ast_names(ia.getMath(), refs)
            expr_refs.setdefault(ia.getSymbol(), set()).update(refs)
        for i in range(sbml_model.getNumRules()):
            rule = sbml_model.getRule(i)
            if rule.isAlgebraic():
                # An algebraic rule constrains initial values implicitly; we do
                # not solve it, so force the correctness-preserving reload path.
                return None
            if rule.isAssignment():
                refs = set()
                self._collect_ast_names(rule.getMath(), refs)
                expr_refs.setdefault(rule.getVariable(), set()).update(refs)
            # Rate rules define d/dt, not the initial value -> ignored here.

        # Seed from expressions that determine a *species'* initial value, then
        # expand through any referenced symbol that is itself expression-defined
        # (e.g. a species initial that depends on a parameter set by a rule).
        dep = set()
        worklist = []
        for symbol, refs in expr_refs.items():
            if symbol in self._species_name_set:
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

    def _needs_structural_reload(self, mut=None, scan_param=None):
        """Whether this evaluation must reload the model from SBML text.

        True only when a parameter/species being changed this evaluation can
        alter a species' initial value (so the engine model's baked-in initial
        concentrations would be stale under in-place ``set_param``). Otherwise
        the fast cached-clone path is exact. See issue #415.
        """
        if self._initial_dep_names is None:
            return True
        if not self._initial_dep_names:
            return False
        changed = set()
        if self.param_set is not None:
            changed.update(self.param_set.keys())
        if mut:
            changed.update(mi.name for mi in mut)
        if scan_param is not None:
            changed.add(scan_param)
        return bool(changed & self._initial_dep_names)

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
        """Apply a value to the engine model. Returns True iff it is a species."""
        if name in self._species_name_set:
            engine_model.set_concentration(name, float(value))
            return True
        if name in self._global_param_names:
            engine_model.set_param(name, float(value))
        return False

    def _get_engine_value_if_present(self, engine_model, name):
        if name in self._species_name_set:
            return float(engine_model.get_concentration(name))
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

    def _prepare_engine_model(self, mut=None, scan_override=None):
        """Clone the cached engine template and apply per-evaluation values.

        The fast-path analogue of ``_build_sbml_doc`` + reload: param_set,
        mutant deltas, and any scan override are applied in place via
        set_param/set_concentration on a cheap clone of the cached model,
        skipping the libSBML reparse + Jacobian re-derivation. Used only when
        the changed names cannot affect a species initial value (see
        ``_needs_structural_reload``). See issue #415.
        """
        engine_model = self._get_engine_template().clone()
        touched_species = self._apply_param_set_engine(engine_model)
        if mut:
            touched_species |= self._apply_mutant_engine(mut, engine_model)
        if scan_override is not None:
            scan_name, scan_value = scan_override
            if self._set_engine_value_if_present(engine_model, scan_name, scan_value):
                touched_species = True
        if touched_species:
            engine_model.save_concentrations()
        engine_model.reset()
        return engine_model

    def _engine_model_for_action(self, mut=None, scan_override=None):
        """Build the engine model for one action: fast cached-clone path when
        safe, else the correctness-preserving reload from SBML text (#415)."""
        scan_param = scan_override[0] if scan_override is not None else None
        if self._needs_structural_reload(mut=mut, scan_param=scan_param):
            doc = self._build_sbml_doc(mut=mut, scan_override=scan_override)
            return self._load_bngsim_model_from_text(_sbml_doc_to_text(doc))
        return self._prepare_engine_model(mut=mut, scan_override=scan_override)

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

    @classmethod
    def _result_to_data(cls, result, *, stochastic=False):
        if stochastic:
            arr = result.as_roadrunner()
            return Data(named_arr=arr)
        species = np.asarray(result.species, dtype=float)
        arr = np.zeros((result.n_times, 1 + species.shape[1]))
        arr[:, 0] = result.time
        arr[:, 1:] = species
        headers = ['time'] + list(result.species_names)
        return cls._data_with_headers(arr, headers)

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
        try:
            return bngsim.Simulator(engine_model, **kwargs)
        except bngsim.SsaValidationError as exc:
            raise ModelError(str(exc)) from exc

    def _run_simulation(self, engine_model, end_time, n_points, *, method='ode',
                        seed=None, timeout=None):
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
                        points = np.linspace(act.min, act.max, act.stepnumber + 1)
                        rows = []
                        headers = None

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

                        data = self._data_with_headers(np.vstack(rows), headers)
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
