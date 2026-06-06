"""Optional SBML simulation using bngsim."""


import copy
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
        raise ModelError('Failed to parse SBML from %s' % source_desc)

    messages = []
    for i in range(doc.getNumErrors()):
        err = doc.getError(i)
        if err.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
            messages.append(err.getMessage())
    if messages:
        raise ModelError(
            'Failed to parse SBML from %s: %s' %
            (source_desc, '; '.join(messages[:3]))
        )

    if doc.getModel() is None:
        raise ModelError('SBML document from %s does not contain a model' % source_desc)

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
    raise RuntimeError('Invalid mutation operation %s' % operation)


class BngsimSbmlModelNoTimeout(Model):
    def __init__(self, file, abs_file, pset=None, actions=(), save_files=False, integrator='cvode',
                 strict_ssa=True):
        if integrator not in _SUPPORTED_INTEGRATORS:
            raise ModelError(
                'sbml_backend = bngsim supports sbml_integrator in %s; got %s' %
                (', '.join(_SUPPORTED_INTEGRATORS), integrator)
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
            'Failed to load model %s.xml - There were errors in parsing this SBML file. See log for details.'
            % self.name
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

    def model_text(self, mut=None):
        logger.info('Generating model text for %s', self.name)
        return _sbml_doc_to_text(self._build_sbml_doc(mut=mut))

    def save(self, file_prefix):
        with open('%s.xml' % file_prefix, 'w') as out:
            out.write(self.model_text())

    def save_all(self, file_prefix):
        for mut in self.mutants:
            with open('%s%s.xml' % (file_prefix, mut.suffix), 'w') as out:
                out.write(self.model_text(mut=mut))

    def add_action(self, action):
        if action.method not in ('ode', 'ssa'):
            raise PybnfError(
                'time_course or param_scan method %s is not currently supported with '
                'sbml_backend = bngsim. Options are ode or ssa.' % action.method
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
                        doc = self._build_sbml_doc(mut=mut)
                        engine_model = self._load_bngsim_model_from_text(_sbml_doc_to_text(doc))
                        result = self._run_simulation(
                            engine_model, act.time, act.stepnumber + 1,
                            method=method, seed=seed_value, timeout=timeout,
                        )
                        data = self._result_to_data(result, stochastic=method == 'ssa')
                        result_dict[suffix_with_mut] = data
                        if self.save_files:
                            self._write_saved_output(
                                '%s/%s_%s%s.gdat' % (folder, filename, act.suffix, mut.suffix),
                                data,
                            )
                    elif isinstance(act, ParamScan):
                        if act.param not in self.param_names:
                            raise PybnfError(
                                'Parameter_scan parameter %s was not found in model %s' %
                                (act.param, self.name)
                            )

                        scan_label = act.param + '_0' if act.param in self._species_name_set else act.param
                        points = np.linspace(act.min, act.max, act.stepnumber + 1)
                        rows = []
                        headers = None

                        for x in points:
                            doc = self._build_sbml_doc(mut=mut, scan_override=(act.param, x))
                            engine_model = self._load_bngsim_model_from_text(_sbml_doc_to_text(doc))
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
                                '%s/%s_%s%s.scan' % (folder, filename, act.suffix, mut.suffix),
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
