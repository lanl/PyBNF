"""The bngsim network-free model: BngsimNfModel (NFsim / RuleMonkey sessions).

Executes BNGL network-free action lines in-process via bngsim's XML-backed
session API, carrying a live session across actions and re-deriving parameters
from the PSet. Reads the bngsim package + capability flags through the _runtime
seam (ADR-0018); delegates result conversion to BngsimModel.
"""


import copy
import logging
import os
import re
import shutil
from types import SimpleNamespace

import numpy as np

from . import _runtime
from ..data import Data
from ..pset import Model, _stage_and_rewrite_tfun_files
from .._seed import resolve_action_seed
from .net_model import BngsimModel
from .parsing import (
    _collapse_action_line_continuations,
    _parse_simulate_action,
    _parse_parameter_scan_action,
    _parse_set_parameter,
    _parse_set_concentration_nf,
    _parse_add_concentration,
    _is_reset_parameters,
    _is_save_parameters,
)
from .expressions import (
    _build_mutant_param_set,
    _evaluate_bngl_params,
    _parse_bngl_param_block,
)
from .scan import _resolve_scan_points
from .output import _write_saved_action_outputs
from .classification import (
    _normalize_nf_action_method,
    _nf_session_backend_for_method,
    _normalize_session_timeout,
    _create_nf_session,
    _destroy_nf_session,
    _first_nf_action_method,
    missing_bngsim_nf_action_support,
)


logger = logging.getLogger(__name__)


class BngsimNfModel(Model):
    """In-process network-free simulation using bngsim's XML-backed session API."""

    def __init__(
        self,
        name,
        acts,
        suffs,
        mutants,
        xml_path,
        bngl_model_lines=None,
        split_line_index=None,
        param_names=(),
        source_dir=None,
        protocol=(),
        save_files=False,
    ):
        if not _runtime.BNGSIM_AVAILABLE:
            raise RuntimeError('bngsim is not available')

        self.name = name
        self.actions = acts
        self.suffixes = suffs
        self.mutants = mutants
        self._xml_path = xml_path
        self._bngl_model_lines = list(bngl_model_lines) if bngl_model_lines is not None else None
        self._split_line_index = split_line_index
        self._source_dir = source_dir
        self._protocol = list(protocol)
        self.save_files = save_files
        self.param_names = tuple(param_names)
        self.param_set = None
        self.bng_command = ''
        self._default_nf_method = _first_nf_action_method(acts)
        missing_nf_support = missing_bngsim_nf_action_support(acts)
        if missing_nf_support:
            raise RuntimeError(
                'bngsim does not provide {} support'.format(', '.join(missing_nf_support))
            )

        if bngl_model_lines is not None:
            self._param_exprs = _parse_bngl_param_block(bngl_model_lines)
        else:
            self._param_exprs = []

    def copy_with_param_set(self, pset):
        """Return a shallow copy with the requested parameter set."""
        newmodel = copy.copy(self)
        newmodel.param_set = pset
        return newmodel

    def _resolve_action_seed(self, *, explicit_seed, action_index, suffix, method):
        """Apply the stochastic_seed policy to one NF/RM action.

        Always returns an int. Under `random*` policies with no honored
        explicit seed, materializes one fresh `secrets.randbits(31)` so that
        callers using arithmetic like `(seed + i)` for per-scan-point
        variation keep working.
        """
        seed_value, overridden, policy = resolve_action_seed(
            self, explicit_seed=explicit_seed, action_index=action_index,
            suffix=suffix, method=method)
        if overridden:
            logger.debug(
                "BngsimNfModel %s action #%d (suffix=%r): overrode explicit BNGL "
                "seed=%s under stochastic_seed=%s",
                self.name, action_index, suffix, explicit_seed, policy,
            )
        if seed_value is None:
            import secrets
            seed_value = secrets.randbits(31) or 1
        return seed_value

    def _initial_param_inputs(self):
        """Return the current PSet as direct parameter inputs for BNGL re-evaluation."""
        if self.param_set is None:
            return {}
        return {
            pname: float(self.param_set[pname])
            for pname in self.param_set.keys()
        }

    def _build_nf_param_overrides(self, input_overrides=None):
        """Compute parameter overrides to apply before network-free initialization."""
        if input_overrides is None:
            input_overrides = self._initial_param_inputs()

        if self._param_exprs:
            return _evaluate_bngl_params(self._param_exprs, input_overrides)

        return {
            pname: float(input_overrides[pname])
            for pname in input_overrides
        }

    @staticmethod
    def _apply_param_overrides(nfsim, param_overrides):
        """Apply all known parameter overrides to one network-free session."""
        try:
            if hasattr(nfsim, 'clear_param_overrides'):
                nfsim.clear_param_overrides()
        except Exception:
            logger.debug('BngsimNfModel: could not clear previous NF parameter overrides')

        for pname, pval in param_overrides.items():
            try:
                nfsim.set_param(pname, pval)
            except Exception:
                logger.debug(
                    'BngsimNfModel: could not set NF parameter %s=%s',
                    pname,
                    pval,
                )

    def _run_nf_parameter_scan(self, ps_params, current_param_inputs, action_index=0,
                               timeout=None):
        """Execute one NF parameter_scan() action using one short session per point."""
        method = _normalize_nf_action_method(ps_params.get('method', 'nf'))
        session_backend = _nf_session_backend_for_method(method)
        scan_timeout = _normalize_session_timeout(timeout, session_backend)

        # sample_times is not supported for network-free bngsim sessions
        # (possible future BNGsim enhancement).
        if ps_params.get('sample_times') is not None:
            logger.warning(
                "sample_times is not supported for bngsim network-free parameter_scan; ignoring")

        param_name = ps_params.get('parameter', '')
        t_start = float(ps_params.get('t_start', 0))
        t_end = float(ps_params.get('t_end', 100))
        n_steps = int(ps_params.get('n_steps', 1))
        suffix = ps_params.get('suffix', 'param_scan')
        print_funcs = bool(int(float(ps_params.get('print_functions', 0))))
        gml = ps_params.get('gml')
        gml_int = int(gml) if gml is not None else None

        explicit_seed = int(float(ps_params['seed'])) if 'seed' in ps_params else None
        scan_base_seed = self._resolve_action_seed(
            explicit_seed=explicit_seed,
            action_index=action_index,
            suffix=suffix,
            method=method,
        )

        points = _resolve_scan_points(ps_params)
        obs_names = []
        expr_names = []
        rows = []

        for i, value in enumerate(points):
            point_inputs = dict(current_param_inputs)
            if param_name:
                point_inputs[param_name] = float(value)
            point_param_overrides = self._build_nf_param_overrides(point_inputs)
            point_seed = (scan_base_seed + i) % (2**31)

            nfsim = _create_nf_session(session_backend, self._xml_path, molecule_limit=gml_int)
            try:
                self._apply_param_overrides(nfsim, point_param_overrides)
                nfsim.initialize(point_seed)
                sim_kwargs = {}
                if scan_timeout is not None:
                    sim_kwargs['timeout'] = scan_timeout
                result = nfsim.simulate(t_start, t_end, n_steps + 1, **sim_kwargs)
                row, row_obs, row_expr = BngsimModel._scan_result_to_row(
                    result, value, print_functions=print_funcs)
                if len(obs_names) == 0:
                    obs_names = row_obs
                    expr_names = row_expr
                rows.append(row)
            finally:
                _destroy_nf_session(nfsim)

        if rows:
            arr = np.vstack(rows)
        else:
            arr = np.zeros((0, 1))

        headers = [param_name] + obs_names + expr_names
        return {suffix: Data.from_columns(arr, headers)}

    @staticmethod
    def _result_to_data(result, print_functions=False):
        """Convert a bngsim Result to a PyBNF Data object."""
        return BngsimModel._result_to_data(result, print_functions=print_functions)

    def execute(self, folder, filename, timeout, with_mutants=True):
        """Execute all NF actions in-process using XML-backed network-free sessions.

        Iterates the model's actions, dispatching each to the matching handler
        (parameter_scan / simulate / set_parameter / setConcentration /
        addConcentration / save_/reset_parameters). The live network-free session
        is carried across actions in ``sess`` and torn down in the ``finally``;
        the per-action handlers mutate ``sess`` in place so cleanup always sees
        the current session, even if a simulation raises mid-action.
        """
        ds = {}
        current_param_inputs = self._initial_param_inputs()
        current_param_overrides = self._build_nf_param_overrides(current_param_inputs)
        saved_param_inputs = dict(current_param_inputs)
        sess = SimpleNamespace(nfsim=None, method=None, gml=None)
        self._pybnf_current_action_info = None

        # Bootstrap seed for sessions that need to be lazily started by a
        # setConcentration / addConcentration action before any simulate runs.
        # action_index=-1 distinguishes this seed from real action seeds.
        bootstrap_seed = self._resolve_action_seed(
            explicit_seed=None,
            action_index=-1,
            suffix='_bootstrap',
            method='nf',
        )

        try:
            for action_index, action_line in enumerate(self.actions):
                line = _collapse_action_line_continuations(action_line).strip()
                if not line or line.startswith('#'):
                    continue

                self._pybnf_current_action_info = {
                    'action_index': action_index,
                    'action_line': line,
                }

                ps_params = _parse_parameter_scan_action(line)
                if ps_params is not None:
                    ds.update(self._run_nf_parameter_scan(
                        ps_params,
                        current_param_inputs,
                        action_index=action_index,
                        timeout=timeout,
                    ))
                    continue

                sim_params = _parse_simulate_action(line)
                if sim_params is not None:
                    self._nf_simulate_action(
                        sim_params, sess, current_param_overrides, ds, action_index, timeout)
                    continue

                sp = _parse_set_parameter(line)
                if sp is not None:
                    param_name, param_value = sp
                    current_param_inputs[param_name] = float(param_value)
                    current_param_overrides = self._build_nf_param_overrides(current_param_inputs)
                    if sess.nfsim is not None:
                        self._apply_param_overrides(sess.nfsim, current_param_overrides)
                    continue

                sc = _parse_set_concentration_nf(line)
                if sc is not None:
                    self._nf_set_concentration_action(
                        sc, sess, current_param_overrides, bootstrap_seed)
                    continue

                ac = _parse_add_concentration(line)
                if ac is not None:
                    self._nf_add_concentration_action(
                        ac, sess, current_param_overrides, bootstrap_seed)
                    continue

                if _is_save_parameters(line):
                    saved_param_inputs = dict(current_param_inputs)
                    continue

                if _is_reset_parameters(line):
                    current_param_inputs = dict(saved_param_inputs)
                    current_param_overrides = self._build_nf_param_overrides(current_param_inputs)
                    if sess.nfsim is not None:
                        self._apply_param_overrides(sess.nfsim, current_param_overrides)
                    continue

                if line and not re.match(r'\s*(begin|end)\s+actions', line):
                    logger.debug("BngsimNfModel: skipping unsupported action: %s", line)
        except Exception as exc:
            self._report_nf_failure(exc, folder, filename, timeout)
        finally:
            _destroy_nf_session(sess.nfsim)

        if self.save_files:
            _write_saved_action_outputs(folder, filename, self.suffixes, ds)

        if with_mutants:
            self._run_nf_mutants(folder, filename, timeout, ds)

        return ds

    def _start_nf_session(self, seed_value, gml_value, method, param_overrides):
        """Create, parameter-override, and initialize a network-free session."""
        session_backend = _nf_session_backend_for_method(method)
        sim = _create_nf_session(session_backend, self._xml_path, molecule_limit=gml_value)
        self._apply_param_overrides(sim, param_overrides)
        sim.initialize(seed_value)
        return sim

    def _nf_simulate_action(self, sim_params, sess, current_param_overrides, ds,
                            action_index, timeout):
        """Run one simulate() action, starting or switching the NF session as needed.

        Mutates ``sess`` (nfsim/method/gml) in place so the caller's ``finally``
        tears down the live session even if ``simulate()`` raises, and stores the
        result in ``ds`` under the action's suffix.
        """
        method = _normalize_nf_action_method(sim_params.get('method', 'nf'))

        # sample_times is not supported for network-free bngsim
        # sessions (possible future BNGsim enhancement).
        if sim_params.get('sample_times') is not None:
            logger.warning(
                "sample_times is not supported for bngsim network-free simulation; ignoring")

        t_start = float(sim_params.get('t_start', 0))
        t_end = float(sim_params.get('t_end', 100))
        n_steps = int(sim_params.get('n_steps', 100))
        print_funcs = bool(int(float(sim_params.get('print_functions', 0))))
        suffix = sim_params.get('suffix', 'time_course')
        gml = sim_params.get('gml')
        gml_int = int(gml) if gml is not None else None
        explicit_seed = int(float(sim_params['seed'])) if 'seed' in sim_params else None
        action_seed = self._resolve_action_seed(
            explicit_seed=explicit_seed,
            action_index=action_index,
            suffix=suffix,
            method=method,
        )

        if sess.nfsim is None:
            sess.nfsim = self._start_nf_session(action_seed, gml_int, method, current_param_overrides)
            sess.method = method
            sess.gml = gml_int
        elif method != sess.method:
            logger.warning(
                "BngsimNfModel: switching network-free backends from %s to %s; "
                "simulator state cannot be transferred",
                sess.method,
                method,
            )
            _destroy_nf_session(sess.nfsim)
            sess.nfsim = self._start_nf_session(action_seed, gml_int, method, current_param_overrides)
            sess.method = method
            sess.gml = gml_int
        elif gml_int is not None and gml_int != sess.gml:
            sess.nfsim.set_molecule_limit(gml_int)
            sess.gml = gml_int

        sim_kwargs = {}
        nf_timeout = _normalize_session_timeout(
            timeout, _nf_session_backend_for_method(method),
        )
        if nf_timeout is not None:
            sim_kwargs['timeout'] = nf_timeout
        self._pybnf_current_action_info.update({
            'method': method,
            'suffix': suffix,
            'seed': action_seed,
            't_start': t_start,
            't_end': t_end,
            'n_steps': n_steps,
            'gml': gml_int,
        })
        result = sess.nfsim.simulate(t_start, t_end, n_steps + 1, **sim_kwargs)
        ds[suffix] = self._result_to_data(result, print_functions=print_funcs)

    def _nf_set_concentration_action(self, sc, sess, current_param_overrides, bootstrap_seed):
        """Apply one setConcentration() action, lazily starting a session if needed.

        Resolves the target count (literal, current parameter override, or a
        session parameter), then adds molecules to reach it. Decreases are not
        supported by the bridge and are warned-and-skipped.
        """
        species_pattern, expr_text = sc
        if sess.nfsim is None:
            sess.method = sess.method or self._default_nf_method
            sess.nfsim = self._start_nf_session(bootstrap_seed, sess.gml, sess.method,
                                                current_param_overrides)

        if expr_text in current_param_overrides:
            count = int(round(current_param_overrides[expr_text]))
        else:
            try:
                count = int(round(float(expr_text)))
            except ValueError:
                try:
                    count = int(round(sess.nfsim.get_parameter(expr_text)))
                except Exception:
                    logger.warning(
                        "BngsimNfModel: cannot evaluate '%s' for setConcentration",
                        expr_text,
                    )
                    return

        mol_type = species_pattern.split('(')[0]
        current = sess.nfsim.get_molecule_count(mol_type)
        to_add = count - current
        if to_add < 0:
            logger.warning(
                "BngsimNfModel: cannot decrease %s from %d to %d with the current bridge; leaving state unchanged",
                mol_type,
                current,
                count,
            )
            return
        if to_add > 0:
            sess.nfsim.add_molecules(mol_type, to_add)

    def _nf_add_concentration_action(self, ac, sess, current_param_overrides, bootstrap_seed):
        """Apply one addConcentration() action, lazily starting a session if needed."""
        species_pattern, delta = ac
        if sess.nfsim is None:
            sess.method = sess.method or self._default_nf_method
            sess.nfsim = self._start_nf_session(bootstrap_seed, sess.gml, sess.method,
                                                current_param_overrides)
        mol_type = species_pattern.split('(')[0]
        to_add = int(round(delta))
        if to_add > 0:
            sess.nfsim.add_molecules(mol_type, to_add)

    def _report_nf_failure(self, exc, folder, filename, timeout):
        """Write a failure report for an exception raised during NF execution, then
        re-raise. A bngsim ``SimulationTimeout`` is surfaced as
        ``FailedSimulationError``; any other exception propagates unchanged. This
        method never returns normally.
        """
        from ..pset import FailedSimulationError
        from .._bngsim_failure import write_failure_report
        write_failure_report(
            folder, filename,
            backend='bngsim-nf',
            bngsim_version=_runtime.BNGSIM_VERSION,
            model=self,
            exception=exc,
            input_path=getattr(self, '_xml_path', None),
            action_info=getattr(self, '_pybnf_current_action_info', None),
        )
        if isinstance(exc, _runtime.bngsim.SimulationTimeout):
            logger.warning(
                "BngsimNfModel %s: wall_time_sim=%s exceeded at %.3fs",
                self.name,
                getattr(exc, 'timeout', timeout),
                float(getattr(exc, 'elapsed', 0.0) or 0.0),
            )
            raise FailedSimulationError(str(exc)) from exc
        raise exc

    def _run_nf_mutants(self, folder, filename, timeout, ds):
        """Execute each mutant model and merge its outputs into ``ds`` (suffixed)."""
        for mut in self.mutants:
            logger.debug('Working on mutant %s', mut.suffix)
            mut_model = self._get_mutant_model_nf(mut)
            mut_data = mut_model.execute(
                folder,
                filename + mut.suffix,
                timeout,
                with_mutants=False,
            )
            for suff in mut_data:
                ds[suff + mut.suffix] = mut_data[suff]
            logger.debug('Finished mutant %s', mut.suffix)

    def _get_mutant_model_nf(self, mut):
        """Create a mutant copy with a mutated parameter set."""
        mut_model = copy.copy(self)
        mut_model.param_set = _build_mutant_param_set(self.param_set, mut)
        return mut_model

    def _saved_bngl_text(self):
        """Build a runnable BNGL copy for export/debugging when source text is available."""
        if (
            self._bngl_model_lines is None or
            self._split_line_index is None or
            self.param_set is None
        ):
            return None

        param_text_lines = [
            f'{k} {str(self.param_set[k])}'
            for k in self.param_names
        ]
        action_lines = ['begin actions\n'] + list(self.actions) + ['end actions']
        protocol_lines = []
        if self._protocol:
            protocol_lines = ['begin protocol\n'] + list(self._protocol) + ['end protocol\n']
        all_lines = (
            self._bngl_model_lines[:self._split_line_index] +
            param_text_lines +
            self._bngl_model_lines[self._split_line_index:] +
            protocol_lines +
            action_lines
        )
        return '\n'.join(all_lines) + '\n'

    def save(self, file_prefix, **kwargs):
        """Save the generated XML plus a runnable BNGL copy when possible."""
        del kwargs
        if self._xml_path and os.path.isfile(self._xml_path):
            shutil.copyfile(self._xml_path, file_prefix + '.xml')

        text = self._saved_bngl_text()
        if text is None:
            return

        text = _stage_and_rewrite_tfun_files(
            text,
            self._source_dir,
            os.path.dirname(file_prefix),
        )
        with open(file_prefix + '.bngl', 'w') as f:
            f.write(text)

    def save_all(self, file_prefix):
        """Save the current model and all mutant exports."""
        self.save(file_prefix)
        for mut in self.mutants:
            self._get_mutant_model_nf(mut).save(file_prefix + mut.suffix)

    def __getstate__(self):
        """Support pickling for worker processes."""
        return self.__dict__.copy()

    def __setstate__(self, state):
        """Restore from pickle."""
        self.__dict__.update(state)

