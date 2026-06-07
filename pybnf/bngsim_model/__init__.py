"""Optional in-process BNGL -> .net simulation using bngsim."""


import copy
import concurrent.futures
import logging
import os
import re
import shutil
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from . import _runtime
# Re-export the bngsim capability flags so ``pybnf.bngsim_model.<flag>`` keeps
# resolving for value-importers (``pybnf.algorithms.base``) and read-only test
# access. Production code in this package reads them as ``_runtime.<flag>`` so a
# patch on the _runtime seam (ADR-0018) bites every reader; these bindings are
# snapshots for re-export only, never read by this package's own logic.
from ._runtime import (
    bngsim as bngsim,
    BNGSIM_AVAILABLE as BNGSIM_AVAILABLE,
    BNGSIM_ERROR as BNGSIM_ERROR,
    BNGSIM_HAS_NFSIM as BNGSIM_HAS_NFSIM,
    BNGSIM_HAS_RULEMONKEY as BNGSIM_HAS_RULEMONKEY,
    BNGSIM_VERSION as BNGSIM_VERSION,
)
from ..data import Data
from ..pset import Model, NetModel, _stage_and_rewrite_tfun_files
from .._seed import resolve_action_seed
# Pure BNGL action-line parsing lives in parsing.py (CI-testable, bngsim-free).
# Re-exported here so the model classes / classification resolve the bare names
# and the package facade (and tests) keep resolving pybnf.bngsim_model.<name>.
from .parsing import (
    _collapse_action_line_continuations as _collapse_action_line_continuations,
    _extract_action_body as _extract_action_body,
    _split_top_level_commas as _split_top_level_commas,
    _parse_action_value as _parse_action_value,
    _parse_action_dict as _parse_action_dict,
    _parse_simulate_action as _parse_simulate_action,
    _parse_parameter_scan_action as _parse_parameter_scan_action,
    _parse_bifurcate_action as _parse_bifurcate_action,
    _parse_set_parameter as _parse_set_parameter,
    _parse_set_concentration as _parse_set_concentration,
    _parse_set_concentration_expr as _parse_set_concentration_expr,
    _parse_set_concentration_nf as _parse_set_concentration_nf,
    _parse_add_concentration as _parse_add_concentration,
    _is_reset_concentrations as _is_reset_concentrations,
    _is_reset_parameters as _is_reset_parameters,
    _is_save_concentrations as _is_save_concentrations,
    _is_save_parameters as _is_save_parameters,
)
# Pure expression / parameter evaluation lives in expressions.py (CI-testable,
# bngsim-free). Re-exported here so the model classes resolve the bare names and
# the facade (and tests) keep resolving pybnf.bngsim_model.<name>.
from .expressions import (
    _build_safe_eval_namespace as _build_safe_eval_namespace,
    _eval_numeric as _eval_numeric,
    _eval_model_expression as _eval_model_expression,
    _model_param_values as _model_param_values,
    _build_mutant_param_set as _build_mutant_param_set,
    _parse_bngl_param_block as _parse_bngl_param_block,
    _evaluate_bngl_params as _evaluate_bngl_params,
    _parse_net_species_initializers as _parse_net_species_initializers,
)
# Scan-point/sample-time resolution + steady-state constants live in scan.py;
# on-disk action-output writing in output.py (both numpy-only, bngsim-free).
# Re-exported so the model classes / classification resolve the bare names.
from .scan import (
    _with_sim_timeout as _with_sim_timeout,
    _resolve_scan_points as _resolve_scan_points,
    _resolve_sample_times as _resolve_sample_times,
    _SS_METHOD_PARITY as _SS_METHOD_PARITY,
    _SS_METHOD_NEWTON as _SS_METHOD_NEWTON,
    _SS_SCAN_N_POINTS as _SS_SCAN_N_POINTS,
)
from .output import (
    _ext_for_simtype as _ext_for_simtype,
    _write_saved_action_outputs as _write_saved_action_outputs,
)
# Backend classification, method/timeout normalization, and NF session
# management live in classification.py (reads the _runtime flag seam).
# Re-exported so the model classes resolve the bare names and the facade keeps
# resolving pybnf.bngsim_model.<name> for algorithms.base and the tests.
from .classification import (
    BNGSIM_BACKEND_NET as BNGSIM_BACKEND_NET,
    BNGSIM_BACKEND_NF as BNGSIM_BACKEND_NF,
    BNGSIM_BACKEND_HYBRID as BNGSIM_BACKEND_HYBRID,
    BNGSIM_NF_BACKEND_NFSIM as BNGSIM_NF_BACKEND_NFSIM,
    BNGSIM_NF_BACKEND_RULEMONKEY as BNGSIM_NF_BACKEND_RULEMONKEY,
    _BNGSIM_ACTION_BACKENDS as _BNGSIM_ACTION_BACKENDS,
    _BNGSIM_NF_CANONICAL_METHODS as _BNGSIM_NF_CANONICAL_METHODS,
    _normalize_ss_method as _normalize_ss_method,
    _coerce_positive_timeout as _coerce_positive_timeout,
    _normalize_sim_timeout as _normalize_sim_timeout,
    _normalize_session_timeout as _normalize_session_timeout,
    _normalize_action_method as _normalize_action_method,
    _bngsim_normalize_method as _bngsim_normalize_method,
    _normalize_nf_action_method as _normalize_nf_action_method,
    _nf_session_backend_for_method as _nf_session_backend_for_method,
    _bngsim_has_nf_session_backend as _bngsim_has_nf_session_backend,
    _nf_session_backend_label as _nf_session_backend_label,
    _nf_method_from_action_params as _nf_method_from_action_params,
    _required_nf_session_backends as _required_nf_session_backends,
    _first_nf_action_method as _first_nf_action_method,
    missing_bngsim_nf_action_support as missing_bngsim_nf_action_support,
    _get_nf_session_class as _get_nf_session_class,
    _create_nf_session as _create_nf_session,
    _destroy_nf_session as _destroy_nf_session,
    _classify_action_method_backend as _classify_action_method_backend,
    _allowed_bngsim_backends_for_action as _allowed_bngsim_backends_for_action,
    classify_actions_for_bngsim as classify_actions_for_bngsim,
    actions_compatible_with_bngsim as actions_compatible_with_bngsim,
)


logger = logging.getLogger(__name__)


@dataclass
class _SimulateActionState:
    """Mutable simulator state carried across the simulate() actions of one run.

    Both ``BngsimModel._execute_actions`` and ``._run_protocol`` walk a list of
    action lines, recreating the simulator when the method changes and tracking
    the running model time for ``continue=>1``. This bundles that shared state so
    the duplicated simulate() handling can live in one place
    (``_prepare_simulate_run``).
    """
    sim: object
    method: str = 'ode'
    poplevel: object = None
    current_time: float = 0.0


@dataclass
class _SimulateRunPlan:
    """A parsed-and-prepared simulate() action, ready to hand to a simulator run.

    ``_prepare_simulate_run`` produces this; ``_run_prepared_simulate`` consumes
    it. The split lets ``_execute_actions`` record per-action info between the
    two steps (the protocol path skips that), without duplicating the parsing.
    """
    sim: object
    method: str
    suffix: str
    print_funcs: bool
    sample_times: object
    t_start: float
    t_end: float
    n_steps: int
    run_kwargs: dict
    stop_if: object


@dataclass
class _ScanSettings:
    """Resolved settings for one parameter_scan()/bifurcate() action.

    Produced by ``BngsimModel._resolve_scan_settings`` and consumed by the
    per-strategy ``_scan_*`` helpers, so the simulation branches read named
    fields instead of a dozen loose locals.
    """
    param_name: str
    t_start: float
    t_end: float
    suffix: str
    use_ss: int
    ss_method: str
    print_funcs: bool
    method: str
    poplevel: object
    scan_seed: object
    sample_times: object
    reset_conc: bool
    points: object
    concentration_overrides: dict
    timeout: object
    scan_timeout: object
    scan_eval_timeout: object


def _try_prepare_codegen(net_path):
    """Attempt to compile ODE RHS to a shared library for faster simulation.

    Returns the path to the compiled ``.so`` or ``""`` if codegen is
    unavailable or compilation fails.
    """
    if os.environ.get('PYBNF_NO_CODEGEN') or os.environ.get('BNGSIM_NO_CODEGEN'):
        return ""
    try:
        from bngsim import prepare_codegen
        return str(prepare_codegen(net_path))
    except Exception as exc:
        logger.warning("Codegen compilation failed (%s); falling back to interpreted ODE RHS (slower)", exc)
        return ""


class BngsimModel(NetModel):
    """In-process simulation model using the optional bngsim engine."""

    def __init__(self, name, acts, suffs, mutants, ls=None, nf=None, source_dir=None, protocol=None,
                 save_files=False):
        super().__init__(
            name,
            acts,
            suffs,
            mutants,
            ls=ls,
            nf=nf,
            source_dir=source_dir,
        )
        if not _runtime.BNGSIM_AVAILABLE:
            raise RuntimeError('bngsim is not available')
        self._protocol = protocol or []
        self.save_files = save_files

        self._net_species_initializers = _parse_net_species_initializers(
            self.netfile_lines
        )
        if nf is not None:
            self._net_path = nf
            self._engine_model = _runtime.bngsim.Model.from_net(nf)
            self._codegen_so = _try_prepare_codegen(nf)
        elif ls is not None:
            raise ValueError('BngsimModel requires nf so the .net path is stable')
        else:
            raise ValueError('Must provide nf')

    def copy_with_param_set(self, pset):
        """Return a shallow copy with a cloned engine model and new PSet."""
        newmodel = copy.copy(self)
        newmodel._engine_model = self._engine_model.clone()
        newmodel._protocol = self._protocol
        newmodel.param_set = pset
        return newmodel

    def _resolve_action_seed(self, *, explicit_seed, action_index, suffix, method):
        """Apply the stochastic_seed policy to one stochastic action.

        Returns the seed integer to pass to bngsim, or None to delegate to
        bngsim's own randomization.
        """
        seed_value, overridden, policy = resolve_action_seed(
            self, explicit_seed=explicit_seed, action_index=action_index,
            suffix=suffix, method=method)
        if overridden:
            logger.debug(
                "BngsimModel %s action #%d (suffix=%r): overrode explicit BNGL "
                "seed=%s under stochastic_seed=%s",
                self.name, action_index, suffix, explicit_seed, policy,
            )
        return seed_value

    def execute(self, folder, filename, timeout, with_mutants=True):
        """Execute all simulation actions in-process using bngsim."""
        from ..pset import FailedSimulationError
        from .._bngsim_failure import write_failure_report

        model = self._engine_model

        if self.param_set is not None:
            for pname in self.param_set.keys():
                try:
                    model.set_param(pname, self.param_set[pname])
                except Exception:
                    # A free parameter that doesn't map to a model parameter is
                    # dropped here -- the optimizer believes it is varying the
                    # parameter while the model never sees it (a confidently
                    # wrong fit). This is legitimate only in multi-model fits,
                    # where the shared PSet spans parameters from other models,
                    # so surface it as a warning rather than swallowing it.
                    logger.warning(
                        "Model %s: could not set free parameter %s=%s "
                        "(not found in this model)",
                        self.name, pname, self.param_set[pname],
                    )

        model.reset()
        self._pybnf_current_action_info = None
        try:
            ds = self._execute_actions(model, timeout=timeout)
        except Exception as exc:
            write_failure_report(
                folder, filename,
                backend='bngsim-net',
                bngsim_version=_runtime.BNGSIM_VERSION,
                model=self,
                exception=exc,
                input_path=getattr(self, '_net_path', None),
                action_info=getattr(self, '_pybnf_current_action_info', None),
            )
            if isinstance(exc, _runtime.bngsim.SimulationTimeout):
                logger.warning(
                    "BngsimModel %s: wall_time_sim=%s exceeded at %.3fs",
                    self.name,
                    getattr(exc, 'timeout', timeout),
                    float(getattr(exc, 'elapsed', 0.0) or 0.0),
                )
                raise FailedSimulationError(str(exc)) from exc
            raise

        if self.save_files:
            _write_saved_action_outputs(folder, filename, self.suffixes, ds)

        if with_mutants:
            for mut in self.mutants:
                logger.debug('Working on mutant %s', mut.suffix)
                mut_model = self._get_mutant_model_bngsim(mut)
                mut_data = mut_model.execute(
                    folder,
                    filename + mut.suffix,
                    timeout,
                    with_mutants=False,
                )
                for suff in mut_data:
                    ds[suff + mut.suffix] = mut_data[suff]
                logger.debug('Finished mutant %s', mut.suffix)

        return ds

    def _execute_actions(self, model, timeout=None):
        """Interpret and execute action lines using bngsim."""
        ds = {}
        state = _SimulateActionState(
            sim=_runtime.bngsim.Simulator(model, method='ode', **self._codegen_kwargs()))

        base_params = {}
        for pname in model.param_names:
            try:
                base_params[pname] = model.get_param(pname)
            except Exception:
                pass
        # Active setConcentration() expressions waiting to be replayed at the
        # next parameter_scan(). Cleared on resetConcentrations and
        # saveConcentrations. See issue #46.
        concentration_overrides: dict[str, str] = {}

        for action_index, action_line in enumerate(self.actions):
            line = _collapse_action_line_continuations(action_line).strip()
            if not line or line.startswith('#'):
                continue

            self._pybnf_current_action_info = {
                'action_index': action_index,
                'action_line': line,
            }

            sim_params = _parse_simulate_action(line)
            if sim_params is not None:
                plan = self._prepare_simulate_run(
                    state, sim_params, model, action_index, timeout,
                    seed_suffix_prefix='', null_nf_sample_times=True,
                )
                self._pybnf_current_action_info.update({
                    'method': plan.method,
                    'suffix': plan.suffix,
                    'seed': plan.run_kwargs.get('seed'),
                    't_start': plan.t_start,
                    't_end': plan.t_end,
                    'n_steps': plan.n_steps,
                })
                result = self._run_prepared_simulate(plan, stop_log_label='stop_if triggered')
                state.current_time = plan.t_end
                ds[plan.suffix] = self._result_to_data(result, print_functions=plan.print_funcs)
                continue

            sp = _parse_set_parameter(line)
            if sp is not None:
                param_name, param_value = sp
                try:
                    model.set_param(param_name, param_value)
                except Exception:
                    logger.warning(
                        "setParameter(%s, %s) failed - param not found",
                        param_name,
                        param_value,
                    )
                continue

            if _is_reset_concentrations(line):
                model.reset()
                concentration_overrides.clear()
                continue

            if _is_reset_parameters(line):
                for pname, pval in base_params.items():
                    try:
                        model.set_param(pname, pval)
                    except Exception:
                        logger.debug(
                            "resetParameters: could not restore %s=%s", pname, pval)
                continue

            if _is_save_concentrations(line):
                model.save_concentrations()
                concentration_overrides.clear()
                continue

            if _is_save_parameters(line):
                for pname in model.param_names:
                    try:
                        base_params[pname] = model.get_param(pname)
                    except Exception:
                        logger.debug(
                            "saveParameters: could not read %s", pname)
                continue

            sc_expr = _parse_set_concentration_expr(line)
            if sc_expr is not None:
                species_name, conc_expr = sc_expr
                try:
                    conc_value = _eval_model_expression(conc_expr, model)
                    model.set_concentration(species_name, conc_value)
                    concentration_overrides[species_name] = conc_expr
                except Exception:
                    logger.warning(
                        "setConcentration(%s, %s) failed",
                        species_name,
                        conc_expr,
                    )
                continue

            ac = _parse_add_concentration(line)
            if ac is not None:
                species_name, delta = ac
                try:
                    current = model.get_concentration(species_name)
                    model.set_concentration(species_name, current + delta)
                except Exception:
                    logger.warning(
                        "addConcentration(%s, %s) failed - species not found",
                        species_name,
                        delta,
                    )
                continue

            ps_params = _parse_parameter_scan_action(line)
            if ps_params is not None:
                ds.update(self._run_parameter_scan(
                    model, ps_params,
                    action_index=action_index,
                    timeout=timeout,
                    concentration_overrides=concentration_overrides,
                ))
                continue

            bf_params = _parse_bifurcate_action(line)
            if bf_params is not None:
                ds.update(self._run_parameter_scan(
                    model, bf_params, is_bifurcate=True,
                    action_index=action_index,
                    timeout=timeout,
                    concentration_overrides=concentration_overrides,
                ))
                continue

            if line and not re.match(r'\s*(begin|end)\s+actions', line):
                logger.debug("BngsimModel: skipping unknown action: %s", line)

        return ds

    def _run_protocol(self, model, timeout=None):
        """Execute the stored protocol: a sequence of action lines.

        Returns the Result from the last simulate action, or None if the
        protocol contains no simulate actions.
        """
        state = _SimulateActionState(
            sim=_runtime.bngsim.Simulator(model, method='ode', **self._codegen_kwargs()))
        last_result = None

        # Baseline saved parameters (used by saveParameters/resetParameters)
        saved_params = {}
        for pname in model.param_names:
            try:
                saved_params[pname] = model.get_param(pname)
            except Exception:
                pass

        for action_index, action_line in enumerate(self._protocol):
            line = _collapse_action_line_continuations(action_line).strip()
            if not line or line.startswith('#'):
                continue

            # ── simulate() ──
            sim_params = _parse_simulate_action(line)
            if sim_params is not None:
                plan = self._prepare_simulate_run(
                    state, sim_params, model, action_index, timeout,
                    seed_suffix_prefix='protocol:', null_nf_sample_times=False,
                )
                last_result = self._run_prepared_simulate(
                    plan, stop_log_label='protocol stop_if triggered')
                state.current_time = plan.t_end
                continue

            # ── setConcentration() ──
            sc = _parse_set_concentration(line)
            if sc is not None:
                species_name, conc_value = sc
                try:
                    model.set_concentration(species_name, conc_value)
                except Exception:
                    logger.warning("protocol: setConcentration(%s, %s) failed",
                                   species_name, conc_value)
                continue

            # ── addConcentration() ──
            ac = _parse_add_concentration(line)
            if ac is not None:
                species_name, delta = ac
                try:
                    current = model.get_concentration(species_name)
                    model.set_concentration(species_name, current + delta)
                except Exception:
                    logger.warning("protocol: addConcentration(%s, %s) failed",
                                   species_name, delta)
                continue

            # ── setParameter() ──
            sp = _parse_set_parameter(line)
            if sp is not None:
                param_name, param_value = sp
                try:
                    model.set_param(param_name, param_value)
                except Exception:
                    logger.warning("protocol: setParameter(%s, %s) failed",
                                   param_name, param_value)
                continue

            # ── resetConcentrations() ──
            if _is_reset_concentrations(line):
                model.reset()
                continue

            # ── saveConcentrations() ──
            if _is_save_concentrations(line):
                model.save_concentrations()
                continue

            # ── saveParameters() ──
            if _is_save_parameters(line):
                saved_params = {}
                for pname in model.param_names:
                    try:
                        saved_params[pname] = model.get_param(pname)
                    except Exception:
                        logger.debug(
                            "protocol: saveParameters could not read %s", pname)
                continue

            # ── resetParameters() ──
            if _is_reset_parameters(line):
                for pname, pval in saved_params.items():
                    try:
                        model.set_param(pname, pval)
                    except Exception:
                        logger.debug(
                            "protocol: resetParameters could not restore %s=%s", pname, pval)
                state.sim = _runtime.bngsim.Simulator(model, method=state.method, **self._codegen_kwargs(state.method))
                continue

            logger.debug("protocol: skipping unrecognized command: %s", line)

        return last_result

    def _prepare_simulate_run(self, state, sim_params, model, action_index, timeout,
                              seed_suffix_prefix, null_nf_sample_times):
        """Parse one simulate() action and prepare it to run.

        Shared by :meth:`_execute_actions` and :meth:`_run_protocol`. Reads the
        ``continue=>1`` start time from ``state.current_time`` and recreates the
        simulator in ``state`` when the method (or psa poplevel) changes, then
        returns a :class:`_SimulateRunPlan`. It deliberately does NOT run the
        simulation: ``_execute_actions`` records per-action diagnostic info
        between preparing and running (the protocol path doesn't), so running is
        a separate step (:meth:`_run_prepared_simulate`).

        ``seed_suffix_prefix`` distinguishes the auto-seed namespace of the two
        callers ('' for actions, 'protocol:' for protocol). ``null_nf_sample_times``
        reproduces the per-caller handling of the unsupported NFsim+sample_times
        combination (the actions path warns and drops sample_times; the protocol
        path never reaches NFsim, so it left it untouched).
        """
        method, poplevel = _normalize_action_method(
            sim_params.get('method', 'ode'),
            sim_params.get('poplevel'),
        )

        # Parse sample_times (list of string values from BNGL)
        sample_times = _resolve_sample_times(sim_params)

        # Gap 1: continue=>1
        continue_flag = bool(int(float(sim_params.get('continue', 0))))
        if continue_flag and 't_start' not in sim_params:
            t_start = state.current_time
        else:
            t_start = float(sim_params.get('t_start', 0))
        t_end = float(sim_params.get('t_end', 100))
        n_steps = int(sim_params.get('n_steps', 100))
        suffix = sim_params.get('suffix', 'time_course')

        # Gap 4: print_functions
        print_funcs = bool(int(float(sim_params.get('print_functions', 0))))

        # Gap 3: atol, rtol, seed
        run_kwargs = {}
        if 'atol' in sim_params:
            run_kwargs['atol'] = float(sim_params['atol'])
        if 'rtol' in sim_params:
            run_kwargs['rtol'] = float(sim_params['rtol'])
        explicit_seed = None
        if 'seed' in sim_params:
            explicit_seed = int(float(sim_params['seed']))
        if method in ('ssa', 'psa'):
            seed_value = self._resolve_action_seed(
                explicit_seed=explicit_seed,
                action_index=action_index,
                suffix=seed_suffix_prefix + suffix,
                method=method,
            )
            if seed_value is not None:
                run_kwargs['seed'] = seed_value
        elif explicit_seed is not None:
            run_kwargs['seed'] = explicit_seed

        # Gap 2: stop_if
        stop_if = sim_params.get('stop_if')
        if stop_if is not None:
            stop_if = stop_if.strip().strip('"').strip("'")

        # sample_times is not supported for NFsim (possible future
        # BNGsim / NFsim enhancement)
        if null_nf_sample_times and method == 'nf' and sample_times is not None:
            logger.warning(
                "sample_times is not supported for NFsim — ignoring")
            sample_times = None

        if method == 'psa':
            if state.method != 'psa' or state.poplevel != poplevel:
                state.sim = _runtime.bngsim.Simulator(
                    model,
                    method='psa',
                    poplevel=poplevel,
                )
                state.method = 'psa'
                state.poplevel = poplevel
        elif state.method != method:
            state.sim = _runtime.bngsim.Simulator(model, method=method, **self._codegen_kwargs(method))
            state.method = method
            state.poplevel = None

        if stop_if:
            state.sim.add_stop_condition(stop_if, label=stop_if)

        run_timeout = _normalize_sim_timeout(timeout, method=method)
        if run_timeout is not None:
            run_kwargs['timeout'] = run_timeout

        return _SimulateRunPlan(
            sim=state.sim, method=method, suffix=suffix, print_funcs=print_funcs,
            sample_times=sample_times, t_start=t_start, t_end=t_end, n_steps=n_steps,
            run_kwargs=run_kwargs, stop_if=stop_if,
        )

    @staticmethod
    def _run_prepared_simulate(plan, stop_log_label):
        """Run a prepared simulate() plan, handling stop_if/StopConditionMet.

        ``stop_log_label`` labels the info-log emitted when a stop condition
        fires ('stop_if triggered' vs 'protocol stop_if triggered').
        """
        try:
            if plan.sample_times is not None:
                result = plan.sim.run(
                    t_span=(plan.sample_times[0], plan.sample_times[-1]),
                    n_points=len(plan.sample_times),
                    sample_times=plan.sample_times,
                    **plan.run_kwargs,
                )
            else:
                result = plan.sim.run(
                    t_span=(plan.t_start, plan.t_end),
                    n_points=plan.n_steps + 1,
                    **plan.run_kwargs,
                )
        except Exception as exc:
            if isinstance(exc, _runtime.bngsim.StopConditionMet):
                logger.info("%s: %s", stop_log_label, plan.stop_if)
                result = exc.result
            else:
                raise

        if plan.stop_if:
            plan.sim.clear_stop_conditions()

        return result

    def _codegen_kwargs(self, method='ode'):
        """Return codegen keyword args for ODE Simulator construction."""
        if method == 'ode' and getattr(self, '_codegen_so', ''):
            return {'codegen': True, 'net_path': self._net_path}
        return {}

    def _make_scan_simulator(self, model, method, poplevel):
        """Construct a fresh simulator for one parameter-scan point."""
        if method == 'psa':
            return _runtime.bngsim.Simulator(
                model,
                method='psa',
                poplevel=poplevel,
            )
        return _runtime.bngsim.Simulator(model, method=method, **self._codegen_kwargs(method))

    def _sync_species_initial_concentrations(self, model):
        """Re-evaluate .net species initializers using the model's current params."""
        if not self._net_species_initializers:
            return

        param_values = {}
        for pname in model.param_names:
            try:
                param_values[pname] = model.get_param(pname)
            except Exception:
                pass

        ns = _build_safe_eval_namespace(param_values)
        for species_name, expr_text in self._net_species_initializers:
            try:
                value = float(eval(expr_text, ns))  # noqa: S307
            except Exception:
                logger.debug(
                    "BngsimModel: could not re-evaluate initial concentration %s = %r",
                    species_name,
                    expr_text,
                )
                continue
            try:
                model.set_concentration(species_name, value)
            except Exception:
                logger.debug(
                    "BngsimModel: could not set initial concentration for %s",
                    species_name,
                )
        model.save_concentrations()

    def _prepare_scan_point_model(
        self, model, param_name, value, concentration_overrides=None,
    ):
        """Clone the base model, apply the scan parameter, and refresh initials.

        ``concentration_overrides`` is the dict of active setConcentration()
        expressions seen since the last reset/save; each is re-evaluated
        against the *cloned* point model's parameter namespace (i.e. after
        the scan parameter has been set), then applied and baked into the
        clone's initial concentrations. See issue #46.
        """
        point_model = model.clone()
        if param_name:
            point_model.set_param(param_name, float(value))
        self._sync_species_initial_concentrations(point_model)
        active_overrides = concentration_overrides or {}
        for species_name, expr in active_overrides.items():
            point_model.set_concentration(
                species_name,
                _eval_model_expression(expr, point_model),
            )
        if active_overrides:
            point_model.save_concentrations()
        point_model.reset()
        return point_model

    def _run_ss_scan_threaded(
        self, model, param_name, points, method, poplevel,
        t_start, t_end, print_funcs, concentration_overrides=None,
        max_workers=4, timeout=None,
    ):
        """Run steady-state parameter scan with threaded parallelism.

        Prepares all point models sequentially (species initializer sync is not
        thread-safe), then submits steady_state() calls to a thread pool.
        Falls back to long time-course per point on non-convergence or error.
        """
        n_workers = min(len(points), max_workers)
        obs_names = []
        expr_names = []
        rows = []

        # Prepare models and simulators sequentially (not thread-safe)
        point_models = []
        point_sims = []
        for value in points:
            pm = self._prepare_scan_point_model(
                model, param_name, value,
                concentration_overrides=concentration_overrides,
            )
            ps = self._make_scan_simulator(pm, method, poplevel)
            point_models.append(pm)
            point_sims.append(ps)

        # Run steady_state() in parallel
        def _solve_ss(idx):
            """Returns (idx, ss_result_or_None, exception_or_None)."""
            try:
                ss_result = point_sims[idx].steady_state()
                return (idx, ss_result, None)
            except Exception as exc:
                return (idx, None, exc)

        ss_outcomes = [None] * len(points)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_solve_ss, i): i for i in range(len(points))}
            for fut in concurrent.futures.as_completed(futures):
                idx, ss_result, exc = fut.result()
                ss_outcomes[idx] = (ss_result, exc)

        # Process results and handle fallbacks
        for i, value in enumerate(points):
            ss_result, exc = ss_outcomes[i]
            ss_ok = False

            if exc is not None:
                logger.warning(
                    "BngsimModel: steady-state solver failed for %s=%s: %s. "
                    "Falling back to long time-course.",
                    param_name, value, exc,
                )
            elif ss_result.converged:
                point_model = point_models[i]
                for j, name in enumerate(ss_result.species_names):
                    point_model.set_concentration(
                        name, ss_result.concentrations[j],
                    )
                point_model.save_concentrations()
                point_model.reset()
                eval_sim = self._make_scan_simulator(point_model, 'ode', None)
                eval_kwargs = {}
                eval_timeout = _normalize_sim_timeout(timeout, method='ode')
                if eval_timeout is not None:
                    eval_kwargs['timeout'] = eval_timeout
                result = eval_sim.run(t_span=(0, 1e-10), n_points=2, **eval_kwargs)
                ss_ok = True
            else:
                logger.warning(
                    "BngsimModel: steady-state solver did not converge for "
                    "%s=%s (residual=%.2e). Falling back to long time-course.",
                    param_name, value, getattr(ss_result, 'residual', None),
                )

            if not ss_ok:
                fb_model = self._prepare_scan_point_model(
                    model, param_name, value,
                    concentration_overrides=concentration_overrides,
                )
                fb_sim = self._make_scan_simulator(fb_model, method, poplevel)
                fb_kwargs = {}
                fb_timeout = _normalize_sim_timeout(timeout, method=method)
                if fb_timeout is not None:
                    fb_kwargs['timeout'] = fb_timeout
                result = fb_sim.run(t_span=(t_start, t_end), n_points=2, **fb_kwargs)

            row, row_obs, row_expr = self._scan_result_to_row(
                result, value, print_functions=print_funcs,
            )
            if len(obs_names) == 0:
                obs_names = row_obs
                expr_names = row_expr
            rows.append(row)

        return rows, obs_names, expr_names

    def _run_parameter_scan(self, model, ps_params, is_bifurcate=False, action_index=0,
                            timeout=None, concentration_overrides=None):
        """Execute a parameter_scan() or bifurcate() action.

        Resolves the action's settings once, dispatches to the matching scan
        strategy, and assembles the per-point rows into a Data object. See the
        ``_scan_*`` helpers for each strategy.
        """
        s = self._resolve_scan_settings(
            ps_params, is_bifurcate, action_index, timeout, concentration_overrides,
        )
        if s.use_ss and s.reset_conc and s.ss_method == _SS_METHOD_NEWTON:
            rows, obs_names, expr_names = self._scan_newton_steady_state(model, s)
        elif s.use_ss and s.reset_conc:
            rows, obs_names, expr_names = self._scan_parity_steady_state(model, s)
        elif s.method == 'protocol':
            rows, obs_names, expr_names = self._scan_protocol(model, s)
        elif not s.reset_conc:
            rows, obs_names, expr_names = self._scan_continuation(model, s)
        else:
            rows, obs_names, expr_names = self._scan_independent(model, s)
        return self._assemble_scan_data(rows, obs_names, expr_names, s)

    def _resolve_scan_settings(self, ps_params, is_bifurcate, action_index, timeout,
                               concentration_overrides):
        """Parse a parameter_scan()/bifurcate() action into a `_ScanSettings` bundle.

        Resolves the seed, sample times, steady-state mode, method/poplevel, and
        per-point timeouts once, applying the bifurcate-specific overrides
        (reset_conc=False, ss_method=>"newton" downgrade) and the
        steady_state/method-compatibility fallback warnings.
        """
        concentration_overrides = concentration_overrides or {}
        param_name = ps_params.get('parameter', '')
        t_start = float(ps_params.get('t_start', 0))
        t_end = float(ps_params.get('t_end', 100))
        suffix = ps_params.get('suffix', 'param_scan')
        use_ss = int(ps_params.get('steady_state', 0))
        ss_method = _normalize_ss_method(ps_params.get('ss_method'))
        print_funcs = bool(int(float(ps_params.get('print_functions', 0))))
        method, poplevel = _normalize_action_method(
            ps_params.get('method', 'ode'),
            ps_params.get('poplevel'),
        )

        # Resolve seed once per scan action; bngsim varies per scan-point in
        # run_batch (base+i), and same-seed-different-θ in per-point fallback
        # produces distinct trajectories per the user's clarification.
        explicit_seed = None
        if 'seed' in ps_params:
            explicit_seed = int(float(ps_params['seed']))
        if method in ('ssa', 'psa'):
            scan_seed = self._resolve_action_seed(
                explicit_seed=explicit_seed,
                action_index=action_index,
                suffix=suffix,
                method=method,
            )
        elif explicit_seed is not None:
            scan_seed = explicit_seed
        else:
            scan_seed = None

        # Resolve sample_times for passthrough to each scan-point simulation
        sample_times = _resolve_sample_times(ps_params)
        if sample_times is not None:
            t_start = sample_times[0]
            t_end = sample_times[-1]

        # bifurcate forces reset_conc=False; parameter_scan defaults to True
        if is_bifurcate:
            reset_conc = False
        else:
            reset_conc = bool(int(float(ps_params.get('reset_conc', 1))))

        # bifurcate is a continuation scan: it carries state between points
        # (reset_conc=False) to trace hysteresis/multistability. Independent
        # per-point Newton (steady_state_batch) finds *a* root and can jump to
        # the wrong branch, destroying the hysteresis signal — so ss_method=>
        # "newton" is rejected here and downgraded to the parity ODE path.
        if is_bifurcate and ss_method == _SS_METHOD_NEWTON:
            logger.warning(
                "BngsimModel: ss_method=>\"newton\" is not supported for "
                "bifurcate continuation scans (it carries state between points "
                "to detect hysteresis, which independent-per-point Newton would "
                "destroy). Downgrading to BNG2.pl-parity integrate-to-steady-state."
            )
            ss_method = _SS_METHOD_PARITY

        if use_ss and method != 'ode':
            logger.warning(
                "BngsimModel: steady_state=>1 is only supported for ODE parameter scans. "
                "Falling back to time-course scan for method=%s.",
                method,
            )
            use_ss = 0

        points = _resolve_scan_points(ps_params)

        scan_timeout = _normalize_sim_timeout(timeout, method=method)
        scan_eval_timeout = _normalize_sim_timeout(timeout, method='ode')

        return _ScanSettings(
            param_name=param_name,
            t_start=t_start,
            t_end=t_end,
            suffix=suffix,
            use_ss=use_ss,
            ss_method=ss_method,
            print_funcs=print_funcs,
            method=method,
            poplevel=poplevel,
            scan_seed=scan_seed,
            sample_times=sample_times,
            reset_conc=reset_conc,
            points=points,
            concentration_overrides=concentration_overrides,
            timeout=timeout,
            scan_timeout=scan_timeout,
            scan_eval_timeout=scan_eval_timeout,
        )

    def _scan_newton_steady_state(self, model, s):
        """steady_state=>1 + ss_method=>"newton"/"kinsol": KINSOL Newton per point.

        Uses the threaded batch path when safe (ODE, no expression-based species
        initializers, >=4 points); otherwise solves each dose-response point
        independently, falling back to a long time-course when the solver fails
        or does not converge.
        """
        param_name = s.param_name
        points = s.points
        method = s.method
        poplevel = s.poplevel
        t_start = s.t_start
        t_end = s.t_end
        print_funcs = s.print_funcs
        concentration_overrides = s.concentration_overrides
        timeout = s.timeout
        scan_timeout = s.scan_timeout
        scan_eval_timeout = s.scan_eval_timeout
        obs_names = []
        expr_names = []
        rows = []
        # ss_method=>"newton" (opt-in accelerator): KINSOL Newton per
        # independent dose-response point. Use threaded path when safe:
        # ODE method, no expression-based species initializers, and enough
        # points to justify threading.
        use_threaded_ss = (
            method == 'ode'
            and not self._net_species_initializers
            and len(points) >= 4
        )
        if use_threaded_ss:
            rows, obs_names, expr_names = self._run_ss_scan_threaded(
                model, param_name, points, method, poplevel,
                t_start, t_end, print_funcs,
                concentration_overrides=concentration_overrides,
                timeout=timeout,
            )
        else:
            for value in points:
                point_model = self._prepare_scan_point_model(
                    model,
                    param_name,
                    value,
                    concentration_overrides=concentration_overrides,
                )
                point_sim = self._make_scan_simulator(
                    point_model,
                    method,
                    poplevel,
                )
                ss_ok = False
                try:
                    ss_result = point_sim.steady_state()
                    if ss_result.converged:
                        for i, name in enumerate(ss_result.species_names):
                            point_model.set_concentration(
                                name,
                                ss_result.concentrations[i],
                            )
                        point_model.save_concentrations()
                        point_model.reset()
                        eval_sim = self._make_scan_simulator(point_model, 'ode', None)
                        result = eval_sim.run(
                            t_span=(0, 1e-10), n_points=2,
                            **_with_sim_timeout({}, scan_eval_timeout),
                        )
                        ss_ok = True
                    else:
                        logger.warning(
                            "BngsimModel: steady-state solver did not converge for "
                            "%s=%s (residual=%.2e). Falling back to long time-course.",
                            param_name, value, getattr(ss_result, 'residual', None),
                        )
                except Exception as exc:
                    logger.warning(
                        "BngsimModel: steady-state solver failed for %s=%s: %s. "
                        "Falling back to long time-course.",
                        param_name, value, exc,
                    )
                if not ss_ok:
                    # Fallback: simulate for a long time and take the final state
                    point_model = self._prepare_scan_point_model(
                        model, param_name, value,
                        concentration_overrides=concentration_overrides,
                    )
                    fallback_sim = self._make_scan_simulator(
                        point_model, method, poplevel,
                    )
                    result = fallback_sim.run(
                        t_span=(t_start, t_end), n_points=2,
                        **_with_sim_timeout({}, scan_timeout),
                    )
                row, row_obs, row_expr = self._scan_result_to_row(
                    result, value, print_functions=print_funcs,
                )
                if len(obs_names) == 0:
                    obs_names = row_obs
                    expr_names = row_expr
                rows.append(row)
        return rows, obs_names, expr_names

    def _scan_parity_steady_state(self, model, s):
        """steady_state=>1, ss_method omitted/"integrate": BNG2.pl-parity per point.

        Integrates each independent point to steady state via
        run(steady_state=True) — the run_network -c ||f||2/n early-stop. The
        final integrated row IS the equilibrium for this point.
        """
        param_name = s.param_name
        points = s.points
        t_start = s.t_start
        t_end = s.t_end
        print_funcs = s.print_funcs
        concentration_overrides = s.concentration_overrides
        scan_eval_timeout = s.scan_eval_timeout
        obs_names = []
        expr_names = []
        rows = []
        for value in points:
            point_model = self._prepare_scan_point_model(
                model, param_name, value,
                concentration_overrides=concentration_overrides,
            )
            point_sim = self._make_scan_simulator(point_model, 'ode', None)
            result = point_sim.run(
                t_span=(t_start, t_end), n_points=_SS_SCAN_N_POINTS,
                steady_state=True,
                **_with_sim_timeout({}, scan_eval_timeout),
            )
            if not int(result.solver_stats.get('steady_state_reached', 0)):
                logger.warning(
                    "BngsimModel: parity steady-state criterion not reached "
                    "for %s=%s within t_end=%s; using final integrated state.",
                    param_name, value, t_end,
                )
            row, row_obs, row_expr = self._scan_result_to_row(
                result, value, print_functions=print_funcs,
            )
            if len(obs_names) == 0:
                obs_names = row_obs
                expr_names = row_expr
            rows.append(row)
        return rows, obs_names, expr_names

    def _scan_protocol(self, model, s):
        """method=>"protocol": run the begin protocol...end protocol block per point."""
        param_name = s.param_name
        points = s.points
        print_funcs = s.print_funcs
        concentration_overrides = s.concentration_overrides
        timeout = s.timeout
        obs_names = []
        expr_names = []
        rows = []
        if not self._protocol:
            raise ValueError(
                'parameter_scan method=>"protocol" but no '
                'begin protocol...end protocol block found'
            )
        for value in points:
            point_model = self._prepare_scan_point_model(
                model, param_name, value,
                concentration_overrides=concentration_overrides,
            )
            last_result = self._run_protocol(point_model, timeout=timeout)
            if last_result is None:
                raise ValueError(
                    'protocol contains no simulate actions'
                )
            row, row_obs, row_expr = self._scan_result_to_row(
                last_result, value, print_functions=print_funcs,
            )
            if len(obs_names) == 0:
                obs_names = row_obs
                expr_names = row_expr
            rows.append(row)
        return rows, obs_names, expr_names

    def _scan_continuation(self, model, s):
        """bifurcate / reset_conc=>0: carry model state between points.

        When steady_state=>1, each point integrates to the parity steady state
        (run_network -c) carrying the previous point's equilibrium forward — the
        continuation that traces hysteresis. ss_method=>"newton" was already
        downgraded to parity in _resolve_scan_settings for bifurcate.
        """
        param_name = s.param_name
        points = s.points
        method = s.method
        poplevel = s.poplevel
        use_ss = s.use_ss
        sample_times = s.sample_times
        scan_seed = s.scan_seed
        t_start = s.t_start
        t_end = s.t_end
        print_funcs = s.print_funcs
        scan_timeout = s.scan_timeout
        obs_names = []
        expr_names = []
        rows = []
        running_model = model.clone()
        for value in points:
            if param_name:
                running_model.set_param(param_name, float(value))
            self._sync_species_initial_concentrations(running_model)
            point_sim = self._make_scan_simulator(
                running_model,
                method,
                poplevel,
            )
            ss_kwargs = {'steady_state': True} if use_ss else {}
            if sample_times is not None:
                result = point_sim.run(
                    t_span=(sample_times[0], sample_times[-1]),
                    n_points=len(sample_times),
                    sample_times=sample_times,
                    seed=scan_seed,
                    **ss_kwargs,
                    **_with_sim_timeout({}, scan_timeout),
                )
            else:
                result = point_sim.run(
                    t_span=(t_start, t_end),
                    n_points=_SS_SCAN_N_POINTS if use_ss else 2,
                    seed=scan_seed,
                    **ss_kwargs,
                    **_with_sim_timeout({}, scan_timeout),
                )
            row, row_obs, row_expr = self._scan_result_to_row(
                result, value, print_functions=print_funcs,
            )
            if len(obs_names) == 0:
                obs_names = row_obs
                expr_names = row_expr
            rows.append(row)
        return rows, obs_names, expr_names

    def _scan_independent(self, model, s):
        """Default independent per-point scan: run_batch() when safe, else sequential.

        Uses bngsim run_batch() (parallel, parameter-only) when there are no
        expression-based species initializers, no active setConcentration()
        overrides, no sample_times, and >=4 points (issue #46); otherwise
        simulates each point independently. Falls back to the sequential path if
        run_batch() raises (re-raising SimulationTimeout).
        """
        param_name = s.param_name
        points = s.points
        method = s.method
        poplevel = s.poplevel
        t_start = s.t_start
        t_end = s.t_end
        print_funcs = s.print_funcs
        concentration_overrides = s.concentration_overrides
        sample_times = s.sample_times
        scan_seed = s.scan_seed
        scan_timeout = s.scan_timeout
        obs_names = []
        expr_names = []
        rows = []
        # Use run_batch() when safe: no expression-based species
        # initializers, no active setConcentration() overrides (each
        # point needs its own concentration setup, which run_batch's
        # parameter-only API cannot express), no sample_times, enough
        # points. Issue #46.
        use_batch = (
            not self._net_species_initializers
            and not concentration_overrides
            and sample_times is None
            and len(points) >= 4
        )
        if use_batch:
            params = [{param_name: float(v)} for v in points]
            n_workers = min(len(points), 4)
            batch_sim = self._make_scan_simulator(model, method, poplevel)
            try:
                batch_results = batch_sim.run_batch(
                    t_span=(t_start, t_end),
                    n_points=2,
                    params=params,
                    num_processors=n_workers,
                    seed=scan_seed,
                    **_with_sim_timeout({}, scan_timeout),
                )
            except Exception as exc:
                if isinstance(exc, _runtime.bngsim.SimulationTimeout):
                    raise
                logger.warning(
                    "BngsimModel: run_batch() failed; falling back to "
                    "sequential scan.",
                    exc_info=True,
                )
                use_batch = False

        if use_batch:
            for i, value in enumerate(points):
                row, row_obs, row_expr = self._scan_result_to_row(
                    batch_results[i], value, print_functions=print_funcs,
                )
                if len(obs_names) == 0:
                    obs_names = row_obs
                    expr_names = row_expr
                rows.append(row)
        else:
            for value in points:
                point_model = self._prepare_scan_point_model(
                    model,
                    param_name,
                    value,
                    concentration_overrides=concentration_overrides,
                )
                point_sim = self._make_scan_simulator(
                    point_model,
                    method,
                    poplevel,
                )
                if sample_times is not None:
                    result = point_sim.run(
                        t_span=(sample_times[0], sample_times[-1]),
                        n_points=len(sample_times),
                        sample_times=sample_times,
                        seed=scan_seed,
                        **_with_sim_timeout({}, scan_timeout),
                    )
                else:
                    result = point_sim.run(
                        t_span=(t_start, t_end), n_points=2,
                        seed=scan_seed,
                        **_with_sim_timeout({}, scan_timeout),
                    )
                row, row_obs, row_expr = self._scan_result_to_row(
                    result, value, print_functions=print_funcs,
                )
                if len(obs_names) == 0:
                    obs_names = row_obs
                    expr_names = row_expr
                rows.append(row)
        return rows, obs_names, expr_names

    def _assemble_scan_data(self, rows, obs_names, expr_names, s):
        """Stack the per-point scan rows into a Data object keyed by the scan suffix."""
        if rows:
            arr = np.vstack(rows)
        else:
            arr = np.zeros((0, 1))

        headers = [s.param_name] + obs_names + expr_names
        return {s.suffix: Data.from_columns(arr, headers)}

    @staticmethod
    def _scan_result_to_row(result, scan_value, print_functions=False):
        """Convert the final point of a scan result into one row plus headers."""
        obs_names = list(result.observable_names)
        obs_array = np.asarray(result.observables)
        if obs_array.ndim == 2 and obs_array.shape[0] > 0:
            final_obs = obs_array[-1, :]
        else:
            final_obs = np.array([])

        if print_functions:
            expr_names = list(result.expression_names)
            expr_array = np.asarray(result.expressions)
            if expr_array.ndim == 2 and expr_array.shape[0] > 0 and expr_array.shape[1] > 0:
                final_expr = expr_array[-1, :]
            else:
                final_expr = np.array([])
        else:
            expr_names = []
            final_expr = np.array([])

        row = np.concatenate((
            np.array([scan_value], dtype=float),
            np.asarray(final_obs, dtype=float),
            np.asarray(final_expr, dtype=float),
        ))
        return row, obs_names, expr_names

    @staticmethod
    def _result_to_data(result, print_functions=False):
        """Convert a bngsim Result to a PyBNF Data object."""
        obs_names = list(result.observable_names)
        n_times = result.n_times
        n_obs = result.n_observables

        if print_functions:
            expr_names = list(result.expression_names)
            expr_array = np.asarray(result.expressions)
            n_expr = len(expr_names)
        else:
            expr_names = []
            expr_array = np.zeros((n_times, 0))
            n_expr = 0

        arr = np.zeros((n_times, 1 + n_obs + n_expr))
        arr[:, 0] = result.time
        obs_array = np.asarray(result.observables)
        arr[:, 1:1 + n_obs] = obs_array
        if n_expr > 0 and expr_array.size > 0:
            arr[:, 1 + n_obs:] = expr_array

        headers = ['time'] + obs_names + expr_names
        return Data.from_columns(arr, headers)

    def _get_mutant_model_bngsim(self, mut):
        """Create a mutant copy using a cloned engine model."""
        mut_model = copy.copy(self)
        mut_model._engine_model = self._engine_model.clone()
        mut_model.param_set = _build_mutant_param_set(self.param_set, mut)
        return mut_model

    def __getstate__(self):
        """Support pickling for Dask workers by dropping the C++ model object."""
        state = self.__dict__.copy()
        state.pop('_engine_model', None)
        state.pop('_codegen_so', None)
        return state

    def __setstate__(self, state):
        """Restore from pickle by re-loading the .net file."""
        self.__dict__.update(state)
        if hasattr(self, '_net_path') and self._net_path:
            self._engine_model = _runtime.bngsim.Model.from_net(self._net_path)
            self._codegen_so = _try_prepare_codegen(self._net_path)
        else:
            raise RuntimeError("Cannot unpickle BngsimModel: no _net_path")

    def save(self, file_prefix, **kwargs):
        """Still write debug/export files via the NetModel implementation."""
        super().save(file_prefix)


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
