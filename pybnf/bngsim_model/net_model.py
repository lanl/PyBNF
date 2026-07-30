"""The bngsim net-backend model: BngsimModel (ODE/SSA/PSA, scans, protocols).

Executes BNGL action lines in-process via the bngsim Simulator, including the
parameter_scan/bifurcate strategies and steady-state handling. Reads the bngsim
package + capability flags through the _runtime seam (ADR-0018).
"""


import concurrent.futures
import copy
import logging
import os
import re
from dataclasses import dataclass

import numpy as np

from . import _runtime
from ..data import Data, OutputSensitivities, stack_scan_sensitivities
from ..pset import NetModel
from ..printing import PybnfError
from .._seed import resolve_action_seed
from .parsing import (
    _collapse_action_line_continuations,
    _parse_simulate_action,
    _parse_parameter_scan_action,
    _parse_bifurcate_action,
    _parse_set_parameter,
    _parse_set_concentration,
    _parse_set_concentration_expr,
    _parse_add_concentration,
    _is_reset_concentrations,
    _is_reset_parameters,
    _is_save_concentrations,
    _is_save_parameters,
)
from .expressions import (
    _build_safe_eval_namespace,
    _eval_numeric,
    _eval_model_expression,
    _build_mutant_param_set,
    _parse_net_species_initializers,
    _net_species_ic_seed_map,
)
from .scan import (
    _with_sim_timeout,
    _resolve_scan_points,
    _resolve_sample_times,
    _SS_METHOD_PARITY,
    _SS_METHOD_NEWTON,
    _SS_SCAN_N_POINTS,
)
from .output import _write_saved_action_outputs
from .classification import (
    _normalize_action_method,
    _normalize_sim_timeout,
    _normalize_ss_method,
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

    ``carried_state`` mirrors bngsim's own "the persistent simulator was advanced
    by a prior ``run()`` with no reset since" notion (the gradient pre-equilibration
    seam, GH #210 / #457): ``True`` once a simulate has run on the current
    ``sim``, cleared by a ``resetConcentrations()`` (``model.reset()``) or a
    simulator (re)creation. On the gradient path it is the trigger for
    ``carry_sensitivities=True`` so the measurement phase's forward-sensitivity
    ICs are seeded from the equilibration steady-state sensitivity ``dx_ss/dθ``
    instead of zero (ADR-0052).
    """
    sim: object
    method: str = 'ode'
    poplevel: object = None
    current_time: float = 0.0
    carried_state: bool = False


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


@dataclass
class _SensitivityRequest:
    """The forward-sensitivity request that activates a model's gradient path.

    Set by :meth:`BngsimModel.enable_output_sensitivities` (#385/#447); ``None``
    on the scalar path. ``params`` are native model parameter ids routed to
    ``Simulator(sensitivity_params=)`` and ``ic`` are species initial values
    routed to ``Simulator(sensitivity_ic=)`` (the routing lists themselves are
    populated by #448). Native parameter space throughout -- no transform here.
    """
    params: list   # native model parameter ids -> sensitivity_params
    ic: list       # species initial-value names -> sensitivity_ic


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

    # Gradient path (#385/#447): None on the scalar path, a _SensitivityRequest
    # once enable_output_sensitivities() activates forward sensitivities. A class
    # attribute (not only an __init__ assignment) so the scalar path stays intact
    # for instances built via object.__new__ (test fakes, pickling).
    _sensitivity_request = None

    # Per-action sensitivity gate (#475): on the gradient path only an action
    # whose output is a SCORED gradient target needs forward sensitivities. An
    # incidental/unscored action -- a stochastic (ssa/nfsim) diagnostic, or a
    # carried-state pre-equilibration parameter_scan (#474) -- carries none, so it
    # runs on the ordinary path and neither computes a wasted sensitivity tensor
    # nor aborts the whole fit at a guard it can never satisfy. ``_scored_suffixes``
    # is the model's scored full-suffix set (set_scored_suffixes, from exp_data);
    # ``_sensitivity_offset`` is this instance's suffix offset (a mutant's suffix,
    # '' for the base) folded onto an action's own suffix to key that set;
    # ``_current_action_suffix`` names the action being prepared. All three are
    # class attributes so an unset (``None``) scored set falls back to the historical
    # all-actions-bearing behavior (any pre-#475 caller of enable_output_sensitivities).
    _scored_suffixes = None
    _sensitivity_offset = ''
    _current_action_suffix = None

    # Transient per-scan sensitivity accumulator (#476): a gradient-supporting scan
    # strategy (reset-to-seed parity steady-state or independent dose-response) fills
    # this with the per-dose-point OutputSensitivities during ``_run_parameter_scan``,
    # which then stacks them onto the scan Data and clears it. A class attribute (like
    # ``_current_action_suffix``) so an object.__new__ instance is safe; set and read
    # synchronously within one scan, never pickled.
    _pending_scan_sens = None

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

    def enable_output_sensitivities(self, *, params=None, ic=None):
        """Activate the gradient path: request forward sensitivities ∂g/∂θ.

        Routes ``params`` (native model parameter ids) to
        ``Simulator(sensitivity_params=)`` and ``ic`` (species initial-value
        names) to ``Simulator(sensitivity_ic=)`` at every subsequent ODE run,
        carrying the resulting tensor onto each simulated :class:`Data`. The
        routing lists themselves come from #448; this method only stores them.

        Gates on the backend capability (#447): a build without forward output
        sensitivities refuses here with an actionable message rather than letting
        a gradient-based fit start and fail deep in the backend. The version floor
        is unaffected -- scalar (metaheuristic) fits never call this.
        """
        if not _runtime.BNGSIM_HAS_OUTPUT_SENS:
            reason = _runtime.feature_missing_reason('output_sensitivities')
            raise PybnfError(
                "Gradient-based fitting needs forward output sensitivities, which "
                "this bngsim build does not provide (%s). Install a bngsim build "
                "with the 'output_sensitivities' feature, or run a gradient-free "
                "fit." % (reason or 'feature unavailable')
            )
        self._sensitivity_request = _SensitivityRequest(
            params=list(params or []), ic=list(ic or []),
        )

    def set_scored_suffixes(self, suffixes):
        """Record which output suffixes are scored gradient targets (#475).

        On the gradient path only a SCORED action's output needs forward
        sensitivities; an incidental/unscored action -- a stochastic (ssa/nfsim)
        diagnostic, or a carried-state pre-equilibration ``parameter_scan`` (#474)
        -- runs sensitivity-free so it neither computes a wasted tensor nor aborts
        the fit at a differentiability guard it can never satisfy. ``suffixes`` is
        the model's ``exp_data`` mapping (or any iterable of scored *full*
        suffixes -- a mutant's own suffix is folded in per-instance via
        :attr:`_sensitivity_offset`, so pass the full set once for the base model
        and every mutant copy shares it). Set by the gradient optimizer's
        ``_setup_gradient_path`` before the model scatter, so it rides the pickle
        to the workers alongside the sensitivity request.
        """
        self._scored_suffixes = set(suffixes)

    def _action_bears_sensitivities(self):
        """Whether the action currently being prepared is a scored gradient target.

        The per-action half of the #475 gate: an action carries forward
        sensitivities only on the gradient path (``_sensitivity_request`` active)
        AND when its output suffix (with this instance's mutant offset) is in the
        scored set. The scalar path returns ``False`` -- no action bears
        sensitivities there, so a carried-state scan runs unguarded exactly as
        before. On the gradient path a scored set that is unknown, or a current
        suffix not yet resolved, returns ``True`` (bearing), so any path that
        activates the request without declaring scored suffixes keeps the
        historical all-actions-bearing behavior.
        """
        if self._sensitivity_request is None:
            return False
        scored = self._scored_suffixes
        if scored is None:
            return True
        suffix = self._current_action_suffix
        if suffix is None:
            return True
        return (suffix + self._sensitivity_offset) in scored

    @property
    def has_discrete_events(self):
        """True iff the engine model contains state-jumping discrete events (#461).

        A discrete event reinitialises the integrator state discontinuously, but
        bngsim's CVODES forward-sensitivity vectors are *not* reinitialised across
        the jump, so the sensitivity columns go silently stale at and after an event
        fires -- bngsim therefore refuses forward output sensitivities outright on
        such a model rather than return wrong derivatives (bngsim GH #205). The
        gradient path reads this as its pre-flight differentiability gate
        (:meth:`GradientOptimizer._require_differentiable_dynamics`) to refuse a
        discrete-event model **up front** -- with an actionable "use a metaheuristic
        fit_type" message -- instead of letting the fit start and fail at the first
        sensitivity-bearing ``simulate()``.

        Only true state-jumping events are counted (the engine core's ``n_events``).
        Discontinuity triggers -- forcing pulses / piecewise-time dosing schedules --
        break the integrator step but do not jump state, so sensitivities through
        them stay valid and are intentionally not counted. ``False`` when the engine
        model or its event count is unavailable (an older/stub backend), so the gate
        never blocks on a missing signal.
        """
        core = getattr(self._engine_model, '_core', None)
        return bool(getattr(core, 'n_events', 0))

    def sensitivity_entity_namespace(self):
        """The bind-by-id namespaces the gradient router classifies free parameters against (#448).

        Returns ``(param_ids, species_initializers, ic_seed_map)``:

        * ``param_ids`` -- the model's ``begin parameters`` namespace (the engine's
          ``param_names``), the kinetic/global ids a free parameter binds to via ``set_param``
          and thus routes to ``Simulator(sensitivity_params=)``;
        * ``species_initializers`` -- the ``(species, initial-expr)`` pairs
          (``_parse_net_species_initializers``); a free parameter that is a species' bare
          initial-value expression binds to ``Simulator(sensitivity_ic=)`` keyed by the
          species (an IC parameter is absent from the ODE RHS, so the parameter axis is zero);
        * ``ic_seed_map`` -- ``{model parameter -> species}`` for a bare species initializer
          (``_net_species_ic_seed_map``), so the router can route a free parameter a condition
          assigns to that parameter onto the species' IC axis (a per-condition estimated initial
          condition, ADR-0076, #511); a non-routable seed maps to ``None``.

        This is the only model coupling :mod:`pybnf.gradient.routing` needs, so the routing
        core stays backend-agnostic. No simulation -- all three are known at build time.
        """
        param_ids = list(self._engine_model.param_names)
        species_initializers = list(self._net_species_initializers)
        return (param_ids, species_initializers,
                _net_species_ic_seed_map(species_initializers, param_ids))

    def _sensitivity_request_kwargs(self, method):
        """Simulator kwargs requesting forward sensitivities on the gradient path.

        Returns ``{}`` on the scalar path (request inactive), so the Simulator
        construction is byte-identical to before this feature existed. On the
        gradient path it returns ``sensitivity_params=``/``sensitivity_ic=`` for an
        ODE Simulator.

        A non-ODE method (ssa/psa/nfsim) is non-differentiable -- forward
        sensitivities are deterministic-ODE only (#447). Whether that is an error
        now depends on whether the action's output is a *scored* gradient target
        (#475): a scored non-ODE action still refuses cleanly (a PyBNF-level error,
        not a backend traceback -- its gradient genuinely cannot be supplied),
        while an incidental/unscored non-ODE action (a stochastic diagnostic that
        no data scores) runs sensitivity-free rather than aborting the whole fit.
        ODE actions are always built sensitivity-bearing: the persistent ODE
        Simulator is reused across actions and carries sensitivities through
        carried states (#457), so an unscored ODE action's unused tensor is left
        harmless rather than risk that continuity by toggling it off mid-protocol.
        """
        req = self._sensitivity_request
        if req is None:
            return {}
        if method != 'ode':
            if self._action_bears_sensitivities():
                raise PybnfError(
                    "Model %s: gradient-based fitting requires deterministic ODE "
                    "integration, but a scored simulate() action requests method=%r. "
                    "Forward output sensitivities are available only for the ODE "
                    "backend; run a gradient-free fit for stochastic/network-free "
                    "simulation." % (self.name, method)
                )
            # Incidental/unscored non-ODE action (#475): no sensitivities needed,
            # so build a plain Simulator and let it run on the ordinary path.
            return {}
        kwargs = {}
        if req.params:
            kwargs['sensitivity_params'] = list(req.params)
        if req.ic:
            kwargs['sensitivity_ic'] = list(req.ic)
        return kwargs

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

        # Re-derive species initial concentrations from the just-applied params.
        # A flattened .net materializes species ICs as concrete numbers at load,
        # and set_param touches only the parameter table / rate laws -- so a free
        # parameter that *only* seeds a species IC (e.g. ``S() <- S0``) would be a
        # silent no-op without this sync (#450). No-op (and byte-unchanged) for
        # models with no param-driven species initializers; the scan paths already
        # sync via _prepare_scan_point_model.
        self._sync_species_initial_concentrations(model)

        model.reset()
        self._pybnf_current_action_info = None
        try:
            ds = self._execute_actions(model, timeout=timeout)
        except PybnfError:
            # Clean PyBNF-level refusal (e.g. a stochastic action on the gradient
            # path) -- surface it as-is, no failure report or generic wrapping.
            raise
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
            if self._sensitivity_request is not None:
                # On the gradient path a backend raise (e.g. bngsim rejecting
                # forward sensitivities for a model with discrete events) must
                # surface as an actionable PyBNF-level message, not a raw
                # backend traceback. The original is chained for diagnostics.
                # Lead with the underlying error and keep the non-differentiable-
                # model diagnosis explicitly conditional (#525): this wrapper also
                # catches failures that are nothing to do with the model's
                # constructs, and naming discrete events unconditionally sent a
                # reporter looking at an event-free model.
                raise PybnfError(
                    "Model %s: simulation failed while computing forward output "
                    "sensitivities for gradient-based fitting: %s. If the model uses "
                    "discrete events or other non-differentiable constructs, forward "
                    "sensitivities are unavailable for it; otherwise the failure report "
                    "written under FailedSimLogs/ names the failing action and parameter "
                    "set. A gradient-free job_type (e.g. job_type = de) needs no "
                    "sensitivities." % (self.name, exc)
                ) from exc
            raise

        if self.save_files:
            _write_saved_action_outputs(folder, filename, self.suffixes, ds)

        if with_mutants:
            for mut in self.mutants:
                # Off-diagonal cross-product pruning (#484): skip building/running a
                # condition mutant entirely when no action pairs with it (its whole column
                # is off-diagonal). A no-op when emit_suffixes is unset.
                if self.emit_suffixes is not None and not any(
                        (s[1] + mut.suffix) in self.emit_suffixes for s in self.suffixes):
                    continue
                logger.debug('Working on mutant %s', mut.suffix)
                mut_model = self._get_mutant_model_bngsim(mut)
                # The mutant runs under its own condition suffix, so _execute_actions keys
                # its emit-set lookups by <action suffix><mut.suffix> (the diagonal for the
                # experiments this condition owns). Shares the base model's emit_suffixes via
                # copy.copy in _get_mutant_model_bngsim (like _sensitivity_offset).
                mut_model._emit_context_suffix = mut.suffix
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
        # No action's suffix is resolved yet, so the initial ODE Simulator is
        # built sensitivity-bearing on the gradient path (harmless -- ODE always
        # bears; see _sensitivity_request_kwargs). Each action then sets
        # _current_action_suffix before (re)constructing its own Simulator (#475).
        self._current_action_suffix = None
        state = _SimulateActionState(
            sim=_runtime.bngsim.Simulator(
                model, method='ode',
                **self._codegen_kwargs(), **self._sensitivity_request_kwargs('ode')))

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
                # Off-diagonal cross-product pruning (#484): skip a simulate whose
                # (action, condition) pair no consumer reads. _emit_context_suffix is ''
                # for the base run and the mutant's suffix on its copy, so this prunes both.
                # A no-op when emit_suffixes is unset; an unregistered pre-equilibration
                # phase (<name>_preequil) is never pruned (see Model._emit_skip). The
                # preceding resetConcentrations() line still ran, so a kept later action
                # starts clean.
                if self._emit_skip(sim_params.get('suffix', 'time_course')):
                    continue
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
                state.carried_state = True   # the simulator now holds an advanced state
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
                state.carried_state = False   # reset clears the carried-over state
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
                # A saveConcentrations() snapshots the current amounts as the new
                # reset point but does NOT un-pin an active param-dependent
                # setConcentration expression: BNG2.pl keeps it live so a
                # following parameter_scan re-evaluates it per point (the
                # preincubate->wash->dose-scan idiom saves the post-wash state and
                # then titrates the competitor per dose -- issue #474). So the
                # overrides carry THROUGH a save (only resetConcentrations(), which
                # returns to the seed, clears them, above).
                model.save_concentrations()
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
                # Off-diagonal cross-product pruning (#484): skip a dose-response scan
                # whose (action, condition) pair no consumer reads (as for simulate above).
                if self._emit_skip(ps_params.get('suffix', 'param_scan')):
                    continue
                ds.update(self._run_parameter_scan(
                    model, ps_params,
                    action_index=action_index,
                    timeout=timeout,
                    concentration_overrides=concentration_overrides,
                    carried_state=state.carried_state,
                ))
                continue

            bf_params = _parse_bifurcate_action(line)
            if bf_params is not None:
                ds.update(self._run_parameter_scan(
                    model, bf_params, is_bifurcate=True,
                    action_index=action_index,
                    timeout=timeout,
                    concentration_overrides=concentration_overrides,
                    carried_state=state.carried_state,
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
            sim=_runtime.bngsim.Simulator(
                model, method='ode',
                **self._codegen_kwargs(), **self._sensitivity_request_kwargs('ode')))
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
                state.carried_state = True   # the simulator now holds an advanced state
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
                state.carried_state = False   # reset clears the carried-over state
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
                state.sim = _runtime.bngsim.Simulator(
                    model, method=state.method,
                    **self._codegen_kwargs(state.method),
                    **self._sensitivity_request_kwargs(state.method))
                state.carried_state = False   # a fresh simulator carries no state
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
        continue_flag = bool(int(_eval_numeric(str(sim_params.get('continue', 0)))))
        if continue_flag and 't_start' not in sim_params:
            t_start = state.current_time
        else:
            t_start = _eval_numeric(str(sim_params.get('t_start', 0)))
        t_end = _eval_numeric(str(sim_params.get('t_end', 100)))
        n_steps = int(_eval_numeric(str(sim_params.get('n_steps', 100))))
        suffix = sim_params.get('suffix', 'time_course')
        # Gate this action's sensitivity request on whether its output is scored
        # (#475): set before any Simulator (re)construction below reads it.
        self._current_action_suffix = suffix

        # Gap 4: print_functions
        print_funcs = bool(int(_eval_numeric(str(sim_params.get('print_functions', 0)))))

        # Gap 3: atol, rtol, seed
        run_kwargs = {}
        if 'atol' in sim_params:
            run_kwargs['atol'] = _eval_numeric(str(sim_params['atol']))
        if 'rtol' in sim_params:
            run_kwargs['rtol'] = _eval_numeric(str(sim_params['rtol']))
        explicit_seed = None
        if 'seed' in sim_params:
            explicit_seed = int(_eval_numeric(str(sim_params['seed'])))
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

        # steady_state=>1 on a simulate() (ADR-0052, new-era pre-equilibration): integrate
        # to steady state (early-stop on ||dx/dt||) rather than to a fixed endpoint. This
        # wires the simulate path to bngsim's existing ``Simulator.run(steady_state=True)``
        # parity primitive -- the same early-stop the steady-state parameter_scan uses
        # (``_scan_parity_steady_state``) -- so an UNMEASURED equilibration phase relaxes to
        # equilibrium, then a subsequent simulate carries that state over (no
        # resetConcentrations between them). ``t_end`` remains the max-time bound for the
        # run if steady state is not reached within the window. Additive: no existing
        # simulate action emits ``steady_state``; only parameter_scan did (its own path).
        if bool(int(_eval_numeric(str(sim_params.get('steady_state', 0))))):
            run_kwargs['steady_state'] = True

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
                    **self._sensitivity_request_kwargs('psa'),
                )
                state.method = 'psa'
                state.poplevel = poplevel
                state.carried_state = False   # a fresh simulator carries no state
        elif state.method != method:
            state.sim = _runtime.bngsim.Simulator(
                model, method=method,
                **self._codegen_kwargs(method), **self._sensitivity_request_kwargs(method))
            state.method = method
            state.poplevel = None
            state.carried_state = False       # a fresh simulator carries no state

        # Pre-equilibration sensitivity continuity (ADR-0052, GH #210 / #457): on the
        # gradient path, a measurement-phase simulate that continues a carried-over species
        # state (a prior run() on this same persistent simulator, no reset since -- e.g. the
        # steady-state equilibration phase) must seed its forward-sensitivity ICs from that
        # phase's final sensitivity ``dx_ss/dθ`` rather than zero. bngsim does the implicit-
        # function-theorem seeding when ``carry_sensitivities=True``; it *requires*
        # sensitivity_params and *raises* if sensitivities are requested on a carried-over
        # state without the flag (and, conversely, if the flag is set on a fresh run), so the
        # condition below mirrors bngsim's own carried-state notion exactly. The scalar path
        # (no _sensitivity_request) never sets it -- byte-identical pre-equilibration there.
        # carry_sensitivities requires sensitivity_params (an initial-condition column cannot be
        # carried across a stable steady state -- ∂x*/∂x(0) = 0 -- and the backend refuses it),
        # so the parameter axis is what gates the flag; an IC-only request never reaches here.
        req = self._sensitivity_request
        if req is not None and req.params and method == 'ode' and state.carried_state:
            run_kwargs['carry_sensitivities'] = True

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
                **self._sensitivity_request_kwargs('psa'),
            )
        return _runtime.bngsim.Simulator(
            model, method=method,
            **self._codegen_kwargs(method), **self._sensitivity_request_kwargs(method))

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
                            timeout=None, concentration_overrides=None, carried_state=False):
        """Execute a parameter_scan() or bifurcate() action.

        Resolves the action's settings once, dispatches to the matching scan
        strategy, and assembles the per-point rows into a Data object. See the
        ``_scan_*`` helpers for each strategy.

        ``carried_state`` is True when a preceding ``simulate`` advanced the model
        off its seed state (a pre-equilibration + intervention, e.g. a
        preincubate->wash->dose-scan protocol; issue #474). Such a scan resets
        each point to the *carried* post-intervention state, not the ``.net``
        seed, so it routes to bngsim's native reset_conc-to-snapshot scan
        (:meth:`_scan_carried_state`, lanl/bngsim#11) rather than the
        seed-re-syncing hand-rolled strategies below (which are correct only for a
        fresh-from-seed dose-response, ADR-0046).
        """
        s = self._resolve_scan_settings(
            ps_params, is_bifurcate, action_index, timeout, concentration_overrides,
        )
        # Gate this scan's sensitivity request on whether its output is scored
        # (#475): set before any per-point Simulator below reads it (all scan
        # strategies build their simulators through _sensitivity_request_kwargs,
        # and _scan_carried_state's refusal consults _action_bears_sensitivities).
        self._current_action_suffix = s.suffix
        # Reset the per-scan sensitivity accumulator (#476): a gradient-supporting
        # strategy (parity steady-state / independent, both reset-to-seed) fills it
        # with the per-dose-point tensors; every other strategy leaves it None.
        self._pending_scan_sens = None
        if carried_state:
            rows, obs_names, expr_names = self._scan_carried_state(model, s, is_bifurcate)
        elif s.use_ss and s.reset_conc and s.ss_method == _SS_METHOD_NEWTON:
            rows, obs_names, expr_names = self._scan_newton_steady_state(model, s)
        elif s.use_ss and s.reset_conc:
            rows, obs_names, expr_names = self._scan_parity_steady_state(model, s)
        elif s.method == 'protocol':
            rows, obs_names, expr_names = self._scan_protocol(model, s)
        elif not s.reset_conc:
            rows, obs_names, expr_names = self._scan_continuation(model, s)
        else:
            rows, obs_names, expr_names = self._scan_independent(model, s)
        ds = self._assemble_scan_data(rows, obs_names, expr_names, s)
        # Gradient path (#476): stack the per-dose-point forward sensitivities the
        # strategy collected onto the assembled scan Data, so gradient assembly can
        # consume d(dose-response)/dtheta. None on the scalar path (accumulator never
        # filled) or when any point lacked a tensor -> byte-identical scalar Data.
        pending = self._pending_scan_sens
        if pending is not None:
            sens = stack_scan_sensitivities(pending)
            if sens is not None:
                ds[s.suffix].output_sensitivities = sens
        self._pending_scan_sens = None
        return ds

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
        t_start = _eval_numeric(str(ps_params.get('t_start', 0)))
        t_end = _eval_numeric(str(ps_params.get('t_end', 100)))
        suffix = ps_params.get('suffix', 'param_scan')
        use_ss = int(_eval_numeric(str(ps_params.get('steady_state', 0))))
        ss_method = _normalize_ss_method(ps_params.get('ss_method'))
        print_funcs = bool(int(_eval_numeric(str(ps_params.get('print_functions', 0)))))
        method, poplevel = _normalize_action_method(
            ps_params.get('method', 'ode'),
            ps_params.get('poplevel'),
        )

        # Resolve seed once per scan action; bngsim varies per scan-point in
        # run_batch (base+i), and same-seed-different-θ in per-point fallback
        # produces distinct trajectories per the user's clarification.
        explicit_seed = None
        if 'seed' in ps_params:
            explicit_seed = int(_eval_numeric(str(ps_params['seed'])))
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
            reset_conc = bool(int(_eval_numeric(str(ps_params.get('reset_conc', 1)))))

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

    def _scan_carried_state(self, model, s, is_bifurcate):
        """Scan invoked with a CARRIED model state -- a ``simulate`` advanced the
        system off its seed before the scan (pre-equilibration + intervention;
        issue #474, the preincubate->wash->dose-scan protocol).

        Uses bngsim's native ``Simulator.parameter_scan``/``bifurcate``
        (lanl/bngsim#11) whose ``reset_conc`` semantics match BNG2.pl: each point
        resets to the state **at scan invocation** (the carried post-intervention
        state -- receptors already loaded during the pre-incubation), assigns the
        scanned parameter, then replays the active ``setConcentration`` overrides
        (species that track the scanned parameter, e.g. the titrated competitor)
        via ``on_point`` -- WITHOUT re-deriving species from the ``.net`` seed
        initializers. Contrast :meth:`_prepare_scan_point_model` (the
        fresh-from-seed dose-response path, ADR-0046), whose seed re-sync would
        discard the pre-equilibrated state.

        bngsim refuses a scan on a sensitivity-configured Simulator (per-point
        seeds off a mid-protocol snapshot would be wrong), so a *scored*
        carried-state scan is unavailable on the gradient path -- surfaced as a
        clear PyBNF error. An incidental/unscored carried-state scan (#475) needs
        no sensitivities, so it runs on the ordinary sensitivity-free path below
        (the simulators built here never pass sensitivity kwargs).
        """
        if self._action_bears_sensitivities():
            raise PybnfError(
                "Model %s: a scored parameter_scan following a pre-equilibration "
                "(a carried, non-seed model state) cannot be run on the gradient "
                "path -- bngsim refuses per-point sensitivity seeds taken off a "
                "mid-protocol snapshot. Run a gradient-free fit for a "
                "pre-equilibrated dose-response experiment." % self.name)
        method = s.method
        if method == 'psa':
            sim = _runtime.bngsim.Simulator(model, method='psa', poplevel=s.poplevel)
        else:
            sim = _runtime.bngsim.Simulator(
                model, method=method, **self._codegen_kwargs(method))

        overrides = s.concentration_overrides or {}

        def on_point(point_model, value):
            # Replay the setConcentration expressions active at scan invocation
            # (issue #46) against each point's post-reset state -- re-evaluated
            # with the just-assigned scanned parameter, so a competitor whose
            # amount tracks the scanned dose (setConcentration("cold",
            # "dose*(NA*Vecf)")) is titrated per point. Species NOT named here
            # keep their carried snapshot value (the pre-equilibrated state).
            for species_name, expr in overrides.items():
                try:
                    point_model.set_concentration(
                        species_name, _eval_model_expression(expr, point_model))
                except Exception:
                    logger.warning(
                        "BngsimModel: scan on_point setConcentration(%s, %s) failed",
                        species_name, expr)

        n_points = len(s.sample_times) if s.sample_times is not None else 2
        scan_kwargs = dict(
            parameter=s.param_name,
            par_scan_vals=list(s.points),
            t_span=(s.t_start, s.t_end),
            n_points=n_points,
            steady_state=bool(s.use_ss),
        )
        if s.scan_seed is not None:
            scan_kwargs['seed'] = s.scan_seed
        if s.scan_timeout is not None:
            scan_kwargs['timeout'] = s.scan_timeout

        # bifurcate is the continuation sibling (reset_conc pinned False, no
        # per-point reset/override -- each point continues from the previous
        # point's end state); parameter_scan resets each point to the invocation
        # snapshot (reset_to=None) and replays overrides via on_point.
        if is_bifurcate:
            results = sim.bifurcate(**scan_kwargs)
        else:
            results = sim.parameter_scan(
                reset_conc=s.reset_conc, reset_to=None, on_point=on_point, **scan_kwargs)
        if not isinstance(results, list):
            results = [results]

        obs_names = []
        expr_names = []
        rows = []
        for value, result in zip(s.points, results):
            row, row_obs, row_expr = self._scan_result_to_row(
                result, value, print_functions=s.print_funcs)
            if len(obs_names) == 0:
                obs_names = row_obs
                expr_names = row_expr
            rows.append(row)
        return rows, obs_names, expr_names

    def _scan_newton_steady_state(self, model, s):
        """steady_state=>1 + ss_method=>"newton"/"kinsol": KINSOL Newton per point.

        Uses the threaded batch path when safe (ODE, no expression-based species
        initializers, >=4 points); otherwise solves each dose-response point
        independently, falling back to a long time-course when the solver fails
        or does not converge.

        Gradient path (#478): the KINSOL algebraic solve returns ``dY_ss/dp``
        exactly (the implicit-function-theorem derivative on the analytical
        Jacobian, not FD), and bngsim>=0.11.35 (lanl/bngsim#12) maps it to
        observable/expression sensitivities on the ``SteadyStateResult`` -- so a
        *scored* Newton scan now collects ``d(dose-response)/dtheta`` per point
        exactly as the parity path does, making ``ss_method=>"newton"`` a genuine
        speed win under a gradient fit rather than the #476 fall-back-to-parity
        refusal. The gradient path runs the points **sequentially** (the KINSOL
        sensitivity solve is not confirmed thread-safe, mirroring how #476
        bypasses ``run_batch``) and reads each point's slice from the
        ``SteadyStateResult`` accessor when KINSOL converges, or -- on the
        non-convergence fallback -- from the CVODE long-time-course ``Result``,
        which is itself sensitivity-bearing; the two are byte-shape-identical so
        they stack down the dose axis together. A build without the accessor
        (bngsim<0.11.35) refuses cleanly with an upgrade hint; an
        incidental/unscored Newton scan runs sensitivity-free unchanged.
        """
        collect_sens = self._action_bears_sensitivities()
        if collect_sens and not _runtime.BNGSIM_HAS_SS_OUTPUT_SENS:
            raise PybnfError(
                "Model %s: a scored parameter_scan with ss_method=>\"newton\"/\"kinsol\" "
                "needs observable-level steady-state forward sensitivities, which this "
                "bngsim build does not provide (%s). Upgrade to bngsim>=0.11.35, omit "
                "ss_method (the BNG2.pl-parity integrate-to-steady-state default IS "
                "differentiable), or run a gradient-free fit."
                % (self.name,
                   _runtime.feature_missing_reason('output_sensitivities')
                   or 'SteadyStateResult.output_sensitivities unavailable'))
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
        # Gradient path: ask KINSOL for dY_ss/dp against the fitted params.
        sens_params = list(self._sensitivity_request.params) if collect_sens else None
        obs_names = []
        expr_names = []
        rows = []
        # ss_method=>"newton" (opt-in accelerator): KINSOL Newton per
        # independent dose-response point. Use threaded path when safe:
        # ODE method, no expression-based species initializers, and enough
        # points to justify threading. The gradient path (#478) takes the
        # sequential loop instead -- it collects a per-point sensitivity tensor
        # (which the threaded path does not) and keeps the KINSOL sensitivity
        # solve off the thread pool.
        use_threaded_ss = (
            method == 'ode'
            and not self._net_species_initializers
            and len(points) >= 4
            and not collect_sens
        )
        if use_threaded_ss:
            rows, obs_names, expr_names = self._run_ss_scan_threaded(
                model, param_name, points, method, poplevel,
                t_start, t_end, print_funcs,
                concentration_overrides=concentration_overrides,
                timeout=timeout,
            )
        else:
            # Per-dose-point forward-sensitivity slices (#478); None on the scalar
            # path, so the accumulator below is left untouched there.
            sens_slices = []
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
                point_sens = None
                try:
                    # sens_params is None on the scalar path and a (possibly empty,
                    # IC-only fit) list on the gradient path; ask KINSOL for
                    # dY_ss/dp only when there is at least one parameter to solve.
                    ss_kwargs = {'sensitivity_params': sens_params} if sens_params else {}
                    ss_result = point_sim.steady_state(**ss_kwargs)
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
                        if collect_sens:
                            # ∂g/∂θ at equilibrium comes from the KINSOL solve's
                            # implicit-function-theorem sensitivity -- NOT the tiny
                            # eval run above, whose fresh-IC sensitivities are ~0.
                            point_sens = self._extract_ss_output_sensitivities(
                                ss_result, print_funcs)
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
                    # Fallback: simulate for a long time and take the final state.
                    # On the gradient path this CVODE run is itself sensitivity-
                    # configured, so its final-row ∂g/∂θ (integrated to t_end, at
                    # equilibrium) is read via the same extractor as the parity
                    # path -- byte-shape-identical to a converged Newton slice.
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
                    if collect_sens:
                        point_sens = self._scan_point_sensitivities(result, print_funcs)
                row, row_obs, row_expr = self._scan_result_to_row(
                    result, value, print_functions=print_funcs,
                )
                if len(obs_names) == 0:
                    obs_names = row_obs
                    expr_names = row_expr
                rows.append(row)
                sens_slices.append(point_sens)
            if collect_sens:
                self._pending_scan_sens = sens_slices
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
        # Gradient path (#476): each dose point is an independent, reset-to-seed
        # sensitivity-configured ODE run, so ∂obs/∂θ at its equilibrium row is
        # well-posed; collect it per point for stacking down the dose axis.
        sens_slices = []
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
            sens_slices.append(self._scan_point_sensitivities(result, print_funcs))
        self._pending_scan_sens = sens_slices
        return rows, obs_names, expr_names

    def _scan_protocol(self, model, s):
        """method=>"protocol": run the begin protocol...end protocol block per point.

        Gradient path (#476): a per-point protocol chains several simulate phases
        (pre-equilibration, interventions) whose forward-sensitivity continuity is
        not yet wired for a scan, so a *scored* protocol scan refuses cleanly; an
        incidental one runs unchanged.
        """
        if self._action_bears_sensitivities():
            raise PybnfError(
                "Model %s: a scored parameter_scan with method=>\"protocol\" cannot be "
                "run on the gradient path -- per-point protocol forward sensitivities "
                "are not yet supported. Run a gradient-free fit." % self.name)
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

        Gradient path (#476): each point's initial state is the previous point's
        theta-dependent end state, so a correct sensitivity seed would have to be
        chained from point to point (dx0/dtheta != 0) -- not yet supported. A
        *scored* continuation / bifurcate refuses cleanly here; an incidental one
        runs unchanged.
        """
        if self._action_bears_sensitivities():
            raise PybnfError(
                "Model %s: a scored continuation parameter_scan (reset_conc=>0) or "
                "bifurcate cannot be run on the gradient path -- each point carries a "
                "parameter-dependent seed from the previous point, whose sensitivity "
                "seed-chaining is not yet supported. Use a reset-to-seed dose-response "
                "(reset_conc=>1, the default) or run a gradient-free fit." % self.name)
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
        # points. Issue #46. On the gradient path (#476) an ODE scan's per-point
        # simulators are sensitivity-configured (ODE always bears, #457) and
        # run_batch() cannot return forward output sensitivities, so such a scan
        # takes the sequential per-point run() loop -- which both collects the
        # tensor (for a scored scan) and avoids a doomed run_batch()+fallback.
        gradient_ode = self._sensitivity_request is not None and method == 'ode'
        use_batch = (
            not self._net_species_initializers
            and not concentration_overrides
            and sample_times is None
            and len(points) >= 4
            and not gradient_ode
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
            # Gradient path (#476): each independent, reset-to-seed point is a
            # sensitivity-configured ODE run, so collect ∂obs/∂θ at its final row
            # for stacking down the dose axis (None on the scalar path).
            sens_slices = []
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
                sens_slices.append(self._scan_point_sensitivities(result, print_funcs))
            self._pending_scan_sens = sens_slices
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
    def _build_data(result, print_functions=False):
        """Convert a bngsim Result to a PyBNF Data object (time + obs + expr).

        The pure scalar-path conversion: no sensitivities. Kept static and
        unchanged so the network-free backend (``nf_model``) can reuse it and so
        the scalar-path :class:`Data` stays byte-identical to before the gradient
        path existed. :meth:`_result_to_data` layers sensitivities on top.
        """
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

    def _result_to_data(self, result, print_functions=False):
        """Convert a bngsim Result to a PyBNF Data object.

        Scalar path: returns :meth:`_build_data` verbatim (``output_sensitivities``
        stays ``None``). Gradient path (``_sensitivity_request`` active): also
        attaches the native-space forward-sensitivity tensor keyed by the same
        observable/expression selectors as the Data columns (#385/#447). The
        attachment never perturbs the value columns -- it is read off the same
        Result -- so a gradient-path Data's numeric trajectory is whatever bngsim
        integrated; only the scalar path is guaranteed byte-identical to before.
        """
        data = self._build_data(result, print_functions=print_functions)
        if self._sensitivity_request is not None and getattr(
                result, 'has_sensitivities', False):
            data.output_sensitivities = self._extract_output_sensitivities(
                result, print_functions)
        return data

    @staticmethod
    def _differentiable_expression_names(result):
        """The expressions on ``result`` that bngsim can hand back an output sensitivity for.

        bngsim differentiates a global function symbolically (lanl/bngsim#198) and
        **refuses**, per function, any body carrying a non-differentiable construct --
        an ``if()`` conditional, a comparison, ``min``/``max``/``abs``/``floor``, a table
        function. It records the per-function verdict on the Result, and
        ``output_sensitivities`` raises ``ValueError`` if a refused function is among the
        requested selectors. ``has_sensitivities_expressions`` is only "some expression
        block exists", not "every expression is differentiable", so it cannot stand in.

        Asking for a refused selector fails the whole simulation, and a *scored* column
        is almost never one of these functions -- so filter them out. This matters for
        any piecewise model: an epidemic model whose rates switch at ``if(t >= tau)``
        has every one of its functions refused, and before this filter every simulation
        of such a model died with ``ValueError`` on the gradient path, making the fit's
        objective ``inf`` everywhere. Should a scored column genuinely be a refused
        expression, its absence surfaces later as a clean "no sensitivity column"
        gradient error rather than a dead simulation.

        Empty support map (an older bngsim, or a Result loaded from disk) ⇒ no verdicts
        to filter on ⇒ every expression is kept, exactly as before.
        """
        names = list(getattr(result, 'expression_names', None) or [])
        support = getattr(result, '_expression_sens_support', None) or {}
        if not support:
            return names
        return [name for name in names if support.get(name) is None]

    @classmethod
    def _extract_output_sensitivities(cls, result, print_functions):
        """Read the native-space ∂g/∂θ tensor off a sensitivity-bearing Result.

        Selectors mirror the Data columns -- ``observable:<name>`` for every
        observable, plus ``expression:<name>`` for each *differentiable* expression
        (:meth:`_differentiable_expression_names`) when ``print_functions`` is on and
        the backend computed expression sensitivities. The ``parameter`` axis is read
        whenever sensitivity params were requested; the ``ic`` axis whenever IC species
        were.
        """
        selectors = ['observable:%s' % name for name in result.observable_names]
        if print_functions and getattr(result, 'has_sensitivities_expressions', False):
            selectors += ['expression:%s' % name
                          for name in cls._differentiable_expression_names(result)]
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
        return OutputSensitivities(
            selectors=selectors, param_names=param_names, ic_species=ic_species,
            d_param=d_param, d_ic=d_ic,
        )

    def _extract_ss_output_sensitivities(self, ss_result, print_functions):
        """Read the observable/expression ∂g/∂θ tensor off a KINSOL ``SteadyStateResult`` (#478).

        The Newton/KINSOL algebraic solve returns ``dY_ss/dp`` **exactly** (the
        implicit-function-theorem derivative on the analytical Jacobian, not FD);
        bngsim>=0.11.35 (lanl/bngsim#12) maps it through the observable/function
        Jacobian and exposes it via ``SteadyStateResult.output_sensitivities``
        with the same ``observable:``/``expression:`` selectors the CVODE
        ``Result`` uses -- so this mirrors :meth:`_extract_output_sensitivities`
        exactly, differing only in two structural ways:

        * A ``SteadyStateResult`` is a single equilibrium point, so its tensor has
          no leading time axis (shape ``(n_selectors, n_params)``). We add a
          singleton row so the layout is ``(1, n_selectors, n_params)`` -- the
          same rank a time-course tensor has, letting
          :func:`~pybnf.data.stack_scan_sensitivities` pick the final row
          (``[-1]``) identically for a converged Newton point and a CVODE
          fallback point.
        * The IC axis is identically zero: a stable steady state forgets its
          initial conditions (``∂x*/∂x(0)=0``, #457), so an IC-seed fit parameter
          contributes no gradient at equilibrium. bngsim declines the ``ic`` axis
          on a ``SteadyStateResult`` for exactly this reason, so we report it as an
          explicit zero tensor (parity with the integrate-to-steady-state path,
          whose IC sensitivities have decayed to ~0) when IC sensitivities were
          requested, and ``None`` otherwise.
        """
        selectors = ['observable:%s' % name for name in ss_result.observable_names]
        if print_functions and getattr(ss_result, 'expression_names', None):
            selectors += ['expression:%s' % name
                          for name in self._differentiable_expression_names(ss_result)]
        param_names = list(ss_result.sensitivity_params)
        d_param = None
        if param_names:
            # (n_selectors, n_params) -> (1, n_selectors, n_params): a singleton
            # equilibrium "row" so the stacker's final-row selection is uniform.
            d_param = np.asarray(
                ss_result.output_sensitivities(selectors, axis='parameter'),
                dtype=float,
            )[np.newaxis, ...]
        req = self._sensitivity_request
        ic_species = list(req.ic) if req is not None else []
        d_ic = None
        if ic_species:
            d_ic = np.zeros((1, len(selectors), len(ic_species)), dtype=float)
        return OutputSensitivities(
            selectors=selectors, param_names=param_names, ic_species=ic_species,
            d_param=d_param, d_ic=d_ic,
        )

    def _scan_point_sensitivities(self, result, print_functions):
        """Per-dose-point forward-sensitivity tensor for a reset-to-seed scan point (#476).

        ``None`` on the scalar path (no request) or when the point's ``Result`` carries
        no tensor (a non-ODE / non-sensitivity run); otherwise the full per-point
        :class:`~pybnf.data.OutputSensitivities` (all integrated rows), whose final row
        the scan stacks down the dose axis (:func:`pybnf.data.stack_scan_sensitivities`).
        Reuses :meth:`_extract_output_sensitivities` so the scan tensor's selectors /
        axis labels are byte-identical to the time-course path's."""
        if self._sensitivity_request is None or not getattr(
                result, 'has_sensitivities', False):
            return None
        return self._extract_output_sensitivities(result, print_functions)

    def _get_mutant_model_bngsim(self, mut):
        """Create a mutant copy using a cloned engine model."""
        mut_model = copy.copy(self)
        mut_model._engine_model = self._engine_model.clone()
        mut_model.param_set = _build_mutant_param_set(self.param_set, mut, self._engine_model)
        # A mutant's action output is scored under ``<action suffix><mut.suffix>``
        # in the parent's dataset, so fold its suffix onto each action's own suffix
        # when keying the scored set for the #475 gate (the shallow copy already
        # shares the base model's _scored_suffixes / _sensitivity_request).
        mut_model._sensitivity_offset = mut.suffix
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
