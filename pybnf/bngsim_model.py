"""Optional in-process BNGL -> .net simulation using bngsim."""


import copy
import logging
import math
import os
import re

import numpy as np

from .data import Data
from .pset import FreeParameter, NetModel, PSet


logger = logging.getLogger(__name__)


try:
    if os.environ.get('PYBNF_NO_BNGSIM'):
        raise ImportError('PYBNF_NO_BNGSIM set')
    import bngsim
    BNGSIM_AVAILABLE = True
except ImportError:
    bngsim = None
    BNGSIM_AVAILABLE = False


_PARAMETER_SCAN_KEY_ALIASES = {
    'param': 'parameter',
    'time': 't_end',
    'min': 'par_min',
    'max': 'par_max',
    'logspace': 'log_scale',
}


def _collapse_action_line_continuations(action_line):
    """Collapse BNGL trailing-backslash line continuations into one action."""
    return re.sub(r'\\\s*\n\s*', '', action_line)


def _extract_action_body(action_line, action_name):
    """Return the body inside action_name({...}) or None if it doesn't match."""
    collapsed = _collapse_action_line_continuations(action_line).strip()
    pattern = r'\s*%s\s*\(\s*\{(.*)\}\s*\)\s*$' % re.escape(action_name)
    match = re.match(pattern, collapsed, re.DOTALL)
    if not match:
        return None
    return match.group(1)


def _split_top_level_commas(text):
    """Split an action body on commas while ignoring nested lists and quotes."""
    items = []
    start = 0
    depth = 0
    quote = None
    escaped = False

    for i, ch in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == quote:
                quote = None
            continue

        if ch in ('"', "'"):
            quote = ch
        elif ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth = max(depth - 1, 0)
        elif ch == ',' and depth == 0:
            item = text[start:i].strip()
            if item:
                items.append(item)
            start = i + 1

    tail = text[start:].strip()
    if tail:
        items.append(tail)
    return items


def _parse_action_value(value_text):
    """Parse a BNGL action value, preserving scalars as strings and lists as lists."""
    value = value_text.strip()
    if not value:
        return value

    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]

    if value.startswith('[') and value.endswith(']'):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_action_value(item) for item in _split_top_level_commas(inner)]

    return value


def _parse_action_dict(action_line, action_name, key_aliases=None):
    """Parse action_name({...}) into a dict, honoring top-level commas only."""
    body = _extract_action_body(action_line, action_name)
    if body is None:
        return None

    params = {}
    aliases = key_aliases or {}
    for item in _split_top_level_commas(body):
        if '=>' not in item:
            logger.debug(
                "BngsimModel: skipping malformed %s item %r",
                action_name,
                item,
            )
            continue
        key_text, value_text = item.split('=>', 1)
        key = aliases.get(key_text.strip(), key_text.strip())
        params[key] = _parse_action_value(value_text)
    return params


def _parse_simulate_action(action_line):
    return _parse_action_dict(action_line, 'simulate')


def _parse_parameter_scan_action(action_line):
    return _parse_action_dict(
        action_line,
        'parameter_scan',
        key_aliases=_PARAMETER_SCAN_KEY_ALIASES,
    )


def _parse_set_parameter(action_line):
    """Parse setParameter("name", value) -> (name, value) or None."""
    match = re.match(
        r'\s*setParameter\s*\(\s*["\'](\w+)["\']\s*,\s*([\d.eE+\-]+)\s*\)',
        action_line,
    )
    if match:
        return match.group(1), float(match.group(2))
    return None


def _parse_set_concentration(action_line):
    """Parse setConcentration("species_name", value) -> (name, value) or None."""
    match = re.match(
        r'\s*setConcentration\s*\(\s*["\']([^"\']+)["\']\s*,\s*([\d.eE+\-]+)\s*\)',
        action_line,
    )
    if match:
        return match.group(1), float(match.group(2))
    return None


def _is_reset_concentrations(action_line):
    return bool(re.match(r'\s*resetConcentrations\s*\(', action_line))


def _is_reset_parameters(action_line):
    return bool(re.match(r'\s*resetParameters\s*\(', action_line))


def _is_save_concentrations(action_line):
    return bool(re.match(r'\s*saveConcentrations\s*\(', action_line))


def _is_save_parameters(action_line):
    return bool(re.match(r'\s*saveParameters\s*\(', action_line))


def _normalize_action_method(method, poplevel_text=None):
    """Normalize BNGL methods to the bngsim simulator API."""
    lower = method.strip().lower()

    poplevel = None
    if poplevel_text is not None:
        try:
            poplevel = float(poplevel_text)
        except (TypeError, ValueError):
            poplevel = None

    if lower == 'pla':
        raise ValueError(
            "method=>'pla' is not supported by the bngsim bridge"
        )

    if lower == 'psa':
        if poplevel is None or poplevel <= 1.0:
            poplevel = 100.0
        return 'psa', poplevel

    if lower == 'ssa' and poplevel is not None and poplevel > 1.0:
        return 'psa', poplevel

    return lower, None


def actions_compatible_with_bngsim(actions):
    """
    Return True if all simulation actions are compatible with BngsimModel.

    Unsupported simulation methods should stay on the subprocess NetModel path.
    """
    for action_line in actions:
        line = _collapse_action_line_continuations(action_line).strip()
        if not line or line.startswith('#'):
            continue

        sim_params = _parse_simulate_action(line)
        if sim_params is not None:
            method = sim_params.get('method', 'ode').strip().lower()
            if method not in ('ode', 'ssa', 'psa') or method == 'pla':
                return False
            continue

        ps_params = _parse_parameter_scan_action(line)
        if ps_params is not None:
            method = ps_params.get('method', 'ode').strip().lower()
            if method not in ('ode', 'ssa', 'psa') or method == 'pla':
                return False

    return True


def _resolve_scan_points(ps_params):
    """Build the parameter-scan point array from explicit values or min/max specs."""
    par_scan_vals = ps_params.get('par_scan_vals')
    if par_scan_vals is not None:
        if isinstance(par_scan_vals, (list, tuple, np.ndarray)):
            raw_points = par_scan_vals
        else:
            raw_points = [par_scan_vals]
        return np.asarray([float(value) for value in raw_points], dtype=float)

    par_min = float(ps_params.get('par_min', 0))
    par_max = float(ps_params.get('par_max', 1))
    n_scan_pts = int(ps_params.get('n_scan_pts', 10))
    log_scale = int(ps_params.get('log_scale', 0))

    if log_scale:
        return np.logspace(np.log10(par_min), np.log10(par_max), n_scan_pts)
    return np.linspace(par_min, par_max, n_scan_pts)


def _build_safe_eval_namespace(seed=None):
    """Build a safe expression-evaluation namespace for .net math."""
    ns = {
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        'log2': math.log2,
        'sqrt': math.sqrt,
        'abs': abs,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'atan2': math.atan2,
        'pi': math.pi,
        'e': math.e,
        'ceil': math.ceil,
        'floor': math.floor,
        'min': min,
        'max': max,
        'pow': pow,
        'if': lambda cond, t, f: t if cond else f,
        'rint': round,
        '__builtins__': {},
    }
    if seed:
        ns.update(seed)
    return ns


def _parse_net_species_initializers(net_lines):
    """Extract (species_name, initial_expr) pairs from a BNG .net file."""
    initializers = []
    in_block = False

    for raw_line in net_lines:
        stripped = raw_line.strip()
        if re.match(r'begin\s+species', stripped):
            in_block = True
            continue
        if re.match(r'end\s+species', stripped):
            break
        if not in_block:
            continue

        line = raw_line.split('#', 1)[0].strip()
        if not line:
            continue

        match = re.match(r'\s*\d+\s+(\S+)\s+(.+?)\s*$', line)
        if match:
            initializers.append((match.group(1), match.group(2).strip()))

    return initializers


class BngsimModel(NetModel):
    """In-process simulation model using the optional bngsim engine."""

    def __init__(self, name, acts, suffs, mutants, ls=None, nf=None, source_dir=None):
        super(BngsimModel, self).__init__(
            name,
            acts,
            suffs,
            mutants,
            ls=ls,
            nf=nf,
            source_dir=source_dir,
        )
        if not BNGSIM_AVAILABLE:
            raise RuntimeError('bngsim is not available')

        self._net_species_initializers = _parse_net_species_initializers(
            self.netfile_lines
        )
        if nf is not None:
            self._net_path = nf
            self._engine_model = bngsim.Model.from_net(nf)
        elif ls is not None:
            raise ValueError('BngsimModel requires nf so the .net path is stable')
        else:
            raise ValueError('Must provide nf')

    def copy_with_param_set(self, pset):
        """Return a shallow copy with a cloned engine model and new PSet."""
        newmodel = copy.copy(self)
        newmodel._engine_model = self._engine_model.clone()
        newmodel.param_set = pset
        return newmodel

    def execute(self, folder, filename, timeout, with_mutants=True):
        """Execute all simulation actions in-process using bngsim."""
        model = self._engine_model

        if self.param_set is not None:
            for pname in self.param_set.keys():
                try:
                    model.set_param(pname, self.param_set[pname])
                except Exception:
                    pass

        model.reset()
        ds = self._execute_actions(model)

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

    def _execute_actions(self, model):
        """Interpret and execute action lines using bngsim."""
        ds = {}
        sim = bngsim.Simulator(model, method='ode')
        current_method = 'ode'
        current_poplevel = None

        base_params = {}
        for pname in model.param_names:
            try:
                base_params[pname] = model.get_param(pname)
            except Exception:
                pass

        for action_line in self.actions:
            line = _collapse_action_line_continuations(action_line).strip()
            if not line or line.startswith('#'):
                continue

            sim_params = _parse_simulate_action(line)
            if sim_params is not None:
                method, poplevel = _normalize_action_method(
                    sim_params.get('method', 'ode'),
                    sim_params.get('poplevel'),
                )
                t_start = float(sim_params.get('t_start', 0))
                t_end = float(sim_params.get('t_end', 100))
                n_steps = int(sim_params.get('n_steps', 100))
                suffix = sim_params.get('suffix', 'time_course')

                if method == 'psa':
                    if current_method != 'psa' or current_poplevel != poplevel:
                        sim = bngsim.Simulator(
                            model,
                            method='psa',
                            poplevel=poplevel,
                        )
                        current_method = 'psa'
                        current_poplevel = poplevel
                elif current_method != method:
                    sim = bngsim.Simulator(model, method=method)
                    current_method = method
                    current_poplevel = None

                result = sim.run(
                    t_span=(t_start, t_end),
                    n_points=n_steps + 1,
                )
                ds[suffix] = self._result_to_data(result)
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
                continue

            if _is_reset_parameters(line):
                for pname, pval in base_params.items():
                    try:
                        model.set_param(pname, pval)
                    except Exception:
                        pass
                continue

            if _is_save_concentrations(line):
                model.save_concentrations()
                continue

            if _is_save_parameters(line):
                for pname in model.param_names:
                    try:
                        base_params[pname] = model.get_param(pname)
                    except Exception:
                        pass
                continue

            sc = _parse_set_concentration(line)
            if sc is not None:
                species_name, conc_value = sc
                try:
                    model.set_concentration(species_name, conc_value)
                except Exception:
                    logger.warning(
                        "setConcentration(%s, %s) failed - species not found",
                        species_name,
                        conc_value,
                    )
                continue

            ps_params = _parse_parameter_scan_action(line)
            if ps_params is not None:
                ds.update(self._run_parameter_scan(model, ps_params))
                continue

            if line and not re.match(r'\s*(begin|end)\s+actions', line):
                logger.debug("BngsimModel: skipping unknown action: %s", line)

        return ds

    def _make_scan_simulator(self, model, method, poplevel):
        """Construct a fresh simulator for one parameter-scan point."""
        if method == 'psa':
            return bngsim.Simulator(
                model,
                method='psa',
                poplevel=poplevel,
            )
        return bngsim.Simulator(model, method=method)

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

    def _prepare_scan_point_model(self, model, param_name, value):
        """Clone the base model, apply the scan parameter, and refresh initials."""
        point_model = model.clone()
        if param_name:
            point_model.set_param(param_name, float(value))
        self._sync_species_initial_concentrations(point_model)
        point_model.reset()
        return point_model

    def _run_parameter_scan(self, model, ps_params):
        """Execute a parameter_scan() action."""
        param_name = ps_params.get('parameter', '')
        t_start = float(ps_params.get('t_start', 0))
        t_end = float(ps_params.get('t_end', 100))
        suffix = ps_params.get('suffix', 'param_scan')
        use_ss = int(ps_params.get('steady_state', 0))
        method, poplevel = _normalize_action_method(
            ps_params.get('method', 'ode'),
            ps_params.get('poplevel'),
        )

        if use_ss and method != 'ode':
            logger.warning(
                "BngsimModel: steady_state=>1 is only supported for ODE parameter scans. "
                "Falling back to time-course scan for method=%s.",
                method,
            )
            use_ss = 0

        points = _resolve_scan_points(ps_params)
        obs_names = []
        expr_names = []
        rows = []

        if use_ss:
            for value in points:
                point_model = self._prepare_scan_point_model(
                    model,
                    param_name,
                    value,
                )
                point_sim = self._make_scan_simulator(
                    point_model,
                    method,
                    poplevel,
                )
                ss_result = point_sim.steady_state()
                for i, name in enumerate(ss_result.species_names):
                    point_model.set_concentration(
                        name,
                        ss_result.concentrations[i],
                    )
                point_model.save_concentrations()
                point_model.reset()
                eval_sim = self._make_scan_simulator(point_model, 'ode', None)
                result = eval_sim.run(t_span=(0, 1e-10), n_points=2)
                row, row_obs, row_expr = self._scan_result_to_row(result, value)
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
                )
                point_sim = self._make_scan_simulator(
                    point_model,
                    method,
                    poplevel,
                )
                result = point_sim.run(t_span=(t_start, t_end), n_points=2)
                row, row_obs, row_expr = self._scan_result_to_row(result, value)
                if len(obs_names) == 0:
                    obs_names = row_obs
                    expr_names = row_expr
                rows.append(row)

        if rows:
            arr = np.vstack(rows)
        else:
            arr = np.zeros((0, 1))

        data = Data(arr=arr)
        headers = [param_name] + obs_names + expr_names
        data.cols = {h: i for i, h in enumerate(headers)}
        data.headers = {i: h for i, h in enumerate(headers)}
        data.indvar = param_name
        return {suffix: data}

    @staticmethod
    def _scan_result_to_row(result, scan_value):
        """Convert the final point of a scan result into one row plus headers."""
        obs_names = list(result.observable_names)
        obs_array = np.asarray(result.observables)
        if obs_array.ndim == 2 and obs_array.shape[0] > 0:
            final_obs = obs_array[-1, :]
        else:
            final_obs = np.array([])

        core = result._core
        expr_names = list(getattr(core, 'expression_names', []))
        expr_array = np.asarray(getattr(core, 'expression_data', np.zeros((0, 0))))
        if expr_array.ndim == 2 and expr_array.shape[0] > 0 and expr_array.shape[1] > 0:
            final_expr = expr_array[-1, :]
        else:
            final_expr = np.array([])

        row = np.concatenate((
            np.array([scan_value], dtype=float),
            np.asarray(final_obs, dtype=float),
            np.asarray(final_expr, dtype=float),
        ))
        return row, obs_names, expr_names

    @staticmethod
    def _result_to_data(result):
        """Convert a bngsim Result to a PyBNF Data object."""
        obs_names = list(result.observable_names)
        n_times = result.n_times
        n_obs = result.n_observables

        core = result._core
        expr_names = list(getattr(core, 'expression_names', []))
        expr_array = np.asarray(getattr(core, 'expression_data', np.zeros((n_times, 0))))
        n_expr = len(expr_names)

        arr = np.zeros((n_times, 1 + n_obs + n_expr))
        arr[:, 0] = result.time
        obs_array = np.asarray(result.observables)
        arr[:, 1:1 + n_obs] = obs_array
        if n_expr > 0 and expr_array.size > 0:
            arr[:, 1 + n_obs:] = expr_array

        data = Data(arr=arr)
        headers = ['time'] + obs_names + expr_names
        data.cols = {h: i for i, h in enumerate(headers)}
        data.headers = {i: h for i, h in enumerate(headers)}
        data.indvar = 'time'
        return data

    def _get_mutant_model_bngsim(self, mut):
        """Create a mutant copy using a cloned engine model."""
        params = {p.name: p.value for p in self.param_set}
        for mi in mut:
            params[mi.name] = mi.mutate(params[mi.name])
        mut_param_list = [
            FreeParameter(
                pname,
                'uniform_var',
                -np.inf,
                np.inf,
                value=params[pname],
                bounded=True,
            )
            for pname in params
        ]
        mut_pset = PSet(mut_param_list)

        mut_model = copy.copy(self)
        mut_model._engine_model = self._engine_model.clone()
        mut_model.param_set = mut_pset
        return mut_model

    def __getstate__(self):
        """Support pickling for Dask workers by dropping the C++ model object."""
        state = self.__dict__.copy()
        state.pop('_engine_model', None)
        return state

    def __setstate__(self, state):
        """Restore from pickle by re-loading the .net file."""
        self.__dict__.update(state)
        if hasattr(self, '_net_path') and self._net_path:
            self._engine_model = bngsim.Model.from_net(self._net_path)
        else:
            raise RuntimeError("Cannot unpickle BngsimModel: no _net_path")

    def save(self, file_prefix, **kwargs):
        """Still write debug/export files via the NetModel implementation."""
        super(BngsimModel, self).save(file_prefix)
