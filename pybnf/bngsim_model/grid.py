"""The output grid of a BNGL ``simulate`` action on the bngsim bridge, by BNG2.pl's rules.

BNG2.pl decides how many rows a ``simulate`` writes, and at which times, in
``Perl2/BNGAction.pm``: ``simulate`` (lines 489-550) for the network methods (ode, ssa,
pla, psa) and ``simulate_nf`` (lines 912-937) for NFsim. The bridge used to read
``n_steps`` and ``t_end`` with defaults of its own (100 and 100), so an action that names
``n_output_steps``, or no step count, ran on a 101-row grid where BNG2.pl writes 21 rows,
or 2, and an action without ``t_end``, which BNG2.pl refuses, ran to t = 100. A data time on
one backend's grid could be missing from the other's (#990 review).

The rules, for a ``simulate`` on a network method:

* ``n_output_steps`` is the same setting as ``n_steps``; naming both is an error.
* ``n_steps`` (or ``n_output_steps``) takes precedence over ``sample_times``, with a warning.
* With neither, and no ``sample_times``, there is one step.
* Without ``sample_times``, ``t_end`` is required.
* ``sample_times`` needs 3 or more points; a ``t_end`` beyond the last is appended.

On NFsim, BNG2.pl reads ``n_steps`` and nothing else for the grid (``n_output_steps`` is not
read, and there is no default step count), requires ``t_end`` greater than ``t_start``, and
refuses ``sample_times``. The bridge keeps one extension there: bngsim's NFsim sessions take
``sample_times``, which the edition-2 experiments write (#427), so an NF action with
``sample_times`` and no ``n_steps`` runs on those times.

A ``parameter_scan`` / ``bifurcate`` runs ``simulate`` at each point in BNG2.pl, so it needs
``t_end`` without ``sample_times`` and may not name both step keys. Its output is one row per
scan point whatever its step count, so the bridge's step count for a scan is unchanged.

Some options BNG2.pl reads decide an action's rows in a way these rules cannot see
(:data:`UNREAD_GRID_OPTIONS`): an ``argfile`` supplies ``t_end``, ``n_steps`` or
``sample_times`` from a file, and ``max_sim_steps``, ``output_step_interval`` and
``print_on_stop=>0`` stop the run early, add rows, or drop the last one. The bridge reads none
of them, so an action naming one is not the bridge's: ``classify_actions_for_bngsim`` reports the
model unsupported, ``bngl_backend = auto`` runs it on BNG2.pl, and ``bngl_backend = bngsim``
refuses it, naming the option.

One difference is not covered, and is lanl/PyBNF#872, open and older than these rules: an action
whose ``sample_times`` start after ``t_start`` is integrated from its first sample time on the
bridge, where BNG2.pl integrates from ``t_start`` and writes that row; and with a ``t_end`` before
its last sample time, BNG2.pl writes rows up to ``t_end`` and the bridge does not. So the grids
agree for every action these rules accept except one with such ``sample_times``.

Every refusal is a :class:`~pybnf.printing.PybnfError` naming the model and the action. The
same functions run when the bridge is chosen for a model (:func:`check_output_grids`, as the
fit sets up its models, before any simulation) and when an action runs, so the two cannot
disagree.
"""

import logging

from ..printing import PybnfError
from .parsing import (
    _collapse_action_line_continuations,
    _parse_action_dict,
    _parse_bifurcate_action,
    _parse_parameter_scan_action,
    _parse_simulate_action,
)
from .scan import _resolve_sample_times


logger = logging.getLogger(__name__)

#: The ``method`` tokens BNG2.pl runs through ``simulate_nf`` (bngsim's NF aliases included).
NETWORK_FREE_METHODS = frozenset(('nf', 'nfsim', 'nf_reject', 'nf_exact', 'rm', 'rulemonkey'))

#: Options BNG2.pl reads that change which rows a ``simulate`` / ``parameter_scan`` writes, and
#: that the bridge does not read (the module docstring says how each does it). ``print_on_stop``
#: is one only when it is not 1, its default.
UNREAD_GRID_OPTIONS = ('argfile', 'max_sim_steps', 'output_step_interval', 'print_on_stop')


def unread_grid_options(params):
    """The :data:`UNREAD_GRID_OPTIONS` a parsed action names, in that order."""
    return [option for option in UNREAD_GRID_OPTIONS if option in params and not (
        option == 'print_on_stop' and str(params[option]).strip() == '1')]


def unread_options_refusal(where, options):
    """The error for an action that names an option the bridge does not read. Not BNG2.pl's
    refusal: BNG2.pl reads it and runs the action."""
    return PybnfError(
        '%s: names %s, which the bngsim bridge does not read' % (where, ', '.join(options)),
        "%s names %s. BNG2.pl reads %s and runs the action, but the bngsim bridge does not, so "
        "it cannot write the rows BNG2.pl writes for it. Run the model with bngl_backend = auto "
        "(which picks BNG2.pl for it) or bngl_backend = bionetgen, or write the action's "
        "t_end and step count into the action itself."
        % (where, ', '.join(options), 'it' if len(options) == 1 else 'them'))


_BOTH_STEP_KEYS = ("'n_steps' and 'n_output_steps' are the same thing, cannot define both. "
                   "Please only define one ('n_output_steps' is preferred).")


def grid_refusal(where, message, note='', bng2pl_says_so=True):
    """The error for an action whose output grid BNG2.pl's rules do not define. ``message`` is
    BNG2.pl's own unless ``bng2pl_says_so`` is false (it has none for the case)."""
    source = ('BNG2.pl refuses the action with that message (BNGAction.pm)' if bng2pl_says_so
              else 'BNG2.pl cannot run the action either')
    return PybnfError(
        '%s: %s' % (where, message),
        "%s cannot run: %s %s, and the bngsim bridge follows BNG2.pl's rules for a "
        "simulation's output grid, so that BNG2.pl and bngsim write the same rows for the "
        "same action.%s" % (where, message, source, note))


def describe_action(model_name, params, kind='simulate'):
    """How a refusal names an action: its model, its kind and its suffix."""
    default = 'time_course' if kind == 'simulate' else 'param_scan'
    return 'Model %s, %s action with suffix "%s"' % (model_name, kind,
                                                     params.get('suffix', default))


def simulate_output_grid(sim_params, where, method, t_start=None, evaluate=None):
    """``(sample_times, t_end, n_steps)`` of one ``simulate`` action, or ``None`` when no
    ``evaluate`` is given (a check of which keys are present, as when the models are set up).

    ``sample_times`` is a sorted list of floats or ``None``; with it, ``t_end`` is its last
    point and ``n_steps`` is ``None``. ``evaluate`` turns an action value into a number, as
    the caller's path always has (``_eval_numeric`` on the ``.net`` bridge, ``float`` on the
    NFsim one); ``t_start`` is the run's start, needed only to check an NFsim ``t_end``."""
    unread = unread_grid_options(sim_params)
    if unread:
        raise unread_options_refusal(where, unread)
    has_steps = 'n_steps' in sim_params
    has_output_steps = 'n_output_steps' in sim_params
    has_sample_times = 'sample_times' in sim_params
    if method in NETWORK_FREE_METHODS:
        if has_steps:
            if has_sample_times:
                logger.warning('n_steps and sample_times both defined. n_steps takes precedence.')
            if 't_end' not in sim_params:
                raise grid_refusal(where, 'Parameter t_end must be defined')
            if evaluate is None:
                return None
            n_steps = int(evaluate(sim_params['n_steps']))
            t_end = evaluate(sim_params['t_end'])
            if n_steps < 1:
                raise grid_refusal(where, 'No simulation output requested: set n_steps>0')
            if not t_end > t_start:
                raise grid_refusal(where, 't_end must be greater than t_start.')
            return None, t_end, n_steps
        if has_sample_times and not has_output_steps:
            sample_times = _resolve_sample_times(sim_params)
            if sample_times is None:
                raise grid_refusal(where, "'sample_times' array must contain 2 or more points.",
                               " BNG2.pl's NFsim takes no sample_times at all; bngsim's takes two "
                               "or more.", bng2pl_says_so=False)
            return (sample_times, sample_times[-1], None) if evaluate is not None else None
        raise grid_refusal(where, 'No simulation output requested: set n_steps>0',
                       " NFsim's simulate in BNG2.pl reads n_steps, not n_output_steps."
                       if has_output_steps else '')
    if has_steps and has_output_steps:
        raise grid_refusal(where, _BOTH_STEP_KEYS)
    if has_steps or has_output_steps or not has_sample_times:
        if has_sample_times:
            key = 'n_steps' if has_steps else 'n_output_steps'
            logger.warning('%s and sample_times both defined. %s takes precedence.', key, key)
        if 't_end' not in sim_params:
            raise grid_refusal(where, 'Parameter t_end must be defined.')
        if evaluate is None:
            return None
        key = 'n_steps' if has_steps else 'n_output_steps' if has_output_steps else None
        n_steps = int(evaluate(sim_params[key])) if key else 1
        if n_steps < 1:
            raise grid_refusal(where, '%s must be at least 1, not %d.' % (key, n_steps),
                           bng2pl_says_so=False)
        return None, evaluate(sim_params['t_end']), n_steps
    raw = sim_params['sample_times']
    if not isinstance(raw, list) or len(raw) < 3:
        raise grid_refusal(where, "'sample_times' array must contain 3 or more points.")
    if evaluate is None:
        return None
    sample_times = _resolve_sample_times(sim_params)
    return sample_times, sample_times[-1], None


def scan_t_end(ps_params, where, sample_times, evaluate=None):
    """The ``t_end`` of a ``parameter_scan`` / ``bifurcate`` point (``None`` without
    ``evaluate``, or for a protocol scan): BNG2.pl runs ``simulate`` at each point, which
    requires ``t_end`` unless ``sample_times`` gives the grid and refuses both step keys. A
    ``method=>"protocol"`` scan runs the protocol block at each point instead
    (``simulate_protocol``), whose own ``simulate`` actions carry their grids."""
    method = str(ps_params.get('method', 'ode')).strip().lower()
    unread = unread_grid_options(ps_params)
    if unread:
        raise unread_options_refusal(where, unread)
    if method == 'protocol':
        return None
    if 'n_steps' in ps_params and 'n_output_steps' in ps_params:
        raise grid_refusal(where, _BOTH_STEP_KEYS)
    if 'sample_times' in ps_params and 'n_steps' not in ps_params \
            and 'n_output_steps' not in ps_params:
        # The points' grid: BNG2.pl's simulate needs 3 or more sample times; bngsim's NFsim,
        # which BNG2.pl's has no sample_times for, takes 2 or more.
        minimum = 2 if method in NETWORK_FREE_METHODS else 3
        raw = ps_params['sample_times']
        if not isinstance(raw, list) or len(raw) < minimum:
            raise grid_refusal(where, "'sample_times' array must contain %d or more points."
                               % minimum, bng2pl_says_so=minimum == 3)
    if sample_times is None and 't_end' not in ps_params:
        raise grid_refusal(where, 'Parameter t_end must be defined.')
    if evaluate is None or 't_end' not in ps_params:
        return None
    return evaluate(ps_params['t_end'])


#: The one network-free ``method`` BNG2.pl has: its ``simulate`` dispatches ``nf`` -- exactly
#: that, in lower case -- to ``simulate_nf`` (NFsim), and rejects every other spelling of a
#: network-free method in :data:`NETWORK_FREE_METHODS`, in any case, as "not a valid option":
#: RuleMonkey (``rm``, ``rulemonkey``, ``nf_exact``), bngsim's other spellings of NFsim
#: (``nfsim``, ``nf_reject``), and ``NF``.
BNG2PL_NETWORK_FREE_METHOD = 'nf'

#: Why BNG2.pl refuses each kind of network-free action: a message states each cause once.
BNG2PL_NF_GRID = ("BNG2.pl's NFsim takes its output grid only as t_end with n_steps of 1 or "
                  "more, never as sample_times")
BNG2PL_NF_METHOD = ("BNG2.pl's one network-free method is 'nf', in lower case: it has no "
                    "RuleMonkey, and none of bngsim's other spellings of NFsim")
BNG2PL_NF_CONTINUE = "BNG2.pl's NFsim cannot continue an earlier run"


def _statement(action_line):
    """An action line as BNG2.pl runs it: continuations joined, and a trailing comment and
    ``;`` dropped. BNGLModel keeps both on a hand-written action
    (``simulate({...});  # equilibrate``), which the bngsim parsers do not read."""
    line = _collapse_action_line_continuations(action_line)
    quote = None
    for i, ch in enumerate(line):
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch in ('"', "'"):
            quote = ch
        elif ch == '#':
            line = line[:i]
            break
    return line.strip().rstrip(';').rstrip()


def _number(value):
    """``value`` as a float, or ``None`` for an expression BNG2.pl evaluates in the model."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bng2pl_nfsim_refusal(params):
    """``(BNG2.pl's message, cause)`` when ``simulate_nf`` (BNGAction.pm 793-937) stops an NFsim
    action with ``params``, checked in its order, or ``None``. A value that is an expression is
    not judged; nor is an action whose ``argfile`` supplies its grid at run time."""
    if 'argfile' in params:
        return None
    resume = params.get('continue')
    if resume is not None and str(resume) not in ('', '0') and _number(resume) != 0:
        return "NFsim does not support 'continue' option.", BNG2PL_NF_CONTINUE
    if 'n_steps' in params:
        n_steps = _number(params['n_steps'])
        if n_steps is not None and n_steps < 1:
            return 'No simulation output requested: set n_steps>0', BNG2PL_NF_GRID
        if 't_end' not in params:
            return 'Parameter t_end must be defined', BNG2PL_NF_GRID
        # Without t_start, simulate_nf starts at the model's current time, which is >= 0.
        t_end, t_start = _number(params['t_end']), _number(params.get('t_start', 0))
        if t_end is not None and t_start is not None and not t_end > t_start:
            return 't_end must be greater than t_start.', BNG2PL_NF_GRID
        return None
    if 'sample_times' in params:
        return 'sample_times not supported in this version of NFsim', BNG2PL_NF_GRID
    return 'No simulation output requested: set n_steps>0', BNG2PL_NF_GRID


def bng2pl_network_free_refusals(action_lines):
    """``[(kind, suffix, BNG2.pl's message, cause)]`` for each network-free action among
    ``action_lines`` that BNG2.pl refuses to run (BNG2.pl 2.9.3, ``Perl2/BNGAction.pm``; #999).
    ``cause`` is one of :data:`BNG2PL_NF_GRID`, :data:`BNG2PL_NF_METHOD`,
    :data:`BNG2PL_NF_CONTINUE`.

    The actions read are ``simulate``, ``parameter_scan`` and ``bifurcate`` with a network-free
    ``method``, and ``simulate_nf``, each as BNG2.pl runs it (:func:`_statement`). Refused:

    * a network-free method BNG2.pl does not have (:data:`BNG2PL_NETWORK_FREE_METHOD`);
    * an NFsim action ``simulate_nf`` stops (:func:`_bng2pl_nfsim_refusal`): ``continue``; no
      ``n_steps`` -- with ``sample_times`` instead ("sample_times not supported in this version
      of NFsim") or with neither; ``n_steps`` below 1; no ``t_end``; ``t_end`` not after
      ``t_start``. A ``parameter_scan`` / ``bifurcate`` runs ``simulate`` at each point with
      its own options, so the same holds for each point. The edition-2 experiments write
      ``sample_times`` for a time course and a scan with no ``n_steps``, which bngsim's NFsim
      runs and BNG2.pl's does not.

    Every other action, network-free or not, is left to :func:`check_output_grids` and BNG2.pl.
    Pure: reads only the action text, so the configuration can call it before any model is
    set up."""
    found = []
    for action_line in action_lines:
        line = _statement(action_line)
        if not line:
            continue
        nfsim = _parse_action_dict(line, 'simulate_nf')
        if nfsim is not None:
            refusal = _bng2pl_nfsim_refusal(nfsim)
            if refusal is not None:
                found.append(('simulate_nf', nfsim.get('suffix', 'time_course')) + refusal)
            continue
        for kind, parse in (('simulate', _parse_simulate_action),
                            ('parameter_scan', _parse_parameter_scan_action),
                            ('bifurcate', _parse_bifurcate_action)):
            params = parse(line)
            if params is None:
                continue
            suffix = params.get('suffix', 'time_course' if kind == 'simulate' else 'param_scan')
            # BNG2.pl compares the method as written (case and all); 'poplevel' makes it psa.
            method = str(params.get('method', ''))
            if 'poplevel' in params:
                refusal = None
            elif method == BNG2PL_NETWORK_FREE_METHOD:
                refusal = _bng2pl_nfsim_refusal(params)
            elif method.strip().lower() in NETWORK_FREE_METHODS:
                refusal = ("Simulation method '%s' is not a valid option." % method,
                           BNG2PL_NF_METHOD)
            else:
                refusal = None
            if refusal is not None:
                found.append((kind, suffix) + refusal)
            break
    return found


def check_output_grids(model_name, action_lines):
    """Refuse, before anything runs, every ``simulate`` / ``parameter_scan`` / ``bifurcate``
    among ``action_lines`` whose output grid BNG2.pl's rules do not define (the module
    docstring lists them). Called when the bngsim bridge is chosen for a model, so the refusal
    comes as the fit sets up its models rather than at the first simulation. It is not one
    the set-up treats as a reason to fall back to BNG2.pl, which refuses the same actions."""
    for action_line in action_lines:
        line = _collapse_action_line_continuations(action_line).strip()
        if not line or line.startswith('#'):
            continue
        sim_params = _parse_simulate_action(line)
        if sim_params is not None:
            method = str(sim_params.get('method', 'ode')).strip().lower()
            simulate_output_grid(sim_params, describe_action(model_name, sim_params), method)
            continue
        for kind, parse in (('parameter_scan', _parse_parameter_scan_action),
                            ('bifurcate', _parse_bifurcate_action)):
            ps_params = parse(line)
            if ps_params is not None:
                scan_t_end(ps_params, describe_action(model_name, ps_params, kind),
                           _resolve_sample_times(ps_params))
                break
