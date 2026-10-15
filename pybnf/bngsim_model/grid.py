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
* ``sample_times`` needs 3 or more points, as listed.

The times of a ``sample_times`` grid are ``run_network``'s (``Network3/src/run_network.cpp``,
lines 338-406), which BNG2.pl hands the sorted list, followed by the action's ``t_end`` when it
gives one (:func:`sample_time_rows`). The run starts at ``t_start`` (0 unless the action says
otherwise, or ``continue=>1`` starts it on the model's clock) and writes that row; then each
listed time after ``t_start`` and before the end; then the end, which is the action's ``t_end``
when it gives one, below the last listed time or above it, and the last listed time otherwise.
So a listed time at or before ``t_start``, or at or after ``t_end``, is not written, and a time
listed twice is written twice. Each time is compared as BNG2.pl hands it to run_network, with
15 significant digits (:func:`bng2pl_time`), so ``0.1*3`` and ``0.3`` are one time. BNG2.pl then
reads the model's species and clock back from the last row the run wrote (BNGAction.pm
757-779), and so does the bridge (``BngsimModel._settle_after_run``): the end of the grid, or,
for a run that stops early on ``stop_if`` or at ``steady_state``, the row it stopped at (#1010).
A ``continue=>1`` action starts there. Edition-2 experiments list t = 0 first, with
``t_start=>0`` and no ``t_end``, so their rows are the listed times. This was lanl/PyBNF#872: the
bridge integrated from the first listed time, reporting the initial state there, and ignored a
``t_end`` below the last one. When the configuration loads, a datum at a time such a grid does
not write is refused (:func:`literal_sample_time_rows`), since no simulation could score it.

On NFsim, BNG2.pl reads ``n_steps`` and nothing else for the grid (``n_output_steps`` is not
read, and there is no default step count), requires ``t_end`` greater than ``t_start``, and
refuses ``sample_times`` and ``continue``. The bridge refuses ``continue`` too: its network-free
runs keep no clock for a continued run to start on, and it used to ignore the option. It keeps
one extension: bngsim's NFsim sessions take ``sample_times``, which the edition-2 experiments
write (#427), so an NF action with two or more ``sample_times`` and no ``n_steps`` runs on them,
at the times a network method's would have.

A ``parameter_scan`` / ``bifurcate`` runs ``simulate`` at each point in BNG2.pl, so it needs
``t_end`` without ``sample_times`` and may not name both step keys. Its output is one row per
scan point, the state at the end of that point's grid, whatever its step count, so the bridge's
step count for a scan is unchanged. With ``sample_times``, that end is the one above, reached
from ``t_start``.

Two ``sample_times`` grids are refused on the bridge, because no grid of it writes what BNG2.pl
does: a negative ``t_start`` (run_network stops: "Negative sample times are not allowed."), and
an end that is not after ``t_start`` (BNG2.pl integrates nothing: it writes the one row at
``t_start``, or, continuing a run, no row, and then stops; the bridge's runs take two or more
times).

A ``continue=>1`` action writes no row at its start in BNG2.pl; the bridge writes it, and
scoring, which matches by time, reads only the rows BNG2.pl also writes.

Some options BNG2.pl reads decide an action's rows in a way these rules cannot see
(:data:`UNREAD_GRID_OPTIONS`): an ``argfile`` supplies ``t_end``, ``n_steps`` or
``sample_times`` from a file, ``max_sim_steps``, ``output_step_interval`` and, beside
``stop_if``, ``print_on_stop`` other than 1 stop the run early, add rows, or drop the last one,
and on a stochastic method run_network checks ``stop_if`` after every reaction, so it stops
between output times, where the bridge checks it at the output times only. The bridge reads
none of them as BNG2.pl does, so an action naming one is not the bridge's, whether it is in the
actions block or in a protocol block: ``classify_actions_for_bngsim`` reports the model
unsupported, ``bngl_backend = auto`` runs it on BNG2.pl, and ``bngl_backend = bngsim`` refuses
it, naming the option. (``print_on_stop`` without ``stop_if`` changes nothing in BNG2.pl, which
reads it only to pass beside the stop condition.)

Every refusal is a :class:`~pybnf.printing.PybnfError` naming the model and the action. The
same functions run when the bridge is chosen for a model (:func:`check_output_grids`, as the
fit sets up its models, before any simulation) and when an action runs, so the two cannot
disagree. When the models are set up, a value that is an expression, or a ``t_start`` taken
from the clock by ``continue=>1``, is not known yet; such an action's times are checked when
it runs.
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


logger = logging.getLogger(__name__)

#: The ``method`` tokens BNG2.pl runs through ``simulate_nf`` (bngsim's NF aliases included).
NETWORK_FREE_METHODS = frozenset(('nf', 'nfsim', 'nf_reject', 'nf_exact', 'rm', 'rulemonkey'))

#: Options BNG2.pl reads that change which rows a ``simulate`` / ``parameter_scan`` writes, and
#: that the bridge does not read as BNG2.pl does (the module docstring says how each does it).
#: ``print_on_stop`` is one only beside ``stop_if`` and when it is not 1, its default: BNG2.pl
#: passes it to run_network only with the stop condition (BNGAction.pm 424-432). ``stop_if`` is
#: one only on a stochastic method (:data:`STOCHASTIC_NETWORK_METHODS`, or ``poplevel``):
#: run_network checks it there after every reaction, so the run stops, and writes its last row,
#: between output times, where the bridge checks it only at the output times (#1010).
UNREAD_GRID_OPTIONS = ('argfile', 'max_sim_steps', 'output_step_interval', 'print_on_stop',
                       'stop_if')

#: The stochastic network methods: run_network's Gillespie loops (ssa, and psa, which an ssa
#: with ``poplevel`` is) check a stop condition after each reaction (network.cpp 6241-6245).
STOCHASTIC_NETWORK_METHODS = frozenset(('ssa', 'psa', 'pla'))


def _unread(option, params):
    if option == 'print_on_stop':
        return 'stop_if' in params and str(params[option]).strip() != '1'
    if option == 'stop_if':
        method = str(params.get('method', 'ode')).strip().lower()
        return method in STOCHASTIC_NETWORK_METHODS or 'poplevel' in params
    return True


def unread_grid_options(params):
    """The :data:`UNREAD_GRID_OPTIONS` a parsed action names, in that order."""
    return [option for option in UNREAD_GRID_OPTIONS if option in params
            and _unread(option, params)]


def unread_options_refusal(where, options):
    """The error for an action that names an option the bridge does not read (as BNG2.pl does).
    Not BNG2.pl's refusal: BNG2.pl reads it and runs the action."""
    stop = ('stop_if' in options)
    return PybnfError(
        '%s: names %s, which the bngsim bridge does not read%s'
        % (where, ', '.join(options), ' as BNG2.pl does' if stop else ''),
        "%s names %s. BNG2.pl reads %s and runs the action, but the bngsim bridge does not%s, so "
        "it cannot write the rows BNG2.pl writes for it.%s Run the model with bngl_backend = auto "
        "(which picks BNG2.pl for it) or bngl_backend = bionetgen, or write the action's "
        "t_end and step count into the action itself."
        % (where, ', '.join(options), 'it' if len(options) == 1 else 'them',
           ' read %s as BNG2.pl does' % ('it' if len(options) == 1 else 'them') if stop else '',
           " On a stochastic method BNG2.pl checks stop_if after every reaction, so it stops, "
           "writes its last row and leaves its state between output times; the bridge checks "
           "it only at the output times." if stop else ''))


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


def bng2pl_time(value):
    """``value`` as BNG2.pl hands a time to run_network: Perl writes a number on the command line
    with 15 significant digits, so ``0.1*3`` reaches run_network as 0.3, and two times that agree
    to 15 digits are the same time there."""
    return float('%.15g' % value)


def sample_time_rows(where, listed, t_start, t_end=None, continued=False, note=''):
    """The times of the rows BNG2.pl writes for an action whose grid is ``sample_times``.

    ``listed`` is the action's ``sample_times`` as numbers, sorted; ``t_start`` is where the
    run starts; ``t_end`` is the action's own ``t_end``, or ``None`` when it gives none;
    ``continued`` says the action sets ``continue=>1``. BNG2.pl hands run_network the sorted
    list followed by ``t_end`` (BNGAction.pm 532-546), each time with 15 significant digits
    (:func:`bng2pl_time`), and run_network writes ``t_start``, then each listed time strictly
    after ``t_start`` and before the end, then the end: ``t_end`` when the action gives one,
    whether it is below the last listed time or above it, else the last listed time
    (run_network.cpp 338-406). A listed time repeated is written each time. The last time
    returned is where the model's clock stands after the run, unless it stops early.

    Refused, naming ``where`` (with ``note`` after the explanation): a negative ``t_start``,
    which run_network stops on, and an end that is not after ``t_start``. For the latter BNG2.pl
    integrates nothing: it writes the one row at ``t_start`` (the bridge, whose runs need a time
    after their start, refuses instead), or, for a continued run, no row at all, and then stops
    because the run left it no state to read back."""
    t_start = bng2pl_time(t_start)
    listed = [bng2pl_time(t) for t in listed]
    end = listed[-1] if t_end is None else bng2pl_time(t_end)
    if t_start < 0:
        raise grid_refusal(where, 'Negative sample times are not allowed.',
                           ' run_network stops with that message: this action starts its '
                           'sample_times grid at t_start = %g.%s' % (t_start, note),
                           bng2pl_says_so=False)
    if not end > t_start:
        bng2pl = ("BNG2.pl writes no row for such a continued run, and then stops (\"Number of "
                  "species in model ... and CDAT file ... differ\"), since the run left it no "
                  "state to read back" if continued else
                  "BNG2.pl integrates nothing for such an action and writes the one row at "
                  "t_start. The bngsim bridge, whose runs need a time after their start, refuses "
                  "it instead")
        raise PybnfError(
            '%s: its sample_times grid ends at t = %g, which is not after t_start = %g'
            % (where, end, t_start),
            "%s cannot run on the bngsim bridge: its sample_times grid starts at t_start = %g and "
            "ends at t = %g (the action's t_end when it gives one, else its last sample time). "
            "%s. Give the action sample times, or a t_end, after t_start.%s"
            % (where, t_start, end, bng2pl, note))
    return [t_start] + [t for t in listed if t_start < t < end] + [end]


def _listed_sample_times(params, evaluate):
    """An action's ``sample_times`` as numbers, in BNG2.pl's numeric sort."""
    return sorted(evaluate(value) for value in params['sample_times'])


def _resumes(params):
    """Whether ``continue`` is set on an action; ``None`` when it is an expression."""
    resume = params.get('continue')
    if resume is None or str(resume).strip() in ('', '0'):
        return False
    number = _number(resume)
    return None if number is None else number != 0


def _check_literal_sample_times(params, where):
    """At set-up, before any simulation: refuse the ``sample_times`` grid of an action whose
    times are all numbers as written (:func:`sample_time_rows`). An expression, or a
    ``t_start`` that ``continue=>1`` takes from the model's clock, is known only when the
    action runs, and is checked then."""
    if 't_start' not in params and _resumes(params) is not False:
        return
    listed = [_number(value) for value in params['sample_times']]
    t_start = _number(params.get('t_start', 0))
    t_end = _number(params['t_end']) if 't_end' in params else None
    if None in listed or t_start is None or ('t_end' in params and t_end is None):
        return
    sample_time_rows(where, sorted(listed), t_start, t_end, continued=bool(_resumes(params)))


def literal_sample_time_rows(params):
    """The times of the rows a ``simulate`` writes for its ``sample_times``, when they are known
    before anything runs, on BNG2.pl and on the bridge alike (:func:`sample_time_rows`); else
    ``None``. They are not known when the grid is a step count, when a time is an expression,
    when ``continue=>1`` takes ``t_start`` from the model's clock, or when the run can stop
    early (``stop_if``, ``steady_state``); nor when the grid is refused (the set-up and
    BNG2.pl say why)."""
    if 'sample_times' not in params or 'n_steps' in params or 'n_output_steps' in params:
        return None
    raw = params['sample_times']
    method = str(params.get('method', 'ode')).strip().lower()
    if not isinstance(raw, list) or len(raw) < (2 if method in NETWORK_FREE_METHODS else 3):
        return None
    if 'stop_if' in params or _number(params.get('steady_state', 0)) != 0:
        return None
    resumes = _resumes(params)
    if resumes is None or (resumes and 't_start' not in params):
        return None
    listed = [_number(value) for value in raw]
    t_start = _number(params.get('t_start', 0))
    t_end = _number(params['t_end']) if 't_end' in params else None
    if None in listed or t_start is None or ('t_end' in params and t_end is None):
        return None
    try:
        return sample_time_rows('', sorted(listed), t_start, t_end, continued=resumes)
    except PybnfError:
        return None


def simulate_output_grid(sim_params, where, method, t_start=None, evaluate=None):
    """``(sample_times, t_end, n_steps)`` of one ``simulate`` action, or ``None`` when no
    ``evaluate`` is given (a check of which keys are present, and of the times written as
    numbers, as when the models are set up).

    ``sample_times`` is ``None``, or the times of the action's rows by BNG2.pl's rules
    (:func:`sample_time_rows`), ``t_start`` first; with it, ``t_end`` is its last time and
    ``n_steps`` is ``None``. ``evaluate`` turns an action value into a number, as the caller's
    path always has (``_eval_numeric`` on the ``.net`` bridge, ``float`` on the NFsim one);
    ``t_start`` is the run's start, required with ``evaluate``."""
    unread = unread_grid_options(sim_params)
    if unread:
        raise unread_options_refusal(where, unread)
    has_steps = 'n_steps' in sim_params
    has_output_steps = 'n_output_steps' in sim_params
    has_sample_times = 'sample_times' in sim_params
    if method in NETWORK_FREE_METHODS:
        if _resumes(sim_params) is not False:
            # BNG2.pl refuses it (simulate_nf), and the network-free sessions keep no clock for
            # it: the bridge ignored it, labelling a continued run's rows from t_start.
            raise grid_refusal(where, "NFsim does not support 'continue' option.",
                               " The bngsim bridge's network-free runs do not read continue "
                               "either; they label a run's rows from its t_start.")
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
            raw = sim_params['sample_times']
            if not isinstance(raw, list) or len(raw) < 2:
                raise grid_refusal(where, "'sample_times' array must contain 2 or more points.",
                               " BNG2.pl's NFsim takes no sample_times at all; bngsim's takes two "
                               "or more.", bng2pl_says_so=False)
            return _sample_time_grid(sim_params, where, t_start, evaluate)
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
    return _sample_time_grid(sim_params, where, t_start, evaluate)


def _sample_time_grid(params, where, t_start, evaluate):
    """:func:`simulate_output_grid` for an action whose grid is its ``sample_times``."""
    if evaluate is None:
        _check_literal_sample_times(params, where)
        return None
    t_end = evaluate(params['t_end']) if 't_end' in params else None
    continued = bool(_resumes(params))
    note = (" Its t_start is the model's clock, where the previous simulate ended, because the "
            "action sets continue=>1." if continued and 't_start' not in params else '')
    rows = sample_time_rows(where, _listed_sample_times(params, evaluate), t_start, t_end,
                            continued=continued, note=note)
    return rows, rows[-1], None


def scan_output_grid(ps_params, where, t_start=None, evaluate=None):
    """``(sample_times, t_end)`` of each point of a ``parameter_scan`` / ``bifurcate``:
    BNG2.pl runs ``simulate`` at each point with the scan's own options, which requires
    ``t_end`` unless ``sample_times`` gives the grid, and refuses both step keys. The point's
    row is its state at ``t_end``, reached from ``t_start``.

    ``sample_times`` is ``None``, or the point's times by :func:`sample_time_rows`
    (``t_start`` first, and ``t_end`` the last of them). Both are ``None`` without
    ``evaluate`` (a check at set-up), and for a ``method=>"protocol"`` scan, which runs the
    protocol block at each point instead (``simulate_protocol``), whose own ``simulate``
    actions carry their grids. ``t_start`` is the points' start, required with ``evaluate``."""
    method = str(ps_params.get('method', 'ode')).strip().lower()
    unread = unread_grid_options(ps_params)
    if unread:
        raise unread_options_refusal(where, unread)
    if method == 'protocol':
        return None, None
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
        grid = _sample_time_grid(ps_params, where, t_start, evaluate)
        return (None, None) if grid is None else grid[:2]
    if 't_end' not in ps_params:
        raise grid_refusal(where, 'Parameter t_end must be defined.')
    if evaluate is None:
        return None, None
    return None, evaluate(ps_params['t_end'])


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
    the set-up treats as a reason to fall back to BNG2.pl, which refuses the same actions, or,
    for a ``sample_times`` grid that ends where it starts, integrates nothing."""
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
                scan_output_grid(ps_params, describe_action(model_name, ps_params, kind))
                break
