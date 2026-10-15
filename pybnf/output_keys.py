"""Which simulation produced an output key: the exact ``(action, condition)`` pair (#850).

A model runs each of its actions once as written and once under each condition (an edition-2
``condition:``, an edition-1 mutant), and names each output by plain concatenation: the action's
suffix followed by the condition's (nothing for the model as written). ``Model.execute`` keys its
result dict that way, and configuration keys each experiment's data the same way.

Reading a key back is therefore a question about the pairs the model has, never about the
key's spelling. The key ``data`` is the action ``data`` as written, or the action ``dat`` under a
condition ``a``, or both, depending only on which actions and conditions exist. A test such as
"the key ends with the condition's suffix" answers it wrongly: it reads the wildtype experiment
``data`` as ``dat`` under ``a`` whenever a condition ``a`` exists, and routes the wrong
condition's chain-rule factors onto its gradient.

So every consumer resolves a key here, from the pairs, and two simulations that spell one key a
consumer reads are refused when the configuration loads (:func:`refuse_ambiguous_output_keys`),
because no lookup can tell them apart and ``execute`` writes one over the other. The simulations
counted include the outputs an action writes without the model registering them -- a
pre-equilibration's unmeasured phase (``<name>_preequil``) -- since ``execute`` stores those under
their own suffix too.
"""

from .printing import PybnfError

__all__ = ['simulation_conditions', 'output_pairs', 'resolve_output_key', 'action_outputs',
           'unregistered_outputs', 'refuse_ambiguous_output_keys',
           'refuse_duplicate_action_suffixes']


def _is_model_as_written(mutant):
    """The SBML backends keep the model as written in their mutant list as an empty
    ``MutationSet``; it is the wildtype, not a condition."""
    return getattr(mutant, 'suffix', '') == '' and not list(getattr(mutant, 'mutations', ()) or ())


def simulation_conditions(model):
    """``[None, condition, ...]``: the model as written (``None``) and every condition or mutant
    it runs, each a :class:`~pybnf.pset.MutationSet`."""
    return [None] + [m for m in (getattr(model, 'mutants', None) or [])
                     if not _is_model_as_written(m)]


def _registered_suffix(entry):
    """A ``model.suffixes`` entry is ``(action kind, suffix)``; RoadRunner's initial list holds
    bare suffixes."""
    return entry if isinstance(entry, str) else entry[1]


def _action_suffixes(model):
    return [_registered_suffix(s) for s in (getattr(model, 'suffixes', None) or [])]


def _truthy(value):
    text = str(value).strip().strip('"\'') if value is not None else ''
    if text in ('', '0'):
        return False
    try:
        return float(text) != 0.0
    except ValueError:
        return True


def action_outputs(model):
    """``[(suffix, kind, continues)]``: every action line of ``model`` that writes an output
    under its own suffix, in order -- a ``simulate`` (any ``simulate_*`` spelling),
    ``parameter_scan`` or ``bifurcate`` line, with the default suffix the bngsim backends give a
    line that names none. ``continues`` is a ``simulate`` with ``continue=>1``, which BNG2.pl
    appends to the output of the same suffix. Empty for a model whose actions are objects
    (SBML, RoadRunner): those write only the outputs they register."""
    import re
    from .bngsim_model.parsing import _collapse_action_line_continuations, _parse_action_dict
    outputs = []
    for action in getattr(model, 'actions', None) or []:
        if not isinstance(action, str):
            continue
        line = _collapse_action_line_continuations(action)
        line = re.sub(r'#.*$', '', line.strip()).strip().rstrip(';').strip()
        match = re.match(r'(\w+)\s*\(', line)
        if match is None:
            continue
        name = match.group(1)
        if name.startswith('simulate'):
            kind, default = 'simulate', 'time_course'
        elif name in ('parameter_scan', 'bifurcate'):
            kind, default = name, 'param_scan'
        else:
            continue
        params = _parse_action_dict(line, name)
        if params is None:
            continue
        outputs.append((str(params.get('suffix', default)), kind,
                        kind == 'simulate' and _truthy(params.get('continue'))))
    return outputs


def unregistered_outputs(model):
    """``{suffix: count}``: the outputs ``model``'s action lines write beyond the ones it
    registers -- a pre-equilibration's equilibration phase (``<name>_preequil``), or a legacy
    action line with no ``suffix=>`` -- counted per suffix, so a phase that shares a suffix with a
    registered experiment counts as a second writer of it. Nothing scores them, but ``execute``
    stores each under its own suffix, once as written and once per condition, so one can
    overwrite an output that is scored."""
    registered = {}
    for suffix in _action_suffixes(model):
        registered[suffix] = registered.get(suffix, 0) + 1
    written = {}
    for suffix, _kind, _cont in action_outputs(model):
        written[suffix] = written.get(suffix, 0) + 1
    return {suffix: n - registered.get(suffix, 0) for suffix, n in written.items()
            if n > registered.get(suffix, 0)}


def output_pairs(model, actions=None):
    """``[(key, action suffix, action, condition)]`` for every registered output ``model`` can
    produce.

    ``actions`` is an iterable of ``(action suffix, action)`` pairs; by default the model's own
    action suffixes, each standing for itself. ``condition`` is ``None`` for the model as
    written. A pair repeated with the same action suffix and condition is listed once, with the
    last of its actions: ``execute`` keeps the output the last one writes
    (:func:`refuse_duplicate_action_suffixes` admits only the repeats where that is the
    output)."""
    if actions is None:
        actions = [(s, s) for s in _action_suffixes(model)]
    pairs, index = [], {}
    for action_suffix, action in actions:
        for condition in simulation_conditions(model):
            identity = (action_suffix, id(condition))
            key = action_suffix + (condition.suffix if condition is not None else '')
            if identity in index:
                pairs[index[identity]] = (key, action_suffix, action, condition)
                continue
            index[identity] = len(pairs)
            pairs.append((key, action_suffix, action, condition))
    return pairs


def _plain_describe(action_suffix, condition):
    what = "the action with suffix '%s'" % action_suffix
    if condition is None:
        return what + ' run with no condition'
    return what + " run under condition '%s'" % condition.suffix


def resolve_output_key(model, key, actions=None):
    """``(action, condition)``: the one simulation of ``model`` whose output is named ``key``.

    ``condition`` is ``None`` for the model as written. ``actions`` is as for
    :func:`output_pairs`; by default the action returned is its suffix. Raises
    :class:`~pybnf.printing.PybnfError` when no simulation produces ``key`` or when several do
    -- a configuration that loads cannot reach the second case for a key anything reads,
    because :func:`refuse_ambiguous_output_keys` refuses it, but a model built another way
    can."""
    pairs = output_pairs(model, None if actions is None else list(actions))
    matches = [(suffix, action, condition) for k, suffix, action, condition in pairs if k == key]
    name = getattr(model, 'name', '?')
    if not matches:
        raise PybnfError(
            "No simulation of model '%s' produces the output '%s'." % (name, key),
            hint="Its simulations produce: %s." % (
                ', '.join(sorted({k for k, _, _, _ in pairs})) or '(none)'))
    if len(matches) > 1:
        raise PybnfError(
            "Model '%s': the output '%s' is produced by more than one simulation (%s), so "
            "which one a measurement, gradient or trajectory belongs to cannot be decided."
            % (name, key, '; '.join(_plain_describe(s, c) for s, _a, c in matches)),
            hint="Rename an action suffix or a condition so that no action suffix followed "
                 "by a condition name spells another action's output.")
    _suffix, action, condition = matches[0]
    return action, condition


def _producers(model):
    """``{key: [(action suffix, condition, registered)]}`` over every output ``execute``
    stores: the registered pairs and the unregistered outputs (:func:`unregistered_outputs`),
    each as written and under each condition."""
    by_key = {}
    for key, suffix, _action, condition in output_pairs(model):
        by_key.setdefault(key, []).append((suffix, condition, True))
    for suffix, count in unregistered_outputs(model).items():
        for _ in range(count):
            for condition in simulation_conditions(model):
                key = suffix + (condition.suffix if condition is not None else '')
                by_key.setdefault(key, []).append((suffix, condition, False))
    return by_key


def _describe(producer, edition, declared, phases):
    suffix, condition, registered = producer
    under = (" run under mutant '%s'" if edition < 2 else " run under condition '%s'")
    where = ' run with no condition' if condition is None else under % condition.suffix
    if not registered:
        if suffix in phases:
            return ("the unmeasured pre-equilibration phase of experiment '%s' (output '%s')%s"
                    % (phases[suffix], suffix, where))
        return "the unscored action with suffix '%s'%s" % (suffix, where)
    if edition >= 2 and suffix in declared:
        return "experiment '%s'%s" % (suffix, where)
    return "the action with suffix '%s'%s" % (suffix, where)


def _off_diagonal(producer, declared):
    """An experiment run under a condition it does not name."""
    suffix, condition, registered = producer
    if not registered or suffix not in declared:
        return False
    return (condition.suffix if condition is not None else '') != declared[suffix]


def refuse_ambiguous_output_keys(model, edition=2, consumed=None, declared=None):
    """Refuse a model two of whose simulations produce one output key that is read (#850, #833).

    ``execute`` writes the second over the first, the data loaded for that key is scored
    against whichever survives, and every consumer that reads a key back -- the gradient's
    condition, its per-condition request, multiple shooting's segment, a trajectory -- could
    pick either.

    ``consumed`` is the set of keys something reads, or ``None`` to count every key the model
    registers (edition 1, where every pair runs and ``am`` records every key). An edition-2
    model runs a pair only when its key is read (the emit set, #484), so two pairs that spell a
    key nothing reads never run and are left alone; two that spell a read key both run, and
    are refused. The outputs an action writes without registering them
    (:func:`unregistered_outputs`) always run; one that spells a read key is refused too. Two
    of those alone overwrite nothing anyone reads.

    ``declared`` maps an edition-2 experiment's name to the condition it names (``''`` for
    none); it chooses the words, and says when a pair is an experiment run under a condition it
    did not name. ``edition`` also only chooses the words."""
    declared = declared or {}
    phases = dict(getattr(model, 'phase_outputs', None) or {})
    for key, producers in _producers(model).items():
        if len(producers) < 2 or not any(registered for _s, _c, registered in producers):
            continue
        if consumed is not None and key not in consumed:
            continue
        first, second = producers[0], producers[1]
        hints = []
        crossed = [p for p in (first, second) if _off_diagonal(p, declared)]
        if crossed:
            suffix, condition, _ = crossed[0]
            named = declared[suffix]
            hints.append(
                "Every experiment of a model is simulated under every condition of that model, "
                "except a pairing PyBNF can tell nothing reads. Experiment '%s' names %s, but it "
                "also runs under condition '%s', because that run's output '%s' is read."
                % (suffix, "condition '%s'" % named if named else 'no condition',
                   condition.suffix if condition is not None else '', key))
        hidden = [p for p in (first, second) if not p[2]]
        if hidden and hidden[0][0] in phases:
            hints.append(
                "A pre-equilibration's unmeasured phase is stored under the experiment's name "
                "followed by '_preequil', so no other experiment can be named that.")
        elif hidden:
            hints.append(
                "An action line the model does not register still stores its output under its "
                "suffix ('time_course' or 'param_scan' when it names none).")
        hints.append(
            "Rename an experiment or a condition so that no experiment name followed by a "
            "condition name spells another experiment's output." if edition >= 2 else
            "Rename an action suffix or a mutant so that no action suffix followed by a mutant "
            "name spells another action's output.")
        raise PybnfError(
            "Model '%s': %s and %s would both produce the output '%s'. %s, so the two cannot "
            "be told apart: one simulation would overwrite the other, and data, gradients and "
            "trajectories for '%s' could be attributed to either."
            % (getattr(model, 'name', '?'), _describe(first, edition, declared, phases),
               _describe(second, edition, declared, phases), key,
               'Both are stored under that name' if hidden else
               "An output is named by its %s followed by its condition's name"
               % ("experiment's name" if edition >= 2 else 'action suffix'), key),
            hint=hints)


def refuse_duplicate_action_suffixes(model, edition=2):
    """Refuse two actions of ``model`` that register one output suffix, unless the output is
    plainly the last one's.

    Two repeats are legitimate, and both are what BNG2.pl does with the model file:

    * a ``simulate`` with ``continue=>1`` that resumes the run before it under the same suffix,
      which BNG2.pl appends to that output;
    * two actions of the same kind in the model file itself, the later writing over the
      earlier's output, as BNG2.pl writes over the file; every PyBNF backend keeps the last
      one too, and so does every lookup (:func:`output_pairs`).

    Any other repeat -- an action the configuration adds (an edition-2 experiment, a
    ``time_course`` / ``param_scan`` key) reusing a suffix, or a ``simulate`` and a
    ``parameter_scan`` sharing one -- leaves the data for that suffix scored against whichever
    ran last, which nothing on the configuration's side says, so it is refused."""
    registered = list(getattr(model, 'suffixes', None) or [])
    from_file = getattr(model, 'file_suffix_count', 0) or 0
    groups = {}
    for i, entry in enumerate(registered):
        groups.setdefault(_registered_suffix(entry), []).append(i)
    repeated = {s: idx for s, idx in groups.items() if len(idx) > 1}
    if not repeated:
        return
    outputs = action_outputs(model)
    for suffix, indices in repeated.items():
        lines = [(kind, cont) for s, kind, cont in outputs if s == suffix]
        if (len(lines) == len(indices) and lines[0][0] == 'simulate'
                and all(kind == 'simulate' and cont for kind, cont in lines[1:])):
            continue
        kinds = [registered[i][0] if not isinstance(registered[i], str) else 'action'
                 for i in indices]
        if all(i < from_file for i in indices) and len(set(kinds)) == 1:
            continue
        raise PybnfError(
            "Model '%s' has %d actions that write the output '%s' (%s). Only one of them would be "
            "kept, so the data for '%s' would be scored against whichever ran last."
            % (getattr(model, 'name', '?'), len(indices), suffix, ', '.join(kinds), suffix),
            hint=["Give each action its own suffix. A simulate that resumes the one before it "
                  "with continue=>1 may share its suffix, and so may two actions of one kind in "
                  "the model file (the later one's output is kept, as BNG2.pl keeps it); "
                  "nothing else may.",
                  "An experiment's name is its simulation's suffix, so a time_course or "
                  "param_scan key must not reuse an experiment's name." if edition >= 2 else
                  "A time_course or param_scan key must not reuse a suffix the model's own "
                  "actions write."])
