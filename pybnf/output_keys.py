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

So every consumer resolves a key here, from the pairs, and two pairs that spell one key are
refused when the configuration loads (:func:`refuse_ambiguous_output_keys`), because no lookup
can tell them apart and ``execute`` writes one over the other.
"""

from .printing import PybnfError

__all__ = ['simulation_conditions', 'output_pairs', 'resolve_output_key',
           'refuse_ambiguous_output_keys']


def _is_model_as_written(mutant):
    """The SBML backends keep the model as written in their mutant list as an empty
    ``MutationSet``; it is the wildtype, not a condition."""
    return getattr(mutant, 'suffix', '') == '' and not list(getattr(mutant, 'mutations', ()) or ())


def simulation_conditions(model):
    """``[None, condition, ...]``: the model as written (``None``) and every condition or mutant
    it runs, each a :class:`~pybnf.pset.MutationSet`."""
    return [None] + [m for m in (getattr(model, 'mutants', None) or [])
                     if not _is_model_as_written(m)]


def _action_suffixes(model):
    return [s[1] for s in (getattr(model, 'suffixes', None) or [])]


def output_pairs(model, actions=None):
    """``[(key, action suffix, action, condition)]`` for every output ``model`` can produce.

    ``actions`` is an iterable of ``(action suffix, action)`` pairs; by default the model's own
    action suffixes, each standing for itself. ``condition`` is ``None`` for the model as
    written. A pair repeated with the same action suffix and condition is listed once, its first
    action kept."""
    if actions is None:
        actions = [(s, s) for s in _action_suffixes(model)]
    pairs, seen = [], set()
    for action_suffix, action in actions:
        for condition in simulation_conditions(model):
            identity = (action_suffix, id(condition))
            if identity in seen:
                continue
            seen.add(identity)
            key = action_suffix + (condition.suffix if condition is not None else '')
            pairs.append((key, action_suffix, action, condition))
    return pairs


def _describe(action_suffix, condition, edition=None):
    """One simulation, in the words of ``edition``: experiments and conditions (2), actions and
    mutants (1), or actions and conditions when the edition is not known."""
    what = ("experiment '%s'" if edition is not None and edition >= 2
            else "the action with suffix '%s'") % action_suffix
    if condition is None:
        return what + ' run with no condition'
    return what + (" run under mutant '%s'" if edition == 1 else " run under condition '%s'") \
        % condition.suffix


def resolve_output_key(model, key, actions=None):
    """``(action, condition)``: the one simulation of ``model`` whose output is named ``key``.

    ``condition`` is ``None`` for the model as written. ``actions`` is as for
    :func:`output_pairs`; by default the action returned is its suffix. Raises
    :class:`~pybnf.printing.PybnfError` when no simulation produces ``key`` or when several do
    -- a configuration that loads cannot reach the second case, because
    :func:`refuse_ambiguous_output_keys` refuses it, but a model built another way can."""
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
            % (name, key, '; '.join(_describe(s, c) for s, _a, c in matches)),
            hint="Rename an action suffix or a condition so that no action suffix followed "
                 "by a condition name spells another action's output.")
    _suffix, action, condition = matches[0]
    return action, condition


def refuse_ambiguous_output_keys(model, edition=2):
    """Refuse a model two of whose simulations produce one output key (#850, #833).

    ``execute`` writes the second over the first, the data loaded for that key is scored
    against whichever survives, and every consumer that reads a key back -- the gradient's
    condition, its per-condition request, multiple shooting's segment, a trajectory --
    could pick either. Refused for every pair the model can produce, scored or not: an
    unscored pair still overwrites a scored one in ``execute``. ``edition`` only chooses the
    words (experiments and conditions, or actions and mutants)."""
    by_key = {}
    for key, action_suffix, _action, condition in output_pairs(model):
        by_key.setdefault(key, []).append((action_suffix, condition))
    for key, producers in by_key.items():
        if len(producers) < 2:
            continue
        (a1, c1), (a2, c2) = producers[0], producers[1]
        raise PybnfError(
            "Model '%s': %s and %s would both produce the output '%s'. An output is named by "
            "its %s followed by its condition's name, so the two cannot be told apart: one "
            "simulation would overwrite the other, and data, gradients and trajectories for "
            "'%s' could be attributed to either."
            % (getattr(model, 'name', '?'), _describe(a1, c1, edition),
               _describe(a2, c2, edition), key,
               "experiment's name" if edition >= 2 else 'action suffix', key),
            hint=("Rename an experiment or a condition so that no experiment name followed by "
                  "a condition name spells another experiment's output." if edition >= 2 else
                  "Rename an action suffix or a mutant so that no action suffix followed by a "
                  "mutant name spells another action's output."))
