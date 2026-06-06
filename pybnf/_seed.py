"""Seed derivation policy for stochastic simulations.

Implements the four `stochastic_seed` config modes from issue #373:

- `auto` (default): derive a 31-bit seed from PyBNF evaluation context;
  override any explicit `seed=>N` in BNGL with a one-time warning.
- `auto_honorbngl`: derive from context, but honor explicit BNGL seeds verbatim.
- `random`: pass `None` to bngsim so it draws fresh per call; override BNGL.
- `random_honorbngl`: pass `None` to bngsim, but honor explicit BNGL seeds.

The derivation hashes `(param_set, model_name, action_index, suffix, method,
replicate_index)` so that:
  - same evaluation -> same trajectory (reproducibility);
  - distinct param vectors / models / actions / replicates -> distinct seeds.
"""


import hashlib


POLICY_AUTO = 'auto'
POLICY_AUTO_HONORBNGL = 'auto_honorbngl'
POLICY_RANDOM = 'random'
POLICY_RANDOM_HONORBNGL = 'random_honorbngl'

_HONORBNGL_POLICIES = frozenset((POLICY_AUTO_HONORBNGL, POLICY_RANDOM_HONORBNGL))
_AUTO_POLICIES = frozenset((POLICY_AUTO, POLICY_AUTO_HONORBNGL))


def _iter_param_items(param_set):
    """Yield (name, value) pairs from a PSet, dict, or None."""
    if param_set is None:
        return
    if hasattr(param_set, 'keys'):
        for name in sorted(param_set.keys()):
            yield name, param_set[name]


def derive_seed(*, param_set, model_name, action_index, suffix, method,
                replicate_index=0):
    """Compute a deterministic 31-bit seed from PyBNF evaluation context.

    Parts are joined with '|' and SHA-256 hashed; the first 4 bytes are
    masked to 31 bits so the result fits any backend's signed-int seed.
    """
    parts = [f'{name}={value!r}' for name, value in _iter_param_items(param_set)]
    parts.append('m:%s' % (model_name or ''))
    parts.append('i:%d' % int(action_index))
    parts.append('s:%s' % (suffix or ''))
    parts.append('x:%s' % (method or ''))
    parts.append('r:%d' % int(replicate_index))
    digest = hashlib.sha256('|'.join(parts).encode('utf-8')).digest()
    return int.from_bytes(digest[:4], byteorder='big') & 0x7fffffff


def resolve_seed(*, explicit_seed, policy, param_set, model_name, action_index,
                 suffix, method, replicate_index=0):
    """Apply the stochastic_seed policy to one simulation invocation.

    Returns `(seed, overridden)`:
      - `seed` is the integer to pass to bngsim, or `None` to let bngsim
        randomize via `secrets.randbits(31)`.
      - `overridden` is True when an explicit BNGL `seed=>N` was discarded
        (i.e., explicit_seed was set but policy is auto/random). Callers
        should emit a one-time warning per (model, action) when this is True.
    """
    if explicit_seed is not None and policy in _HONORBNGL_POLICIES:
        return int(explicit_seed), False
    overridden = explicit_seed is not None
    if policy in _AUTO_POLICIES:
        seed = derive_seed(
            param_set=param_set,
            model_name=model_name,
            action_index=action_index,
            suffix=suffix,
            method=method,
            replicate_index=replicate_index,
        )
        return seed, overridden
    return None, overridden


def resolve_action_seed(model, *, explicit_seed, action_index, suffix, method):
    """Resolve one stochastic action's seed from a bngsim model's policy context.

    Reads the ``stochastic_seed`` policy, ``param_set``, and replicate index off
    ``model`` and applies :func:`resolve_seed`. Returns
    ``(seed_value, overridden, policy)``; the caller emits its own backend-
    specific override log and applies any None-materialization or method gate
    (which is where the three bngsim backends genuinely differ).
    """
    policy = getattr(model, '_pybnf_stochastic_seed_policy', POLICY_AUTO)
    seed_value, overridden = resolve_seed(
        explicit_seed=explicit_seed,
        policy=policy,
        param_set=getattr(model, 'param_set', None),
        model_name=model.name,
        action_index=action_index,
        suffix=suffix,
        method=method,
        replicate_index=getattr(model, '_pybnf_replicate_index', 0),
    )
    return seed_value, overridden, policy
