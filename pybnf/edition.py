"""The select-and-freeze ``edition`` marker (ADR-0031).

A ``.conf`` may declare ``edition = <integer>`` to opt into a frozen set of
modernized conventions. Editions are **select-and-freeze** in the Rust sense: an
``edition = N`` conf is interpreted under edition-N conventions *forever*, even as
later PyBNF releases change other defaults under higher editions. This is why the
marker is an ``edition`` and not a ``min_version`` floor: a floor pins tooling,
not semantics, so the next default change would silently re-interpret every conf
that declared an older floor -- exactly the drift this mechanism exists to
prevent.

**Absence of the key means legacy semantics:** the implicit
:data:`LEGACY_EDITION` (1), byte-identical to PyBNF's historical behavior. New-era
syntax requires an explicit ``edition`` at or above the edition that introduced it
(:func:`require_edition`); using it without one is an immediate, named error
rather than a silent reinterpretation.

The value is a plain integer **decoupled from PyBNF release numbers and years**
(editions change only when a convention changes, not every release). The tool
*derives* the minimum-supporting PyBNF version for an edition it knows
(:data:`EDITION_INTRODUCED_IN`) and reports it; an edition newer than this PyBNF
understands is reported against the running version.

This module is pure policy data plus small predicates -- it imports nothing from
``config`` so the config layer (and the future modern-surface steps) can depend on
it without a cycle.
"""

from . import __version__
from .printing import PybnfError


#: The implicit edition when ``edition`` is absent: legacy (historical) behavior.
LEGACY_EDITION = 1

#: The newest edition this PyBNF release understands. A conf declaring a higher
#: edition is rejected -- this PyBNF predates those conventions.
CURRENT_EDITION = 2

#: The first PyBNF version that supports each known edition -- the
#: minimum-supporting version the tool reports. Edition 1 is the historical
#: behavior (supported from the start); edition 2 introduces the ADR-0031
#: median-universal centering and the modernized objective surface. (Version at
#: release time: edition 2 ships in the next release after v1.4.0; update here if
#: that number changes.)
EDITION_INTRODUCED_IN = {
    LEGACY_EDITION: '0.0.0',
    2: '1.5.0',
}

#: The editions this PyBNF understands (1 .. CURRENT_EDITION).
SUPPORTED_EDITIONS = frozenset(range(LEGACY_EDITION, CURRENT_EDITION + 1))


def resolve_edition(value):
    """The effective edition for a raw ``edition`` config value.

    ``None`` (the key absent) resolves to :data:`LEGACY_EDITION`; an explicit value
    is validated and returned as an int. Raises ``PybnfError`` for a malformed or
    unsupported edition (see :func:`validate_edition`).
    """
    if value is None:
        return LEGACY_EDITION
    return validate_edition(value)


def validate_edition(value):
    """Validate an explicit ``edition`` value and return it as an int.

    Raises ``PybnfError`` when ``value`` is not a positive integer this PyBNF
    supports. A *future* edition (greater than :data:`CURRENT_EDITION`) is reported
    against the running version, since this PyBNF predates that edition's
    conventions and cannot name the release that introduced it.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise PybnfError(
            f'Invalid edition {value!r}',
            f"Config key 'edition' must be a positive integer (a supported edition "
            f"is one of {_supported_list()}); got {value!r}.")
    if value < LEGACY_EDITION:
        raise PybnfError(
            f'Invalid edition {value}',
            f"Config key 'edition' must be a positive integer (a supported edition "
            f"is one of {_supported_list()}); got {value}.")
    if value > CURRENT_EDITION:
        raise PybnfError(
            f'Unsupported edition {value}',
            f"Config 'edition = {value}' needs a newer PyBNF than {__version__}, which "
            f"understands editions up to {CURRENT_EDITION}. Upgrade PyBNF, or set "
            f"'edition = {CURRENT_EDITION}' (or omit it for legacy behavior).")
    return value


def is_modern(edition):
    """True when an already-resolved ``edition`` is past the legacy edition -- i.e.
    it opts into the ADR-0031 modernized conventions."""
    return edition > LEGACY_EDITION


def min_version_for(edition):
    """The minimum PyBNF version that supports ``edition`` (a known edition), for
    reporting. An edition this PyBNF does not know falls back to the running
    version."""
    return EDITION_INTRODUCED_IN.get(edition, __version__)


def require_edition(edition, min_edition, feature, *, key='edition'):
    """Guard a modern-era syntax ``feature`` behind an explicit ``edition``.

    ``edition`` is the conf's already-resolved edition (absence -> legacy 1).
    Raises ``PybnfError`` -- naming the ``key`` and the fix -- when it is below
    ``min_edition``, so using new-era syntax without opting in is an immediate,
    explanatory error rather than a silent reinterpretation (ADR-0031). This is the
    seam every later modern-surface step (the three-key objective surface, the new
    sigma-source verbs, the legacy-token desugaring) calls before honoring its
    syntax; it is a no-op once the conf has opted in.
    """
    if edition >= min_edition:
        return
    at = (f'at edition {edition} (legacy)' if edition <= LEGACY_EDITION
          else f'at edition {edition}')
    raise PybnfError(
        f'{feature} requires edition {min_edition}',
        f"{feature} is edition-{min_edition} syntax, but this .conf is {at}. Add "
        f"'{key} = {min_edition}' to your .conf to use it (minimum PyBNF version "
        f"{min_version_for(min_edition)}).")


def _supported_list():
    return ', '.join(str(e) for e in sorted(SUPPORTED_EDITIONS))
