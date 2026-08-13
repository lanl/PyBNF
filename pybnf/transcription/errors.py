"""Constrained-transcription exceptions (#563).

A single dependency-free home for the layer's exception, mirroring
:mod:`pybnf.gradient.errors`: the layout, the equality interface, and the outer loop
all raise it, and a consumer catches it without importing any of them.
"""

from ..printing import PybnfError


class TranscriptionError(PybnfError):
    """A constrained transcription could not be built, stepped, or reconciled.

    Raised for the *structural* faults of the layer -- a variable block whose name
    collides with a reported free parameter, a Jacobian block outside its declared
    shape, a non-positive constraint scale, a penalty schedule that cannot start.
    These are consumer wiring errors, not bad points to be stepped over: an
    augmented layout that does not describe what the residual assembly is about to
    write into is silently wrong in a way no fit result would reveal.

    It is a :class:`~pybnf.printing.PybnfError` so the message reaches the user
    intact when a fit is driven from a configuration rather than from a test.
    """
