"""The augmented variable layout: reported free parameters plus internal auxiliary blocks (#563).

A constrained transcription solves a *different* problem from the one the user asked
about. Multiple shooting (#563's first consumer) splits an experiment at knots and adds
one segment-start state ``z_j`` per knot, so the decision vector grows from the fit's
``k`` free parameters to ``k + sum_j dim(z_j)``. Direct collocation would add a state per
collocation node; latent-state estimation adds a state per unobserved species. In every
case the added coordinates are **internal to the transcription**: they are searched, they
are bounded, they carry gradient columns -- and they are *not* biological fit parameters.
Reporting them in ``sorted_params_*.txt`` would claim the fit estimated 3x as many
quantities as it did, and would put a quantity with no scientific meaning next to ones
that have it.

:class:`AugmentedLayout` is the bookkeeping that keeps those two populations apart while
letting one flat vector carry both. It owns exactly one thing: the map between

    ``u_aug = [ u_reported | z_1 | z_2 | ... | z_K ]``

and its named parts. The reported block is always first and always contiguous, so
``u_aug[:n_reported]`` is the vector every existing PyBNF seam already understands (a
PSet's coordinates in sampling space, ADR-0029) with no slicing ceremony -- and a
consumer that forgets to unpack gets the *reported* parameters, not a silently
misaligned mixture.

Space
-----
Every coordinate is in the space the optimizer walks. For the reported block that is the
free parameters' **sampling space** ``u`` (``log10(theta)`` for a ``logvar``), exactly as
``trf`` / ``lbfgs`` / ``gntr`` step in; for an internal block it is whatever space that
block declares its bounds in. The layout does not transform anything -- the ``d theta/d u``
chain rule stays where it already lives, in
:mod:`pybnf.gradient.assembly`, applied once when the Jacobian is built. A block that
wants to be searched in log space says so by being *built* in log space.

The homotopy seam
-----------------
The #563 prototype's central finding is that the **segment-count homotopy is the
mechanism**, not a refinement to add later (issue #563, finding 5.2): coarsening
``4 -> 2 -> 1`` is what converts a segmented stage that scores worse than a flat line into
a solve. A homotopy is a sequence of transcriptions of *the same fit*, so the layer needs
a way to carry a point from one layout to the next. :meth:`AugmentedLayout.carry_over`
is that: the reported block always survives (it is the same fit), an internal block
survives iff the next layout still declares a block of that name and size, and a block the
next layout adds is seeded from its own :attr:`VariableBlock.initial`. Matching is **by
name**, which is what makes the rule generic -- the layout never learns what a knot is.
"""

import numpy as np

from .errors import TranscriptionError

#: Separator between an internal block's name and one component's label in a qualified
#: name (``'seg2::A_state'``). Chosen because no PyBNF free-parameter name can contain
#: it, so a qualified internal name can never be mistaken for -- or collide with -- a
#: reported one.
QUALIFIER = '::'


class VariableBlock:
    """One named group of internal auxiliary variables.

    :param name: The block's identity. Unique within a layout, and the key
        :meth:`AugmentedLayout.carry_over` matches on across a homotopy stage change.
    :param labels: One label per component, for diagnostics and defect reports (for
        multiple shooting: the state names, e.g. ``('A_state', 'Y_state', 'Z_state')``).
    :param lower: Per-component lower bounds, in the block's own space.
    :param upper: Per-component upper bounds.
    :param initial: The value a layout that newly introduces this block starts it at --
        the consumer's best guess for the auxiliary variable (for multiple shooting, the
        state read off a nominal trajectory at the knot). Also what
        :meth:`AugmentedLayout.initial_point` seeds.

    Bounds are part of the block because the inner optimizers this layer feeds are
    bound-constrained (the Coleman-Li reflective step in ``trf`` / ``gntr``), and a
    segment-start concentration that is allowed to go negative is not a state the
    simulator can restart from.
    """

    def __init__(self, name, labels, lower, upper, initial):
        self.name = str(name)
        if not self.name:
            raise TranscriptionError('An internal variable block must have a non-empty name.')
        if QUALIFIER in self.name:
            raise TranscriptionError(
                "Internal variable block name %r contains the reserved qualified-name separator "
                "%r." % (self.name, QUALIFIER))
        self.labels = tuple(str(x) for x in labels)
        self.lower = np.asarray(lower, dtype=float).reshape(-1)
        self.upper = np.asarray(upper, dtype=float).reshape(-1)
        self.initial = np.asarray(initial, dtype=float).reshape(-1)
        n = len(self.labels)
        if n == 0:
            raise TranscriptionError('Internal variable block %r is empty.' % self.name)
        if not (len(self.lower) == len(self.upper) == len(self.initial) == n):
            raise TranscriptionError(
                'Internal variable block %r declares %i labels but %i lower / %i upper / %i '
                'initial values.' % (self.name, n, len(self.lower), len(self.upper),
                                     len(self.initial)))
        if len(set(self.labels)) != n:
            raise TranscriptionError(
                'Internal variable block %r has duplicate component labels.' % self.name)
        if np.any(self.lower > self.upper):
            raise TranscriptionError(
                'Internal variable block %r has a lower bound above its upper bound.' % self.name)
        if not np.all(np.isfinite(self.initial)):
            raise TranscriptionError(
                'Internal variable block %r has a non-finite initial value.' % self.name)

    @property
    def size(self):
        return len(self.labels)

    @property
    def qualified_names(self):
        """This block's components as ``'<block>::<label>'`` -- the names that appear in a
        defect report or a diagnostic, and that are guaranteed disjoint from every reported
        free-parameter name."""
        return tuple('%s%s%s' % (self.name, QUALIFIER, label) for label in self.labels)

    def clipped(self, values):
        """``values`` projected into this block's box -- what a consumer applies after a
        :meth:`AugmentedLayout.carry_over` whose source stage had looser bounds."""
        return np.clip(np.asarray(values, dtype=float).reshape(-1), self.lower, self.upper)

    def __repr__(self):
        return 'VariableBlock(%r, size=%i)' % (self.name, self.size)


class AugmentedLayout:
    """The map between a flat augmented vector and its reported / internal parts.

    :param reported_names: The fit's free parameters, in the order every existing PyBNF
        seam already uses (``Configuration.variables``). These occupy the leading,
        contiguous slice of every augmented vector.
    :param lower: Reported-block lower bounds, in sampling space.
    :param upper: Reported-block upper bounds, in sampling space.
    :param blocks: The internal :class:`VariableBlock`\\ s, in the order they are laid out
        after the reported block.

    The layout is immutable and cheap to build, so a homotopy builds one per stage.
    """

    def __init__(self, reported_names, lower, upper, blocks=()):
        self.reported_names = tuple(str(x) for x in reported_names)
        self._lower_reported = np.asarray(lower, dtype=float).reshape(-1)
        self._upper_reported = np.asarray(upper, dtype=float).reshape(-1)
        self.blocks = tuple(blocks)

        k = len(self.reported_names)
        if len(self._lower_reported) != k or len(self._upper_reported) != k:
            raise TranscriptionError(
                'The augmented layout declares %i reported parameters but %i lower / %i upper '
                'bounds.' % (k, len(self._lower_reported), len(self._upper_reported)))
        if len(set(self.reported_names)) != k:
            raise TranscriptionError('The augmented layout has duplicate reported parameter names.')
        if np.any(self._lower_reported > self._upper_reported):
            raise TranscriptionError(
                'The augmented layout has a reported lower bound above its upper bound.')

        seen = set(self.reported_names)
        self._slices = {}
        offset = k
        for block in self.blocks:
            if not isinstance(block, VariableBlock):
                raise TranscriptionError(
                    'An augmented layout takes VariableBlock objects; got %r.' % type(block).__name__)
            if block.name in self._slices:
                raise TranscriptionError(
                    'Duplicate internal variable block name %r in the augmented layout.'
                    % block.name)
            clash = seen.intersection(block.qualified_names)
            if clash:
                raise TranscriptionError(
                    'Internal variable %s collides with a name already in the augmented layout. '
                    'Internal auxiliary variables must be disjoint from the reported free '
                    'parameters -- they are never reported as fit results.'
                    % ', '.join(sorted(clash)))
            seen.update(block.qualified_names)
            self._slices[block.name] = slice(offset, offset + block.size)
            offset += block.size
        self._size = offset

    # -- shape ------------------------------------------------------------------

    @property
    def size(self):
        """Length of an augmented vector."""
        return self._size

    @property
    def n_reported(self):
        """Number of reported free parameters -- the fit's own ``k``."""
        return len(self.reported_names)

    @property
    def n_internal(self):
        """Number of internal auxiliary coordinates the transcription added."""
        return self._size - len(self.reported_names)

    @property
    def block_names(self):
        return tuple(block.name for block in self.blocks)

    @property
    def reported_slice(self):
        """The leading slice every augmented vector carries the reported parameters in."""
        return slice(0, len(self.reported_names))

    def slice_of(self, block_name):
        """The slice ``block_name`` occupies, raising rather than returning a wrong one."""
        try:
            return self._slices[block_name]
        except KeyError:
            raise TranscriptionError(
                'The augmented layout has no internal variable block %r (it has %s).'
                % (block_name, ', '.join(repr(n) for n in self.block_names) or 'none'))

    def block(self, block_name):
        """The :class:`VariableBlock` named ``block_name``."""
        self.slice_of(block_name)          # raises with the good message
        return next(b for b in self.blocks if b.name == block_name)

    @property
    def names(self):
        """Every coordinate's name: the reported free parameters, then each block's
        qualified component names. Guaranteed unique, and guaranteed to mark which
        coordinates are internal (they alone contain :data:`QUALIFIER`)."""
        out = list(self.reported_names)
        for block in self.blocks:
            out.extend(block.qualified_names)
        return tuple(out)

    def is_internal(self, index):
        """Whether coordinate ``index`` is an internal auxiliary variable rather than a
        reported free parameter -- the predicate any reporting path filters on."""
        return index >= len(self.reported_names)

    # -- bounds -----------------------------------------------------------------

    @property
    def lower(self):
        """Stacked lower bounds over the whole augmented vector."""
        return np.concatenate([self._lower_reported] + [b.lower for b in self.blocks]) \
            if self.blocks else self._lower_reported.copy()

    @property
    def upper(self):
        """Stacked upper bounds over the whole augmented vector."""
        return np.concatenate([self._upper_reported] + [b.upper for b in self.blocks]) \
            if self.blocks else self._upper_reported.copy()

    # -- packing ----------------------------------------------------------------

    def pack(self, reported, internals=None):
        """Build an augmented vector from the reported parameters and a
        ``{block name: values}`` mapping. Every declared block must be supplied."""
        reported = np.asarray(reported, dtype=float).reshape(-1)
        if len(reported) != len(self.reported_names):
            raise TranscriptionError(
                'The augmented layout expects %i reported parameters; got %i.'
                % (len(self.reported_names), len(reported)))
        internals = dict(internals or {})
        unknown = set(internals) - set(self.block_names)
        if unknown:
            raise TranscriptionError(
                'Unknown internal variable block(s) %s for this augmented layout.'
                % ', '.join(sorted(repr(n) for n in unknown)))
        out = np.empty(self._size, dtype=float)
        out[self.reported_slice] = reported
        for block in self.blocks:
            if block.name not in internals:
                raise TranscriptionError(
                    'No values supplied for internal variable block %r.' % block.name)
            values = np.asarray(internals[block.name], dtype=float).reshape(-1)
            if len(values) != block.size:
                raise TranscriptionError(
                    'Internal variable block %r takes %i values; got %i.'
                    % (block.name, block.size, len(values)))
            out[self._slices[block.name]] = values
        return out

    def unpack(self, u):
        """Split an augmented vector into ``(reported, {block name: values})``."""
        u = self._check(u)
        internals = {block.name: u[self._slices[block.name]].copy() for block in self.blocks}
        return u[self.reported_slice].copy(), internals

    def reported_of(self, u):
        """Just the reported free parameters -- the only part of the vector that is a fit
        result. Every reporting, certification, and PSet path goes through this."""
        return self._check(u)[self.reported_slice].copy()

    def internal_of(self, u, block_name):
        """Just block ``block_name``'s values."""
        return self._check(u)[self.slice_of(block_name)].copy()

    def initial_point(self, reported):
        """The augmented start point: ``reported`` as given, every internal block at its
        declared :attr:`VariableBlock.initial`."""
        return self.pack(reported, {b.name: b.initial for b in self.blocks})

    # -- embedding --------------------------------------------------------------

    def embed_gradient(self, gradient):
        """Zero-pad a reported-space gradient into augmented space.

        For the corner where a term genuinely has no dependence on the auxiliary variables
        -- a prior, a parameter-only penalty. A term that *does* depend on them (the data
        fit of a multiple-shooting segment, which reads ``z_j`` through the ``IC`` route)
        must be assembled in augmented space directly, not embedded.
        """
        gradient = np.asarray(gradient, dtype=float).reshape(-1)
        if len(gradient) != len(self.reported_names):
            raise TranscriptionError(
                'A reported-space gradient has %i entries; got %i.'
                % (len(self.reported_names), len(gradient)))
        out = np.zeros(self._size, dtype=float)
        out[self.reported_slice] = gradient
        return out

    def embed_jacobian(self, jacobian):
        """Zero-pad a reported-space ``(m, k)`` Jacobian's columns into augmented space."""
        jacobian = np.atleast_2d(np.asarray(jacobian, dtype=float))
        if jacobian.shape[1] != len(self.reported_names):
            raise TranscriptionError(
                'A reported-space Jacobian has %i columns; got %i.'
                % (len(self.reported_names), jacobian.shape[1]))
        out = np.zeros((jacobian.shape[0], self._size), dtype=float)
        out[:, self.reported_slice] = jacobian
        return out

    # -- homotopy ---------------------------------------------------------------

    def carry_over(self, u, target):
        """Move a point from this layout into ``target``'s -- one step of the homotopy.

        The reported block always survives: it is the same fit, and its value is the whole
        reason the previous stage ran. An internal block survives iff ``target`` declares a
        block of the same **name and size**; a block ``target`` adds is seeded from its own
        :attr:`VariableBlock.initial`; a block ``target`` dropped is discarded (that is what
        coarsening *is*). Carried values are clipped into the target block's box, since two
        stages need not bound an auxiliary variable identically.

        A name that matches with a *different* size is a consumer bug -- two stages disagree
        about what that block means -- and raises rather than being silently reseeded.
        """
        if not isinstance(target, AugmentedLayout):
            raise TranscriptionError('carry_over takes an AugmentedLayout target.')
        u = self._check(u)
        if target.reported_names != self.reported_names:
            raise TranscriptionError(
                'Cannot carry a point between augmented layouts with different reported free '
                'parameters -- a homotopy re-transcribes one fit, it does not change which '
                'parameters that fit estimates.')
        carried = {}
        for block in target.blocks:
            if block.name in self._slices:
                source = self.block(block.name)
                if source.size != block.size:
                    raise TranscriptionError(
                        'Internal variable block %r is %i wide in the source layout and %i wide '
                        'in the target; one block name must mean one thing across a homotopy.'
                        % (block.name, source.size, block.size))
                carried[block.name] = block.clipped(u[self._slices[block.name]])
            else:
                carried[block.name] = block.initial
        return target.pack(np.clip(u[self.reported_slice],
                                   target._lower_reported, target._upper_reported), carried)

    # -- reporting --------------------------------------------------------------

    def describe(self):
        """One line for the run log: how many coordinates the transcription added, and where."""
        if not self.blocks:
            return ('%i reported free parameters, no internal auxiliary variables '
                    '(this transcription is the plain single-shoot problem)'
                    % len(self.reported_names))
        return ('%i reported free parameters + %i internal auxiliary variables in %i block(s): %s'
                % (len(self.reported_names), self.n_internal, len(self.blocks),
                   ', '.join('%s[%i]' % (b.name, b.size) for b in self.blocks)))

    def _check(self, u):
        u = np.asarray(u, dtype=float).reshape(-1)
        if len(u) != self._size:
            raise TranscriptionError(
                'This augmented layout is %i wide; got a vector of length %i.'
                % (self._size, len(u)))
        return u

    def __repr__(self):
        return 'AugmentedLayout(k=%i, internal=%i, blocks=%i)' % (
            len(self.reported_names), self.n_internal, len(self.blocks))
