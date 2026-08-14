"""The segment-simulation seam: the one place multiple shooting touches a simulator (#563).

Everything else in :mod:`pybnf.shooting` is arithmetic over what this seam returns -- knot
placement, the ``IC``-routed objective assembly, the continuity block, the inner solver, the
homotopy. Narrowing the simulator coupling to a single method is what lets all of that be
verified offline against a closed-form backend, exactly as
:mod:`pybnf.transcription` is verified against a closed-form transcription: an offline
implementation of :class:`SegmentBackend` exercises the same code path bngsim does.

What a segment simulation is
----------------------------
One span ``[t_j, t_{j+1}]`` of one experiment, integrated **from a state the transcription
supplies** rather than from the model's own initial conditions, with output at the
segment's own data times and at its end knot. Two properties make it different from an
ordinary PyBNF simulation, and both are why this cannot ride the propose/score loop:

* the initial state is an *argument*, not a property of the parameter set -- segment ``j``'s
  start is the auxiliary variable ``z_j``, which is never a fit result and never enters a
  :class:`~pybnf.pset.PSet`; and
* the run must come back carrying its forward sensitivities on **both** axes. The parameter
  axis gives ``dPhi/dtheta``; the initial-condition axis gives ``dPhi/dz_j``, which is the
  block the continuity Jacobian is built from and the reason #563 needed
  ``sensitivity_ic`` to exist at all.

The prototype's structural finding is what keeps the objective half cheap: a segment-start
state enters the data fit as an ``IC`` route with chain-rule factor 1, so the *same*
tensor this seam already returns feeds
:func:`~pybnf.gradient.assembly.assemble_gradient_and_fisher_hessian` with no new residual
math. This module therefore returns the tensor as-is, in the native units the assembly
expects, and does no differentiation of its own.

Segment 0 is not special-cased
------------------------------
The first segment starts from the model's own initial conditions -- which are frequently
*fitted* (``init_Z_state`` and friends), so they are reported free parameters rather than
auxiliary ones. A caller passes ``initial_state = None`` for it, and the backend simulates
the model as configured. Everything downstream reads that back as "this segment has no
auxiliary block", which is the same shape as the ``m = 1`` stage having none at all.
"""

from abc import ABC, abstractmethod

import numpy as np

from ..printing import PybnfError


class SegmentSimulationFailed(Exception):
    """One segment did not integrate.

    Raised by a backend rather than returned, because a non-integrable segment is a
    property of the *point*, not of the run: the caller converts it into a non-finite local
    model, the inner solver's trust region shrinks, and the search backs off -- the same way
    a failed simulation is handled everywhere else on the gradient path (#492). It is not a
    fit-ending error, which is exactly the robustness the #563 prototype measured (a segment
    that fails does not kill the whole trajectory: median ``-166.95`` against single
    shooting's ``-105.50`` from an uninformed box draw).
    """


class SegmentTrace:
    """One simulated segment: its scored rows, and its end-knot state with derivatives.

    :param data: The segment's :class:`~pybnf.data.Data` over its own output grid,
        carrying the :class:`~pybnf.data.OutputSensitivities` payload. This is the object
        the objective assembly consumes, unchanged.
    :param end_state: The model state at the segment's **end knot**, in the order of
        ``state_names``. The left half of the continuity defect
        ``c_j = Phi_j(z_j, theta) - z_{j+1}``.
    :param d_end_param: ``d(end state)/d(native parameter)``, shape
        ``(n_state, len(param_axis))``, or ``None`` when no parameter axis was requested.
    :param d_end_ic: ``d(end state)/d(initial state)``, shape ``(n_state, len(ic_axis))``,
        or ``None``. For an interior segment this is the ``dPhi_j/dz_j`` block; for
        segment 0 it is the derivative with respect to the model's own initials, which is
        how a *fitted* initial condition reaches the continuity rows.
    :param param_axis: Native parameter ids labelling ``d_end_param``'s columns.
    :param ic_axis: Species labelling ``d_end_ic``'s columns.
    """

    __slots__ = ('data', 'end_state', 'd_end_param', 'd_end_ic', 'param_axis', 'ic_axis')

    def __init__(self, data, end_state, d_end_param=None, d_end_ic=None,
                 param_axis=(), ic_axis=()):
        self.data = data
        self.end_state = np.asarray(end_state, dtype=float).reshape(-1)
        self.d_end_param = None if d_end_param is None else np.atleast_2d(
            np.asarray(d_end_param, dtype=float))
        self.d_end_ic = None if d_end_ic is None else np.atleast_2d(
            np.asarray(d_end_ic, dtype=float))
        self.param_axis = tuple(param_axis)
        self.ic_axis = tuple(ic_axis)

    def is_finite(self):
        for array in (self.end_state, self.d_end_param, self.d_end_ic):
            if array is not None and not np.all(np.isfinite(array)):
                return False
        return True

    def __repr__(self):
        return 'SegmentTrace(states=%i, dparam=%s, dic=%s)' % (
            len(self.end_state),
            None if self.d_end_param is None else self.d_end_param.shape,
            None if self.d_end_ic is None else self.d_end_ic.shape)


class SegmentBackend(ABC):
    """What multiple shooting needs from a simulator, and nothing else.

    One instance per scored experiment (one ``(model, condition)`` pair), built once per
    fit and reused across every evaluation -- the #563 prototype measured that constructing
    a sensitivity-bearing simulator costs ~17 ms warm against ~50 ms for the integration
    itself, so at ``m`` segments per evaluation construction would dominate.
    """

    #: Segment integrations this backend has performed. The cost the #563 acceptance
    #: benchmark reports and the prototype's paired sweeps measured multiple shooting's
    #: 2-7x overhead in -- and the number a run reports as its completed simulations, which
    #: is deliberately *not* the count of augmented-model evaluations: one evaluation is
    #: ``m`` integrations, so reporting evaluations would understate the cost by the very
    #: factor the method is being judged on. An implementation increments it per
    #: :meth:`simulate`.
    n_simulations = 0

    def open_lanes(self, pset, n_lanes):
        """Prepare up to ``n_lanes`` independent simulation contexts at ``pset``; return how
        many are actually available (at least 1).

        A *lane* is whatever state :meth:`simulate` restarts to run one span -- for the
        bngsim backends, a warm engine model and ``Simulator``. Two segments cannot share
        one, because a segment is run by resetting that state to its own start knot, so a
        second segment on the same object would integrate from the first's start. Lanes are
        what makes :class:`~pybnf.shooting.parallel.SegmentPool` able to run ``L`` segments
        of one experiment at once.

        Called on the **calling thread**, before any worker starts, because preparing a lane
        is where a backend touches the model it owns. The default implementation offers one
        lane, which is the serial behaviour and needs no preparation of its own.
        """
        del pset, n_lanes
        return 1

    @property
    @abstractmethod
    def state_names(self):
        """The model state carried across a knot, in a fixed order.

        For an ODE model this is its species. It is the vector an auxiliary block holds,
        the vector a continuity row is written on, and the axis of ``d_end_ic``.
        """

    @property
    @abstractmethod
    def nominal_state(self):
        """A representative magnitude for each state, in ``state_names`` order.

        Used for two things a fit cannot do without: the strictly positive
        :class:`~pybnf.transcription.equality.EqualityModel` scales -- a continuity defect
        is a difference of states, so a model whose species span six orders of magnitude
        would otherwise hand the penalty term a condition number for free -- and the floor
        under an auxiliary variable's box.
        """

    @abstractmethod
    def simulate(self, pset, sample_times, initial_state=None, lane=0):
        """Integrate one span and return its :class:`~pybnf.data.Data`.

        :param pset: The reported parameter set, exactly as an ordinary evaluation would
            apply it (conditions, bind-by-id, initial assignments and all).
        :param sample_times: Strictly increasing output times. ``sample_times[0]`` is
            where integration starts and ``sample_times[-1]`` is the segment's end knot.
        :param initial_state: ``{state name: value}`` overriding the model's own initial
            conditions, or ``None`` for the first segment, which starts from the model as
            configured.
        :param lane: Which of the contexts :meth:`open_lanes` prepared to run in. ``0`` is
            the only one a serial pass ever uses, and the only one a backend that offers no
            lanes has to honour. A caller must never run two segments in one lane at once.

        Raises :class:`SegmentSimulationFailed` for a point that does not integrate.
        """


def trace_from_data(data, state_names, selector_prefix='species'):
    """Read a :class:`SegmentTrace` off a simulated :class:`~pybnf.data.Data`.

    The end-knot state is the last row of the state columns, and its derivatives are the
    last row of the sensitivity tensor -- which is the point of putting the end knot in the
    segment's own output grid rather than simulating the span twice.

    A ``Data`` with no ``output_sensitivities`` yields a value-only trace, which is what the
    certification path (an ordinary single-shoot score, no derivatives) and a scalar
    evaluation need.
    """
    state_names = tuple(state_names)
    missing = [name for name in state_names if name not in data.cols]
    if missing:
        raise PybnfError(
            'A multiple-shooting segment simulation returned no column for state(s) %s. The '
            'state carried across a knot must be a simulated column of the same experiment.'
            % ', '.join(sorted(missing)))
    rows = data.data
    end_state = np.array([rows[-1, data.cols[name]] for name in state_names], dtype=float)

    sens = data.output_sensitivities
    if sens is None:
        return SegmentTrace(data, end_state)

    selectors = ['%s:%s' % (selector_prefix, name) for name in state_names]
    unknown = [s for s in selectors if s not in sens.selectors]
    if unknown:
        raise PybnfError(
            'A multiple-shooting segment simulation returned no sensitivity column for %s. '
            'The continuity Jacobian is built from the state at the end knot, so every '
            'carried state must be a sensitivity selector of the same run.'
            % ', '.join(sorted(unknown)))
    d_end_param = None
    if sens.d_param is not None:
        d_end_param = np.array([sens.slice_for(s, axis='parameter')[-1] for s in selectors],
                               dtype=float)
    d_end_ic = None
    if sens.d_ic is not None:
        d_end_ic = np.array([sens.slice_for(s, axis='ic')[-1] for s in selectors], dtype=float)
    return SegmentTrace(data, end_state, d_end_param=d_end_param, d_end_ic=d_end_ic,
                        param_axis=tuple(sens.param_names), ic_axis=tuple(sens.ic_species))
