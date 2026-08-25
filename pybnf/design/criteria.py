"""Reducing an information matrix to one number, so two designs can be compared (#574).

The expected Fisher information ``F`` is a square matrix, one row and column per free parameter,
in **sampling space** (ADR-0029 -- so a log-scaled parameter's entry is about its order of
magnitude, which is the scale it is fitted on). Its inverse is the covariance matrix the fit
would have, so ``(F^-1)_kk`` is the variance of parameter ``k`` and ``sqrt(threshold *
(F^-1)_kk)`` is the half-width of that parameter's confidence interval in the quadratic
approximation -- the same interval a profile-likelihood run traces, for the same threshold.

A criterion turns that matrix into a single score so designs can be ranked:

* ``'a'`` -- the summed variance of the parameters, or of a named subset. With one parameter
  named this is the classical c-criterion.
* ``'d'`` -- the log determinant, which is the volume of the joint confidence region.
* ``'e'`` -- the smallest eigenvalue, which is the worst-determined direction.

Singular information is not a numerical accident to be smoothed away. A parameter the data cannot
constrain at all leaves a direction with no information in it, and the honest reading is an
infinite variance, not a large one. Every function here decides that with one shared rule: an
eigenvalue below :data:`SINGULAR_TOL` times the largest one counts as zero, and a parameter with
any weight on such a direction has infinite variance.
"""

import numpy as np

#: An eigenvalue this far below the largest one carries no information. Relative, because the
#: information matrix has the units of the data and can sit anywhere on the number line.
SINGULAR_TOL = 1e-10

#: How much of a parameter's own axis has to lie in the uninformed directions before its variance
#: is infinite rather than merely large. Squared weights, so this is a very small angle.
NULL_COMPONENT_TOL = 1e-10

#: The criteria a design run may be scored with.
CRITERIA = ('a', 'd', 'e')

#: The full name of each criterion, for messages and report headers.
CRITERION_NAMES = {
    'a': 'A-optimal (average parameter variance)',
    'd': 'D-optimal (confidence region volume)',
    'e': 'E-optimal (worst-determined direction)',
}


def lower_is_better(criterion):
    """Whether a smaller :func:`criterion_value` is the better design.

    The A-criterion is a variance, so smaller is better; the other two are amounts of
    information, so larger is. :func:`criterion_score` hides this (it is always maximized) but a
    report has to say which way its numbers read."""
    return criterion == 'a'


def _spectrum(information):
    """The eigenvalues and eigenvectors of ``information``, plus which eigenvalues count as zero.

    Symmetrized first: every term summed into the matrix is a symmetric outer product, so any
    asymmetry is rounding, and ``eigh`` reads only one triangle anyway."""
    matrix = np.asarray(information, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    largest = float(values[-1]) if values.size else 0.0
    cutoff = SINGULAR_TOL * largest if largest > 0.0 else 0.0
    uninformed = values <= cutoff
    return values, vectors, uninformed


def is_singular(information):
    """Whether any direction in parameter space carries no information at all.

    A singular information matrix means some combination of the parameters is invisible to the
    data. No amount of care with the criterion changes that; the design has to add a measurement
    that sees the missing direction."""
    if np.size(information) == 0:
        return False
    _values, _vectors, uninformed = _spectrum(information)
    return bool(uninformed.any())


def parameter_variances(information):
    """Each parameter's variance -- the diagonal of the inverse information -- in sampling space.

    Infinite for a parameter that lies (even partly) along a direction the data does not
    constrain, which is the correct reading rather than a failure: no finite confidence interval
    exists for it. Computed from the eigendecomposition rather than by inverting, so the
    uninformed directions can be recognized instead of producing a huge finite number."""
    values, vectors, uninformed = _spectrum(information)
    informed = ~uninformed
    weights = vectors ** 2                                   # row k: parameter k's weight per axis
    variances = np.full(values.shape, np.inf)
    lost = weights[:, uninformed].sum(axis=1) if uninformed.any() else np.zeros(values.shape)
    usable = lost <= NULL_COMPONENT_TOL
    if informed.any():
        finite = (weights[:, informed] / values[informed]).sum(axis=1)
        variances[usable] = finite[usable]
    return variances


def log_determinant(information):
    """``log det F``, or ``-inf`` when the information is singular (a confidence region of
    unbounded volume)."""
    values, _vectors, uninformed = _spectrum(information)
    if uninformed.any():
        return -np.inf
    return float(np.log(values).sum())


def smallest_eigenvalue(information):
    """The information in the worst-determined direction, floored at zero (a tiny negative
    eigenvalue is rounding: every term summed into the matrix is positive semi-definite)."""
    values, _vectors, _uninformed = _spectrum(information)
    return max(float(values[0]), 0.0) if values.size else 0.0


def criterion_value(information, criterion, targets=None):
    """The criterion read the way a person would state it: a summed variance for ``'a'``, a log
    determinant for ``'d'``, the smallest eigenvalue for ``'e'``.

    ``targets`` is the list of parameter indices the A-criterion sums over (``None`` -> all of
    them). Use :func:`lower_is_better` to know which direction is an improvement."""
    if criterion == 'a':
        variances = parameter_variances(information)
        chosen = variances if targets is None else variances[np.asarray(targets, dtype=int)]
        return float(chosen.sum())
    if criterion == 'd':
        return log_determinant(information)
    if criterion == 'e':
        return smallest_eigenvalue(information)
    raise ValueError('unknown design criterion %r' % (criterion,))


def criterion_score(information, criterion, targets=None):
    """The criterion as something to **maximize**, so the selection loop never has to ask which
    way a given criterion runs. The A-criterion's summed variance is negated; the others are
    already amounts of information."""
    value = criterion_value(information, criterion, targets)
    return -value if lower_is_better(criterion) else value


def null_space_gain(information, block, targets=None):
    """How much of ``block`` falls in the directions ``information`` currently knows nothing about.

    While the information is singular every criterion is pinned at its worst value -- an infinite
    variance, a log determinant of ``-inf``, a smallest eigenvalue of zero -- so none of them can
    tell two candidates apart. This can, and it asks the right question at that moment: of the
    directions the data does not yet see, how much does this measurement see? The selection loop
    uses it until the information becomes invertible and then goes back to the requested
    criterion.

    ``targets`` restricts the accounting to the uninformed directions that the target parameters
    actually lie along, so a c-criterion run is not sent off to fix a direction nobody asked
    about."""
    values, vectors, uninformed = _spectrum(information)
    if not uninformed.any():
        return 0.0
    axes = vectors[:, uninformed]                            # columns spanning the unseen space
    if targets is not None:
        idx = np.asarray(targets, dtype=int)
        # Keep only the unseen directions the targets have weight on. An unseen direction
        # orthogonal to every target cannot be why a target's variance is infinite.
        relevant = (axes[idx, :] ** 2).sum(axis=0) > NULL_COMPONENT_TOL
        if not relevant.any():
            return 0.0
        axes = axes[:, relevant]
    return float(np.einsum('ij,jk,ki->', axes.T, np.asarray(block, dtype=float), axes))


def unidentified_parameters(information, param_names):
    """The parameters this information matrix leaves with an infinite variance.

    Called on the information of the *largest possible* design -- everything already measured plus
    every candidate at once -- it names the parameters no experiment in the candidate space can
    pin down. That is a structural statement about the model and the observables, not a shortage
    of data, so it is reported rather than optimized around."""
    variances = parameter_variances(information)
    return [name for name, var in zip(param_names, variances) if not np.isfinite(var)]


def interval_half_widths(information, threshold):
    """Each parameter's confidence-interval half-width in **sampling space**, at the ``Delta
    chi2`` threshold a profile-likelihood run would use.

    This is the quadratic (Wald) approximation to the profile interval: the profile of a
    parameter near the optimum is the parabola ``Delta chi2 = (theta_k - theta*_k)^2 /
    (F^-1)_kk``, which crosses the threshold at ``+- sqrt(threshold * (F^-1)_kk)``. For a
    linear model the approximation is exact. Infinite for a parameter with infinite variance,
    which reads as an open interval."""
    return np.sqrt(threshold * parameter_variances(information))
