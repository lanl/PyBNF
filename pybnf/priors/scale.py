"""Parameter Scale -- the space a free parameter is sampled, proposed, and
stored in (ADR-0003, ADR-0010).

A ``Scale`` owns the ``theta <-> u`` transform between a parameter's stored
value ``theta`` and the sampling space ``u`` the prior family and the proposal
arithmetic both operate in. ``Linear`` is the identity; ``Log10`` is base-10
log (``u = log10(theta)``). The transform lives here, in one place, so the
``log10``/``10**`` boundary is not smeared across ``FreeParameter.add`` /
``_reflect`` / ``prior_logpdf`` / ``sample_value``.

The two scales are a closed set (not an extension point like the prior
families), so they are concrete singletons -- ``LINEAR`` and ``LOG10`` -- rather
than a registry. ``Log10.inverse`` is ``10.0 ** u`` to match ``exp10`` and the
inline ``10**`` already used by the proposal arithmetic, bit-for-bit.
"""

import numpy as np


class Scale:
    """Base class for a parameter scale (an abstract ``theta <-> u`` transform)."""

    is_log = False

    def forward(self, theta):
        """Map a stored value ``theta`` into the sampling space ``u``."""
        raise NotImplementedError

    def inverse(self, u):
        """Map a sampling-space value ``u`` back to a stored value ``theta``."""
        raise NotImplementedError


class Linear(Scale):
    is_log = False

    def forward(self, theta):
        return theta

    def inverse(self, u):
        return u


class Log10(Scale):
    is_log = True

    def forward(self, theta):
        return np.log10(theta)

    def inverse(self, u):
        return 10.0 ** u


LINEAR = Linear()
LOG10 = Log10()
