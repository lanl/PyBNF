"""Parameter Scale -- the space a free parameter is sampled, proposed, and
stored in (ADR-0003, ADR-0010).

A ``Scale`` owns the ``theta <-> u`` transform between a parameter's stored
value ``theta`` and the sampling space ``u`` the prior family and the proposal
arithmetic both operate in. ``Linear`` is the identity; ``Log10`` is base-10
log (``u = log10(theta)``); ``Ln`` is natural log (``u = ln(theta)``). The
transform lives here, in one place, so the log/exp boundary is not smeared
across ``FreeParameter.add`` / ``_reflect`` / ``prior_logpdf`` / ``sample_value``
(which all go through ``_scale``) -- so a new base composes for free there.

The scales are concrete singletons -- ``LINEAR``, ``LOG10``, ``LN`` -- not a
registry. Each carries an explicit ``name`` so its base is never ambiguous in
output (ADR-0022: every log scale names its base; there is no bare "log"); the
native ``parameter:`` record (ADR-0043) selects one by name
(``parameter_scale: linear|log10|ln``). ``Log10.inverse`` is ``10.0 ** u`` to
match ``exp10`` and the inline ``10**`` of the proposal arithmetic, bit-for-bit;
``Ln.inverse`` is ``np.exp(u)``.
"""

import numpy as np


class Scale:
    """Base class for a parameter scale (an abstract ``theta <-> u`` transform)."""

    is_log = False
    name = 'linear'

    def forward(self, theta):
        """Map a stored value ``theta`` into the sampling space ``u``."""
        raise NotImplementedError

    def inverse(self, u):
        """Map a sampling-space value ``u`` back to a stored value ``theta``."""
        raise NotImplementedError

    def inverse_jax(self, u):
        """JAX-traceable peer of :meth:`inverse` (the ``u -> theta`` map, ADR-0059).

        The gradient-based ``hmc`` sampler evaluates the model's NLL at
        ``theta = scale.inverse(u)`` and needs that step inside the autodiff graph,
        so a log-scaled parameter composes with NUTS. ``10.0 ** u`` already traces
        under JAX, but ``np.exp`` / ``np.log10`` do not, so each scale supplies a
        ``jnp`` peer. No change-of-variables Jacobian is added for this transform --
        the prior is defined in ``u`` (ADR-0010), so ``theta`` enters only through the
        likelihood. The default raises, mirroring :meth:`inverse`."""
        raise NotImplementedError


class Linear(Scale):
    is_log = False
    name = 'linear'

    def forward(self, theta):
        return theta

    def inverse(self, u):
        return u

    def inverse_jax(self, u):
        return u


class Log10(Scale):
    is_log = True
    name = 'log10'

    def forward(self, theta):
        return np.log10(theta)

    def inverse(self, u):
        return 10.0 ** u

    def inverse_jax(self, u):
        # 10.0 ** u traces unchanged under JAX (no jnp needed), and matches
        # ``inverse`` bit-for-bit -- kept as an explicit peer for symmetry.
        return 10.0 ** u


class Ln(Scale):
    is_log = True
    name = 'ln'

    def forward(self, theta):
        return np.log(theta)

    def inverse(self, u):
        return np.exp(u)

    def inverse_jax(self, u):
        import jax.numpy as jnp
        return jnp.exp(u)


LINEAR = Linear()
LOG10 = Log10()
LN = Ln()
