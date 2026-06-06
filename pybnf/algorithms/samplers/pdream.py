"""PDreamAlgorithm — Preconditioned DREAM (the ``p_dream`` fit type).

DREAM(ZS) with proposals computed in a covariance-whitened parameter space, for
better sampling of correlated posteriors. Extracted byte-identical (M1 Step 4).
Subclasses DreamAlgorithm (a sibling leaf in this sub-package).
"""


from .dream import DreamAlgorithm, DreamConfig
from ...registry import register_fit_type

from typing import Any

import logging
import numpy as np


# Preserve the original module logger name so log records keep the
# 'pybnf.algorithms' channel.
logger = logging.getLogger('pybnf.algorithms')


class PDreamConfig(DreamConfig):
    """Config for preconditioned DREAM (p_dream), co-located with the method
    (ADR-0006). Extends :class:`DreamConfig` with the one preconditioning key
    ``PDreamAlgorithm`` adds; everything else (incl. the β-ladder hook) is
    inherited."""

    precondition_adapt: Any = None


@register_fit_type('p_dream', family='sampler', display_name='Preconditioned DREAM',
                   schema=PDreamConfig)
class PDreamAlgorithm(DreamAlgorithm):
    """
    P-DREAM: Preconditioned DREAM.

    Extends DREAM(ZS) by computing DE proposals in a covariance-whitened parameter space.
    An online covariance estimate C is learned from the chain history (as in Adaptive Metropolis).
    Donors are transformed to whitened coordinates z = L_inv @ x before computing DE differences,
    and crossover is applied in whitened space where dimensions are decorrelated.

    The proposal remains symmetric (DE differences from an external archive), so standard
    Metropolis-Hastings acceptance is valid without additional Hastings correction.

    After a configurable adaptation period, the covariance is updated every generation from
    the pooled chain history.  Before adaptation begins, the sampler behaves identically to
    DREAM(ZS).
    """

    def __init__(self, config):
        super().__init__(config)
        pa = config.config['precondition_adapt']
        self.precondition_adapt = pa if pa is not None else self.burn_in // 2
        self._cov_L = None       # Cholesky factor of the covariance estimate
        self._cov_L_inv = None   # Inverse of Cholesky factor (whitening matrix)
        self._preconditioned = False

    def _update_covariance(self):
        """
        Estimate the covariance from pooled chain history and compute Cholesky factors.
        Discards the first 50% of each chain as warmup (matching the convention used by
        diagnostics.split_chains for R-hat/ESS) so the early burn-in transient does not
        inflate the preconditioner. Pooling across chains is intentional: P-DREAM's global
        preconditioner wants a proposal scale large enough for archive-based mode hopping.
        """
        # Pool the post-warmup half of each chain history into one matrix
        all_samples = []
        for chain in self.chain_history:
            if len(chain) > 1:
                all_samples.extend(chain[len(chain) // 2:])
        if len(all_samples) < 2 * self.n_dim:
            return  # Not enough samples yet

        X = np.array(all_samples)
        n = X.shape[0]
        d = X.shape[1]

        # Sample covariance with Haario-style regularization: C = Cov(X) + eps*I
        cov = np.cov(X, rowvar=False)
        eps = 1e-6 * np.trace(cov) / d if np.trace(cov) > 0 else 1e-6
        cov += eps * np.eye(d)

        try:
            L = np.linalg.cholesky(cov)
            self._cov_L = L
            self._cov_L_inv = np.linalg.solve(L, np.eye(d))
            if not self._preconditioned:
                self._preconditioned = True
                logger.info('P-DREAM: preconditioning activated at iteration %d '
                            'with %d pooled samples (d=%d)'
                            % (min(self.iteration), n, d))
            else:
                logger.debug('P-DREAM: covariance updated with %d samples' % n)
        except np.linalg.LinAlgError:
            logger.warning('P-DREAM: Cholesky decomposition failed, '
                           'skipping covariance update')

    def _whiten(self, x_vec):
        """Transform a parameter vector to whitened space: z = L_inv @ x."""
        return self._cov_L_inv @ x_vec

    def _unwhiten_diff(self, dz_vec):
        """Transform a difference vector from whitened space back: dx = L @ dz."""
        return self._cov_L @ dz_vec

    def got_result(self, res):
        """Override to update covariance estimate after each generation sync."""
        result = super().got_result(res)

        # After a full generation sync with new proposals, update the covariance
        if isinstance(result, list) and len(result) > 0:
            if min(self.iteration) >= self.precondition_adapt:
                self._update_covariance()

        return result

    def calculate_new_pset(self, idx):
        """
        DE proposal in whitened space.

        When preconditioning is active:
        1. Transform current state and archive donors to z = L_inv @ x
        2. Compute DE difference in z-space
        3. Apply crossover in z-space (dimensions are decorrelated)
        4. Scale and add perturbation in z-space
        5. Convert the total jump back: dx = L @ dz_total
        6. Propose x_new = x_current + dx

        Before preconditioning activates, falls back to standard DREAM proposals.
        """
        if not self._preconditioned:
            return super().calculate_new_pset(idx)

        x0 = self.current_pset[idx]
        x0_vec = self._param_vec(x0)

        # Draw 2*delta donor states from the ZS archive (without replacement)
        sel = self.chain_rngs[idx].choice(len(self.archive), 2 * self.delta, replace=False)

        # Whiten the donor states
        z_donors = []
        for s in sel:
            z_donors.append(self._whiten(self._param_vec(self.archive[s])))

        # Sample crossover value and mask (in whitened space where dims are independent)
        cr_idx = self.chain_rngs[idx].choice(self.ncr_count, p=self.cr_probs)
        cr = self.ncr[cr_idx]
        while True:
            ds = self.chain_rngs[idx].uniform(size=self.n_dim) <= cr
            if np.any(ds):
                break

        # Gamma selection
        if self.chain_rngs[idx].uniform() < self.g_prob:
            gamma = 1
            ds[:] = True  # mode jump: update all dimensions
        else:
            d_prime = int(np.sum(ds))
            if self.adaptive_step_size:
                gamma = 2.38 / np.sqrt(2.0 * self.delta * d_prime)
            else:
                gamma = self.step_size

        # Compute DE difference in whitened space
        dz_total = np.zeros(self.n_dim)
        for j in range(self.delta):
            dz_total += z_donors[j] - z_donors[self.delta + j]

        # Apply crossover mask in whitened space
        dz_masked = np.where(ds, dz_total, 0.0)

        # Small perturbations in whitened space
        zeta_z = self.chain_rngs[idx].normal(0, self.config.config['zeta'], size=self.n_dim)
        lamb = self.chain_rngs[idx].uniform(-self.config.config['lambda'], self.config.config['lambda'])

        # Total jump in whitened space, then transform back to original space
        dz_jump = zeta_z + (1.0 + lamb) * gamma * dz_masked
        dx_jump = self._unwhiten_diff(dz_jump)

        # Build proposed PSet in original space (reject rather than reflect)
        xp_vec = x0_vec + dx_jump
        proposal = self._proposal_pset(xp_vec)
        if proposal is None:
            logger.debug("Proposed parameter outside of bounds")
            return None, cr_idx

        return proposal, cr_idx
