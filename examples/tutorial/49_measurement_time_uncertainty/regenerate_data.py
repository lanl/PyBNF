#!/usr/bin/env python
"""Regenerate ``decay.exp`` for lesson 49 (measurement-time uncertainty).

Deterministic (all draws come from seed 7). Unlike the other tutorials -- whose
data is the model's own output at the truth, produced by the shared
``examples/tutorial/regenerate_data.py`` -- this lesson's data carries a *timing*
perturbation the shared generator has no mode for, so it lives here.

The decay ``x(t) = exp(-k t)`` is closed-form, so the data is that exact solution
evaluated at the true (perturbed) sampling times, plus Gaussian measurement noise:

    reported times  t_k    = linspace(0.5, 4.0, 8)          # what the .exp LABELS
    true times      tau_k ~ TruncNormal(t_k, sigma_t) on [t0, tmax]   # unobserved
    measurement     y_k    = exp(-theta * tau_k) + N(0, sigma_y)

A fit that scores at the reported ``t_k`` (standard.conf) is therefore biased;
one that marginalizes ``tau_k`` (marginal.conf / estimate_sigma_t.conf) recovers
``theta``. The manifest records only the recovery target (k = 1); the perturbation
magnitude sigma_t = 0.5 is the truth ``estimate_sigma_t.conf`` re-discovers.

    python examples/tutorial/49_measurement_time_uncertainty/regenerate_data.py
"""
from pathlib import Path

import numpy as np
from scipy.stats import truncnorm

THETA_TRUE = 1.0     # true decay rate (the recovery target)
SIGMA_T = 0.5        # timing-uncertainty scale (truth for estimate_sigma_t.conf)
SIGMA_Y = 0.05       # measurement-noise sd (known; the confs fix sigma at this)
T0, TMAX = 0.0, 10.0 # timing-prior support
SEED = 7


def main():
    reported = np.linspace(0.5, 4.0, 8)
    rng = np.random.default_rng(SEED)
    a, b = (T0 - reported) / SIGMA_T, (TMAX - reported) / SIGMA_T
    actual = truncnorm.rvs(a, b, loc=reported, scale=SIGMA_T, random_state=rng)
    y = np.exp(-THETA_TRUE * actual) + rng.normal(0, SIGMA_Y, size=len(reported))
    lines = ['# time\tX_obs'] + [f'{t:.10g}\t{v:.10g}' for t, v in zip(reported, y)]
    Path(__file__).with_name('decay.exp').write_text('\n'.join(lines) + '\n')
    print('wrote decay.exp')


if __name__ == '__main__':
    main()
