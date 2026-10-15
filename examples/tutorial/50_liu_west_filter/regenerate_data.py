"""Write cases.exp: negative-binomial counts, dispersion R (variance mu + mu**2 / R), around the
daily increase of ``cases`` in sir_outbreak.bngl at the model file's values.

The model is integrated by hand (fourth-order Runge-Kutta, step 1/100 day), so this needs only
numpy, and the draws come from ``RandomState``, whose streams numpy keeps fixed (NEP 19).

    python make_data.py
"""

from pathlib import Path

import numpy as np

BETA, GAMMA, N, I0 = 0.6, 0.25, 50000.0, 10.0   # the model file's values
R = 20.0          # dispersion of the count noise
DAYS = 42         # one count per day, days 1 to 42
STEPS = 100       # Runge-Kutta steps per day
SEED = 2026


def _rates(s, i):
    """dS/dt, dI/dt and dC/dt of sir_outbreak.bngl."""
    infection = BETA / N * s * i
    return -infection, infection - GAMMA * i, infection


def daily_means(days=DAYS):
    """The increase of ``cases`` over each day 1..days, from S = N - I0, I = I0, C = 0."""
    s, i, c = N - I0, I0, 0.0
    dt = 1.0 / STEPS
    means = []
    for _ in range(days):
        start = c
        for _ in range(STEPS):
            k1 = _rates(s, i)
            k2 = _rates(s + dt / 2 * k1[0], i + dt / 2 * k1[1])
            k3 = _rates(s + dt / 2 * k2[0], i + dt / 2 * k2[1])
            k4 = _rates(s + dt * k3[0], i + dt * k3[1])
            s += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            i += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            c += dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        means.append(c - start)
    return np.array(means)


def counts(means, seed=SEED):
    stream = np.random.RandomState(seed)
    return [int(stream.negative_binomial(R, R / (R + mu))) for mu in means]


def write_data(path):
    rows = ['%d\t%d' % (day, n) for day, n in enumerate(counts(daily_means()), start=1)]
    Path(path).write_text('\n'.join(['# time\tcases'] + rows) + '\n')


if __name__ == '__main__':
    write_data(Path(__file__).resolve().parent / 'cases.exp')
