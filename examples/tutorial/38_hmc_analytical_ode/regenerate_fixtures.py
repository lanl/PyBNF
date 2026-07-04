"""Regenerate the committed .exp fixtures for lesson 38 (HMC on an analytical ODE
likelihood) directly from the models' CLOSED-FORM solutions -- no simulator.

That is the whole point of the lesson: each ODE here has an exact analytical
solution, so we can both (a) synthesize the data from the closed form and
(b) fit it with that same closed form as the likelihood, sampled by HMC/NUTS.

Both models are drawn from the analytical-ODE catalog
(~/Code/BNGL-Models/analytical_odes). We use only their closed forms; the BNGL
files themselves are not needed here (and their `Analytical_*` output functions
would trip up a simulator anyway -- see the suite notes).

Run from this folder:   python regenerate_fixtures.py
The true parameters below are the recovery targets recorded in ../_manifest.py.
"""
import numpy as np

HERE = __file__.rsplit("/", 1)[0]


def _write(path, header, t, y):
    with open(f"{HERE}/{path}", "w") as f:
        f.write("# " + "\t".join(header) + "\n")
        for row in zip(t, y):
            f.write("\t".join(f"{v:.6f}" for v in row) + "\n")


def viral_decay():
    """HIV/ART post-therapy viral decay:  dV/dt = -c V  =>  V(t) = V0 exp(-c t).
    (catalog: hiv_art_viral_decay). Mono-exponential, well identified."""
    V0, c, sigma = 100.0, 0.60, 2.5
    t = np.round(np.arange(0.0, 8.01, 0.5), 3)
    rng = np.random.default_rng(7)
    V = V0 * np.exp(-c * t) + rng.normal(0.0, sigma, t.size)
    _write("viral_decay.exp", ("time", "Vobs"), t, V)


def damped_oscillation():
    """Underdamped harmonic oscillator (phase 0):  x(t) = C exp(-a t) cos(w t).
    (catalog: damped_harmonic_oscillator). Oscillatory, C/a correlated."""
    C, a, w, sigma = 5.0, 0.35, 3.0, 0.15
    t = np.round(np.linspace(0.0, 10.0, 60), 4)
    rng = np.random.default_rng(11)
    x = C * np.exp(-a * t) * np.cos(w * t) + rng.normal(0.0, sigma, t.size)
    _write("damped_oscillation.exp", ("time", "xobs"), t, x)


if __name__ == "__main__":
    viral_decay()
    damped_oscillation()
    print("wrote viral_decay.exp, damped_oscillation.exp")
