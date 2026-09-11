"""Stochastic parameter recovery benchmark (lanl/PyBNF#663, small version).

Six published stochastic models, each with a frozen problem definition (true
parameter values, search bounds, simulation settings, observables, sampling times,
replicate count and seed for the data) and committed synthetic data, plus a scoring
protocol and a runner that scores PyBNF's own methods against them.

``protocol`` is pure Python (problem definitions, scoring, aggregation) and imports
nothing from PyBNF, so the frozen definitions and the scoring rules can be read and
applied without a simulation backend. ``harness`` is the PyBNF-dependent half: it
generates the data, builds and runs fits, and records what each fit did.
"""
