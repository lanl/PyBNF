"""The ``SigmaSource`` abstraction (ADR-0021): where a noise model's noise
parameter comes from.

ADR-0011 made the ``NoiseModel`` a pure per-point kernel and named the *noise
parameter source* as the objective wrapper's job, but left it hard-coded -- the
``_SD`` data column, the magic free parameters ``sigma__FREE`` / ``r__FREE``, and
the ``neg_bin_r`` constant each lived in a different objfunc subclass. This module
lifts those three into one first-class abstraction so they can be selected **per
observable** (the #410 engine) and named freely (dissolving the magic strings, as
M2.3 did for priors).

A ``SigmaSource`` answers two questions the per-point engine needs: its
``value(...)`` for one observation, and whether it is ``estimated``. The
**estimated** flag is load-bearing -- it is what decides whether the family's
likelihood normalizer is summed: a fixed source (a data column, a constant)
contributes only the family's ``data_fit``, while an estimated source (a free
parameter) contributes the full ``nll`` including the normalizer. This is exactly
ADR-0011's "normalizer retained iff the noise parameter is estimated", now keyed
off the source rather than hard-coded per objfunc.
"""

from abc import ABC, abstractmethod

from ..printing import PybnfError


class SigmaSource(ABC):
    """Where a noise model reads its noise parameter for one observation.

    ``estimated`` decides normalizer inclusion (see the module docstring); it is
    ``False`` by default and overridden to ``True`` by the free-parameter source.
    """

    estimated = False

    @abstractmethod
    def value(self, owner, exp_data, exp_row, col_name):
        """The noise-parameter value for one observation. ``owner`` is the objective
        (used by the free-parameter source to read the resolved pset values);
        ``exp_data``/``exp_row``/``col_name`` locate the point (used by the data-
        column source)."""

    def required_free_param(self):
        """The ``__FREE`` parameter name this source requires the fit to declare,
        or ``None`` if it sources no free parameter (used by ``_load_variables``
        validation, ADR-0021)."""
        return None

    def exp_column(self, col_name):
        """The experimental-data column this source consumes, or ``None`` if it
        reads no data column -- so ``_check_columns`` can exempt it from the
        unused-column error."""
        return None


class DataColumnSigma(SigmaSource):
    """The noise parameter read per point from an experimental-data column named
    ``<observable><suffix>`` (``chi_sq`` / ``lognormal``: the ``_SD`` column). It is
    a *fixed* source -- the value is data, not estimated -- so the caller drops the
    likelihood normalizer. The suffix is explicit (default ``_SD``) so a non-Gaussian
    family can read a differently-named column without the "standard deviation"
    misnomer (ADR-0021)."""

    estimated = False

    def __init__(self, suffix='_SD'):
        self.suffix = suffix

    def exp_column(self, col_name):
        return col_name + self.suffix

    def value(self, owner, exp_data, exp_row, col_name):
        column = col_name + self.suffix
        try:
            idx = exp_data.cols[column]
        except KeyError:
            # Todo: Check for this and throw the error before all the workers get created.
            raise PybnfError(f'Column {column} not found',
                 f"Column {column} was not found in the experimental data. A noise model that reads its "
                 f"scale from the data requires a {column} column corresponding to {col_name}, giving the "
                 "per-point noise scale (e.g. the standard deviation) of that variable. ")
        return exp_data.data[exp_row, idx]


class FreeParameterSigma(SigmaSource):
    """The noise parameter estimated as a free parameter, resolved by name from the
    pset (``chi_sq_dynamic``'s ``sigma__FREE``, ``neg_bin_dynamic``'s ``r__FREE``,
    or a per-observable ``__FREE`` parameter). It is *estimated*, so the caller
    keeps the likelihood normalizer."""

    estimated = True

    def __init__(self, name):
        self.name = name

    def required_free_param(self):
        return self.name

    def value(self, owner, exp_data, exp_row, col_name):
        # ``owner._pset_values`` is the {name: value} map the objective's
        # evaluate_multiple builds once per evaluation from the pset (ADR-0021).
        return owner._pset_values[self.name]


class ConstantSigma(SigmaSource):
    """The noise parameter held at a fixed configuration constant (``neg_bin``'s
    ``neg_bin_r``; the native ``fix_at`` source). Fixed, so the caller drops the
    likelihood normalizer."""

    estimated = False

    def __init__(self, value):
        self.const = value

    def value(self, owner, exp_data, exp_row, col_name):
        return self.const
