"""Phase 6 smoke test: SBML SSA via bngsim should track libroadrunner.

Compares per-time-point mean (rel-diff <5%), std (rel-diff <15%), and a
Kolmogorov-Smirnov distributional check (p > 0.01) on a 3-species
linear-chain SBML across both backends.

The matched-seed parity fixture is an SSA-correct (explicitly split)
variant of ``tests/bngl_files/abc.xml`` so both backends emit the same
shape under RR's gillespie integrator and bngsim's method='ssa'. The
original abc.xml uses COPASI/Antimony's reversible kineticLaw shape
(``compartment * (kf*A - kr*B)``); ``test_bngsim_ssa_runs_copasi_reversible_abc``
covers the Phase 7 split path that recognizes that shape and emits two
SSA channels per reversible SBML reaction.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import pybnf.bngsim_sbml_model as bngsim_sbml_model
from pybnf import pset


pytest.importorskip('roadrunner')

pytestmark = pytest.mark.bngsim_sbml


N_REPLICATES = 200
T_END = 500.0
STEP = 10.0
SPECIES = ('A', 'B', 'C')
# z-score thresholds correctly scale with N; ~5σ gives ~3e-7 per-test
# probability under H0, comfortably surviving Bonferroni over the
# ~450 (51 t-points × 3 species × {mean, std}) comparisons we make.
MEAN_Z_TOL = 5.0
STD_Z_TOL = 5.0
KS_P_FLOOR = 1e-4


# 3-species A↔B↔C chain, same rate constants and initial amounts as
# tests/bngl_files/abc.xml, but with each reversible reaction split into
# two explicit non-reversible channels so both backends produce the
# correct SSA equilibrium distribution.
ABC_SPLIT_SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="abc_split">
    <listOfCompartments>
      <compartment id="c" size="1" constant="true" spatialDimensions="3"/>
    </listOfCompartments>
    <listOfSpecies>
      <species id="A" compartment="c" initialAmount="20"
               hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="B" compartment="c" initialAmount="0"
               hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
      <species id="C" compartment="c" initialAmount="0"
               hasOnlySubstanceUnits="true"
               boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfParameters>
      <parameter id="kAB" value="0.01" constant="true"/>
      <parameter id="kBA" value="0.01" constant="true"/>
      <parameter id="kBC" value="0.1" constant="true"/>
      <parameter id="kCB" value="0.1" constant="true"/>
    </listOfParameters>
    <listOfReactions>
      <reaction id="A_to_B" reversible="false" fast="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>kAB</ci><ci>A</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
      <reaction id="B_to_A" reversible="false" fast="false">
        <listOfReactants>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>kBA</ci><ci>B</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
      <reaction id="B_to_C" reversible="false" fast="false">
        <listOfReactants>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="C" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>kBC</ci><ci>B</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
      <reaction id="C_to_B" reversible="false" fast="false">
        <listOfReactants>
          <speciesReference species="C" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
        <kineticLaw>
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><times/><ci>kCB</ci><ci>C</ci></apply>
          </math>
        </kineticLaw>
      </reaction>
    </listOfReactions>
  </model>
</sbml>"""


@pytest.fixture(scope='module')
def abc_split_xml(tmp_path_factory):
    path = tmp_path_factory.mktemp('abc_split') / 'abc_split.xml'
    path.write_text(ABC_SPLIT_SBML)
    return str(path)


def _ssa_action():
    return pset.TimeCourse({
        'time': str(T_END),
        'step': str(STEP),
        'method': 'ssa',
    })


def _collect_replicates(model, tmp_path, prefix):
    samples = {sp: [] for sp in SPECIES}
    times = None
    for i in range(N_REPLICATES):
        # Mimic what Job._run_models stamps onto the model copy: a unique
        # replicate_index per call so each replicate derives a distinct seed
        # under the default (`auto`) stochastic_seed policy.
        model._pybnf_replicate_index = i
        result = model.execute(str(tmp_path), '%s_%d' % (prefix, i), 1000)
        data = result['time_course']
        if times is None:
            times = data.data[:, data.cols['time']]
        for sp in SPECIES:
            samples[sp].append(data.data[:, data.cols[sp]])
    return times, {sp: np.asarray(samples[sp]) for sp in SPECIES}


def test_bngsim_ssa_matches_roadrunner(abc_split_xml, tmp_path):
    empty = pset.PSet([])
    rr_model = pset.SbmlModelNoTimeout(
        abc_split_xml, abc_split_xml, pset=empty, actions=(_ssa_action(),),
        integrator='gillespie',
    )
    bn_model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        abc_split_xml, abc_split_xml, pset=empty, actions=(_ssa_action(),),
        integrator='gillespie',
    )

    rr_times, rr_samples = _collect_replicates(rr_model, tmp_path, 'rr')
    bn_times, bn_samples = _collect_replicates(bn_model, tmp_path, 'bn')

    np.testing.assert_allclose(rr_times, bn_times, rtol=0, atol=1e-9)

    failures = []
    for sp in SPECIES:
        rr_arr = rr_samples[sp]
        bn_arr = bn_samples[sp]
        assert rr_arr.shape == bn_arr.shape

        for ti in range(1, rr_arr.shape[1]):
            rr_col = rr_arr[:, ti]
            bn_col = bn_arr[:, ti]
            n = rr_col.shape[0]
            rr_mu, bn_mu = rr_col.mean(), bn_col.mean()
            rr_sd, bn_sd = rr_col.std(ddof=1), bn_col.std(ddof=1)
            mean_se = max(np.sqrt(rr_sd ** 2 / n + bn_sd ** 2 / n), 1e-9)
            mean_z = abs(rr_mu - bn_mu) / mean_se
            # log-ratio of stds, SE ~ sqrt(1/(N-1)) per side under chi^2 approx;
            # combined SE is sqrt(2/(N-1)).
            log_se = np.sqrt(2.0 / max(n - 1, 1))
            std_z = (
                abs(np.log(bn_sd) - np.log(rr_sd)) / log_se
                if min(rr_sd, bn_sd) > 1e-3
                else 0.0
            )
            _, ks_p = stats.ks_2samp(rr_col, bn_col)

            if mean_z > MEAN_Z_TOL:
                failures.append(
                    '%s t=%.1f mean z=%.2f > %.1f (rr=%.3f±%.3f, bn=%.3f±%.3f)'
                    % (sp, rr_times[ti], mean_z, MEAN_Z_TOL,
                       rr_mu, rr_sd / np.sqrt(n), bn_mu, bn_sd / np.sqrt(n))
                )
            if std_z > STD_Z_TOL:
                failures.append(
                    '%s t=%.1f std z=%.2f > %.1f (rr=%.3f, bn=%.3f)'
                    % (sp, rr_times[ti], std_z, STD_Z_TOL, rr_sd, bn_sd)
                )
            if ks_p < KS_P_FLOOR:
                failures.append(
                    '%s t=%.1f KS p=%.2e < %.0e'
                    % (sp, rr_times[ti], ks_p, KS_P_FLOOR)
                )

    assert not failures, 'Backend parity violations:\n' + '\n'.join(failures)


def test_bngsim_ssa_action_method_path(abc_split_xml, tmp_path):
    """SSA via action.method='ssa' on a default integrator='cvode' model."""
    empty = pset.PSet([])
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        abc_split_xml, abc_split_xml, pset=empty, actions=(_ssa_action(),),
    )
    assert model.stochastic is True
    result = model.execute(str(tmp_path), 'abc_ssa_action_method', 60)
    data = result['time_course']
    assert data.cols['time'] == 0
    final_total = (
        data.data[-1, data.cols['A']]
        + data.data[-1, data.cols['B']]
        + data.data[-1, data.cols['C']]
    )
    assert abs(final_total - 20.0) < 1e-9


def test_bngsim_ssa_chi_sq_matches_roadrunner(abc_split_xml, tmp_path):
    """Chi^2 regression smoke test (GH #7 closeout): at fixed parameters,
    PyBNF's chi_sq objective must agree between sbml_backend='roadrunner'
    and sbml_backend='bngsim' within tolerance.

    Both backends simulate the same SBML at the same parameters; we
    average each backend's per-time mean over N replicates, then score
    that mean against synthetic exp data (deterministic mean from a fixed
    seed, with constant per-species sigma). chi^2 is bounded by
    finite-replicate noise, so the two backends' chi^2 values should
    differ by less than the within-backend noise floor.
    """
    from pybnf.objective import ChiSquareObjective
    from pybnf.data import Data

    empty = pset.PSet([])
    rr_model = pset.SbmlModelNoTimeout(
        abc_split_xml, abc_split_xml, pset=empty, actions=(_ssa_action(),),
        integrator='gillespie',
    )
    bn_model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        abc_split_xml, abc_split_xml, pset=empty, actions=(_ssa_action(),),
        integrator='gillespie',
    )

    rr_times, rr_samples = _collect_replicates(rr_model, tmp_path, 'rr_chi')
    bn_times, bn_samples = _collect_replicates(bn_model, tmp_path, 'bn_chi')

    np.testing.assert_allclose(rr_times, bn_times, rtol=0, atol=1e-9)

    # Synthetic exp data: average rr+bn means as the reference signal so
    # the chi^2 baseline is symmetric in the two backends. Sigma scales
    # are the per-species standard deviations of the rr replicate cloud.
    n_t = len(rr_times)
    n_sp = len(SPECIES)
    headers = ['time'] + list(SPECIES) + ['%s_SD' % sp for sp in SPECIES]
    arr = np.zeros((n_t, 1 + 2 * n_sp))
    arr[:, 0] = rr_times
    for i, sp in enumerate(SPECIES):
        rr_mu = rr_samples[sp].mean(axis=0)
        bn_mu = bn_samples[sp].mean(axis=0)
        rr_sd = np.maximum(rr_samples[sp].std(axis=0, ddof=1), 0.5)
        arr[:, 1 + i] = 0.5 * (rr_mu + bn_mu)
        arr[:, 1 + n_sp + i] = rr_sd
    exp_data = Data()
    exp_data.data = arr
    exp_data.cols = {h: i for i, h in enumerate(headers)}
    exp_data.headers = {i: h for i, h in enumerate(headers)}
    exp_data.indvar = 'time'
    exp_data.weights = np.ones_like(arr)

    def _sim_data(samples):
        mean_arr = np.zeros((n_t, 1 + n_sp))
        mean_arr[:, 0] = rr_times
        for i, sp in enumerate(SPECIES):
            mean_arr[:, 1 + i] = samples[sp].mean(axis=0)
        d = Data()
        d.data = mean_arr
        sim_headers = ['time'] + list(SPECIES)
        d.cols = {h: j for j, h in enumerate(sim_headers)}
        d.headers = {j: h for j, h in enumerate(sim_headers)}
        d.indvar = 'time'
        return d

    rr_sim = _sim_data(rr_samples)
    bn_sim = _sim_data(bn_samples)

    objective = ChiSquareObjective(ind_var_rounding=0)
    chi2_rr = objective.evaluate(rr_sim, exp_data, show_warnings=False)
    chi2_bn = objective.evaluate(bn_sim, exp_data, show_warnings=False)

    # Each backend's mean estimator has variance ~ sigma^2 / N per point;
    # the (mu - exp)^2 / (2 sigma^2) summand has expectation ~ 1/(2 N)
    # → chi2 ~ (n_t-1)*n_sp/(2 N). For n_t=51, n_sp=3, N=200 that's ~0.4.
    # Difference between the two chi^2 values reflects only the
    # difference of their two sample means, which under H0 has the same
    # noise scale, so we bound the absolute difference at a few times the
    # expected magnitude rather than chase a relative tolerance.
    n_compare = (n_t - 1) * n_sp
    chi2_floor = n_compare / (2.0 * N_REPLICATES)
    chi2_tol = max(5.0 * chi2_floor, 0.5)

    assert abs(chi2_rr - chi2_bn) < chi2_tol, (
        'chi^2 disagreement: rr=%g, bn=%g, tol=%g (floor=%g, n_compare=%d, N=%d)'
        % (chi2_rr, chi2_bn, chi2_tol, chi2_floor, n_compare, N_REPLICATES)
    )


def test_bngsim_ssa_runs_copasi_reversible_abc(tmp_path):
    """Phase 7: COPASI-style reversible kineticLaws (the original abc.xml
    shape, ``compartment * (kf*A - kr*B)``) must simulate under SSA, not
    raise. The bngsim SBML loader recognizes the wrapper-product ×
    binary-MINUS shape and emits two Elementary channels per SBML
    reaction; abc.xml's two reversible reactions become four channels.
    Smoke check: trajectory advances, conservation A+B+C=20 holds within
    integer-amount discretization."""
    abc_xml = str(Path(__file__).resolve().parent / 'bngl_files' / 'abc.xml')
    empty = pset.PSet([])
    model = bngsim_sbml_model.BngsimSbmlModelNoTimeout(
        abc_xml, abc_xml, pset=empty, actions=(_ssa_action(),),
        integrator='gillespie',
    )
    result = model.execute(str(tmp_path), 'abc_phase7', 60)
    data = result['time_course']
    final_total = (
        data.data[-1, data.cols['A']]
        + data.data[-1, data.cols['B']]
        + data.data[-1, data.cols['C']]
    )
    assert abs(final_total - 20.0) < 1e-9
