"""The fit's total wall-clock budget: ``wall_time_fit`` (#529, ADR-0093).

Two layers are covered here — the stopwatch (:mod:`pybnf.budget`) and the
configuration surface that admits or refuses the key. The *behavior* the budget
drives (the run loop stopping and finalizing anyway) lives with the rest of the
run-loop orchestration tests, in ``test_run_loop.py``.

The clock is injected everywhere, so nothing here sleeps or depends on how fast
the test machine is.
"""
import os

import numpy as np
import pytest

from . import integration_harness as H
from .context import algorithms, budget as budget_mod, config, printing


class _Clock:
    """A monotonic clock the test drives by hand."""

    def __init__(self, t=0.0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt
        return self.t


# --------------------------------------------------------------------------- #
# FitBudget: the stopwatch
# --------------------------------------------------------------------------- #
def test_budget_counts_down_and_expires_exactly_at_the_limit():
    clock = _Clock()
    b = budget_mod.FitBudget(10.0, clock=clock)

    assert b.elapsed() == 0.0 and b.remaining() == 10.0 and not b.expired()
    clock.advance(4.0)
    assert b.elapsed() == 4.0 and b.remaining() == 6.0 and not b.expired()
    clock.advance(6.0)  # exactly at the deadline
    assert b.expired() and b.remaining() == 0.0
    clock.advance(100.0)  # and it stays expired, with remaining clamped at 0
    assert b.expired() and b.remaining() == 0.0


def test_budget_charges_time_already_spent_before_it_was_built():
    """Configuration loading and network generation happen before the budget object
    exists; that head start is charged to it, so the budget bounds the whole run and
    not just the part after setup."""
    clock = _Clock()
    b = budget_mod.FitBudget(10.0, elapsed=7.0, clock=clock)

    assert b.remaining() == 3.0
    clock.advance(3.0)
    assert b.expired()


def test_from_config_returns_none_when_unbounded():
    """0 (the default) is 'no budget' -- represented by no FitBudget at all, so every
    consumer's check is `budget is not None` rather than a sentinel comparison."""
    assert budget_mod.FitBudget.from_config(_conf(0)) is None
    assert budget_mod.FitBudget.from_config(_conf(None)) is None

    b = budget_mod.FitBudget.from_config(_conf(60), clock=_Clock())
    assert b is not None and b.limit == 60.0


def test_from_config_charges_the_time_since_process_start():
    import time
    b = budget_mod.FitBudget.from_config(_conf(1000), started_at=time.time() - 100.0,
                                         clock=_Clock())
    # ~100 s of process startup is already gone; allow a wide margin for slow CI.
    assert 850.0 < b.remaining() < 901.0


def test_format_duration_matches_the_run_time_line():
    assert budget_mod.format_duration(0) == '0:00:00'
    assert budget_mod.format_duration(59.9) == '0:00:59'
    assert budget_mod.format_duration(3661) == '1:01:01'
    assert budget_mod.format_duration(-5) == '0:00:00'


def _conf(wall_time_fit, fit_type='de'):
    cfg = object.__new__(config.Configuration)
    cfg.config = {'wall_time_fit': wall_time_fit, 'fit_type': fit_type}
    return cfg


# --------------------------------------------------------------------------- #
# The configuration surface
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('value', [0, 1, 10800])
def test_valid_budgets_are_accepted_and_normalized_to_int(value):
    cfg = _conf(value)
    cfg._check_wall_time_fit()
    assert cfg.config['wall_time_fit'] == value
    assert isinstance(cfg.config['wall_time_fit'], int)


@pytest.mark.parametrize('value', [-1, -10800, 'an hour', 3.5, True])
def test_a_budget_that_is_not_a_nonnegative_integer_is_refused(value):
    with pytest.raises(printing.PybnfError, match='wall_time_fit'):
        _conf(value)._check_wall_time_fit()


def test_a_fit_type_that_cannot_honor_the_budget_refuses_it_rather_than_ignoring_it():
    """``hmc`` runs its own in-process sampling loop, so a deadline set on it would
    never fire. Refuse, with the reason and a remedy (#527's rule: a refusal states
    what is wrong AND what to do)."""
    with pytest.raises(printing.PybnfError) as exc:
        _conf(3600, fit_type='hmc')._check_wall_time_fit()
    assert 'hmc' in exc.value.message
    assert 'num_samples' in exc.value.message      # the hint survives into the user message
    assert 'wall_time_fit' in exc.value.log_message


def test_a_fit_type_that_cannot_honor_the_budget_is_unaffected_when_none_is_asked_for():
    _conf(0, fit_type='hmc')._check_wall_time_fit()  # no budget named -> nothing to refuse


def test_model_checking_strips_the_budget():
    """A check is one evaluation of given parameters, not a search, so there is no run
    to budget; the key is dropped alongside refine/bootstrap rather than pretending to
    bound something."""
    d = {'wall_time_fit': 3600, 'refine': 1, 'bootstrap': 2, 'population_size': 10}
    config.Configuration._strip_uncheckable_keys(d)
    assert 'wall_time_fit' not in d
    assert d == {'population_size': 10}


# --------------------------------------------------------------------------- #
# End to end: a real optimizer, a real Configuration, a budgeted stop
# --------------------------------------------------------------------------- #
class _TickingDE(algorithms.DifferentialEvolution):
    """Differential Evolution that charges a fixed slice of wall clock to the budget
    per completed simulation, so a deadline can be driven deterministically instead
    of by sleeping. Module-level (not a closure) because a fit pickles itself for its
    periodic backup."""

    seconds_per_result = 100.0

    def got_result(self, res):
        self.clock.advance(self.seconds_per_result)
        return super().got_result(res)


def test_a_budgeted_fit_stops_on_the_deadline_and_finalizes(tmp_path, monkeypatch):
    """The whole path, with only dask faked (see integration_harness): a real
    Differential Evolution fit of a 2-D Gaussian, given a max_iterations it could
    never reach, is ended by the wall-time budget alone -- and still writes the
    results a converged fit writes.

    This is the benchmark protocol the key exists for (Grein et al. 2026): a fit
    given a fixed wall-time allocation, whose result is scored as "the best
    objective reached within budget B".
    """
    H.install(monkeypatch)
    mean, var = [2.0, -1.0], [1.0, 1.0]
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(mean, var))
    conf = H.make_config(tmp_path, 'de', tgt, exp, n_params=2,
                         population_size=8, max_iterations=10 ** 6,  # unreachable
                         wall_time_fit=3600)
    assert conf.config['wall_time_fit'] == 3600  # the key survives the config pipeline

    alg = _TickingDE(conf)
    clock = _Clock()
    alg.clock = clock
    alg.budget = budget_mod.FitBudget(3600, clock=clock)
    H.drive(alg)

    # It stopped on the deadline, not on its own criterion or its iteration cap.
    assert clock.t >= 3600.0
    assert alg.stop_reason is not None and 'Wall-time budget reached' in alg.stop_reason
    assert alg.max_iterations == 10 ** 6   # nowhere near exhausted
    # ... having made real progress toward the known mode, and recorded it the usual way.
    assert len(alg.trajectory) > 0
    assert np.all(np.abs(H.best_params(alg, 2) - mean) < 5.0)
    res_dir = os.path.join(str(tmp_path), 'out', 'Results')
    assert os.path.isfile(os.path.join(res_dir, 'sorted_params_final.txt'))
    assert os.path.isfile(os.path.join(res_dir, 'stop_reason.txt'))


# --------------------------------------------------------------------------- #
# The phases after the fit: refine, and bootstrap replicates
#
# The budget bounds the *run*, not the fit alone, so neither of these starts once
# it is spent -- and while it is live, they run under the same deadline object
# rather than each getting a fresh one.
# --------------------------------------------------------------------------- #
def _expired(limit=60.0):
    """A budget whose whole limit was already spent before it was handed over."""
    return budget_mod.FitBudget(limit, elapsed=limit, clock=_Clock())


def _finished_de_fit(tmp_path, monkeypatch, **overrides):
    """A completed 2-D Gaussian DE fit, ready for the post-fit phases."""
    H.install(monkeypatch)
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec([2.0, -1.0], [1.0, 1.0]))
    conf = H.make_config(tmp_path, 'de', tgt, exp, n_params=2, population_size=12,
                         max_iterations=15, **overrides)
    alg = algorithms.DifferentialEvolution(conf)
    H.drive(alg)
    return conf, alg


def test_a_spent_budget_skips_the_refine(tmp_path, monkeypatch):
    """A refine is new work. The fit has already finalized, so the run's outputs are
    complete -- just unpolished."""
    import types
    from pybnf import pybnf as pybnf_main

    conf, alg = _finished_de_fit(tmp_path, monkeypatch, refine=1)
    alg.budget = _expired()
    ran = []
    monkeypatch.setattr(algorithms.SimplexAlgorithm, 'run',
                        lambda self, client, resume=None, debug=False: ran.append(self))
    before = alg.trajectory.best_score()

    pybnf_main._refine_best_fit(conf, alg, types.SimpleNamespace(client=H.FakeClient()),
                                debug=False)

    assert ran == []
    assert alg.trajectory.best_score() == before


def test_a_live_budget_is_handed_to_the_refiner(tmp_path, monkeypatch):
    """One deadline bounds the whole run: the refiner inherits the fit's budget
    object rather than starting a fresh one."""
    import types
    from pybnf import pybnf as pybnf_main

    conf, alg = _finished_de_fit(tmp_path, monkeypatch, refine=1)
    alg.budget = budget_mod.FitBudget(60.0, clock=_Clock())
    seen = []
    monkeypatch.setattr(algorithms.SimplexAlgorithm, 'run',
                        lambda self, client, resume=None, debug=False: seen.append(self.budget))

    pybnf_main._refine_best_fit(conf, alg, types.SimpleNamespace(client=H.FakeClient()),
                                debug=False)

    assert seen == [alg.budget]


class _BootstrapStub:
    """Stands in for the algorithm _run_bootstrapping drives; every replicate entry
    point raises, so a test can tell whether the budget guard stopped the loop before
    it or let it through."""

    def __init__(self, budget):
        self.budget = budget
        self.bootstrap_number = None
        self.bootstrap_attempt = 0

    def reset(self, bootstrap):
        raise AssertionError('started bootstrap replicate %d' % bootstrap)


def _bootstrap_conf(tmp_path, n=3):
    import types
    return types.SimpleNamespace(
        config={'output_dir': str(tmp_path), 'bootstrap': n, 'bootstrap_max_obj': 10.0,
                'num_to_output': 100},
        variables=[])


def test_a_spent_budget_starts_no_further_bootstrap_replicate(tmp_path):
    """A replicate is a whole fit's worth of new work, so the run stops with the
    replicates already accepted (each was appended to
    bootstrapped_parameter_sets.txt as it finished)."""
    from pybnf import pybnf as pybnf_main

    alg = _BootstrapStub(_expired())

    pybnf_main._run_bootstrapping(_bootstrap_conf(tmp_path), alg, cluster=None, debug=False)
    # returns without ever calling alg.reset -> no replicate was started


def test_a_live_budget_does_not_stop_bootstrapping(tmp_path):
    """The guard is the budget's, not a general off switch: with time left, the loop
    proceeds into the replicate as before."""
    from pybnf import pybnf as pybnf_main

    alg = _BootstrapStub(budget_mod.FitBudget(60.0, clock=_Clock()))

    with pytest.raises(AssertionError, match='started bootstrap replicate 0'):
        pybnf_main._run_bootstrapping(_bootstrap_conf(tmp_path), alg, cluster=None, debug=False)
