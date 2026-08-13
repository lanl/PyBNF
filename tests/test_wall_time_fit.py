"""The fit's total wall-clock budget: ``wall_time_fit`` (#529, ADR-0093), and the
slice of it reserved for the refine (``wall_time_refine_frac``, #564, ADR-0107).

Two layers are covered here — the stopwatch (:mod:`pybnf.budget`) and the
configuration surface that admits or refuses the key. The *behavior* the budget
drives (the run loop stopping and finalizing anyway) lives with the rest of the
run-loop orchestration tests, in ``test_run_loop.py``; the artifacts a budgeted
run leaves behind are in ``test_method_chain.py``.

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


def _conf(wall_time_fit, fit_type='de', **extra):
    cfg = object.__new__(config.Configuration)
    cfg.config = {'wall_time_fit': wall_time_fit, 'fit_type': fit_type}
    cfg.config.update(extra)
    return cfg


# --------------------------------------------------------------------------- #
# The refine's reserved slice of the budget (#564, ADR-0107)
#
# `refine = 1` asks for a METHOD -- search globally, then polish locally. A
# wall-clock-budgeted search has no reason to leave anything behind, so unless the
# budget holds a tail back the polish essentially never runs.
# --------------------------------------------------------------------------- #
def test_a_reserve_bounds_the_phase_while_the_limit_still_bounds_the_run():
    clock = _Clock()
    b = budget_mod.FitBudget(100.0, clock=clock, reserve=10.0)

    assert b.remaining() == 90.0 and not b.expired()   # the search's share, not the run's
    clock.advance(90.0)
    assert b.expired() and b.remaining() == 0.0        # the search is done at 90 s
    with budget_mod.spend_reserve(b):
        assert not b.expired() and b.remaining() == 10.0   # ... the refine's tail is not
    assert b.expired()   # and it is put back, for the next bootstrap replicate's search


def test_a_search_that_converges_early_hands_its_leftovers_to_the_refine():
    """The reserve is a floor under the refine, not a cap on it: the refine gets the
    run's whole remaining time."""
    clock = _Clock()
    b = budget_mod.FitBudget(100.0, clock=clock, reserve=10.0)
    clock.advance(20.0)   # the search stopped on its own criterion, 70 s early

    with budget_mod.spend_reserve(b):
        assert b.remaining() == 80.0


def test_a_reserve_cannot_swallow_more_than_the_budget():
    b = budget_mod.FitBudget(10.0, clock=_Clock(), reserve=99.0)
    assert b.reserve == 10.0 and b.expired()


def test_spend_reserve_is_a_no_op_on_an_unbudgeted_run():
    with budget_mod.spend_reserve(None) as b:
        assert b is None


def test_spend_reserve_puts_the_reserve_back_even_when_the_phase_raises():
    b = budget_mod.FitBudget(100.0, clock=_Clock(), reserve=10.0)
    with pytest.raises(RuntimeError):
        with budget_mod.spend_reserve(b):
            raise RuntimeError('the refine blew up')
    assert b.reserve == 10.0


@pytest.mark.parametrize('overrides,expected', [
    ({}, 150.0),                                        # the default tenth of the budget
    ({'wall_time_refine_frac': 0.5}, 750.0),
    ({'wall_time_refine_frac': 0.0}, 0.0),              # opt out: the pre-#564 split
    ({'refine': 0}, 0.0),                               # no refine to protect
    ({'wall_time_fit': 0}, 0.0),                        # no budget to divide
    ({'refine_method': 'de'}, 0.0),                     # == fit_type: the refine is skipped
])
def test_the_reserve_is_taken_only_when_a_refine_will_actually_run(overrides, expected):
    conf = {'wall_time_fit': 1500, 'fit_type': 'de', 'refine': 1, 'refine_method': 'sim',
            'wall_time_refine_frac': 0.1}
    conf.update(overrides)
    assert budget_mod.refine_reserve_seconds(conf) == expected


def test_from_config_sizes_the_reserve_it_builds_the_budget_with():
    b = budget_mod.FitBudget.from_config(
        _conf(1500, refine=1, refine_method='sim', wall_time_refine_frac=0.1),
        clock=_Clock())
    assert b.limit == 1500.0 and b.reserve == 150.0 and b.remaining() == 1350.0


def test_a_fit_that_asks_for_no_refine_gets_the_ADR_0093_budget_exactly():
    """Nothing about an unrefined run changes: the whole budget is the search's."""
    b = budget_mod.FitBudget.from_config(_conf(1500), clock=_Clock())
    assert b.reserve == 0.0 and b.remaining() == 1500.0


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


@pytest.mark.parametrize('value', [0, 0.1, 0.5, 0.999])
def test_valid_refine_fractions_are_accepted_and_normalized_to_float(value):
    cfg = _conf(1500, wall_time_refine_frac=value)
    cfg._check_wall_time_refine_frac()
    assert cfg.config['wall_time_refine_frac'] == float(value)
    assert isinstance(cfg.config['wall_time_refine_frac'], float)


@pytest.mark.parametrize('value', [-0.1, 1, 1.0, 1.5, 'a tenth', True])
def test_a_refine_fraction_outside_zero_to_one_is_refused(value):
    """1 is excluded with the negatives: reserving the whole budget for the polish
    would leave the search nothing at all, which is not what any conf means."""
    with pytest.raises(printing.PybnfError, match='wall_time_refine_frac'):
        _conf(1500, wall_time_refine_frac=value)._check_wall_time_refine_frac()


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


def test_a_budgeted_fit_stops_early_so_the_refine_it_asked_for_can_run(tmp_path, monkeypatch):
    """#564, the reported defect: `cmaes` + `refine = 1, refine_method = gntr` +
    `wall_time_fit` ran plain `cmaes` in 15 of 15 benchmark runs. The search filled the
    budget -- as a wall-clock-budgeted search always will -- and the polish, being new
    work, never started.

    With a reserve the search stops at its own share and the refine runs on the rest,
    so the executed method chain is the requested one.
    """
    import types
    from pybnf import pybnf as pybnf_main

    H.install(monkeypatch)
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec([2.0, -1.0], [1.0, 1.0]))
    conf = H.make_config(tmp_path, 'de', tgt, exp, n_params=2, population_size=8,
                         max_iterations=10 ** 6,   # unreachable: only the clock can stop it
                         wall_time_fit=3600, refine=1, refine_method='sim',
                         wall_time_refine_frac=0.1)

    alg = _TickingDE(conf)
    clock = _Clock()
    alg.clock = clock
    alg.budget = budget_mod.FitBudget.from_config(conf, clock=clock)
    assert alg.budget.reserve == 360.0
    H.drive(alg)

    # The search stopped on the deadline, but on ITS deadline: short of wall_time_fit
    # by the reserve, and saying so.
    assert 3240.0 <= clock.t < 3600.0
    assert 'reserved for the refine' in alg.stop_reason

    left = []
    monkeypatch.setattr(algorithms.SimplexAlgorithm, 'run',
                        lambda self, client, resume=None, debug=False: left.append(self.budget.remaining()))
    pybnf_main._refine_best_fit(conf, alg, types.SimpleNamespace(client=H.FakeClient()),
                                debug=False)

    assert left and left[0] > 0.0        # the refine ran, with time to spend
    assert alg.budget.reserve == 360.0   # and the split is intact for a bootstrap replicate


def test_opting_out_of_the_reserve_restores_the_old_first_come_first_served_split(tmp_path, monkeypatch):
    """`wall_time_refine_frac = 0` is the pre-#564 behavior, kept reachable for anyone
    who wants every second spent searching -- but now it is a choice the conf states,
    not the only thing that could happen."""
    import types
    from pybnf import pybnf as pybnf_main

    H.install(monkeypatch)
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec([2.0, -1.0], [1.0, 1.0]))
    conf = H.make_config(tmp_path, 'de', tgt, exp, n_params=2, population_size=8,
                         max_iterations=10 ** 6, wall_time_fit=3600, refine=1,
                         refine_method='sim', wall_time_refine_frac=0.0)

    alg = _TickingDE(conf)
    clock = _Clock()
    alg.clock = clock
    alg.budget = budget_mod.FitBudget.from_config(conf, clock=clock)
    assert alg.budget.reserve == 0.0
    H.drive(alg)
    assert clock.t >= 3600.0

    ran = []
    monkeypatch.setattr(algorithms.SimplexAlgorithm, 'run',
                        lambda self, client, resume=None, debug=False: ran.append(self))
    pybnf_main._refine_best_fit(conf, alg, types.SimpleNamespace(client=H.FakeClient()),
                                debug=False)
    assert ran == []


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
