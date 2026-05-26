from .context import data, algorithms, pset, config, printing, raises
from os import mkdir
from shutil import rmtree
from copy import deepcopy


class TestScatter:
    @classmethod
    def setup_class(cls):
        cls.data1s = [
            '# time    v1_result    v2_result    v3_result\n',
            ' 1 2.1   3.1   6.1\n',
        ]
        cls.d1s = data.Data()
        cls.d1s.data = cls.d1s._read_file_lines(cls.data1s, '\s+')

        # Note mutation_rate is set to 1.0 because for tests with few params, with a lower mutation_rate might randomly
        # create a duplicate parameter set, causing the "not in individuals" tests to fail.
        cls.config = config.Configuration({
            'population_size': 7, 'max_iterations': 20, 'fit_type': 'ss',
            ('uniform_var', 'v1__FREE'): [0, 10], ('uniform_var', 'v2__FREE'): [0, 10], ('uniform_var', 'v3__FREE'): [0, 10],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'}, 'initialization': 'lh',
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp']})

        cls.config_path = 'bngl_files/parabola.conf'
        mkdir('test_ss_output')
        mkdir('test_ss_output/Simulations')
        mkdir('test_ss_output/Results')
        mkdir('bnf_out')

    @classmethod
    def teardown_class(cls):
        rmtree('bnf_out')
        rmtree('test_ss_output')

    def test_start(self):
        ss = algorithms.ScatterSearch(deepcopy(self.config))
        ss.start_run()
        assert len(ss.refs) == 0
        assert len(ss.pending) == 30
        assert len(ss.reserve) == 20

    def test_updates(self):
        ss = algorithms.ScatterSearch(deepcopy(self.config))
        start_params = ss.start_run()
        ss.iteration = 1  # Avoid triggering output on iter 0.

        iter2run = []
        for i in range(30):
            res = algorithms.Result(start_params[i], self.data1s, start_params[i].name)
            res.score = 42.
            torun = ss.got_result(res)
            if i < 29:
                assert torun == []
            elif i == 29:
                assert len(torun) == 42  #pop_size*(pop_size-1)
                iter2run = torun
        assert len(ss.refs) == 7
        for p in ss.refs:
            assert len(ss.received[p[0]]) == 0

        # 2nd iteration
        i = 0
        out = ss.refs[3]
        notout = ss.refs[4]
        newref = None
        for pi in range(7):
            for hi in range(7):
                if pi == hi:
                    continue
                ps = iter2run[i]
                i += 1
                res = algorithms.Result(ps, self.data1s, ps.name)
                if pi == 3 and hi == 5:
                    res.score = 37.
                    newref = (ps, 37.)
                else:
                    res.score = 50.
                ss.got_result(res)

        assert out not in ss.refs
        assert notout in ss.refs
        assert newref in ss.refs
        assert ss.stuckcounter[notout[0]] == 1
        assert ss.stuckcounter[newref[0]] == 0

    def test_exp10(self):
        assert algorithms.exp10(2.) == 100.

    @raises(printing.PybnfError)
    def test_exp10_overflow(self):
        algorithms.exp10(100000.)


# --------------------------------------------------------------------------- #
# ScatterSearch construction, reset, and the got_result decision logic: the
# reference-set update (replace / stuck-counter / local-min archival) and the
# Egea-2009 combination that builds new candidates around each reference point.
# The combination's only randomness is add_rand's uniform draw; freezing it to
# the midpoint reduces a candidate to the closed form  pi - alpha*beta*d.
# --------------------------------------------------------------------------- #
import numpy as np
import numpy.testing as npt
import re as _re

SS_NAMES = ('v1__FREE', 'v2__FREE', 'v3__FREE')


def _ss_pset(vals):
    return pset.PSet([pset.FreeParameter(n, 'uniform_var', -1000., 1000., v)
                      for n, v in zip(SS_NAMES, vals)])


def _ss_config(tmp_path, **over):
    base = {
        'population_size': 3, 'max_iterations': 100, 'fit_type': 'ss',
        ('uniform_var', 'v1__FREE'): [-1000, 1000], ('uniform_var', 'v2__FREE'): [-1000, 1000],
        ('uniform_var', 'v3__FREE'): [-1000, 1000],
        'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'},
        'bngl_files/parabola.bngl': ['bngl_files/par1.exp'],
        'output_dir': str(tmp_path / 'ss_out')}
    base.update(over)
    return config.Configuration(base)


class TestScatterSearchConstruction:

    def test_population_floored_to_three(self, tmp_path):
        """Oracle (minimum population): scatter search needs >= 3 references; a
        smaller population is bumped to 3."""
        assert algorithms.ScatterSearch(_ss_config(tmp_path, population_size=2)).popsize == 3

    def test_explicit_init_size_floored_to_population(self, tmp_path):
        """Oracle (init_size >= population_size): an explicit init_size below the
        population size is raised to the population size."""
        ss = algorithms.ScatterSearch(_ss_config(tmp_path, population_size=5, init_size=2))
        assert ss.init_size == 5

    def test_default_init_size_floored_to_population(self, tmp_path):
        """Oracle (default init_size): the default 10*n_vars is used, but never
        below the population size. With 3 vars (default 30) and population 40, the
        default is floored up to 40."""
        ss = algorithms.ScatterSearch(_ss_config(tmp_path, population_size=40))
        assert ss.init_size == 40

    def test_explicit_reserve_size_used(self, tmp_path):
        """Oracle (reserve_size config): an explicit reserve_size overrides the
        default (which is max_iterations)."""
        ss = algorithms.ScatterSearch(_ss_config(tmp_path, reserve_size=5))
        assert ss.reserve_size == 5

    def test_zero_reserve_size_gives_empty_reserve(self, tmp_path):
        """Oracle (reserve disabled): reserve_size <= 0 leaves the reserve empty
        after start_run (the random-restart pool is disabled)."""
        ss = algorithms.ScatterSearch(_ss_config(tmp_path, reserve_size=0))
        ss.start_run()
        assert ss.reserve == []

    def test_reset_clears_state(self, tmp_path):
        """Oracle (reset invariant): reset() returns the reference set, pending,
        received, counters and archive to their constructed empty state."""
        ss = algorithms.ScatterSearch(_ss_config(tmp_path))
        ss.start_run()
        ss.iteration = 9
        ss.local_mins = [(_ss_pset((1., 1., 1.)), 0.5)]
        ss.reset()
        assert ss.refs == [] and ss.pending == {} and ss.received == {}
        assert ss.iteration == 0 and ss.local_mins == [] and ss.reserve == []

    def test_non_lh_initialization(self, tmp_path):
        """Oracle (initialization branch): initialization != 'lh' draws init_size
        independent random psets as the starting pool."""
        ss = algorithms.ScatterSearch(_ss_config(tmp_path, init_size=8, initialization='rand'))
        out = ss.start_run()
        assert len(out) == 8 and len(ss.pending) == 8

    def test_get_backup_every_formula(self, tmp_path):
        """Oracle (backup cadence): scatter search runs pop*(pop-1) sims per
        iteration, so get_backup_every = backup_every * pop * (pop-1) * smoothing."""
        ss = algorithms.ScatterSearch(_ss_config(tmp_path, population_size=4))
        cfg = ss.config.config
        expected = cfg['backup_every'] * 4 * 3 * cfg['smoothing']
        assert ss.get_backup_every() == expected


class TestScatterSearchUpdate:

    d1s = data.Data()
    d1s.data = d1s._read_file_lines(
        ['# time v1_result v2_result v3_result\n', ' 1 2.1 3.1 6.1\n'], r'\s+')

    def _primed(self, tmp_path, refvals, scores, child_score, **over):
        """Build a scatter search whose reference set is exactly the given
        (refvals, scores) and whose pending children (all scoring child_score)
        are one result away from completing an iteration."""
        ss = algorithms.ScatterSearch(_ss_config(tmp_path, **over))
        refs = [(_ss_pset(v), s) for v, s in zip(refvals, scores)]
        ss.refs = list(refs)
        ss.stuckcounter = {r[0]: 0 for r in refs}
        ss.iteration = 1
        # All references but the last have already received a (non-improving) child;
        # the final pending child completes the generation when its result arrives.
        ss.received = {r[0]: [(_ss_pset([x + 0.001 for x in v]), child_score)]
                       for (r, v) in zip(refs[:-1], refvals[:-1])}
        ss.received[refs[-1][0]] = []
        last_child = _ss_pset([x + 0.002 for x in refvals[-1]])
        last_child.name = 'pendingchild'
        ss.pending = {last_child: refs[-1][0]}
        return ss, refs, last_child

    def test_combination_matches_egea_closed_form(self, tmp_path, monkeypatch):
        """Oracle (Egea-2009 combination, midpoint draw): each candidate for
        parent pi using helper hi draws uniformly in
        [pi - d(1+alpha*beta), pi + d(1-alpha*beta)] with d = ref[hi]-ref[pi],
        alpha = sign(hi-pi), beta = (|hi-pi|-1)/(pop-2). With the draw frozen to
        the midpoint this collapses to  new = pi - alpha*beta*d, i.e. adjacent
        references (beta=0) reproduce the parent exactly. Pins alpha=sign(hi-pi),
        the beta denominator, and the d direction. Children score high so the
        reference set is unchanged going into the combination."""
        refvals = [(10., 20., 30.), (40., 50., 60.), (12., 24., 33.)]
        ss, refs, last_child = self._primed(tmp_path, refvals, [1., 2., 3.], child_score=100.,
                                            population_size=3)
        monkeypatch.setattr(np.random, 'uniform', lambda lb, ub: (lb + ub) / 2.)
        res = algorithms.Result(last_child, self.d1s, last_child.name); res.score = 100.
        query = ss.got_result(res)

        refsvals = [r[0] for r in ss.refs]            # references are unchanged & sorted by score
        assert [s for _, s in ss.refs] == [1., 2., 3.]
        assert query, 'a generation should have been proposed'
        for q in query:
            pi, hi = (int(x) for x in _re.search(r'p(\d+)h(\d+)', q.name).groups())
            for k, nm in enumerate(SS_NAMES):
                d = refsvals[hi][nm] - refsvals[pi][nm]
                alpha = np.sign(hi - pi)
                beta = (abs(hi - pi) - 1) / (3 - 2)
                npt.assert_allclose(q[nm], refsvals[pi][nm] - alpha * beta * d, atol=1e-9)

    def test_better_child_replaces_reference_and_resets_counter(self, tmp_path):
        """Oracle (reference update): a child strictly better than its parent
        reference replaces it and resets that slot's stuck counter to 0, while a
        non-improving reference's counter increments."""
        refvals = [(10., 20., 30.), (40., 50., 60.), (12., 24., 33.)]
        ss, refs, last_child = self._primed(tmp_path, refvals, [10., 20., 30.],
                                            child_score=100., population_size=3,
                                            local_min_limit=5)
        # Give the worst reference (refs[2], score 30) a winning child.
        winner = _ss_pset((5., 5., 5.))
        ss.received[refs[2][0]] = [(winner, 1.0)]
        res = algorithms.Result(last_child, self.d1s, last_child.name); res.score = 100.
        ss.got_result(res)
        assert (winner, 1.0) in ss.refs                 # replaced parent
        assert ss.stuckcounter[winner] == 0
        assert ss.stuckcounter[refs[0][0]] == 1         # non-improving slot incremented

    def test_stuck_reference_is_archived_and_replaced_from_reserve(self, tmp_path):
        """Oracle (local-minimum archival): a reference that fails to improve for
        local_min_limit generations is moved to local_mins (kept sorted, capped at
        pop) and replaced by a reserve pset with score inf. With local_min_limit=1
        every non-improving reference is archived this generation."""
        refvals = [(10., 20., 30.), (40., 50., 60.), (12., 24., 33.)]
        ss, refs, last_child = self._primed(tmp_path, refvals, [1., 2., 3.],
                                            child_score=100., population_size=3,
                                            local_min_limit=1)
        reserve = [_ss_pset((100., 100., 100.)), _ss_pset((200., 200., 200.)),
                   _ss_pset((300., 300., 300.))]
        ss.reserve = list(reserve)
        res = algorithms.Result(last_child, self.d1s, last_child.name); res.score = 100.
        ss.got_result(res)
        archived = [m[0] for m in ss.local_mins]
        for r in refs:
            assert r[0] in archived                     # all three stuck refs archived
        assert len(ss.local_mins) <= ss.popsize
        assert all(np.isinf(s) for _, s in ss.refs)     # every slot refilled from reserve
        assert ss.reserve == []                         # reserve drained

    def test_stuck_reference_falls_back_to_random_when_reserve_empty(self, tmp_path):
        """Oracle (empty-reserve replacement): when a stuck reference must be
        replaced but the reserve pool is exhausted, the slot is refilled with a
        fresh random pset (score inf) drawn from inside the box rather than from
        the reserve."""
        refvals = [(10., 20., 30.), (40., 50., 60.), (12., 24., 33.)]
        ss, refs, last_child = self._primed(tmp_path, refvals, [1., 2., 3.],
                                            child_score=100., population_size=3,
                                            local_min_limit=1)
        ss.reserve = []                                 # force the random_pset() fallback
        res = algorithms.Result(last_child, self.d1s, last_child.name); res.score = 100.
        ss.got_result(res)
        assert all(np.isinf(s) for _, s in ss.refs)     # all stuck slots refilled
        for r in ss.refs:                                # fresh psets lie inside the box
            for nm in SS_NAMES:
                assert -1000. <= r[0][nm] <= 1000.

    def test_stop_at_max_iterations(self, tmp_path):
        """Oracle (termination): completing the generation that brings iteration
        up to max_iterations returns 'STOP'."""
        refvals = [(10., 20., 30.), (40., 50., 60.), (12., 24., 33.)]
        ss, refs, last_child = self._primed(tmp_path, refvals, [1., 2., 3.],
                                            child_score=100., population_size=3,
                                            max_iterations=2)
        ss.iteration = 1                                # next completion -> iteration 2 == max
        res = algorithms.Result(last_child, self.d1s, last_child.name); res.score = 100.
        assert ss.got_result(res) == 'STOP'


class TestScatterSearchRound1Init:

    d1s = data.Data()
    d1s.data = d1s._read_file_lines(
        ['# time v1_result v2_result v3_result\n', ' 1 2.1 3.1 6.1\n'], r'\s+')

    def test_round1_keeps_best_half_and_seeds_counters(self, tmp_path):
        """Oracle (round_1_init selection): the first reference set is the top
        ceil(pop/2) initial points by score plus floor(pop/2) drawn at random from
        the rest. So the ceil(pop/2) best-scoring inits are always references, the
        set has exactly pop entries, and every stuck counter starts at 0."""
        ss = algorithms.ScatterSearch(_ss_config(tmp_path, population_size=7, init_size=8))
        start = ss.start_run()
        ss.iteration = 1                                # avoid iter-0 output side effect
        for i, ps in enumerate(start):
            res = algorithms.Result(ps, self.d1s, ps.name); res.score = float(i)
            ss.got_result(res)
        topcount = int(np.ceil(7 / 2.))                 # = 4
        ref_psets = [r[0] for r in ss.refs]
        assert len(ss.refs) == 7
        for i in range(topcount):                       # the 4 best-scoring inits are kept
            assert start[i] in ref_psets
        assert all(c == 0 for c in ss.stuckcounter.values())
