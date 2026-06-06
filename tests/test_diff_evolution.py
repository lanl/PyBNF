from .context import data, algorithms, pset, config, printing

from types import SimpleNamespace
import shutil
import numpy as np
import numpy.testing as npt
import pytest


class TestDiffEvolution:
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
            'population_size': 20, 'max_iterations': 20, 'islands': 2, 'migrate_every': 3, 'num_to_migrate': 2,
            'mutation_rate': 1.0, 'fit_type': 'de',
            ('uniform_var', 'v1__FREE'): [0, 10], ('uniform_var', 'v2__FREE'): [0, 10], ('uniform_var', 'v3__FREE'): [0, 10],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'}, 'initialization': 'lh',
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp'],
            'output_dir': 'test_init'})

    @classmethod
    def teardown_class(cls):
        shutil.rmtree('test_init')

    def test_start(self):
        de = algorithms.DifferentialEvolution(self.config)
        assert de.num_per_island == 10
        start_params = de.start_run()
        assert len(start_params) == 20
        assert len(de.individuals) == 2
        assert len(de.individuals[0]) == 10
        assert de.waiting_count == [10, 10]

    def test_updates(self):
        de = algorithms.DifferentialEvolution(self.config)
        start_params = de.start_run()

        for i in range(9):
            res = algorithms.Result(start_params[i], self.data1s, start_params[i].name)
            res.score = 42.
            torun = de.got_result(res)
            assert torun == []
        # Finish island 1 iter 0, should get some new params.
        res = algorithms.Result(start_params[9], self.data1s, start_params[9].name)
        res.score = 42.
        torun = de.got_result(res)
        assert len(torun) == 10
        next_params = torun
        assert de.iter_num == [1, 0]
        for i in range(10, 20):
            res = algorithms.Result(start_params[i], self.data1s, start_params[i].name)
            res.score = 150.
            torun = de.got_result(res)
            next_params += torun
        # End of iteration 0
        assert de.iter_num == [1, 1]

        params_gen2 = []
        for i in range(20):
            res = algorithms.Result(next_params[i], self.data1s, next_params[i].name)
            res.score = max(1., i ** 2)
            if i < 10:
                assert de.island_map[next_params[i]] == (0, i)
            else:
                assert de.island_map[next_params[i]] == (1, i-10)
            torun = de.got_result(res)
            # Replace if i**2 is better than previous value
            if i <= 6:
                assert next_params[i] == de.individuals[0][i]
            elif 7 <= i <= 9:
                assert start_params[i] == de.individuals[0][i]
            elif 10 <= i <= 12:
                assert next_params[i] == de.individuals[1][i-10]
            elif 12 < i:
                assert start_params[i] == de.individuals[1][i-10]
            if i == 9 or i == 19:
                assert len(torun) == 10
            else:
                assert len(torun) == 0
            params_gen2 += torun
        # End of iteration 1
        assert de.iter_num == [2, 2]

        # After iteration 2, migration will trigger
        params_gen3 = []
        for i in range(10):
            res = algorithms.Result(params_gen2[i], self.data1s, params_gen2[i].name)
            res.score = 9999.
            torun = de.got_result(res)
            params_gen3 += torun
        assert de.migration_ready == [1, 0]
        assert de.migration_done == [0, 0]
        assert len(de.migration_indices[1]) == 2
        assert len(de.migration_perms[1]) == 2
        assert len(de.migration_transit[1][0]) == 2
        assert len(de.migration_transit[1][1]) == 0

        for i in range(10, 20):
            res = algorithms.Result(params_gen2[i], self.data1s, params_gen2[i].name)
            res.score = 9999.
            torun = de.got_result(res)
            params_gen3 += torun

        assert de.migration_ready == [1, 1]
        assert de.migration_done == [0, 1]


# --------------------------------------------------------------------------- #
# DifferentialEvolutionBase.new_individual: the DE mutation/recombination math.
# new_individual builds a trial vector base + F*(donor differences) for the
# mutated dimensions. We freeze the two stochastic inputs -- the donor index
# draw (np.random.choice) and the per-dimension crossover coin (np.random.random)
# -- so the proposal reduces to a closed form we can check exactly. Wide bounds
# keep reflection out of the picture.
# --------------------------------------------------------------------------- #
NAMES = ('v1__FREE', 'v2__FREE', 'v3__FREE')


def _wide_pset(vals):
    return pset.PSet([pset.FreeParameter(n, 'uniform_var', -100., 100., v)
                      for n, v in zip(NAMES, vals)])


def _ade_config(tmp_path, **over):
    base = {
        'population_size': 6, 'max_iterations': 100, 'mutation_rate': 1.0,
        'mutation_factor': 0.5, 'de_strategy': 'rand1', 'fit_type': 'ade',
        ('uniform_var', 'v1__FREE'): [-100, 100], ('uniform_var', 'v2__FREE'): [-100, 100],
        ('uniform_var', 'v3__FREE'): [-100, 100],
        'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'},
        'bngl_files/parabola.bngl': ['bngl_files/par1.exp'],
        'output_dir': str(tmp_path / 'ade_out')}
    base.update(over)
    return config.Configuration(base)


class TestNewIndividual:

    def _alg(self, tmp_path, **over):
        return algorithms.AsynchronousDifferentialEvolution(_ade_config(tmp_path, **over))

    def test_invalid_strategy_raises(self, tmp_path):
        """Oracle (documented guard): de_strategy must be one of the six named
        strategies; anything else raises PybnfError."""
        with pytest.raises(printing.PybnfError):
            self._alg(tmp_path, de_strategy='wander7')

    def test_rand1_is_base_plus_F_times_donor_diff(self, tmp_path, monkeypatch):
        """Oracle (DE/rand/1 closed form): with picks = [0, 1, 2] and every
        crossover coin firing (mutation_rate=1), the mutant is exactly
        base + F*(donor_a - donor_b) = ind[0] + F*(ind[1] - ind[2]) per dimension.
        Pins the donor difference (a-b, not b-a) and the F scaling."""
        alg = self._alg(tmp_path, de_strategy='rand1', mutation_factor=0.5)
        monkeypatch.setattr(alg, 'rng', SimpleNamespace(
            choice=lambda n, k, replace=False: np.array([0, 1, 2]), random=lambda: 0.0))
        inds = [_wide_pset((1., 2., 3.)), _wide_pset((10., 20., 30.)), _wide_pset((4., 5., 6.))]
        mut = alg.new_individual(inds)
        for name in NAMES:
            expected = inds[0][name] + 0.5 * (inds[1][name] - inds[2][name])
            npt.assert_allclose(mut[name], expected)

    def test_rand2_sums_two_donor_differences(self, tmp_path, monkeypatch):
        """Oracle (DE/rand/2 closed form): a '2' strategy draws 5 donors and the
        mutant is base + F*(a-b) + F*(c-d) = ind[0] + F*(ind1-ind2) + F*(ind3-ind4).
        Pins pickn=5 and the two-difference accumulation."""
        alg = self._alg(tmp_path, de_strategy='rand2', mutation_factor=0.5)
        monkeypatch.setattr(alg, 'rng', SimpleNamespace(
            choice=lambda n, k, replace=False: np.array([0, 1, 2, 3, 4]), random=lambda: 0.0))
        inds = [_wide_pset((1., 2., 3.)), _wide_pset((10., 20., 30.)), _wide_pset((4., 5., 6.)),
                _wide_pset((0., -1., -2.)), _wide_pset((7., 8., 9.))]
        mut = alg.new_individual(inds)
        for name in NAMES:
            expected = (inds[0][name] + 0.5 * (inds[1][name] - inds[2][name])
                        + 0.5 * (inds[3][name] - inds[4][name]))
            npt.assert_allclose(mut[name], expected)

    def test_no_crossover_keeps_base(self, tmp_path, monkeypatch):
        """Oracle (crossover gate): with mutation_rate=0 no coin ever fires, so
        every dimension keeps the base value and the mutant is identical to the
        base individual ind[picks[0]] = ind[0]."""
        alg = self._alg(tmp_path, mutation_rate=0.0)
        monkeypatch.setattr(alg, 'rng', SimpleNamespace(
            choice=lambda n, k, replace=False: np.array([0, 1, 2]), random=lambda: 0.0))
        inds = [_wide_pset((1., 2., 3.)), _wide_pset((10., 20., 30.)), _wide_pset((4., 5., 6.))]
        assert alg.new_individual(inds) == inds[0]

    def test_base_index_inside_picks_is_preserved_uniquely(self, tmp_path, monkeypatch):
        """Oracle (base_index swap path): when base_index is requested and the
        random draw already contains it (picks=[0,1,2], base_index=2), the code
        swaps it out of its slot into picks[0] then sets picks[0]=base_index,
        leaving distinct donors. Result: base=ind[2], donors ind[1] & ind[0], so
        mutant = ind[2] + F*(ind[1] - ind[0])."""
        alg = self._alg(tmp_path, de_strategy='rand1', mutation_factor=0.5)
        monkeypatch.setattr(alg, 'rng', SimpleNamespace(
            choice=lambda n, k, replace=False: np.array([0, 1, 2]), random=lambda: 0.0))
        inds = [_wide_pset((1., 2., 3.)), _wide_pset((10., 20., 30.)), _wide_pset((4., 5., 6.))]
        mut = alg.new_individual(inds, base_index=2)
        for name in NAMES:
            expected = inds[2][name] + 0.5 * (inds[1][name] - inds[0][name])
            npt.assert_allclose(mut[name], expected)

    def test_base_index_outside_picks_overwrites_first(self, tmp_path, monkeypatch):
        """Oracle (base_index overwrite path): when base_index is not among the
        drawn picks ([0,1,3], base_index=2), picks[0] is overwritten with
        base_index, giving base=ind[2], donors ind[1] & ind[3], so
        mutant = ind[2] + F*(ind[1] - ind[3])."""
        alg = self._alg(tmp_path, de_strategy='rand1', mutation_factor=0.5)
        monkeypatch.setattr(alg, 'rng', SimpleNamespace(
            choice=lambda n, k, replace=False: np.array([0, 1, 3]), random=lambda: 0.0))
        inds = [_wide_pset((1., 2., 3.)), _wide_pset((10., 20., 30.)),
                _wide_pset((4., 5., 6.)), _wide_pset((-2., -4., -6.))]
        mut = alg.new_individual(inds, base_index=2)
        for name in NAMES:
            expected = inds[2][name] + 0.5 * (inds[1][name] - inds[3][name])
            npt.assert_allclose(mut[name], expected)

    def test_base_class_methods_return_not_implemented(self, tmp_path):
        """Oracle (abstract API): DifferentialEvolutionBase leaves start_run and
        got_result unimplemented; both return (do not raise) NotImplementedError."""
        base = algorithms.DifferentialEvolutionBase(_ade_config(tmp_path))
        assert isinstance(base.start_run(), NotImplementedError)
        assert isinstance(base.got_result(None), NotImplementedError)


# --------------------------------------------------------------------------- #
# DifferentialEvolution: population sizing, reset, strategy dispatch, and the
# termination/migration bookkeeping that only shows up when got_result is driven.
# --------------------------------------------------------------------------- #
def _de_config(tmp_path, **over):
    base = {
        'population_size': 3, 'max_iterations': 100, 'islands': 1, 'migrate_every': 20,
        'num_to_migrate': 1, 'mutation_rate': 1.0, 'mutation_factor': 0.5, 'fit_type': 'de',
        'de_strategy': 'rand1',
        ('uniform_var', 'v1__FREE'): [-100, 100], ('uniform_var', 'v2__FREE'): [-100, 100],
        ('uniform_var', 'v3__FREE'): [-100, 100],
        'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'},
        'bngl_files/parabola.bngl': ['bngl_files/par1.exp'],
        'output_dir': str(tmp_path / 'de_out')}
    base.update(over)
    return config.Configuration(base)


class TestDifferentialEvolutionPlumbing:

    d1s = data.Data()
    d1s.data = d1s._read_file_lines(
        ['# time v1_result v2_result v3_result\n', ' 1 2.1 3.1 6.1\n'], r'\s+')

    def test_population_floored_to_three_per_island(self, tmp_path):
        """Oracle (minimum population): DE needs >= 3 individuals per island, so a
        too-small population is bumped to 3 per island, single- or multi-island."""
        assert algorithms.DifferentialEvolution(_de_config(tmp_path, population_size=2,
                                                           islands=1)).num_per_island == 3
        assert algorithms.DifferentialEvolution(_de_config(tmp_path, population_size=2,
                                                           islands=2)).num_per_island == 3

    def test_population_reduced_to_divide_islands(self, tmp_path):
        """Oracle (even split): num_per_island = floor(population_size/islands), so
        7 over 2 islands gives 3 per island (one individual dropped)."""
        de = algorithms.DifferentialEvolution(_de_config(tmp_path, population_size=7, islands=2))
        assert de.num_per_island == 3

    def test_single_island_disables_migration(self, tmp_path):
        """Oracle (no migration with one island): migrate_every is forced to inf
        so the migration branch is never entered."""
        de = algorithms.DifferentialEvolution(_de_config(tmp_path, islands=1, migrate_every=5))
        assert de.migrate_every == np.inf

    def test_reset_clears_state(self, tmp_path):
        """Oracle (reset invariant): reset() returns the island bookkeeping to the
        constructed empty/zeroed state."""
        de = algorithms.DifferentialEvolution(_de_config(tmp_path, islands=2, population_size=6))
        de.start_run()
        de.iter_num = [3, 4]
        de.reset()
        assert de.individuals == [] and de.island_map == {}
        assert de.iter_num == [0, 0] and de.migration_ready == [0, 0]
        assert de.migration_transit == {}

    def test_non_lh_initialization(self, tmp_path):
        """Oracle (initialization branch): initialization != 'lh' draws independent
        random psets; start_run lays out islands x per-island individuals."""
        de = algorithms.DifferentialEvolution(_de_config(tmp_path, islands=2, population_size=6,
                                                         initialization='rand'))
        out = de.start_run()
        assert len(out) == 6
        assert len(de.proposed_individuals) == 2 and len(de.proposed_individuals[0]) == 3

    def _run_one_island_generation(self, de, scores):
        """Feed one full generation on a single-island DE; return got_result's
        output from the result that completes the iteration."""
        start = de.start_run()
        out = None
        for ps, sc in zip(start, scores):
            res = algorithms.Result(ps, self.d1s, ps.name)
            res.score = sc
            out = de.got_result(res)
        return out

    @pytest.mark.parametrize("strategy, expected", [
        ('best1', 'argmin'),     # base is always the fittest individual
        ('all1', 'index'),       # base cycles through every index jj
        ('rand1', None),         # base is random (base_index None)
    ])
    def test_strategy_selects_base_index(self, tmp_path, strategy, expected):
        """Oracle (strategy -> base_index dispatch): 'best' strategies pass the
        argmin-fitness index as the proposal base, 'all' strategies pass the
        slot index jj, and plain 'rand' strategies pass None. A spy on
        new_individual records the base_index of every proposal in the next
        generation."""
        de = algorithms.DifferentialEvolution(_de_config(tmp_path, islands=1, population_size=3,
                                                         de_strategy=strategy))
        recorded = []
        orig = de.new_individual
        de.new_individual = lambda inds, base_index=None: recorded.append(base_index) or orig(inds, base_index)
        self._run_one_island_generation(de, [5.0, 3.0, 7.0])  # argmin at index 1
        assert len(recorded) == 3
        if expected == 'argmin':
            assert recorded == [1, 1, 1]
        elif expected == 'index':
            assert recorded == [0, 1, 2]
        else:
            assert recorded == [None, None, None]

    def test_convergence_stop(self, tmp_path):
        """Oracle (convergence criterion): when max(fitness)/min(fitness) <
        1 + stop_tolerance (and min != 0) the island reports 'STOP'. Equal
        fitnesses give ratio exactly 1, below any positive tolerance."""
        de = algorithms.DifferentialEvolution(_de_config(tmp_path, islands=1, population_size=3,
                                                         stop_tolerance=0.002))
        assert self._run_one_island_generation(de, [5.0, 5.0, 5.0]) == 'STOP'

    def test_single_island_stop_at_max_iterations(self, tmp_path):
        """Oracle (termination): a single island that completes max_iterations
        generations stops the whole run."""
        de = algorithms.DifferentialEvolution(_de_config(tmp_path, islands=1, population_size=3,
                                                         max_iterations=1))
        assert self._run_one_island_generation(de, [5.0, 3.0, 7.0]) == 'STOP'

    def test_multi_island_waits_at_max_iterations(self, tmp_path):
        """Oracle (multi-island termination): when one island reaches
        max_iterations but the others have not, it submits no new jobs (returns
        []) rather than stopping the run."""
        de = algorithms.DifferentialEvolution(_de_config(tmp_path, islands=2, population_size=6,
                                                         max_iterations=1))
        start = de.start_run()
        out = None
        for ps in start[:3]:                          # finish only island 0
            res = algorithms.Result(ps, self.d1s, ps.name); res.score = 5.0
            out = de.got_result(res)
        assert out == [] and de.iter_num == [1, 0]

    def test_migration_completes_and_frees_transit(self, tmp_path):
        """Oracle (migration teardown): with migrate_every=1 and two islands, the
        actual exchange happens and, once both islands have completed a given
        migration, its transit/perm/index bookkeeping is deleted. Drives several
        synced generations and checks migration 1's data is gone while both
        islands record having migrated."""
        de = algorithms.DifferentialEvolution(_de_config(tmp_path, islands=2, population_size=6,
                                                         migrate_every=1, num_to_migrate=1,
                                                         max_iterations=50))
        current = de.start_run()
        for _ in range(4):                            # several full generations
            nxt = []
            for i, ps in enumerate(current):
                res = algorithms.Result(ps, self.d1s, ps.name)
                res.score = 10.0 + i                  # distinct, nonzero: no convergence STOP
                out = de.got_result(res)
                if isinstance(out, list):
                    nxt += out
            current = nxt
        assert min(de.migration_done) >= 1            # both islands migrated at least once
        assert 1 not in de.migration_transit          # migration 1 bookkeeping freed
        assert 1 not in de.migration_indices


# --------------------------------------------------------------------------- #
# AsynchronousDifferentialEvolution: no islands; each finished pset immediately
# spawns a replacement at the same index using the current population.
# --------------------------------------------------------------------------- #
class TestAsyncDifferentialEvolution:

    d1s = data.Data()
    d1s.data = d1s._read_file_lines(
        ['# time v1_result v2_result v3_result\n', ' 1 2.1 3.1 6.1\n'], r'\s+')

    def test_population_floored_to_three(self, tmp_path):
        """Oracle (minimum population): a population below 3 is bumped to 3."""
        ade = algorithms.AsynchronousDifferentialEvolution(_ade_config(tmp_path, population_size=2))
        assert ade.population_size == 3

    def test_reset_clears_state(self, tmp_path):
        """Oracle (reset invariant): reset() empties the population and fitness
        lists and zeroes the completion counter."""
        ade = algorithms.AsynchronousDifferentialEvolution(_ade_config(tmp_path, population_size=3))
        ade.start_run()
        ade.sims_completed = 9
        ade.reset()
        assert ade.individuals == [] and ade.fitnesses == [] and ade.sims_completed == 0

    def test_non_lh_initialization(self, tmp_path):
        """Oracle (initialization branch): initialization != 'lh' draws
        population_size independent random psets."""
        ade = algorithms.AsynchronousDifferentialEvolution(
            _ade_config(tmp_path, population_size=4, initialization='rand'))
        out = ade.start_run()
        assert len(out) == 4 and len(ade.individuals) == 4

    def test_better_result_replaces_and_increments_generation(self, tmp_path):
        """Oracle (accept rule + naming): a result whose fitness is <= the stored
        fitness at its index replaces that individual, and the spawned trial is
        named gen(g+1)ind(j); a worse result leaves the individual untouched."""
        ade = algorithms.AsynchronousDifferentialEvolution(_ade_config(tmp_path, population_size=3))
        start = ade.start_run()                       # gen0ind0..2, fitnesses inf
        better = algorithms.Result(start[1], self.d1s, start[1].name); better.score = 4.0
        out = ade.got_result(better)
        assert ade.individuals[1] == start[1] and ade.fitnesses[1] == 4.0
        assert out[0].name == 'gen1ind1'
        worse = algorithms.Result(start[1], self.d1s, start[1].name); worse.score = 99.0
        ade.got_result(worse)
        assert ade.fitnesses[1] == 4.0                # not replaced

    def test_stop_at_max_iterations(self, tmp_path):
        """Oracle (termination): after population_size * max_iterations completed
        sims the run stops."""
        ade = algorithms.AsynchronousDifferentialEvolution(_ade_config(tmp_path, population_size=3,
                                                                       max_iterations=1))
        start = ade.start_run()
        out = None
        for ps in start:
            res = algorithms.Result(ps, self.d1s, ps.name); res.score = 5.0 + start.index(ps)
            out = ade.got_result(res)
        assert out == 'STOP'

    def test_convergence_stop(self, tmp_path):
        """Oracle (convergence criterion): at an iteration boundary, equal
        fitnesses make max/min = 1 < 1 + stop_tolerance, so the run stops."""
        ade = algorithms.AsynchronousDifferentialEvolution(_ade_config(tmp_path, population_size=3,
                                                                       max_iterations=100,
                                                                       stop_tolerance=0.002))
        start = ade.start_run()
        out = None
        for ps in start:
            res = algorithms.Result(ps, self.d1s, ps.name); res.score = 5.0
            out = ade.got_result(res)
        assert out == 'STOP'

    @pytest.mark.parametrize("strategy, expected", [
        ('best1', 'argmin'),
        ('all1', 'index'),
        ('rand1', None),
    ])
    def test_strategy_selects_base_index(self, tmp_path, strategy, expected):
        """Oracle (strategy -> base_index dispatch): the per-result replacement
        proposes from the fittest individual ('best'), the just-finished index
        ('all'), or a random base ('rand', base_index None). Spy on new_individual;
        fitnesses [5,3,7] put argmin at index 1 while the finished pset is index 0,
        so the three strategies give distinct base_index values."""
        ade = algorithms.AsynchronousDifferentialEvolution(_ade_config(tmp_path, population_size=3,
                                                                       de_strategy=strategy))
        start = ade.start_run()
        ade.fitnesses = [5.0, 3.0, 7.0]
        ade.individuals = list(start)
        recorded = []
        orig = ade.new_individual
        ade.new_individual = lambda inds, base_index=None: recorded.append(base_index) or orig(inds, base_index)
        res = algorithms.Result(start[0], self.d1s, start[0].name); res.score = 5.0  # index j=0
        ade.got_result(res)
        if expected == 'argmin':
            assert recorded == [1]
        elif expected == 'index':
            assert recorded == [0]
        else:
            assert recorded == [None]
