from .context import data, algorithms, pset, objective, config
from types import SimpleNamespace
import numpy as np
import numpy.testing as npt
from os import path
from shutil import rmtree
from copy import deepcopy


class TestParticleSwarm:
    @classmethod
    def setup_class(cls):
        cls.data1e = [
            '# time    v1_result    v2_result    v3_result  v1_result_SD  v2_result_SD  v3_result_SD\n',
            ' 1 2   3   6   0.1   0.1   0.1\n'
        ]

        cls.d1e = data.Data()
        cls.d1e.data = cls.d1e._read_file_lines(cls.data1e, r'\s+')

        cls.data1s = [
            '# time    v1_result    v2_result    v3_result\n',
            ' 1 2.1   3.1   6.1\n',
        ]
        cls.d1s = data.Data()
        cls.d1s.data = cls.d1s._read_file_lines(cls.data1s, r'\s+')

        cls.data2s = [
            '# time    v1_result    v2_result    v3_result\n',
            ' 1 2.2   3.2   6.2\n',
        ]
        cls.d2s = data.Data()
        cls.d2s.data = cls.d2s._read_file_lines(cls.data2s, r'\s+')

        cls.variables = ['v1__FREE', 'v2__FREE', 'v3__FREE']

        cls.chi_sq = objective.ChiSquareObjective()

        cls.p0 = pset.FreeParameter('v1__FREE', 'uniform_var', 0, 10, 3.14)
        cls.p1 = pset.FreeParameter('v2__FREE', 'uniform_var', 0, 10, 1.0)
        cls.p2 = pset.FreeParameter('v3__FREE', 'uniform_var', 0, 10, 0.1)

        cls.params = pset.PSet([cls.p0, cls.p1, cls.p2])

        cls.config = config.Configuration({'population_size': 15, 'max_iterations': 20, 'cognitive': 1.5, 'social': 1.5,
                      ('uniform_var', 'v1__FREE'): [0, 10], ('uniform_var', 'v2__FREE'): [0, 10], ('uniform_var', 'v3__FREE'): [0, 10],
                      'models': {'bngl_files/parabola.bngl'}, 'exp_data':{'bngl_files/par1.exp'},
                      'bngl_files/parabola.bngl':['bngl_files/par1.exp'],
                      'fit_type': 'pso', 'output_dir': 'test_pso'})

        cls.config2 = config.Configuration({'population_size': 15, 'max_iterations': 20, 'cognitive': 1.5, 'social': 1.5,
                           ('uniform_var', 'v1__FREE'): [0, 10], ('loguniform_var', 'v2__FREE'): [0.01, 1e5],
                           ('lognormal_var', 'v3__FREE'): [0, 1],
                           'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'},
                           'bngl_files/parabola.bngl': ['bngl_files/par1.exp'],
                           'fit_type': 'pso', 'output_dir': 'test_pso2'})

        cls.config_path = 'bngl_files/parabola.conf'

        cls.lh_config = config.Configuration(
            {'population_size': 10, 'max_iterations': 20, 'cognitive': 1.5, 'social': 1.5,
            ('uniform_var', 'v1__FREE'): [0, 10], ('uniform_var', 'v2__FREE'): [0, 10], ('uniform_var', 'v3__FREE'): [0, 10],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'},
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp'], 'output_dir': 'test_pso_lh',
            'initialization': 'lh', 'fit_type': 'pso'})

    @classmethod
    def teardown_class(cls):
        if path.isdir('test_pso_lh'):
            rmtree('test_pso_lh')
        if path.isdir('test_pso2'):
            rmtree('test_pso2')
        if path.isdir('test_pso'):
            rmtree('test_pso')

    def test_random_pset(self):
        ps = algorithms.ParticleSwarm(deepcopy(self.config2))
        params = ps.random_pset()
        assert 0 <= params['v1__FREE'] <= 10
        assert 0.01 < params['v2__FREE'] < 1e5
        assert 1e-4 < params['v3__FREE'] < 1e4

    def test_start(self):
        ps = algorithms.ParticleSwarm(self.config)
        start_params = ps.start_run()
        assert len(start_params) == 15

    def test_bests_slots_are_independent(self):
        # CQ-7: each particle's best slot must be a distinct list object, not
        # aliased copies of one inner list (the old [[None, inf]] * n foot-gun).
        ps = algorithms.ParticleSwarm(self.config)
        assert len({id(slot) for slot in ps.bests}) == ps.num_particles

    def test_updates(self):
        ps = algorithms.ParticleSwarm(self.config)
        start_params = ps.start_run()
        next_params = []
        for p in start_params:
            new_result = algorithms.Result(p, self.d2s, 'sim_1')
            new_result.score = ps.objective.evaluate(self.d2s, self.d1e)
            next_params += ps.got_result(new_result)

        assert ps.global_best[0] in start_params

        new_result = algorithms.Result(next_params[7], self.d1s, 'sim_1')
        new_result.score = ps.objective.evaluate(self.d1s, self.d1e)
        ps.got_result(new_result)  # better than the previous ones
        assert ps.global_best[0] == next_params[7]

        # Exactly 1 individual particle should have its best as that global best, the rest should be one of start_params
        count = 0
        for i in range(15):
            if ps.bests[i][0] == next_params[7]:
                count += 1
            else:
                assert ps.bests[i][0] in start_params
        assert count == 1

    def test_latin_hypercube(self):
        ps = algorithms.ParticleSwarm(self.lh_config)
        ps.start_run()
        for i in range(10):
            # Latin hypercube should distribute starting values evenly (one in each bin) in each dimension.
            assert len([x for x in ps.swarm if i < x[0]['v1__FREE'] < i+1]) == 1

    def test_latin_hypercube_uses_initialization_distribution(self):
        # Regression for #413 at the shared Algorithm startup path: a tight
        # objective prior no longer concentrates LH start points near the mean
        # when the parameter's initializer is bounds-uniform.
        alg = SimpleNamespace(
            variables=[pset.FreeParameter(
                'x__FREE', 'normal_var', 0.0, 0.01, lb=-10.0, ub=10.0,
                initialization_distribution=pset.INITIALIZATION_BOUNDS)],
            rng=np.random.default_rng(1),
        )
        starts = algorithms.Algorithm.random_latin_hypercube_psets(alg, 10)
        vals = [p['x__FREE'] for p in starts]
        assert all(-10.0 <= v <= 10.0 for v in vals)
        assert min(vals) < -8.0
        assert max(vals) > 8.0


# --------------------------------------------------------------------------- #
# The deterministic decision logic of got_result: the velocity/position update,
# the inertia-weight schedule, bounds reflection, and the stopping criteria.
# These drive got_result directly on a hand-built single-particle swarm so the
# only stochastic input (the two np.random.random() draws per dimension) is the
# coefficient on the cognitive/social pull, which we either zero out (cognitive=
# social=0) or freeze via monkeypatch to get a closed-form oracle.
# --------------------------------------------------------------------------- #
def _uniform_pset(values, names=('v1__FREE', 'v2__FREE', 'v3__FREE'), lo=0., hi=10.):
    return pset.PSet([pset.FreeParameter(n, 'uniform_var', lo, hi, v)
                      for n, v in zip(names, values)])


class TestParticleSwarmUpdate:
    @classmethod
    def setup_class(cls):
        cls.d1s = data.Data()
        cls.d1s.data = cls.d1s._read_file_lines(
            ['# time v1_result v2_result v3_result\n', ' 1 2.1 3.1 6.1\n'], r'\s+')

    def _pso(self, tmp_path, **over):
        base = {
            'population_size': 1, 'max_iterations': 100, 'cognitive': 1.5, 'social': 1.5,
            ('uniform_var', 'v1__FREE'): [0, 10], ('uniform_var', 'v2__FREE'): [0, 10],
            ('uniform_var', 'v3__FREE'): [0, 10],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'},
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp'],
            'fit_type': 'pso', 'output_dir': str(tmp_path / 'pso_out')}
        base.update(over)
        return algorithms.ParticleSwarm(config.Configuration(base))

    def _single_particle(self, ps, cur, vel, best, gbest, best_score=1.0, gbest_score=0.5):
        """Install a one-particle swarm at a known state so got_result is
        deterministic given the random coefficient(s)."""
        ps.swarm = [[cur, dict(vel)]]
        ps.particle_of_name = {cur.name: 0}
        ps.bests = [[best, best_score]]
        ps.global_best = [gbest, gbest_score]
        ps.num_evals = 0
        ps.last_best = np.inf

    def test_velocity_update_is_pure_inertia_when_accel_zero(self, tmp_path):
        """Oracle (inertia term): with cognitive=social=0 the acceleration terms
        vanish, and with nv=0 the inertia weight is w=w0=particle_weight, so the
        velocity update is exactly v_new = w0 * v_old componentwise, regardless of
        the personal/global best. Catches a wrong inertia coefficient."""
        ps = self._pso(tmp_path, cognitive=0., social=0., particle_weight=0.7)
        cur = _uniform_pset((5., 5., 5.)); cur.name = 'iter0p0'
        vel = {'v1__FREE': 0.4, 'v2__FREE': -0.3, 'v3__FREE': 0.2}
        self._single_particle(ps, cur, vel, _uniform_pset((1., 2., 9.)),
                              _uniform_pset((9., 8., 1.)))
        res = algorithms.Result(cur, self.d1s, 'iter0p0'); res.score = 20.0  # worse: no best update
        ps.got_result(res)
        for v in ('v1__FREE', 'v2__FREE', 'v3__FREE'):
            npt.assert_allclose(ps.swarm[0][1][v], 0.7 * vel[v])

    def test_velocity_update_separates_cognitive_and_social(self, tmp_path, monkeypatch):
        """Oracle (full PSO velocity update): freeze the random coefficient to
        k=0.5; then v_new = w0*v_old + c1*k*(best-cur) + c2*k*(gbest-cur). With
        distinct c1!=c2 and best!=gbest, swapping the cognitive and social
        coefficients (or flipping a diff sign) changes the result per dimension."""
        ps = self._pso(tmp_path, cognitive=2.0, social=3.0, particle_weight=0.7)
        # Freeze the per-dimension velocity-pull coefficient to k=0.5 (the alg now
        # draws from its own Generator, whose methods can't be patched in place).
        monkeypatch.setattr(ps, 'rng', SimpleNamespace(random=lambda: 0.5))
        cur = _uniform_pset((5., 5., 5.)); cur.name = 'iter0p0'
        vel = {'v1__FREE': 0.4, 'v2__FREE': -0.3, 'v3__FREE': 0.2}
        best = _uniform_pset((6., 4., 5.5)); gbest = _uniform_pset((5.5, 6., 4.))
        self._single_particle(ps, cur, vel, best, gbest)
        res = algorithms.Result(cur, self.d1s, 'iter0p0'); res.score = 20.0
        ps.got_result(res)
        for v, bv, gv in [('v1__FREE', 6., 5.5), ('v2__FREE', 4., 6.), ('v3__FREE', 5.5, 4.)]:
            expected = 0.7 * vel[v] + 2.0 * 0.5 * (bv - 5.) + 3.0 * 0.5 * (gv - 5.)
            npt.assert_allclose(ps.swarm[0][1][v], expected)

    def test_inertia_weight_schedule_midpoint_at_nmax(self, tmp_path):
        """Oracle (inertia-decay schedule): w = w0 + (wf-w0)*nv/(nv+nmax). At
        nv=nmax this is the midpoint (w0+wf)/2. With cognitive=social=0 the update
        is w*v_old, so reading the velocity ratio pins the schedule (catches an
        off-by-one in nv/(nv+nmax) or a swapped w0/wf)."""
        ps = self._pso(tmp_path, cognitive=0., social=0., particle_weight=1.0,
                       particle_weight_final=0.2, adaptive_n_max=4)
        ps.nv = 4  # == nmax
        cur = _uniform_pset((5., 5., 5.)); cur.name = 'iter0p0'
        vel = {'v1__FREE': 0.4, 'v2__FREE': 0.4, 'v3__FREE': 0.4}
        self._single_particle(ps, cur, vel, cur, cur, best_score=5.0, gbest_score=5.0)
        res = algorithms.Result(cur, self.d1s, 'iter0p0'); res.score = 20.0
        ps.got_result(res)
        npt.assert_allclose(ps.swarm[0][1]['v1__FREE'], ((1.0 + 0.2) / 2) * 0.4)

    def test_out_of_bounds_zeros_velocity_and_reflects_position(self, tmp_path):
        """Oracle (bounds handling): if the proposed step would carry a parameter
        past its box, that velocity component is reset to 0 and the position is
        reflected back inside. With cognitive=social=0, w0=1, v_old=3 on a
        particle at 9 in [0,10] gives 9+3=12 -> velocity zeroed, position folds to
        8. In-bounds dimensions keep their (inertia-scaled) velocity."""
        ps = self._pso(tmp_path, cognitive=0., social=0., particle_weight=1.0)
        cur = _uniform_pset((9., 5., 5.)); cur.name = 'iter0p0'
        vel = {'v1__FREE': 3.0, 'v2__FREE': 0.1, 'v3__FREE': -0.1}
        self._single_particle(ps, cur, vel, cur, cur, best_score=5.0, gbest_score=5.0)
        res = algorithms.Result(cur, self.d1s, 'iter0p0'); res.score = 20.0
        ps.got_result(res)
        assert ps.swarm[0][1]['v1__FREE'] == 0.0
        npt.assert_allclose(ps.swarm[0][1]['v2__FREE'], 0.1)
        npt.assert_allclose(ps.swarm[0][1]['v3__FREE'], -0.1)
        npt.assert_allclose(ps.swarm[0][0]['v1__FREE'], 8.0)  # 12 folded into [0,10]
        assert 0. <= ps.swarm[0][0]['v1__FREE'] <= 10.

    def test_log_space_out_of_bounds_zeros_velocity(self, tmp_path):
        """Oracle (log-space bounds branch): for a log-space parameter the
        out-of-box test compares 10**(log10(value)+step) to the bounds. A
        loguniform param at 50 in [0.01,100] with step 1 reaches 10**(log10 50 +1)
        ~= 500 > 100, so its velocity is zeroed (exercises line 1707)."""
        cfg = {
            'population_size': 1, 'max_iterations': 100, 'cognitive': 0., 'social': 0.,
            'particle_weight': 1.0,
            ('loguniform_var', 'v1__FREE'): [0.01, 100], ('uniform_var', 'v2__FREE'): [0, 10],
            ('uniform_var', 'v3__FREE'): [0, 10],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'},
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp'],
            'fit_type': 'pso', 'output_dir': str(tmp_path / 'pso_log')}
        ps = algorithms.ParticleSwarm(config.Configuration(cfg))
        cur = pset.PSet([pset.FreeParameter('v1__FREE', 'loguniform_var', 0.01, 100, 50.),
                         pset.FreeParameter('v2__FREE', 'uniform_var', 0, 10, 5.),
                         pset.FreeParameter('v3__FREE', 'uniform_var', 0, 10, 5.)])
        cur.name = 'iter0p0'
        vel = {'v1__FREE': 1.0, 'v2__FREE': 0.0, 'v3__FREE': 0.0}
        self._single_particle(ps, cur, vel, cur, cur, best_score=5.0, gbest_score=5.0)
        res = algorithms.Result(cur, self.d1s, 'iter0p0'); res.score = 20.0
        ps.got_result(res)
        assert ps.swarm[0][1]['v1__FREE'] == 0.0

    def test_unproductive_flight_increments_nv(self, tmp_path):
        """Oracle (unproductive-iteration counter): at the end of a pseudoflight
        (num_evals % num_particles == 0), if the global best improved by less than
        abs_tol + rel_tol*last_best, nv increments. Seed a finite last_best equal
        to the current global best and feed a non-improving result; with abs_tol=1
        the (zero) improvement is below threshold, so nv goes 0 -> 1. With nv held
        productive (abs_tol=0) the same result would leave nv untouched."""
        ps = self._pso(tmp_path, population_size=1, max_iterations=100, adaptive_abs_tol=1.0)
        cur = _uniform_pset((5., 5., 5.)); cur.name = 'iter0p0'
        ps.swarm = [[cur, {'v1__FREE': 0., 'v2__FREE': 0., 'v3__FREE': 0.}]]
        ps.particle_of_name = {cur.name: 0}
        ps.bests = [[cur, 10.0]]; ps.global_best = [cur, 10.0]
        ps.num_evals = 0; ps.last_best = 10.0; ps.nv = 0
        res = algorithms.Result(cur, self.d1s, 'iter0p0'); res.score = 12.0  # no improvement
        ps.got_result(res)
        assert ps.nv == 1                     # |10 - 10| = 0 < 1.0 -> unproductive

    def test_stop_at_max_evals(self, tmp_path):
        """Oracle (termination): max_evals = population_size * max_iterations; once
        num_evals reaches it, got_result returns 'STOP'. pop=1, max_iterations=1
        => the first result stops the run."""
        ps = self._pso(tmp_path, population_size=1, max_iterations=1)
        cur = _uniform_pset((5., 5., 5.)); cur.name = 'iter0p0'
        self._single_particle(ps, cur, {'v1__FREE': 0., 'v2__FREE': 0., 'v3__FREE': 0.},
                              cur, cur, best_score=5.0, gbest_score=5.0)
        res = algorithms.Result(cur, self.d1s, 'iter0p0'); res.score = 5.0
        assert ps.got_result(res) == 'STOP'

    def test_stop_when_max_speed_below_v_stop(self, tmp_path):
        """Oracle (speed-based termination): at a pseudoflight boundary, if the
        maximum particle speed is below v_stop the run stops. The check reads the
        current velocities (before the update), so all-zero velocities with
        v_stop=0.001 stop on the first result."""
        ps = self._pso(tmp_path, population_size=1, max_iterations=100, v_stop=0.001)
        cur = _uniform_pset((5., 5., 5.)); cur.name = 'iter0p0'
        self._single_particle(ps, cur, {'v1__FREE': 0., 'v2__FREE': 0., 'v3__FREE': 0.},
                              cur, cur, best_score=5.0, gbest_score=5.0)
        res = algorithms.Result(cur, self.d1s, 'iter0p0'); res.score = 5.0
        assert ps.got_result(res) == 'STOP'

    def test_add_iterations_extends_max_evals(self, tmp_path):
        """Oracle (resume bookkeeping): add_iterations(n) extends max_evals by
        exactly n * population_size."""
        ps = self._pso(tmp_path, population_size=5, max_iterations=10)
        before = ps.max_evals
        ps.add_iterations(3)
        assert ps.max_evals == before + 3 * 5

    def test_reset_clears_state(self, tmp_path):
        """Oracle (reset invariant): reset() restores the swarm bookkeeping to the
        constructed empty state, with one [None, inf] best slot per particle."""
        ps = self._pso(tmp_path, population_size=3)
        ps.start_run()
        ps.nv = 7; ps.num_evals = 42; ps.last_best = 1.0
        ps.global_best = [object(), 0.1]
        ps.reset()
        assert ps.swarm == [] and ps.particle_of_name == {}
        assert ps.nv == 0 and ps.num_evals == 0 and ps.last_best == np.inf
        assert ps.global_best == [None, np.inf]
        assert ps.bests == [[None, np.inf]] * 3

    def test_non_lh_initialization(self, tmp_path):
        """Oracle (initialization branch): with initialization != 'lh', start_run
        draws population_size independent random psets inside the box."""
        ps = self._pso(tmp_path, population_size=6, initialization='rand')
        out = ps.start_run()
        assert len(out) == 6 and len(ps.swarm) == 6
        for p in out:
            assert 0. <= p['v1__FREE'] <= 10.


# --------------------------------------------------------------------------- #
# Two particles at one position (#807, #721).
#
# pso used to look a particle up by its *position*, which is not an identity: a
# PSet hashes and compares by value, so a value-keyed map cannot hold two
# particles at one point. The second entry overwrote the first, and the swarm
# either lost a particle without a word (its per-particle best left at
# [None, inf], its score credited to the survivor) or raised KeyError on the
# second result back -- whichever particle happened to move first (#807). On the
# update path the collision was instead broken by jittering one position off the
# other, a step that could not move a large linear parameter at all and so spun
# forever (#721).
#
# The map is now keyed by the in-flight PSet's *name*, which is unique per
# particle by construction, so two particles may share a position and each still
# gets its own result. These tests assert that property rather than the mechanism
# that used to paper over it, and every one of them fails on the previous code:
# the four start-path ones on the lost or misfiled result, and the two update-path
# ones because the position is stepped off the collision -- measured through the
# draw budget below, which is also what turns a reinstated unbounded jitter loop
# into a failure rather than the hang #721 was.
# --------------------------------------------------------------------------- #
class _BudgetedRng:
    """A Generator stand-in that refuses past ``budget`` draws.

    An update makes exactly two draws per parameter (the cognitive and social
    coefficients) and nothing else, so budgeting the draws is how these tests say
    *no step was taken to separate the particles*. It is also the guard that a
    reinstated jitter loop fails rather than hangs -- the #721 behaviour is an
    unbounded loop, and there is no pytest-timeout here to catch one."""

    def __init__(self, budget, seed=0):
        self._inner = np.random.default_rng(seed)
        self._left = budget

    def random(self):
        if self._left <= 0:
            raise AssertionError('the update made more draws than this test allows; '
                                 'something is stepping the position to separate it')
        self._left -= 1
        return self._inner.random()


class TestParticleSwarmDuplicatePositions:
    @classmethod
    def setup_class(cls):
        cls.d1s = data.Data()
        cls.d1s.data = cls.d1s._read_file_lines(
            ['# time v1_result v2_result v3_result\n', ' 1 2.1 3.1 6.1\n'], r'\s+')

    def _pso(self, tmp_path, n=4, lo=0, hi=10, **over):
        base = {
            'population_size': n, 'max_iterations': 100, 'initialization': 'rand',
            ('uniform_var', 'v1__FREE'): [lo, hi], ('uniform_var', 'v2__FREE'): [lo, hi],
            ('uniform_var', 'v3__FREE'): [lo, hi],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'},
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp'],
            'fit_type': 'pso', 'output_dir': str(tmp_path / 'pso_dup')}
        base.update(over)
        return algorithms.ParticleSwarm(config.Configuration(base))

    def _collide_starts(self, ps, dupes):
        """Start ``ps`` with the particles in ``dupes`` drawn to one shared position.

        Distinct PSet objects carrying equal values, which is what two independent
        draws landing together produce -- a box narrow beside the spacing of doubles
        at its magnitude makes it a birthday problem (#807)."""
        real = ps.random_pset
        shared = real()
        drawn = []

        def draw():
            i = len(drawn)
            drawn.append(i)
            return deepcopy(shared) if i in dupes else real()

        ps.random_pset = draw
        return ps.start_run()

    def _feed(self, ps, starts, order, scores):
        for i, score in zip(order, scores):
            res = algorithms.Result(starts[i], self.d1s, starts[i].name)
            res.score = score
            ps.got_result(res)

    def test_colliding_start_draws_keep_every_particle(self, tmp_path):
        """Regression (#807, outcome a): two start draws on one point, and the shared
        position's own result first. All three velocity terms are then zero, so the
        surviving particle does not move and re-registers under the same key, absorbing
        the duplicate's result as its own -- silently, with nothing logged and nothing
        raised. The lost particle's best stayed [None, inf] for the whole run."""
        ps = self._pso(tmp_path)
        starts = self._collide_starts(ps, {0, 2})
        assert starts[0] == starts[2] and starts[0] is not starts[2]   # the collision
        assert ps.particle_of_name == {'iter0p%i' % i: i for i in range(4)}
        self._feed(ps, starts, [0, 1, 2, 3], [10.0, 11.0, 12.0, 13.0])
        assert all(b[0] is not None for b in ps.bests)   # no particle went unscored
        assert len(ps.particle_of_name) == len(ps.swarm) == 4

    def test_colliding_start_draws_survive_any_result_order(self, tmp_path):
        """Regression (#807, outcome b): the same collision, but another particle
        reports a better score first. That pulls the shared-position particle off the
        point, its key goes with it, and the next result for that position used to die
        in ``pset_map.pop``."""
        ps = self._pso(tmp_path)
        starts = self._collide_starts(ps, {0, 2})
        self._feed(ps, starts, [1, 0, 2, 3], [5.0, 12.0, 13.0, 14.0])   # KeyError before
        assert all(b[0] is not None for b in ps.bests)
        assert len(ps.particle_of_name) == len(ps.swarm) == 4

    def test_each_score_goes_to_the_particle_that_ran_it(self, tmp_path):
        """The name says which particle a result came back from, so a score is credited
        to the particle that ran it and not to whichever one the position last belonged
        to. This is what a map from position to a *list* of particles could not do: with
        two coincident particles it would have to assign the two scores in arrival order,
        which for a stochastic objective mixes up which particle saw which draw."""
        ps = self._pso(tmp_path)
        starts = self._collide_starts(ps, {0, 2})
        self._feed(ps, starts, [0, 1, 2, 3], [10.0, 11.0, 12.0, 13.0])
        assert [b[1] for b in ps.bests] == [10.0, 11.0, 12.0, 13.0]

    def test_every_particle_of_a_pinned_box_gets_its_own_result(self, tmp_path):
        """The one declaration that makes the collision certain rather than unlikely:
        ``uniform_var = x 5 5`` is accepted (see #808) and pins the parameter, so with
        every parameter pinned every draw is the same point. The swarm used to collapse
        onto one particle for the whole run -- four particles, one map entry, all four
        results credited to particle 3, bests [inf, inf, inf, 10.0]. It is still a fit
        with nothing to search, but no particle is lost and no result is misfiled."""
        ps = self._pso(tmp_path, lo=5, hi=5)
        starts = ps.start_run()
        assert len(ps.particle_of_name) == 4
        self._feed(ps, starts, [0, 1, 2, 3], [10.0, 11.0, 12.0, 13.0])
        assert [b[1] for b in ps.bests] == [10.0, 11.0, 12.0, 13.0]
        assert len(ps.particle_of_name) == 4

    # --- the update path (#721) --------------------------------------------- #
    def _collide_update(self, ps, hi, here, step, budget=6):
        """Stage the update-path collision: particle 0 sits at ``here`` with velocity
        ``step``, so its next position is ``here + step`` -- exactly where particle 1 is
        already being simulated. The budget is the six draws the velocity update itself
        makes (two per parameter, both against a zeroed acceleration); a seventh means
        something is stepping the position to separate it."""
        cur = _uniform_pset((here,) * 3, lo=0., hi=hi); cur.name = 'iter0p0'
        other = _uniform_pset((here + step,) * 3, lo=0., hi=hi); other.name = 'iter0p1'
        vel = {'v1__FREE': step, 'v2__FREE': step, 'v3__FREE': step}
        ps.swarm = [[cur, dict(vel)], [other, dict(vel)]]
        ps.particle_of_name = {'iter0p0': 0, 'iter0p1': 1}
        ps.bests = [[cur, 5.0], [other, 5.0]]
        ps.global_best = [cur, 5.0]
        ps.num_evals = 0
        ps.last_best = np.inf
        ps.rng = _BudgetedRng(budget)
        res = algorithms.Result(cur, self.d1s, 'iter0p0'); res.score = 20.0
        return other, res

    def test_a_particle_may_step_onto_another_particles_position(self, tmp_path):
        """Regression (#721): a step that would leave the box zeroes that velocity
        component, so a swarm converging on a corner piles up there and this is the
        ordinary case, not the exotic one. The particle now goes exactly where the
        velocity update sends it and both particles stay tracked under their own names.
        At a magnitude of 6e12 one ULP is 1.2e-4, so the +-1e-6 step that used to
        separate them rounded away entirely and the loop taking it never terminated."""
        ps = self._pso(tmp_path, n=2, hi=1e13, cognitive=0., social=0., particle_weight=1.0)
        other, res = self._collide_update(ps, hi=1e13, here=5e12, step=1e12)
        out = ps.got_result(res)
        assert len(out) == 1
        assert out[0] == other                     # coincident, and that is allowed
        assert all(out[0][v] == 6e12 for v in ('v1__FREE', 'v2__FREE', 'v3__FREE'))
        assert ps.particle_of_name == {'iter0p1': 1, 'iter1p0': 0}   # both still tracked

    def test_an_ordinary_magnitude_collision_is_also_left_alone(self, tmp_path):
        """The same on the [0, 10] box, where the old step did work: the position is no
        longer moved off the collision at any magnitude, so a fit's particles sit where
        the velocity update puts them rather than a jitter's width away from it."""
        ps = self._pso(tmp_path, n=2, hi=10., cognitive=0., social=0., particle_weight=1.0)
        other, res = self._collide_update(ps, hi=10., here=5., step=1.)
        out = ps.got_result(res)
        assert out[0] == other
        assert all(out[0][v] == 6.0 for v in ('v1__FREE', 'v2__FREE', 'v3__FREE'))
        assert ps.particle_of_name == {'iter0p1': 1, 'iter1p0': 0}
