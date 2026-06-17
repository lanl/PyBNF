from .context import algorithms
from .context import data
from .context import pset
from .context import config
from os import mkdir
from os import environ
from os import getcwd
from os.path import isfile
from os.path import isdir
from shutil import rmtree
from .context import raises
import copy

import numpy as np


class _ConcreteAlgorithm(algorithms.Algorithm):
    """Concrete Algorithm: the base's start_run/got_result are @abstractmethod
    (ADR-0007), so the bare base can't be instantiated. add_to_trajectory (the
    method under test) is inherited unchanged."""

    def start_run(self):
        return []

    def got_result(self, res):
        return []


class TestJob(object):
    @classmethod
    def setup_class(cls):
        cls.output_root = f"test_job_class_{environ.get('PYTEST_XDIST_WORKER', 'local')}"
        rmtree(cls.output_root, ignore_errors=True)
        mkdir(cls.output_root)
        cls.model = pset.BNGLModel('bngl_files/Tricky.bngl')
        d = [
            pset.FreeParameter('koff__FREE', 'normal_var', 0, 1, value=0.1),
            pset.FreeParameter('__koff2__FREE', 'normal_var', 0, 1, value=0.1),
            pset.FreeParameter('kase__FREE', 'normal_var', 0, 1, value=1),
            pset.FreeParameter('pase__FREE', 'normal_var', 0, 1, value=1),
        ]
        cls.pset = pset.PSet(d)
        cls.bng_command = environ['BNGPATH'] + '/BNG2.pl'
        cls.model.bng_command = cls.bng_command
        cls.job = algorithms.Job([cls.model], cls.pset, 'sim_1', cls.output_root, calc_future=None, norm_settings=None,
                                 timeout=None, postproc_settings=dict())
        cls.job_to = algorithms.Job([cls.model], cls.pset, 'sim_to', cls.output_root, calc_future=None, norm_settings=None,
                                    timeout=0, postproc_settings=dict())

    @classmethod
    def teardown_class(cls):
        rmtree(cls.output_root, ignore_errors=True)

    def _folder(self, name):
        return f'{self.output_root}/{name}'

    def test_job_components(self):
        folder = self._folder('sim_x')
        mkdir(folder)
        self.job.folder = getcwd() + '/' + folder
        sim_data = self.job._run_models()
        assert len(sim_data.keys()) == 1
        assert 'Tricky' in sim_data.keys()
        assert sim_data['Tricky'].keys() == set(['p1_5', 'thing'])
        assert isinstance(list(sim_data['Tricky'].values())[0], data.Data)
        assert isfile(f'{folder}/Tricky_sim_1.bngl')
        assert isdir(f'{folder}/Tricky_sim_1_thing')

    def test_job_run(self):
        folder = self._folder('sim_1')
        self.job.folder = getcwd() + '/' + folder
        res = self.job.run_simulation()
        assert isinstance(res, algorithms.Result)
        sim_data = res.simdata
        assert len(sim_data.keys()) == 1
        assert 'Tricky' in sim_data.keys()
        assert sim_data['Tricky'].keys() == set(['p1_5', 'thing'])
        assert isinstance(list(sim_data['Tricky'].values())[0], data.Data)
        assert isfile(f'{folder}/Tricky_sim_1.bngl')
        assert isdir(f'{folder}/Tricky_sim_1_thing')

    def test_net_job(self):
        netmodel = pset.NetModel('TrickyWP_p1_5', ['simulate({method=>"ode",t_start=>0,t_end=>1,n_steps=>10})'], [], [], nf='bngl_files/TrickyWP_p1_5.net')
        netmodel.bng_command = self.bng_command
        folder = self._folder('sim_net')
        mkdir(folder)
        job = algorithms.Job([netmodel], pset.PSet([pset.FreeParameter('f', 'normal_var', 0, 1, value=0.5)]), 'test', self.output_root, calc_future=None, norm_settings=None, timeout=None, postproc_settings=dict())

        job.folder = getcwd() + '/' + folder
        job._run_models()
        assert isfile(f'{folder}/TrickyWP_p1_5_test.net')
        assert isfile(f'{folder}/TrickyWP_p1_5_test.bngl')
        assert isfile(f'{folder}/TrickyWP_p1_5_test.cdat')
        assert isfile(f'{folder}/TrickyWP_p1_5_test.gdat')
        assert isfile(f'{folder}/TrickyWP_p1_5_test.log')

    def test_timeout(self):
        res = self.job_to.run_simulation()
        assert isinstance(res, algorithms.FailedSimulation)

    def test_add_failedsimulation(self):
        a = _ConcreteAlgorithm(
            config.Configuration({"models": {"bngl_files/parabola.bngl"}, 'exp_data': {'bngl_files/par1.exp'},
                                  'bngl_files/parabola.bngl': ['bngl_files/par1.exp'], 'max_iterations': 10,
                                  'population_size': 10,
                                  ('uniform_var', 'v1__FREE'): [0., 10.], ('uniform_var', 'v2__FREE'): [0., 10.],
                                  ('uniform_var', 'v3__FREE'): [0., 10.],
                                  'output_dir': self._folder('pybnf_output')}))
        res = self.job_to.run_simulation()
        assert res.fail_type == 0
        a.add_to_trajectory(res)
        assert a.trajectory.best_score() == np.inf

    def test_postprocess_result(self):
        d = data.Data('bngl_files/special_cases/postprocess/unit_test_data.gdat')
        simdata = {'model1': {'data1': d, 'data2': copy.deepcopy(d)}}
        res = algorithms.Result(self.pset, simdata, 'test')
        postproc_settings = {('model1', 'data1'): 'bngl_files/special_cases/postprocess/myscript.py'}
        res.postprocess_data(postproc_settings)
        assert res.simdata['model1']['data1']['x'][1] == 10
        assert res.simdata['model1']['data2']['x'][1] == 5

    @raises(TypeError)
    def test_failed_postprocess(self):
        d = data.Data('bngl_files/special_cases/postprocess/unit_test_data.gdat')
        simdata = {'model1': {'data1': d, 'data2': copy.deepcopy(d)}}
        res = algorithms.Result(self.pset, simdata, 'test')
        postproc_settings = {('model1', 'data1'): 'bngl_files/special_cases/postprocess/bugscript.py'}
        res.postprocess_data(postproc_settings)
