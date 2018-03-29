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


import numpy as np


class TestJob(object):
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        cls.model = pset.BNGLModel('bngl_files/Tricky.bngl')
        d = [
            pset.FreeParameter('koff__FREE__', 'normal_var', 0, 1, value=0.1),
            pset.FreeParameter('__koff2__FREE__', 'normal_var', 0, 1, value=0.1),
            pset.FreeParameter('kase__FREE__', 'normal_var', 0, 1, value=1),
            pset.FreeParameter('pase__FREE__', 'normal_var', 0, 1, value=1),
        ]
        cls.pset = pset.PSet(d)
        cls.bngpath = environ['BNGPATH'] + '/BNG2.pl'
        cls.job = algorithms.Job([cls.model], cls.pset, 'sim_1', cls.bngpath, '.', timeout=None)
        cls.job_to = algorithms.Job([cls.model], cls.pset, 'sim_to', cls.bngpath, '.', timeout=0)

    @classmethod
    def teardown_class(cls):
        rmtree('bnf_out')
        rmtree('sim_net')
        rmtree('sim_x')
        rmtree('sim_1')
        rmtree('sim_to')
        rmtree('sim_to_rerun1')

    def test_job_components(self):
        mkdir('sim_x')
        self.job.folder = getcwd() + '/sim_x'
        model_files = self.job._write_models()
        self.job.execute(model_files)
        sim_data = self.job.load_simdata()
        assert len(sim_data.keys()) == 1
        assert 'Tricky' in sim_data.keys()
        assert sim_data['Tricky'].keys() == set(['p1_5', 'thing'])
        assert isinstance(list(sim_data['Tricky'].values())[0], data.Data)
        assert isfile('sim_x/Tricky_sim_1.bngl')
        assert isdir('sim_x/Tricky_sim_1_thing')

    def test_job_run(self):
        self.job.folder = getcwd() + '/sim_1'
        res = self.job.run_simulation()
        assert isinstance(res, algorithms.Result)
        sim_data = res.simdata
        assert len(sim_data.keys()) == 1
        assert 'Tricky' in sim_data.keys()
        assert sim_data['Tricky'].keys() == set(['p1_5', 'thing'])
        assert isinstance(list(sim_data['Tricky'].values())[0], data.Data)
        assert isfile('sim_1/Tricky_sim_1.bngl')
        assert isdir('sim_1/Tricky_sim_1_thing')

    def test_net_job(self):
        netmodel = pset.NetModel('TrickyWP_p1_5', ['simulate({method=>"ode",t_start=>0,t_end=>1,n_steps=>10})'], [], nf='bngl_files/TrickyWP_p1_5.net')
        mkdir('sim_net')
        job = algorithms.Job([netmodel], pset.PSet([pset.FreeParameter('f', 'normal_var', 0, 1, value=0.5)]), 'test', self.bngpath, '.', timeout=None)
        job.folder = getcwd() + '/sim_net'
        job.execute(job._write_models())
        assert isfile('sim_net/TrickyWP_p1_5_test.net')
        assert isfile('sim_net/TrickyWP_p1_5_test.bngl')
        assert isfile('sim_net/TrickyWP_p1_5_test.cdat')
        assert isfile('sim_net/TrickyWP_p1_5_test.gdat')

    def test_timeout(self):
        res = self.job_to.run_simulation()
        assert isinstance(res, algorithms.FailedSimulation)

    def test_add_failedsimulation(self):
        a = algorithms.Algorithm(
            config.Configuration({"models": {"bngl_files/parabola.bngl"}, 'exp_data': {'bngl_files/par1.exp'},
                                  'bngl_files/parabola.bngl': ['bngl_files/par1.exp'], 'max_iterations': 10,
                                  'population_size': 10,
                                  ('uniform_var', 'v1__FREE__'): [0., 10.], ('uniform_var', 'v2__FREE__'): [0., 10.],
                                  ('uniform_var', 'v3__FREE__'): [0., 10.]}))
        res = self.job_to.run_simulation()
        assert res.fail_type == 0
        a.add_to_trajectory(res)
        assert next(iter(a.trajectory.trajectory.values())) == np.inf