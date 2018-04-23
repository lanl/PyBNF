import re
from .context import pset
from nose.tools import raises


class TestTrajectory:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        """Define constants to be used in tests"""
        params0 = [
            pset.FreeParameter('x', 'normal_var', 0, 1, value=1.0),
            pset.FreeParameter('y', 'normal_var', 0, 1, value=2.0),
            pset.FreeParameter('z', 'normal_var', 0, 1, value=3.14)
        ]
        params1 = [
            pset.FreeParameter('x', 'normal_var', 0, 1, value=3.0),
            pset.FreeParameter('y', 'normal_var', 0, 1, value=700.3),
            pset.FreeParameter('z', 'normal_var', 0, 1, value=5.2e-4)
        ]
        params2 = [
            pset.FreeParameter('x', 'normal_var', 0, 1, value=1.0),
            pset.FreeParameter('y', 'normal_var', 0, 1, value=2.0),
            pset.FreeParameter('z', 'normal_var', 0, 1, value=3.141)
        ]
        params3 = [
            pset.FreeParameter('x', 'normal_var', 0, 1, value=3.2),
            pset.FreeParameter('y', 'normal_var', 0, 1, value=10000.0),
            pset.FreeParameter('z', 'normal_var', 0, 1, value=45.78)
        ]
        params5 = [
            pset.FreeParameter('kk', 'normal_var', 0, 1, value=5.2e-39),
            pset.FreeParameter('xx', 'normal_var', 0, 1, value=3.00),
            pset.FreeParameter('yy', 'normal_var', 0, 1, value=700.3),
            pset.FreeParameter('ww', 'normal_var', 0, 1, value=52e-5)
        ]
        cls.ps0 = pset.PSet(params0)
        cls.ps1 = pset.PSet(params1)
        cls.ps2 = pset.PSet(params2)
        cls.ps3 = pset.PSet(params3)
        cls.ps4 = pset.PSet(params5)
        cls.obj0 = 0.0
        cls.obj1 = 1.0
        cls.obj2 = float("inf")
        cls.obj3 = float("NaN")

    @classmethod
    def teardown_class(cls):
        pass

    def test_build(self):
        traj = pset.Trajectory(1000000)
        traj.add(self.ps0, self.obj0, 'p0')
        assert len(traj.trajectory) == 1
        traj.add(self.ps1, self.obj1, 'p1')
        assert len(traj.trajectory) == 2
        traj.add(self.ps2, self.obj2, 'p2')
        assert len(traj.trajectory) == 3
        traj.add(self.ps3, self.obj3, 'p3')
        assert len(traj.trajectory) == 4

    @raises(Exception)
    def test_incompatible_psets(self):
        traj = pset.Trajectory(1000000)
        traj.add(self.ps0, self.obj0, 'p0')
        traj.add(self.ps4, self.obj2, 'p4')

    def test_write(self):
        traj = pset.Trajectory(1000000)
        traj.add(self.ps0, self.obj0, 'p0')
        traj.add(self.ps1, self.obj1, 'p1')
        traj.add(self.ps2, self.obj2, 'p2')
        traj.add(self.ps3, self.obj3, 'p3')
        s = traj._write()
        assert re.match('#\tSimulation\tObj\tx\ty\tz\n', s)
        assert re.search('\t1.0\t3.0\t700.3\t0.00052\n', s)

    def test_best_fit(self):
        traj = pset.Trajectory(1000000)
        traj.add(self.ps0, self.obj0, 'p0')
        traj.add(self.ps1, self.obj1, 'p1')
        traj.add(self.ps2, self.obj2, 'p2')
        traj.add(self.ps3, self.obj3, 'p3')
        assert traj.best_fit() == self.ps0
        assert traj.best_fit_name() == 'p0'

    def test_max_output(self):
        traj = pset.Trajectory(2)
        traj.add(self.ps0, self.obj0, 'p0')
        traj.add(self.ps1, self.obj1, 'p1')
        traj.add(self.ps2, self.obj2, 'p2')
        traj.add(self.ps3, self.obj3, 'p3')
        s = traj._write()
        assert s == '#\tSimulation\tObj\tx\ty\tz\n\tp0\t0.0\t1.0\t2.0\t3.14\n\tp1\t1.0\t3.0\t700.3\t0.00052\n'

    def test_load_trajectory(self):
        traj = pset.Trajectory.load_trajectory('bngl_files/traj.txt', self.ps0, 1000)
        assert len(traj.trajectory) == 16
        assert traj.trajectory[traj.best_fit()] == 199.84014809103564