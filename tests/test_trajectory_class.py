import re
from .context import pset
from nose.tools import raises


class TestTrajectory:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        """Define constants to be used in tests"""
        cls.ps0 = pset.PSet({'x': 1.0, 'y': 2.0, 'z': 3.14})
        cls.ps1 = pset.PSet({'x': 3.0, 'y': 700.3, 'z': 5.2e-4})
        cls.ps2 = pset.PSet({'x': 1.0, 'y': 2.0, 'z': 3.141})
        cls.ps3 = pset.PSet({'x': 3.2, 'y': 10000.0, 'z': 45.78})
        cls.ps4 = pset.PSet({'kk': 5.2e-39, 'xx': 3.00, 'yy': 700.3, 'ww': 52e-5})
        cls.obj0 = 0.0
        cls.obj1 = 1.0
        cls.obj2 = float("inf")
        cls.obj3 = float("NaN")

    @classmethod
    def teardown_class(cls):
        pass

    def test_build(self):
        traj = pset.Trajectory()
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
        traj = pset.Trajectory()
        traj.add(self.ps0, self.obj0, 'p0')
        traj.add(self.ps4, self.obj2, 'p4')

    def test_write(self):
        traj = pset.Trajectory()
        traj.add(self.ps0, self.obj0, 'p0')
        traj.add(self.ps1, self.obj1, 'p1')
        traj.add(self.ps2, self.obj2, 'p2')
        traj.add(self.ps3, self.obj3, 'p3')
        s = traj._write()
        print(s)
        assert re.match('#\tSimulation\tObj\tx\ty\tz\n', s)
        assert re.search('\t1.0\t3.0\t700.3\t0.00052\n', s)

    def test_best_fit(self):
        traj = pset.Trajectory()
        traj.add(self.ps0, self.obj0, 'p0')
        traj.add(self.ps1, self.obj1, 'p1')
        traj.add(self.ps2, self.obj2, 'p2')
        traj.add(self.ps3, self.obj3, 'p3')
        assert traj.best_fit() == self.ps0
        assert traj.best_fit_name() == 'p0'

    def test_max_output(self):
        traj = pset.Trajectory(max_output=2)
        traj.add(self.ps0, self.obj0, 'p0')
        traj.add(self.ps1, self.obj1, 'p1')
        traj.add(self.ps2, self.obj2, 'p2')
        traj.add(self.ps3, self.obj3, 'p3')
        s = traj._write()
        assert s == '#\tSimulation\tObj\tx\ty\tz\n\tp0\t0.0\t1.0\t2.0\t3.14\n\tp1\t1.0\t3.0\t700.3\t0.00052\n'
