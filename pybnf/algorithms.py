"""pybnf.algorithms: contains the Algorithm class and subclasses as well as support classes and functions"""


from .pset import Trajectory


class Result(object):
    def __init__(self, paramset, simdata):
        self.pset = paramset
        self.simdata = simdata


class Job:
    def __init__(self, model, params):
        self.model = model
        self.params = params

    def run_simulation(self):
        pass


class Algorithm(object):
    def __init__(self, exp_data, objective):
        """
        Instantiates an Algorithm with a set of experimental data and an objective function.  Also
        initializes a Trajectory instance to track the fitting progress

        :param exp_data: List of experimental Data objects to be fit
        :type exp_data: iterable
        :param objective: The objective function
        :type objective: ObjectiveFunction
        """
        self.exp_data = exp_data
        self.objective = objective
        self.trajectory = Trajectory()

