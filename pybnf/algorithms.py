"""pybnf.algorithms: contains the Algorithm class and subclasses as well as support classes and functions"""


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
