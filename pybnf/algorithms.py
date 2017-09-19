"""pybnf.algorithms: contains the Algorithm class and subclasses as well as support classes and functions"""


from distributed import as_completed
from distributed import Client

from .pset import Trajectory


import logging


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

    def start_run(self):
        """
        Called by the scheduler at the start of a fitting run.
        Must return a list of PSets that the scheduler should run.

        :return: list of PSets
        """
        logging.info("Initializing algorithm")
        raise NotImplementedError("Subclasses must implement start_run()")

    def got_result(self, res):
        """
        Called by the scheduler when a simulation is completed, with the pset that was run, and the resulting simulation
        data

        :param res: result from the completed simulation
        :type res: Result
        :return: List of PSet(s) to be run next.
        """
        logging.info("Retrieved result")
        raise NotImplementedError("Subclasses must implement got_result()")

    def add_to_trajectory(self, res):
        score = self.objective.evaluate(res.simdata, self.exp_data)
        self.trajectory.add(res.pset, score)

    def run(self):
        client = Client()
        psets = self.start_run()
        jobs = [Job(p) for p in psets]
        futures = [client.submit(job.run_simulation) for job in jobs]
        pool = as_completed(futures, with_results=True)
        while True:
            f, res = next(pool)
            self.add_to_trajectory(res)
            response = self.got_result(res)
            if response == 'STOP':
                logging.info("Stop criterion satisfied")
                break
            else:
                pool.update([client.submit(j.run_simulation) for j in response])
        logging.info("Fitting complete")
        client.close()
