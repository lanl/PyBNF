"""pybnf.algorithms: contains the Algorithm class and subclasses as well as support classes and functions"""


from distributed import as_completed
from distributed import Client

from .pset import Model
from .pset import PSet
from .pset import Trajectory

import logging


import logging


class Result(object):
    """
    Container for the results of a single evaluation in the fitting algorithm
    """

    def __init__(self, paramset, simdata):
        """
        Instantiates a Result

        :param paramset: The parameters corresponding to this evaluation
        :type paramset: PSet
        :param simdata: The simulation results corresponding to this evaluation
        :type simdata: list of Data instances
        """
        self.pset = paramset
        self.simdata = simdata


class Job:
    """
    Container for information necessary to perform a single evaluation in the fitting algorithm
    """

    def __init__(self, model, params):
        """
        Instantiates a Job

        :param model: The model to evaluate
        :type model: Model
        :param params: The parameter set with which to evaluate the model
        :type params: PSet
        """
        self.model = model
        self.params = params

    def run_simulation(self):
        """Runs the simulation and reads in the result"""
        
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
        """Adds information from a Result to the Trajectory instance"""

        score = self.objective.evaluate(res.simdata, self.exp_data)
        self.trajectory.add(res.pset, score)

    def run(self):
        """Main loop for executing the algorithm"""
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

class ParticleSwarm(Algorithm):
    """
    Implements particle swarm optimization.

    The implementation roughly follows Moraes et al 2015, although is reorganized to better suit PyBNF's format.
    Note the global convergence criterion discussed in that paper is not used (would require too long a
    computation), and instead uses ????

    """

    def __init__(self, expdata, objective, variable_list, num_particles, max_evals, cognitive=1.5, social=1.5, w0=1.,
                 wf=0.1, nmax=30, n_stop=np.inf, absolute_tol=0., relative_tol=0.):
        """
        Initial configuration of particle swarm optimizer

        :param expdata: Data object
        :param objective: ObjectiveFunction object
        :param max_evals: Halt after this many simulations are run.
        :param cognitive: Parameter c1 - how much is a particle accelerated toward its own previous best
        :param social: Parameter c2 - how much is a particle accelerated toward the global best.

        The remaining parameters relate to the complicated method presented is Moraes et al for adjusting the inertia
        weight as you go. These features have not been directly unit tested. It remains to be seen whether they are
        helpful for our application.
        To disable this option, set w0 and wf to the same value, and set n_stop to infinity (the default)
        :param w0: Inertia weight at the start of the simulation
        :param wf: Inertia weight at the end of the simulation
        :param nmax: Controls how quickly we approach wf - After nmax "unproductive" iterations, we are halfway from
        w0 to wf
        :param n_stop: End the entire run if we have had this many "unproductive" iterations (should be more than nmax)
        :param absolute_tol: Tolerance for determining if an iteration was "unproductive". A run is unproductive if the
        change in global_best is less than absolute_tol + relative_tol * global_best
        :param relative_tol: Tolerance 2 for determining if an iteration was "unproductive" (see above)
        """

        super(ParticleSwarm, self).__init__(expdata, objective)

        # Save config parameters
        self.c1 = cognitive
        self.c2 = social
        self.max_evals = max_evals
        self.w0 = w0
        self.wf = wf
        self.nmax = nmax
        self.n_stop = n_stop
        self.absolute_tol = absolute_tol
        self.relative_tol = relative_tol

        self.num_particles = num_particles
        self.variable_list = variable_list

        self.nv = 0  # Counter that controls the current weight. Counts number of "unproductive" iterations.
        self.num_evals = 0  # Counter for the total number of results received

        # Initialize storage for the swarm data
        self.swarm = []  # List of lists of the form [PSet, velocity]. Velocity is stored as a dict with the same keys
        # as PSet
        self.pset_map = dict()  # Maps each PSet to it s particle number, for easy lookup.
        self.bests = [[None, np.inf]] * num_particles  # The best result for each particle: list of the
        # form [PSet, objective]
        self.global_best = [None, np.inf]  # The best result for the whole swarm
        self.last_best = np.inf

    def start_run(self):
        """
        Start the run by initializing n particles at random positions and velocities
        :return:
        """

        for i in range(self.num_particles):
            new_params = pset.PSet({xi: np.random.uniform(0, 4) for xi in self.variable_list})
            # Todo: Smart way to initialize velocity?
            new_velocity = {xi: np.random.uniform(-1, 1) for xi in self.variable_list}
            self.swarm.append([new_params, new_velocity])
            self.pset_map[new_params] = i

        return [particle[0] for particle in self.swarm]

    def got_result(self, paramset, simdata):
        """
        Updates particle velocity and position after a simulation completes.

        :param paramset: PSet simulated
        :param simdata: Resulting Data object
        :return:
        """

        self.num_evals += 1

        if self.num_evals % self.num_particles == 0:
            # End of one "pseudoflight", check if it was productive.
            if (self.last_best != np.inf and
                    np.abs(self.last_best - self.global_best[1]) <
                    self.absolute_tol + self.relative_tol * self.last_best):
                self.nv += 1
            self.last_best = self.global_best[1]

        score = self.objective.evaluate(simdata, self.exp_data)
        p = self.pset_map.pop(paramset)  # Particle number

        # Update best scores if needed.
        if score < self.bests[p][1]:
            self.bests[p] = [paramset, score]
            if score < self.global_best[1]:
                self.global_best = [paramset, score]

        # Update own position and velocity
        # The order matters - updating velocity first seems to make the best use of our current info.
        w = self.w0 + (self.wf - self.w0) * self.nv / (self.nv + self.nmax)
        self.swarm[p][1] = {v:
                                w * self.swarm[p][1][v] + self.c1 * np.random.random() * (
                                self.bests[p][0][v] - self.swarm[p][0][v]) +
                                self.c2 * np.random.random() * (self.global_best[0][v] - self.swarm[p][0][v])
                            for v in self.variable_list}
        new_pset = pset.PSet({v: self.swarm[p][0][v] + self.swarm[p][1][v] for v in self.variable_list},
                             allow_negative=True)  # Todo: Smarter handling of negative values
        self.swarm[p][0] = new_pset
        self.pset_map[new_pset] = p

        # Check for stopping criteria
        if self.num_evals >= self.max_evals or self.nv >= self.n_stop:
            return 'STOP'

        return [new_pset]
