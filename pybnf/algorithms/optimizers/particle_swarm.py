"""ParticleSwarm optimizer (the ``pso`` fit type).

Extracted from the algorithms package (M1 Step 4). Subclasses Algorithm and
inherits its run loop + execution seam; it makes no core.* call of its own.
"""


from ..base import Algorithm
from ...config_schema import PyBNFConfigModel
from ...pset import PSet
from ...printing import print1, print2
from ...registry import register_fit_type

import logging
import numpy as np
import re


# Preserve the original module logger name so log records keep the
# 'pybnf.algorithms' channel.
logger = logging.getLogger('pybnf.algorithms')


class PSOConfig(PyBNFConfigModel):
    """Particle-swarm config fields, co-located with the method (ADR-0002,
    ADR-0006). These defaults previously lived in ``config_schema.GlobalConfig``;
    moving them here makes ``pso``'s config knowledge travel with its algorithm
    class. The values are byte-identical to the old global defaults -- the
    int-literal ``adaptive_n_max = 30`` stays an int (pydantic does not validate
    defaults), matching the golden-config net.

    ``particle_weight_final`` is intentionally NOT here: it is not a default but a
    runtime fallback (set to ``particle_weight`` when absent, in ``__init__``
    below), so it stays a pass-through extra.
    """

    particle_weight: float = 0.7
    adaptive_n_max: float = 30
    adaptive_n_stop: float = float('inf')
    adaptive_abs_tol: float = 0.0
    adaptive_rel_tol: float = 0.0
    cognitive: float = 1.5
    social: float = 1.5
    v_stop: float = 0.0

    # particle_weight_final is a runtime fallback (-> particle_weight when absent, set
    # in __init__), not a schema default, but it IS a valid pso key (#401).
    RUNTIME_KEYS = frozenset({'particle_weight_final'})


@register_fit_type('pso', family='optimizer', display_name='Particle Swarm Optimization',
                   schema=PSOConfig)
class ParticleSwarm(Algorithm):
    """
    Implements particle swarm optimization.

    The implementation roughly follows Moraes et al 2015, although is reorganized to better suit PyBNF's format.
    Note the global convergence criterion discussed in that paper is not used (would require too long a
    computation), and instead uses ????

    """

    def __init__(self, config):

        # Former params that are now part of the config
        # variable_list, num_particles, max_evals, cognitive=1.5, social=1.5, w0=1.,
        # wf=0.1, nmax=30, n_stop=np.inf, absolute_tol=0., relative_tol=0.)
        """
        Initial configuration of particle swarm optimizer
        :param conf_dict: The fitting configuration
        :type conf_dict: Configuration

        The config should contain the following definitions:

        population_size - Number of particles in the swarm
        max_iterations - Maximum number of iterations. More precisely, the max number of simulations run is this times
        the population size.
        cognitive - Acceleration toward the particle's own best
        social - Acceleration toward the global best
        particle_weight - Inertia weight of the particle (default 1)

        The following config parameters relate to the complicated method presented is Moraes et al for adjusting the
        inertia weight as you go. These are optional, and this feature will be disabled (by setting
        particle_weight_final = particle_weight) if these are not included.
        It remains to be seen whether this method is at all useful for our applications.

        particle_weight_final -  Inertia weight at the end of the simulation
        adaptive_n_max - Controls how quickly we approach wf - After nmax "unproductive" iterations, we are halfway from
        w0 to wf
        adaptive_n_stop - nd the entire run if we have had this many "unproductive" iterations (should be more than
        adaptive_n_max)
        adaptive_abs_tol - Tolerance for determining if an iteration was "unproductive". A run is unproductive if the
        change in global_best is less than absolute_tol + relative_tol * global_best
        adaptive_rel_tol - Tolerance 2 for determining if an iteration was "unproductive" (see above)

        """

        super().__init__(config)

        # This default value gets special treatment because if missing, it should take the value of particle_weight,
        # disabling the adaptive weight change entirely.
        if 'particle_weight_final' not in self.config.config:
            self.config.config['particle_weight_final'] = self.config.config['particle_weight']

        # Save config parameters
        self.c1 = self.config.config['cognitive']
        self.c2 = self.config.config['social']
        self.max_evals = self.config.config['population_size'] * self.config.config['max_iterations']
        self.output_every = self.config.config['population_size'] * self.config.config['output_every']

        self.num_particles = self.config.config['population_size']
        # Todo: Nice error message if a required key is missing

        self.w0 = self.config.config['particle_weight']

        self.wf = self.config.config['particle_weight_final']
        self.nmax = self.config.config['adaptive_n_max']
        self.n_stop = self.config.config['adaptive_n_stop']
        self.absolute_tol = self.config.config['adaptive_abs_tol']
        self.relative_tol = self.config.config['adaptive_rel_tol']

        self.nv = 0  # Counter that controls the current weight. Counts number of "unproductive" iterations.
        self.num_evals = 0  # Counter for the total number of results received

        # Initialize storage for the swarm data
        self.swarm = []  # List of lists of the form [PSet, velocity]. Velocity is stored as a dict with the same keys
        # as PSet
        self.pset_map = dict()  # Maps each PSet to it s particle number, for easy lookup.
        # One independent [PSet, objective] slot per particle. A list-multiply
        # ([[None, inf]] * n) would alias one inner list across every particle --
        # harmless today (got_result reassigns the whole slot) but a foot-gun if
        # anyone ever mutates a slot in place.
        self.bests = [[None, np.inf] for _ in range(self.num_particles)]  # The best result for each particle
        self.global_best = [None, np.inf]  # The best result for the whole swarm
        self.last_best = np.inf

    def reset(self, bootstrap=None):
        super().reset(bootstrap)
        self.nv = 0
        self.num_evals = 0
        self.swarm = []
        self.pset_map = dict()
        self.bests = [[None, np.inf] for _ in range(self.num_particles)]
        self.global_best = [None, np.inf]
        self.last_best = np.inf

    def start_run(self):
        """
        Start the run by initializing n particles at random positions and velocities
        :return:
        """
        print2('Running Particle Swarm Optimization with %i particles for %i total simulations' %
               (self.num_particles, self.max_evals))

        if self.config.config['initialization'] == 'lh':
            new_params_list = self.random_latin_hypercube_psets(self.num_particles)
        else:
            new_params_list = [self.random_pset() for i in range(self.num_particles)]

        for i in range(len(new_params_list)):
            p = new_params_list[i]
            p.name = 'iter0p%i' % i

            # As suggested by Engelbrecht 2012, set all initial velocities to 0
            new_velocity = dict({v.name: 0. for v in self.variables})

            self.swarm.append([p, new_velocity])
            self.pset_map[p] = len(self.swarm)-1  # Index of the newly added PSet.

        return [particle[0] for particle in self.swarm]

    def got_result(self, res):
        """
        Updates particle velocity and position after a simulation completes.

        :param res: Result object containing the run PSet and the resulting Data.
        :return:
        """

        paramset = res.pset
        score = res.score

        self.num_evals += 1

        if self.num_evals % self.num_particles == 0:
            if (self.num_evals / self.num_particles) % 10 == 0:
                print1('Completed %i of %i simulations' % (self.num_evals, self.max_evals))
            else:
                print2('Completed %i of %i simulations' % (self.num_evals, self.max_evals))
            print2(f'Current best score: {self.global_best[1]:f}')
            # End of one "pseudoflight", check if it was productive.
            if (self.last_best != np.inf and
                    np.abs(self.last_best - self.global_best[1]) <
                    self.absolute_tol + self.relative_tol * self.last_best):
                self.nv += 1
            self.last_best = self.global_best[1]

            # Check stop criterion
            if self.config.config['v_stop'] > 0:
                max_speed = max([abs(v) for p in self.swarm for v in p[1].values()])
                if max_speed < self.config.config['v_stop']:
                    logger.info(f'Stopping particle swarm because the max speed is {max_speed}')
                    return 'STOP'

        if self.num_evals % self.output_every == 0:
            self.output_results()

        p = self.pset_map.pop(paramset)  # Particle number

        # Update best scores if needed.
        if score <= self.bests[p][1]:
            self.bests[p] = [paramset, score]
            if score <= self.global_best[1]:
                self.global_best = [paramset, score]

        # Update own position and velocity
        # The order matters - updating velocity first seems to make the best use of our current info.
        w = self.w0 + (self.wf - self.w0) * self.nv / (self.nv + self.nmax)
        self.swarm[p][1] = \
            {v.name:
                w * self.swarm[p][1][v.name] +
                self.c1 * np.random.random() * self.bests[p][0].get_param(v.name).diff(self.swarm[p][0].get_param(v.name)) +
                self.c2 * np.random.random() * self.global_best[0].get_param(v.name).diff(self.swarm[p][0].get_param(v.name))
            for v in self.variables}

        # Manually check to determine if reflection occurred (i.e. attempted assigning of variable outside its bounds)
        # If so, update based on reflection protocol and set velocity to 0
        new_vars = []
        for v in self.swarm[p][0]:
            new_vars.append(v.add(self.swarm[p][1][v.name]))
            if v.log_space:
                new_val = 10.**(np.log10(v.value) + self.swarm[p][1][v.name])
            else:
                new_val = v.value + self.swarm[p][1][v.name]
            if new_val < v.lower_bound or v.upper_bound < new_val:
                self.swarm[p][1][v.name] = 0.0

        new_pset = PSet(new_vars)
        self.swarm[p][0] = new_pset

        # This will cause a crash if new_pset happens to be the same as an already running pset in pset_map.
        # This could come up in practice if all parameters have hit a box constraint.
        # As a simple workaround, perturb the parameters slightly
        while new_pset in self.pset_map:
            new_pset = PSet([v.add_rand(-1e-6, 1e-6) for v in self.swarm[p][0]])

        self.pset_map[new_pset] = p

        # Set the new name: the old pset name is iter##p##
        # Extract the iter number
        iternum = int(re.search('iter([0-9]+)', paramset.name).groups()[0])
        new_pset.name = 'iter%ip%i' % (iternum+1, p)

        # Check for stopping criteria
        if self.num_evals >= self.max_evals or self.nv >= self.n_stop:
            return 'STOP'

        return [new_pset]

    def add_iterations(self, n):
        self.max_evals += n * self.config.config['population_size']
