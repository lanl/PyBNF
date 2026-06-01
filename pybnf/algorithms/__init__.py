
"""Contains the Algorithm class and subclasses as well as support classes and functions for running simulations"""


from . import core
# Re-export the core execution primitives so the package facade
# (algorithms.Result, algorithms.run_job, ...) keeps resolving for external
# callers and tests. The run loop itself resolves the patched names as
# core.run_job / core.as_completed / core.Job (ADR-0001), so it does not rely
# on these bare names. `X as X` marks them as intentional re-exports for ruff.
from .core import (
    Result as Result,
    FailedSimulation as FailedSimulation,
    Job as Job,
    JobGroup as JobGroup,
    MultimodelJobGroup as MultimodelJobGroup,
    HybridJobGroup as HybridJobGroup,
    run_job as run_job,
    result_from_completed as result_from_completed,
)
# The Algorithm base class and its module-level helpers live in base.py. Every
# leaf class below subclasses Algorithm, so it must be in this namespace before
# those definitions execute; exp10/latin_hypercube are re-exported so leaves
# (SimplexAlgorithm) and the facade (algorithms.exp10) keep resolving them.
from .base import (
    Algorithm as Algorithm,
    latin_hypercube as latin_hypercube,
    exp10 as exp10,
)
# BayesianAlgorithm (sampler base + R-hat/ESS diagnostics) lives in
# samplers/base.py. The Bayesian leaf classes below subclass it, so it must be
# in this namespace before their definitions execute; the facade re-export also
# keeps algorithms.BayesianAlgorithm resolving for tests.
from .samplers.base import BayesianAlgorithm as BayesianAlgorithm
# Leaf fit types, re-exported so pybnf.py's algs.* dispatch and the test facade
# (algorithms.ParticleSwarm, ...) keep resolving. Peeled into optimizers/ and
# samplers/ one file per commit (M1 Step 4).
from .optimizers.particle_swarm import ParticleSwarm as ParticleSwarm
from .optimizers.differential_evolution import (
    DifferentialEvolutionBase as DifferentialEvolutionBase,
    DifferentialEvolution as DifferentialEvolution,
    AsynchronousDifferentialEvolution as AsynchronousDifferentialEvolution,
)
from .optimizers.scatter_search import ScatterSearch as ScatterSearch
from .optimizers.simplex import SimplexAlgorithm as SimplexAlgorithm
from .samplers.dream import DreamAlgorithm as DreamAlgorithm
from .samplers.pdream import PDreamAlgorithm as PDreamAlgorithm
from .samplers.basic_mcmc import BasicBayesMCMCAlgorithm as BasicBayesMCMCAlgorithm
from .samplers.adaptive_mcmc import Adaptive_MCMC as Adaptive_MCMC
from ..pset import PSet
from ..printing import print0, print1
from ..objective import ConstraintCounter

import logging
import os
import copy
import traceback
# Facade re-export: the run loop (now in base.py) catches CancelledError, and
# tests construct algorithms.CancelledError to drive that path.
from concurrent.futures import CancelledError as CancelledError



logger = logging.getLogger(__name__)


class ModelCheck(object):
    """
    An algorithm that just checks the fit quality for a job with no free parameters.

    Does not subclass Algorithm. To run, instead call run_check() with no Cluster.
    """

    def __init__(self, config):
        """
        Instantiates ModelCheck with a Configuration object.
        :param config: The fitting configuration
        :type config: Configuration
        """
        self.config = config
        self.exp_data = self.config.exp_data
        self.objective = self.config.obj
        self.bootstrap_number = None

        logger.debug('Creating output directory')
        if not os.path.isdir(self.config.config['output_dir']):
            os.mkdir(self.config.config['output_dir'])

        if self.config.config['simulation_dir']:
            self.sim_dir = self.config.config['simulation_dir'] + '/Simulations'
        else:
            self.sim_dir = self.config.config['output_dir'] + '/Simulations'

        # Store a list of all Model objects.
        self.model_list = copy.deepcopy(list(self.config.models.values()))

    def run_check(self, debug=False):
        """Main loop for executing the algorithm"""

        print1('Running model checking on the given model(s)')

        empty = PSet([])
        empty.name = 'check'
        job = core.Job(self.model_list, empty, 'check', self.sim_dir, self.config.config['wall_time_sim'], None,
                  None, dict(), delete_folder=False,
                  stochastic_seed_policy=self.config.config['stochastic_seed'])
        result = core.run_job(job, debug, self.sim_dir)

        if isinstance(result, FailedSimulation):
            print0('Simulation failed.')
            return

        result.normalize(self.config.config['normalization'])
        try:
            result.postprocess_data(self.config.postprocessing)
        except Exception:
            logger.exception('User-defined post-processing script failed')
            traceback.print_exc()
            print0('User-defined post-processing script failed. Exiting')
            return

        result.score = self.objective.evaluate_multiple(result.simdata, self.exp_data, result.pset,
                                                         self.config.constraints)
        if result.score is None:
            print0('Simulation contained NaN or Inf values. Cannot calculate objective value.')
            return
        print0('Objective value is %s' % result.score)
        if len(self.config.constraints) > 0:
            counter = ConstraintCounter()
            fail_count = counter.evaluate_multiple(result.simdata, self.exp_data, self.config.constraints)
            total = sum([len(cset.constraints) for cset in self.config.constraints])
            print('Satisfied %i out of %i constraints' % (total-fail_count, total))
            for cset in self.config.constraints:
                cset.output_itemized_eval(result.simdata, self.sim_dir)
