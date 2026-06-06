"""ModelCheck -- the ``check`` fit type, a first-class checking method.

A checking run (statistical model checking): evaluates the objective value and
constraint satisfaction for the given parameters without searching parameter
space. It registers in the ``checker`` family -- a peer of ``optimizer`` and
``sampler``, not a utility afterthought -- but, being neither an optimizer nor a
sampler, it lives at the algorithms package top level rather than under
optimizers/ or samplers/. Extracted byte-identical (M1 Step 4). It does not
subclass Algorithm, but it does use the execution machinery: the patched names
are resolved as core.Job / core.run_job through the core module object (ADR-0001
seam), and ConstraintCounter is read from this module's namespace (tests patch
it here).
"""


from . import core
from .core import FailedSimulation
from ..pset import PSet
from ..printing import print0, print1
from ..objective import ConstraintCounter
from ..registry import register_fit_type

import logging
import os
import copy
import traceback


# Preserve the original module logger name so log records keep the
# 'pybnf.algorithms' channel.
logger = logging.getLogger('pybnf.algorithms')


@register_fit_type('check', family='checker', display_name='Model Check')
class ModelCheck:
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
