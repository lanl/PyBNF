"""Public facade for the ``pybnf.algorithms`` package.

Re-exports the core execution primitives, the Algorithm / BayesianAlgorithm
bases, every fit-type class, and ModelCheck so ``pybnf.algorithms.<Name>`` keeps
resolving for ``pybnf.py``'s dispatch and the test suite. The implementations
live in core.py / base.py / optimizers/ / samplers/ / model_check.py; this
module contains no logic of its own.
"""


from . import core as core
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
# The Algorithm base class and its module-level helpers live in base.py.
# exp10 / latin_hypercube are re-exported so the SimplexAlgorithm leaf and the
# facade (algorithms.exp10, used by test_scatter) keep resolving them.
from .base import (
    Algorithm as Algorithm,
    latin_hypercube as latin_hypercube,
    exp10 as exp10,
)
# BayesianAlgorithm (sampler base + R-hat/ESS diagnostics) lives in
# samplers/base.py; the facade re-export keeps algorithms.BayesianAlgorithm
# resolving for tests.
from .samplers.base import BayesianAlgorithm as BayesianAlgorithm
# Leaf fit types, re-exported so pybnf.py's algs.* dispatch and the test facade
# (algorithms.ParticleSwarm, ...) keep resolving.
from .optimizers.particle_swarm import ParticleSwarm as ParticleSwarm
from .optimizers.differential_evolution import (
    DifferentialEvolutionBase as DifferentialEvolutionBase,
    DifferentialEvolution as DifferentialEvolution,
    AsynchronousDifferentialEvolution as AsynchronousDifferentialEvolution,
)
from .optimizers.scatter_search import ScatterSearch as ScatterSearch
from .optimizers.simplex import SimplexAlgorithm as SimplexAlgorithm
from .optimizers.simulated_annealing import SimulatedAnnealing as SimulatedAnnealing
from .optimizers.powell import PowellAlgorithm as PowellAlgorithm
from .optimizers.cmaes import CMAESAlgorithm as CMAESAlgorithm
# Gradient-based local optimizers (#386): trf (trust-region least-squares /
# Levenberg–Marquardt, primary) consumes #385's residual Jacobian; lbfgs (bounded
# limited-memory BFGS, the scalar-gradient fallback) consumes the scalar gradient and
# so handles the objectives trf refuses (estimated noise scale, Laplace/count,
# constraints). gntr (#481) is the missing cell: a trust-region step with trf's
# Gauss-Newton/EFIM Hessian extended to those same general-NLL objectives (a Fisher
# Hessian atop the scalar gradient). Each is a GradientOptimizer
# (optimizers/gradient_base.py); importing the leaf runs its @register_fit_type.
from .optimizers.trf import TRFAlgorithm as TRFAlgorithm
from .optimizers.lbfgs import LBFGSAlgorithm as LBFGSAlgorithm
from .optimizers.gntr import GNTRAlgorithm as GNTRAlgorithm
# Profile likelihood (#446/#466): a standalone new-era job_type that reuses the same
# gradient path -- a multi-start TRF polish to the optimum, then one adaptive re-optimized
# profile per parameter for confidence intervals + identifiability. Importing the leaf runs
# its @register_fit_type.
from .optimizers.profile_likelihood import (
    ProfileLikelihoodAlgorithm as ProfileLikelihoodAlgorithm,
)
from .samplers.dream import DreamAlgorithm as DreamAlgorithm
from .samplers.pdream import PDreamAlgorithm as PDreamAlgorithm
from .samplers.basic_mcmc import BasicBayesMCMCAlgorithm as BasicBayesMCMCAlgorithm
from .samplers.adaptive_mcmc import Adaptive_MCMC as Adaptive_MCMC
# hmc (ADR-0059) is the gradient-based reference sampler; its module imports no jax at
# import time (jax/blackjax load lazily on the run path), so registering it here is safe
# without the optional pybnf[jax] extra installed.
from .samplers.hmc import HMCSampler as HMCSampler
# ModelCheck (fit_type 'check') is a utility run -- neither optimizer nor
# sampler -- so it lives at the algorithms top level (model_check.py).
from .model_check import ModelCheck as ModelCheck
# Facade re-export: the run loop (in base.py) catches CancelledError, and tests
# construct algorithms.CancelledError to drive that path.
from concurrent.futures import CancelledError as CancelledError
