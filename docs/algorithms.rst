Fitting Algorithms
==================

PyBNF contains eight fitting algorithms described here.

Differential Evolution | Population-based | Synchronous or Asynchronous | General-purpose parameter fitting
Scatter Search | Population-based | Synchronous | Difficult problems with high dimensions or many local minima
Particle Swarm | Population-based | Asynchronous | Optimal use on very large numbers of processors
Simulated Annealing | Metropolis sampling | Independent Markov Chains | Problem-specific applications
Markov Chain Monte Carlo | Metropolis sampling |Independent Markov Chains | Finding probability distributions of parameters
Parallel Tempering | Metropolis sampling | Synchronized Markov Chains | Finding probability distributions in challenging probablity landscapes
DREAM | Hybrid Population-based / Metropolis Sampling | Synchronous | 
Simplex | Local search | Synchronous | Local optimization, or refinement of a result from another algorithm.

Differential Evolution
----------------------

Algorithm
^^^^^^^^^
A population of individuals (points in parameter space) are iteratively evaluated with an objective function.  Parent individuals from the current iteration are selected to form new individuals in the next iteration.  The new individual's parameters are derived by combining parameters from the parents

Parallelization
^^^^^^^^^^^^^^^
PyBNF offers parallel, synchronous differential evolution. In each iteration, n simulations are run in parallel, but all must complete before moving on to the next iteration. 

It also offers parallel, asynchronous differential evolution, in which the current population consists of m islands. Each island is able to move on to the next iteration even if other islands are still in progress. If m is set to the number of available processors, then processors will never sit idle. Note however that this might not be the most efficient thing to do. 

Applications
^^^^^^^^^^^^
In our experience, differential evolution tends to be the best general-purpose algorithm, and we suggest it as a starting point for a new fitting problem if you are unsure which algorithm to choose. 


Scatter Search
--------------

Algorithm
^^^^^^^^^
Scatter Search functions similarly to differential evolution, but maintains a smaller current population than the number of available processors. In each iteration, every possible pair of individuals are combined to propose a new individual.

Parallelization
^^^^^^^^^^^^^^^
In a scatter search run of population size n, each iteration requires n*(n-1) independent simulations that can all be run in parallel. Scatter search requires synchronization at the end of each iteration, waiting for all simulations to complete before moving to the next iteration. 

Applications
^^^^^^^^^^^^
We find scatter search is also a good general-purpose fitting algorithm. It performs especially well on fitting problems that are difficult due to a search space that is high dimensional or contains many local minima. 
 

Particle Swarm Optimization
---------------------------

Algorithm
^^^^^^^^^
In particle swarm optimization, each parameter set is represented by a particle moving through parameter space at some velocity. Each particle accelerates towards the 

Parallelization
^^^^^^^^^^^^^^^
Particle swarm optimization is fundamentally an asynchronous, parallel algorithm. As soon as one simulation completes, that particle can calculate its next parameter set and begin a new simulation. Processors will never remain idle, and adding an arbitrarily large number of processors will continue to improve the performance of the algorithm [citation needed].

Applications
^^^^^^^^^^^^
Particle swarm optimization becomes advantageous over the other available algorithms when many processors are available (>100). 
Be warned that if your problem is under-constrained, this algorithm tends to choose parameters that sit on the edge of box constraints. This solution is arguably fine, but makes it very obvious to a reader that your model is under-constrained. 


Markov Chain Monte Carlo
------------------------

Algorithm
^^^^^^^^^
Markov chain Monte Carlo is a Bayesian method in which points in parameter space are sampled with a frequency proportional to the probability that the parameter set is correct given the data. The result is a probability distribution over parameter space that expresses the likelihood of each possible parameter set. With this algorithm, we obtain not just a point estimate of the best fit, but a means to quantify the uncertainty in each parameter value. 

When running Markov chain Monte Carlo, PyBNF outputs additional files containing this probability distribution information. The files in ``Results/Histograms/`` give histograms of the marginal probability distributions for each free parameter. The files ``credible##.txt`` (e.g., ``credible95.txt``) use the marginal histogram for each parameter to calculate a *credible interval* - an interval in which the parameter value is expected to fall with the specified probability (e.g. 95%).  Finally, ``samples.txt`` contains all parameter sets sampled over the course of the fitting run, allowing the user to perform further custom analysis on the sampled probability distribution. 

Implementation details
^^^^^^^^^^^^^^^^^^^^^^
Our implementation is described in [Kozer2013]_. We start at a random point in parameter space, and make a step of size ``step_size`` to move to a new location in parameter space. We take the value of the objective function to be the probability of the data given the parameter set (the *likelihood* in Bayesian statistics).  We assume a prior distribution based on the parameter definitions in the config file -- a uniform, loguniform, normal, or lognormal distribution, depending on the config key used. Note: If a uniform or loguniform prior is used, the prior does not affect the result other than to confine the distribution within the specified range. If a normal or lognormal prior is used, the prior does affect the probability of accepting each proposed move, and therefore the choice of prior affects the final sampled probability distribution. 

The Bayesian *posterior* distribution -- the probability of the parameters given the data -- is given by the product of the above likelihood and prior. We use the value of the posterior to determine whether to accept the proposed move. 

Moves are accepted according to the Metropolis criterion. If a move increases the value of the posterior, it is always accepted. If it decreases the value of the posterior, it is accepted with probability e^(-BDF), where DF is the change in the posterior, and B (an analog for 1/Temperature) is taken to be 1 in this simple implementation.

Parallelization
^^^^^^^^^^^^^^^
Markov chain Monte Carlo is not an inherently parallel algorithm. In the Markov chain, we need to know the current state before proposing the next one. However, PyBNF supports running several independent Markov chains by specifying the number of chains with the ``population_size`` key. All samples from all parallel chains are pooled to obtain a better estimate of the final posterior probability distribution. 

Note that each chain must independently go through the burn-in period, but after the burn-in time, your rate of sampling will be improved proportional to the number of parallel chains in your run. 

Applications
^^^^^^^^^^^^
Markov chain Monte Carlo is the simplest method available in PyBNF to generate a probability distribution in parameter space. 


Simulated Annealing
-------------------

Algorithm
^^^^^^^^^
Simulated annealing is another Markov chain-based algorithm, but our goal is not to find a full probability distribution, just find the optimal parameter set. To do so, we start the Markov chain at a high temperature, where unfavorable moves are accepted frequently, and gradually reduce the temperature over the course of the simulation. The idea is that we will explore parameter space broadly at the start of the fitting run, and become more confined to the optimal region of parameter space as the run proceeds. 

Implementation Details
^^^^^^^^^^^^^^^^^^^^^^
The Markov chain is implemented in the same way as described above for the Markov chain Monte Carlo algorithm, incorporating both the objective function value and the prior distribution to calculate the posterior probability density. 

The difference is in the Metropolis criterion for acceptance of a proposed move. Here, a move that decreases the value of the posterior is accepted with probability e^(-BDF), where B changes over the course of the fitting run. 

Parallelization
^^^^^^^^^^^^^^^
Simulated annealing is not an inherently parallel algorithm. The trajectory is a Markov chain in which we need to know the current state before proposing the next one. However, PyBNF supports running several independent simulated annealing chains 


Applications
^^^^^^^^^^^^


Parallel Tempering
------------------

DREAM
-----

Simplex
-------

.. [Kozer2013] Kozer, N.; Barua, D.; Orchard, S.; Nice, E. C.; Burgess, A. W.; Hlavacek, W. S.; Clayton, A. H. A. Exploring Higher-Order EGFR Oligomerisation and Phosphorylation—a Combined Experimental and Theoretical Approach. Mol. BioSyst. Mol. BioSyst 2013, 9 (9), 1849–1863.
