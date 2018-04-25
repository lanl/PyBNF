Fitting Algorithms
==================

Summary of Available Fitting Algorithms
---------------------------------------

Differential Evolution | Population-based | Synchronous or Asynchronous | General-purpose parameter fitting
Scatter Search | Population-based | Synchronous | Difficult problems with high dimensions or many local minima
Particle Swarm | Population-based | Asynchronous | Optimal use on very large numbers of processors
Simulated Annealing | Metropolis sampling | Independent Markov Chains | Problem-specific applications
Markov Chain Monte Carlo | Metropolis sampling |Independent Markov Chains | Finding probability distributions of parameters
Parallel Tempering | Metropolis sampling | Synchronized Markov Chains | Finding probability distributions in challenging probablity landscapes
DREAM | Hybrid Population-based / Metropolis Sampling | Synchronous | 
Simplex | Local search | Synchronous | Local optimization, or refinement of a result from another algorithm.

General implementation features for all algorithms
--------------------------------------------------



Differential Evolution
----------------------

Algorithm
^^^^^^^^^
A population of individuals (points in parameter space) are iteratively evaluated with an objective function.  Parent individuals from the current iteration are selected to form new individuals in the next iteration.  The new individual's parameters are derived by combining parameters from the parents. New individuals are accepted into the population if the value 

Parallelization
^^^^^^^^^^^^^^^
PyBNF offers parallel, synchronous differential evolution. In each iteration, n simulations are run in parallel, but all must complete before moving on to the next iteration. 

It also offers parallel, asynchronous differential evolution, in which the current population consists of m islands. Each island is able to move on to the next iteration even if other islands are still in progress. If m is set to the number of available processors, then processors will never sit idle. Note however that this might not be the most efficient thing to do. 

Implementation details
^^^^^^^^^^^^^^^^^^^^^^
PyBNF uses the implementation of asynchronous differential evolution described in [Penas2015]_. The synchronous algorithm is a special case in which there is only one island and no migration step. 

We maintain a list of ``population_size`` current parameter sets, and in each iteration, ``population_size`` new parameter sets are proposed. To propose a parameter set, we choose 3 random parameter sets p1, p2, and p3 in the current population. For each free parameter P, the new parameter set is assigned the value p1[P] + ``mutation_factor`` * (p2[P]-p3[P]) with probability ``mutation_rate``, or p1[P] with probability 1 - ``mutation_rate``. The new parameter set replaces the parameter set with the same index in the current population if it has a lower objective value. 

In the asynchronous version of the algorithm, the population is divided into ``num_islands`` islands, which each follow the above update procedure independently. Every ``migrate_every`` iterations, a migration step occurs in which ``num_to_migrate`` individuals from each island are transferred randomly to others (according to a random permutation of the islands, keeping the number of individuals on each island constant). The migration step does not require synchronization of the islands; it is performed when the last island reaches the appropriate iteration number, regardless of whether other islands are already further along. 

Applications
^^^^^^^^^^^^
In our experience, differential evolution tends to be the best general-purpose algorithm, and we suggest it as a starting point for a new fitting problem if you are unsure which algorithm to choose. 


Scatter Search
--------------

Algorithm
^^^^^^^^^
Scatter Search [Glover2000]_ functions similarly to differential evolution, but maintains a smaller current population than the number of available processors. In each iteration, every possible pair of individuals are combined to propose a new individual.

Parallelization
^^^^^^^^^^^^^^^
In a scatter search run of population size n, each iteration requires n\*(n-1) independent simulations that can all be run in parallel. Scatter search requires synchronization at the end of each iteration, waiting for all simulations to complete before moving to the next iteration. 

Implementation details
^^^^^^^^^^^^^^^^^^^^^^
The PyBNF implementation follows the outline presented in the introduction of [Penas2017]_ and uses the recombination method described in [Egea2009]_.

We maintain a reference set of ``population_size`` individuals, recommended to be a small number (~ 12-18). Each newly proposed parameter set is based on a "parent" parameter set and a "helper" parameter set, both from the current reference set. In each iteration, we consider all possible parent-helper combinations, for a total of n\*(n-1) parameter sets. The new parameter set depends on the position of the parent and helper in the sorted reference set.   


Applications
^^^^^^^^^^^^
We find scatter search is also a good general-purpose fitting algorithm. It performs especially well on fitting problems that are difficult due to a search space that is high dimensional or contains many local minima. 
 

Particle Swarm Optimization
---------------------------

Algorithm
^^^^^^^^^
In particle swarm optimization, each parameter set is represented by a particle moving through parameter space at some velocity. The acceleration of each particle is set in a way that moves it toward more favorable areas of parameter space: the acceleration has contributions pointing toward both the best parameter set seen so far by the individual particle, and the global best parameter set seen by any particle in the population. 

Parallelization
^^^^^^^^^^^^^^^
Particle swarm optimization in PyBNF is an asynchronous, parallel algorithm. As soon as one simulation completes, that particle can calculate its next parameter set and begin a new simulation. Processors will never remain idle, and adding an arbitrarily large number of processors will continue to improve the performance of the algorithm [citation needed].

Implementation details
^^^^^^^^^^^^^^^^^^^^^^
The PyBNF implementation is based on the description in [Moraes2015]_. Each particle keeps track of its current position, velocity, and the best parameter set it has seen during the run. 

After each simulation completes, the velocity of the particle is updated according to the formula v[i+1] = w\*v[i] + c1\*u1*(xi-x_min) + c2\*u2\*(xi-x_globalmin). The constants in the above formula may be set with config keys: w is ``particle_weight``, c1 is ``cognitive``, and c2 is ``social``. xi is the current particle position, v[i] is the current velocity, v[i+1] is the updated velocity, x_min is the best parameter set this particle has seen, and x_globalmin is the best parameter set any particle has seen. u1 and u2 are uniform random numbers in the range [0,1]. Following the velocity update, the position of the particle is updated by adding its current velocity. 

We apply a special treatment if a ``uniform_var`` or ``loguniform_var`` moves outside of the specified box constraints. As with other algorithms, the particle position is reflected back inside the boundaries. In addition, the component of the velocity corresponding to the parameter that moved out of bounds is set to zero, to prevent the particle from immediately crossing the same boundary again. 

An optional feature (discussed in [Moraes2015]_) allows the particle weight w to vary over the course of the simulation. In the original algorithm descirption, w was called "inertia weight", but when w takes a value less than 1, it can be thought of as friction - a force that decelerates particles regardless of the objective function evaluations. The idea is to reduce w (increase friction) over the course of the fitting run, to make the particles come to a stop at a good objective value by the end of the run. 

When using the adaptive friction feature, w starts at ``particle_weight``, and approaches ``particle_weight_final`` by the end of the simulation. The value of w changes based on how many iterations we deem "unproductive" according to the following criterion: An iteration is unproductive if the global best objective function obj_min changes by less than ``adaptive_abs_tol`` + ``adaptive_rel_tol`` \* obj_min, where ``adaptive_abs_tol`` and ``adaptive_rel_tol`` can be set in the config. Then, we keep track of N, the total number of unproductive iterations so far. At each iteration we set w = ``particle_weight`` + (``particle_weight_final`` - ``particle_weight``) \* N / (N + ``adaptive_n_max``). As can be seen in the above formula, the config key ``adaptive_n_max`` sets the number of unproductive iterations it takes to reach halfway between ``particle_weight`` and ``particle_weight_final``.

Applications
^^^^^^^^^^^^
Particle swarm optimization becomes advantageous over the other available algorithms when many processors are available (>100), and when the runtime per simulation can vary greatly depending on the parameter set (such as in some SSA and NFSim runs). In these cases, the asynchronicity of the particle swarm allows you to take full advantage of all available processors at all times. 


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
Ryan: please write this one

Simplex
-------

.. [Egea2009] Egea, J. A.; Balsa-Canto, E.; García, M.-S. G.; Banga, J. R. Dynamic Optimization of Nonlinear Processes with an Enhanced Scatter Search Method. Ind. Eng. Chem. Res. 2009, 48 (9), 4388–4401.
.. [Glover2000] Glover, F.; Laguna, M.; Martí, R. Fundamentals of Scatter Search and Path Relinking. Control Cybern. 2000, 29 (3), 652–684.
.. [Kozer2013] Kozer, N.; Barua, D.; Orchard, S.; Nice, E. C.; Burgess, A. W.; Hlavacek, W. S.; Clayton, A. H. A. Exploring Higher-Order EGFR Oligomerisation and Phosphorylation—a Combined Experimental and Theoretical Approach. Mol. BioSyst. Mol. BioSyst 2013, 9 (9), 1849–1863.
.. [Moraes2015] Moraes, A. O. S.; Mitre, J. F.; Lage, P. L. C.; Secchi, A. R. A Robust Parallel Algorithm of the Particle Swarm Optimization Method for Large Dimensional Engineering Problems. Appl. Math. Model. 2015, 39 (14), 4223–4241.
.. [Penas2015] Penas, D. R.; González, P.; Egea, J. A.; Banga, J. R.; Doallo, R. Parallel Metaheuristics in Computational Biology: An Asynchronous Cooperative Enhanced Scatter Search Method. Procedia Comput. Sci. 2015, 51 (1), 630–639.
.. [Penas2017] Penas, D. R.; González, P.; Egea, J. A.; Doallo, R.; Banga, J. R. Parameter Estimation in Large-Scale Systems Biology Models: A Parallel and Self-Adaptive Cooperative Strategy. BMC Bioinformatics 2017, 18 (1), 52.
