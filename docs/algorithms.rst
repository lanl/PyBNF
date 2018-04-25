Fitting Algorithms
==================

PyBNF contains eight fitting algorithms described here.

Differential Evolution
----------------------

#### How it works
A population of individuals (points in parameter space) are iteratively evaluated with an objective function.  Parent individuals from the current iteration are selected to form new individuals in the next iteration.  The new individual's parameters are derived by combining parameters from the parents

#### Running in Parallel
PyBNF offers parallel, synchronous differential evolution. In each iteration, n simulations are run in parallel, but all must complete before moving on to the next iteration. 

It also offers parallel, asynchronous differential evolution, in which the current population consists of m islands. Each island is able to move on to the next iteration even if other islands are still in progress. If m is set to the number of available processors, then processors will never sit idle. Note however that this might not be the most efficient thing to do. 

#### When to use it
In our experience, differential evolution tends to be the best general-purpose algorithm, and we suggest it as a starting point for a new fitting problem if you are unsure which algorithm to choose. 


Scatter Search
--------------

#### How it works
Scatter Search functions similarly to differential evolution, but maintains a smaller current population than the number of available processors. In each iteration, every possible pair of individuals are combined to propose a new individual. 

Particle Swarm Optimization
---------------------------

#### How it works
In particle swarm optimization, each parameter set is represented by a particle moving through parameter space at some velocity. Each particle accelerates towards the 

#### Running in Parallel
Particle swarm optimization is fundamentally an asynchronous, parallel algorithm. As soon as one simulation completes, that particle can calculate its next parameter set and begin a new simulation. Processors will never remain idle, and adding an arbitrarily large number of processors will continue to improve the performance of the algorithm [citation needed].

#### When to use it
Particle swarm optimization becomes advantageous over the other available algorithms when many processors are available (>100). 
Be warned that if your problem is under-constrained, this algorithm tends to choose parameters that sit on the edge of box constraints. This solution is arguably fine, but makes it very obvious to a reader that your model is under-constrained. 

Simulated Annealing
-------------------

Markov Chain Monte Carlo
------------------------

Parallel Tempering
------------------

DREAM
-----

Simplex
-------
