.. _config_keys:

Configuration Keys
==================

The following sections give all possible configuration keys that may be used in your .conf file to configure your
fitting run.  Configuration keys are mapped to their values with the following syntax::

    key = value


Required Keys
-------------
**model**
  Specifies the mapping between model files (.bngl or .xml) and .exp files (.exp or .con). If no experimental files are
  associated with a model write ``none`` instead of a file path.  Model paths and files are followed by a ':' and then
  a comma-delimited list of experimental data files or constraint files corresponding to the model files

  Examples:
    * ``model = path/to/model1.bngl : path/to/data1.exp``
    * ``model = path/to/model2.xml : path/to/data2.con, path/to/data2.exp``

**fit_type**
  The choice of fitting algorithm. Options:
    * ``de`` - :ref:`alg-de`
    * ``ade`` - :ref:`Asynchronous Differential Evolution <alg-de>`
    * ``ss`` - :ref:`alg-ss`
    * ``pso`` - :ref:`Particle Swarm Optimization <alg-pso>`
    * ``bmc`` - :ref:`Bayesian Markov chain Monte Carlo <alg-mcmc>`
    * ``sim`` - :ref:`Simplex <alg-sim>` local search
    * ``sa`` - :ref:`Simulated Annealing <alg-sa>`
    * ``pt`` - :ref:`Parallel tempering <alg-pt>`

  Examples:
    * ``fit_type = de``

**population_size**
  The number parameter sets to maintain in a single iteration of the algorithm. See algorithm descriptions for more
  information.

  Examples:
    * ``population_size = 50``

**max_iterations**
  Maximum number of iterations. This key is required.

  Examples:
    * ``max_iterations = 200``


Other Path Keys
---------------
**bng_command**
  Path to BNG2.pl, including the BNG2.pl file name. This key is required if your fitting includes any .bngl files,
  unless the BioNetGen path is specified with the BNGPATH env variable.

  Default: Uses the BNGPATH environmental variable

  Examples:
    * ``bng_command = path/to/BNG2.pl``

**output_dir**
  Directory where we should save the output.

  Default: "bnf_out"

  Examples:
    * ``output_dir = dirname``

**mutant**
  Declares a model that does not have its own model file, but instead is defined based on another model with some name
  (e.g. ``basemodel``). Following ``basemodel`` is the name of the mutant model; this name is appended to the suffixes
  of the base model. That is, if the base model has data files ``data1.exp`` and ``data2.exp``, a corresponding mutant
  model with the name  "m1" should use the files ``data1m1.exp`` and ``data2m1.exp``. ``statement1``, ``statement2``,
  etc. specify how to change ``basemodel`` to make the mutant model. The statements have the format
  [variable][operator][value] ; for example ``a__FREE=0`` or ``b__FREE*2``. Supported operators are ``=``, ``+``, ``-``,
  ``*``, ``/``.

  Default: None

  Examples:
    * ``mutant = model0 no_a a__FREE=0 : data1no_a.exp, data2no_a.exp``


Parameter Specification
-----------------------
**uniform_var**
  A bounded uniformly distributed variable defined by a 3-tuple corresponding to the variable name, minimum
  value, and maximum value

  Examples:
    * ``uniform_var = k__FREE 10 20``

**normal_var**
  A normally distributed variable defined by a 3-tuple: the name, mean value, and standard deviation. The distribution
  is truncated at 0 to prevent negative values

  Examples:
    * ``normal_var = d__FREE 0 1``

**loguniform_var**
  A variable distributed uniformly in logarithmic space. The value syntax is identical to the **uniform_var** syntax

  Examples:
    * ``loguniform_var = p__FREE 0.001 100``

**lognormal_var**
  A variable normally distributed in logarithmic space.  The value syntax is a 3-tuple specifying the variable name,
  the base 10 logarithm of the mean, and the base 10 logarithm of the standard deviation

  Examples:
    * ``lognormal_var = l__FREE 1 0.1``


The following keys are to be used only with the :ref:`simplex <alg-sim>` algorithm. Simplex should not use any of the
other parameter specifications. If you are using another algorithm with the flag ``refine``, you must set the simplex
algorithm's parameters with ``simplex_step`` or ``simplex_log_step``.

**var**
  The starting point for a free parameter.  It is defined by a 3-tuple, corresponding to the variable's name, its initial
  value and an initial step size (optional).  If not specified, the initial step size defaults to the value specified
  by the simplex-specific parameter ``simplex_step`` (see :ref:`simplex <alg-sim>`)

  Examples:
    * ``var = k__FREE 10``
    * ``var = d__FREE 2 0.05``

**logvar**
  Syntax and sematics are identical to the ``var`` key above, but the initial value and initial step should be specified
  in base 10 logarithmic space.

  Examples:
    * ``logvar = k__FREE -3 1``

Parallel Computing
------------------
``parallel_count = int``
  For a local (non-cluster) fitting run, how many jobs to run in parallel. Default: Use all available cores.
``cluster_type = str``
  Type of cluster used for running the fit. This key may be omitted, and instead specified on the command line with the ``-t`` flag. Currently suports ``slurm`` or ``none``. Will support ``torque`` and ``pbs`` in the future. Default: None (local fitting run).
``scheduler_node = str``
  Manually set node used for creating the distributed Client -- takes a string identifying a machine on a network. If running on a cluster with SLURM, it is recommended to use :ref:`automatic configuration <cluster>` with the flag ``-t slurm`` instead of using this key. Default: None 
``worker_nodes = str1 str2 str3``
  Manually set nodes used for computation - takes one or more strings separated by whitespace identifying machines on a network. If running on a cluster with SLURM, it is recommended to use :ref:`automatic configuration <cluster>` with the flag ``-t slurm`` instead of using this key.  Default: None 

General Options
---------------

Output Options
^^^^^^^^^^^^^^
``delete_old_files = int``
  If 1, delete simulation folders immediately after they complete. If 2, delete both old simulation folders and old sorted_params.txt result files. If 0, do not delete any files (warning, could consume a large amount of disk space). Default: 1
``num_to_output = int``
  The maximum number of PSets to write when writing the trajectory. Default: 5000
``output_every = int``
  Write the Trajectory to file every x iterations. Default: 20
``verbosity = int``
  Specifies the amount of information output to the terminal. 0 - Quiet; user prompts and errors only. 1 - Normal; Warnings and concise progress updates. 2 - Verbose; Information and detailed progress updates. Default: 1

Algorithm Options
^^^^^^^^^^^^^^^^^
``objfunc = str``
  Which :ref:`objective function <objective>` to use. Options: ``chi_sq`` - Chi Squared, ``sos`` - Sum of squares, ``norm_sos`` - Sum of squares, normalized by the value at each point,
  ``ave_norm_sos`` - Sum of squares, normalized by the average value of the variable. Default: chi_sq
``bootstrap = int`` 
  If assigned a positive value, estimate confidence intervals through a bootstrapping procedure.  The assigned integer is the number of bootstrap replicates to perform.  Default: 0 (no bootstrapping)
``bootstrap_max_obj = float``
  The maximum value of a fitting run's objective function to be considered valid in the bootstrapping procedure. If a fit ends with a larger objective value, it is discarded. 
  Default: None
``constraint_scale = float``
  Scale all weights in all constraint files by this multiplicative factor. For convenience only: The same thing could be achieved by editing constraint files, but this option is useful for tuning the relative contributions of quantitative and qualitative data. Default: 1 (no scaling)
``ind_var_rounding = int``
  If 1, make sure every exp row is used by rounding it to the nearest available value of the independent variable in the simulation data. (Be careful with this! Usually, it is better to set up your simulation so that all experimental points are hit exactly) Default: 0
``initialization = str``
  How to initialize parameters. ``rand`` - initialize params randomly according to the distributions. ``lh`` - For ``random_var``\ s and ``loguniform_var``\ s, initialize with a latin hypercube distribution, to more uniformly cover the search space.
``local_objective_eval = int``
  If 1, evaluate the objective function locally, instead of parallelizing this calculation on the workers. This option is automatically enabled when using the ``smoothing`` feature. 
  Default: 0 (unless smoothing is enabled)
``min_objective = float``
  Stop fitting if an objective function lower than this value is reached. Default: None; always run for the maximum iterations
``normalization = type`` ; ``normalization = type : d1.exp, d2.exp`` ; ``normalization = type: (d1.exp: var1,var2)``
  Indicates that simulation data must be normalized in order to compare with exp files. Choices for ``type`` are: ``init`` - normalize to the initial value,  ``peak`` - normalize to the maximum value, ``zero`` - normalize such that each column has a mean of 0 and a standard deviation of 1, ``unit`` - Scales data so that the range of values is between (min-init)/(max-init) and 1 (if the maximum value is 0 (i.e. max == init), then the data is scaled by the minimum value after subtracting the initial value so that the range of values is between 0 and -1). If only the type is specified, the normalization is applied to all exp files. If one or more exp files included, it applies to only those exp files. Additionally, you may enclose an exp file in parentheses, and specify which columns of that exp file get normalized, as in ``(data1.exp: 1,3-5)`` or ``(data1.exp: var1,var2)`` Multiple lines with this key can be used. Default: No normalization
``refine = int``
  If 1, after fitting is completed, refine the best fit parameter set by a local search with the simplex algorithm. Default: 0
``smoothing = int``
  Number of replicate runs to average together for each parameter set (useful for stochastic simulations). Default: 1
``wall_time_gen = int``
  Maximum time (in seconds) to wait to generate the network for a BNGL model. Will cause the program to exit if exceeded. Default: 3600
``wall_time_sim = int``
  Maximum time (in seconds) to wait for a simulation to finish.  Exceeding this results in an infinite objective function value. Caution: For SBML models, using this option has an overhead cost, so don't use it unless needed. Default: 3600  


Algorithm-specific Options
--------------------------

:ref:`Simplex <alg-sim>`
^^^^^^^^^^^^^^^^^^^^^^^^

These settings for the :ref:`simplex <alg-sim>` algorithm may also be used when running other algorithms with ``refine = 1``.

``simplex_step = float``
  In initialization, we perturb each parameter by this step size. If you specify a step size for a specific variable via ``var`` or ``logvar``, it overrides this setting. Default: 1
``simplex_log_step = float``
  Equivalent of ``simplex_step``, for variables that move in log space. Default: ``simplex_step``
``simplex_reflection = float``
  When we reflect a point through the centroid, what is the ratio of dilation on the other side? Default: 1.0
``simplex_expansion = float``
  If the reflected point was the global minimum, how far do we keep moving in that direction? (as a ratio to the initial distance to centroid) Default: 1.0
``simplex_contraction = float``
  If the reflected point was not an improvement, we retry at what distance from the centroid? (as a ratio of the initial distance to centroid) Default: 0.5
``simplex_shrink = float``
  If a whole iteration was unproductive, shrink the simplex by setting simplex point :math:`s[i]` to :math:`x*s[0] + (1-x)*s[i]`, where *x* is the value of this key and :math:`s[0]` is the best point in the simplex. Default: 0.5
``simplex_max_iterations = int``
  If specified, overrides the ``max_iterations`` setting. Useful if you are using the ``refine`` flag and want ``max_iterations`` to refer to your main algorithm.
``simplex_stop_tol = float`` 
  Stop the algorithm if all parameters have converged to within this value (specifically, if all reflections in an iteration move the parameter by less than this 
  value) Default: 0 (don't use this criterion)


:ref:`Differential Evolution <alg-de>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyBNF offers two versions of :ref:`differential evoltution <alg-de>`: synchronous differential evolution (``fit_type = de``) and asynchronous differential evolution (``fit_type = ade``). Both versions may be configured with the follwing keys.

``mutation_rate = float``
  When generating a new individual, mutate each parameter with this probability. Default: 0.5
``mutation_factor = float``
  When mutating a parameter x, change it by mutation_factor*(PS1[x] - PS2[x]) where PS1 and PS2 are random other PSets in the population.  Default: 1.0
``stop_tolerance = float``
  Stop the run if within the current popluation :math:`max(objective) / min(objective) < 1 + e`, where *e* = this value. This criterion triggers when the entire population has converged to roughly the same objective. Default: 0.002
``de_strategy = str``
  Specifies how new parameter sets are chosen. Options are: ``rand1``, ``rand2``, ``best1``, ``best2``, ``all1``, ``all2``. The parameter set we mutate is: 'rand' - a random one, 'best' - the one with the lowest objective value, 'all' - the one we are proposing to replace (so all psets are mutated once per iteration). The amount of mutation is based on: '1' - 1 pair of other parameter sets :math:`(p_1-p_2)`, '2' - 2 pairs of other parameter sets :math:`(p1-p2 + p3-p4)`. Default: rand1

The following options are only available with ``fit_type = de``, and serve to make the algorithm more asynchronous. If used, these options enable :ref:`island-based <alg-island>` differential evolution, which is asynchronous in that each island can independently proceed to the next iteration. 

``islands = int``
  Number of separate populations to evolve. Default: 1
``migrate_every = int``
  After this number of generations, migrate some individuals between islands. Default: 20 (but Inf if ``islands = 1``)
``num_to_migrate = int``
  How many individuals to migrate off of each island during migration. Default: 3


:ref:`Scatter Search <alg-ss>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``init_size = int``
  Number of PSets to test to generate the initial population. Default: 10 * number of variables
``local_min_limit = int``
  If a point is stuck for this many iterations without improvement, it is assumed to be a local min and replaced with a random parameter set. Default: 5
``reserve_size = int``
  Scatter Search maintains a latin-hypercube-distributed "reserve" of parameter sets. When it needs to pick a random new parameter set, it takes one from the reserve, so it's not similar to a previous random choice. The initial size of the reserve is this value. If the reserve becomes empty, we revert to truly random pset choices. Default: max_iterations


:ref:`Particle Swarm <alg-pso>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``cognitive = float``
  Acceleration toward a particle's own best fit
``social = float``
  Acceleration toward the global best fit
``particle_weight = float`` 
  Inertia weight of particle. A value less than 1 can be thought of as friction that contiuously decelerates the particle. Default: 1
``v_stop = float``
  Stop the algorithm if the speeds of all parameters in all particles are less than this value. Default: 0 (don't use this criterion)

A variant of particle swarm that adaptively changes the ``particle_weight`` over the course of the fitting run is configured with the following parameters. See the :ref:`algorithm documentation <pso-adaptive>` for more information. 

``particle_weight_final``
  The final particle weight after the adaptive changing. Default: the value of ``particle_weight``, effectively disabling this feature. 
``adaptive_n_max``
  After this many "unproductive" iterations, we have moved halfway from the initial weight to the final weight. Default: 30
``adaptive_n_stop``
  Afer this many "unproductive" iterations, stop the fitting run. Default: Inf
``adaptive_abs_tol``
  Parameter for checking if an iteration was "unproductive" Default: 0
``adaptive_rel_tol``
  Parameter for checking if an iteration was "unproductive" Default: 0

:ref:`Bayesian Algorithms (bmc, pt, sa) <alg-mcmc>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the family of Bayesian algoritms with Metropolis sampling, PyBNF includes :ref:`MCMC <alg-mcmc>` (``fit_type = bmc``), :ref:`Parallel Tempering <alg-pt>` (``fit_type = pt``), :ref:`Simulated Annealing <alg-sa>` (``fit_type = sa``). These algorithms have many configuration keys in common, as described below. 


For all Bayesian algorithms
"""""""""""""""""""""""""""

``step_size = float``
  When proposing a Monte Carlo step, the step in n-dimensional parameter space has this length. Default: 0.2

``beta = int`` ; ``beta = b1 b2 b3`` 
  Sets the initial beta (1/temperature). A smaller beta corresponds to a more broad exploration of parameter space. If a single value is provided, that beta is used for all replicates. If multiple values are provided, an equal number of replicates uses each value. 
  
  For ``mcmc``, should be set to 1 (the default) to get the true probability distribution. 
  
  For ``pt``, should specify multiple values: the number of values should equal ``population_size``/``reps_per_beta``. Or you may instead use the ``beta_range`` key. Only the largest beta value in the list will constribute to statistical samples, and to get the true probability distribution, this maximum value should be 1.
  
  For ``sa``, should typically be set to a single, small value which will increase over the course of the fitting run. 


For all Bayesian algorithms except ``sa``
"""""""""""""""""""""""""""""""""""""""""

``sample_every = int``
  Every x iterations, save the current PSet into the sampled population. Default: 100
``burn_in = int``
  Don't sample for this many iterations at the start, to let the system equilibrate. Default: 10000
``output_hist_every = int`` 
  Every x samples (i.e every x*sample_every iterations), save a historgram file for each variable, and the credible interval files, based on what has been sampled so far. Regardless, we also output these files at the end of the run.  Default: 100
``hist_bins = int`` 
  Number of bins used when writing the histogram files. Default: 10
``credible_intervals = n1 n2 n3``
  Specify one or more numbers here. For each n, the algorithm will save a file giving bounds for each variable such that in n% of the samples the variable lies within the bounds.  Default: 68 95


For Simulated Annealing
"""""""""""""""""""""""

``beta_max = float`` 
  Stop the algorithm if all replicates reach this beta (1/temperature) value. Default: Inf (don't use this stop criterion)
``cooling = float``
  Each time a move to a higher energy state is accepted, increase beta (1/temperature) by this value. Default: 0.01


For Parallel Tempering
""""""""""""""""""""""

``exchange_every = int``
  Every x iterations, perform replica exchange, swapping replicas that are adjacent in temperature with a statistically correct probability
``reps_per_beta = int``
  How many identical replicas to run at each temperature. Must be a divisor of population_size
``beta_range=min max``
  As an alternative to setting ``beta``, the range of values of beta to use. The replicates will use population_size/reps_per_beta evenly spaced beta values within this range. Only the replicas at the max beta value will be sampled. For the true probability distribution, max should be 1.


.. For DREAM
.. """""""""

.. step_size: As in Bayesian settings, but here it can be set to 'auto' (Not implemented)
.. ``crossover_number = int``
..   The number of distinct crossover probabilities for performing Gibbs sampling on the parameter set.  Random numbers are generated for each parameter and if they are less than the sampled crossover probability, then a new value is calculated in the updated PSet. Default: 3
.. ``zeta = float``
..   A (very) small number for perturbing the calculated update for a particular parameter (applies to all parameters).  Default: 1e-6
.. ``lambda = float``
..   A small number for perturbing parameters selected by the crossover procedure.  Default: 0.1
.. ``gamma_prob = float``
..   A probability that determines how often a jump in parameter space is assigned a value of 1 instead of ``step_size``.  Helps with jumping to the mode of the distribution.  Default: 0.1



