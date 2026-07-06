.. _config:

Configuring a Fitting Job
=========================

The Configuration File
----------------------

The configuration file is a plain text file with the extension “.conf” that specifies all of the information that PyBNF needs to perform the fitting: the location of the model and data files, and the details of the fitting algorithm to be run.

Several examples of .conf files are included in the examples/ folder.

Each line of a conf file has the general format config_key=value, which assigns the configuration key “config_key” to the value “value”.

The available configuration keys to be specified are detailed in :ref:`config_keys`.


Command-Line Options
--------------------

PyBNF accepts the following command-line flags:

``-c CONF_FILE``
  Path to the configuration file (required).

``-t CLUSTER_TYPE``
  Cluster type (e.g. ``slurm``). See :ref:`cluster` for details.

``-s SCHEDULER_FILE``
  Path to a Dask scheduler file. See :ref:`Manual configuration with Dask <manualdask>`.

``-o``
  Automatically overwrite existing output folders without prompting.

``-r [N]``
  Resume a previous fitting run. Optionally pass a number to add that many iterations.

``-l LOG_PREFIX``
  Custom prefix for log file names.

``-L LOG_LEVEL``
  Set the verbosity of the log file. Options in decreasing order of verbosity: ``debug``, ``info`` (default), ``warning``, ``error``, ``critical``, ``none``. Single-letter abbreviations (``d``, ``i``, ``w``, ``e``, ``c``, ``n``) are also accepted. Use ``-L debug`` for detailed per-iteration output, or ``-L warning`` for minimal logs.

``-d``
  Output a separate debug log file (can be very large).


Model Files
-----------

BioNetGen
^^^^^^^^^

BioNetGen models are specified in plain text files written in BioNetGen language (BNGL). Documentation for BNGL can be found at https://bionetgen.org/.

Two small modifications of a BioNetGen-compatible BNGL file are necessary to use the file with PyBNF

.. highlight:: none

1) Replace each value to be fit with a name that ends in the string “__FREE”.

For example, if the parameters block in our original file was the following:

::

    begin parameters

        v1 17
        v2 42
        v3 37
        NA 6.02e23

    end parameters

the revised version for PyBNF should look like:

::

    begin parameters

        v1 v1__FREE
        v2 v2__FREE
        v3 v3__FREE
        NA 6.02e23

    end parameters

We have replaced each fixed parameter value in the original file with a “FREE” parameter to be fit. Parameters that we do not want to fit (such as the physical constant NA) are left as is.

2) Use the “suffix” argument to create a correspondence between your simulation command and your experimental data file.

For example, if your simulation call ``simulate({method=>”ode”})`` generates data to be fit using the data file data1.exp, you should edit your call to ``simulate({method=>”ode”, suffix=>”data1”})``.

SBML
^^^^

SBML files can be used with PyBNF as is, with no modifications required. PyBNF will match parameter names given in the configuration file, with the IDs of parameters or species in the SBML file. If the name of a species is given, PyBNF fits for the initial concentration of that species. 

PyBNF assumes that any parameters and species that are not named in the config file are not meant to be fit - such values are held constant at the value specified in the SBML file. 

To avoid mistakes in configuration, you may optionally append “__FREE” to the names of parameters to be fit, as with BioNetGen models. PyBNF will raise an error if it finds a parameter ending in “__FREE” in the SBML that is not specified in the configuration file.

Caution: If you are using `COPASI`_ to export SBML files, renaming a parameter is not straightforward. Typically, renaming a parameter only changes its ``name`` field, but PyBNF reads the ``id`` field.

Note that SBML files do not contain information about what time course or parameter scan simulations should be run on the model. Therefore, when using SBML files, it is required to specify this information in the configuration file with the :ref:`time_course <time_course_key>` and :ref:`param_scan <param_scan_key>` keys. For BNGL models, simulation actions should be specified in the BNGL file's ``begin actions`` block, which supports the full set of BioNetGen action arguments.

.. _exp-file:

Experimental Data Files
-----------------------

Experimental data file are plain text files with the extension “.exp” that contain whitespace-delimited tables of data to be used for fitting.

The first line of the .exp file is the header. It should contain the character ``#`` (optional, to match the output format of BioNetGen), followed by the names of each column. The first column name should be the name of the independent variable (e.g. “time” for a time course simulation). The rest of the column names should match the names of observables or functions in a BNGL file, or species in an SBML file (in this section, we refer to all of these options as "observables"). The following lines should contain data, with numbers separated by whitespace. Use “nan” to indicate missing data. Here is a simple example of an exp file. In this case, the corresponding BNGL file should contain observables named X and Y:

::

    #    time    X    Y
        0    5    1e4
        5    7    1.5e4
        10    9    4e4
        15    nan    6.5e4
        20    15    1.1e5

If your are fitting with the chi-squared objective function, you also need to provide a standard deviation for each experimental data point. To do so, include a column in the .exp file with "_SD" appended to the variable name. For example:

::

    #    time    X    Y        X_SD    Y_SD
        0    5    1e4        1    2e2
        5    7    1.5e4    1.2    2e2
        10    9    4e4        1.4    4e2
        15    nan    6.5e4    nan    5e2
        20    15    1.1e5    0.9    5e2

.. _con-file:

Property (BPSL) files
---------------------

Property files are plain text files with the extension ".prop" that define qualitative system properties. In PyBNF, properties are expressed as inequality constraints to be imposed on the outputs of the model. Such constraints can be used to formalize qualitative data known about the biological system of interest. The syntax for writing .prop files, described in this section, is called the  Biological Property Specification Language (BPSL).

Each line of the .prop file should contain constraint declaration consisting of three parts: an inequality to be satisfied, an enforcement condition that specifies when in the simulation time course the constraint is applied, and a clause that specifies how the constraint should be incorporated into the objective function.

Formal grammar
^^^^^^^^^^^^^^

The following EBNF grammar defines the complete BPSL syntax. All keywords are case-insensitive. ::

    constraint     = ( inequality enforcement | split ) [ penalty ] [ comment ]

    inequality     = ( observable | number ) operator ( observable | number )
    operator       = "<" | "<=" | ">" | ">="
    observable     = letter { letter | digit | "_" | "." }
    number         = [ "+" | "-" ] digit { digit } [ "." { digit } ]
                     [ "E" [ "+" | "-" ] digit { digit } ]

    enforcement    = at_clause | between_clause | "once" | "always"
    at_clause      = "at" criterion [ "everytime" | "first" ] [ "before" ]
    between_clause = ( "between" | "once between" ) criterion "," criterion
    criterion      = number | observable "=" number

    split          = observable at_clause operator observable at_clause

    penalty        = weight_clause | likelihood_clause | logit_clause
    weight_clause  = "weight" number [ "altpenalty" inequality ] [ "min" number ]
    likelihood_clause = ( "confidence" number | "pmin" number "pmax" number )
                        [ "tolerance" number ]
    logit_clause   = "logit" "scale" number [ "pmin" number "pmax" number ]

    comment        = "#" { any printable character }

Three methods of incorporating constraints are supported. A static penalty model (the 2018 "hinge") can be used by providing a `Weight`_ clause. In this case, if a constraint of the form :math:`A<B` with weight :math:`w` is violated, then the value added to the objective function is :math:`w*(A-B)`. Alternatively, a likelihood-based model can be used by providing a `Confidence`_ clause (the 2020 "probit") or a `Logit`_ clause (the 2025 logit). In these cases, the contribution is the negative log probability of constraint satisfaction under a proper noise model. The likelihood-based models should be used when statistically rigorous results are important, such as when performing Bayesian uncertainty quantification. It is not recommended to mix penalty models within the same fitting problem (but see `qualitative_loss`_, which re-runs one problem under a single chosen model).

If none of a weight, confidence, or logit clause is provided, a static penalty model is assumed with a weight of 1.

Inequality
^^^^^^^^^^

The inequality can consist of any relationship (<, >, <=, or >=) between two observables, or between one observable and a constant. For example ``A < 5`` , or ``A >= B``. 
Note that < and <= are equivalent unless the ``min`` keyword is used (see `Weight`_).

Enforcement
^^^^^^^^^^^

Four keywords are available to specify when the inequality is enforced. 

* ``always`` - Enforce the inequality at all time points during the simulation.
  
  ``A < 5 always``

* ``once`` - Require that the inequality be true at at least one time point during the simulation.  

  ``A < 5 once``

* ``at`` - Enforce the inequality at one specific time point. This could be a constant time point:  
  
  ``A < 5 at 6`` or equivalently, ``A < 5 at time=6``  

  It is also possible to specify the time point in terms of another observable.
  
  ``A < 5 at B=6`` - Enforce the inequality at the first time point such that B=6 (more exactly, the first time such that B crosses the value of 6 between two consecutive time steps)  

  Using similar syntax, we can specify that the constraint is enforced at every time B=6, not just the first, using the ``everytime`` keyword 
 
  ``A < 5 at B=6 everytime``

  The ``first`` keyword says that the constraint should only (this is the default behavior, so this keyword is optional)  

  ``A<5 at B=6 first``
  
  The ``before`` keyword moves the enforcement the last time point before the condition is met
  
  ``A<5 at B=6 before`` - Enforce the inequality at the last time point before B=6.

  If the specified condition (B=6 in the example) is never met, then the constraint is not applied. It is often useful to add a second constraint to ensure that an "at" constraint is enforced. In this example, assuming the initial value of B is below 6, we could add the constraint ``B>=6 once``
  
  It is also possible to write inequalities with two ``at`` keywords to compare observables at two different values of the independent variable:
  
  ``A at 5 < B at C=6`` - Compares the value of A at time 5 to the value of B at the first time point such that C=6. 

* ``between`` - Enforce the inequality at all times between the two specified time points. The time points may be specified in the same format as with the at keyword above, and should be separated by a comma.  

  ``A < 5 between 7, B=6`` would enforce the inequality from time=7 to the first time after time=7 such that B=6. 

  If the first condition (time=7 in the example) is never met, then the constraint is never enforced. If the second condition (B=6 in the example) is never met, then the constraint is enforced from the start time until the end of the simulation. 
  
* ``once between`` - Require that the inequality be satisfied at at least one point within the specified time range. The syntax is the same as for a ``between`` constraint. 

  ``A < 5 once between 7, B=6`` would require that A<7 at some point between time=7 and the first time after time=7 such that B=6. 

The above definitions assume that time is the independent variable, but note that the same keywords can be used for parameter scans with a different independent variable. 

Weight
^^^^^^

The weight clause consists of the ``weight`` keyword followed by a number. This number is multiplied by the extent of constraint violation to give the value to be added to the objective function. For example:  

``A < 5 at 6 weight 2`` - If the inequality A < 5 is not satisfied at time 6, then a penalty of 2*(A-5) is added to the objective function. 

The ``min`` keyword indicates the minimum possible penalty to apply if the constraint is violated. This minimum is still multiplied by the constraint weight. 

``A < 5 at 6 weight 2 min 4`` - If the inequality A < 5 is not satisfied at time 6, the penalty is :math:`2*\textrm{max}((A-5), 4)`. Since we used the strict < operator, the minimum penalty of 8 is applied even if A=5 at time 6. 

In some unusual cases, it is desirable to use a different observable for calculating penalties than the one used in the inequality. For example, the variable in the inequality might be a discrete variable, and it would be desirable to calculate the penalty with a corresponding continuous variable. This substitution may be made using the ``altpenalty`` keyword in the weight clause, followed by the new inequality to use for calculating the penalty. 

``A < 5 at B=3 weight 10 altpenalty A2<4 min 1`` - This constraint would check if A<5 when B reaches 3. If A >= 5 at that time, it instead calculates the penalty based on the inequality A2<4 with a weight of 10: :math:`10*\textrm{max}(0, A2-4)`. If the initial inequality is violated but the penalty inequality is satisfied, then the penalty is equal to the weight times the min value (10\*1 in the example), or zero if no min was declared. 

Confidence
^^^^^^^^^^

A confidence clause can be provided instead of a weight clause to use a likelihood-based model to incorporate the constraint into the objective function. The clause consists of the ``confidence`` keyword, followed by a number, followed by the ``tolerance`` keyword, followed by a number. 

Under this model, the inequality is rewritten in the form :math:`g<0` for a function :math:`g`. For example, in the constraint ``A<B at time=5``, :math:`g = A(5)-B(5)`, and in the constraint ``A>5 always``, :math:`g = 5 - \textrm{min}(A)` The ``tolerance`` represents the standard deviation of the quantity :math:`g`, which is assumed to have a Gaussian distribution.  The ``confidence`` represents the probability that the constraint should be satisfied by the model. With probability 1-confidence, there is assumed to be a model discrepancy, such that whether the constraint is satisfied is unrelated to the model or its parameters.

The value added to the objective function given confidence :math:`conf`, tolerance :math:`tol`, and :math:`g` as defined above is :math:`-\textrm{log}( conf + (1-2conf) \textrm{cdf}(g,tol,0))`, where :math:`\textrm{cdf}(\mu,\sigma,0)` is the cumulative distribution function of a Gaussian distribution with mean :math:`\mu` and standard deviation :math:`\sigma`, evaluated at 0. 

If ``tolerance`` is omitted, it is assumed to be 0, resulting in a step function.

The following examples illustrate the use of confidence clauses:

* ``A < 5 at time = 4 confidence 0.98 tolerance 1`` - The term added to the objective function would be :math:`-\textrm{log}(0.01 + 0.98*\textrm{cdf}(A(4)-5, 1, 0))`

* ``A > 5 always confidence 0.98`` - Tolerance is assumed to be 0. The term added to the objective function would be :math:`-\textrm{log}(0.99)` if :math:`min(A)>5` or :math:`-\textrm{log}(0.01)` otherwise. 

The keywords ``pmin`` and ``pmax`` may be used in place of ``confidence`` to specify different minimum and maximum probabilities of the constraint. In this case, the term added to the objectve function is :math:`-\textrm{log}( p_{min} + (p_{max}-p_{min}) \textrm{cdf}(g,tol,0))`. For example

* ``A < 5 at time = 4 pmin 0.01 pmax 0.98`` - The term added to the objective function would be :math:`-\textrm{log}(0.01 + 0.97*\textrm{cdf}(A(4)-5, 1, 0))`

Logit
^^^^^

A ``logit`` clause selects the logit (Bernoulli) likelihood model of Miller et al. 2025. As with the confidence clause, the inequality is first rewritten in the form :math:`g<0`; :math:`g` is the signed constraint margin (negative when satisfied). The clause consists of the ``logit`` keyword, the ``scale`` keyword, and a positive number :math:`s`. The term added to the objective function is the softplus

.. math::

    -\textrm{log}\,\sigma(-g/s) = \textrm{log}(1 + e^{g/s}),

where :math:`\sigma` is the logistic function. The scale :math:`s` controls how sharply the penalty turns on as the constraint is violated. As :math:`s \to 0` (with :math:`\textrm{weight} = 1/s`), the logit penalty converges to the 2018 hinge :math:`\textrm{weight} \cdot \textrm{max}(0, g)` — the large-margin limit — so the logit model can be read as a smooth, likelihood-grounded generalization of the static penalty. The logit and probit models agree closely near the decision boundary when their scales are matched by :math:`s \approx \textrm{tolerance}/1.6` (the :math:`\Phi(x) \approx \sigma(1.6x)` link).

Because the softplus is smooth and its gradient with respect to the model parameters is available (via the same forward-sensitivity machinery as the other penalties), the logit model works with the gradient-based optimizers.

* ``A < 5 at time = 4 logit scale 1.0`` - The term added to the objective function is :math:`\textrm{log}(1 + e^{(A(4)-5)/1.0})`.

Optional ``pmin`` and ``pmax`` keywords add probit-style label smoothing for apples-to-apples comparison with the confidence model, clipping the satisfaction probability to :math:`p_{min} + (p_{max}-p_{min})\,\sigma(-g/s)` before the negative log. They are off by default (faithful to the original derivation).

* ``A < 5 at time = 4 logit scale 1.0 pmin 0.01 pmax 0.98`` - The term added is :math:`-\textrm{log}(0.01 + 0.97\,\sigma(-(A(4)-5)/1.0))`.

.. _qualitative_loss:

Choosing one penalty model for the whole fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default (``qualitative_loss = auto``), each constraint's penalty model is chosen by its own clause: a ``weight`` clause selects the hinge, a ``confidence``/``pmin``/``tolerance`` clause selects the probit, and a ``logit scale`` clause selects the logit. The ``qualitative_loss`` configuration key overrides this per-constraint choice, forcing **every** constraint in the fit to a single family — ``hinge``, ``probit``, or ``logit`` — without re-authoring the ``.prop`` files. The scale of the target family is derived from whatever each constraint authored, using the scale-matching identities above (logit :math:`s` ↔ hinge :math:`\textrm{weight} = 1/s` ↔ probit :math:`\textrm{tolerance} = 1.6\,s`). This is a benchmarking convenience for comparing the three models on the same problem; the recommended default for ordinary use is ``auto``. Example::

    qualitative_loss = logit

.. _qualitative_scale:

Estimating the scale as a fitting parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The likelihood models (logit and probit) each carry a scale — the logit ``scale`` :math:`s` or the probit ``tolerance`` :math:`\sigma` — that is normally a fixed number you supply. The ``qualitative_scale`` key instead ties that scale to a **fittable free parameter**, so a fit estimates it jointly with the model parameters (the proper-likelihood framework's payoff: the noise scale need not be elicited in advance). The value is ``fit <parameter>``, naming a free parameter you have declared elsewhere in the ``.conf``::

    qualitative_scale = fit s_qual
    loguniform_var = s_qual 0.01 100

Every logit/probit constraint's scale then reads its live value from ``s_qual`` at each evaluation, and the fit sees the closed-form :math:`\partial(\textrm{penalty})/\partial(\textrm{scale})` in its gradient (so the gradient-based optimizers estimate it efficiently). Declare the scale parameter **positive** — a ``log``-scaled variable (e.g. ``loguniform_var``) is the natural choice, and the log-space transform is handled automatically.

A single scale is tied across **all** qualitative constraints (a globally-tied scale). This is the identifiable case: because BPSL constraints are single-sided (each asserts its inequality *holds*), the scale is informed only by the tension between constraints the best fit can and cannot satisfy, so per-observation scales are not identifiable while one shared scale is. The tie applies to logit/probit constraints; a static (hinge) constraint has no scale to estimate, so combine ``qualitative_scale`` with ``qualitative_loss = logit`` (or ``probit``) when your ``.prop`` files author hinge weights.


Constraints involving multiple models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, observables in property files are assumed to come from the model that the .prop file is mapped to, and the simulation suffix matching the .prop file's name (the same convention as for .exp files). However, it is possible to use "dot notation" to refer to observables in other simulations, as in the following example.

fit.conf::
    
    model = model1.bngl : wt.exp
    model = model2.bngl : mut.prop

mut.prop::
    
    A < wt.A always 
    
In this example, the constraint would check that the value of ``A`` in the simulation of model2 with suffix "mut" is less than the value of ``A`` in the simulation of model1 with suffix "wt". In this way, it is possible to write constraints involving the outputs of multiple models. 

To use this feature, all simulation suffixes must be unique across all models. In addition all observables used in a single constraint must have the same independent variable with the same step size. 

.. _COPASI: http://copasi.org/

