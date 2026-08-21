.. _config_keys:

Configuration Keys
==================

The following sections give all possible configuration keys that may be used in your .conf file to configure your
fitting run.  Each line of the .conf file sets the value of a configuration key with the general syntax::

    key = value


Required Keys
-------------
.. _model_legacy:

**model** (legacy form, ``model = …``)
  Specifies the mapping between model files (.bngl or .xml) and data files (.exp or .prop). Model paths and files are
  followed by a ':' and then a comma-delimited list of experimental data files or property files corresponding to the
  model files. If no experimental files are associated with a model write ``none`` instead of a file path.

  Examples:

    * ``model = path/to/model1.bngl : path/to/data1.exp``
    * ``model = path/to/model2.xml : path/to/data2.prop, path/to/data2.exp``
    * ``model = path/to/model3.xml : none``

.. _model_decl:

**model** (new-era declaration form, ``model: …``)
  Under a modern :ref:`edition <edition>` (``edition >= 2``) a model is *declared* with
  the colon form, which carries **no** data binding — data is introduced separately
  through an experiment's measurements, not on the model line (this retires the legacy
  coupling of data onto the model). One or more model files follow the ``:``; the
  ``modelId`` is the filename **stem**, which must be unique across all declarations.

  ``model:`` lines are repeatable and accumulate, so a many-model job reads as one line
  per model; a comma list is shorthand for a few. Requires ``edition >= 2``; the legacy
  ``model = … : …`` form above continues to work at every edition.

  A declaration of a **single** SBML (``.xml``) or Antimony (``.ant``) model on
  ``sbml_backend = bngsim`` may also state that model's own CVODE tolerances, in any
  order after the file:

  * **atol:** ``<number>``, ``auto``, or ``tracking [<decades>]`` — the per-model form of
    :ref:`sbml_atol <sbml_rtol>`, meaning exactly what that key means, for this model only.
  * **rtol:** ``<number>`` — the per-model form of :ref:`sbml_rtol <sbml_rtol>`.
  * **species_atol:** ``<species> <number>[, <species> <number>…]`` — a hand-written
    absolute tolerance for named species of this model. Every species it does *not* name
    keeps whatever it would otherwise have had, so this is a set of exceptions rather than
    a replacement for the derivation.

  ``sbml_atol`` and ``sbml_rtol`` remain the fit-wide defaults; a field here overrides the
  matching global key for the one model it names, and states nothing about the others.
  That is the point of putting them here: a global key applies to *every* SBML/Antimony
  model in the fit, which is why neither could ever take a per-species vector — a
  positional one is ordered against a species list a conf author cannot see, and a
  species-keyed one has no reading across models that do not share species names.

  A ``model:`` line carrying any of these fields must declare exactly one model, and that
  model must be one with a CVODE tolerance to state. A BNGL model writes ``atol``/``rtol``
  in its own ``begin actions`` block; the RoadRunner backend has its own integrator
  settings; and ``sbml_integrator = gillespie`` runs every action stochastically, where
  there are no CVODE tolerances at all. All three are refused here rather than accepted and
  ignored. Requires ``edition >= 2``.

  **The species names are bngsim's, not the document's**, and the difference is checked
  rather than papered over: a name the model does not integrate is an error at config
  load, listing the names it does have. Two cases make the two lists differ. bngsim
  renames a species id that collides with an Antimony reserved word, so a model file
  declaring ``NULL`` is integrated as ``_ant_NULL`` (the error says so when it recognizes
  the case). And a parameter or compartment driven by a rate rule or an event assignment
  becomes a state of its own, appended after every declared species — nameable here, and
  reachable by nothing else, since it has no declared initial value for a derivation to
  read.

  **A number you write is used verbatim.** Neither clamp of the derivation applies to a
  ``species_atol`` entry: not the ``1e-8`` ceiling, and not the model-wide floor. Those
  bound how far a rule reading *initial values* may go on its own; a number in the conf is
  a statement, exactly as a plain ``sbml_atol`` number is. It must be finite and strictly
  positive — zero would be legal in a bare CVODE per-species vector and is refused as a
  ``tracking`` ceiling, so the stricter rule is the one applied, and it is applied
  everywhere.

  The fields compose. ``atol: auto`` chooses the base the map is laid over; a plain
  ``atol`` number makes that base the number, broadcast; ``atol: tracking`` makes the
  resulting vector the trajectory-following **ceiling**. The steady-state convergence cutoff stays
  the model-wide number in every case, for the reason :ref:`sbml_atol <sbml_rtol>` gives.
  A ``species_atol`` needs a bngsim with per-species tolerance support and is refused —
  not quietly dropped — without one.

  Examples:

    * ``model: egfr.bngl`` (one model; ``modelId`` = ``egfr``)
    * ``model: egfr.bngl, erbb2.bngl`` (comma list)
    * ``model: egfr.bngl`` then ``model: erbb2.bngl`` (multiple lines, union)
    * ``model: weber.xml, atol: auto`` (this model trusts its own scale; the rest of the
      fit is unaffected)
    * ``model: weber.xml, species_atol: PKD 1e-3, CERT 1e-2`` (two species by hand,
      everything else derived)
    * ``model: weber.xml, atol: 1e-6, rtol: 1e-10, species_atol: PKD 1e-3`` (everything at
      ``1e-6`` except ``PKD``)

.. _condition:

**condition** (new-era, ``condition: …``)
  Under a modern :ref:`edition <edition>` (``edition >= 2``) a **condition** is a *named*
  set of parameter perturbations applied to a base model — a PyBNF Mutant, equal to a
  PEtab v2 Condition. It is the perturbation half of the legacy ``mutant`` line, with
  **no** data binding (data is introduced separately, through an experiment).

  The line is ``condition: <name>, perturbations: <var op val>[, <var op val>…]``. Each
  perturbation is a variable, an operator, and a number: ``=`` sets the value absolutely,
  while ``*`` ``/`` ``+`` ``-`` apply relative to the parameter's nominal value. An
  optional ``model: <file>`` field (placed before ``perturbations:``) names the base
  model; it is omittable when the job declares a single model, and required when it
  declares more than one. Requires ``edition >= 2``.

  Examples:

    * ``condition: dimer_dead, perturbations: kdimer = 0``
    * ``condition: overexpr, perturbations: erbb2_tot * 20, kdeg / 2``
    * ``condition: overexpr, model: erbb2.bngl, perturbations: erbb2_tot * 20`` (multi-model)

.. _experiment:

**experiment** (new-era, ``experiment: …``)
  Under a modern :ref:`edition <edition>` (``edition >= 2``) an **experiment** is a
  *named* simulation bound to its measurement files — a PEtab v2 Experiment. The
  experiment **name** replaces the legacy filename→suffix convention as the simulation's
  identity, so a data file can be named anything and the data↔simulation link is *stated*,
  not inferred from filenames.

  The line is ``experiment: <name>, data: <file1.exp>[, <file2.exp>…]`` plus the optional
  labeled fields ``condition: <name>``, ``preequilibrate: <name>``, ``model: <file>``,
  ``type: <type>``, ``method: <ode|ssa|pla|nf>``, and
  ``measurement_params: <file.tsv>``, which may appear in any order after the name; only
  ``data:`` is required.

    * **data:** a comma list of ``.exp`` files. **Multiple files are replicates** — all
      their rows become measurements under the one experiment (stacked, not averaged), the
      thing the legacy surface cannot express without pre-averaging.
    * **measurement_params:** optionally names a tab-separated per-measurement placeholder
      sidecar. Replicate-aware sidecars use the columns ``replicate``, ``column``, ``time``,
      ``placeholder``, and ``token`` (``replicate`` is 1-based); the original four-column
      format without ``replicate`` remains valid and shares each time binding across all
      replicate files.
    * **The simulation outputs at exactly the data's points.** The independent-variable
      column of the data supplies the simulation's output grid (the BNGL ``begin actions``
      block is no longer needed for fitting); PyBNF synthesizes the ``simulate`` action
      from the data, so the scoring grid always lines up with the measurements.
    * **condition:** names a :ref:`condition <condition>` to apply (omitted ⇒ wildtype,
      "model as is").
    * **preequilibrate:** names a :ref:`condition <condition>` that puts the model in an
      unmeasured **steady state before the measured time course begins** — the PEtab v2
      pre-equilibration protocol. The named condition (and the measurement ``condition:``,
      if any) is applied inline as a ``setParameter`` change: PyBNF runs the model to
      steady state under it, then applies the measurement condition and simulates the data
      grid from that equilibrated state. A pre-equilibration condition may use only
      **absolute** (``=``) perturbations. The conditions it names are consumed by the
      experiment, so they are not also run as standalone conditions. Available for BNGL
      models and for SBML/Antimony models under ``sbml_backend = bngsim``; the
      RoadRunner SBML backend has no state carry-over between phases and refuses it.
    * **model:** names the base model by filename stem; omittable when the job declares a
      single model, required when it declares more than one.
    * **type:** is **inferred** from the data — a ``time`` column ⇒ a time course, a ``time``
      column whose every value is ``inf`` ⇒ a **steady state**, any other independent
      variable ⇒ a dose-response ``parameter_scan`` — and stated (``time_course`` /
      ``steady_state`` / ``parameter_scan``) only to make the choice explicit.
    * **t_end:** for a ``parameter_scan``, the fixed measurement time of each dose (omitted
      ⇒ each dose runs to steady state). For a ``steady_state`` experiment it is instead the
      **max-time bound** on the relaxation, default ``1e6``. Inert for a time course, whose
      grid comes from the data.
    * **method:** the simulator, default ``ode``.

  Requires ``edition >= 2``.

  **Steady-state measurements.** Data measured at equilibrium is written with ``time = inf``
  in the ``.exp`` — PEtab's own steady-state time. PyBNF then relaxes the model to
  equilibrium (early-stopping on ``||dx/dt||``, bounded by ``t_end:``) instead of integrating
  to a fixed endpoint, and scores the datum against that final state. NFsim has no
  steady-state solve, so ``method: nf`` is refused; and one experiment cannot mix ``inf``
  with finite times (a steady state and a time course are two different simulations — give
  each its own ``experiment:``). (ADR-0086)

  Examples:

    * ``experiment: egf_high, data: high_wt_r1.exp, high_wt_r2.exp`` (two replicates)
    * ``experiment: egf_high_dd, condition: dimer_dead, data: high_dd.exp``
    * ``experiment: eq, data: equilibrium.exp`` (an ``.exp`` measured at ``time = inf``)
    * ``experiment: dose, preequilibrate: serum_starve, data: dose.exp`` (equilibrate under
      ``serum_starve``, then measure)
    * ``experiment: egf_high, model: egfr.bngl, data: high.exp`` (multi-model)

.. _observable:

**observable** (new-era, ``observable: …``)
  Under a modern :ref:`edition <edition>` (``edition >= 2``) an **observable** line remaps
  a data-file **column header** to a model observable/function **name** when the two
  differ. By default a ``.exp`` column header *is* the model observable name, and the
  objective matches experimental columns to simulation columns by that name — so this line
  is needed only when the measured column is named something else (common with real data).
  Without it, a differently-named data column has no matching simulation column and the fit
  raises.

  The line is ``observable: <entity>, column: <header>`` — the model **entity** first, the
  data column **header** second. It renames the ``<header>`` column to ``<entity>`` (and
  its ``<header>_SD`` per-point :ref:`noise <noise_model_key>` companion, where present, to
  ``<entity>_SD``) in every experimental data file, so the by-name match succeeds.

  The override is **global** (a top-level line, not per-experiment): it applies across all
  experimental data. A data file that does not contain ``<header>`` (an experiment that
  doesn't measure that observable) is left unchanged; a ``<header>`` present in *no* data
  file is treated as a typo and raises, listing the columns actually present. The
  independent-variable column cannot be remapped, and a remap that would collide with an
  existing column raises. Requires ``edition >= 2``.

  Example:

    * ``observable: pErk, column: pErk_measured`` (the model observable ``pErk`` is
      measured by the data column named ``pErk_measured``)

.. note::

   **PEtab v2 export.** The new-era problem surface above (``model:`` / ``condition:`` /
   ``experiment:`` / ``data:`` / ``observable:``, together with the free parameters and the
   modern objective) is exactly what the PEtab v2 exporter reads, and export is a
   *transcription*: an ``experiment:`` becomes a PEtab Experiment (its name the
   ``experimentId``, its ``data:`` replicates the measurement rows), a ``condition:``
   becomes a PEtab Condition, and an ``observable:`` renames a column before it is
   classified. The exporter is **new-era only**: it refuses a legacy data linkage
   (``model = X : Y.exp`` / ``mutant`` / ``param_scan``) under a modern edition, requiring
   the surface above. A dose-response (``parameter_scan``) experiment runs each dose to
   steady state by default (PEtab ``time = inf``), with an optional ``t_end:`` fixed
   endpoint; it exports to N steady-state Conditions/Experiments and imports back, closing
   the dose-response round trip (#426). A plain **steady-state** experiment needs no
   translation at all: its ``.exp`` time already *is* PEtab's ``inf``, so it exports and
   imports as an ordinary measurement row. (ADR-0028, ADR-0046, ADR-0086)

.. _fit_type:

**fit_type**
  The **legacy** (:ref:`edition <edition>` 1) name for the run-selector key. Under a
  modern edition (``edition >= 2``) it is renamed to :ref:`job_type <job_type>` and
  naming ``fit_type`` is an error; in the legacy edition it works exactly as before.
  Selects the procedure to run:

    * ``de`` - :ref:`alg-de`
    * ``ade`` - :ref:`Asynchronous Differential Evolution <alg-de>`
    * ``ss`` - :ref:`alg-ss`
    * ``pso`` - :ref:`Particle Swarm Optimization <alg-pso>`
    * ``mh`` - :ref:`Metropolis-Hastings MCMC (Not recommended) <alg-mcmc>`
    * ``sim`` - :ref:`Simplex <alg-sim>` local search
    * ``powell`` - :ref:`Powell <alg-powell>` local search
    * ``cmaes`` - :ref:`CMA-ES <alg-cmaes>` (local search, or global search over a bounded box)
    * ``sa`` - :ref:`Simulated Annealing (Not recommended) <alg-sa>`
    * ``pt`` - :ref:`Parallel tempering (Not recommended) <alg-pt>`
    * ``am`` - :ref:`Adaptive MCMC <alg-am>`
    * ``dream`` - :ref:`DREAM <alg-dream>`
    * ``p_dream`` - :ref:`DREAM <alg-dream>` with preconditioning (P-DREAM)
    * ``hmc`` - :ref:`Hamiltonian Monte Carlo (NUTS) <alg-hmc>` (analytical / ``expression`` objectives only; requires ``edition >= 2`` and the ``pybnf[jax]`` extra)
    * ``check`` - Run :ref:`model checking <model_check>` instead of fitting


  Example:

    * ``fit_type = de``

.. _job_type:

**job_type**
  The **modern** (:ref:`edition <edition>` ``>= 2``) name for the run-selector key,
  taking the same values as :ref:`fit_type <fit_type>` above. It replaces ``fit_type``
  because that name was a misnomer -- the key selects across point-estimate
  *optimizers* (``de`` / ``ade`` / ``pso`` / ``ss`` / ``sim`` / ``powell`` / ``cmaes``
  / ``sa``, and the gradient-based :ref:`trf / lbfgs / gntr <gradient_fitting>` plus
  :ref:`multiple shooting <multiple_shooting>`, ``ms``), Bayesian
  *samplers* (``am`` / ``dream`` / ``p_dream`` / ``pt`` / ``mh``, and the
  gradient-based :ref:`hmc <alg-hmc>` for analytical objectives), the
  :ref:`profile-likelihood <gradient_fitting>` identifiability analysis
  (``profile_likelihood``), and the model *checker*
  (``check``), not just fitting. The value names the specific
  procedure; the key names the kind of job. Requires :ref:`edition <edition>` ``>= 2``,
  and like the modern objective surface there is **no implicit default** -- the run
  must be named. Under a modern edition the legacy ``fit_type`` key is rejected.

  Example:

    * ``job_type = de`` (with ``edition = 2``)

**objfunc**
  The **legacy** (:ref:`edition <edition>` 1) objective-function key. It still works
  exactly as before when no modern ``edition`` is declared, but under a modern edition
  (``edition >= 2``) it is an error -- name the objective with the modern three-key
  surface instead (:ref:`objective <objective_key>` / :ref:`noise_model <noise_model_key>`
  / :ref:`profile_objective <profile_objective_key>`).

   - ``chi_sq`` - Chi squared (Gaussian noise; sigma per point from the data's ``_SD`` column)
   - ``chi_sq_dynamic`` - Chi squared with sigma as a free parameter (Requires sigma__FREE in the model and the configuration file)
   - ``lognormal`` - Lognormal noise (Gaussian on the log10 scale; sigma per point from the data's ``_SD`` column)
   - ``laplace`` - Laplace (double-exponential) noise with the scale b as a free parameter (Requires b__FREE in the model and the configuration file)
   - ``neg_bin`` - Negative Binomial (Requires neg_bin_r set to a number in the configuration file i.e neg_bin_r = 2, Default = 24)
   - ``neg_bin_dynamic`` - Negative Binomial with r as a free parameter (Requires r__FREE in the model and the configuration file)
   - ``kl`` - Kullback-Leibler
   - ``sos`` - Sum of squares
   - ``sod`` - Sum of differences
   - ``norm_sos`` - Sum of squares, normalized by the value at each point,
   - ``ave_norm_sos`` - Sum of squares, normalized by the average value of the variable.

   This sets one noise model for the whole fit. To use a different noise model for
   particular observables, override them with :ref:`noise_model <noise_model_key>` keys.

  Default: chi_sq

  Example:

    * ``objfunc = chi_sq``


.. _objective_key:

**objective**
  The modern named objective key (requires :ref:`edition <edition>` ``>= 2``). It
  accepts the same per-point token vocabulary as the legacy ``objfunc``
  (``chi_sq`` / ``chi_sq_dynamic`` / ``lognormal`` / ``laplace`` / ``sos`` / ``sod`` /
  ``norm_sos`` / ``ave_norm_sos`` / ``neg_bin`` / ``neg_bin_dynamic``), plus ``score``
  (pass a single ``score`` value straight through, ignoring the data) and the modern-only
  ``lnnormal`` natural-log Gaussian synonym. Each token
  **desugars** to the equivalent per-point noise model on the
  :ref:`noise_model <noise_model_key>` engine -- e.g. ``objective = sos`` is
  ``noise_model = gaussian, sigma = fix_at 1``, ``objective = chi_sq`` is
  ``noise_model = gaussian, sigma = read_exp_file _SD``. The recommended modern form is
  a ``noise_model`` line directly; the tokens are kept as familiar synonyms. The
  desugared least-squares forms restore the statistically-proper ``1/2`` the legacy
  ``sos`` / ``norm_sos`` / ``ave_norm_sos`` drop -- the located best fit is identical,
  only the reported objective value is halved. Column-joint objectives (``kl`` /
  ``wasserstein``) go under :ref:`profile_objective <profile_objective_key>` instead.

  The ``objective`` key also names a **closed-form analytical objective** with no model
  file or simulator (see :ref:`Analytical and user-defined objectives
  <analytical_objectives>`): a built-in target (``objective = banana, a = 1, b = 100`` and
  the ``gaussian`` / ``rotated_gaussian`` / ``rotated_quartic`` / ``multimodal`` menu), an
  inline math :ref:`expression <expression_key>` (``objective = expression``), or a Python
  :ref:`callable <callable_key>` (``objective = callable``). These read no experimental data
  unless a :ref:`data <data_key>` key binds it.

  Per-observable :ref:`noise_model <noise_model_key>` overrides may accompany an
  ``objective``. Specify exactly one of ``objective`` / a whole-fit ``noise_model`` /
  ``profile_objective`` per fit (there is no implicit default under a modern edition).

  Example:

    * ``edition = 2``
    * ``objective = sos``


.. _expression_key:

**expression**
  The companion to ``objective = expression`` (requires :ref:`edition <edition>` ``>= 2``):
  a closed-form negative log-likelihood (or cost) written as **PEtab math** over the declared
  free parameters, with no model file and no simulator. The symbols bind to the free
  parameters **by name**; PEtab math uses ``^`` for exponentiation (not ``**``). With a
  :ref:`data <data_key>` key the expression also references the bound ``.exp`` column headers
  and is summed per data row (a per-observation NLL). Requires the optional PEtab/sympy extra
  (``pip install pybnf[petab]``). See :ref:`Analytical and user-defined objectives
  <analytical_objectives>`.

  Example:

    * ``objective = expression``
    * ``expression = 0.5*((1 - x1)^2 + 100*(x2 - x1^2)^2)``


.. _callable_key:

**callable**
  The companion to ``objective = callable`` (requires :ref:`edition <edition>` ``>= 2``): a
  ``<module>:<function>`` entry point to a Python callable computing the objective, the escape
  hatch for densities a single expression cannot express. The left side is an importable dotted
  module (``mypkg.mymodule``) or a file path (``path/to/model.py``); the right side is the
  function name. The function has the signature ``f(params, data=None) -> float`` -- ``params``
  is the ``{name: value}`` parameter dict (bind-by-name), ``data`` the bound experimental data
  (see :ref:`data <data_key>`) or ``None`` -- and returns the scalar cost. The entry point is
  resolved and validated at config load. A general callable is not differentiable, so this works
  with the gradient-free algorithms but not :ref:`hmc <alg-hmc>`. See :ref:`Analytical and
  user-defined objectives <analytical_objectives>`.

  Example:

    * ``objective = callable``
    * ``callable = mymodule:negative_log_likelihood``
    * ``callable = path/to/model.py:negative_log_likelihood``


.. _data_key:

**data**
  Binds experimental data to a bring-your-own analytical objective (``objective = expression``
  or ``objective = callable``; requires :ref:`edition <edition>` ``>= 2``). The value is a comma
  list of ``.exp`` files, **each one experiment**, presented to the objective as a
  ``{experiment_name: Data}`` mapping keyed by file stem. A ``callable`` receives the whole
  mapping as its ``data`` argument and reduces it however it likes; an ``expression`` references
  the data columns by header and is summed per row over every experiment. The ``data`` key is
  valid only with these two objectives (any other objective binds data through a model /
  experiment). See :ref:`Analytical and user-defined objectives <analytical_objectives>`.

  Examples:

    * ``data = dose_response.exp``
    * ``data = replicate1.exp, replicate2.exp``


.. _profile_objective_key:

**profile_objective**
  A modern **column-joint** objective key (requires :ref:`edition <edition>` ``>= 2``):
  it compares the *shape* of a whole observable column at once, rather than scoring
  each point independently.

   - ``kl`` - Kullback-Leibler (the multinomial cross-entropy of the normalized profile)
   - ``wasserstein`` - the 1-Wasserstein (earth-mover) distance between the normalized
     simulated and experimental profiles, over the row index (unit spacing)

  Specify exactly one of ``objective`` / a whole-fit ``noise_model`` /
  ``profile_objective`` per fit; a column-joint objective does not take per-observable
  ``noise_model`` overrides.

  Example:

    * ``edition = 2``
    * ``profile_objective = wasserstein``


**noise_location**
  The whole-fit default for which summary of the noise distribution the model
  prediction is taken to be -- ``mean`` or ``median`` -- applied to the
  :ref:`objfunc <objective>`'s noise model (the analog of the per-observable
  ``location`` field on a :ref:`noise_model <noise_model_key>` key, which overrides
  it). ``median`` (the default when unset) means the prediction is the
  distribution's median; ``mean`` means its expected value. The two differ only for
  a ``lognormal`` observable (where ``mean`` adds the moment correction
  ``mu = log10(prediction) - sigma**2*ln10/2``) and a ``neg_bin`` observable. Only
  valid with a likelihood ``objfunc`` (``chi_sq`` / ``lognormal`` / ``laplace`` /
  ``neg_bin`` / ...). ``neg_bin`` is parameterized directly by its mean, so ``mean``
  is redundant; ``median`` interprets the prediction as the count distribution's
  0.5-quantile, solved for by a per-point CDF inversion (issue #419).

  Example:

    * ``objfunc = lognormal``
    * ``noise_location = mean``


.. _noise_profiling:

**noise_profiling**
  Profile every **estimated noise scale** out of the search analytically instead of
  fitting it as an ordinary free parameter (``1`` to enable; default ``0``, off).
  Applies to each noise parameter declared ``= fit <parameter>`` (a scale that *is* a
  free parameter). At every evaluation such a scale is replaced by its closed-form
  maximum-likelihood value over the scored points that share it -- the weighted residual
  RMS ``sqrt(sum w r**2 / sum w)`` for a ``normal`` / ``lognormal`` / ``lnnormal``
  observable, the weighted mean absolute residual ``sum w |r| / sum w`` for ``laplace``
  -- so those dimensions leave the search entirely and every parameter set the fit
  evaluates is scale-optimal by construction.

  Why it helps: at a random point in the box the sampled scale is nowhere near its
  optimum, so the ``log sigma`` term dominates and a global search ranks candidates
  mostly by *how wrong their sigma happens to be* rather than by how well their dynamics
  fit. It also removes 1 to k search dimensions, and a profiled scale has no box to run
  into (a searched one can optimize into its upper bound and silently absorb model
  misfit as "measurement noise").

  The switch is **all-or-nothing** within a fit, and is refused before the run starts --
  naming the reason -- when any estimated scale has no closed-form profile: a
  ``formula`` / ``prediction_formula`` / per-measurement sigma, a ``student_t`` ``df``,
  the ``neg_bin`` dispersion, or a ``location = mean`` prediction on a log scale. A
  **fixed** scale (a data column, ``fix_at``, ``relative``) is not searched, so it is
  simply left alone. Also refused for the Bayesian samplers (``mcmc``, ``pt``, ``am``,
  ``dream``, ...): profiling *maximizes* the nuisance out where a posterior *integrates*
  it out, so the draws would not be posterior draws.

  The profiled parameters must still be **declared** as free parameters (the same
  ``.conf`` runs with and without the key); their bounds and prior simply become inert.
  They remain estimated quantities, so they keep counting in ``k`` in
  ``Results/information_criteria.txt``, and their fitted values are written to
  ``Results/profiled_noise.txt`` -- they are not coordinates of the best parameter set,
  so they appear in no ``sorted_params_*.txt`` row.

  Supported by every optimizer, including ``lbfgs`` and ``gntr``. ``job_type = trf``
  refuses a profiled fit (as it already refuses a searched free scale): under profiling
  the least-squares residual norm is constant, so a trust-region residual model carries
  no information -- use ``lbfgs``. It is the recommended pairing for :ref:`multiple
  shooting <multiple_shooting>` (``job_type = ms``): a profiled scale is defined by the
  data residuals alone, which continuity defects never enter, so the reported objective
  stays comparable to a single-shoot one.

  See :ref:`normalization <normalization_key>` ``= scale`` for the same trick applied to
  an unknown multiplicative scale on the data.

  Example::

    edition = 2
    job_type = cmaes
    noise_model = normal, sigma = fit sigma_obs
    loguniform_var = sigma_obs 1e-3 1e3   # still declared; bounds now inert
    noise_profiling = 1


.. _noise_model_key:

**noise_model**
  A per-point noise model, either for the **whole fit** (no observable -- the modern
  replacement for ``objfunc``, requires :ref:`edition <edition>` ``>= 2``) or as an
  **override for a single observable** (with an observable name), so different
  observables in one fit can use different noise models. The whole-fit line (or the
  legacy ``objfunc``) is the default for every observable not named by a per-observable
  ``noise_model``. Each key names the distribution family and, for each of the family's
  noise parameters, where its value comes from::

    noise_model [<observable>] = <family>, <parameter> = <source>[, <parameter> = <source> ...][, location = mean|median][, cumulative]

  The **family** is one of ``normal``, ``lognormal``, ``lnnormal``, ``laplace``, ``neg_bin``, or
  ``student_t``. Each **parameter** is named by its standard statistical name --
  ``sigma`` for ``normal`` / ``lognormal`` / ``lnnormal``, ``scale`` for ``laplace``, ``dispersion``
  for ``neg_bin``, and ``sigma`` plus ``df`` for ``student_t`` (the only
  two-parameter family). Each **source** is one of:

   - ``read_exp_file <suffix>`` - read it per point from the experimental data
     column ``<observable><suffix>`` (conventionally ``_SD``).
   - ``fit <name>__FREE`` - estimate it as a free parameter, declared the usual way
     (e.g. ``uniform_var = <name>__FREE <lower> <upper>``).
   - ``fix_at <number>`` - hold it at a fixed numeric constant.
   - ``relative [<cv>]`` - constant coefficient of variation: ``sigma = cv * |value|``,
     so the noise scales with the measurement (``cv`` defaults to 1). This is the
     heteroscedastic model the legacy ``norm_sos`` fits.
   - ``column_mean`` - ``sigma`` is the observable's experimental column mean (one
     scale per column). This is the model the legacy ``ave_norm_sos`` fits.
   - ``formula <expr>`` - an arithmetic expression over free parameters (and constants),
     evaluated per point against the current fit; the PEtab ``noiseFormula`` source.
   - ``prediction_formula <expr>`` - an expression whose ``sigma`` scales with the
     **simulated output**: the combined additive+proportional error model
     ``sigma = sd_abs + sd_rel * <observable>`` where ``<observable>`` is a model
     species/observable/function read from the current simulation and ``sd_abs`` / ``sd_rel``
     are estimated free parameters. Use this (not ``formula``) whenever the noise scales with
     the prediction. Score path only -- a gradient/EFIM fit is unsupported for this source.

  The **student_t** family is the heavy-tailed, outlier-robust likelihood (robust
  regression) -- a ``normal`` with a tail-heaviness knob ``df`` (degrees of freedom):
  small ``df`` gives fat tails that downweight outliers, and ``df`` toward infinity
  recovers the Gaussian. Both of its parameters are sourced independently, so a fit may
  estimate 0, 1, or 2 of them (e.g. ``sigma = fit s__FREE`` with a fixed ``df``, or both
  free). ``df`` is the one parameter that may be **omitted**: it then defaults to a fixed
  ``4`` (the standard robust default), so ``noise_model = student_t, sigma = fix_at 0.7``
  is a valid robust fit. Estimating ``df`` (``df = fit nu__FREE``) is weakly identified,
  so pair it with a positive prior on ``nu__FREE`` (e.g. ``gamma_var`` / ``half_normal_var``).

  The optional **location** field sets which summary of the noise distribution the
  model prediction is taken to be: ``median`` (the default -- the prediction is the
  distribution's median, matching PEtab) or ``mean`` (the prediction is its
  expected value). The two differ for a ``lognormal`` / ``lnnormal`` observable, where
  ``mean`` adds the moment correction ``mu = log10(prediction) - sigma**2*ln10/2`` or
  ``mu = ln(prediction) - sigma**2/2`` respectively
  (the symmetric families are unaffected). ``neg_bin`` is parameterized directly by
  its mean, so ``location = mean`` is redundant (accepted); ``location = median``
  interprets the prediction as the count distribution's 0.5-quantile, solved for by a
  per-point continuous-CDF inversion (issue #419).

  The optional **cumulative** flag (per-observable only) declares the observable a
  *cumulative* count: its simulated prediction is differenced row-to-row (cumulative
  total -> per-interval increment, with the first row kept as-is) before scoring. It is
  a prediction transform, independent of the noise family, so it pairs with any family
  (e.g. ``normal``, ``laplace``, ``neg_bin``). Legacy configs that relied on the older
  convention -- a data column whose name contains ``_Cum``, recognized only by
  ``objfunc = neg_bin_dynamic`` -- keep working unchanged; the explicit ``cumulative``
  flag is the family-independent replacement (issue #418). A cumulative observable
  cannot be exported to PEtab (PEtab has no cumulative-count operator).

  Examples:

    * ``noise_model = gaussian, sigma = fix_at 1`` (whole-fit default; ``edition = 2``)
    * ``noise_model obs2 = laplace, scale = fit b_obs2__FREE``
    * ``noise_model obs3 = normal, sigma = read_exp_file _SD``
    * ``noise_model obs4 = neg_bin, dispersion = fix_at 10``
    * ``noise_model obs5 = lognormal, sigma = read_exp_file _SD, location = mean``
    * ``noise_model obs_ln = lnnormal, sigma = read_exp_file _SD`` (natural-log Gaussian)
    * ``noise_model cases = neg_bin, dispersion = fit r__FREE, cumulative``
    * ``noise_model obs6 = student_t, sigma = fit s__FREE`` (robust; ``df`` defaults to 4)
    * ``noise_model obs7 = student_t, sigma = fit s__FREE, df = fit nu__FREE`` (estimate both)
    * ``noise_model obs8 = gaussian, sigma = formula 0.1 + 0.05 * cv__FREE`` (sigma an
      expression over free parameters)
    * ``noise_model obs9 = gaussian, sigma = prediction_formula sd_abs__FREE + sd_rel__FREE * obs9``
      (combined additive+proportional error: sigma scales with the *predicted* value of ``obs9``)


.. _edition:

**edition**
  An optional integer that opts the .conf into a frozen set of modernized PyBNF
  conventions. Editions are *select-and-freeze*: a conf written for ``edition = 2``
  is interpreted under edition-2 conventions forever, even as later PyBNF releases
  change other defaults under higher editions, so upgrading PyBNF never silently
  reinterprets your existing config. Omitting the key selects *legacy* behavior
  (the implicit edition 1), byte-identical to PyBNF's historical defaults; the
  newest syntax requires opting in with an explicit ``edition``. The value is a
  plain integer, decoupled from PyBNF release numbers -- editions change only when
  a convention changes.

  Under a modern edition (``edition >= 2``) the objective is named through the modern
  three-key surface -- :ref:`objective <objective_key>` (or a whole-fit
  :ref:`noise_model <noise_model_key>` line) for per-point noise models, or
  :ref:`profile_objective <profile_objective_key>` for column-joint ones -- and the
  legacy :ref:`objfunc <objective>` key is rejected. Exactly one objective must be
  named; there is no implicit default.

  Also under a modern edition the universal default for prediction centering is the
  **median** (consistent with PEtab v2). This is byte-identical for the location-scale
  noise models (``chi_sq`` / ``lognormal`` / ``laplace``), which already default to the
  median. The one place the number differs is ``neg_bin``, whose legacy default was the
  mean: under a modern edition a ``neg_bin`` fit with no explicit location resolves to
  the median (a per-point CDF inversion, issue #419) and **warns**, since the value
  changes from legacy and median ``neg_bin`` is rarely intended -- set
  :ref:`noise_location <objective>` (``= mean`` to keep the legacy behavior, or
  ``= median`` to silence the warning) explicitly.

  Default: unset (legacy, edition 1)

  Example:

    * ``edition = 2``


**population_size**
  The number parameter sets to maintain in a single iteration of the algorithm. See algorithm descriptions for more
  information. This is also the key that decides how many simulations can run at the same time, so on a cluster it
  should be chosen against the number of processors reserved -- see :ref:`Sizing a run <sizing>`.

  Example:
  
    * ``population_size = 50``

**max_iterations**
  Maximum number of iterations

  Example:

    * ``max_iterations = 200``

**n_starts**
  Number of independent multi-start runs for the metaheuristic optimizers (``de``, ``ade``, ``ss``, ``pso``). A single run collapses its population into one basin, so on a multimodal objective it returns only a local minimum. With ``n_starts > 1``, that many independent searches are run one after another -- each a fresh random / Latin-hypercube population, each up to ``max_iterations`` iterations or until it converges -- and the best fit over all of them is kept. ``1`` (the default) is a single run, identical to the historical behavior. (``cmaes`` has its own multimodal restart, ``cmaes_restarts``; the gradient optimizers use ``population_size`` as their start count.)

  Example:

    * ``n_starts = 10``


Other Path Keys
---------------

.. _bng_command:

**bng_command**
  Path to BNG2.pl, including the BNG2.pl file name. This key is required if your fitting includes any .bngl files,
  unless the BioNetGen path is specified with the BNGPATH env variable.

  Default: Uses the BNGPATH environmental variable

  Example:
  
    * ``bng_command = path/to/BNG2.pl``


.. _bngl_backend:

**bngl_backend**
  Backend selection for BNGL simulations. Options are ``auto``, ``bionetgen``, or ``bngsim``. With ``auto``,
  PyBNF uses BNGsim for supported BNGL network and NFsim paths when BNGsim is available, and otherwise uses the
  BioNetGen subprocess path. Use ``bionetgen`` to force the legacy BioNetGen path, or ``bngsim`` to require BNGsim
  and fail if the model's actions are unsupported by the BNGsim bridge. Setting the environment variable
  ``PYBNF_NO_BNGSIM=1`` disables BNGsim auto-selection.

  BNGL workflows still need ``bng_command`` when PyBNF must run BNG2.pl to generate ``.net`` or XML files.

  Default: auto

  Example:

    * ``bngl_backend = bionetgen``


.. _stochastic_seed:

**stochastic_seed**
  Policy controlling how PyBNF supplies RNG seeds to stochastic simulations
  (``ssa``, ``psa``, NFsim, RuleMonkey) on the BNGsim backend. Affects BNGL
  ``.net`` / ``.xml``, SBML, and Antimony models. Four modes:

    * ``auto`` *(default)* — PyBNF derives a deterministic 31-bit seed from
      the evaluation context (parameter set, model name, action index, suffix,
      method, smoothing replicate index). Same evaluation reproduces the same
      trajectory; distinct evaluations get distinct seeds. Any explicit
      ``seed=>N`` written in a BNGL action is **overridden** with a one-time
      warning per (model, action) at fit start.
    * ``auto_honorbngl`` — Same derivation as ``auto``, but explicit BNGL
      ``seed=>N`` is honored verbatim for that one action.
    * ``random`` — PyBNF passes no seed; BNGsim draws fresh entropy
      (``secrets.randbits(31)``) per call. Each run produces different
      trajectories. Explicit BNGL seeds are overridden with a warning.
    * ``random_honorbngl`` — Random by default, but explicit BNGL ``seed=>N``
      is honored verbatim for that one action.

  The default ``auto`` is recommended for fitting workflows: it gives a
  well-defined stochastic objective (same parameter point → same chi²) and
  makes failed fits reproducible. Use ``random`` for one-shot exploratory
  Monte Carlo runs where you want fresh entropy each invocation. The
  ``_honorbngl`` variants are escape hatches for power users with
  deliberate per-action explicit seeds.

  Under the ``_honorbngl`` modes, combining ``smoothing > 1`` with a model
  that contains an explicit BNGL ``seed=>N`` is rejected at config load,
  because it would cause every smoothing replicate to produce the same
  trajectory.

  Default: auto

  Example:

    * ``stochastic_seed = random``


**output_dir**
  Directory where we should save the output.

  Default: "pybnf_output"

  Example:
  
    * ``output_dir = dirname``


Parameter and Model Specification
---------------------------------
**mutant**
  Declares a model that does not have its own model file, but instead is defined based on another model (the "base model"), changing only a small number 
  of parameter values. The first word of the declaration gives the name of the base model (not including the path or  .bngl/.xml extension).
  The second word is the name of the mutant model; this name is appended to the suffixes
  of the base model. That is, if the base model has data files ``data1.exp`` and ``data2.exp``, a corresponding mutant
  model with the name  "m1" should use the files ``data1m1.exp`` and ``data2m1.exp``. Following the name of the mutant
  model is a series of statements that specify how to change ``basemodel`` to make the mutant model. The statements 
  have the format [variable][operator][value] ; for example ``a__FREE=0`` or ``b__FREE*2``. Supported operators are 
  ``=``, ``+``, ``-``, ``*``, ``/``.

  Default: None

  Example:
    
    Elsewhere in your .conf file, you have specified model1:
    
      * ``model = path/to/model1.bngl : data1.exp``
    
    Then you can use this key as follows:
    
      * ``mutant = model1 no_a a__FREE=0 : data1no_a.exp, data2no_a.exp``
      * ``mutant = model1 extra_ab a__FREE*2 b__FREE*2 : data1extra_ab.exp``

**uniform_var**
  A bounded uniformly distributed variable defined by a 3-tuple corresponding to the variable name, minimum
  value, and maximum value. If the tag ``U`` is added to the end, the bounds are enforced only during initialization, 
  not during fitting. 

  Examples:
  
    * ``uniform_var = k__FREE 10 20``
    * ``uniform_var = k__FREE 10 20 U``

**normal_var**
  A normally distributed variable defined by a 3-tuple: the name, mean value, and standard deviation. The distribution
  is truncated at 0 to prevent negative values

  Example:
  
    * ``normal_var = d__FREE 10 1``

**loguniform_var**
  A variable distributed uniformly in logarithmic space. The value syntax is identical to the **uniform_var** syntax

  Examples:
  
    * ``loguniform_var = p__FREE 0.001 100``
    * ``loguniform_var = p__FREE 0.001 100 U``

**lognormal_var**
  A variable normally distributed in logarithmic space.  The value syntax is a 3-tuple specifying the variable name,
  the base 10 logarithm of the mean, and the base 10 logarithm of the standard deviation

  Example:

    * ``lognormal_var = l__FREE 1 0.1``

**Additional prior families**
  Beyond the four above, PyBNF ships a catalog of single- and two-parameter ``scipy.stats``-backed
  prior families, each with a ``<family>_var`` keyword (and ``log<family>_var`` for the same prior in
  base-10 log space). The value is the variable name followed by the family's parameters, **already in
  the parameter's scale**:

    * one-parameter — ``exponential`` (scale), ``chisquare`` (dof), ``rayleigh`` (scale),
      ``half_normal`` (scale), ``half_cauchy`` (scale): e.g. ``half_normal_var = sigma__FREE 2``
    * two-parameter — ``cauchy`` (location, scale), ``laplace`` (location, scale), ``gamma`` (shape,
      scale), ``inv_gamma`` (shape, scale), ``weibull`` (shape, scale), ``gumbel`` (location, scale),
      ``logistic`` (location, scale), ``beta`` (alpha, beta): e.g. ``beta_var = frac__FREE 2 5``

  ``half_normal`` / ``half_cauchy`` are the standard weakly-informative *scale* priors; ``beta`` is for
  a fraction or probability on ``[0, 1]``; ``inv_gamma`` is the conjugate variance prior; ``student_t``
  (below) is the heavy-tailed *robust* prior.

**student_t** (new-era ``parameter:`` record only)
  The Student-t prior is parameterized by **three** numbers — ``df`` (degrees of freedom; small values
  give fatter tails, ``df → ∞`` approaches a Normal), ``location``, and ``scale`` — one more than the
  positional ``*_var`` grammar can carry. It is therefore declared with the new-era labeled
  ``parameter:`` record (requires :ref:`edition <edition>` ``>= 2``), which names every field:

    * ``parameter: x__FREE, prior: student_t, df: 4, location: 0, scale: 2.5``
    * ``parameter: k__FREE, prior: student_t, parameter_scale: log10, df: 3, location: 1, scale: 0.5, lower: 0.1, upper: 100``

  (A family's own ``scale`` field is its distribution parameter, distinct from the record's
  ``parameter_scale`` sampling-space transform.) Every family above is *also* expressible through this
  record form (``prior: <family>``, with the family's parameters as named fields); the positional
  ``*_var`` keyword is the legacy shorthand for the one- and two-parameter families.


The following two keys (``var`` and ``logvar``) are the single-value **start point** used by the
start-point optimizers — :ref:`Simplex <alg-sim>`, :ref:`Powell <alg-powell>`, :ref:`CMA-ES <alg-cmaes>`,
the gradient methods :ref:`trf / lbfgs / gntr <alg-gradient>`, and multiple shooting (``ms``).

Each of those ``job_type``\ s accepts **either** style, and not a mix:

* every parameter given a single value with ``var`` / ``logvar`` — a purely local search from
  that point; or
* every parameter given a **bounded** prior (``uniform_var`` / ``loguniform_var``, or any family
  truncated to a finite box with ``lower:``/``upper:`` on a ``parameter:`` record) — a search over
  that box, beginning at its centre unless you name a different start with
  :ref:`start_point <start_point>`.

An **unbounded** prior (``normal_var``, ``gamma_var``, …) is refused for these ``job_type``\ s:
a box search needs a box. For any other algorithm, define parameters with the prior-based
specifications, not ``var`` / ``logvar``.

The rule is about what each parameter *is*, not how it is spelled: a ``parameter:`` record with
no ``prior:`` and no ``lower:``/``upper:`` is a start point exactly as ``var`` is, and is accepted
and refused in the same places.

When refining a result (``refine = 1``), the optimizer is chosen by ``refine_method`` (``sim``
(default), ``powell``, or ``cmaes``); it reads that optimizer's own settings (e.g. ``simplex_step``
for Simplex), so you do not need to add ``var`` / ``logvar`` lines.

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

  Example:

    * ``logvar = k__FREE -3 1``

.. _start_point:

**start_point**
  The point the fit starts from, one line per parameter: the parameter's name followed by its value.
  Use it to seed a run at a published optimum, at a PEtab ``nominalValue``, or at any point you want to
  test the local behaviour of a method around — while keeping the box and priors declared above.

  The value is always in the parameter's **own units**, never in log space, whatever its scale: a
  ``loguniform_var`` or ``parameter_scale: log10`` parameter takes ``start_point = k 0.017``, not
  ``-1.77``. (A non-positive value on a log-scaled parameter is an error, since that is the likely
  symptom of writing the log.)

  It is **partial by design**. Name only the parameters you want to pin; every other parameter starts
  exactly where it would have without the key — at the centre of its box, or at its ``var`` / ``logvar``
  value. For a multi-start fit (``population_size`` / ``n_starts`` greater than 1) the declared point is
  start 0 and the remaining starts stay independent draws, so the scatter that makes a multi-start
  worth running is preserved; use ``population_size = 1`` for a single run from exactly this point.

  A value outside the parameter's declared bounds is **refused**, not moved. PyBNF does not clamp an
  out-of-box value to the nearest bound — it reflects it back inside, landing on an arbitrary interior
  point — so a start point that would be moved is reported as a configuration error instead.

  Where the fit actually started is recorded in ``Results/start_point.txt``, with each parameter's
  value, its source, and the box it was checked against.

  Equivalent to the ``initial_value:`` field of a new-era ``parameter:`` record (see
  :ref:`edition <edition>`), which states the same fact inline; ``start_point`` works at every edition
  and alongside the legacy ``*_var`` declarations. Declaring both for one parameter is fine if they
  agree and an error if they disagree.

  One exception to the equivalence: for ``job_type = profile_likelihood``, an ``initial_value:``
  given for **every** parameter means those values *are* the optimum θ\* to profile around, and the
  polish is skipped. ``start_point`` never carries that meaning — it names where the polish starts,
  and the polish still runs — so a value you believe is optimal should be written as
  ``initial_value:`` there.

  Default: None (the box centre, or the ``var`` / ``logvar`` point)

  Examples:

    * ``start_point = k__FREE 0.017``
    * ``start_point = kcat 17.5``

Simulation Actions
------------------

These keys specify what simulations should be performed with the models. For SBML models, simulation actions are required. For BNGL models, the same information can be specified in the actions block of the BNGL file, so use of these keys is optional.

.. note::

   For BNGL models, we recommend specifying simulation actions in the BNGL file's ``begin actions`` block rather than in the configuration file. The BNGL actions block supports the full set of BioNetGen action arguments (e.g., ``steady_state``, ``atol``, ``rtol``, ``sparse``, ``continue``, ``stop_if``), whereas the configuration file keys below only support a subset. Configuration file actions are primarily intended for SBML models, which have no native action syntax.

.. _time_course_key:

**time_course**
  Run a time course simulation on the model. Specify a comma-delimited list of ``key:value`` pairs, with the following possible keys:
  
    * ``time``: The simulation time. Required.
    * ``suffix``: The suffix of the data file to save. You should map the model to a .exp file of the same name. Default: time_course
    * ``step``: The simulation time step. Default: 1
    * ``model``: The name of the model to run (not including the path or .bngl/.xml extension). Default: All models in the fitting run.
    * ``subdivisions``: Only for use with ``sbml_integrator=euler``, specifies the number of internal Euler steps to perform between each output step specified by ``step``. Default: 1
    * ``method`` The simulation method to use. Default is ``ode``. Options are:
    
       * ``ode``: Numerical integration of differential equations
       * ``ssa``: Stochastic simulation algorithm (BioNetGen's "ssa" algorithm for BNGL models; Gillespie's direct method for SBML models)
       * ``pla``: Partitioned-leaping algorithm (BNGL models only)
       * ``nf``: Network-free simulation with NFsim (BNGL models only)
  
  Example:
  
    * ``time_course = time:60, model:model1, suffix:data1``

.. _param_scan_key:

**param_scan**
  Run a parameter scan on the model. Specify a comma-delimited list of ``key:value`` pairs, with the following possible keys:
  
    * ``param``: Name of the parameter to scan. Required.
    * ``min``: Minimum value of the parameter. Required
    * ``max``: Maximum value of the parameter. Required. 
    * ``step``: Change in the parameter value between consecutive simulations in the scan. Required.
    * ``time``: The simulation time. Required.
    * ``suffix``: The suffix of the data file to save. You should map the model to a .exp file of the same name. Default: param_scan
    * ``logspace``: If 1, take ``step`` to be in log (base 10) space, and scan the parameter in log (base 10) space. Default: 0
    * ``model``: The name of the model to run (not including the path or .bngl/.xml extension). Default: All models in the fitting run.
    * ``subdivisions``: Only for use with ``sbml_integrator=euler``, specifies the number of internal Euler steps to perform for each simulation. Default: 1000
    * ``method``: The simulation method to use. Options are the same as in ``time_course``. Default: ode
  
  Example:
  
    * ``param_scan = param:x, min:1, max:1000, step:0.5, logspace:1, time:60, model:model1, suffix:data1``


Parallel Computing
------------------
**parallel_count**
  The number of jobs to run in parallel. This may be set for both local and cluster fitting runs. For cluster runs, this number is divided by the number of available nodes (and rounded up) to determine the number of parallel jobs per node.

  Each parallel job runs in its own **single-threaded worker process**, whether or not this key is set: the simulation backends hold process-wide state that is not thread-safe, so PyBNF never places two concurrently running jobs in one process. This key therefore sets a process count, not a thread count. Lowering it is the way to reduce the memory a run uses, since each worker process holds its own copy of the models.

  Default: Use all available cores -- one single-threaded worker per core. Locally, the core count comes from Dask, which honors CPU affinity and cgroup quotas (so a run confined to 4 cores gets 4 workers, not the host's full count). On a cluster, with either ``cluster_type``, it is the number of CPUs the job was granted on a node (``$SLURM_CPUS_ON_NODE``), falling back to the affinity- and cgroup-aware count Dask derives and then to the machine's whole processor count; PyBNF logs the number it used and which of the three it came from. See :ref:`How many workers run on each node <workercount>`.

  Example:
  
    * ``parallel_count = 7``

**cluster_type**
  Type of cluster used for running the fit. This key may be omitted, and instead specified on the command line with the
  ``-t`` flag. Currently supports ``slurm``, ``slurm-srun``, or ``none``.

  ``slurm`` starts the workers by logging in to each allocated node over SSH. ``slurm-srun`` starts them with SLURM's
  ``srun`` command instead, which requires no login and therefore works on clusters whose nodes authenticate to each
  other by host-based or Kerberos SSH -- where the SSH route cannot work at all. It must be run from the shell that
  holds the allocation. See :ref:`Starting workers without SSH <srun>`.

  Default: None (local fitting run).

  Example:
  
    * ``cluster_type = slurm``
    
**parallelize_models**
  For fitting jobs that include multiple models, run those models on different cores, utilizing a total of this number of cores per parameter set evaluation. 
  Should not be set higher than the total number of models. Using this option incurs additional communication overhead, and causes the objective function
  to be evaluated locally, not in parallel. Therefore, only certain types of problems will benefit from this option. This option can be used with
  ``smoothing``; PyBNF will partition the model list for each smoothing replicate, merge the model results, then average the replicates.
  
  Default: 1
  
  Example:
  
    * ``parallelize_models = 3``

**scheduler_file**
  Provide a scheduler file to link PyBNF to a Dask scheduler already created outside of PyBNF. See :ref:`Manual configuration with Dask <manualdask>` for more information. 
  This option may also be specified on the command line with the ``-s`` flag. 

  With ``cluster_type = slurm-srun``, PyBNF starts the scheduler itself, so this key instead chooses the path that
  scheduler writes its connection information to. It must be on a filesystem the compute nodes share. Default in that
  case: ``dask_scheduler.json`` inside ``output_dir``.
  
  Default: None
  
  Example: 
  
    * ``scheduler_file = cluster.json``

**scheduler_node**
  Manually set node used for creating the distributed Client -- takes a string identifying a machine on a network. If
  running on a cluster with SLURM, it is recommended to use :ref:`automatic configuration <cluster>` with the flag
  ``-t slurm`` instead of using this key.

  Default: None

  Example:
  
    * ``scheduler_node = cn180``

**simulation_dir**
  Optional setting for a different directory where we should save (or temporarily store) simulation output. Usually
  not necessary to set separately from `output_dir`, and on an ordinary shared network filesystem there is no reason
  to. It is worth setting when `output_dir` is on a parallel filesystem such as Lustre or GPFS, which handles the many
  small files a fit creates and deletes worst; point this key at storage better suited to that traffic and only the
  results are written to the parallel filesystem. It must exist and be writable on every node, and should be on shared
  storage -- see :ref:`Where simulation files are written <simdir>`.

  Default: Use the same directory as `output_dir`.
  
  Example:
  
    * ``simulation_dir = /scratch/sim_output``

**worker_nodes**
  Manually set nodes used for computation - takes one or more strings separated by whitespace identifying machines on a
  network. If running on a cluster with SLURM, it is recommended to use :ref:`automatic configuration <cluster>` with
  the flag ``-t slurm`` instead of using this key.

  Default: None

  Example:
  
    * ``worker_nodes = cn102 cn104 cn10511``

General Options
---------------

Output Options
^^^^^^^^^^^^^^
**delete_old_files**
  Takes an integer for a value.  If 1, delete simulation folders immediately after they complete. If 2, delete both
  old simulation folders and old sorted_params.txt result files. If 0, do not delete any files (warning, could consume
  a large amount of disk space).

  Default: 1

  Example:
  
    * ``delete_old_files = 2``

**num_to_output**
  The maximum number of parameter sets to output when writing the trajectory to file. The parameter sets are ordered
  by their corresponding objective function value to ensure the best fits are outputted.

  Default: 5000

  Example:
  
    * ``num_to_output = 100000``

**output_every**
  The number of iterations in between consecutive events writing the trajectory to file.

  Default: 20

  Example:

    * ``output_every = 1000``

**backup_every**
  The number of iterations between writes of the run's checkpoint — the saved state that a
  ``-r`` :ref:`resume <config>` reads to continue an interrupted run. A larger value checkpoints
  less often, trading resume granularity for lower I/O.

  Default: 1

  Example:

    * ``backup_every = 10``

**backup_information_criteria**
  If 1, each checkpoint also writes ``Results/information_criteria_backup.txt`` — the AIC/BIC/AICc
  and full normalized log-likelihood of the best fit so far — alongside
  ``Results/sorted_params_backup.txt``. Both halves of a scoreable result are then on disk
  throughout the run, so a fit that is still going, or that was killed or crashed, can be scored
  from its own artifacts; without it, ``Results/information_criteria.txt`` appears only when the
  run terminates normally. The checkpoint file has the same format as the final one, with extra
  ``#`` comment lines marking it as a snapshot of a run in progress; the final artifact is
  unchanged.

  Cost is one extra simulation per checkpoint, and only when the checkpoint has something new to
  report: nothing is spent while the best fit is unchanged (the file already describes it), and
  nothing is spent at all unless the objective is a proper likelihood, since no information
  criterion is defined otherwise. The cadence is ``backup_every``. Set this to 0 for a model
  where even that is too expensive.

  Default: 1

  Example:

    * ``backup_information_criteria = 0``

**save_best_data**
  If 1, run an extra simulation at the end of fitting using the best-fit parameters, and save the best-fit .gdat and .scan files to the Results directory. 
  
  Default: 0

  Example:

    * ``save_best_data = 1``

**embed_best_fit_data**
  Opt-in, new-era (``edition = 2``) only. When 1, the end-of-run ``Results/<model>_bestfit.bngl`` artifact additionally embeds each time-indexed observable's experimental data **inline** as a ``tfun([t...],[y...], time)`` reference function (ADR-0054; was a sidecar ``.tfun`` file under ADR-0048), so the saved model self-contains its comparison curves in one file. ``tfun`` is a bngsim feature (BNG2.pl parses no ``tfun`` form), so the embedded overlay is read through a bngsim path. A no-op in the legacy edition and when unset.

  Default: 0

  Example:

    * ``embed_best_fit_data = 1``

**smooth_plot_points**
  Opt-in, new-era (``edition = 2``) only. In the new era a fitting job's output times come from the data, so the end-of-run ``Results/<model>_bestfit.bngl`` artifact reproduces a *ragged* trajectory (only the measured time points). When set to a positive integer N, the artifact's data-derived time-course actions are re-rendered onto a uniform grid of N output steps over ``[t_start, t_max]`` instead of the data's ``sample_times``, so running the artifact yields a smooth plot curve. This affects only the saved artifact (a post-fit re-render) -- never the fit itself -- and is honored by both BNG2.pl and bngsim. Parameter-scan (dose-response) actions and the steady-state pre-equilibration phase are left untouched. A no-op (ragged grid, as authored) when 0.

  Default: 0

  Example:

    * ``smooth_plot_points = 500``

.. _output_inference_data:

**output_inference_data**
  Opt-in, MCMC fits only (``am`` / ``dream`` / ``p_dream`` / ``pt`` / ``mh``). When 1, the end of a Bayesian sampler run also writes ``Results/inference_data.nc`` -- an `ArviZ <https://python.arviz.org>`_ ``InferenceData`` built from the saved ``Results/samples.txt`` (ADR-0055) -- so the posterior is ready for the ArviZ / bayesplot / loo ecosystem (trace, rank, forest, pair plots, ``az.summary``, ``az.compare``) with no extra step. (Adaptive MCMC (``am``) records its draws per chain in ``Results/A_MCMC/Runs/params_*.txt`` rather than ``samples.txt``; the bridge reads those directly, and -- since ``am`` records draws only, not a per-draw log-posterior -- its object carries no ``sample_stats.lp`` group.) Load it with ``arviz.from_netcdf("Results/inference_data.nc")``, or build one post-hoc from any finished run with ``pybnf.inference_data.from_pybnf("path/to/Results")``. Log-scaled parameters are emitted in their sampling space (``log10`` / ``ln``, e.g. ``log10_k``), the space PyBNF samples and computes R-hat/ESS in; the ``posterior`` group carries one variable per parameter and (except for ``am``) ``sample_stats`` carries ``lp`` (the log-posterior). Because ``samples.txt`` is the thinned (by ``sample_every``), post-burn-in saved sample, ArviZ recomputes diagnostics on fewer draws than ``Results/diagnostics.txt``, so ``az.ess`` reads lower by design (PyBNF's own final R-hat/ESS ride along in the object's attributes); lower ``sample_every`` for denser ArviZ diagnostics. **For LOO/WAIC model comparison** (ADR-0056), when the fit uses a per-point likelihood objfunc (``chi_sq`` / ``chi_sq_dynamic`` / ``lognormal`` / ``laplace`` / ``neg_bin`` / ``neg_bin_dynamic``, or the ``objective`` / ``noise_model`` surface), the run also records ``Results/log_likelihood.txt`` and the InferenceData gains a ``log_likelihood`` group, so ``az.loo`` / ``az.waic`` / ``az.compare`` work directly on it. Its values are genuine, *unweighted* per-observation log-densities (not ``-score``). With a non-likelihood objfunc (least-squares, ``kl``, ``direct_pass``, ...) there is no normalized density, so no sidecar is written, the group is omitted, and a one-time note explains that LOO/WAIC needs a likelihood objfunc. Requires the optional ArviZ extra (``pip install pybnf[arviz]``); a no-op (with a log note) if the extra is absent, and on a non-sampler fit.

  Default: 0

  Example:

    * ``output_inference_data = 1``

**verbosity**
  An integer value that specifies the amount of information output to the terminal.
  
   - 0 - Quiet: User prompts and errors only
   - 1 - Normal: Warnings and concise progress updates
   - 2 - Verbose: Information and detailed progress updates

  Default: 1

  Example:
  
    * ``verbosity = 0``

Algorithm Options
^^^^^^^^^^^^^^^^^
  
**bootstrap**
  If assigned a positive value, estimate confidence intervals through a :ref:`bootstrapping <bootstrap>` procedure.  The assigned integer is the number of bootstrap replicates to perform.
  
  Default: 0 (no bootstrapping)
  
  Example:
  
    * ``bootstrap = 10``
    
**bootstrap_max_obj**
  The maximum value of a fitting run's objective function to be considered valid in the bootstrapping procedure. If a fit ends with a larger objective value, it is discarded.
  
  Default: None
  
  Example:
  
    * ``bootstrap_max_obj = 1.5``
    
**constraint_scale**
  Scale all weights in all .prop files by this multiplicative factor. For convenience only - The same thing could be achieved by editing .prop files, but this option is useful for tuning the relative contributions of quantitative and qualitative data.

  Default: 1 (no scaling)

  Example:

    * ``constraint_scale = 1.5``

**qualitative_loss**
  Global override for the qualitative (.prop / BPSL) penalty family. Forces every constraint in the fit to one model, deriving a scale-matched parameter from whatever each constraint authored (see the weight/confidence/logit clauses in :ref:`Property files <config>`). A benchmarking convenience for comparing the three models on one problem; the recommended default is ``auto``.

   - ``auto`` - each constraint keeps its authored model (``weight`` → hinge, ``confidence``/``pmin``/``tolerance`` → probit, ``logit scale`` → logit).
   - ``hinge`` - force every constraint to the static (2018) hinge penalty.
   - ``probit`` - force every constraint to the Gaussian-CDF (2020) likelihood.
   - ``logit`` - force every constraint to the logit (2025) softplus likelihood.

  Default: auto

  Example:

    * ``qualitative_loss = logit``

**qualitative_scale**
  Tie every qualitative (logit/probit) constraint's scale (the logit ``scale`` or probit ``tolerance``) to a fittable free parameter, so a fit estimates it jointly with the model parameters. The value is ``fit <parameter>``, naming a free parameter declared elsewhere in the .conf; declare it positive (log-scaled). A single scale is shared across all qualitative constraints (globally tied — the identifiable case). Applies to logit/probit constraints only; pair with ``qualitative_loss = logit`` (or ``probit``) if your .prop files author hinge weights.

  Default: none (scales fixed as authored)

  Example:

    * ``qualitative_scale = fit s_qual`` (with ``loguniform_var = s_qual 0.01 100``)

**ind_var_rounding**
  If 1, make sure every exp row is used by rounding it to the nearest available value of the independent variable in the simulation data. (Be careful with this! Usually, it is better to set up your simulation so that all experimental points are hit exactly) 
  
  Default: 0
  
  Example:
  
    * ``ind_var_rounding = 1``
    
**initialization**
  How to arrange the initial parameter sets.
  
   - ``rand`` - initialize parameters with independent random draws.
   - ``lh`` - initialize bounded parameters with a latin hypercube distribution, to more uniformly cover the search space.
   
  Default: lh
  
  Example: 
  
    * ``initialization = rand``

**initialization_distribution**
  Which distribution to draw start points from. This is separate from the objective prior.

   - ``prior`` - draw start points from each parameter's prior distribution. This is the backward-compatible default.
   - ``bounds`` - draw start points uniformly over each parameter's finite bounds in PyBNF's sampling space. Linear parameters use linear bounds; log parameters use log10 bounds.

  Default: prior

  Example:

    * ``initialization_distribution = bounds``

**random_seed**
  Seed for PyBNF's NumPy random number generator. If specified, the same seed will reproduce PyBNF-side random draws
  such as parameter initialization, proposal generation, acceptance decisions, and bootstrap resampling when results are
  processed in the same order. If omitted, PyBNF chooses a seed from system entropy and logs it as ``Random seed: N`` so
  the run can be repeated later.

  Some parallel runs can still differ if simulation jobs finish in a different order, because several algorithms draw
  random numbers when each result is processed. Stochastic simulators also require their own simulator-level seed to
  make simulation output reproducible.

  Default: None

  Example:

    * ``random_seed = 12345``
    
**local_objective_eval**
  If 1, evaluate the objective function locally, instead of parallelizing this calculation on the workers. This option is automatically enabled when using the ``smoothing`` or ``parallelize_models`` feature.
   
  Default: 0 (unless ``smoothing`` or ``parallelize_models`` is enabled)
  
  Example: 
  
    * ``local_objective_eval = 1``
  
**min_objective**
  Stop fitting if an objective function lower than this value is reached. 
  
  Default: None; always run for the maximum iterations
  
  Example: 
  
    * ``min_objective = 0.01``
  
.. _normalization_key:

**normalization**
  Normalize a simulation's predicted observable before it is compared to the data -- useful
  when the experimental values are themselves reported on a normalized scale (fold-change,
  percent-of-maximum, arbitrary fluorescence units, ...). Specify one of the following types:

   - ``init`` - normalize to the initial value
   - ``peak`` - normalize to the maximum value
   - ``zero`` - normalize such that each column has a mean of 0 and a standard deviation of 1
   - ``unit`` - Scales data so that the range of values is between (min-init)/(max-init) and 1 (if the maximum value is 0 (i.e. max == init), then the data is scaled by the minimum value after subtracting the initial value so that the range of values is between 0 and -1).
   - ``floor <rho>`` - add a measurement-noise **floor** ``x' = x + rho*max(x)`` (``rho`` a small fraction, default ``0.03``), so a log / relative objective stays finite where a series legitimately touches zero. Applied **identically to the simulated and the experimental** column.
   - ``scale`` - profile out each series' **optimal multiplicative scale** at scoring time (analytic / hierarchical scaling), so an overall model-vs-data scale difference on arbitrary-unit data is not penalized -- with no extra fitted parameter. Family-appropriate: the geometric-mean ratio for a log objective (:ref:`lognormal <objective_key>`), the least-squares optimum ``c* = sum(s d)/sum(s^2)`` otherwise.

  Normalization is a per-observable *prediction* transform -- a sibling of the per-observable
  :ref:`noise_model <noise_model_key>` / ``cumulative`` surface -- so under a modern
  :ref:`edition <edition>` (``>= 2``) it is keyed by **observable**, never by filename. Three
  forms layer into a single most-specific-wins rule, and the value may be an ordered **chain**
  of transforms separated by commas (applied left to right)::

    normalization = <chain>                          # whole-fit default (every observable)
    normalization <observable> = <chain>             # per-observable (every experiment)
    normalization <experiment>.<observable> = <chain>  # per-(experiment, observable) override

  For any observable column of any experiment the most specific rule wins:
  ``<experiment>.<observable>`` beats ``<observable>`` beats the whole-fit default; an
  observable matched by no rule is left un-normalized. ``<observable>`` is the model
  observable/function name (the data column name as remapped by any
  :ref:`observable <observable>` override) and ``<experiment>`` is the experiment name (see
  :ref:`experiment <experiment>`). A standard-deviation (``_SD``) column is never normalized
  on its own.

  ``peak`` / ``init`` / ``zero`` / ``unit`` rescale the **simulated** column only (the data is
  assumed pre-normalized by the user); ``floor`` and ``scale`` are applied symmetrically to
  the model and the data (a floor or an analytic scale is only meaningful applied to both).
  Together with ``objective = lognormal`` the chain ``floor 0.03, scale`` spells the
  "sum of squared log-differences of geometric-mean-normalized trajectories" objective common
  to arbitrary-unit fluorescence / blot fits.

  Normalization has no PEtab v2 representation (peak / initial-value / z-score / floor scaling
  is a whole-trajectory reduction, and ``scale`` is an analytic per-series optimum, neither a
  pointwise observable formula), so a job that uses it **cannot be exported to PEtab** -- the
  exporter refuses it rather than silently scoring the raw, un-normalized columns. Every
  transform here, ``floor`` and ``scale`` included, *is* differentiable, so a
  :ref:`gradient-based fit <gradient_fitting>` (``trf`` / ``lbfgs`` / ``gntr``) can run on a
  normalized or analytically scaled column -- including a **chain** of them
  (``floor 0.03, peak``), which the gradient composes stage by stage, each stage's chain rule
  read in the values that stage consumed and produced.

  Default: No normalization

  Examples (modern, ``edition = 2``):

     * ``normalization = init`` (whole-fit default)
     * ``normalization pErk = peak`` (pErk in every experiment)
     * ``normalization egf_high.pAkt = init`` (pAkt in experiment egf_high only)
     * ``normalization = floor 0.03, scale`` (floor then analytic scale, every observable)
     * ``normalization pStat = scale`` (analytic per-series scaling for pStat)

  **Legacy form** (no ``edition``): normalization is keyed by ``.exp`` *filename* instead. If
  only the type is specified it applies to all exp files; a type followed by a ``':'`` and a
  comma-delimited list of exp files applies to only those, and an exp file may be enclosed in
  parentheses with a column list, as in ``(data1.exp: 1,3-5)`` or ``(data1.exp: var1,var2)``.
  Multiple ``normalization`` lines may be used. This filename form is **not** available under
  ``edition >= 2`` (which keys data by experiment name, not filename) -- use the per-observable
  forms above instead.

     * ``normalization = init: data1.exp, data2.exp`` (legacy)
     * ``normalization = init: (data1.exp: 1,3-5), (data2.exp: var1,var2)`` (legacy)

.. _postproc_key:

**postprocess**
  Used to specify a custom Python script for postprocessing simulation results before evaluating the objective function. Specify the path to the Python script, followed by a list of all of the simulation suffixes for which that postprocessing script should be applied. For how to set up a postprocessing script, see :ref:`Custom Postprocessing <postproc>`. 
 
  Default: No postprocessing
  
  Example:
  
    * ``postprocess = path/to/script.py suff1 suff2``
  
**refine**
  If 1, after fitting is completed, refine the best fit parameter set by a local search. The optimizer used is set by ``refine_method`` (Simplex by default). Set that optimizer's config keys in addition to the config for your main algorithm.

  This asks for a *method* -- search globally, then polish locally -- so under a total wall-clock budget the refine is given a share of that budget up front rather than whatever the search happens to leave; see ``wall_time_refine_frac``. ``Results/method_chain.json`` records whether it ran.

  Default: 0

  Example:

    * ``refine = 1``

**refine_method**
  Which local optimizer to use for refinement when ``refine = 1``: ``sim`` (Nelder–Mead Simplex), ``powell`` (Powell's conjugate-direction method), or ``cmaes`` (CMA-ES). See :ref:`refinement <refinement>`. Has no effect unless ``refine = 1``.

  Default: sim

  Example:

    * ``refine_method = powell``

**sbml_integrator**
  Which integrator to use for SBML models. Options are ``cvode``, ``rk4``, ``gillespie``, or ``euler``, and are described in the `libroadrunner documentation <https://libroadrunner.readthedocs.io/en/latest/>`_. If your ``time_course`` or ``param_scan`` key specifies ``method: ssa``, then ``gillespie`` is used for that action, overriding this setting. 
  
  Default: cvode
  
  Example:
  
    * ``sbml_integrator = rk4``

.. _sbml_backend:

**sbml_backend**
  Which simulation engine runs SBML models: ``roadrunner`` (the default, libRoadRunner) or
  ``bngsim`` (the BioNetGen simulator). The ``bngsim`` backend supports a restricted set of
  integrators (see the ``sbml_integrator`` key) and only the ``ode`` and ``ssa`` simulation
  methods.

  Default: roadrunner

  Example:

    * ``sbml_backend = bngsim``

**sbml_ssa_strict**
  Relevant only when ``sbml_backend = bngsim`` and a simulation uses a stochastic (``ssa``)
  method: whether the SBML-to-network conversion runs in strict mode. Set to 0 to relax the
  strict checks.

  Default: 1

.. _sbml_rtol:

**sbml_rtol**, **sbml_atol**
  Relevant only when ``sbml_backend = bngsim``: the CVODE relative and absolute error
  tolerances used for every deterministic (``ode``) simulation of every SBML/Antimony
  model in the fit. For BNGL models, write ``rtol``/``atol`` in the BNGL file's
  ``begin actions`` block instead — that is BioNetGen's own surface for them.

  These two keys are the fit-wide **default**. Under ``edition >= 2`` a single model can
  state its own instead, on its :ref:`model: <model_decl>` declaration line
  (``atol:`` / ``rtol:``), and only there can a *per-species* vector be written by hand
  (``species_atol:``) — a global key applies to every model in the fit, so a species name
  on one has no unambiguous reading. Everything below describes what a value means; where
  it is written decides how much of the fit it means it for.

  Leave both unset unless you have a reason not to. ``sbml_rtol`` then takes the
  backend default (``1e-8``), and the absolute tolerance is **derived from the model**
  in two steps.

  First a single number for the model: ``sbml_rtol`` times the *median* strictly-positive
  species initial value in the SBML file, clamped to at most the backend default
  (``1e-8``) and at least ``1e-16``. Because that clamp only ever tightens, a model whose
  species are of order one keeps the backend default exactly. The median, rather than the
  smallest species, keeps one negligible transient intermediate from driving the tolerance
  for the whole model.

  That number exists because a constant absolute tolerance is a statement about the
  model's units: CVODE weights each state by ``rtol*|y| + atol``, so ``atol = 1e-8``
  declares values below ``1e-8`` to be noise. That is true of a model in molecule
  counts and false of, say, a population-*fraction* epidemic model whose species sit
  around ``1e-7``, whose whole early trajectory then carries no significant digits —
  and whose forward sensitivities, which a gradient fit reads, carry fewer still.

  Then, **per species**, each one is released back toward the backend default as far as
  its own initial value allows: species *i* is integrated at ``sbml_rtol`` times its own
  initial value, but never below the model-wide number above and never above ``1e-8``.
  A species whose initial value is at or below the model's median keeps the model-wide
  number, as does one declared at zero.

  The point of the second step is that one number over-charges the large species. A model
  with principal species at ``10`` and one transient intermediate at ``1e-9`` gets a
  tolerance that holds the former far tighter than ``sbml_rtol`` alone would, which costs
  steps and sometimes the simulation, and buys nothing. Note the direction: this step only
  ever *loosens*, so no species is integrated more tightly than the model-wide number, and
  a model that simulates today cannot start failing. Resolving the *small* species better
  than the model-wide number is deliberately not attempted — it was measured, and it
  roughly doubled the failed simulations on the model it was meant to help.

  The derivation is a property of the **model file**, not of the point being fitted: it
  is read off the species initial values in the SBML document once, at load, and held
  for the whole fit. A tolerance that moved with a fitted initial condition would put a
  step in the objective wherever the derivation crossed a rounding boundary.

  **Both clamps above only ever tighten, and that is a no-regression rule rather than a
  statement about your model.** A model whose species all sit far *above* one has a real,
  computable tolerance need, and the derivation computes it and then discards it:
  ``Weber_BMC2015``'s seven species live at ``1.24e+02 .. 4.21e+07``, so it asks for
  ``4.7e-03`` and is handed ``1e-08``, 5.7 decades tighter. What that costs is CVODE's
  error test read forwards: an ``atol`` far under the state's own magnitude is one the
  absolute term can never reach, so the integrator is held to a *relative* accuracy far
  tighter than the ``sbml_rtol`` that is supposed to govern it, and pays in steps for
  resolution nobody asked for. A forward-sensitivity solve pays most, since CVODES derives
  its sensitivity tolerances from the state ones. Say ``sbml_atol = auto`` to lift the
  ceiling:

  ``sbml_atol = auto``
    Trust the derivation in **both** directions. The two steps above are unchanged except
    that neither is capped at ``1e-08`` any more, so a species is integrated at
    ``sbml_rtol`` times its own initial value however large that is, and the model-wide
    number is ``sbml_rtol`` times the median however large *that* is. The lower clamp
    stays: no species is resolved below the model-wide number, which is the part that was
    measured (releasing it roughly doubled the failed simulations on the model it was
    meant to help). Nothing changes for a model whose species sit at or below one — those
    never reached the ceiling — so this only moves models the clamp was serving badly.

  ``sbml_atol = tracking``, ``sbml_atol = tracking <decades>``
    ``auto``, plus an absolute tolerance that follows the **trajectory** rather than
    staying where the initial values put it. A vector read off initial values fixes the
    cross-species compromise and stops there: whatever number a species gets, it keeps for
    the whole run, so a species that starts at order one and decays to something tiny
    outgrows its own tolerance partway through and stops being error-controlled. This
    installs bngsim's ``CVodeWFtolerances`` rule
    ``atol_i(y) = clamp(sbml_rtol*|y_i|, ceiling_i * 10**-decades, ceiling_i)``, with the
    ``auto`` vector as the ceiling — so it is never *looser* than ``auto``, only deeper,
    and ``tracking 0`` is ``auto`` exactly.

    ``<decades>`` is how far below its own ceiling a species keeps being resolved
    relatively; omit it for bngsim's measured default (12). It is not free — roughly 1.6x
    the integrator steps at the default depth on a real-model sweep, ~4x on a pure decay —
    and it cannot conjure digits that are not there: a species formed as a *difference* of
    large fluxes carries roundoff of order ``eps * flux``, and asking for accuracy below
    that collapses the step size rather than sharpening the answer. Lower the depth if a
    model that integrated before stops. Needs a bngsim with lanl/bngsim#213; PyBNF refuses
    the key rather than integrating at something else if yours does not have it.

  Set ``sbml_atol`` to a **number** if your model lives below the ``1e-16`` floor, or if
  you want a different tolerance from the one the derivation picks. A number replaces the
  whole derivation: it integrates every species at that value, which is also the way to
  pin the pre-per-species behaviour exactly. ``auto`` and ``tracking`` are not that — they
  say how far the model's own state may set its tolerance, so the derivation stays on.

  If what you want is a tolerance for **one species** that the derivation does not
  produce, that is ``species_atol:`` on the :ref:`model: <model_decl>` line rather than
  anything here: it names the species, it is checked against the model's own species list
  at config load, and it leaves every other species exactly where this key would have put
  it.

  What a looser tolerance costs is **smoothness**, not accuracy, and the distinction
  matters when choosing between ``auto`` and a hand-set number. On ``Weber_BMC2015`` four
  decades of tolerance move the objective at the reference point in its *sixth* decimal,
  and the assembled gradient does not move at all; what degrades is the finite-difference
  reference — i.e. the objective surface a trust-region line search consumes. So a
  gradient fit on a model with a wide species spread may do better at a number somewhere
  under the derivation's own answer, and ``auto`` is the setting that tells you what that
  answer is.

  A bngsim build without per-species tolerance support stops after the first step and
  integrates every species at the model-wide number, so an older bngsim runs every fit it
  ran before — and still honours ``auto``, which loosens that number.

  Default: unset (see above)

  Examples:

    * ``sbml_atol = 1e-20``
    * ``sbml_atol = auto``
    * ``sbml_atol = tracking 6``

.. _sensitivity_fallback:

**sensitivity_fallback**
  Relevant only to a gradient-based ``job_type`` (``trf``, ``lbfgs``, ``gntr``, ``ms``,
  ``profile_likelihood``): what to do when bngsim declines the **analytic sensitivity
  right-hand side** for one of your models.

  bngsim derives ``∂f/∂θ`` in closed form and compiles it, but CVODES takes one
  sensitivity-RHS callback for every column, so a single rate law it cannot
  differentiate — an ``abs()``/``floor()``/``erf()`` term, a comparison in a place it
  cannot solve, or simply a derivation that ran out of its build-time budget — declines
  the analytic path for the **whole model**. CVODES' internal difference quotient carries
  every column instead. That answer is still correct; it costs an extra right-hand-side
  evaluation per column per step, so an ``N``-parameter fit pays roughly ``N`` times the
  sensitivity cost. On a fit measured in hours that is often the difference between a
  result and a run that times out with nothing.

  PyBNF checks once per model at job start, before the fit has spent anything, by
  building one sensitivity-bearing simulator and reading back whether the compiled
  artifact carries the analytic RHS. The settings are:

  * ``warn`` (default) — name the model, the number of sensitivity columns, the expected
    cost multiplier, and bngsim's own reason when it gave one, then run the fit.
  * ``error`` — refuse the fit instead, for a long unattended run where discovering this
    from the log afterwards is too late.
  * ``ignore`` — skip the check entirely, including the one simulator construction per
    model that it costs.

  Two limits are worth knowing. bngsim reports its *reason* while generating codegen
  source, which a warm codegen cache skips, so the reason is present the first time a
  model is built and absent afterwards — the verdict itself is read off the artifact and
  is unaffected. And the check describes the model at the fit's start point: one step of
  bngsim's codegen reads parameter values, so a model that branches on a fitted threshold
  could in principle change its answer mid-search.

  Default: warn

  Examples:

    * ``sensitivity_fallback = error``
    * ``sensitivity_fallback = ignore``

**smoothing**
  Number of replicate runs to average together for each parameter set (useful for stochastic simulations). This option can be used with
  ``parallelize_models`` to run model partitions independently within each replicate.

  Each replicate gets a distinct deterministic seed under the default
  :ref:`stochastic_seed` policy (``auto``), so smoothing replicates yield
  different stochastic trajectories while remaining reproducible across
  runs. If you set ``stochastic_seed = auto_honorbngl`` or
  ``random_honorbngl`` and any of your BNGL actions specifies an explicit
  ``seed=>N``, PyBNF rejects the run at config load — that combination
  would force every replicate to share the same trajectory.

  Default: 1

  Example:

    * ``smoothing = 2``
    
**generate_network**
  Model-scoped options for BNGL network generation (edition 2). When a model carries no
  ``begin actions`` block — the edition-2 convention, where the ``experiment:`` lines
  synthesize the simulations — PyBNF generates the reaction network with a bare
  ``generate_network({overwrite=>1})``. For a model whose reaction network is finite only
  under a stoichiometry / aggregation / iteration cap (crosslinking, aggregation,
  polymerization), that bare default would generate an *unbounded* network and never
  terminate. This key supplies the cap the stripped actions block used to carry: its value is
  a free-form BNGL ``generate_network`` options fragment, injected as
  ``generate_network({overwrite=>1, <options>})``. An explicit ``generate_network`` line in
  the model always takes precedence (this only fills the synthesized default).

  Default: none (the bare ``generate_network({overwrite=>1})``)

  Examples:

    * ``generate_network = max_stoich=>{EGF=>4,EGFR=>4}``
    * ``generate_network = max_stoich=>{EGF=>4,EGFR=>4}, max_iter=>3, max_agg=>8``

**wall_time_gen**
  Maximum time (in seconds) to wait to generate the network for a BNGL model. Will cause the program to exit if exceeded.

  Default: 3600

  Example:

    * ``wall_time_gen = 600``
    
**wall_time_sim**
  Maximum time (in seconds) to wait for a simulation to finish.  Exceeding this results in an infinite objective function value. Caution: For SBML models, using this option has an overhead cost, so only use it when needed. 
  
  Default: 3600 for BNGL models; No limit for SMBL models
  
  Example: 
  
    * ``wall_time_sim = 600``

**wall_time_fit**
  Total wall-clock budget (in seconds) for the whole fit -- the peer of ``wall_time_sim``
  and ``wall_time_gen``, which bound one simulation and one network generation but never
  the run itself. When the budget runs out, PyBNF stops launching new work, abandons what
  is still in flight, and then runs the **normal** end-of-fit path against the best
  parameter set found so far: ``sorted_params_final.txt``, the best-fit simulations,
  ``information_criteria.txt``, and the rest. A budgeted result is therefore scoreable
  exactly like a completed one -- only the stop *reason* differs, and that is logged,
  printed, and written to ``Results/stop_reason.txt``.

  The clock starts when PyBNF starts, so configuration loading and network generation are
  inside the budget; pair it with ``parallel_count`` to reproduce a "N cores for T hours"
  compute allocation. It bounds the run, not each phase: no further bootstrap replicate
  begins once the budget is spent, and the ``refine`` gets the slice of the budget
  ``wall_time_refine_frac`` holds back for it rather than a deadline of its own. Two
  things are deliberately outside it -- one in-flight simulation may overrun the deadline
  by up to ``wall_time_sim`` before it is abandoned, and finalizing re-simulates the best
  fit once to report it.

  A fit stopped this way is *not* a converged fit; check ``Results/stop_reason.txt`` before
  reading its results as a completed search. ``Results/method_chain.json`` records which
  methods actually ran, and why any requested phase did not.

  Not available for ``job_type = hmc`` (which samples in process, with no simulation loop
  for the budget to stop) -- naming it there is an error rather than a silent no-op.

  Default: 0 (no limit)

  Examples:

    * ``wall_time_fit = 10800``   (3 hours)
    * ``wall_time_fit = 600``

**wall_time_refine_frac**
  The share of ``wall_time_fit`` held back from the search so that the ``refine`` can run.

  ``refine = 1`` asks for a *method* -- search globally, then polish the result with a
  local optimizer -- but a wall-clock-budgeted search has no reason to leave anything
  behind: it runs until the clock stops. Without a reserve the polish is new work that
  never starts, so ``wall_time_fit`` + ``refine = 1`` would silently execute the global
  search alone. This key makes the phase split explicit: the search may spend
  ``(1 - wall_time_refine_frac)`` of the budget, and the refine gets the rest.

  The reserve is a floor, not a cap. A search that converges before its share is up hands
  everything it did not spend to the refine, so the run's total is still ``wall_time_fit``
  -- one deadline for the whole run.

  Inert unless both ``wall_time_fit`` and ``refine = 1`` are set, and no reserve is taken
  when ``refine_method`` names the algorithm the fit itself ran (which PyBNF skips anyway).
  Set it to ``0`` to spend every second searching and accept that the refine may not run;
  PyBNF then says so at every verbosity level, and records the downgrade in
  ``Results/method_chain.json``.

  Default: 0.1

  Examples:

    * ``wall_time_refine_frac = 0.1``   (a tenth of the budget for the polish)
    * ``wall_time_refine_frac = 0``     (no reserve; the search may spend it all)

**max_failed_simulations**
  Maximum number of simulation failures allowed before any successful simulation completes. If this many jobs fail (crash, not timeout) before the first success, PyBNF aborts. Increase this value if your model has a high failure rate at many parameter sets but can still succeed at others.

  Default: 100

  Example:

    * ``max_failed_simulations = 500``


Algorithm-specific Options
--------------------------

:ref:`Simplex <alg-sim>`
^^^^^^^^^^^^^^^^^^^^^^^^

These settings for the :ref:`simplex <alg-sim>` algorithm may also be used when running other algorithms with ``refine = 1``.

**simplex_step**
  In initialization, we perturb each parameter by this step size. If you specify a step size for a specific variable via ``var`` or ``logvar``, it overrides this setting. 
  
  Default: 1
  
  Example:
  
    * ``simplex_step = 0.5``
  
**simplex_log_step**
  Equivalent of ``simplex_step``, for variables that move in log space. 
  
  Default: Value of ``simplex_step``
  
  Example:
  
    * ``simplex_log_step = 0.5``

**simplex_reflection**
  When we reflect a point through the centroid, what is the ratio of dilation on the other side? 
  
  Default: 1.0
  
  Example:
  
    * ``simplex_reflection = 0.5``

**simplex_expansion**
  If the reflected point was the global minimum, how far do we keep moving in that direction? (as a ratio to the initial distance to centroid) 
  
  Default: 1.0
  
  Example:
  
    * ``simplex_expansion = 0.5``
  
**simplex_contraction**
  If the reflected point was not an improvement, we retry at what distance from the centroid? (as a ratio of the initial distance to centroid) 
  
  Default: 0.5
  
  Example:
  
    * ``simplex_contraction = 0.3``
    
**simplex_shrink**
  If a whole iteration was unproductive, shrink the simplex by setting simplex point :math:`s[i]` to :math:`x*s[0] + (1-x)*s[i]`, where *x* is the value of this key and :math:`s[0]` is the best point in the simplex. 
  
  Default: 0.5
  
  Example:
  
    * ``simplex_shrink = 0.3``

**simplex_max_iterations**
  If specified, overrides the ``max_iterations`` setting. Useful if you are using the ``refine`` flag and want ``max_iterations`` to refer to your main algorithm.
  
  Example:
  
    * ``simplex_max_iterations = 20``
    
**simplex_stop_tol**
  Stop the algorithm if all parameters have converged to within this value (specifically, if all reflections in an iteration move the parameter by less than this
  value)

  Default: 0 (don't use this criterion)

  Example:
    * ``simplex_stop_tol = 0.01``


:ref:`Powell <alg-powell>`
^^^^^^^^^^^^^^^^^^^^^^^^^^

These settings for the :ref:`Powell <alg-powell>` optimizer apply both to ``job_type = powell`` and to any algorithm run with ``refine = 1`` and ``refine_method = powell``.

**powell_step**
  Initial bracketing step along each search direction, in the parameter sampling space (a factor of ``10**powell_step`` for a log-scaled parameter). Each line search starts by probing this far, then expands a bracket around the minimum and refines it (see ``powell_line_tol``); it is no longer the only step the search can take.

  Default: 1.0

  Example:

    * ``powell_step = 0.3``

**powell_line_tol**
  Fractional tolerance to which each 1-D (Brent) line minimum is resolved. Smaller values locate each line minimum more precisely at the cost of more objective evaluations per line search; the default is ample for refining a near-quadratic objective.

  Default: 1e-4

  Example:

    * ``powell_line_tol = 1e-3``

**powell_stop_tol**
  Stop when a whole cycle of line searches improves the objective by less than this fraction.

  Default: 1e-5

  Example:

    * ``powell_stop_tol = 1e-4``

**powell_max_iterations**
  If specified, the number of Powell cycles (one line search along each direction), overriding ``max_iterations``. Useful when using ``refine`` and you want ``max_iterations`` to refer to your main algorithm.

  Default: value of ``max_iterations``

  Example:

    * ``powell_max_iterations = 20``


:ref:`CMA-ES <alg-cmaes>`
^^^^^^^^^^^^^^^^^^^^^^^^^

These settings for the :ref:`CMA-ES <alg-cmaes>` optimizer apply both to ``job_type = cmaes`` and to any algorithm run with ``refine = 1`` and ``refine_method = cmaes``. CMA-ES uses ``population_size`` as its population size (lambda, at least 4) and ``max_iterations`` as its generation budget. Standalone, ``job_type = cmaes`` accepts either a single ``var`` / ``logvar`` start point (local search) or a bounded ``uniform_var`` / ``loguniform_var`` box (its global-start mode, starting from the box center); see :ref:`CMA-ES <alg-cmaes>`.

**cmaes_sigma0**
  Initial overall step size of the search distribution, in the parameter sampling space (a factor of ``10**cmaes_sigma0`` for a log-scaled parameter). In box / global-start mode (bounded ``uniform_var`` / ``loguniform_var`` priors) it is instead read as a fraction of each box width, so the initial per-coordinate standard deviation is ``cmaes_sigma0`` × (box width).

  Default: 0.3

  Example:

    * ``cmaes_sigma0 = 0.5``

**cmaes_stop_tol**
  Stop when the largest principal standard deviation of the search distribution falls below this value. This is a step length in the parameter sampling space, and in restart mode (``cmaes_restarts > 0``) it is also the threshold below which every individual coordinate step counts as collapsed. The stagnation threshold on the *objective* is ``cmaes_tolfun``.

  Default: 1e-11

  Example:

    * ``cmaes_stop_tol = 1e-8``

**cmaes_tolfun**
  Stagnation tolerance on the objective, used only in restart mode (``cmaes_restarts > 0``): a run is declared finished — and yields to the next restart — when the range of its best objective over the last ``10 + ceil(30 × (number of parameters) / population_size)`` generations falls to this value or below. It is an absolute range in the units of your objective function, unlike ``cmaes_stop_tol``, which is a step length in the parameter sampling space; set it to the smallest objective improvement per window that you still consider progress. Unset, it follows ``cmaes_stop_tol``, which is rarely what you want if you rely on stagnation restarts: a value loose enough to detect a stalled run is far looser than a converged search distribution.

  Default: unset (follows ``cmaes_stop_tol``)

  Example:

    * ``cmaes_tolfun = 1e-3``

**cmaes_restarts**
  Maximum number of IPOP / BIPOP restarts for multimodal search. A single CMA-ES run descends into the one basin its start lands in, so on a multimodal objective it reaches only a local minimum. With ``cmaes_restarts > 0``, whenever a run *finishes* by converging (its search distribution shrinks below ``cmaes_stop_tol``, or its step size degenerates) or reaching ``cmaes_run_maxgen`` — as distinct from exhausting the generation budget ``max_iterations`` — CMA-ES reinitializes from a fresh random point in the prior box with a rescaled population and keeps searching, up to this many restarts, keeping the global best across all runs. Requires the box / global-start mode (bounded ``uniform_var`` / ``loguniform_var`` priors), which provides the box restarts resample from. ``0`` (the default) is a single run.

  Default: 0

  Example:

    * ``cmaes_restarts = 9``

**cmaes_run_maxgen**
  Optional generation cap for each individual CMA-ES run, including the initial run and every IPOP / BIPOP restart. Reaching the cap yields to the next restart when one remains; on the final run it ends the fit. This prevents a single steadily improving local basin from consuming the global ``max_iterations`` generation budget before the configured restarts can run. The global budget still applies across all runs. BIPOP small runs also have an automatic evaluation-balancing cap; the smaller of that cap and ``cmaes_run_maxgen`` applies.

  Default: unset (no per-run cap)

  Example:

    * ``cmaes_run_maxgen = 100``

**cmaes_restart_strategy**
  The restart schedule (used only when ``cmaes_restarts > 0``). ``ipop`` grows the population geometrically each restart (``population_size`` × ``cmaes_ipop_factor``\ :sup:`k`), a progressively broader global search (Auger & Hansen 2005). ``bipop`` interleaves that increasing-population regime with a small-population regime, launching whichever has spent fewer evaluations so far, which balances broad sweeps against many quick fine-grained searches (Hansen 2009).

  Default: ipop

  Example:

    * ``cmaes_restart_strategy = bipop``

**cmaes_ipop_factor**
  Geometric population-growth factor per restart, used by IPOP and by BIPOP's large regime. ``2.0`` is the standard population doubling.

  Default: 2.0

  Example:

    * ``cmaes_ipop_factor = 2.0``


:ref:`Differential Evolution <alg-de>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyBNF offers two versions of :ref:`differential evoltution <alg-de>`: synchronous differential evolution (``job_type = de``) and asynchronous differential evolution (``job_type = ade``). Both versions may be configured with the follwing keys.

**mutation_rate**
  When generating a new individual, mutate each parameter with this probability. 
  
  Default: 0.5
  
  Example:
  
    * ``mutation_rate = 0.7``
    
**mutation_factor**
  When mutating a parameter x, change it by mutation_factor*(PS1[x] - PS2[x]) where PS1 and PS2 are random other PSets in the population.  
  
  Default: 1.0
  
  Example:
  
    * ``mutation_factor = 0.7``

**stop_tolerance**
  Stop the run when the current population has converged to roughly the same objective function value, measured as an **absolute range** of the objective across the finite members of the population: the run ends once :math:`max\_objective - min\_objective \le e`, where *e* is the value of this key. Failed simulations (which score infinity) are ignored, so one dead candidate can neither trigger nor block the stop. This is a range in the units of your objective function; on a likelihood objective (a negative log-likelihood, which is unbounded below) set it to the smallest population spread you still consider unconverged. Prior to #561 this was a *ratio* test (:math:`max/min < 1 + e`), which stopped the run at generation 0 on any objective that can go negative; see ``de_tolfun`` to set the range independently of this key's magnitude.

  Default: 0.002

  Example:

    * ``stop_tolerance = 0.001``


**de_tolfun**
  The convergence tolerance the run actually uses: the absolute objective range (see ``stop_tolerance``) below which the population counts as converged. It is a range in the units of your objective function, whereas ``stop_tolerance`` was historically a dimensionless ratio; the two are separated so a fit can set a meaningful objective-range stop without reinterpreting the legacy key. Unset, it follows ``stop_tolerance``, so an existing config keeps the threshold magnitude it had.

  Default: unset (follows ``stop_tolerance``)

  Example:

    * ``de_tolfun = 1e-3``


**de_strategy**
  Specifies how new parameter sets are chosen. The following options are available:
  
   - ``rand1``
   - ``rand2``
   - ``best1`` 
   - ``best2``
   - ``all1``
   - ``all2``

  The first part of the string determines which parameter set we mutate:
  
   - ``rand`` - a random one
   - ``best`` - the one with the lowest objective value
   - ``all`` - the one we are proposing to replace (so all psets are mutated once per iteration). 

  The second part of the string specifies how we calculate the amount by which to mutate each parameter: 
  
   - ``1`` - Use 1 pair of other parameter sets: :math:`(p_1-p_2)`
   - ``2`` - Use 2 pairs of other parameter sets: :math:`(p1-p2 + p3-p4)`. 
  
  Default: rand1
  
  Example:
  
    * ``de_strategy = rand2``

The following options are only available with ``job_type = de``, and serve to make the algorithm more asynchronous. If used, these options enable :ref:`island-based <alg-island>` differential evolution, which is asynchronous in that each island can independently proceed to the next iteration. 

**islands**
  Number of separate populations to evolve.
  
  Default: 1
  
  Example: 
  
    * ``islands = 2``
    
**migrate_every**
  After this number of generations, migrate some individuals between islands. 
  
  Default: 20 (but Infinity if ``islands = 1``)
  
  Example:
  
    * ``migrate_every = 10``
    
**num_to_migrate**
  How many individuals to migrate off of each island during migration. 
  
  Default: 3
  
  Example:
  
    * ``num_to_migrate = 5``


:ref:`Scatter Search <alg-ss>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**init_size**
  Number of parameter sets to test to generate the initial population. 
  
  Default: 10 * number of parameters
  
  Example:
  
    * ``init_size = 100``
  
  
**local_min_limit**
  If a point is stuck for this many iterations without improvement, it is assumed to be a local min and replaced with a random parameter set. 
  
  Default: 5
  
  Example:
  
    * ``local_min_limit = 10``
    
**reserve_size**
  Scatter Search maintains a latin-hypercube-distributed "reserve" of parameter sets. When it needs to pick a random new parameter set, it takes one from the reserve, so it's not similar to a previous random choice. The initial size of the reserve is this value. If the reserve becomes empty, we revert to truly random pset choices. 
  
  Default: Value of ``max_iterations``
  
  Example:
  
    * ``reserve_size = 100``


:ref:`Particle Swarm <alg-pso>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**cognitive**
  Acceleration toward a particle's own best fit
  
  Default: 1.5
  
  Example:
  
    * ``cognitive = 1.7``
  
**social**
  Acceleration toward the global best fit
  
  Default: 1.5
  
  Example:
  
    * ``social = 1.7``
    
**particle_weight**
  Inertia weight of particle. A value less than 1 can be thought of as friction that contniuously decelerates the particle.
  
  Default: 0.7
  
  Example:
  
    * ``particle_weight = 0.9``
    
**v_stop**
  Stop the algorithm if the speeds of all parameters in all particles are less than this value. 
  
  Default: 0 (don't use this criterion)
  
  Example:
  
    * ``v_stop = 0.01``

A variant of particle swarm that adaptively changes the ``particle_weight`` over the course of the fitting run is configured with the following parameters. See the :ref:`algorithm documentation <pso-adaptive>` for more information. 

**particle_weight_final**
  The final particle weight after the adaptive weight changing. 
  
  Default: the value of ``particle_weight``, effectively disabling this feature. 
  
  Example:
  
    * ``particle_weight_final = 0.5``
    
**adaptive_n_max**
  After this many "unproductive" iterations, we have moved halfway from the initial weight to the final weight. 
  
  Default: 30
  
  Example: 
  
    * ``adaptive_n_max = 20``
    
**adaptive_n_stop**
  Afer this many "unproductive" iterations, stop the fitting run. 
  
  Default: Inf
  
  Example:
  
    * ``adaptive_n_stop = 50``
    
**adaptive_abs_tol**
  Parameter for checking if an iteration was "unproductive" 
  
  Default: 0
  
  Example:
  
    * ``adaptive_abs_tol = 0.01``
    
**adaptive_rel_tol**
  Parameter for checking if an iteration was "unproductive" 
  
  Default: 0
  
  Example:
  
    * ``adaptive_rel_tol = 0.01``

:ref:`Bayesian Algorithms (mh, pt, sa) <alg-mcmc>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the family of Bayesian algoritms with Metropolis sampling, PyBNF includes :ref:`Metropolis-Hastings MCMC <alg-mcmc>` (``job_type = mh``), :ref:`Parallel Tempering <alg-pt>` (``job_type = pt``), :ref:`Simulated Annealing <alg-sa>` (``job_type = sa``), and :ref:`DREAM <alg-dream>` (``job_type = dream``). These algorithms have many configuration keys in common, as described below. 


For all Bayesian algorithms
"""""""""""""""""""""""""""

**step_size**
  When proposing a Monte Carlo step, the step in n-dimensional parameter space has this length. 
  
  Default: 0.2
  
  Example:
  
    * ``step_size = 0.5``

**beta**
  Sets the initial beta (1/temperature). A smaller beta corresponds to a more broad exploration of parameter space. If a single value is provided, that beta is used for all replicates. If multiple values are provided, an equal number of replicates uses each value. 
  
  For ``mh``, should be set to 1 (the default) to get the true probability distribution. 
  
  For ``pt``, should specify multiple values: the number of values should equal ``population_size``/``reps_per_beta``. Or you may instead use the ``beta_range`` key. Only the largest beta value in the list will constribute to statistical samples, and to get the true probability distribution, this maximum value should be 1.
  
  For ``sa``, should typically be set to a single, small value which will increase over the course of the fitting run. 
  
  Default: 1
  
  Examples:
  
    * ``beta = 0.9``
    * ``beta = 0.7 0.8 0.9 1``


For all Bayesian algorithms except ``sa``
"""""""""""""""""""""""""""""""""""""""""

**sample_every**
  Every x iterations, save the current PSet into the sampled population. Default: 100
  
  Example:
  
    * ``sample_every = 20``
    
**burn_in**
  Don't sample for this many iterations at the start, to let the system equilibrate. 
  
  Default: 10000
  
  Example:
  
    * ``burn_in = 1000``
    
**output_hist_every**
  Every x samples (i.e every x*sample_every iterations), save a historgram file for each parameter, and the credible interval files, based on what has been sampled so far. Regardless, we also output these files at the end of the run.  
  
  Default: 100
  
  Example: 
  
    * ``output_hist_every = 10``
    
**hist_bins** 
  Number of bins used when writing the histogram files. 
  
  Default: 10
  
  Example:
  
    * ``hist_bins = 20``

**credible_intervals**
  Specify one or more numbers here. For each n, the algorithm will save a file giving bounds for each parameter such that in n% of the samples, the parameter lies within the bounds.
  
  Default: 68 95
  
  Examples:
  
    * ``credible_intervals = 95``
    * ``credible_intervals = 20 68 95``


For Simulated Annealing
"""""""""""""""""""""""

**beta_max** 
  Stop the algorithm if all replicates reach this beta (1/temperature) value. 
  
  Default: Infinity (don't use this stop criterion)
  
  Example:
  
    * ``beta_max = 1.5``
    
**cooling**
  Each time a move to a higher energy state is accepted, increase beta (1/temperature) by this value. 
  
  Default: 0.01
  
  Example:
  
    * ``cooling = 0.001``


For Parallel Tempering
""""""""""""""""""""""

**exchange_every**
  Every x iterations, perform replica exchange, swapping replicas that are adjacent in temperature with a statistically correct probability
  
  Default: 20
  
  Example:
  
    * ``exchange_every = 10``
    
    
**reps_per_beta**
  How many identical replicas to run at each temperature. Must be a divisor of ``population_size``.
  
  Default: 1
  
  Example:
  
    * ``reps_per_beta = 5``
  
  
**beta_range**
  As an alternative to setting ``beta``, the range of values of beta to use. Specify the minimum value, followed by the maximum value. The replicates will use ``population_size``/``reps_per_beta`` geometrically spaced beta values within this range. Only the replicas at the max beta value will be sampled. For the true probability distribution, the maximum value should be 1.
  
  Default: None (betas are set with the ``beta`` key)
  
  Example:
  
    * ``beta_range = 0.5 1`` 
    
    
For Adaptive MCMC
""""""""""""""""""""""

**stablizingCov**
  Stabilize the covariant matrix of the proposal. 
  
  Default: 0.001
  
  Example:
  
    * ``stablizingCov = 0.1``
    
    
**adaptive**
  The number of iterations that the simulation will spend collecting data to observe the data for calcualtion of the differential matrix.``.
  
  Default: 10000
  
  Example:
  
    * ``adaptive = 50000``
  
  
**output_noise_trajectory (Only for use with neg_bin and neg_bin_dynamic functions)**
  Calculate and add the negative binomial noise to the specified observables or functions then save the output of the user defined observable or function from the simulation output to a .txt file.
  
  Default: None (multiple values can be defined separated by a comma)
  
  Note: output_trajectory and output_noise_trajectory can both be declared in the same configuration file but may       result in slower performance
  
  Example:
  
    * ``output_noise_trajectory = ObservableA`` 
    * ``output_noise_trajectory = ObservableA, ObservableB, FunctionA``
    

**output_trajectory**
  Save the output of the user defined observable or function from the simulation output to a .txt file.
  
  Default: None (multiple values can be defined separated by a comma)
  
  Example:
  
    * ``output_trajectory = ObservableA``
    * ``output_trajectory = ObservableA, ObservableB, FunctionA``
    
    
**continue_run**
  When set to 1 the chains began at the MAP parameters, calculated covarience matrix, and diffusivity from the previous chain. 
  
  Default: 0
  
  Example:
  
    * ``continue_run = 1`` 
    
**calculate_covari**
  Calculate the covairance matrix of a defined segment of the previous run 
  
  Default: None
  
  Example:
  
    * ``calculate_covari = 1 50000``
**starting_params**
  Seed every chain of a **sampler** from a defined set of starting parameters, listed in the same order
  the free parameters are declared in the .conf file, separated by spaces.

  This key is read only by the samplers (``am``, ``mh``, ``pt``, ``dream``, ``p_dream``, ``hmc``).
  Setting it on any other ``job_type`` is a configuration error: no optimizer has ever read it, so it
  used to be accepted and then silently discarded, and the fit started somewhere else entirely.

  Prefer :ref:`start_point <start_point>` for new configurations. It is read by every ``job_type``, it is
  matched by **name** rather than by position — ``starting_params`` is positional against declaration
  order, while every result file PyBNF writes is alphabetical, so round-tripping a result row back into
  it silently permutes the values — and it is validated against the declared bounds. Note also that
  ``continue_run = 1`` overrides ``starting_params`` entirely.

  Default: None

  Example:

    * ``starting_params = 5.5 2 3``

For DREAM
"""""""""

``step_size = float``
  Fixed jump rate for the differential evolution proposal. If not specified, an adaptive jump rate of
  :math:`2.38/\sqrt{2\delta d'}` is used automatically (recommended). Setting this key explicitly disables
  adaptive scaling.

``adaptive_step_size = bool``
  Toggle for the adaptive jump-rate scaling above (owned by ``dream`` / ``p_dream``). Set to 0 to
  disable adaptation and use a fixed step, the same off-state as specifying an explicit ``step_size``.
  Default: on

``crossover_number = int``
  The number of distinct crossover probabilities for subspace sampling. Defines the set
  :math:`\{1/n, 2/n, \ldots, 1\}`. Selection probabilities are adapted during the first half of burn-in.
  Default: 3

``zeta = float``
  Standard deviation of the small normal perturbation added to each parameter for detailed balance.
  Default: 1e-6

``lambda = float``
  Half-width of the uniform perturbation applied to parameters selected by the crossover procedure.
  Default: 0.1

``gamma_prob = float``
  Probability of a mode jump (:math:`\gamma = 1`, all dimensions updated) instead of the standard proposal.
  Default: 0.1

``archive_size = int``
  Initial size of the ZS archive (number of random prior draws). Default: :math:`10d` where :math:`d` is
  the number of free parameters.

``archive_thin_rate = int``
  Every this many generations, current chain states are appended to the archive.  Default: 10

``snooker_prob = float``
  Probability of using a snooker update instead of a parallel direction proposal each generation.
  Default: 0.1

``delta = int``
  Number of chain pairs used in the differential evolution proposal. Higher values increase proposal
  diversity at the cost of needing a larger archive. Default: 1

``outlier_method = str``
  Method for detecting outlier chains during burn-in. Options: ``iqr`` (interquartile range) or
  ``grubbs`` (Grubbs test at alpha=0.01). Default: ``iqr``

``proposal = str``
  The DREAM proposal operator. ``de`` (the default) is the classic DREAM(ZS) parallel-direction
  differential-evolution proposal. ``whitened`` computes the proposal in an online
  covariance-whitened space for better sampling of correlated posteriors — this is what the
  ``p_dream`` job type selects (``p_dream`` is simply ``dream`` with ``proposal = whitened``
  pinned), and it can also be requested explicitly on a ``dream`` run. ``kalman`` is the
  Kalman-inspired proposal (DREAM(KZS); Zhang, Vrugt et al. 2020): during a burn-in window it
  steers each proposal toward the data using a Kalman gain built from the archive's
  parameter↔model-output cross-covariance, which accelerates burn-in on informative, mildly
  non-linear problems, then reverts to ``de`` for the sampling phase. ``kalman`` requires a
  linear-scale Gaussian likelihood (``objfunc = chi_sq`` or ``chi_sq_dynamic`` — the source of the
  measurement covariance ``R = diag(σ²)``) and ``n_try = 1``. Default: ``de``

``precondition_adapt = int``
  Used only by the ``whitened`` proposal (the ``p_dream`` job type, or ``dream`` with
  ``proposal = whitened``). The iteration at which the sampler switches to proposing in its learned
  covariance-whitened space; until then the online covariance is still being estimated and plain
  DREAM proposals are used. Default: half of ``burn_in``.

``kalman_burnin_frac = float``
  Used only by the ``kalman`` proposal. The fraction of ``burn_in`` over which the Kalman-inspired
  proposal is active before the chain reverts to the ``de`` proposal. The Kalman jump deliberately
  breaks detailed balance (there is no Hastings correction), so its samples are burn-in and
  discarded; it must switch off before the sampling phase, so this must be between 0 and 1.
  Default: ``0.3`` (matching Zhang et al. 2020's ``T_K = 0.3 T``, here a fraction of ``burn_in``).

``n_try = int``
  Number of candidate proposals drawn per chain per generation — the Multi-Try DREAM
  (MT-DREAM(ZS)) count (Laloy & Vrugt 2012). ``n_try = 1`` (the default) is the classic single-try
  engine. With ``n_try = k > 1`` each chain proposes ``k`` candidates, selects one in proportion to
  its posterior importance weight, and accepts it over the current state with a multiple-try
  Metropolis ratio evaluated against a reference set (``2k - 1`` evaluations per chain per
  generation). Multiple tries per generation raise the per-generation acceptance rate and help
  parameter-rich or strongly correlated posteriors mix. Composes with the ``de`` and ``whitened``
  proposals and with the snooker update; ``proposal = kalman`` requires ``n_try = 1``. Default: ``1``
  If set to a positive value, the algorithm stops automatically once all parameters have
  :math:`\hat{R}` below this threshold (checked after burn-in). Set to 0 to disable. A common
  threshold is 1.05. Default: 0 (disabled)

``diagnostics_every = int``
  How often (in iterations) to compute and report the convergence diagnostics
  (:math:`\hat{R}`, bulk/tail ESS). Each computation rank-normalizes the last half of the chain
  history, whose length grows with the run, so computing it on a fixed cadence makes the total
  diagnostic cost scale with the *square* of ``max_iterations``. Striding it instead caps the
  number of computations and keeps the cost roughly linear; the diagnostic value reported at any
  given iteration is unchanged. Set to 0 (the default) to auto-scale as ``max(10, max_iterations //
  100)`` (~100 reports per run); set a positive value to force a fixed cadence. Default: 0 (auto)


For Hamiltonian Monte Carlo (HMC)
"""""""""""""""""""""""""""""""""

The :ref:`HMC sampler <alg-hmc>` (``job_type = hmc``) uses window adaptation in place of the
shared MCMC ``burn_in`` / ``sample_every`` thinning (NUTS draws are near-independent, so every
post-warmup draw is kept), and ``population_size`` as the number of independent chains. It adds
three keys. Requires :ref:`edition <edition>` ``>= 2`` and the ``pybnf[jax]`` extra.

**num_warmup**
  Window-adaptation (warmup) steps per chain -- NUTS tunes its step size (dual averaging) and mass
  matrix over these, then discards them.

  Default: 1000

  Example:

    * ``num_warmup = 800``

**num_samples**
  Post-warmup draws kept per chain (each becomes one row of the samples file). With ``population_size``
  chains the total sample count is ``population_size * num_samples``.

  Default: 1000

  Example:

    * ``num_samples = 2000``

**target_accept**
  The NUTS dual-averaging target acceptance probability (Stan's default is 0.8). Raising it toward 1
  shrinks the step size, which traverses sharp curvature more reliably -- the fix for divergent
  transitions on a hard geometry (e.g. a tight banana) -- at the cost of more gradient evaluations per
  draw.

  Default: 0.8

  Example:

    * ``target_accept = 0.95``
