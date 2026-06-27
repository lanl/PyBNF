"""Grammar and methods for parsing the configuration file"""


from .printing import PybnfError, print1
from .priors import var_keyword_grammar
from .config import Configuration

from string import punctuation

import logging
import pyparsing as pp
import re


logger = logging.getLogger(__name__)


_one_of = pp.one_of if hasattr(pp, 'one_of') else pp.oneOf
_DelimitedList = pp.DelimitedList if hasattr(pp, 'DelimitedList') else pp.delimitedList


def _parse_all(parser, text):
    if hasattr(parser, 'parse_string'):
        return parser.parse_string(text, parse_all=True)
    return parser.parseString(text, parseAll=True)


numkeys_int = ['verbosity', 'parallel_count', 'delete_old_files', 'population_size',
               'smoothing', 'max_iterations',
               'num_to_output', 'output_every', 'islands', 'migrate_every', 'num_to_migrate', 'init_size',
               'local_min_limit', 'reserve_size', 'burn_in', 'sample_every', 'output_hist_every',
               'hist_bins', 'refine', 'simplex_max_iterations', 'wall_time_sim', 'wall_time_gen', 'verbosity',
               'exchange_every', 'backup_every', 'bootstrap', 'crossover_number', 'ind_var_rounding',
               'local_objective_eval', 'reps_per_beta', 'save_best_data', 'embed_best_fit_data',
               'smooth_plot_points', 'output_inference_data',
               'parallelize_models', 'adaptive', 'continue_run',
               'delta', 'archive_size', 'archive_thin_rate', 'adaptive_step_size', 'powell_max_iterations',
               'max_failed_simulations', 'random_seed', 'sbml_ssa_strict', 'diagnostics_every', 'edition',
               # HMC (job_type = hmc, ADR-0059): per-chain warmup/draw counts.
               'num_warmup', 'num_samples']
numkeys_float = ['min_objective', 'cognitive', 'social', 'particle_weight',
                 'particle_weight_final', 'adaptive_n_max', 'adaptive_n_stop', 'adaptive_abs_tol', 'adaptive_rel_tol',
                 'mutation_rate', 'mutation_factor', 'stop_tolerance', 'step_size', 'simplex_step', 'simplex_log_step',
                 'simplex_reflection', 'simplex_expansion', 'simplex_contraction', 'simplex_shrink', 'cooling',
                 'beta_max', 'bootstrap_max_obj', 'simplex_stop_tol', 'v_stop', 'gamma_prob', 'zeta', 'lambda',
                 'constraint_scale', 'neg_bin_r', 'stablizingCov',
                 'rhat_threshold', 'snooker_prob',
                 'powell_step', 'powell_stop_tol', 'cmaes_sigma0', 'cmaes_stop_tol',
                 # HMC (job_type = hmc, ADR-0059): NUTS dual-averaging target acceptance.
                 'target_accept']
multnumkeys = ['credible_intervals', 'beta', 'beta_range', 'starting_params', 'calculate_covari']
# The prior-family var keywords are derived from the registry (ADR-0010): each
# family yields {base}_var (linear) + log{base}_var (log10). Bounded-support
# families (b_var_def_keys) take the optional b/u flag in the grammar and have
# their bound read in ploop; unbounded ones (var_def_keys) don't. var/logvar are
# the no-prior Simplex start-point keywords (one or two numbers, no family).
b_var_def_keys, var_def_keys, one_param_var_keys = var_keyword_grammar()
# The two-number keywords are every var keyword except the one-parameter families.
two_param_var_keys = [k for k in var_def_keys + b_var_def_keys
                      if k not in one_param_var_keys]
var_def_keys_1or2nums = ['var', 'logvar']
strkeylist = ['bng_command', 'output_dir', 'fit_type', 'job_type', 'objfunc', 'objective',
              'profile_objective', 'initialization',
              'initialization_distribution',
              'cluster_type', 'scheduler_node', 'scheduler_file', 'de_strategy', 'sbml_integrator',
              'sbml_backend', 'bngl_backend', 'stochastic_seed', 'simulation_dir',
              'outlier_method', 'refine_method', 'noise_location']
multstrkeys = ['worker_nodes', 'postprocess', 'output_trajectory', 'output_noise_trajectory']
dictkeys = ['time_course', 'param_scan']
punctuation_safe = re.sub('[:,]', '', punctuation)


def parse(s):
    equals = pp.Suppress('=')
    colon = pp.Suppress(':')
    comment = pp.Suppress(pp.Optional(pp.Literal('#') - pp.ZeroOrMore(pp.Word(pp.printables))))
    # set up multiple grammars

    # single str value
    strkeys = _one_of(' '.join(strkeylist), caseless=True)
    string = pp.Word(pp.alphanums + punctuation)
    strgram = strkeys - equals - string - comment

    # single num value
    numkeys = _one_of(' '.join(numkeys_int + numkeys_float), caseless=True)
    point = pp.Literal(".")
    e = pp.CaselessLiteral("E")
    # An optionally-signed infinity (an open truncation side, ADR-0047 -- 'inf' / '-inf')
    # or a real number. inf is tried first so '-inf' matches it whole rather than the
    # real-number branch consuming the lone sign.
    inf_num = pp.Combine(pp.Optional(pp.Word("+-", exact=1)) + pp.CaselessLiteral("inf"))
    real_num = pp.Combine(pp.Word("+-" + pp.nums, pp.nums) +
                         pp.Optional(point + pp.Optional(pp.Word(pp.nums))) +
                         pp.Optional(e + pp.Word("+-" + pp.nums, pp.nums)))
    num = inf_num | real_num
    numgram = numkeys - equals - num - comment

    # variable definition grammar, split by the family's parameter count so the arity is
    # enforced at parse time (a clean error, not a downstream TypeError). The two-number
    # families (location/scale: mean/sd, location/b, shape/scale; and the bounded box) take
    # ``<p> <num> <num>`` plus the optional reflecting-bounds b/u flag a bounded family
    # carries; the one-parameter unbounded families (exponential scale, chisquare dof,
    # rayleigh scale -- ADR-0010/#417) take a single ``<p> <num>``.
    bng_parameter = pp.Word(pp.alphas, pp.alphanums + "_")
    two_param_keys = _one_of(' '.join(two_param_var_keys), caseless=True)
    two_param_nums = bng_parameter - num - num - pp.Optional(pp.Word("ubBU"))
    one_param_keys = _one_of(' '.join(one_param_var_keys), caseless=True)
    one_param_nums = bng_parameter - num
    strnumgram = ((two_param_keys - equals - two_param_nums)
                  | (one_param_keys - equals - one_param_nums)) - comment

    # multiple string value grammar
    multstrkey = _one_of(' '.join(multstrkeys), caseless=True)
    multstrgram = multstrkey - equals - pp.OneOrMore(string)

    # var and logvar alt grammar (only one number given)
    varkeys = _one_of(' '.join(var_def_keys_1or2nums), caseless=True)
    vargram = varkeys - equals - bng_parameter - num - pp.Optional(num) - comment

    # multiple num value
    multnumkey = _one_of(' '.join(multnumkeys), caseless=True)
    multnumgram = multnumkey - equals - pp.OneOrMore(num) - comment

    # model-data mapping grammar
    mdmkey = pp.CaselessLiteral("model")
    nonetoken = pp.Suppress(pp.CaselessLiteral("none"))
    model_file = pp.Regex(r".*?\.(bngl|xml|ant|target)")
    exp_file = pp.Regex(r".*?\.(exp|con|prop)")
    # A per-measurement binding-table sidecar (ADR-0045): a .tsv referenced by an
    # experiment's ``measurement_params:`` field, carrying a row-varying placeholder's
    # per-row token (an estimated id that cannot ride a float .exp column).
    param_file = pp.Regex(r".*?\.tsv")
    mdmgram = mdmkey - equals - model_file - colon - (_DelimitedList(exp_file) ^ nonetoken) - comment

    # new-era model declaration grammar (ADR-0028):
    #   model: <file>[, <file>...]
    # A pure model *declaration* -- data never binds here (it is introduced only by an
    # experiment's `data:` sub-field). Repeatable/accumulating across lines; modelId =
    # filename stem (uniqueness enforced when models load). It shares the ``model``
    # keyword with the legacy ``model = file : exp`` form, so it is tried first and
    # ``mdmkey + colon`` (a non-error-stop ``+``) lets a legacy ``model =`` line
    # backtrack cleanly to ``mdmgram``; once the colon commits, the file list is
    # error-stopped (``-``). A parse action tags it ``model_decl`` so ploop can route
    # the colon form -- whose tokens are otherwise shaped like ``mdmgram``'s -- to the
    # declaration handler. Edition-gated (>= 2) in config.py.
    model_decl_gram = mdmkey + colon - _DelimitedList(model_file) - comment
    model_decl_gram.set_parse_action(lambda t: ['model_decl'] + list(t)[1:])

    # normalization mapping grammar
    normkey = pp.CaselessLiteral("normalization")
    anything = pp.Word(pp.alphanums+punctuation+' ')
    normgram = normkey - equals - anything  # The set of legal grammars for normalization is too complicated,
    # Will handle with separate code.

    # new-era per-observable normalization grammar (ADR-0053, #444):
    #   normalization <obs> = <type>           -- per-observable (every experiment)
    #   normalization <exp>.<obs> = <type>      -- per-(experiment, observable) override
    # Normalization is a per-observable *prediction* transform -- a sibling of the
    # per-observable noise_model / cumulative surface (ADR-0021/0051) -- so the new era
    # keys it by observable (and optionally an experiment), never a filename. The bare
    # token before the ``=`` is what distinguishes this from the legacy / whole-fit
    # ``normalization = <type>[: <files>]`` form, so it is tried first and the recoverable
    # ``normkey + norm_target`` backtracks cleanly to ``normgram`` when no token precedes
    # the ``=`` (the whole-fit / legacy forms). The ``.`` in norm_target carries the
    # optional ``<experiment>.`` qualifier. The type is kept permissive (validated in
    # config.py) so a typo'd type gives a clear "Invalid normalization type" rather than a
    # parse failure. Tagged ``normalization_obs`` so ploop routes it; edition-gated (>= 2)
    # in config.py.
    norm_target = pp.Word(pp.alphas, pp.alphanums + '_.')
    norm_modern_gram = normkey + norm_target - equals - pp.Word(pp.alphas) - comment
    norm_modern_gram.set_parse_action(lambda t: ['normalization_obs'] + list(t)[1:])

    # Grammar for dictionary-like specification of simulation actions
    # We are intentionally over-permissive here, because the Action class will be able to give more helpful error
    # messages than a failed parse.
    dict_entry = pp.Word(pp.alphas) - colon - pp.Word(pp.alphanums + punctuation_safe)
    dict_key = _one_of(' '.join(dictkeys), caseless=True)
    dictgram = dict_key - equals - _DelimitedList(dict_entry) - comment

    # native noise model grammar (ADR-0021, ADR-0031):
    #   noise_model [<obs>] = <family>, <param> = <verb> [<arg>][, <param> = <verb> [<arg>]]...
    # e.g. ``noise_model obs2 = laplace, scale = fit b_obs2__FREE`` (per-observable) or
    # ``noise_model = gaussian, sigma = fix_at 1`` (the whole-fit default, no
    # observable). The verbs map to the SigmaSource kinds (fit -> free parameter,
    # read_exp_file -> data column, fix_at -> constant, relative -> constant-CV,
    # column_mean -> the column's mean); the arg is permissive (a __FREE name, a column
    # suffix like _SD, a number) and is interpreted in objective.py. ``relative`` takes
    # an optional CV and ``column_mean`` takes none, so the source arg is optional.
    noise_model_key = pp.CaselessLiteral('noise_model')
    nm_token = pp.Word(pp.alphas, pp.alphanums + '_')   # observable / family / param name
    nm_verb = _one_of('fit read_exp_file fix_at relative column_mean', caseless=True)
    nm_arg = pp.Word(pp.alphanums + '_+-.')
    # An optional ``location = mean|median`` field (the prediction's interpretation,
    # ADR-0024) rides alongside the ``<param> = <source>`` fields. MatchFirst tries
    # the ``location`` literal first, so a real noise-parameter name falls through to
    # a source field (and an invalid ``location = <x>`` errors rather than silently
    # parsing as a source, since the literal has committed).
    nm_location_field = pp.Group(pp.CaselessLiteral('location') - equals - _one_of('mean median', caseless=True))
    # The ``formula`` source (ADR-0044): an expression sigma over free parameters, e.g.
    # ``sigma = formula 0.1 + 0.05*scaling``. Its arg is the rest of the field (a PEtab
    # math expression: operators / parens / whitespace), captured up to the next comma or
    # comment so the comma-delimited field grammar is untouched (a noiseFormula carries no
    # comma). Uses ``+`` (not ``-``) so a non-``formula`` field backtracks to nm_source_field.
    nm_formula_arg = pp.Regex(r'[^,#\n]+').set_parse_action(lambda t: t[0].strip())
    nm_formula_field = pp.Group(nm_token + equals + pp.CaselessLiteral('formula') + nm_formula_arg)
    nm_source_field = pp.Group(nm_token - equals - nm_verb - pp.Optional(nm_arg))
    # An optional bare ``cumulative`` flag field (ADR-0051, #418): declares the column's
    # prediction a cumulative count, differenced to its per-interval increment before scoring.
    # A *prediction* transform, orthogonal to the noise family/source -- it rides the
    # per-observable noise_model line for ergonomics, but ploop stores it under its own
    # ('cumulative', observable) structural key. A bare literal (no ``=``), matched first so it
    # never shadows a ``<param> = <source>`` field.
    nm_cumulative_field = pp.Group(pp.CaselessLiteral('cumulative'))
    nm_field = nm_cumulative_field | nm_location_field | nm_formula_field | nm_source_field
    # The observable is optional: present -> a per-observable override; absent
    # (``noise_model = <family>``) -> the whole-fit default (ADR-0031). pyparsing
    # distinguishes them by whether a bare token precedes the ``=``.
    noise_model_gram = noise_model_key - pp.Optional(nm_token) - equals - nm_token - pp.Suppress(',') - \
        _DelimitedList(nm_field) - comment

    # native named-analytical-target grammar (ADR-0059 item 6):
    #   objective = <target>[, <const> = <number>]...
    # e.g. ``objective = banana, a = 1, b = 100``. The target is a built-in closed-form
    # objective whose scalar constants ride the objective line -- the noise_model field
    # grammar's shape, but with bare ``<name> = <number>`` fields (no verbs). The target name
    # is an explicit keyword set (canonical home: analytical_model.INLINE_TARGET_TYPES; kept
    # literal here to avoid an import cycle), so a *bare* token like ``objective = score`` is
    # NOT a target and backtracks -- via the leading ``+`` -- to the plain strgram. Constants
    # are optional; their defaults are documented and echoed at run start (config.py).
    objective_key = pp.CaselessLiteral('objective')
    objective_target_name = _one_of('banana', caseless=True)
    obj_const_field = pp.Group(nm_token + equals + num)
    objective_target_gram = (objective_key + equals + objective_target_name
                             + pp.Optional(pp.Suppress(',') + _DelimitedList(obj_const_field))
                             - comment)
    objective_target_gram.set_parse_action(lambda t: ['objective_spec'] + list(t)[1:])

    # bring-your-own analytical objective expression (ADR-0050):
    #   objective   = expression
    #   expression  = 0.5*((1 - x1)^2 + 100*(x2 - x1^2)^2)
    # ``objective = expression`` selects it (a plain ``objective`` string value -- not a named
    # target -- so it flows through ``strgram``), and the companion ``expression`` key carries
    # the user's PEtab-math NLL over the declared free parameters. The value grammar is kept
    # deliberately permissive (the rest of the line, up to a ``#`` comment) -- mirroring
    # ``normgram``'s ``anything`` and the measurement-model ``obs_formula`` -- because PEtab
    # math (operators / parens / ``^`` powers / function calls with internal commas) is too
    # rich to enumerate here; config.py compiles and validates it (an unparseable expression or
    # an undeclared symbol gives a pointed error there, not a parse failure). NB PEtab math uses
    # ``^`` for exponentiation, not ``**``.
    expression_key = pp.CaselessLiteral('expression')
    expression_value = pp.Regex(r'[^#\n]+').set_parse_action(lambda t: t[0].strip())
    expression_gram = expression_key - equals - expression_value - comment

    # mutant model grammar
    mutkey = pp.CaselessLiteral('mutant')
    mut_op = pp.Group(pp.Word(pp.alphas+'_', pp.alphanums+'_') - _one_of('+ - * / =') - num)
    mutgram = mutkey - equals - string - string - pp.Group(pp.OneOrMore(mut_op)) - \
        pp.Group(colon - (_DelimitedList(exp_file) ^ nonetoken)) - comment

    # new-era condition grammar (ADR-0028) -- a PyBNF Mutant = a PEtab Condition:
    #   condition: <name>[, model: <file>], perturbations: <var op val>[, <var op val>...]
    # A named set of parameter perturbations (op in = * / + -; ``=`` absolute, the rest
    # relative to the nominal value) -- the perturbation half of a legacy ``mutant``,
    # with NO data binding (data is introduced only by an experiment's ``data:``). The
    # ``model:`` sub-field is optional (omittable when there is a single model). Output:
    # ``['condition', <name>, <model-ref group>?, <perturbations group>]`` -- the model
    # ref (when present) and the perturbations are each a single ``pp.Group``, so their
    # positions are fixed (perturbations last; the model ref present iff len == 4) and
    # ploop reads them unambiguously. Edition-gated (>= 2) in config.py.
    condition_key = pp.CaselessLiteral('condition')
    cond_name = pp.Word(pp.alphas, pp.alphanums + '_')
    cond_model_key = pp.Suppress(pp.CaselessLiteral('model'))
    perturbations_key = pp.Suppress(pp.CaselessLiteral('perturbations'))
    cond_op = pp.Group(pp.Word(pp.alphas+'_', pp.alphanums+'_') - _one_of('+ - * / =') - num)
    cond_model_ref = pp.Group(pp.Suppress(',') + cond_model_key + colon + model_file)
    cond_perts = pp.Group(_DelimitedList(cond_op))
    condition_gram = condition_key + colon - cond_name + pp.Optional(cond_model_ref) + \
        pp.Suppress(',') + perturbations_key + colon - cond_perts - comment

    # new-era experiment grammar (ADR-0028) -- a PEtab Experiment carrying its data:
    #   experiment: <name>[, condition: <c>][, preequilibrate: <c0>][, model: <f>], data: <f1>[, <f2>...][, type: ...][, method: ...][, t_end: <t>][, t_start: <t0>][, n_steps: <n>]
    # A named simulation bound to its measurement files. The experiment NAME replaces the
    # legacy BNGL Suffix as the simulation's identity (it becomes both the action suffix and
    # the exp_data key); ``data:`` is a comma list whose multiple files are REPLICATES (all
    # measurements under the one experiment). The optional ``condition:`` names the Condition
    # to apply (omitted => wildtype), ``preequilibrate:`` names the Condition to equilibrate
    # under first (unmeasured, to steady state -- ADR-0052 pre-equilibration #440),
    # ``model:`` resolves the base model (omittable when one
    # model), ``type:`` overrides the data-driven type inference, ``method:`` the simulator,
    # and ``t_end:`` a parameter_scan's fixed endpoint time (absent => steady state, the
    # new-era scan default -- ADR-0046). Each labeled sub-field is a single pp.Group, combined
    # with pp.Each (``&``) so they may appear in ANY order after the name; only ``data:`` is
    # required. ploop reads the groups by their label, so order does not matter. Output:
    # ``['experiment', <name>, <field group>, ...]``. Edition-gated (>= 2) in config.py.
    experiment_key = pp.CaselessLiteral('experiment')
    exp_name = pp.Word(pp.alphas, pp.alphanums + '_')
    exp_field_token = pp.Word(pp.alphas, pp.alphanums + '_')
    exp_condition_field = pp.Group(pp.Suppress(',') + pp.CaselessLiteral('condition') + colon + cond_name)
    # The optional ``preequilibrate:`` field (ADR-0052, new-era pre-equilibration #440): names
    # the Condition the system equilibrates UNDER (unmeasured, to steady state -- PEtab time=-inf)
    # before the measurement ``condition:`` perturbs and the data grid is measured. Its presence
    # triggers the two-phase action synthesis in config.py (equilibrate -> setParameter -> measure,
    # state carried over). Mirrors ``condition:`` (a single condition name). config.py by label.
    exp_preequilibrate_field = pp.Group(
        pp.Suppress(',') + pp.CaselessLiteral('preequilibrate') + colon + cond_name)
    exp_model_field = pp.Group(pp.Suppress(',') + pp.CaselessLiteral('model') + colon + model_file)
    exp_data_field = pp.Group(pp.Suppress(',') + pp.CaselessLiteral('data') + colon + _DelimitedList(exp_file))
    exp_type_field = pp.Group(pp.Suppress(',') + pp.CaselessLiteral('type') + colon + exp_field_token)
    exp_method_field = pp.Group(pp.Suppress(',') + pp.CaselessLiteral('method') + colon + exp_field_token)
    # The optional ``t_end:`` field: a fixed simulation endpoint time. For a parameter_scan
    # (ADR-0046) it is the scan's measurement time (absent => steady state, PEtab time=inf,
    # the new-era default). For a *constraint-only* experiment (``data:`` is .con/.prop only,
    # so there is no measurement time grid to derive -- ADR-0028 addendum) it is the
    # time-course endpoint, the new-era home for the timing a legacy job kept in the model's
    # begin actions block. Ignored for a time course that has .exp data (its grid comes from
    # the data). A single number; config.py reads it by label.
    exp_tend_field = pp.Group(pp.Suppress(',') + pp.CaselessLiteral('t_end') + colon + num)
    # The optional ``t_start:`` field (ADR-0028 addendum): the integration start time for a
    # constraint-only experiment's synthesized time course (with ``t_end:``); the legacy
    # begin-actions ``t_start``. Absent => 0 (every config-action time course's default).
    # Inert when the grid comes from data (which forces a t=0 baseline). config.py by label.
    exp_tstart_field = pp.Group(pp.Suppress(',') + pp.CaselessLiteral('t_start') + colon + num)
    # The optional ``n_steps:`` field (ADR-0028 addendum): the number of uniform output steps
    # for a constraint-only experiment's synthesized time course (with ``t_end:``); the legacy
    # begin-actions ``n_steps``. Absent => the TimeCourse default (step = 1). Inert when the
    # grid comes from data. A single number; config.py reads it by label.
    exp_nsteps_field = pp.Group(pp.Suppress(',') + pp.CaselessLiteral('n_steps') + colon + num)
    # The optional per-measurement binding-table sidecar (ADR-0045): names a .tsv whose
    # per-row placeholder tokens config.py attaches to this experiment's exp Data.
    exp_measparams_field = pp.Group(
        pp.Suppress(',') + pp.CaselessLiteral('measurement_params') + colon + param_file)
    experiment_gram = experiment_key + colon - exp_name + \
        (pp.Optional(exp_condition_field) & pp.Optional(exp_preequilibrate_field)
         & pp.Optional(exp_model_field) & exp_data_field
         & pp.Optional(exp_type_field) & pp.Optional(exp_method_field)
         & pp.Optional(exp_tend_field) & pp.Optional(exp_tstart_field)
         & pp.Optional(exp_nsteps_field) & pp.Optional(exp_measparams_field)) - comment

    # new-era observable grammar (ADR-0028) -- a column-header override:
    #   observable: <entity>, column: <header>
    # By default a .exp column header IS the model observable/function name, and the
    # objective matches an experimental column to a simulation column BY NAME. This line is
    # the opt-in override for the common case where the measured data column is named
    # something other than the model entity: it maps the model <entity> to the data
    # <header>, so config.py can rename the data column <header> -> <entity> (and its
    # <header>_SD per-point noise companion, ADR-0021) and the by-name match succeeds --
    # without it a differently-named data column has no matching sim column and the
    # objective raises. One required ``column:`` field, no optionals. The key is the model
    # entity, the value the data header. Output: ['observable', <entity>, <header>].
    # Edition-gated (>= 2) in config.py.
    observable_key = pp.CaselessLiteral('observable')
    obs_entity = pp.Word(pp.alphas, pp.alphanums + '_')
    obs_column_key = pp.Suppress(pp.CaselessLiteral('column'))
    obs_column = pp.Word(pp.alphas, pp.alphanums + '_')
    # New-era measurement-model alternative (ADR-0036): ``observable: <id>, formula: <expr>``
    # declares a *measurement model* -- a PEtab observableFormula evaluated post-simulation
    # over the output trajectory (the observation layer), not a column rename. The ``formula``
    # keyword is *kept* in the output (not suppressed) so ``ploop`` distinguishes the two
    # forms by length; the formula is the rest of the line (a PEtab math expression -- internal
    # commas/spaces/parens allowed) up to an optional ``#`` comment. Output:
    # ``['observable', <id>, 'formula', <expr>]`` vs the column form's ``['observable', <entity>,
    # <header>]``.
    obs_formula_kw = pp.CaselessLiteral('formula')
    obs_formula = pp.Regex(r'[^#\n]+')
    observable_gram = observable_key + colon - obs_entity + pp.Suppress(',') + \
        ((obs_column_key + colon - obs_column)
         | (obs_formula_kw + colon - obs_formula)) - comment

    # new-era free-parameter record (ADR-0043) -- every part of the line is named:
    #   parameter: <id>[, prior: <family>][, parameter_scale: lin|log10|ln][, <field>: <num> ...]
    #             [, lower: <num>, upper: <num>][, initial_value: <num>]
    # (``parameter_scale`` is the sampling-space transform; a family's own ``scale`` field, e.g.
    #  cauchy/gamma/student_t, is a distinct distribution parameter -- no collision.)
    # The fully-labeled replacement for the legacy positional ``<family>_var = id p1 p2``
    # (which stays, edition-gated): no positional numbers, the family names its own params
    # (normal -> mean/sd, ...), bounds are named lower/upper, and the prior-truncation box of
    # #417 is just the lower/upper fields. Parsed permissively into ordered (field, value)
    # pairs -- the noise_model/observable pattern -- and config.py validates the field set
    # against the family + builds the FreeParameter. A field value is a number or a bare word
    # (a family name like ``normal`` or a scale like ``log10``). Output:
    # ``['parameter', <id>, [<field>, <value>], ...]``. Edition-gated (>= 2) in config.py.
    parameter_key = pp.CaselessLiteral('parameter')
    param_id = pp.Word(pp.alphas, pp.alphanums + '_')
    param_field_name = pp.Word(pp.alphas, pp.alphanums + '_')
    param_field_value = num | pp.Word(pp.alphas, pp.alphanums + '_')
    parameter_field = pp.Group(pp.Suppress(',') + param_field_name + colon - param_field_value)
    parameter_gram = parameter_key + colon - param_id - pp.ZeroOrMore(parameter_field) - comment

    # check each grammar and output somewhat legible error message
    parser = model_decl_gram | mdmgram | noise_model_gram | objective_target_gram | expression_gram | condition_gram | experiment_gram | observable_gram | parameter_gram | strgram | numgram | strnumgram | multnumgram | multstrgram | vargram | norm_modern_gram | normgram | dictgram | mutgram
    line = _parse_all(parser, s).asList()

    return line


def load_config(path):
    try:
        infile = open(path, 'r', encoding='utf-8', errors='replace')
    except FileNotFoundError:
        raise PybnfError(f'Configuration file {path} not found')
    with infile:
        param_dict = ploop(infile.readlines())
    return Configuration(param_dict)


def flatten(vs):
    return vs[0] if len(vs) == 1 else vs


def ploop(ls):  # parse loop
    d = {}
    models = set()
    exp_data = set()
    for i, line in enumerate(ls):
        if re.match(r'\s*$', line) or re.match(r'\s*#', line):
            continue
        try:
            logger.debug(f'Parsing line {line.strip()}')
            l = parse(line)

            # Find parameter assignments that reference distinct parameters
            if l[0] in b_var_def_keys:
                key = (l[0], l[1])
                values = [float(x) for x in l[2:4]]
                if len(l) == 5:
                    values.append(re.fullmatch('b', l[4], flags=re.IGNORECASE) is not None)
                else:
                    values.append(True)
            elif l[0] in var_def_keys_1or2nums or l[0] in var_def_keys:
                key = (l[0], l[1])
                values = [float(x) for x in l[2:]]
            elif l[0] in numkeys_int:
                key = l[0]
                values = int(l[1])
            elif l[0] in numkeys_float:
                key = l[0]
                values = float(l[1])
            elif l[0] in multnumkeys:
                key = l[0]
                values = [float(x) for x in l[1:]]
            elif l[0] in multstrkeys:
                key = l[0]
                values = l[1:]
            elif l[0] not in ('model', 'model_decl'):
                key = l[0]
                values = flatten(l[1:])

            # Find parameter assignments defining model and experimental data
            if l[0] == 'model':
                key = l[1]
                values = l[2:]
                d[key] = values  # individual data files remain in list
                models.add(key)
                exp_data.update(values)
            elif l[0] == 'model_decl':
                # New-era `model:` declaration (ADR-0028): a pure model declaration with
                # no data binding (data is introduced only by an experiment's `data:`).
                # Fold each file exactly like a legacy `model = file : none` line -- add
                # it to the models set with an empty exp list -- and accumulate the
                # declared files in the structural 'model' marker so config.py can
                # edition-gate the new syntax (>= edition 2). modelId = filename stem;
                # stem-uniqueness is enforced when models load (Model.name).
                for mf in l[1:]:
                    if mf not in d:
                        d[mf] = []
                    models.add(mf)
                d.setdefault('model', []).extend(l[1:])
            elif l[0] in dictkeys:
                # Multiple declarations allowed; config dict entry should contain a list of all the declarations.
                # Convert the line into a dict of key-value pairs. Keep everything as strings, check later
                entry = dict()
                for xi in range(0, len(values), 2):
                    if values[xi] in entry:
                        raise PybnfError(f'For config key {l[0]}, attribute {values[xi]} is specified multiple times')
                    entry[values[xi]] = values[xi+1]
                if l[0] in d:
                    d[l[0]].append(entry)
                else:
                    d[l[0]] = [entry]
            elif l[0] == 'mutant':
                if 'mutant' in d:
                    d['mutant'].append(l[1:])
                else:
                    d['mutant'] = [l[1:]]
                exp_data.update(l[-1])
            elif l[0] == 'condition':
                # New-era `condition:` (ADR-0028) -- a named set of parameter
                # perturbations on a base model (a PyBNF Mutant = a PEtab Condition),
                # with NO data binding. Store as a structural ('condition', name) tuple
                # key (like a noise_model key) -> (model_ref or None, [(var, op, val),
                # ...]); config.py edition-gates these and maps each to a MutationSet.
                # The perturbations are always the last group; the optional model ref is
                # l[2][0], present iff len(l) == 4 (one optional + one required group).
                name = l[1]
                perts = [tuple(op) for op in l[-1]]
                model_ref = l[2][0] if len(l) == 4 else None
                cond_key = ('condition', name)
                if cond_key in d:
                    raise PybnfError(f"Condition '{name}' is specified multiple times")
                d[cond_key] = (model_ref, perts)
            elif l[0] == 'experiment':
                # New-era `experiment:` (ADR-0028) -- a named simulation bound to its data:
                # files. Store as a structural ('experiment', name) tuple key (like a
                # condition / noise_model key) -> a dict of the labeled sub-fields. Each
                # field group is ['<label>', <value>...] (data carries a list, the rest a
                # single value); reading by label means the grammar's any-order pp.Each is
                # handled here without depending on group order. config.py edition-gates
                # these and synthesizes the TimeCourse/ParamScan action + exp_data entry.
                # The data files are also staged into the exp_data set so the normalization
                # key can validate against them (as legacy model/mutant lines do).
                name = l[1]
                fields = {}
                for grp in l[2:]:
                    label = grp[0].lower()
                    if label == 'data':
                        fields['data'] = list(grp[1:])
                    else:
                        fields[label] = grp[1]
                exp_key = ('experiment', name)
                if exp_key in d:
                    raise PybnfError(f"Experiment '{name}' is specified multiple times")
                d[exp_key] = fields
                exp_data.update(fields.get('data', []))
            elif l[0] == 'observable':
                # New-era `observable:` -- either a column-header override (ADR-0028, Chunk 4)
                # or a measurement-model formula (ADR-0036). The grammar keeps the 'formula'
                # keyword in the output so the two forms are distinguished here by length:
                #   column form  -> ['observable', <entity>, <header>]        (len 3)
                #   formula form -> ['observable', <id>, 'formula', <expr>]   (len 4)
                # The column form stores a structural ('observable', entity) tuple key ->
                # the data column header (config.py renames the data column <header> ->
                # <entity>, and <header>_SD -> <entity>_SD, so the objective's by-name match
                # succeeds). The formula form stores a ('measurement', id) tuple key -> the
                # PEtab observableFormula string; config.py compiles it into the measurement-
                # model observation layer (evaluated post-simulation, ADR-0036).
                if len(l) == 4 and str(l[2]).lower() == 'formula':
                    obs_id, expr = l[1], l[3].strip()
                    meas_key = ('measurement', obs_id)
                    if meas_key in d:
                        raise PybnfError(f"Observable '{obs_id}' is specified multiple times")
                    d[meas_key] = expr
                else:
                    entity, header = l[1], l[2]
                    obs_key = ('observable', entity)
                    if obs_key in d:
                        raise PybnfError(f"Observable '{entity}' is specified multiple times")
                    d[obs_key] = header
            elif l[0] == 'parameter':
                # New-era parameter record (ADR-0043): ['parameter', <id>, [field, value], ...].
                # Store under a structural ('parameter', id) tuple key -> an ordered dict of the
                # named string fields (prior/scale/<family params>/lower/upper/initial_value).
                # config.py edition-gates these and builds the FreeParameter; keeping the values
                # as strings here lets config.py do the float/family-aware interpretation.
                pid = l[1]
                fields = {}
                for grp in l[2:]:
                    fname, fval = grp[0].lower(), grp[1]
                    if fname in fields:
                        raise PybnfError(f"Parameter '{pid}': field '{fname}' is specified multiple times")
                    fields[fname] = fval
                pkey = ('parameter', pid)
                if pkey in d:
                    raise PybnfError(f"Parameter '{pid}' is specified multiple times")
                d[pkey] = fields
            elif l[0] == 'noise_model':
                # noise_model [<obs>] = <family>, <param> = <verb> [<arg>][, location = mean|median]
                # (ADR-0021, ADR-0024, ADR-0031). Store as a structural ('noise_model',
                # observable) tuple key (like a free-parameter key) -> (family,
                # {param: (verb, arg)}, location); objective.py interprets the tokens into
                # a (NoiseModel, SigmaSource). The observable is None for the whole-fit
                # default line ``noise_model = <family>, ...``. The observable is present
                # iff a bare token precedes the family (l[2] is then the family string;
                # otherwise l[1] is the family and l[2] is the first field group).
                if isinstance(l[2], str):
                    observable, family, raw_fields = l[1], l[2], l[3:]
                else:
                    observable, family, raw_fields = None, l[1], l[2:]
                where = "the whole-fit noise_model" if observable is None else f"noise_model for {observable}"
                fields = {}
                location = None
                cumulative = False
                for field in raw_fields:
                    if field[0].lower() == 'cumulative':
                        # A bare flag (ADR-0051, #418): a per-observable prediction transform,
                        # orthogonal to the noise spec -- stored under its own ('cumulative',
                        # observable) key below, not folded into the (family, fields, location)
                        # noise tuple.
                        if cumulative:
                            raise PybnfError(f"In {where}, cumulative is specified multiple times")
                        cumulative = True
                        continue
                    if field[0].lower() == 'location':
                        if location is not None:
                            raise PybnfError(f"In {where}, location is specified multiple times")
                        location = field[1].lower()
                        continue
                    param, verb = field[0], field[1]
                    arg = field[2] if len(field) > 2 else None   # relative/column_mean may omit it
                    if param in fields:
                        raise PybnfError(f"In {where}, noise parameter '{param}' "
                                         "is specified multiple times")
                    fields[param] = (verb, arg)
                nm_key = ('noise_model', observable)
                if nm_key in d:
                    target = "The whole-fit noise_model" if observable is None else f"noise_model for observable '{observable}'"
                    raise PybnfError(f"{target} is specified multiple times")
                d[nm_key] = (family, fields, location)
                if cumulative:
                    # The transform differences ONE column's cumulative counts into per-interval
                    # increments, so it is inherently per-observable; a whole-fit ('cumulative',
                    # None) would mean "every column is cumulative", an easy foot-gun -- reject it.
                    if observable is None:
                        raise PybnfError(
                            "The whole-fit noise_model line cannot be 'cumulative'",
                            "The cumulative->incident prediction transform (#418) is per-observable: "
                            "it differences one column's cumulative counts to per-interval "
                            "increments. Declare it on a per-observable line, e.g. "
                            "'noise_model <obs> = <family>, <param> = <source>, cumulative'.")
                    d[('cumulative', observable)] = True
            elif l[0] == 'objective_spec':
                # Named analytical objective with inline constants (ADR-0059 item 6):
                # ``objective = banana, a = 1, b = 100``. Record the target name as the modern
                # ``objective`` value (so the edition-gated objective-key machinery and the
                # "exactly one objective" check engage), plus a structural
                # ('objective_target', None) key carrying the (name, {const: value}) the model
                # synthesis reads -- the structural-tuple-key pattern noise_model uses.
                # Constants are optional; config.py applies + echoes the documented defaults.
                target_name = l[1]
                consts = {}
                for field in l[2:]:
                    cname, cval = field[0], field[1]
                    if cname in consts:
                        raise PybnfError(f"In 'objective = {target_name}', constant "
                                         f"'{cname}' is specified multiple times")
                    consts[cname] = float(cval)
                if 'objective' in d or ('objective_target', None) in d:
                    raise PybnfError("The 'objective' key is specified multiple times")
                d['objective'] = target_name
                d[('objective_target', None)] = (target_name, consts)
            elif l[0] == 'postprocess':
                if len(values) < 2:
                    raise PybnfError("Config key 'postprocess' should specify a python file, followed by one or more "
                                     "suffixes.")
                if 'postprocess' in d:
                    d['postprocess'].append(values)
                else:
                    d['postprocess'] = [values]
            elif l[0] == 'normalization':
                # Normalization defined with way too many possible options
                # At the end of all this, the config dict has one of the following formats:
                # 'normalization' : 'type'
                # 'normalization' : {'expfile':'type', 'expfile2':[('type1', [numbers]), ('type2', [colnames]), ...]}

                parsed = parse_normalization_def(values)
                if type(parsed) == str:
                    if 'normalization' in d:
                        raise PybnfError('contradictory normalization keys',
                                         "Config file contains multiple 'normalization' keys, one of which specifies"
                                         " no specific exp files, thereby applying to all of them. If you are using "
                                         "this option, you should only have one 'normalization' key in the config file.")
                    d['normalization'] = parsed
                else:
                    if 'normalization' in d:
                        if type(d['normalization']) != dict:
                            raise PybnfError('contradictory normalization keys',
                                             "Config file contains multiple 'normalization' keys, one of which specifies"
                                             " no specific exp files, thereby applying to all of them. If you are using "
                                             "this option, you should only have one 'normalization' key in the config file.")
                    else:
                        d['normalization'] = dict()
                    for k in parsed:
                        if k in d['normalization'] and (type(parsed[k]) == str or type(d['normalization'][k]) == str):
                            raise PybnfError(f'contradictory normalization keys for {k}',
                                             f"File {k} has normalization specified multiple times in a way that is "
                                             "contradictory.")
                        if type(parsed[k]) == str:
                            d['normalization'][k] = parsed[k]
                        else:
                            if k not in d['normalization']:
                                d['normalization'][k] = []
                            d['normalization'][k].append(parsed[k])
            elif l[0] == 'normalization_obs':
                # New-era per-observable normalization (ADR-0053, #444):
                # ``normalization <target> = <type>`` where <target> is an observable, or
                # ``<experiment>.<observable>`` for a per-experiment override. Stored as a
                # structural ('normalization', target) tuple key -- a sibling of
                # ('noise_model', obs) / ('cumulative', obs) -- so it rides the structural-key
                # path (ADR-0014, never flagged unused) and is resolved against the
                # (experiment x column) grid in config.py::_postprocess_normalization, where
                # the type is also validated. The whole-fit default stays under the plain
                # 'normalization' string key (above); the two coexist as specificity layers.
                target, ntype = l[1], l[2]
                norm_key = ('normalization', target)
                if norm_key in d:
                    raise PybnfError(f"normalization for '{target}' is specified multiple times")
                d[norm_key] = ntype.lower()
            else:
                if key in d:
                    if d[key] == values:
                        print1(f"Warning: Config key '{key}' is specified multiple times")
                    else:
                        raise PybnfError(f"Config key '{key}' is specified multiple times with different values.")
                d[key] = values

        except pp.ParseBaseException:
            # Split on space, '=', and ':' so a colon-form key (the new-era
            # ``model:`` / ``experiment:`` / ... syntax) reports the bare keyword.
            key = re.split('[ =:]', line)[0].lower()
            fmt = ''
            if key in numkeys_int:
                fmt = f"'{key}=x' where x is an integer"
            elif key in numkeys_float:
                fmt = f"'{key}=x' where x is a decimal number"
            elif key in multnumkeys:
                fmt = f"'{key}=x1 x2 ...' where x1, x2, ... is a list of numbers"
            elif key in var_def_keys:
                fmt = f"'{key}=v x y' where v is a variable name, and x and y are numbers"
            elif key in b_var_def_keys:
                fmt = f"'{key}=v x y z' where v is a variable name, x and y are numbers, and z is optional and specifies " \
                      "whether or not the variable should be bounded ('u' is unbounded, 'b' or left blank is bounded)"
            elif key in var_def_keys_1or2nums:
                fmt = f"'{key}=v x' or '{key}=v x y' where v is a variable name, and x and y are decimal numbers"
            elif key in strkeylist:
                fmt = f"'{key}=s' where s is a string"
            elif key == 'model':
                fmt = "'model=modelfile.bngl : datafile.exp' or 'model=modelfile.bngl : datafile1.exp, datafile2.exp'" \
                      " (legacy), or the new-era declaration 'model: modelfile.bngl' or " \
                      "'model: modelfile1.bngl, modelfile2.bngl' (requires edition >= 2)." \
                      " Supported modelfile extensions are .bngl, .xml, .ant, and .target"
            elif key == 'normalization':
                fmt = f"'{key}=s' or '{key}=s : datafile1.exp, datafile2.exp' where s is a string ('init', 'peak', " \
                      "'unit', or 'zero')"
            elif key in dictkeys:
                fmt = f"'{key}=key1: value1, key2: value2,...' where key1, key2, etc are attributes of the {key} (see " \
                      "documentation for available options)"
            elif key == 'mutant':
                fmt = "'mutant=base model var1=val1 var2*val2 ... : datafile1.exp, datafile2.exp' where mutation " \
                      "operations (var1=val1 etc) have the format [variable_name][operator][number] and other " \
                      "arguments are strings"
            elif key == 'condition':
                fmt = "'condition: name, perturbations: var1 op val1, var2 op val2, ...' where op is one of " \
                      "= * / + - , optionally with 'model: modelfile' before perturbations (requires edition >= 2)"
            elif key == 'experiment':
                fmt = "'experiment: name, data: file1.exp[, file2.exp ...]' (data files may be .exp " \
                      "measurements and/or .con/.prop constraints) optionally with 'condition: c', " \
                      "'preequilibrate: c0' (equilibrate under c0 to steady state, unmeasured, before " \
                      "measuring -- ADR-0052), " \
                      "'model: modelfile', 'type: time_course|parameter_scan', 'method: ode|ssa|pla|nf', " \
                      "'t_end: <number>' (a parameter_scan's fixed endpoint, or a constraint-only " \
                      "experiment's time-course endpoint), 't_start: <number>' / 'n_steps: <number>' " \
                      "(a constraint-only experiment's integration start / output resolution), or " \
                      "'measurement_params: file.tsv' in any order (requires edition >= 2)"
            elif key == 'observable':
                fmt = "'observable: entity, column: header' mapping a model observable/function name to a " \
                      "differently-named data column header (requires edition >= 2)"
            elif key == 'parameter':
                fmt = "'parameter: id, prior: <family>, <field>: <num>, ...' with named fields -- e.g. " \
                      "'parameter: k, prior: normal, mean: 0, sd: 1, lower: -5, upper: 5' or " \
                      "'parameter: k, prior: uniform, lower: 0, upper: 10' (optionally parameter_scale: log10|ln, " \
                      "initial_value: x) (requires edition >= 2)"

            message = f"Parsing configuration key '{key}' on line {i}.\n"
            if fmt == '':
                message += f'{key} is not a valid configuration key.'
            else:
                message += f'{key} should be specified in the format {fmt}'

            raise PybnfError(f"Misconfigured config key '{line.strip()}' at line: {i}", message)

    d['models'] = models
    d['exp_data'] = exp_data
    return d


def parse_normalization_def(s):
    """
    Parse the complicated normalization grammar
    If the grammar is specified incorrectly, it will end up calling something invalid the normalization type or the
    exp file, and this error will be caught later.

    :param s: The string following the equals sign in the normalization key
    :return: What to write in the config dictionary: A string, or a dictionary {expfile: string} or
    {expfile: (string, index_list)} or {expfile: (string, name_list)}
    """

    def parse_range(x):
        """Parse a string as a set of numbers like 10,"""
        result = []
        for part in x.split(','):
            if '-' in part:
                a, b = part.split('-')
                a, b = int(a), int(b)
                result.extend(range(a, b + 1))
            else:
                a = int(part)
                result.append(a)
        return result

    # Remove all spaces
    s = re.sub(r'\s', '', s)
    if ':' in s:
        # List of exp files
        res = dict()
        i = s.index(':')
        normtype = s[:i]
        explist = s[i+1:]
        exps = re.split(r',(?![^()]*\))', explist) # Dark magic: split on commas that aren't inside parentheses
        # Achievement unlocked: Use 16 punctuation marks in a row
        for e in exps:
            if e[0] == '(' and e[-1] == ')':
                # It's an exp in parentheses with column-wise specs
                pair = e[1:-1].split(':')
                if len(pair) == 1:
                    res[pair[0]] = normtype
                elif len(pair) == 2:
                    e, cols = pair
                    if re.match(r'^[\d,\-]+$', cols):
                        col_nums = parse_range(cols)
                        res[e] = (normtype, col_nums)
                    else:
                        col_names = cols.split(',')
                        res[e] = (normtype, col_names)
                else:
                    raise PybnfError(f"Parsing normalization key - the item '{e}' has too many colons in it")
            else:
                # It's just an exp
                res[e] = normtype
        return res
    else:
        # Single string for all
        return s
