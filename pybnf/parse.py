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
               'local_objective_eval', 'reps_per_beta', 'save_best_data', 'parallelize_models', 'adaptive', 'continue_run',
               'delta', 'archive_size', 'archive_thin_rate', 'adaptive_step_size', 'powell_max_iterations',
               'max_failed_simulations', 'random_seed', 'sbml_ssa_strict', 'diagnostics_every']
numkeys_float = ['min_objective', 'cognitive', 'social', 'particle_weight',
                 'particle_weight_final', 'adaptive_n_max', 'adaptive_n_stop', 'adaptive_abs_tol', 'adaptive_rel_tol',
                 'mutation_rate', 'mutation_factor', 'stop_tolerance', 'step_size', 'simplex_step', 'simplex_log_step',
                 'simplex_reflection', 'simplex_expansion', 'simplex_contraction', 'simplex_shrink', 'cooling',
                 'beta_max', 'bootstrap_max_obj', 'simplex_stop_tol', 'v_stop', 'gamma_prob', 'zeta', 'lambda',
                 'constraint_scale', 'neg_bin_r', 'stablizingCov',
                 'rhat_threshold', 'snooker_prob',
                 'powell_step', 'powell_stop_tol', 'cmaes_sigma0', 'cmaes_stop_tol']
multnumkeys = ['credible_intervals', 'beta', 'beta_range', 'starting_params', 'calculate_covari']
# The prior-family var keywords are derived from the registry (ADR-0010): each
# family yields {base}_var (linear) + log{base}_var (log10). Bounded-support
# families (b_var_def_keys) take the optional b/u flag in the grammar and have
# their bound read in ploop; unbounded ones (var_def_keys) don't. var/logvar are
# the no-prior Simplex start-point keywords (one or two numbers, no family).
b_var_def_keys, var_def_keys = var_keyword_grammar()
var_def_keys_1or2nums = ['var', 'logvar']
strkeylist = ['bng_command', 'output_dir', 'fit_type', 'objfunc', 'initialization',
              'cluster_type', 'scheduler_node', 'scheduler_file', 'de_strategy', 'sbml_integrator',
              'sbml_backend', 'bngl_backend', 'stochastic_seed', 'simulation_dir',
              'outlier_method', 'refine_method']
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
    num = pp.Combine(pp.Word("+-" + pp.nums, pp.nums) +
                         pp.Optional(point + pp.Optional(pp.Word(pp.nums))) +
                         pp.Optional(e + pp.Word("+-" + pp.nums, pp.nums)))
    numgram = numkeys - equals - num - comment

    # variable definition grammar
    strnumkeys = _one_of(' '.join(var_def_keys + b_var_def_keys), caseless=True)
    bng_parameter = pp.Word(pp.alphas, pp.alphanums + "_")
    varnums = bng_parameter - num - num - pp.Optional(pp.Word("ubBU"))
    strnumgram = strnumkeys - equals - varnums - comment

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
    mdmgram = mdmkey - equals - model_file - colon - (_DelimitedList(exp_file) ^ nonetoken) - comment

    # normalization mapping grammar
    normkey = pp.CaselessLiteral("normalization")
    anything = pp.Word(pp.alphanums+punctuation+' ')
    normgram = normkey - equals - anything  # The set of legal grammars for normalization is too complicated,
    # Will handle with separate code.

    # Grammar for dictionary-like specification of simulation actions
    # We are intentionally over-permissive here, because the Action class will be able to give more helpful error
    # messages than a failed parse.
    dict_entry = pp.Word(pp.alphas) - colon - pp.Word(pp.alphanums + punctuation_safe)
    dict_key = _one_of(' '.join(dictkeys), caseless=True)
    dictgram = dict_key - equals - _DelimitedList(dict_entry) - comment

    # mutant model grammar
    mutkey = pp.CaselessLiteral('mutant')
    mut_op = pp.Group(pp.Word(pp.alphas+'_', pp.alphanums+'_') - _one_of('+ - * / =') - num)
    mutgram = mutkey - equals - string - string - pp.Group(pp.OneOrMore(mut_op)) - \
        pp.Group(colon - (_DelimitedList(exp_file) ^ nonetoken)) - comment

    # check each grammar and output somewhat legible error message
    parser = mdmgram | strgram | numgram | strnumgram | multnumgram | multstrgram | vargram | normgram | dictgram | mutgram
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
            elif l[0] != 'model':
                key = l[0]
                values = flatten(l[1:])

            # Find parameter assignments defining model and experimental data
            if l[0] == 'model':
                key = l[1]
                values = l[2:]
                d[key] = values  # individual data files remain in list
                models.add(key)
                exp_data.update(values)
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
            else:
                if key in d:
                    if d[key] == values:
                        print1(f"Warning: Config key '{key}' is specified multiple times")
                    else:
                        raise PybnfError(f"Config key '{key}' is specified multiple times with different values.")
                d[key] = values

        except pp.ParseBaseException:
            key = re.split('[ =]', line)[0].lower()
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
