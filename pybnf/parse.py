import pyparsing as pp
import re
from string import punctuation

numkeys_int = ['verbosity', 'parallel_count', 'seed', 'delete_old_files', 'max_generations', 'population_size',
               'smoothing', 'max_parents', 'force_different_parents', 'keep_parents', 'divide_by_init',
               'log_transform_sim_data', 'standardize_sim_data', 'standardize_exp_data', 'max_iterations',
               'num_to_output', 'output_every', 'islands', 'migrate_every', 'num_to_migrate', 'init_size',
               'local_min_limit', 'reserve_size', 'burn_in', 'sample_every', 'output_hist_every',
               'hist_bins', 'refine']
numkeys_float = ['extra_weight', 'swap_rate', 'min_objfunc_value', 'cognitive', 'social', 'particle_weight',
                 'particle_weight_final', 'adaptive_n_max', 'adaptive_n_stop', 'adaptive_abs_tol', 'adaptive_rel_tol',
                 'mutation_rate', 'mutation_factor', 'stop_tolerance', 'step_size', 'simplex_step', 'simplex_log_step',
                 'simplex_reflection', 'simplex_expansion', 'simplex_contraction', 'simplex_shrink']
multnumkeys = ['credible_intervals']
var_def_keys = ['random_var', 'lognormrandom_var', 'loguniform_var', 'normrandom_var', 'static_list_var', 'mutate']
var_def_keys_1or2nums = ['var', 'logvar']

def parse(s):
    equals = pp.Suppress('=')
    colon = pp.Suppress(':')
    comment = pp.Suppress(pp.Optional(pp.Literal('#') - pp.ZeroOrMore(pp.Word(pp.printables))))
    # set up multiple grammars

    # single str value
    strkeys = pp.oneOf(
        'bng_command\
        job_name\
        output_dir\
        fit_type\
        objfunc\
        initialization\
        ', caseless=True)
    string = pp.Word(pp.alphanums + punctuation)
    strgram = strkeys - equals - string - comment

    # single num value
    numkeys = pp.oneOf(' '.join(numkeys_int + numkeys_float), caseless=True)
    point = pp.Literal(".")
    e = pp.CaselessLiteral("E")
    num = pp.Combine(pp.Word("+-" + pp.nums, pp.nums) +
                         pp.Optional(point + pp.Optional(pp.Word(pp.nums))) +
                         pp.Optional(e + pp.Word("+-" + pp.nums, pp.nums)))
    numgram = numkeys - equals - num - comment

    # multiple str and num value
    strnumkeys = pp.oneOf(' '.join(var_def_keys), caseless=True)
    bng_parameter = pp.Word(pp.alphas, pp.alphanums + "_")
    varnums = bng_parameter - num - num
    strnumgram = strnumkeys - equals - varnums - comment

    # var and logvar alt grammar (only one number given)
    varkeys = pp.oneOf(' '.join(var_def_keys_1or2nums), caseless=True)
    vargram = varkeys - equals - bng_parameter - num - pp.Optional(num) - comment

    # static_list_var grammar
    slvkey = pp.oneOf('static_list_var', caseless=True)
    slvgram = slvkey - equals - bng_parameter - pp.OneOrMore(num) - comment

    # multiple num value
    multnumkey = pp.oneOf(' '.join(multnumkeys), caseless=True)
    multnumgram = multnumkey - equals - pp.OneOrMore(num) - comment

    # model-data mapping grammar
    mdmkey = pp.CaselessLiteral("model")
    bngl_file = pp.Regex(".*?\.bngl")
    exp_file = pp.Regex(".*?\.exp")
    mdmgram = mdmkey - equals - bngl_file - colon - pp.delimitedList(exp_file) - comment

    # check each grammar and output somewhat legible error message
    line = (mdmgram | strgram | numgram | strnumgram | slvgram | multnumgram | vargram).parseString(s, parseAll=True).asList()

    return line


def load_config(path):
    infile = open(path, 'r')
    param_dict = ploop(infile.readlines())
    infile.close()
    return param_dict


def flatten(vs):
    return vs[0] if len(vs) == 1 else vs


def ploop(ls):  # parse loop

    d = {}
    models = set()
    exp_data = set()
    for i, line in enumerate(ls):
        if re.match('\s*$', line) or re.match('\s*#', line):
            continue
        try:
            l = parse(line)

            # Find parameter assignments that reference distinct parameters
            if l[0] in var_def_keys or l[0] in var_def_keys_1or2nums:
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
            else:
                key = l[0]
                values = flatten(l[1:])

            # Find parameter assignments defining model and experimental data
            if l[0] == 'model':
                key = l[1]
                values = l[2:]
                d[key] = values  # individual data files remain in list
                models.add(key)
                exp_data.update(values)
            else:
                d[key] = values

        except:
            message = "misconfigured parameter '%s' at line: %s" % (line.strip(), i)
            print(message)
            raise  # Rethrow the error that was raised, for easier debugging.
            #raise Exception(message)

    d['models'] = models
    d['exp_data'] = exp_data
    return d


class IllegalParamException(Exception):
    def __init__(self, value):
        print(self.message)
