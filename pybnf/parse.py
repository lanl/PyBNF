import pyparsing as pp
import re


def parse(s):
    equals = pp.Suppress('=')
    comment = pp.Suppress(pp.Optional(pp.Literal('#') - pp.ZeroOrMore(pp.Word(pp.printables))))
    # set up multiple grammars

    # single str value
    strkeys = pp.oneOf('bng_command job_name', caseless=True)
    string = pp.Word(pp.alphanums + "_")
    strgram = strkeys - equals - string - comment

    # single num value
    numkeys = pp.oneOf(
        'verbosity\
        parallel_count\
        seed\
        delete_old_files\
        max_generations\
        population_size\
        smoothing\
        min_objfunc_value\
        objfunc\
        extra_weight\
        swap_rate\
        max_parents\
        force_different_parents\
        keep_parents\
        divide_by_init\
        log_transform_sim_data\
        standardize_sim_data\
        standardize_exp_data',
        caseless=True)
    point = pp.Literal(".")
    e = pp.CaselessLiteral("E")
    num = pp.Combine(pp.Word("+-" + pp.nums, pp.nums) +
                         pp.Optional(point + pp.Optional(pp.Word(pp.nums))) +
                         pp.Optional(e + pp.Word("+-" + pp.nums, pp.nums)))
    numgram = numkeys - equals - num - comment

    # multiple str value
    strskeys = pp.oneOf('output_dir model exp_file', caseless=True)
    strings = pp.OneOrMore(pp.Word(pp.printables))
    strsgram = strskeys - equals - strings - comment

    # multiple str and num value
    strnumkeys = pp.oneOf('mutate random_var lognormrandom_var loguniform_var', caseless=True)
    bng_parameter = pp.Word(pp.alphas, pp.alphanums + "_")
    varnums = bng_parameter - num - num
    strnumgram = strnumkeys - equals - varnums - comment

    # static_list_var grammar
    slvkey = pp.oneOf('static_list_var', caseless=True)
    slvgram = slvkey - equals - bng_parameter - pp.OneOrMore(num) - comment

    # check each grammar and output somewhat legible error message
    line = (strgram | numgram | strsgram | strnumgram | slvgram).parseString(s, parseAll=True).asList()

    return line


def load_config(path):
    infile = open(path, 'r')
    param_dict = ploop(infile.readlines())
    infile.close()
    return param_dict


def ploop(ls):  # parse loop
    d = {}

    for i, line in enumerate(ls):
        if re.match('\s*$', line) or re.match('\s*#', line):
            continue
        try:
            l = parse(line)

            # Find parameter assignments that reference distinct parameters
            var_def_keys = set(['random_var', 'lognormrandom_var', 'loguniform_var', 'static_list_var', 'mutate'])
            if l[0] in var_def_keys:
                key = (l[0], l[1])
                values = l[2:]
            else:
                key = l[0]
                values = l[1:]

            # Assign values to keys
            if len(values) == 1:
                d[key] = values[0]
            else:
                d[key] = values  # set key to values
        except:
            message = "misconfigured parameter '%s' at line: %s" % (line.strip(), i)
            #               print (message)
            raise Exception(message)

    return d


class IllegalParamException(Exception):
    def __init__(self, value):
        print(self.message)
