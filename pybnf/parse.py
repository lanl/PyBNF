import pyparsing as pp
import re

def parse(s):

    equals = pp.Suppress('=')

    #set up multiple grammars

    #single str value 
    strkeys = pp.oneOf('bng_command job_name')
    string = pp.Word(pp.alphas)
    strgram = strkeys + equals + string
    
    #single num value
    numkeys = pp.oneOf('verbosity parallel_count seed delete_old_files max_generations population_size smoothing min_objfunc_value objfunc extra_weight swap_rate max_parents force_different_parents keep_parents divide_by_init log_transform_sim_data standardize_sim_data standardize_exp_data')
    num = pp.Word(pp.nums)
    numgram = numkeys + equals + num
    
    #multiple str value
    strskeys = pp.oneOf('output_dir model exp_file')
    strings = pp.OneOrMore(pp.Word(pp.alphas))
    strsgram = strskeys + equals + strings
    
    #multiple str and num value
    strnumkeys = pp.oneOf('mutate')
    nums = pp.OneOrMore(pp.Word(pp.nums))
    strnumgram = strnumkeys + equals + strings + nums
    
    line = (strgram | numgram | strsgram | strnumgram).parseString(s, parseAll=True).asList() 
    
    return line
         
    
def ploop(path):  # parse loop
    d = {}
    
    with open(path, "r") as infile:
        for line in infile:
            if re.match('\s*$', line) or re.match('\s*#', line):
                continue
            l = parse(line)
            print (l)
            key = l[0]
            values = l[1:]
            d[key] = values #set key to values
            
    print (d)
    return d
    
ploop("con.txt")

