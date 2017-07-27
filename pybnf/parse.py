import pyparsing as pp
import re
from collections import defaultdict

parameters = {}  # define empty dict
d = defaultdict(list)
def parse(s):
    
    lkeys = pp.oneOf('model exp_files')
    keys = pp.oneOf('output_dir bng_command population_size fit_type cluster_software parallel_count')('key') 
    equals = pp.Suppress('=')
    value = pp.Word(pp.alphanums)('value')
    values = pp.OneOrMore(value)

    linegrammar = keys + equals  + values  # <-- grammar defined here
    
    line = linegrammar.parseString(s, parseAll=True) #parse string
    
  #  print(line.values)
    d[line.key] = line.values  # set the key to the values
    #print(line.values)
    print(d.items())


def ploop(path):  # parse loop
    with open(path, "r") as infile:
        for line in infile:
            if re.match('\s*$', line) or re.match('\s*#', line):
                continue
            parse(line)
    return parameters

ploop('con.txt')