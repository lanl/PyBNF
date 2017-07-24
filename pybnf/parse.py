from pyparsing import *
import re

parameters = {}  # define empty dict

def parse(str):
    
    keys = oneOf('output_dir bng_command population_size fit_type cluster_software parallel_count')('key') 
    equals = Suppress('=')
    value = Word(alphanums)('value')
    
    linegrammar = keys + equals  + value  # <-- grammar defined here
    
    line = linegrammar.parseString(str) #parse string
    
    parameters[line.key] = line.value  # set the key to the value
    #print(parameters)


def ploop(path):  # parse loop
    with open(path, "r") as infile:
        for line in infile:
            if re.match('\s*$', line) or re.match('\s*#', line):
                continue
            parse(line)
    return parameters

