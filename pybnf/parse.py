import pyparsing as pp
import re

parameters = {}  # define empty dict


def separate(toks):  # separate the key from the value
    l = toks.asList()
    return l


def parse(str):
    linegrammar = pp.Word(pp.alphanums) + "=" + pp.Word(pp.alphanums)  # <-- grammar defined here
    lineaction = linegrammar.setParseAction(separate)  # set lineaction equal to the result of parse action
    #   print (str, "->", lineaction.parseString(str))
    parameters[lineaction.parseString(str)[0]] = lineaction.parseString(str)[2]  # set the key to the value
    # print(parameters)


def ploop(path):  # parse loop
    with open(path, "r") as infile:
        for line in infile:
            if re.match('\s*$', line) or re.match('\s*#', line):
                continue
            parse(line)
    return parameters

