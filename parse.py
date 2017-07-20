import pyparsing as pp

parameters = {} #define empty dict

def separate(toks): #separate the key from the value
        l = toks.asList()
        return l
        
def parse(str):
    linegrammar = pp.Word(pp.alphanums) + "=" + pp.Word(pp.alphanums) # <-- grammar defined here
    lineaction = linegrammar.setParseAction(separate) # set lineaction equal to the result of parse action
 #   print (str, "->", lineaction.parseString(str))
    parameters[lineaction.parseString(str)[0]] = lineaction.parseString(str)[2] #set the key to the value
    print (parameters)
                
def ploop(path): #parse loop
    with open(path, "r") as infile:
        for line in infile:
            if not(len(line.strip())) == 0 and not(line.startswith("#")):
                parse(line)
                
