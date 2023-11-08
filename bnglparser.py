"""
*made to turn parameters into free parameters
*very rudimentary "parser"
"""
def parse(s):
	nocomment = s.split("#")
	plist = nocomment[0].split(" ")
	return(plist)

def load(path, path2, vlist):
    try:
        infile = open(path, 'r')
    except FileNotFoundError:
        raise PybnfError('Configuration file %s not found' % path)
    ploop(infile.readlines(), path2, vlist)
    infile.close()

def ploop(path, path2, vlist):
	lol = []
	for i, line in enumerate(path):
		l = parse(line)
		for n, i in enumerate(l):
			if i.strip() != "":
				for j in vlist:
					h = j.replace('__FREE', '')
					if i.strip() == h:
						try:
							val = l[n+1]
						except:
							val = ""
						l[n+1] = str(h + "__FREE")
						l.append(str("#" + h + "=" + val))
		#make sure the new lines aren't deleted
		if l[-1].strip() != "\n":
			l.append("\n")
		#append to list of lists
		lol.append(l)
	#write list of lists to file
	pwrite(path2, lol)

def pwrite(path, lisp):
	file = open(path, 'w')
	for n in lisp:
		string = ' '.join(n)
		file.write(string)

#load("/Users/alex/Desktop/PyBNF/examples/demo/parabola.bngl", "/Users/alex/Desktop/tarabola.bngl", ['v1__FREE', 'v2__FREE', 'v3__FREE'])