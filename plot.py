"""
python script for parsing and plotting gdat files
"""

import matplotlib.pyplot as plt
import os
import re
#x
xtitle = ""
xdata = []
#y
ytitles = []
ydata = []

title = ""

def load(path):
    try:
        infile = open(path, 'r')
        title = os.path.basename(path)
    except FileNotFoundError:
        raise PybnfError('Configuration file %s not found' % path)
    plotloop(infile.readlines())
    infile.close()

def parse(s):
	plist = re.split('\s', s)
	data = []
	for n, i in enumerate(plist):
		cl = i.strip()
		if cl != "" and cl != "#":
			try: 
				data.append(float(cl))
			except:
				data.append(cl)
	print(data)
	return(data)

def plotloop(path):
	for i, line in enumerate(path):
		l = parse(line)
		if i == 0:
			xtitle = l[0]
			print(xtitle)
			ytitles = l[1:]
			print(ytitles)
			for n in range(len(ytitles)):
				ydata.append([])
		else:
			xdata.append(l[0])
			for n in range(len(ytitles)):
				ydata[n].append(l[n+1])
			print(ydata)
	for n in range(len(ytitles)):
		plt.plot(xdata, ydata[n])
		plt.xlabel(xtitle)
		plt.legend(ytitles, loc='upper left')
		plt.title(title)
	plt.show()

load("/Users/alex/Desktop/example1/timecourse.exp")

