"""pybnf.pybnf: defines the entry point for the PyBNF application"""


__version__ = "0.1"

import argparse
from pybnf.parse import ploop

def main():
    print("PyBNF-- version %s" % __version__)
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', action='store', dest='conf_file',
                         help='Path to the BioNetFit configuration file' metavar='config.txt')
    
    results = parser.parse_args()
   # print(results.conf_file)
    ploop(results.conf_file)