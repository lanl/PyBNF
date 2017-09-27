"""pybnf.pybnf: defines the entry point for the PyBNF application"""


import logging
import argparse
from pybnf.parse import ploop

__version__ = "0.1"


def main():
    log_format = "%(levelname)s\t%(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
            
    logging.info("PyBNF v%s" % __version__)
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', action='store', dest='conf_file',
    help='Path to the BioNetFit configuration file' metavar='config.txt')
    
    results = parser.parse_args()
    ploop(results.conf_file)