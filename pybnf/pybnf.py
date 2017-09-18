"""pybnf.pybnf: defines the entry point for the PyBNF application"""


__version__ = "0.1"

import argparse

def main():
    print("PyBNF-- version %s" % __version__)
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-c', action='store_const', dest='conf_file',
                        const='value-to-store', help='config file path')
    
    results = parser.parse_args()
    print(results.conf_file)