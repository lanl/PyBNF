#import libraries
import argparse

#import other modules
import parse

parser = argparse.ArgumentParser(prog='BioNetFit2')
parser.add_argument('-c', help='set configure file path', type=str)
args = parser.parse_args()
#parse config file
ploop(args.c)