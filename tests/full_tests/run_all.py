#!/usr/bin/env python3
"""
This script runs a series of full-length test problems. 
It should be run to validate that PyBNF is working before any release.
The problems are designed to be as short as possible while testing all / as many as possible features of PyBNF
The total run time is about 30 minutes

Run with no arguments to run locally
Run with the argument ssh to use PyBNF's automatic cluster setup with dask-ssh
Run with the argument sf to use manual cluster setup, with a preconfigured dask cluster with scheduler file named sf
Bash files are provided in this folder to run this script using the cluster options. 

This test script is not Windows compatible due to extensive use of `grep`

The output file test_summary.txt should be manually checked to confirm success of the tests. 
The output can be compared to example_summary.txt, which is the result from running on commit e34ba3b (14 Mar 2019)
shortly before the v1.0.0 release.
"""

import subprocess as sp
from sys import argv
import traceback as tb

def run_all():
    """
    Run all test cases and write summary to test_summary.txt
    """

    outstream = open(output_name,'w')
    
    try:
        run_test('T1','T1-ssprop','polynomial.conf',outstream)
    except Exception:
        outstream.write('Error on T1!\n')
        outstream.write(tb.format_exc())
    
    try:
        run_test('T2','T2-ade-abcd','fit_ade.conf',outstream)
        # Also check refined fit
        bestname, bestvalue = read_best_fit('T2-ade-abcd/fit/Results/sorted_params_refine_final.txt')
        outstream.write('After refinement, best objective was %s with param set %s\n' % (bestvalue, bestname))
    except Exception:
        outstream.write('Error on T2!\n')
        outstream.write(tb.format_exc())
    
    try:
        run_test('T3','T3-de-egg','egg-de.conf',outstream)
        # Also check bootstrap results
        bestname, bestvalue = read_best_fit('T3-de-egg/fit/Results-boot0/sorted_params_final.txt')
        outstream.write('Bootstrap 0 best objective was %s with param set %s\n' % (bestvalue, bestname))
        bestname, bestvalue = read_best_fit('T3-de-egg/fit/Results-boot1/sorted_params_final.txt')
        outstream.write('Bootstrap 1 best objective was %s with param set %s\n' % (bestvalue, bestname))
        bestname, bestvalue = read_best_fit('T3-de-egg/fit/Results-boot2/sorted_params_final.txt')
        outstream.write('Bootstrap 2 best objective was %s with param set %s\n' % (bestvalue, bestname))
    except Exception:
        outstream.write('Error on T3!\n')
        outstream.write(tb.format_exc())
    
    try:
        run_test('T4','T4-pso-nf','rnf.conf',outstream)
    except Exception:
        outstream.write('Error on T4!\n')
        outstream.write(tb.format_exc())
    
    try:
        run_test('T5','T5-pt-trivial','trivial_pt.conf',outstream)
        # Also check credible intervals
        with open('T5-pt-trivial/fit/Results/credible68_final.txt') as f:
            f.readline() # header
            parts_a = f.readline().split()
            parts_b = f.readline().split()
            parts_c = f.readline().split()
            outstream.write('68%% credible for a = [%s, %s] (ground truth [9, 11])\n' % (parts_a[1], parts_a[2]))
            outstream.write('68%% credible for b = [%s, %s] (ground truth [6.6, 13.4])\n' % (parts_b[1], parts_b[2]))
            outstream.write('68%% credible for c = [%s, %s] (ground truth [9, 11])\n' % (parts_c[1], parts_c[2]))
        with open('T5-pt-trivial/fit/Results/credible95_final.txt') as f:
            f.readline() # header
            parts_a = f.readline().split()
            parts_b = f.readline().split()
            parts_c = f.readline().split()
            outstream.write('95%% credible for a = [%s, %s] (ground truth [8, 12])\n' % (parts_a[1], parts_a[2]))
            outstream.write('95%% credible for b = [%s, %s] (ground truth [5.25, 14.75])\n' % (parts_b[1], parts_b[2]))
            outstream.write('95%% credible for c = [%s, %s] (ground truth [8, 12])\n' % (parts_c[1], parts_c[2]))
    except Exception:
        outstream.write('Error on T5!\n')
        outstream.write(tb.format_exc())
        
    try:
        run_test('T6','T6-check','polynomial.conf',outstream, display_best=False)
        try:
            grep = sp.run(['grep', 'Objective', 'T6_stdout.out'],
                      check=True, stdout=sp.PIPE, universal_newlines=True)
        except sp.CalledProcessError:
            outstream.write('Failed to read Objective!\n')
        else:
            outstream.write(grep.stdout)
        try:
            grep = sp.run(['grep', 'Satisfied', 'T6_stdout.out'],
                      check=True, stdout=sp.PIPE, universal_newlines=True)
        except sp.CalledProcessError:
            outstream.write('Failed to read number satisfied!\n')
        else:
            outstream.write(grep.stdout)
    except Exception:
        outstream.write('Error on T6!\n')
        outstream.write(tb.format_exc())

def run_test(name, folder, conffile, outstream, display_best=True):
    """
    Run the specified test case, writing summary to outstream
    """
    
    outstream.write('\nTest %s:\n' % name)
    

    proc = sp.run(['pybnf', '-c', '%s/%s' % (folder, conffile), '-l', name, '-o'],
                  stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)
    with open('%s_stdout.out' % name, 'w') as out:
        out.write(proc.stdout)
    with open('%s_stderr.out' % name, 'w') as out:
        out.write(proc.stderr)

    if proc.returncode != 0:
        outstream.write('Failed!\n')
        return
    
    outstream.write('Completed\n')
    
    if display_best:
        bestname, bestvalue = read_best_fit('%s/fit/Results/sorted_params_final.txt' % folder)
        outstream.write('Best objective was %s with param set %s\n' % (bestvalue,bestname))
    
    try:
        grep = sp.run(['grep', '-i', 'fitting time', '%s.log' % name],
                      check=True, stdout=sp.PIPE, universal_newlines=True)
    except sp.CalledProcessError:
        outstream.write('Failed to read run time!\n')
    else:
        outstream.write(grep.stdout)
    
    outstream.write('Log has %s lines\n' % wc_summary('%s.log' % name))
    outstream.write('Stdout has %s lines\n' % wc_summary('%s_stdout.out' % name))
    outstream.write('Stderr has %s lines\n' % wc_summary('%s_stderr.out' % name))

def wc_summary(path):
    """
    Return the number of lines in the file, or None if failed
    """
    try:
        wc = sp.run(['wc','-l',path], stdout=sp.PIPE, check=True, universal_newlines=True)
    except sp.CalledProcessError:
        return None
    
    parts = wc.stdout.split()
    return int(parts[0])

def read_best_fit(path):
    """
    Return a 2-tuple consisting of the name and the best-fit objective function value, 
    given the path to sorted_params.txt
    """    
    
    with open(path) as f:
        f.readline() # header
        line = f.readline() # first pset (best)
        parts = line.split()
        name = parts[0]
        value = float(parts[1])
    return (name, value)
    

if __name__=='__main__':
    if len(argv) == 1:
        extra_args = []
        output_name = 'test_summary.txt'
    elif argv[1] == 'ssh':
        extra_args = ['-t', 'slurm']
        output_name = 'test_summary_ssh.txt'
    elif argv[1] == 'sf':
        extra_args = ['-s', 'sf']
        output_name = 'test_summary_sf.txt'
    else:
        raise ValueError('Invalid argument '+argv[1])
    run_all()
