"""
Printing commands that respect the application-wide verbosity setting.
"""

verbosity = 1


def print0(s):
    print(s)


def print1(s):
    if verbosity >= 1:
        print(s)


def print2(s):
    if verbosity >= 2:
        print(s)
