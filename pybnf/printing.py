"""
Printing commands that respect the application-wide verbosity setting.
"""

verbosity = 1


def print0(s):
    """Print the statement at any verbosity level"""
    print(s)


def print1(s):
    """Print the statement only if the verbosity level is at least 1"""
    if verbosity >= 1:
        print(s)


def print2(s):
    """Print the statement only if the verbosity level is 2"""
    if verbosity >= 2:
        print(s)


class PybnfError(RuntimeError):
    """
    Represents a user-generated error for which we can provide an informative message to the user about what
    went wrong with the input before quitting.
    """
    def __init__(self, message):
        self.message = message
