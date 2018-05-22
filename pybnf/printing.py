"""Contains printing commands that respect the application-wide verbosity setting."""


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
    def __init__(self, log_message, user_message=None):
        """
        :param log_message: The message to print to the log
        :param user_message: The message to output to the user. If omitted, the user gets the same message as
        log_message.
        """
        self.log_message = log_message
        if user_message:
            self.message = user_message
        else:
            self.message = log_message
