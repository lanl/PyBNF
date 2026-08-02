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

    Two independent knobs shape what the user sees. ``user_message`` **replaces** the log message -- for a
    diagnosis that is too technical to show as-is, restated in user-facing terms. ``hint`` **appends** to it --
    for the remedy the user needs *alongside* the diagnosis, not instead of it. A refusal that carries only a
    generic remedy in ``user_message`` silently discards its own reason (#527): the reason reaches the log and
    the user is told to give up without being told what to fix.
    """
    def __init__(self, log_message, user_message=None, hint=None):
        """
        :param log_message: The message to print to the log
        :param user_message: The message to output to the user *in place of* log_message. If omitted, the user
        gets the same message as log_message.
        :param hint: One suggested remedy, or a sequence of them, appended to the user-facing message as an
        indented ``->`` line each. Adds to the message rather than replacing it, so the user gets both what
        went wrong and what to do about it.
        """
        self.log_message = log_message
        self.hints = [hint] if isinstance(hint, str) else list(hint or ())
        self.message = user_message if user_message else log_message
        for h in self.hints:
            self.message += '\n  -> ' + h
