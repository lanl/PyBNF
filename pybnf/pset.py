import math
import numpy as np
import re


class Pset(object):
    """Class representing a parameter set"""

    def __init__(self):
        pass
        # Initialize parameters (self.x = ...)

    @staticmethod
    def to_number(x):
        """
        Attempts to convert a string to a float

        :param x: str
        :return: float
        """
        if re.match('\b[nN][aA][nN]\b', x):
            return math.nan
        elif re.match('\b[iI][nN][fF]\b', x):
            return math.inf
        elif re.match('\b-[iI][nN][fF]\b', x):
            return -math.inf
        else:
            return float(x)
