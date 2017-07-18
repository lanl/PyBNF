import math
import numpy as np
import re


class Data(object):
    """Top level class for managing data"""

    def __init__(self):
        self.cols = dict()  # dictionary with column header, column index pairs
        self.data = np.array()  # Numpy array for data

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

