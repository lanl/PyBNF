import math
import numpy as np
import re


class Data(object):
    """Top level class for managing data"""

    def __init__(self):
        self.cols = dict()  # dictionary with column header, column index pairs
        self.data = None  # Numpy array for data

    def __getitem__(self, col_header):
        """
        Gets a column of data based on its column header
        
        :param col_header: Data column name
        :type col_header: str
        :return: Numpy array corresponding to name
        """
        idx = self.cols[col_header]
        return self.data[:, idx]

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

    def load_data(self, file_name, sep='\s+'):
        """
        Loads column data from a text file

        :param file_name: Name of data file
        :type file_name: str
        :param sep: String that separates columns
        :type sep: str
        :return: None
        """
        with open(file_name) as f:
            lines = f.readlines()

        self.data = self._read_file_lines(lines, sep)

    def _read_file_lines(self, lines, sep):
        """Helper function that reads lines from BNGL gdat files"""

        header = re.split(sep, lines[0].strip().strip('#').strip())

        for c in header:
            self.cols[c] = len(self.cols)

        data = []
        for l in lines[1:]:
            data.append([self.to_number(x) for x in re.split(sep, l.strip())])

        return np.array(data)
