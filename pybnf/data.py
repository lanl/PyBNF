import math
import numpy as np
import re


class Data(object):
    """Top level class for managing data"""

    def __init__(self, file_name=None, arr=None):
        self.cols = dict()  # dict of column headers to column indices
        self.data = None  # Numpy array for data
        if file_name is not None:
            self.load_data(file_name)
        elif arr is not None:
            self.data = arr

    def __getitem__(self, col_header):
        """
        Gets a column of data based on its column header
        
        :param col_header: Data column name
        :type col_header: str
        :return: Numpy array corresponding to name
        """
        idx = self.cols[col_header]
        return self.data[:, idx]

    def get_row(self, col_header, value):
        """
        Returns the (first) data row in which field col_header is equal to value.
        This should typically be used for col_header as the independent variable.

        :param col_header: Data column name
        :type col_header str
        :param value:
        :type value: str
        :return: 1D numpy array consisting of the requested row
        """

        c_idx = self.cols[col_header]
        rows = self.data[self.data[:, c_idx] == value, :]

        if rows.size == 0:
            return None

        return rows[0, :]

    @staticmethod
    def _to_number(x):
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
            l = len(self.cols)
            self.cols[c] = l

        data = []
        for l in lines[1:]:
            data.append([self._to_number(x) for x in re.split(sep, l.strip())])

        return np.array(data)

    def _dep_cols(self, idx):
        """
        Returns all data columns without independent variable

        :param idx: Column index for independent variable (defaults to 0)
        :type idx: int
        :return: Numpy array of observable data
        """
        return np.delete(self.data, idx, axis=1)

    def _ind_col(self, idx):
        """
        Returns data column corresponding to independent variable

        :param idx: Column index for independent variable (defaults to 0)
        :type idx: int
        :return: 1-D Numpy array of independent variable values
        """
        return self.data[:, idx]

    def normalize_to_peak(self, idx=0):
        """
        Normalizes all data columns (except the independent variable) to the peak
        value in their respective columns

        :param idx: Index of independent variable
        :type idx: int
        :return: Normalized Numpy array (including independent variable column)
        """
        ind = self._ind_col(idx)
        l = ind.shape[0]
        dep = self._dep_cols(idx)
        dep_rel = dep / np.amax(dep, axis=0)
        return np.hstack((ind.reshape((l, 1)), dep_rel))

    def normalize_to_init(self, idx=0):
        """
        Normalizes all data columns (except the independent variable) to the initial
        value in their respective columns

        :param idx: Index of independent variable
        :type idx: int
        :return: Normalized Numpy array (including independent variable column)
        """
        ind = self._ind_col(idx)
        l = ind.shape[0]
        dep = self._dep_cols(idx)
        dep_rel = dep / dep[0, :]
        return np.hstack((ind.reshape((l, 1)), dep_rel))

    def normalize_to_zero(self, idx=0, bc=True):
        """
        Normalizes data so that each column's mean is 0

        :param idx: Index of independent variable
        :type idx: int
        :return: Normalized Numpy array (including independent variable column)
        """
        ind = self._ind_col(idx)
        l = ind.shape[0]
        dep = self._dep_cols(idx)
        col_means = np.mean(dep, axis=0)
        ddof = 0 if not bc else 1

        def norm_by_std(col):
            col_std = np.std(col, ddof=ddof)
            if col_std == 0.:
                return col
            else:
                return col / col_std

        col_norm = np.apply_along_axis(norm_by_std, 0, (dep - col_means))
        return np.hstack((ind.reshape((l, 1)), col_norm))
