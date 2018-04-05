"""pybnf.data: class with methods to manage experimental and simulation data"""


import math
import numpy as np
import re
from .printing import PybnfError


class Data(object):
    """Top level class for managing data"""

    def __init__(self, file_name=None, arr=None):
        self.cols = dict()  # dict of column headers to column indices
        self.data = None  # Numpy array for data
        self.indvar = None # Name of the independent variable
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

    def load_data(self, file_name, sep='\s+', flags=()):
        """
        Loads column data from a text file

        :param file_name: Name of data file
        :type file_name: str
        :param sep: String that separates columns
        :type sep: str
        :param flags: Additional specifications about how to load the file. Current options: 'copasi-scan' -
        triggers some additional processing to read the output of a Copasi parameter scan. 'time' - force the name of
        the independent variable to be 'time', instead of the actual header contents (in Copasi, the header instead
        says 'Time'.)
        :type flags: Iterable of str
        :return: None
        """
        with open(file_name) as f:
            lines = f.readlines()

        self.data = self._read_file_lines(lines, sep, file_name=file_name, flags=flags)

    def _read_file_lines(self, lines, sep, file_name='', flags=()):
        """Helper function that reads lines from BNGL gdat files"""

        header = re.split(sep, lines[0].strip().strip('#').strip())
        # Ignore parentheses added to functions in BNG 2.3, and [] added to species names in COPASI
        header = [h.strip('()[]') for h in header]
        if 'copasi-scan' in flags:
            header[0] = re.sub('Values\[', '', header[0])
            header[0] = re.sub('\]\.InitialValue', '', header[0])
            header[0] = re.sub('\]', '', header[0])  # For scan on initial conditions
        if 'time' in flags:
            header[0] = 'time'
        ncols = len(header)
        self.indvar = header[0]

        for c in header:
            l = len(self.cols)
            self.cols[c] = l

        data = []
        for i, l in enumerate(lines[1:]):
            if re.match('^\s*$', l) or re.match('\s*#', l):
                continue
            try:
                num_list = [self._to_number(x) for x in re.split(sep, l.strip())]
            except ValueError as err:
                raise PybnfError('Parsing %s on line %i: %s' % (file_name, i+2, err.args[0]))
            if len(num_list) != ncols:
                raise PybnfError('Parsing %s on line %i: Found %i values, expected %i' %
                                 (file_name, i+2, len(num_list), ncols))
            data.append(num_list)

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

    def normalize_to_peak(self, idx=0, cols='all'):
        """
        Normalizes all data columns (except the independent variable) to the peak
        value in their respective columns

        Updates the data array in this object, returns none.

        :param idx: Index of independent variable
        :type idx: int
        :param cols: List of column indices to normalize, or 'all' for all columns but independent variable
        :return: Normalized Numpy array (including independent variable column)
        """
        if cols == 'all':
            cols = list(range(self.data.shape[1]))
            cols.remove(idx)
        for c in cols:
            self.data[:, c] = self.data[:, c] / np.max(self.data[:, c])

    def normalize_to_init(self, idx=0, cols='all'):
        """
        Normalizes all data columns (except the independent variable) to the initial
        value in their respective columns

        Updates the data array in this object, returns none.

        :param idx: Index of independent variable
        :type idx: int
        :param cols: List of column indices to normalize, or 'all' for all columns but independent variable
        """
        if cols == 'all':
            cols = list(range(self.data.shape[1]))
            cols.remove(idx)
        for c in cols:
            self.data[:, c] = self.data[:, c] / self.data[0, c]

    def normalize_to_zero(self, idx=0, bc=True, cols='all'):
        """
        Normalizes data so that each column's mean is 0

        Updates the data array in this object, returns none.

        :param idx: Index of independent variable
        :type idx: int
        :param bc: If True, the standard deviation is normalized by 1/(N-1). If False, by 1/N.
        :type bc: bool
        :param cols: List of column indices to normalize, or 'all' for all columns but independent variable
        """
        if cols == 'all':
            cols = list(range(self.data.shape[1]))
            cols.remove(idx)
        ddof = 0 if not bc else 1
        for c in cols:
            col = self.data[:, c]
            col -= np.mean(col)
            std = np.std(col, ddof=ddof)
            if std != 0:
                col /= std
            self.data[:, c] = col

    def _subtract_baseline(self, idx=0, cols='all'):
        if cols == 'all':
            cols = list(range(self.data.shape[1]))
            cols.remove(idx)
        for c in cols:
            col = self.data[:, c]
            self.data[:, c] = col - self.data[0, c]

    def normalize_to_unit_scale(self, idx=0, cols='all'):
        """
        Scales data so that the range of values is between (min-init)/(max-init) and 1.  If the maximum value is 0
        (i.e. max == init), then the data is scaled by the minimum value after subtracting the initial value
        so that the range of values is between 0 and -1

        :param idx: Index of independent variable
        :type idx: int
        :param cols: List of column indices to normalize, or 'all' for all columns but independent variable
        :type: list or str
        :return:
        """
        if cols == 'all':
            cols = list(range(self.data.shape[1]))
            cols.remove(idx)
        self._subtract_baseline(idx, cols)
        for c in cols:
            cmax = np.max(self.data[:, c])
            if cmax == 0.0:
                self.data[:, c] = self.data[:, c] / np.abs(np.min(self.data[:, c]))
            else:
                self.data[:, c] = self.data[:, c] / np.max(self.data[:, c])

    @staticmethod
    def average(datas):
        """
        Calculates the average of several data objects.
        The input Data objects should have the same column labels and independent variable values (NOT CURRENTLY
        CHECKED)

        :param datas: Iterable of Data objects of identical size to be averaged
        :return: Data object
        """
        output = Data()
        output.cols = datas[0].cols
        output.data = np.mean(np.stack([d.data for d in datas]), axis=0)
        return output

    def normalize(self, method):
        """
        Normalize the data according to the specified method: 'init', 'peak', 'unit', or 'zero'
        The method could also be a list of ordered pairs [('init', [columns]), ('peak', [columns])], where columns
        is a list of integers or column labels

        Updates the data array in this object, returns none.
        """

        def normalize_once(m, cols):
            if m == 'init':
                self.normalize_to_init(cols=cols)
            elif m == 'peak':
                self.normalize_to_peak(cols=cols)
            elif m == 'zero':
                self.normalize_to_zero(cols=cols)
            elif m == 'unit':
                self.normalize_to_unit_scale(cols=cols)
            else:
                # Should have caught a user-defined invalid setting in config before getting here.
                raise ValueError('Invalid method %s for Data.normalize()' % m)

        if type(method) == str:
            normalize_once(method, 'all')
        else:
            for mi, cols_i in method:
                if type(cols_i[0]) == str:
                    # Convert to int indices
                    cols_i = [self.cols[c] for c in cols_i]
                normalize_once(mi, cols_i)
