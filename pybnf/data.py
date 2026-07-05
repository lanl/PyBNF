"""Class with methods to manage experimental and simulation data"""


import logging
import numpy as np
import re
from dataclasses import dataclass
from typing import Optional
from .printing import PybnfError


logger = logging.getLogger(__name__)


@dataclass
class OutputSensitivities:
    """Forward output sensitivities (∂g/∂θ) attached to a simulated :class:`Data`.

    Carried through from the bngsim ``Result.output_sensitivities`` tensor on the
    gradient path (#385/#447), in **native parameter space** (no log/scale
    transform -- PyBNF owns that, ADR-0029). Purely additive: a scalar-path
    ``Data`` leaves :attr:`Data.output_sensitivities` ``None`` and is byte-identical
    to before this feature existed.

    The ``d_param`` / ``d_ic`` tensors have shape ``(n_times, n_selectors, n_axis)``,
    aligned column-for-column with :attr:`selectors` (typed selectors matching the
    observable/expression columns of the owning ``Data``). The third axis is labelled
    by :attr:`param_names` (for ``d_param``) or :attr:`ic_species` (for ``d_ic``).
    Consumers (#449) address a column by selector via :meth:`slice_for`.
    """

    selectors: list           # typed selectors, e.g. ['observable:Atot', ...]
    param_names: list         # axis labels for d_param (sensitivity_params order)
    ic_species: list          # axis labels for d_ic (sensitivity_ic_species order)
    d_param: Optional[np.ndarray] = None   # (n_times, n_selectors, n_params) or None
    d_ic: Optional[np.ndarray] = None      # (n_times, n_selectors, n_ic) or None

    def slice_for(self, selector, axis='parameter'):
        """Return the ``(n_times, n_axis)`` sensitivity slice for one selector.

        ``axis='parameter'`` reads :attr:`d_param`; ``axis='ic'`` reads
        :attr:`d_ic`. Raises ``KeyError`` if the selector was not requested and
        ``ValueError`` if the requested axis was not computed.
        """
        try:
            col = self.selectors.index(selector)
        except ValueError:
            raise KeyError(selector)
        if axis == 'parameter':
            tensor = self.d_param
        elif axis == 'ic':
            tensor = self.d_ic
        else:
            raise ValueError("axis must be 'parameter' or 'ic', got %r" % (axis,))
        if tensor is None:
            raise ValueError("sensitivity axis %r was not computed" % (axis,))
        return tensor[:, col, :]


@dataclass
class NormalizationRecord:
    """How one column of a simulated :class:`Data` was normalized (#453/#385).

    :meth:`Data.normalize` rescales a *predicted* observable before scoring (``init`` /
    ``peak`` / ``zero`` / ``unit``, ADR-0053) -- a θ-dependent transform of the moving
    trajectory, so the gradient path must thread the normalizer's own derivative through
    ``∂(raw/N)/∂θ`` (a quotient/chain rule that couples rows). The transform happens at the
    ``Data`` level and overwrites the raw column in place, so the few facts the chain rule
    needs -- the divisor ``N`` and the reference row(s) it is read from -- are recorded here
    at normalize time, before the raw values are gone. Purely additive: a ``Data`` that is
    never normalized leaves :attr:`Data.normalization` ``None`` and is byte-identical, and
    recording changes no data value (only this sidecar). The chain-rule *interpretation*
    lives in :mod:`pybnf.gradient.assembly` (this is a plain fact holder; ``data.py`` knows
    no gradient math).

    Every method's ``∂(normalized_i)/∂θ`` is a function of the **raw** per-row sensitivity
    ``s_k = ∂raw_k/∂θ`` (the #447 tensor, untouched by normalization) and the **normalized**
    column values ``n_k`` (read back from the now-rescaled ``Data``):

    * ``peak``/``init``: ``n_i = raw_i / N`` with ``N`` the column max (``ref_row`` =
      argmax) or the initial value (``ref_row`` = 0) -- ``∂n_i/∂θ = (s_i - n_i·s_ref)/N``.
    * ``unit``: ``n_i = (raw_i - raw_0)/N`` (baseline-subtracted, ``baseline_row`` = 0) with
      ``N`` the max-after-baseline (``sign`` = +1, ``ref_row`` = argmax) or, in the
      degenerate max==baseline branch, ``|min|`` (``sign`` = -1, ``ref_row`` = argmin) --
      ``∂n_i/∂θ = ((s_i - s_base) - sign·n_i·(s_ref - s_base))/N``.
    * ``zero`` (z-score): ``n_i = (raw_i - μ)/σ`` couples **all** rows through ``σ`` --
      ``scale`` = ``σ`` (0 when std is 0, where ``Data`` leaves the column un-divided) and
      ``ddof`` carries the ``K - ddof`` denominator of ``∂σ/∂θ``.
    """

    method: str                          # 'peak' | 'init' | 'zero' | 'unit'
    scale: float                         # the divisor N (peak: max; init: raw_0; zero: std; unit: max/|min|)
    ref_row: Optional[int] = None        # the row N is read from (peak/init/unit); None for zero
    baseline_row: Optional[int] = None   # unit subtracts this row first (0); None for peak/init/zero
    sign: float = 1.0                    # unit-min branch flips the reference term's sign
    ddof: int = 0                        # zero: the K - ddof denominator of the std derivative


class Data:
    """Top level class for managing data"""

    def __init__(self, file_name=None, arr=None, named_arr=None):
        """
        Initializes a Data instance.  Must specify either a file name or an array

        :param file_name:
        :param arr:
        """
        self.cols = dict()  # dict of column headers to column indices
        self.headers = dict()  # dict of column indices to headers
        self._data = None  # Numpy array for data
        self._observers = []  # For implementing the observer pattern
        self.weights = None  # Numpy array for bootstrapping weights
        self.indvar = None  # Name of the independent variable
        # Forward output sensitivities (#385/#447): None on the scalar path,
        # an OutputSensitivities payload on the gradient path. Additive only.
        self.output_sensitivities = None
        # Per-column normalization records (#453/#385): None until normalize() runs,
        # then {col_name: NormalizationRecord}. The gradient path reads it to thread the
        # normalizer's own derivative; absent -> byte-identical. Additive only.
        self.normalization = None
        self.bind_to(self.update_weights)
        if file_name is not None:
            self.load_data(file_name)
        elif arr is not None:
            self.data = arr
        elif named_arr is not None:
            # Initialize with RoadRunner named array
            # NamedArray is not pickleable, so we need to copy the contents into a regular array.
            self.data = np.array(named_arr)
            self.load_rr_header(named_arr.colnames)

    @classmethod
    def from_columns(cls, arr, headers, indvar=None):
        """
        Build a Data object from a 2-D array and an ordered list of column headers.

        Populates the cols (header->index) and headers (index->header) maps and
        the independent-variable name (defaults to the first header). Used by the
        simulator backends to assemble time-course and parameter-scan outputs
        without re-deriving the same cols/headers/indvar wiring at every site.

        :param arr: 2-D numpy array, one column per header
        :param headers: ordered list of column names; headers[0] is the indvar
        :param indvar: name of the independent variable (defaults to headers[0])
        :return: Data
        """
        obj = cls(arr=arr)
        obj.cols = {h: i for i, h in enumerate(headers)}
        obj.headers = {i: h for i, h in enumerate(headers)}
        obj.indvar = headers[0] if indvar is None else indvar
        return obj

    def rename_column(self, old, new):
        """Rename a data column header from ``old`` to ``new`` in place.

        Rewires both the header->index (``cols``) and index->header (``headers``)
        maps; the underlying data array is untouched (a column is the same numbers
        under a different name). Used by the new-era ``observable:`` override
        (ADR-0028) to remap a data-file column header to a model observable/function
        name, so the objective's by-name exp<->sim column match succeeds.

        Guards (each a clear ``PybnfError`` rather than a silent corruption):

        * ``old`` must be a present column (a missing header is almost always a typo);
        * ``new`` must not already name a *different* column (which would silently
          merge two columns / clobber existing data);
        * ``old`` must not be the independent variable (column 0) -- remapping the
          time / scanned-parameter axis is a mistake, not a rename.

        Renaming a column to its own name is a no-op.
        """
        if old == new:
            return
        if old not in self.cols:
            raise PybnfError(f"Cannot rename data column '{old}': there is no column with "
                             f"that name (columns are {sorted(self.cols)}).")
        if new in self.cols:
            raise PybnfError(f"Cannot rename data column '{old}' to '{new}': a column named "
                             f"'{new}' already exists.")
        if old == self.indvar:
            raise PybnfError(f"Cannot rename the independent-variable column '{old}'; "
                             "remapping the independent variable (the time or scanned-"
                             "parameter axis) is not allowed.")
        idx = self.cols.pop(old)
        self.cols[new] = idx
        self.headers[idx] = new

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        for callback in self._observers:
            callback(self._data)

    def bind_to(self, callback):
        self._observers.append(callback)

    def update_weights(self, data):
        self.weights = np.ones(data.shape)

    def _valid_indices(self):
        """Finds indices in Data.data that are valid for bootstrap sampling"""
        valid_indices = []
        for i in range(self.data.shape[0]):
            for j in range(1, self.data.shape[1]):
                if re.search('_SD$', self.headers[j]):
                    continue
                if np.isfinite(self.data[i, j]):
                    valid_indices.append((i, j))
        return valid_indices

    def gen_bootstrap_weights(self, rng):
        """
        Generates a integer weight for each point in the set of dependent variables.  Equivalent
        to sampling with replacement.  Weights are used when calculating the objective function
        for bootstrapped data.  Used for experimental data sets

        :param rng: the caller's np.random.Generator (the algorithm's root rng)
        :return:
        """
        indices = np.array(self._valid_indices())
        samples = indices[rng.choice(indices.shape[0], size=indices.shape[0], replace=True)]
        self.weights = np.zeros(self.data.shape)
        for s in samples:
            self.weights[s[0], s[1]] += 1

    def __getitem__(self, col_header):
        """
        Gets a column of data based on its column header
        
        :param col_header: Data column name
        :type col_header: str
        :return: Numpy array corresponding to name
        """
        idx = self.cols[col_header]
        return self.data[:, idx]

    def __setitem__(self, key, value):
        """
        Sets a column of data based on its column header
        :param key: Column name to modify
        :type key: str
        :param value: New column contents
        :type value: np.array
        """
        idx = self.cols[key]
        self.data[:, idx] = value

    def get_row(self, col_header, value):
        """
        Returns the (first) data row in which field col_header is equal to value.
        This should typically be used for col_header as the independent variable.

        :param col_header: Data column name
        :type col_header: str
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
        # float() natively parses 'nan', 'inf', '-inf' (any case), so no special
        # handling is needed. (The old '\b...' regex branches were dead: '\b' in
        # a non-raw string is a backspace char, not a regex word boundary, so they
        # never matched real data and everything already fell through to float().)
        return float(x)

    def load_data(self, file_name, sep=r'\s+'):
        """
        Loads column data from a text file

        :param file_name: Name of data file
        :type file_name: str
        :param sep: String that separates columns
        :type sep: str
        :return: None
        """
        with open(file_name, encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        self.data = self._read_file_lines(lines, sep, file_name=file_name)

    def load_rr_header(self, header):
        """
        Loads the header from a RoadRunner NamedArray
        :param header: The colnames attribute of a RoadRunner NamedArray (a list of str)
        """
        self.indvar = header[0].strip('[]')
        self.cols = {header[i].strip('[]'): i for i in range(len(header))}

    def _read_file_lines(self, lines, sep, file_name=''):
        """Helper function that reads lines from BNGL gdat files"""

        header = re.split(sep, lines[0].strip().strip('#').strip())
        # Ignore parentheses added to functions in BNG 2.3, and [] added to species names in COPASI
        header = [h.strip('()[]') for h in header]
        if header[0] == 'Time':
            header[0] = 'time'  # Allow either capitalization because Copasi uses capital, BNG uses lowercase
        ncols = len(header)
        self.indvar = header[0]

        self.cols = dict()
        self.headers = dict()
        for c in header:
            l = len(self.cols)
            if c in self.cols:
                raise DuplicateColumnError(f'Data file contains duplicate column name "{c}"')
            self.cols[c] = l
            self.headers[l] = c

        data = []
        for i, l in enumerate(lines[1:]):
            if re.match(r'^\s*$', l) or re.match(r'\s*#', l):
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
        with np.errstate(all='ignore'):  # Suppress divide by 0 warnings printed to terminal
            if cols == 'all':
                cols = list(range(self.data.shape[1]))
                cols.remove(idx)
            for c in cols:
                column = self.data[:, c]
                # Record N = peak and its row before the in-place divide overwrites them
                # (#453): the gradient threads d(raw/N)/d theta. Additive, value-preserving.
                self._record_normalization(c, NormalizationRecord(
                    'peak', float(np.max(column)), ref_row=int(np.argmax(column))))
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
        with np.errstate(all='ignore'):  # Suppress divide by 0 warnings printed to terminal
            if cols == 'all':
                cols = list(range(self.data.shape[1]))
                cols.remove(idx)
            for c in cols:
                # Record N = initial value before the in-place divide overwrites row 0
                # (#453): ref_row 0 is the divisor's source row for the gradient chain rule.
                self._record_normalization(c, NormalizationRecord(
                    'init', float(self.data[0, c]), ref_row=0))
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
        with np.errstate(all='ignore'):  # Suppress divide by 0 warnings printed to terminal
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
                # Record the z-score scale (std; 0 means the column was left un-divided) and
                # the K - ddof denominator the gradient's d std/d theta uses (#453). z-score
                # couples every row, so only these scalars are recorded -- the per-row mean of
                # the sensitivities is recomputed from the tensor at gradient time.
                self._record_normalization(c, NormalizationRecord('zero', float(std), ddof=ddof))

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
                # Degenerate branch: the baseline-subtracted column tops out at 0, so it is
                # scaled by |min| instead. The divisor N = |min| depends on raw, so its row's
                # sensitivity enters with a flipped sign (#453); the baseline is still row 0.
                self._record_normalization(c, NormalizationRecord(
                    'unit', float(np.abs(np.min(self.data[:, c]))),
                    ref_row=int(np.argmin(self.data[:, c])), baseline_row=0, sign=-1.0))
                self.data[:, c] = self.data[:, c] / np.abs(np.min(self.data[:, c]))
            else:
                # N = the max-after-baseline; ref_row is its argmax, baseline is row 0 (#453).
                self._record_normalization(c, NormalizationRecord(
                    'unit', float(cmax), ref_row=int(np.argmax(self.data[:, c])),
                    baseline_row=0, sign=1.0))
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
        # Carry over the column-index->header map and the independent-variable
        # name so the averaged Data stays as usable as its inputs (constraints
        # read .indvar, save() reads .headers).
        output.headers = datas[0].headers
        output.indvar = datas[0].indvar
        output.data = np.mean(np.stack([d.data for d in datas]), axis=0)
        return output

    def _record_normalization(self, col_index, record):
        """Stash how column ``col_index`` was normalized, keyed by its header name (#453).

        The gradient path reads :attr:`normalization` to thread the normalizer's own
        derivative through ``∂(raw/N)/∂θ``; every other path ignores it. Keyed by **name**
        (stable identifier) rather than index, mirroring how the gradient assembly addresses a
        scored column. Lazily creates the dict, so a never-normalized ``Data`` keeps
        ``normalization is None`` (byte-identical)."""
        if self.normalization is None:
            self.normalization = {}
        # Key by header name when known (the gradient looks up a scored column by name); fall
        # back to the index for a headerless Data (no gradient path reads it -- just no crash).
        self.normalization[self.headers.get(col_index, col_index)] = record

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
                raise ValueError(f'Invalid method {m} for Data.normalize()')

        if type(method) == str:
            normalize_once(method, 'all')
        else:
            for mi, cols_i in method:
                if type(cols_i[0]) == str:
                    # Convert to int indices
                    cols_i = [self.cols[c] for c in cols_i]
                normalize_once(mi, cols_i)

    def weights_to_file(self, file_name):
        logger.info(f"Saving weights in file {file_name}")
        np.savetxt(file_name, self.weights, fmt='%d', header='\t'.join(sorted(self.cols, key=self.cols.get)))


class DuplicateColumnError(ValueError):
    """
    Error thrown if a loaded data file has duplicate column names.
    Should be reraised as a PybnfError only if it was a user-supplied file
    """
    pass
