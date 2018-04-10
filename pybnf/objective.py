"""pybnf.objective: defines objective functions used for evaluating points in parameter space"""


from .printing import PybnfError, print1

import numpy as np


class ObjectiveFunction(object):
    """
    Abstract class representing an objective function
    Subclasses customize how the objective value is calculated from the quantitative exp data
    The base class includes all the support we need for constraints.
    """

    def evaluate_multiple(self, sim_data_dict, exp_data_dict, constraints=()):
        """
        Compute the value of the objective function on several data sets, and return the total.
        Optionally may pass an iterable of ConstraintSets whose penalties will be added to the total

        :param sim_data_dict: Dictionary of the form {modelname: {suffix1: Data1}} containing the simulated data objects
        :type sim_data_dict: dict
        :param exp_data_dict: Dictionary mapping suffix strings to experimental Data objects
        :type exp_data_dict: dict
        :param constraints: Iterable of ConstraintSet objects containing the constraints that we should evaluate using
        the simulated data
        :type constraints: Iterable of ConstraintSet
        :return:
        """
        total = 0.

        # Dictionary mapping suffix strings to experimental Data objects
        # exp_data_dict = self.conf.exp_data

        if not sim_data_dict:
            return np.inf
        else:
            for model in sim_data_dict:
                for suffix in sim_data_dict[model]:
                    # Suffixes might exist in sim_data_dict that do not have experimental data.
                    # Need to check for that here.
                    if suffix in exp_data_dict:
                        val = self.evaluate(sim_data_dict[model][suffix], exp_data_dict[suffix])
                        if val is None:
                            return None
                        total += val
            for cset in constraints:
                total += cset.total_penalty(sim_data_dict)

            return total

    def evaluate(self, sim_data, exp_data):
        """
        :param sim_data: A Data object containing simulated data
        :type sim_data: Data
        :param exp_data: A Data object containing experimental data
        :type exp_data: Data
        :return: float, value of the objective function, with a lower value indicating a better fit.
        """
        raise NotImplementedError("Subclasses must override evaluate()")


class SummationObjective(ObjectiveFunction):
    """
    Represents a type of objective function in which we perform some kind of summation over all available experimental
    data points.
    Currently, this describes all objective functions in PyBNF.
    """

    def __init__(self):
        # Keep track of which warnings we've printed, so we only print each one once.
        self.warned = set()

    def evaluate(self, sim_data, exp_data):
        """
        :param sim_data: A Data object containing simulated data
        :type sim_data: Data
        :param exp_data: A Data object containing experimental data
        :type exp_data: Data
        :return: float, value of the objective function, with a lower value indicating a better fit.
        """

        indvar = min(exp_data.cols, key=exp_data.cols.get)  # Get the name of column 0, the independent variable

        compare_cols = set(exp_data.cols).intersection(set(sim_data.cols))  # Set of columns to compare
        # Warn if experiment columns are going unused
        self._check_columns(exp_data.cols, compare_cols)
        try:
            compare_cols.remove(indvar)
        except KeyError:
            raise PybnfError('The independent variable "%s" in your exp file was not found in the simulation data.'
                             % indvar)

        func_value = 0.0
        # Iterate through rows of experimental data
        for rownum in range(exp_data.data.shape[0]):

            # Figure out the corresponding row number in the simulation data
            # Find the row number of sim_data column 0 that is almost equal to exp_data[rownum, 0]
            sim_row = np.argmax(np.isclose(sim_data.data[:, 0], exp_data.data[rownum, 0], atol=0.))
            # If no such column existed, sim_row will come out as 0; need to check for this and skip if it happened
            if sim_row == 0 and not np.isclose(sim_data.data[0, 0], exp_data.data[rownum, 0], atol=0.):
                warnstr = indvar + str(exp_data.data[rownum, 0])  # An identifier so we only print the warning once
                if warnstr not in self.warned:
                    print1("Warning: Ignored " + indvar + " " + str(exp_data.data[rownum, 0]) +
                           " because that " + indvar + " was not in the simulation data.")
                    self.warned.add(warnstr)
                continue

            for col_name in compare_cols:
                if np.isnan(exp_data.data[rownum, exp_data.cols[col_name]]):
                    continue

                cur_sim_val = sim_data.data[sim_row, sim_data.cols[col_name]]

                if np.isnan(cur_sim_val) or np.isinf(cur_sim_val):
                    return None
                func_value += self.eval_point(sim_data, exp_data, sim_row, rownum, col_name)

        return func_value

    def eval_point(self, sim_data, exp_data, sim_row, exp_row, col_name):
        """
        Calculate the objective function for a single point in the data

        This evaluation is what differentiates the different objective functions.

        :param sim_data: The simulation Data object
        :param exp_data: The experimental Data object
        :param sim_row: The row number to look at in sim_data
        :param exp_row: The row number to look at in exp_data
        :param col_name: The column name to look at  (same for the sim_data and the exp_data)
        :return:
        """
        raise NotImplementedError('Subclasses of SummationObjective must override eval_point')

    def _check_columns(self, exp_cols, compare_cols):
        """
        Check that all exp_cols are being read in compare_cols; give a warning if not.
        :param exp_cols: Iterable of all experimental data column names
        :param compare_cols: Iterable of the names being used
        :return: None
        """
        missed = set(exp_cols).difference(set(compare_cols))
        for name in missed:
            if name not in self.warned:
                print1("Warning: Ignoring experimental data column '%s' because that name is not in the simulation "
                       "output" % name)
                self.warned.add(name)


class ChiSquareObjective(SummationObjective):

    def eval_point(self, sim_data, exp_data, sim_row, exp_row, col_name):
        sim_val = sim_data.data[sim_row, sim_data.cols[col_name]]
        exp_val = exp_data.data[exp_row, exp_data.cols[col_name]]
        try:
            # Todo: Check for this and throw the error before all the workers get created.
            sd_col = exp_data.cols[col_name + '_SD']
        except KeyError:
            raise PybnfError('Column %s_SD not found' % col_name,
                 "Column %s_SD was not found in the experimental data. When using the chi_sq objective function, your "
                 "data file must include a _SD column corresponding to each experimental variable, giving the standard "
                 "deviations of that variable. " % col_name)
        exp_sigma = exp_data.data[exp_row, sd_col]
        return 1. / (2. * exp_sigma ** 2.) * (sim_val - exp_val) ** 2.

    def _check_columns(self, exp_cols, compare_cols):
        """
        Check that all exp_cols are being read in compare_cols; give a warning if not.
        :param exp_cols: Iterable of all experimental data column names
        :param compare_cols: Iterable of the names being used
        :return: None
        """
        missed = set(exp_cols).difference(set(compare_cols).union(set(['%s_SD' % s for s in compare_cols])))
        for name in missed:
            if name not in self.warned:
                print1("Warning: Ignoring experimental data column '%s' because that name is not in the simulation "
                       "output" % name)
                self.warned.add(name)


class SumOfSquaresObjective(SummationObjective):

    def eval_point(self, sim_data, exp_data, sim_row, exp_row, col_name):

        sim_val = sim_data.data[sim_row, sim_data.cols[col_name]]
        exp_val = exp_data.data[exp_row, exp_data.cols[col_name]]
        return (sim_val - exp_val) ** 2.


class NormSumOfSquaresObjective(SummationObjective):
    """
    Sum of squares where each point is normalized by the y value at that point, ((y-y')/y)^2
    """
    def eval_point(self, sim_data, exp_data, sim_row, exp_row, col_name):

        sim_val = sim_data.data[sim_row, sim_data.cols[col_name]]
        exp_val = exp_data.data[exp_row, exp_data.cols[col_name]]
        return ((sim_val - exp_val)/exp_val) ** 2.


class AveNormSumOfSquaresObjective(SummationObjective):
    """
    Sum of squares where each point is normalized by the average value of that variable,
    ((y-y')/ybar)^2
    """
    def evaluate(self, sim_data, exp_data):
        # Precalculate the average of each exp column to use for all points in this call.
        self.aves = {name: np.average(exp_data[name]) for name in exp_data.cols}
        return super().evaluate(sim_data, exp_data)

    def eval_point(self, sim_data, exp_data, sim_row, exp_row, col_name):
        sim_val = sim_data.data[sim_row, sim_data.cols[col_name]]
        exp_val = exp_data.data[exp_row, exp_data.cols[col_name]]
        return ((sim_val - exp_val) / self.aves[col_name]) ** 2.
