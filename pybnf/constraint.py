from .printing import PybnfError
import pyparsing as pp
import numpy as np

class ConstraintSet:
    """
    Represents the set of all constraints provided in one con file
    """
    def __init__(self):
        self.constraints = []

    def total_penalty(self, sim_data_dict):
        """
        Evaluate the sim_data_dict against all constraints, and return the total penalty

        :param sim_data_dict: Dictionary of the form {modelname: {suffix1: Data1}} containing the simulated data objects
        :return:
        """
        return sum([c.penalty(sim_data_dict) for c in self.constraints])

    def _load_constraint_file(self, filename):
        """
        Parse the constraint file filename and load them all into my constraint list
        """
        with open(filename) as f:
            for line in f:
                p = self.parse_constraint_line(line)

    def parse_constraint_line(self, line):
        obs = pp.Word(pp.alphas, pp.alphanums+'_.')
        point = pp.Literal(".")
        e = pp.CaselessLiteral("E")
        const = pp.Combine(pp.Word("+-" + pp.nums, pp.nums) +
                         pp.Optional(point + pp.Optional(pp.Word(pp.nums))) +
                         pp.Optional(e + pp.Word("+-" + pp.nums, pp.nums)))
        iop = pp.oneOf("< <= > >=")
        ineq0 = obs - iop - (obs ^ const)
        ineq1 = const - iop - obs
        ineq = ineq0 ^ ineq1
        equals = pp.Literal('=')
        obs_crit = obs - equals - const
        enforce_crit = const ^ obs_crit
        enforce_at = pp.CaselessLiteral('at') - enforce_crit - pp.Optional(pp.oneOf('everytime first', caseless=True))
        enforce_between = enforce_crit - pp.Suppress(',') - enforce_crit
        enforce_other = pp.oneOf('once always', caseless=True)
        enforce = enforce_at ^ enforce_between ^ enforce_other
        min = pp.CaselessLiteral('min') - const
        penalty = pp.CaselessLiteral('altpenalty') - const
        wt_expr = const - pp.Optional(penalty) - pp.Optional(min)
        weight = pp.CaselessLiteral('weight') - wt_expr
        constraint = ineq - enforce - pp.Optional(weight)

        return constraint.parseString(line, parseAll=True)





class Constraint:
    """
    Abstract class representing an optimization constraint with a penalty for violating the constraint
    """

    def __init__(self, quant1, sign, quant2, base_suffix):
        """
        Create a constraint of the form (quant1) (sign) (quant2)
        :param quant1: String observable name or float. String could be in the form suffix.Observable to refer to any
        observable in the fitting run.
        :param sign: One of '<', '<=', '>', '>='
        :param quant2: String observable name or float
        :param base_suffix: Suffix to assume for observables that dont use the 'suffix.Observable' notation
        """

        self.quant1 = quant1
        self.sign = sign
        self.quant2 = quant2

    def penalty(self, sim_data_dict):
        """
        penalty function for violating the constraint. Returns 0 if constraint is satisfied, or a positive value
        if the constraint is volated. Implementation depends on the type of constraint.

        :param sim_data_dict: Dictionary of the form {modelname: {suffix1: Data1}} containing the simulated data objects
        :type sim_data_dict: dict
        """

        raise NotImplementedError('Subclasses of Constraint must override penalty()')

    def _parse_inequality(self, ineq):
        """
        Set this constraint's inequality according to the specified inequality string
        Updates the target_var, target_val, and sign fields.

        :param ineq: String representing the inequality
        :return:
        """
        sign = None
        for asign in ['<=', '>=', '<', '>']:
            if asign in ineq:
                sign = asign
                break
        if sign is None:
            raise ParseError('Unable to parse inequality ' + ineq)
        parts = ineq.split(sign)
        if len(parts) != 2:
            raise ParseError('Unable to parse inequality ' + ineq)
        self.target_var = parts[0]
        try:
            self.target_val = float(parts[1])
        except ValueError:
            # Alternate option to give a variable name here.
            self.target_val = parts[1]
        self.sign = sign

    def str_info(self, penalty=None):
        return (' ({:.3g}) '.format(penalty)) + self.target_var + self.sign + float(self.target_val)


class AtConstraint(Constraint):
    def __init__(self, inequality, when_condition, never_penalty=1e5):
        """
        Creates a new constraint of the form

        X1>value when X2=value

        :param inequality: String specifying the inequality, eg 'X1>4'
        :param when_condition: String specifying the when condition eg 'X2=3'
        :param never_penalty: The large penalty for never satisfying the when condition within the data
        """

        super().__init__()
        self.never_penalty = never_penalty
        self._parse_inequality(inequality)

        parts = when_condition.split('=')
        if len(parts) != 2:
            raise ParseError("Unable to parse 'when' condition " + when_condition)
        self.when_var = parts[0]
        self.when_val = float(parts[1])
        self.weight = 1.0

    def penalty(self, data):
        """
        Compute the when penalty.
        If the when condition is never met, the penalty is the large never_penalty times the closest approach to the
        when condition.

        In the normal case, the penalty is 0 if the constraint is met, or the distance from the constraint if it
        is failed, at the index where the when condition is first met.

        Edge cases where the the when condition is hit exactly at a particular index are not currently implemented
        correctly

        Now supports declaring a weight of the constraint by setting the .weight attribute. This is a constant
        multiplier on the penalty function

        :param data:
        :type data: TysonData
        :return: float penalty falue
        """

        # Find the index where the when condition is met
        when_col = data[self.when_var] < self.when_val
        first_true = find_first(True, when_col)
        first_false = find_first(False, when_col)

        if first_true == -1:
            # When_var is never lower than when_val
            return self.never_penalty * np.min(data[self.when_var] - self.when_val)

        if first_false == -1:
            # When_var is never higher than when_val
            return self.never_penalty * np.min(self.when_val - data[self.when_var])

        keyindex = max(first_false, first_true)

        # Check the appropriate index
        diff = data[self.target_var][keyindex] - self.target_val  # positive if data > target

        if self.sign == '>' or self.sign == '>=':
            diff = -diff

        return max(0., diff) * self.weight

    def str_info(self, penalty=None):
        return super().str_info(penalty) + ' when ' + self.when_var + '=' + str(self.when_val)


class BetweenConstraint(Constraint):
    def __init__(self, inequality, when_condition):
        """
        Creates a new constraint of the form

        X1>value until X2=value

        :param inequality: String specifying the inequality, eg 'X1>4'
        :param when_condition: String specifying the when condition eg 'X2=3'
        """

        super().__init__()
        self._parse_inequality(inequality)

        parts = when_condition.split('=')
        if len(parts) != 2:
            raise ParseError("Unable to parse 'until' condition " + when_condition)
        self.when_var = parts[0]
        self.when_val = float(parts[1])
        self.weight = 1.0

    def penalty(self, data):
        """
        Compute the until penalty.

        The until constraint indicates that the constraint must be satisfied at every time point until the condition
        is satisfied.
        If the until condition is never met, there's no penalty for that, but then the constraint must be satisfied
        for the entire simulation.

        The penalty is given by the worst violation of the constraint within the time range that the constraint is active.

        Edge cases where the the when condition is hit exactly at a particular index are not currently implemented
        correctly

        Now supports declaring a weight of the constraint by setting the .weight attribute. This is a constant
        multiplier on the penalty function

        :param data:
        :type data: TysonData
        :return: float penalty falue
        """

        # Find the index where the when condition is met
        when_col = data[self.when_var] < self.when_val
        first_true = find_first(True, when_col)
        first_false = find_first(False, when_col)

        if first_true == -1 or first_false == -1:
            keyindex = len(data[self.when_var])
        else:
            keyindex = max(first_false, first_true)

        # Check the appropriate indices (everything up to keyindex). Like an AlwaysConstraint for just this range.
        diffs = data[self.target_var][:keyindex] - self.target_val  # positive if data > target

        if self.sign == '>' or self.sign == '>=':
            diffs = -diffs

        worstdiff = np.max(diffs)

        return max(0., worstdiff) * self.weight

    def str_info(self, penalty=None):
        return super().str_info(penalty) + ' until ' + self.when_var + '=' + str(self.when_val)


class AlwaysConstraint(Constraint):
    def __init__(self, inequality):
        super().__init__()
        self._parse_inequality(inequality)
        self.weight = 1.0

    def penalty(self, data):
        """
        Compute the always penalty
        The penalty is given by the worst miss of the constraint over the entire data column


        :param data: TysonData object
        :return:
        """

        diffs = data[self.target_var] - self.target_val
        if self.sign == '>' or self.sign == '>=':
            diffs = -diffs

        worstdiff = np.max(diffs)

        return max(0., worstdiff) * self.weight

    def str_info(self, penalty=None):
        return super().str_info(penalty) + ' always'



class OnceConstraint(Constraint):
    def __init__(self, inequality):
        super().__init__()
        self._parse_inequality(inequality)
        self.weight = 1.0

    def penalty(self, data):
        """
        Compute the notalways penalty
        The penalty is given by the smallest miss of the opposite of the constraint, over the entire data column

        :param data: TysonData object
        :return: float
        """

        diffs = data[self.target_var] - self.target_val

        if self.sign == '<' or self.sign == '<=':
            diffs = -diffs

        bestdiff = np.min(diffs)

        return max(0., bestdiff) * self.weight

    def str_info(self, penalty=None):
        return super().str_info(penalty) + ' notalways'

class ParseError(Exception):
    pass

def find_first(x, y):
    raise NotImplementedError('This was some function that Im not planning to transfer over to PyBNF')