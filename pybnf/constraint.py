from .printing import PybnfError
import pyparsing as pp
import numpy as np
import re

class ConstraintSet:
    """
    Represents the set of all constraints provided in one con file
    """
    def __init__(self):
        self.constraints = []
        self.base_suffix = 'idk'  # Todo

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
            linenum = 0
            for line in f:
                linenum += 1
                if re.match(r'\s*(#|$)', line):  # Blank or comment
                    continue
                # Parse the line
                try:
                    p = self.parse_constraint_line(line)
                except pp.ParseBaseException:
                    raise PybnfError("Unable to parse constraint '%s' at line %i of %s" % (line, linenum, filename))
                # Convert all the attributes of the parsed line into args to use to create the constraint
                try:
                    quant1 = float(p.ineq[0])
                except ValueError:
                    quant1 = p.ineq[0]
                sign = p.ineq[1]
                try:
                    quant2 = float(p.ineq[2])
                except ValueError:
                    quant2 = p.ineq[2]
                if p.weight_expr:
                    weight = float(p.weight_expr.weight)
                    altpenalty = p.weight_expr.altpenalty if p.weight_expr.altpenalty else None
                    minpenalty = float(p.weight_expr.min) if p.weight_expr.min else 0.
                else:
                    weight = 1.
                    altpenalty = None
                    minpenalty = 0.

                # Check the constraint type based on the parse object, extract the constraint-type-specific args, and
                # make the constraint
                if p.enforce[0] == 'at':
                    if len(p.enforce[1]) == 1:
                        atval = float(p.enforce[1][0])
                        atvar = None
                    else:
                        atvar = p.enforce[1][0]
                        atval = float(p.enforce[1][1])
                    repeat = (len(p.enforce) == 3 and p.enforce[2] == 'everytime')
                    con = AtConstraint(quant1, sign, quant2, self.base_suffix, weight, altpenalty=altpenalty,
                                           minpenalty=minpenalty, atvar=atvar, atval=atval, repeat=repeat)
                elif p.enforce[0] == 'always':
                    con = AlwaysConstraint(quant1, sign, quant2, self.base_suffix, weight, altpenalty=altpenalty,
                                           minpenalty=minpenalty)
                elif p.enforce[0] == 'once':
                    con = OnceConstraint(quant1, sign, quant2, self.base_suffix, weight, altpenalty, minpenalty)
                elif p.enforce[0] == 'between':
                    if len(p.enforce[1]) == 1:
                        startval = float(p.enforce[1][0])
                        startvar = None
                    else:
                        startvar = p.enforce[1][0]
                        startval = float(p.enforce[1][1])
                    if len(p.enforce[2]) == 1:
                        endval = float(p.enforce[2][0])
                        endvar = None
                    else:
                        endvar = p.enforce[2][0]
                        endval = float(p.enforce[2][1])
                    con = BetweenConstraint(quant1, sign, quant2, self.base_suffix, weight, altpenalty=altpenalty,
                                           minpenalty=minpenalty, startvar=startvar, startval=startval, endvar=endvar,
                                           endval=endval)
                else:
                    raise RuntimeError('Unknown enforcement keyword %s' % p.enforce[0])
                self.constraints.append(con)


    def parse_constraint_line(self, line):
        obs = pp.Word(pp.alphas, pp.alphanums+'_.')
        point = pp.Literal(".")
        e = pp.CaselessLiteral("E")
        const = pp.Combine(pp.Word("+-" + pp.nums, pp.nums) +
                         pp.Optional(point + pp.Optional(pp.Word(pp.nums))) +
                         pp.Optional(e + pp.Word("+-" + pp.nums, pp.nums)))
        iop = pp.oneOf("< <= > >=")
        ineq0 = obs + iop + (obs ^ const)
        ineq1 = const + iop + obs
        ineq = ineq0 ^ ineq1
        equals = pp.Suppress('=')
        obs_crit = obs - equals - const
        enforce_crit = const | obs_crit
        enforce_at = pp.CaselessLiteral('at') - pp.Group(enforce_crit) - pp.Optional(pp.oneOf('everytime first', caseless=True))
        enforce_between = pp.CaselessLiteral('between') - pp.Group(enforce_crit) - pp.Suppress(',') - pp.Group(enforce_crit)
        enforce_other = pp.oneOf('once always', caseless=True)
        enforce = enforce_at ^ enforce_between ^ enforce_other
        min = pp.CaselessLiteral('min') - const.setResultsName('min')
        penalty = pp.CaselessLiteral('altpenalty') - pp.Group(ineq).setResultsName('altpenalty')
        wt_expr = const.setResultsName('weight') - pp.Optional(penalty) - pp.Optional(min)
        weight = pp.CaselessLiteral('weight') - wt_expr
        comment = pp.Suppress(pp.Literal('#') - pp.ZeroOrMore(pp.Word(pp.printables)))
        constraint = pp.Group(ineq).setResultsName('ineq') + pp.Group(enforce).setResultsName('enforce') + \
                     pp.Optional(pp.Group(weight).setResultsName('weight_expr')) + pp.Optional(comment)

        return constraint.parseString(line, parseAll=True)


class Constraint:
    """
    Abstract class representing an optimization constraint with a penalty for violating the constraint
    """

    def __init__(self, quant1, sign, quant2, base_suffix, weight, altpenalty=None, minpenalty=0.):
        """
        Create a constraint of the form (quant1) (sign) (quant2)
        :param quant1: String observable name or float. String could be in the form suffix.Observable to refer to any
        observable in the fitting run.
        :param sign: One of '<', '<=', '>', '>='
        :param quant2: String observable name or float
        :param base_suffix: Suffix to assume for observables that dont use the 'suffix.Observable' notation
        :param weight: Weight of the constraint, to multiply the extent of violation
        :param altpenalty: a 3-tuple [quantity, sign, quantity], giving an inequality to use instead of the regular
        constraint inequality to calculate the penalty
        :param minpenalty: The minimum penalty that must be applied if the constraint is violated, regardless of how
        low the extent of violation is.
        """
        # Flip the inequality if it's a '>', so we can always assume a '<'
        if sign in ('>', '>='):
            self.quant1 = quant2
            self.quant2 = quant1
        else:
            self.quant1 = quant1
            self.quant2 = quant2
        if '=' in sign:
            self.or_equal = True
        else:
            self.or_equal = False

        self.base_suffix = base_suffix
        self.weight = weight
        self.min_penalty = minpenalty

        if altpenalty:
            # Also force the altpenalty to be a '<'.
            # Never matters whether this is an 'or equal' or not.
            alt1, altsign, alt2 = altpenalty
            if altsign in ('>', '>='):
                self.alt1 = alt2
                self.alt2 = alt1
            else:
                self.alt1 = alt1
                self.alt2 = alt2
        else:
            self.alt1 = None
            self.alt2 = None

    def penalty(self, sim_data_dict):
        """
        penalty function for violating the constraint. Returns 0 if constraint is satisfied, or a positive value
        if the constraint is violated. Implementation depends on the type of constraint.

        :param sim_data_dict: Dictionary of the form {modelname: {suffix1: Data1}} containing the simulated data objects
        :type sim_data_dict: dict
        """

        raise NotImplementedError('Subclasses of Constraint must override penalty()')


class AtConstraint(Constraint):
    def __init__(self, quant1, sign, quant2, base_suffix, weight, atvar, atval, altpenalty=None, minpenalty=0.,
                 repeat=False):
        """
        Creates a new constraint of the form

        X1>value at X2=value

        :param atvar: Variable checked to determine the 'at' condition, or None for the independent variable
        :param atval: Value that the atvar must take to activate the 'at' condition
        :param repeat: If True, enforce the constraint every time the 'at' condition is met. If False, only enforce
        the first time
        """

        super().__init__(quant1, sign, quant2, base_suffix, weight, altpenalty, minpenalty)
        self.atvar = atvar
        self.atval = atval
        self.repeat = repeat

    def penalty(self, data):
        """
        Compute the penalty
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


class BetweenConstraint(Constraint):
    def __init__(self, quant1, sign, quant2, base_suffix, weight, startvar, startval, endvar, endval, altpenalty=None,
                 minpenalty=0.):
        """
        Creates a new constraint of the form

        X1 < X2 between X3=value  X4=value

        :param startvar: Variable checked to determine the start of the interval, or None for the independent variable
        :param startval: Value that the startvar must take to trigger the start of the interval
        :param endvar: Variable checked to determine the end of the interval, or None for the independent variable
        :param endval: Value that the endvar must take to trigger the start of the interval
        """

        super().__init__(quant1, sign, quant2, base_suffix, weight, altpenalty, minpenalty)

        self.startvar = startvar
        self.startval = startval
        self.endvar = endvar
        self.endval = endval

    def penalty(self, data):
        """
        Compute the penalty
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


class AlwaysConstraint(Constraint):
    def __init__(self, quant1, sign, quant2, base_suffix, weight, altpenalty=None, minpenalty=0.):
        """
        Creates a new constraint of the form

        X1>X2 always
        """

        super().__init__(quant1, sign, quant2, base_suffix, weight, altpenalty, minpenalty)


    # def __init__(self, inequality):
    #     super().__init__()
    #     self._parse_inequality(inequality)
    #     self.weight = 1.0

    def penalty(self, data):
        """
        Compute the always penalty
        The penalty is given by the worst miss of the constraint over the entire data column
        """

        diffs = data[self.target_var] - self.target_val
        if self.sign == '>' or self.sign == '>=':
            diffs = -diffs

        worstdiff = np.max(diffs)

        return max(0., worstdiff) * self.weight

    def str_info(self, penalty=None):
        return super().str_info(penalty) + ' always'


class OnceConstraint(Constraint):
    def __init__(self, quant1, sign, quant2, base_suffix, weight, altpenalty=None, minpenalty=0.):
        """
        Creates a new constraint of the form

        X1>X2 once
        """

        super().__init__(quant1, sign, quant2, base_suffix, weight, altpenalty, minpenalty)

    def penalty(self, data):
        """
        Compute the penalty
        """

        diffs = data[self.target_var] - self.target_val

        if self.sign == '<' or self.sign == '<=':
            diffs = -diffs

        bestdiff = np.min(diffs)

        return max(0., bestdiff) * self.weight

class ParseError(Exception):
    pass

def find_first(x, y):
    raise NotImplementedError('This was some function that Im not planning to transfer over to PyBNF')