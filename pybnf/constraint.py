"""
Classes for defining various constraints that can be applied to the fitting run.  Used when incorporating qualitative
data into the fit
"""


from .printing import PybnfError
import pyparsing as pp
import numpy as np
import re
import logging


logger = logging.getLogger(__name__)


class ConstraintSet:
    """
    Represents the set of all constraints provided in one prop file
    """
    def __init__(self, base_model, base_suffix):
        self.constraints = []
        self.base_suffix = base_suffix
        self.base_model = base_model

    def total_penalty(self, sim_data_dict):
        """
        Evaluate the sim_data_dict against all constraints, and return the total penalty

        :param sim_data_dict: Dictionary of the form {modelname: {suffix1: Data1}} containing the simulated data objects
        :return:
        """
        return sum([c.penalty(sim_data_dict) for c in self.constraints])

    def number_failed(self, sim_data_dict):
        """
        Return the number of satisfied constraints

        :param sim_data_dict:
        :return:
        """
        return sum([0 if c.penalty(sim_data_dict) == 0 else 1 for c in self.constraints])

    def output_itemized_eval(self, sim_data_dict, output_dir):
        """
        Output a text file in which we evaluate the given simulation data: each line of the text file contains
        the penalty associated with the corresponding line in the constraint file.

        :param sim_data_dict:
        :param output_dir: Directory where the file should be saved
        :return:
        """
        with open('%s/%s_constraint_eval.txt' % (output_dir, self.base_suffix), 'w') as out:
            for c in self.constraints:
                out.write('%s\n' % c.penalty(sim_data_dict))

    def load_constraint_file(self, filename, scale=1.0):
        """
        Parse the constraint file filename and load them all into my constraint list
        :param filename: Path of constraint file
        :param scale: Factor by which we multiply all constraint weights
        """
        logger.info('Loading constraints for %s suffix %s from %s' % (self.base_model, self.base_suffix, filename))
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
                    if p.weight_expr.altpenalty:
                        altpenalty = p.weight_expr.altpenalty
                        for i in (0, 2):
                            try:
                                altpenalty[i] = float(altpenalty[i])
                            except ValueError:
                                pass  # If float conversion fails, it's a variable name, so leave it
                    else:
                        altpenalty = None
                    minpenalty = float(p.weight_expr.min) if p.weight_expr.min else 0.
                else:
                    weight = 1.
                    altpenalty = None
                    minpenalty = 0.

                weight *= scale  # Scale the weight by the specified factor

                # Check the constraint type based on the parse object, extract the constraint-type-specific args, and
                # make the constraint
                if p.enforce[0] == 'at':
                    if len(p.enforce[1]) == 1:
                        atval = float(p.enforce[1][0])
                        atvar = None
                    else:
                        atvar = p.enforce[1][0]
                        atval = float(p.enforce[1][1])
                    repeat = (len(p.enforce) >= 3 and p.enforce[2] == 'everytime')
                    before = (len(p.enforce) >= 3 and p.enforce[-1] == 'before')
                    con = AtConstraint(quant1, sign, quant2, self.base_model, self.base_suffix, weight, altpenalty=altpenalty,
                                           minpenalty=minpenalty, atvar=atvar, atval=atval, repeat=repeat, before=before)
                elif p.enforce[0] == 'always':
                    con = AlwaysConstraint(quant1, sign, quant2, self.base_model, self.base_suffix, weight, altpenalty=altpenalty,
                                           minpenalty=minpenalty)
                elif p.enforce[0] == 'once':
                    con = OnceConstraint(quant1, sign, quant2, self.base_model, self.base_suffix, weight, altpenalty, minpenalty)
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
                    con = BetweenConstraint(quant1, sign, quant2, self.base_model, self.base_suffix, weight, altpenalty=altpenalty,
                                           minpenalty=minpenalty, startvar=startvar, startval=startval, endvar=endvar,
                                           endval=endval)
                else:
                    raise RuntimeError('Unknown enforcement keyword %s' % p.enforce[0])
                self.constraints.append(con)
        logger.info('Loaded %i constraints' % len(self.constraints))

    @staticmethod
    def parse_constraint_line(line):
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
        enforce_at = pp.CaselessLiteral('at') - pp.Group(enforce_crit) - pp.Optional(pp.oneOf('everytime first', caseless=True)) -\
            pp.Optional(pp.CaselessLiteral('before'))
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

    def __init__(self, quant1, sign, quant2, base_model, base_suffix, weight, altpenalty=None, minpenalty=0.):
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

        self.base_model = base_model
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

        # Key tuples need to be initialized after we receive the first data result, so we can figure out the keys.
        self.qkeys1 = None
        self.qkeys2 = None
        self.akeys1 = None
        self.akeys2 = None

    def find_keys(self, sim_data_dict):
        """
        Function to be called on the first evaluation of the penalty.
        Read through the data dictionary and figure out what keys I need to use to access each relevant variable.

        This needs to be done kind of by brute force, because we were never told the model file to use, and it's even
        possible the input is poorly defined by referring to a suffix that exists in multiple models.
        :param sim_data_dict:
        :return:
        """
        keylist = []
        for q in (self.quant1, self.quant2, self.alt1, self.alt2):
            key = self.get_key(q, sim_data_dict)
            keylist.append(key)
        logger.debug("Got keys for constraint %s<%s: %s" % (self.quant1, self.quant2, keylist))
        self.qkeys1, self.qkeys2, self.akeys1, self.akeys2 = keylist

    def get_key(self, q, sim_data_dict):
        """
        Converts a string from a constraint file ('Observable' or 'suffix.observable') into a tuple of the 3 keys you
        would need to index into the sim_data_dict to get the right column of data.

        :param q:
        :return:
        """
        if not isinstance(q, str):
            return None
        if '.' not in q:
            if self.base_model not in sim_data_dict:
                raise PybnfError('Constraint file is associated with model %s, but that model was not found' %
                                 self.base_model)
            if self.base_suffix not in sim_data_dict[self.base_model]:
                raise PybnfError("Constraint file %s.prop refers to suffix '%s' in model '%s', but that model does not "
                                 "produce output with that suffix" %
                                 (self.base_suffix, self.base_suffix, self.base_model))
            if q not in sim_data_dict[self.base_model][self.base_suffix].cols:
                raise PybnfError("Observable '%s' specified in constraint file was not found in the output of model "
                                 "'%s', suffix '%s'" % (q, self.base_model, self.base_suffix))
            return (self.base_model, self.base_suffix, q)
        else:
            parts = q.split('.')
            if len(parts) > 2:
                raise PybnfError('Parsing constraint variable %s' % q,
                                 "Parsing constraint variable %s - The format should be 'suffix.Observable' Should "
                                 "not contain multiple '.' characters.")
            options = []
            for m in sim_data_dict:
                if parts[0] in sim_data_dict[m]:
                    options.append(m)
            if len(options) == 0:
                raise PybnfError("Could not find any experimental data corresponding to the suffix '%s' specified "
                                 "in your constraint file" % parts[0])
            if len(options) > 1:
                raise PybnfError("Suffix '%s' in your constraint file appears in multiple model files. Please "
                                 "change your suffix names so it isn't ambiguous" % parts[0])
            if parts[1] not in sim_data_dict[options[0]][parts[0]].cols:
                raise PybnfError("Obserable '%s' in constraint file - The output with suffix %s does not contain an "
                                 "observable %s" % (q, parts[0], parts[1]))
            return (options[0], parts[0], parts[1])

    def index(self, sim_data_dict, keys):
        """
        Shortcut function for applying all 3 indices to the data object
        :return:
        """
        return sim_data_dict[keys[0]][keys[1]][keys[2]]

    def get_penalty(self, sim_data_dict, imin, imax, once=False, require_length=None):
        """
        Helper function for calculating the penalty, that can be called from the subclasses.
        Enforces the constraint for the entire interval unless the once option is set.

        :param sim_data_dict: The dictionary of data objects
        :param imin: First index at which to check the constraint
        :param imax: Last index at which to check the constraint (exclusive)
        :param once: If true, enforce that the constraint holds once at some point during the time interval
        :param require_length: If set to an integer, raise an error if the length of the selected data column(s) is not
        equal to that value. (Used to check that "at" and "between" constraints are not encountering an unsupported
        case)
        :return:
        """

        def length_error():
            raise PybnfError('Applying constraint %s<%s. Constraints involving observables that have different time '
                             'points in their simulation outputs are not currently supported' %
                             (self.quant1, self.quant2))

        if isinstance(self.quant1, str):
            q1_col = self.index(sim_data_dict, self.qkeys1)
            if require_length and len(q1_col) != require_length:
                length_error()
            q1 = q1_col[imin:imax]
        else:
            q1 = self.quant1
        if isinstance(self.quant2, str):
            q2_col = self.index(sim_data_dict, self.qkeys2)
            if require_length and len(q2_col) != require_length:
                length_error()
            q2 = q2_col[imin:imax]
        else:
            q2 = self.quant2

        try:
            if once:
                penalty = np.min(q1-q2)
            else:
                penalty = np.max(q1 - q2)
        except ValueError:
            length_error()

        if penalty > 0 or (penalty == 0. and not self.or_equal):
            # Failed constraint
            if self.alt1:
                # Replace the penalty with the alt penalty
                if isinstance(self.alt1, str):
                    a1 = self.index(sim_data_dict, self.akeys1)[imin:imax]
                else:
                    a1 = self.alt1
                if isinstance(self.alt2, str):
                    a2 = self.index(sim_data_dict, self.akeys2)[imin:imax]
                else:
                    a2 = self.alt2

                if once:
                    penalty = np.min(a1 - a2)
                else:
                    penalty = np.max(a1 - a2)
                penalty = max(0., penalty)
            # Apply the minimum penalty
            penalty = max(self.min_penalty, penalty)

        # If constraint satisfied or penalty is negative for any other reason, return 0.
        penalty = max(0., penalty)

        return penalty * self.weight

    def penalty(self, sim_data_dict):
        """
        penalty function for violating the constraint. Returns 0 if constraint is satisfied, or a positive value
        if the constraint is violated. Implementation depends on the type of constraint.

        :param sim_data_dict: Dictionary of the form {modelname: {suffix1: Data1}} containing the simulated data objects
        :type sim_data_dict: dict
        """

        raise NotImplementedError('Subclasses of Constraint must override penalty()')


class AtConstraint(Constraint):
    def __init__(self, quant1, sign, quant2, base_model, base_suffix, weight, atvar, atval, altpenalty=None, minpenalty=0.,
                 repeat=False, before=False):
        """
        Creates a new constraint of the form

        X1>value at X2=value

        :param atvar: Variable checked to determine the 'at' condition, or None for the independent variable
        :param atval: Value that the atvar must take to activate the 'at' condition
        :param repeat: If True, enforce the constraint every time the 'at' condition is met. If False, only enforce
        the first time
        :param before: If True, enforce the constraint at the time point immediately before the 'at' condition is met
        """

        super().__init__(quant1, sign, quant2, base_model, base_suffix, weight, altpenalty, minpenalty)
        self.atvar = atvar
        self.atval = atval
        self.repeat = repeat
        self.before = before

        self.atkeys = None
        logger.debug("Created 'at' constraint %s<%s" % (self.quant1, self.quant2))

    def find_keys(self, sim_data_dict):
        """
        Function to be called on the first evaluation of the penalty.
        Read through the data dictionary and figure out what keys I need to use to access each relevant variable.

        This needs to be done kind of by brute force, because we were never told the model file to use, and it's even
        possible the input is poorly defined by referring to a suffix that exists in multiple models.
        :param sim_data_dict:
        :return:
        """

        keylist = []
        for q in (self.quant1, self.quant2, self.alt1, self.alt2, self.atvar):
            key = self.get_key(q, sim_data_dict)
            keylist.append(key)
        self.qkeys1, self.qkeys2, self.akeys1, self.akeys2, self.atkeys = keylist
        if self.atkeys is None:
            # No name was specified for atvar; we default to the independent variable of the default suffix
            self.atkeys = (self.base_model, self.base_suffix,
                           sim_data_dict[self.base_model][self.base_suffix].indvar)
        logger.debug("Got keys for 'at' constraint %s<%s: %s" %
                      (self.quant1, self.quant2, [self.qkeys1, self.qkeys2, self.akeys1, self.akeys2, self.atkeys]))

    def penalty(self, sim_data_dict):
        """
        Compute the penalty
        """

        # Load the keys if we haven't yet done so
        if not self.atkeys:
            self.find_keys(sim_data_dict)

        # Find the index where the when condition is met
        atdata = self.index(sim_data_dict, self.atkeys)
        at_col = atdata < self.atval
        # Make array that is True at every point where the atvar crosses the atval
        flip_points = at_col[:-1] != at_col[1:]
        flip_inds = np.nonzero(flip_points)[0]

        if not self.repeat:
            flip_inds = flip_inds[:1]

        penalty = 0.
        for fi in flip_inds:
            # Make sure we pick the correct end of the interval if there's a point equal to the "at"
            if np.isclose(atdata[fi+1], self.atval, atol=0.) and not (np.isclose(atdata[fi], self.atval, atol=0.)):
                fi += 1
            # If constraint was declared with "before", go back 1.
            if self.before and fi > 0:
                fi -= 1
            penalty += self.get_penalty(sim_data_dict, fi, fi+1, require_length=len(at_col))
            # Todo - if atvar and quant1 were simulated on different time scales, need to scour independent variable cols

        return penalty


class BetweenConstraint(Constraint):
    def __init__(self, quant1, sign, quant2, base_model, base_suffix, weight, startvar, startval, endvar, endval, altpenalty=None,
                 minpenalty=0.):
        """
        Creates a new constraint of the form

        X1 < X2 between X3=value  X4=value

        :param startvar: Variable checked to determine the start of the interval, or None for the independent variable
        :param startval: Value that the startvar must take to trigger the start of the interval
        :param endvar: Variable checked to determine the end of the interval, or None for the independent variable
        :param endval: Value that the endvar must take to trigger the start of the interval
        """

        super().__init__(quant1, sign, quant2, base_model, base_suffix, weight, altpenalty, minpenalty)

        self.startvar = startvar
        self.startval = startval
        self.endvar = endvar
        self.endval = endval

        self.startkeys = None
        self.endkeys = None
        logger.debug("Created 'between' constraint %s<%s" % (self.quant1, self.quant2))

    def find_keys(self, sim_data_dict):
        """
        Function to be called on the first evaluation of the penalty.
        Read through the data dictionary and figure out what keys I need to use to access each relevant variable.

        This needs to be done kind of by brute force, because we were never told the model file to use, and it's even
        possible the input is poorly defined by referring to a suffix that exists in multiple models.
        :param sim_data_dict:
        :return:
        """

        keylist = []
        for q in (self.quant1, self.quant2, self.alt1, self.alt2, self.startvar, self.endvar):
            key = self.get_key(q, sim_data_dict)
            keylist.append(key)
        self.qkeys1, self.qkeys2, self.akeys1, self.akeys2, self.startkeys, self.endkeys = keylist
        if self.startkeys is None:
            # No name was specified for startvar; we default to the independent variable of the default suffix
            self.startkeys = (self.base_model, self.base_suffix,
                           sim_data_dict[self.base_model][self.base_suffix].indvar)
        if self.endkeys is None:
            # Same for endvar
            self.endkeys = (self.base_model, self.base_suffix,
                           sim_data_dict[self.base_model][self.base_suffix].indvar)
        logger.debug("Got keys for 'between' constraint %s<%s: %s" %
                      (self.quant1, self.quant2, [self.qkeys1, self.qkeys2, self.akeys1, self.akeys2, self.startkeys, self.endkeys]))


    def penalty(self, sim_data_dict):
        """
        Compute the penalty
        """

        if not self.startkeys:
            self.find_keys(sim_data_dict)

        # Find the indices bounding the condition
        startdat = self.index(sim_data_dict, self.startkeys)
        startcol = startdat < self.startval
        start = np.nonzero(startcol[:-1] != startcol[1:])[0]
        if len(start) == 0:
            # Interval never started
            return 0.
        else:
            start = start[0]
            # If a point is exactly equal, make sure we pick the right end of the interval
            if start < len(startdat) - 1 and np.isclose(startdat[start+1], self.startval, atol=0.) \
                    and not (np.isclose(startdat[start], self.startval, atol=0.)):
                start += 1

        enddat = self.index(sim_data_dict, self.endkeys)
        endcol = enddat < self.endval
        if len(startcol) != len(endcol):
            raise PybnfError('Applying constraint %s<%s. Constraints involving observables that have different time '
                             'points in their simulation outputs are not currently supported' %
                             (self.quant1, self.quant2))
        # Note: [0] strips a useless extra dimension
        end = np.nonzero(endcol[start+1:-1] != endcol[start+2:])[0]
        if len(end) == 0:
            # Interval never ended
            end = len(startcol) - 1
        else:
            end = end[0]  # Take the first point satisfying the end condition
            end += start + 1
        if end < len(endcol) - 1 and np.isclose(enddat[end+1], self.endval, atol=0.) \
                and not (np.isclose(enddat[end], self.endval, atol=0.)):
            end += 1

        penalty = self.get_penalty(sim_data_dict, start, end+1, require_length=len(endcol))

        return penalty


class AlwaysConstraint(Constraint):
    def __init__(self, quant1, sign, quant2, base_model, base_suffix, weight, altpenalty=None, minpenalty=0.):
        """
        Creates a new constraint of the form

        X1>X2 always
        """

        super().__init__(quant1, sign, quant2, base_model, base_suffix, weight, altpenalty, minpenalty)
        logger.debug("Created 'always' constraint %s<%s" % (self.quant1, self.quant2))

    def penalty(self, sim_data_dict):
        """
        Compute the always penalty
        The penalty is given by the worst miss of the constraint over the entire data column
        """
        if not self.qkeys1 and not self.qkeys2:
            self.find_keys(sim_data_dict)

        # Todo - if q1 and q2 are on different time scales
        penalty = self.get_penalty(sim_data_dict, 0, None)  # Note: Indexing by 0:None takes the entire column

        return penalty


class OnceConstraint(Constraint):
    def __init__(self, quant1, sign, quant2, base_model, base_suffix, weight, altpenalty=None, minpenalty=0.):
        """
        Creates a new constraint of the form

        X1>X2 once
        """

        super().__init__(quant1, sign, quant2, base_model, base_suffix, weight, altpenalty, minpenalty)
        logger.debug("Created 'once' constraint %s<%s" % (self.quant1, self.quant2))

    def penalty(self, sim_data_dict):
        """
        Compute the penalty
        """
        if not self.qkeys1 and not self.qkeys2:
            self.find_keys(sim_data_dict)

        # Note: Indexing by 0:None takes the entire column
        penalty = self.get_penalty(sim_data_dict, 0, None, once=True)

        return penalty
