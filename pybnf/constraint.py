"""
Classes for defining various constraints that can be applied to the fitting run.  Used when incorporating qualitative
data into the fit
"""


from .printing import PybnfError
import pyparsing as pp
import numpy as np
import re
import logging
from math import erf


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
        Return the number of failed (unsatisfied) constraints, i.e. those with a
        nonzero penalty. (Matches the method name; the body counts 1 per nonzero
        penalty.)

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
        with open(f'{output_dir}/{self.base_suffix}_constraint_eval.txt', 'w') as out:
            for c in self.constraints:
                out.write(f'{c.penalty(sim_data_dict)}\n')

    def load_constraint_file(self, filename, scale=1.0):
        """
        Parse the constraint file filename and load them all into my constraint list
        :param filename: Path of constraint file
        :param scale: Factor by which we multiply all constraint weights
        """
        logger.info(f'Loading constraints for {self.base_model} suffix {self.base_suffix} from {filename}')
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
                if p.ineq:
                    # Normal case - for everything except "split at" constraint
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
                    pmin = None
                    pmax = None
                    tolerance = None
                    if p.likelihood_expr:
                        # Should not happen if parsing is working right
                        raise ValueError('Parsed with both weight_expr and likelihood_expr')
                elif p.likelihood_expr:
                    if p.likelihood_expr.confidence:
                        confidence = float(p.likelihood_expr.confidence)
                        pmin = (1. - confidence) / 2.
                        pmax = 1. - pmin
                    else:
                        pmin = float(p.likelihood_expr.pmin)
                        pmax = float(p.likelihood_expr.pmax)
                    tolerance = float(p.likelihood_expr.tolerance) if p.likelihood_expr.tolerance else 0.
                    weight = None
                    altpenalty = None
                    minpenalty = None
                else:
                    weight = 1.
                    altpenalty = None
                    minpenalty = 0.
                    pmin = None
                    pmax = None
                    tolerance = None

                if weight is not None:
                    weight *= scale  # Scale the weight by the specified factor

                # Check the constraint type based on the parse object, extract the constraint-type-specific args, and
                # make the constraint
                if p.split:
                    # Special case for "split at" combining inequality and enforcement clauses
                    quant1 = p.split.obs1
                    quant2 = p.split.obs2
                    if len(p.split.at1[1]) == 1:
                        atvar1 = None
                        atval1 = float(p.split.at1[1][0])
                    else:
                        atvar1 = p.split.at1[1][0]
                        atval1 = float(p.split.at1[1][1])
                    if len(p.split.at2[1]) == 1:
                        atvar2 = None
                        atval2 = float(p.split.at2[1][0])
                    else:
                        atvar2 = p.split.at2[1][0]
                        atval2 = float(p.split.at2[1][1])
                    sign = p.split.sign
                    repeat = (len(p.split.at1) >= 3 and p.split.at1[2] == 'everytime') or \
                             (len(p.split.at2) >= 3 and p.split.at2[2] == 'everytime')
                    before1 = (len(p.split.at1) >= 3 and p.split.at1[-1] == 'before')
                    before2 = (len(p.split.at2) >= 3 and p.split.at2[-1] == 'before')
                    con = SplitAtConstraint(quant1, atvar1, atval1, sign, quant2, atvar2, atval2, self.base_model,
                                            self.base_suffix, weight, altpenalty=altpenalty, minpenalty=minpenalty,
                                            repeat=repeat, before1=before1, before2=before2, pmin=pmin, pmax=pmax,
                                            tolerance=tolerance)
                elif p.enforce[0] == 'at':
                    if len(p.enforce[1]) == 1:
                        atval = float(p.enforce[1][0])
                        atvar = None
                    else:
                        atvar = p.enforce[1][0]
                        atval = float(p.enforce[1][1])
                    repeat = (len(p.enforce) >= 3 and p.enforce[2] == 'everytime')
                    before = (len(p.enforce) >= 3 and p.enforce[-1] == 'before')
                    con = AtConstraint(quant1, sign, quant2, self.base_model, self.base_suffix, weight, altpenalty=altpenalty,
                                           minpenalty=minpenalty, atvar=atvar, atval=atval, repeat=repeat, before=before,
                                       pmin=pmin, pmax=pmax, tolerance=tolerance)
                elif p.enforce[0] == 'between' or p.enforce[0] == 'once between':
                    once = (p.enforce[0] == 'once between')
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
                                           endval=endval, pmin=pmin, pmax=pmax, tolerance=tolerance, once=once)
                elif p.enforce[0] == 'always':
                    con = AlwaysConstraint(quant1, sign, quant2, self.base_model, self.base_suffix, weight, altpenalty=altpenalty,
                                           minpenalty=minpenalty, pmin=pmin, pmax=pmax, tolerance=tolerance)
                elif p.enforce[0] == 'once':
                    con = OnceConstraint(quant1, sign, quant2, self.base_model, self.base_suffix, weight, altpenalty,
                                         minpenalty, pmin, pmax, tolerance)
                else:
                    raise RuntimeError(f'Unknown enforcement keyword {p.enforce[0]}')
                con.source_line = line.strip()
                self.constraints.append(con)
        logger.info('Loaded %i constraints' % len(self.constraints))

    @staticmethod
    def parse_constraint_line(line):
        obs = pp.Word(pp.alphas, pp.alphanums+'_.')
        point = pp.Literal(".")
        e = pp.CaselessLiteral("E")
        number = pp.Combine(pp.Word("+-" + pp.nums, pp.nums) +
                         pp.Optional(point + pp.Optional(pp.Word(pp.nums))) +
                         pp.Optional(e + pp.Word("+-" + pp.nums, pp.nums)))
        iop = pp.one_of("< <= > >=")
        ineq0 = obs + iop + (obs ^ number)
        ineq1 = number + iop + obs
        ineq = ineq0 ^ ineq1
        equals = pp.Suppress('=')
        obs_crit = obs - equals - number
        enforce_crit = number | obs_crit
        enforce_at = pp.CaselessLiteral('at') - pp.Group(enforce_crit) - pp.Optional(pp.one_of('everytime first', caseless=True)) -\
            pp.Optional(pp.CaselessLiteral('before'))
        enforce_between = pp.Or([pp.CaselessLiteral('once between'), pp.CaselessLiteral('between')]) - \
                          pp.Group(enforce_crit) - pp.Suppress(',') - pp.Group(enforce_crit)
        enforce_other = pp.one_of('once always', caseless=True)
        enforce = enforce_at ^ enforce_between ^ enforce_other
        split = obs.set_results_name('obs1') - enforce_at.set_results_name('at1') - iop.set_results_name('sign') - \
                obs.set_results_name('obs2') - enforce_at.set_results_name('at2')
        min = pp.CaselessLiteral('min') - number.set_results_name('min')
        penalty = pp.CaselessLiteral('altpenalty') - pp.Group(ineq).set_results_name('altpenalty')
        wt_expr = number.set_results_name('weight') - pp.Optional(penalty) - pp.Optional(min)
        weight = pp.CaselessLiteral('weight') - wt_expr
        confidence = pp.CaselessLiteral('confidence') - number.set_results_name('confidence')
        pmin_pmax = pp.CaselessLiteral('pmin') - number.set_results_name('pmin') - \
                    pp.CaselessLiteral('pmax') - number.set_results_name('pmax')
        likelihood = (confidence ^ pmin_pmax) - \
                     pp.Optional(pp.CaselessLiteral('tolerance') - number.set_results_name('tolerance'))
        comment = pp.Suppress(pp.Literal('#') - pp.ZeroOrMore(pp.Word(pp.printables)))
        constraint = ((pp.Group(ineq).set_results_name('ineq') + pp.Group(enforce).set_results_name('enforce')) ^
                      pp.Group(split).set_results_name('split')) + \
                     pp.Optional(pp.Group(weight).set_results_name('weight_expr') ^
                                 pp.Group(likelihood).set_results_name('likelihood_expr')) + pp.Optional(comment)

        return constraint.parse_string(line, parse_all=True)


class Constraint:
    """
    Abstract class representing an optimization constraint with a penalty for violating the constraint
    """

    def __init__(self, quant1, sign, quant2, base_model, base_suffix, weight, altpenalty=None, minpenalty=0.,
                 pmin=None, pmax=None, tolerance=None):
        """
        Create a constraint of the form (quant1) (sign) (quant2)
        Depending on which parameters are passed, the constraint automatically chooses a model for penalty calculation
        If weight and/or altpenalty are passed, uses the static penalty method
        If confidence and/or tolerance are passed, uses the log likelihood method with a Gaussian CDF.
        The two penalty models should not be mixed.

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
        :param pmin: Using the log likelihood penalty, the minimum possible probability of this constraint.
        Represents the probability that the constraint is satisfied regardless of the output of the model
        :param pmax: Using the log likelihood penalty, the maximum possible probability of this constraint.
        Represents the probability that the constraint is violated regardless of the output of the model
        :param tolerance: Using the log likelihood penalty, the standard deviation of the Gaussian CDF.
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
        self.source_line = ''  # Original text from .prop file, set during parsing

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

        self.pmin = pmin
        self.pmax = pmax
        self.tolerance = tolerance

        # Choose penalty model depending on what is set to None
        # Check for invalid combinations of keys, but these should have been caught during parsing.
        if weight is not None:
            self.penalty_model = 'static'
            if pmin is not None or pmax is not None or tolerance is not None:
                raise ValueError(f'Constraint {quant1}{sign}{quant2} has a weight, and so should not have a pmin, pmax and/or tolerance')
        elif pmin is not None:
            self.penalty_model = 'likelihood'
            if pmax is None:
                raise ValueError(f'Constraint {quant1}{sign}{quant2} has a pmin but not a pmax')
            if tolerance is None:
                raise ValueError(f'Constraint {quant1}{sign}{quant2} has a pmin and pmax but not a tolerance')
            if altpenalty is not None:
                raise ValueError(f'Constraint {quant1}{sign}{quant2} should not have both a confidence and a altpenalty')
            if pmin < 0. or pmax > 1.:
                raise PybnfError(f'Constraint {quant1}{sign}{quant2} has invalid probabilities. Settings for confidence, pmin, and/or '
                                 'pmax must be between 0 and 1.')
            if pmin > pmax:
                raise PybnfError(f'Constraint {quant1}{sign}{quant2} has pmin set larger than pmax.')
        else:
            raise ValueError(f'Constraint {quant1}{sign}{quant2} must have either a weight or a confidence')

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

        Subclasses that introduce additional "at"/"between" variables override
        :meth:`_find_extra_keys` to resolve those; the common quant/alt key
        resolution lives here so it isn't duplicated per subclass.
        :param sim_data_dict:
        :return:
        """
        self.qkeys1, self.qkeys2, self.akeys1, self.akeys2 = (
            self.get_key(q, sim_data_dict) for q in (self.quant1, self.quant2, self.alt1, self.alt2))
        self._find_extra_keys(sim_data_dict)
        logger.debug(f"Got keys for constraint {self.quant1}<{self.quant2}: {[self.qkeys1, self.qkeys2, self.akeys1, self.akeys2]}")

    def _find_extra_keys(self, sim_data_dict):
        """
        Hook for subclasses to resolve the keys of any additional variables they
        reference (the "at"/"start"/"end" conditions). The base constraint has
        no such variables, so this does nothing.
        """
        pass

    def _atvar_key(self, atvar, sim_data_dict):
        """
        Resolve the data key for an "at"-style condition variable, defaulting to
        the independent variable of the base suffix when the variable name was
        omitted (atvar is None).
        """
        key = self.get_key(atvar, sim_data_dict)
        if key is None:
            key = (self.base_model, self.base_suffix,
                   sim_data_dict[self.base_model][self.base_suffix].indvar)
        return key

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
                raise PybnfError(f'Constraint file is associated with model {self.base_model}, but that model was not found')
            if self.base_suffix not in sim_data_dict[self.base_model]:
                raise PybnfError(f"Constraint file {self.base_suffix}.prop refers to suffix '{self.base_suffix}' in model '{self.base_model}', but that model does not "
                                 "produce output with that suffix")
            if q not in sim_data_dict[self.base_model][self.base_suffix].cols:
                raise PybnfError(f"Observable '{q}' specified in constraint file was not found in the output of model "
                                 f"'{self.base_model}', suffix '{self.base_suffix}'")
            return (self.base_model, self.base_suffix, q)
        else:
            parts = q.split('.')
            if len(parts) > 2:
                raise PybnfError(f'Parsing constraint variable {q}',
                                 "Parsing constraint variable %s - The format should be 'suffix.Observable' Should "
                                 "not contain multiple '.' characters.")
            options = []
            for m in sim_data_dict:
                if parts[0] in sim_data_dict[m]:
                    options.append(m)
            if len(options) == 0:
                raise PybnfError(f"Could not find any experimental data corresponding to the suffix '{parts[0]}' specified "
                                 "in your constraint file")
            if len(options) > 1:
                raise PybnfError(f"Suffix '{parts[0]}' in your constraint file appears in multiple model files. Please "
                                 "change your suffix names so it isn't ambiguous")
            if parts[1] not in sim_data_dict[options[0]][parts[0]].cols:
                raise PybnfError(f"Obserable '{q}' in constraint file - The output with suffix {parts[0]} does not contain an "
                                 f"observable {parts[1]}")
            return (options[0], parts[0], parts[1])

    def index(self, sim_data_dict, keys):
        """
        Shortcut function for applying all 3 indices to the data object
        :return:
        """
        return sim_data_dict[keys[0]][keys[1]][keys[2]]

    def get_penalty(self, sim_data_dict, imin, imax, once=False, require_length=None, imin2=None, imax2=None):
        """
        Helper function for calculating the penalty, that can be called from the subclasses.
        Chooses to call either get_static_penalty or get_log_likelihood depending on the penalty model
        used for this constraint
        """
        if self.penalty_model == 'static':
            return self.get_static_penalty(sim_data_dict, imin, imax, once, require_length, imin2, imax2)
        elif self.penalty_model == 'likelihood':
            return self.get_log_likelihood(sim_data_dict, imin, imax, once, require_length, imin2, imax2)
        else:
            raise ValueError(f'Invalid penalty model: {self.penalty_model}')

    def get_difference(self, sim_data_dict, imin, imax, once=False, require_length=None, imin2=None, imax2=None):
        """
        Helper function for calculating the penalty, that can be called from the subclasses.
        Calculates the difference between the two sides of the inequality. A negative value means the inequality is
        satisfied.
        Considers the worst case over the specified interval, or the best case if once=True.

        The result can be used to calculate the penalty using a static penalty model or a likelihood model.

        :param sim_data_dict: The dictionary of data objects
        :param imin: First index at which to check the constraint
        :param imax: Last index at which to check the constraint (exclusive)
        :param once: If true, enforce that the constraint holds once at some point during the time interval
        :param require_length: If set to an integer, raise an error if the length of the selected data column(s) is not
        equal to that value. (Used to check that "at" and "between" constraints are not encountering an unsupported
        case)
        :param imin2: If specified, use this different index for quantity 2
        :param imax2: If specified, use this different index for quantity 2
        :return:
        """

        def length_error():
            raise PybnfError(f'Applying constraint {self.quant1}<{self.quant2}. Constraints involving observables that have different time '
                             'points in their simulation outputs are not currently supported')

        if imin2 is None:
            imin2 = imin
        if imax2 is None:
            imax2 = imax
        if (imax is not None and imax-imin != imax2-imin2) or \
                (imax is None and (imin != imin2 or imax2 is not None)):
            # Note that the case of imax=None occurs if the desired max index is the end of the time course.
            # Should not happen, because currently only the Split At constraint uses imin2 and imax2
            raise ValueError('imin2:imax2 has a different length than imin:imax')

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
            q2 = q2_col[imin2:imax2]
        else:
            q2 = self.quant2

        try:
            if once:
                difference = np.min(q1 - q2)
            else:
                difference = np.max(q1 - q2)
        except ValueError:
            length_error()

        return difference

    def get_static_penalty(self, sim_data_dict, imin, imax, once=False, require_length=None, imin2=None, imax2=None):
        """
        Helper function for calculating the static penalty, that can be called from the subclasses.

        :param sim_data_dict: The dictionary of data objects
        :param imin: First index at which to check the constraint
        :param imax: Last index at which to check the constraint (exclusive)
        :param once: If true, enforce that the constraint holds once at some point during the time interval
        :param require_length: If set to an integer, raise an error if the length of the selected data column(s) is not
        equal to that value. (Used to check that "at" and "between" constraints are not encountering an unsupported
        case)
        :param imin2: If specified, use this different index for quantity 2
        :param imax2: If specified, use this different index for quantity 2
        :return:
        """
        penalty = self.get_difference(sim_data_dict, imin, imax, once, require_length, imin2, imax2)

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

    def get_log_likelihood(self, sim_data_dict, imin, imax, once=False, require_length=None, imin2=None, imax2=None):
        """
        Helper function for calculating the negative log likelihood of constraint satisfaction given the parameters,
        i.e. a likelihood-based penalty function.
        The likelihood is calculated with a Gaussian CDF defined in terms of this constraint's confidence
        and tolerance

        :param sim_data_dict: The dictionary of data objects
        :param imin: First index at which to check the constraint
        :param imax: Last index at which to check the constraint (exclusive)
        :param once: If true, enforce that the constraint holds once at some point during the time interval
        :param require_length: If set to an integer, raise an error if the length of the selected data column(s) is not
        equal to that value. (Used to check that "at" and "between" constraints are not encountering an unsupported
        case)
        :param imin2: If specified, use this different index for quantity 2
        :param imax2: If specified, use this different index for quantity 2
        :return:
        """

        def cdf(x):
            """
            The cumulative distribution function of a Gaussian distribution with mean 0 and variance 1

            Previous code versions used the approximation cdf(x) = 1./(1. + np.exp(-1.7*x))
            """
            return (1. + erf(x / np.sqrt(2.))) / 2.

        difference = self.get_difference(sim_data_dict, imin, imax, once, require_length, imin2, imax2)
        if self.tolerance == 0:
            # Edge case where cdf is a step function
            if difference < 0 or (difference == 0 and self.or_equal):
                log_likelihood = -np.log(self.pmax)
            else:
                log_likelihood = -np.log(self.pmin)
        else:
            # Standard case
            # Note that a negative difference is good, hence the sign convention in the calculation below.
            k = self.tolerance
            prob = cdf(-difference/k)
            adjusted_prob = (self.pmax - self.pmin) * prob + self.pmin
            log_likelihood = -np.log(adjusted_prob)
        return log_likelihood



    def penalty(self, sim_data_dict):
        """
        penalty function for violating the constraint. Returns 0 if constraint is satisfied, or a positive value
        if the constraint is violated. Implementation depends on the type of constraint.

        :param sim_data_dict: Dictionary of the form {modelname: {suffix1: Data1}} containing the simulated data objects
        :type sim_data_dict: dict
        """

        raise NotImplementedError('Subclasses of Constraint must override penalty()')

    def penalty_gradient(self, sim_data_dict):
        """The gradient of this constraint's penalty w.r.t. the free parameters --
        the constraint counterpart of the objective's residual seam (#449/#385).

        A constraint penalty is a piecewise function of an at-/between-time
        observable readout, so its gradient is the readout's forward sensitivity
        times the local penalty slope -- a distinct chain from the per-point
        likelihood residual. That is layer I of #385 (not yet implemented), so the
        stub raises :class:`GradientNotSupported`: a gradient fit with active
        constraints falls back to a gradient-free step rather than silently dropping
        the constraint's contribution to the gradient."""
        from .gradient.errors import GradientNotSupported
        raise GradientNotSupported(
            "Constraint gradients are layer I of #385 (not yet implemented); run a "
            "gradient-free fit, or drop constraints from the gradient-based fit.")


class AtConstraint(Constraint):
    def __init__(self, quant1, sign, quant2, base_model, base_suffix, weight, atvar, atval, altpenalty=None, minpenalty=0.,
                 repeat=False, before=False, pmin=None, pmax=None, tolerance=None):
        """
        Creates a new constraint of the form

        X1>value at X2=value

        :param atvar: Variable checked to determine the 'at' condition, or None for the independent variable
        :param atval: Value that the atvar must take to activate the 'at' condition
        :param repeat: If True, enforce the constraint every time the 'at' condition is met. If False, only enforce
        the first time
        :param before: If True, enforce the constraint at the time point immediately before the 'at' condition is met
        """

        super().__init__(quant1, sign, quant2, base_model, base_suffix, weight, altpenalty, minpenalty, pmin, pmax,
                         tolerance)
        self.atvar = atvar
        self.atval = atval
        self.repeat = repeat
        self.before = before

        self.atkeys = None
        logger.debug(f"Created 'at' constraint {self.quant1}<{self.quant2}")

    def _find_extra_keys(self, sim_data_dict):
        """Resolve the 'at' condition variable, defaulting to the indvar."""
        self.atkeys = self._atvar_key(self.atvar, sim_data_dict)

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

class SplitAtConstraint(Constraint):
    def __init__(self, quant1, atvar1, atval1, sign, quant2, atvar2, atval2, base_model, base_suffix,
                 weight, altpenalty=None, minpenalty=0.,
                 repeat=False, before1=False, before2=False, pmin=None, pmax=None, tolerance=None):
        """
        Creates a new constraint of the form

        X1 at X2=value < X3 at X4=value

        :param atvar: Variable checked to determine the 'at' condition, or None for the independent variable
        :param atval: Value that the atvar must take to activate the 'at' condition
        :param repeat: If True, enforce the constraint every time the 'at' condition is met. If False, only enforce
        the first time
        :param before: If True, enforce the constraint at the time point immediately before the 'at' condition is met
        """

        super().__init__(quant1, sign, quant2, base_model, base_suffix, weight, altpenalty, minpenalty, pmin, pmax,
                         tolerance)

        # Since the superclass flips the sign for ">", we need to also flip the "at" conditions
        if sign in ('>', '>='):
            self.atvar1 = atvar2
            self.atval1 = atval2
            self.atvar2 = atvar1
            self.atval2 = atval1
            self.before1 = before2
            self.before2 = before1
        else:
            self.atvar1 = atvar1
            self.atval1 = atval1
            self.atvar2 = atvar2
            self.atval2 = atval2
            self.before1 = before1
            self.before2 = before2
        if repeat:
            raise PybnfError('The "repeat" keyword is not currently supported for constraints with two "at" conditions')

        self.atkeys1 = None
        self.atkeys2 = None
        logger.debug(f"Created split 'at' constraint {self.quant1}<{self.quant2}")

    def _find_extra_keys(self, sim_data_dict):
        """Resolve both 'at' condition variables, each defaulting to the indvar."""
        self.atkeys1 = self._atvar_key(self.atvar1, sim_data_dict)
        self.atkeys2 = self._atvar_key(self.atvar2, sim_data_dict)

    def penalty(self, sim_data_dict):
        """
        Compute the penalty
        """

        # Load the keys if we haven't yet done so
        if not self.atkeys1:
            self.find_keys(sim_data_dict)

        # Find the index where the when condition is met
        atdata1 = self.index(sim_data_dict, self.atkeys1)
        at_col1 = atdata1 < self.atval1
        # Make array that is True at every point where the atvar crosses the atval
        flip_points1 = at_col1[:-1] != at_col1[1:]
        flip_inds1 = np.nonzero(flip_points1)[0]

        # Repeat for 2nd at condition
        atdata2 = self.index(sim_data_dict, self.atkeys2)
        at_col2 = atdata2 < self.atval2
        flip_points2 = at_col2[:-1] != at_col2[1:]
        flip_inds2 = np.nonzero(flip_points2)[0]

        if len(flip_inds1) == 0 or len(flip_inds2) == 0:
            # One of the at conditions was not met, so constraint is not enforced.
            return 0.

        fi1 = flip_inds1[0]
        fi2 = flip_inds2[0]

        # Make sure we pick the correct end of the interval if there's a point equal to the "at"
        if np.isclose(atdata1[fi1+1], self.atval1, atol=0.) and not (np.isclose(atdata1[fi1], self.atval1, atol=0.)):
            fi1 += 1
        if np.isclose(atdata2[fi2+1], self.atval2, atol=0.) and not (np.isclose(atdata2[fi2], self.atval2, atol=0.)):
            fi2 += 1
        # If constraint was declared with "before", go back 1.
        if self.before1 and fi1 > 0:
            fi1 -= 1
        if self.before2 and fi2 > 0:
            fi2 -= 1
        penalty = self.get_penalty(sim_data_dict, fi1, fi1+1, require_length=len(at_col1), imin2=fi2, imax2=fi2+1)
        # Todo - if atvar and quant1 were simulated on different time scales, need to scour independent variable cols

        return penalty

class BetweenConstraint(Constraint):
    def __init__(self, quant1, sign, quant2, base_model, base_suffix, weight, startvar, startval, endvar, endval, altpenalty=None,
                 minpenalty=0., pmin=None, pmax=None, tolerance=None, once=False):
        """
        Creates a new constraint of the form

        X1 < X2 between X3=value  X4=value

        or, if once=True,
        X1 < X2 once between X3=value  X4=value

        :param startvar: Variable checked to determine the start of the interval, or None for the independent variable
        :param startval: Value that the startvar must take to trigger the start of the interval
        :param endvar: Variable checked to determine the end of the interval, or None for the independent variable
        :param endval: Value that the endvar must take to trigger the start of the interval
        """

        super().__init__(quant1, sign, quant2, base_model, base_suffix, weight, altpenalty, minpenalty, pmin, pmax,
                         tolerance)

        self.startvar = startvar
        self.startval = startval
        self.endvar = endvar
        self.endval = endval

        self.startkeys = None
        self.endkeys = None
        self.once = once
        logger.debug(f"Created 'between' constraint {self.quant1}<{self.quant2}")

    def _find_extra_keys(self, sim_data_dict):
        """Resolve the start/end interval variables, each defaulting to the indvar."""
        self.startkeys = self._atvar_key(self.startvar, sim_data_dict)
        self.endkeys = self._atvar_key(self.endvar, sim_data_dict)

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
            raise PybnfError(f'Applying constraint {self.quant1}<{self.quant2}. Constraints involving observables that have different time '
                             'points in their simulation outputs are not currently supported')
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

        penalty = self.get_penalty(sim_data_dict, start, end+1, require_length=len(endcol), once=self.once)

        return penalty


class AlwaysConstraint(Constraint):
    def __init__(self, quant1, sign, quant2, base_model, base_suffix, weight, altpenalty=None, minpenalty=0.,
                 pmin=None, pmax=None, tolerance=None):
        """
        Creates a new constraint of the form

        X1>X2 always
        """

        super().__init__(quant1, sign, quant2, base_model, base_suffix, weight, altpenalty, minpenalty, pmin, pmax,
                         tolerance)
        logger.debug(f"Created 'always' constraint {self.quant1}<{self.quant2}")

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
    def __init__(self, quant1, sign, quant2, base_model, base_suffix, weight, altpenalty=None, minpenalty=0.,
                 pmin=None, pmax=None, tolerance=None):
        """
        Creates a new constraint of the form

        X1>X2 once
        """

        super().__init__(quant1, sign, quant2, base_model, base_suffix, weight, altpenalty, minpenalty, pmin, pmax,
                         tolerance)
        logger.debug(f"Created 'once' constraint {self.quant1}<{self.quant2}")

    def penalty(self, sim_data_dict):
        """
        Compute the penalty
        """
        if not self.qkeys1 and not self.qkeys2:
            self.find_keys(sim_data_dict)

        # Note: Indexing by 0:None takes the entire column
        penalty = self.get_penalty(sim_data_dict, 0, None, once=True)

        return penalty
