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


def _sigmoid(x):
    """The logistic function ``1 / (1 + e^{-x})``, evaluated in the numerically stable branch
    (``e^{x}/(1+e^{x})`` for ``x < 0``) so large ``|x|`` never overflows. Scalar in, scalar out --
    the local slope factor of the logit penalty."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    ex = np.exp(x)
    return ex / (1.0 + ex)


class ConstraintSet:
    """
    Represents the set of all constraints provided in one prop file
    """
    def __init__(self, base_model, base_suffix):
        self.constraints = []
        self.base_suffix = base_suffix
        self.base_model = base_model

    def total_penalty(self, sim_data_dict, pset_values=None):
        """
        Evaluate the sim_data_dict against all constraints, and return the total penalty

        :param sim_data_dict: Dictionary of the form {modelname: {suffix1: Data1}} containing the simulated data objects
        :param pset_values: The objective's live ``{name: value}`` map, threaded so a constraint with an
            estimated scale parameter reads its current value; ``None`` uses the literal.
        :return:
        """
        return sum([c.penalty(sim_data_dict, pset_values=pset_values) for c in self.constraints])

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

    def load_constraint_file(self, filename, scale=1.0, qualitative_loss='auto', qualitative_scale=None):
        """
        Parse the constraint file filename and load them all into my constraint list
        :param filename: Path of constraint file
        :param scale: Factor by which we multiply all constraint weights
        :param qualitative_loss: One of 'auto', 'hinge', 'probit', 'logit'. 'auto' (default) keeps
        each constraint's authored penalty family (the keyword-driven per-constraint selection);
        any other value coerces every constraint to that family via
        :meth:`Constraint.coerce_penalty_model` -- a benchmarking convenience.
        :param qualitative_scale: If set, the name of a free parameter every constraint's scale is
        tied to (:meth:`Constraint.bind_scale_param`), so a fit estimates the qualitative scale
        jointly. Applied after any ``qualitative_loss`` coercion.
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
                    logit_scale = None
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
                    logit_scale = None
                elif p.logit_expr:
                    # Logit (Miller et al. 2025) softplus penalty. `scale` is the s of ln(1+e^{diff/s}).
                    # Optional pmin/pmax add probit-style label smoothing (default off = faithful to
                    # the arXiv derivation).
                    logit_scale = float(p.logit_expr.scale)
                    if p.logit_expr.pmin:
                        pmin = float(p.logit_expr.pmin)
                        pmax = float(p.logit_expr.pmax)
                    else:
                        pmin = None
                        pmax = None
                    tolerance = None
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
                    logit_scale = None

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
                                            tolerance=tolerance, scale=logit_scale)
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
                                       pmin=pmin, pmax=pmax, tolerance=tolerance, scale=logit_scale)
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
                                           endval=endval, pmin=pmin, pmax=pmax, tolerance=tolerance, once=once, scale=logit_scale)
                elif p.enforce[0] == 'always':
                    con = AlwaysConstraint(quant1, sign, quant2, self.base_model, self.base_suffix, weight, altpenalty=altpenalty,
                                           minpenalty=minpenalty, pmin=pmin, pmax=pmax, tolerance=tolerance, scale=logit_scale)
                elif p.enforce[0] == 'once':
                    con = OnceConstraint(quant1, sign, quant2, self.base_model, self.base_suffix, weight, altpenalty,
                                         minpenalty, pmin, pmax, tolerance, scale=logit_scale)
                else:
                    raise RuntimeError(f'Unknown enforcement keyword {p.enforce[0]}')
                con.source_line = line.strip()
                if qualitative_loss != 'auto':
                    con.coerce_penalty_model(qualitative_loss)
                if qualitative_scale is not None:
                    con.bind_scale_param(qualitative_scale)
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
        # Logit (Miller et al. 2025) penalty: `logit scale <s>` selects the softplus loss
        # ln(1 + e^{difference/s}); the optional `pmin/pmax` add probit-style label smoothing for
        # apples-to-apples benchmarking (default off).
        logit = pp.CaselessLiteral('logit') - pp.CaselessLiteral('scale') - number.set_results_name('scale') - \
                pp.Optional(pmin_pmax)
        comment = pp.Suppress(pp.Literal('#') - pp.ZeroOrMore(pp.Word(pp.printables)))
        constraint = ((pp.Group(ineq).set_results_name('ineq') + pp.Group(enforce).set_results_name('enforce')) ^
                      pp.Group(split).set_results_name('split')) + \
                     pp.Optional(pp.Group(weight).set_results_name('weight_expr') ^
                                 pp.Group(likelihood).set_results_name('likelihood_expr') ^
                                 pp.Group(logit).set_results_name('logit_expr')) + pp.Optional(comment)

        return constraint.parse_string(line, parse_all=True)


class Constraint:
    """
    Abstract class representing an optimization constraint with a penalty for violating the constraint
    """

    def __init__(self, quant1, sign, quant2, base_model, base_suffix, weight, altpenalty=None, minpenalty=0.,
                 pmin=None, pmax=None, tolerance=None, scale=None):
        """
        Create a constraint of the form (quant1) (sign) (quant2)
        Depending on which parameters are passed, the constraint automatically chooses a model for penalty calculation
        If weight and/or altpenalty are passed, uses the static (hinge) penalty method.
        If confidence and/or tolerance are passed, uses the log likelihood method with a Gaussian CDF (probit).
        If scale is passed, uses the logit softplus penalty (Miller et al. 2025).
        The three penalty models should not be mixed.

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
        :param scale: Using the logit penalty, the scale s of the softplus loss ln(1 + e^{difference/s}).
        Must be positive. Smaller s approaches the hinge (weight = 1/s); larger s is softer.
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
        self.scale = scale
        # Name of a free parameter this constraint's scale (logit s / probit sigma) is tied to,
        # or None for a fixed authored scale. Set by bind_scale_param when a fit ties the
        # qualitative scale to an estimated parameter (qualitative_scale = fit).
        self.scale_param = None

        # Choose penalty model depending on what is set to None
        # Check for invalid combinations of keys, but these should have been caught during parsing.
        if weight is not None:
            self.penalty_model = 'static'
            if pmin is not None or pmax is not None or tolerance is not None or scale is not None:
                raise ValueError(f'Constraint {quant1}{sign}{quant2} has a weight, and so should not have a pmin, pmax, tolerance and/or scale')
        elif scale is not None:
            # Logit (softplus) penalty. Checked before the probit branch because a logit constraint
            # may also carry optional pmin/pmax (label smoothing).
            self.penalty_model = 'logit'
            if scale <= 0.:
                raise PybnfError(f'Constraint {quant1}{sign}{quant2} has a logit scale of {scale}; the scale must be positive.')
            if tolerance is not None:
                raise ValueError(f'Constraint {quant1}{sign}{quant2} should not have both a logit scale and a tolerance')
            if altpenalty is not None:
                raise ValueError(f'Constraint {quant1}{sign}{quant2} should not have both a logit scale and an altpenalty')
            if (pmin is None) != (pmax is None):
                raise ValueError(f'Constraint {quant1}{sign}{quant2} logit penalty needs both pmin and pmax, or neither')
            if pmin is not None:
                if pmin < 0. or pmax > 1.:
                    raise PybnfError(f'Constraint {quant1}{sign}{quant2} has invalid probabilities. Settings for pmin '
                                     'and pmax must be between 0 and 1.')
                if pmin > pmax:
                    raise PybnfError(f'Constraint {quant1}{sign}{quant2} has pmin set larger than pmax.')
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
            raise ValueError(f'Constraint {quant1}{sign}{quant2} must have either a weight, a confidence, or a logit scale')

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

    def _intrinsic_scale(self):
        """This constraint's authored scale expressed as a **logit-equivalent** ``s`` -- the common
        currency the three qualitative losses convert through. Static hinge
        ``weight w`` -> ``1/w`` (the large-margin identity logit ~ hinge with weight ``1/s``);
        logit ``scale s`` -> ``s``; probit ``tolerance sigma`` -> ``sigma/1.6`` (the
        ``Phi(x) ~ sigma(1.6 x)`` link). Raises on a degenerate scale (zero weight, or a probit
        step function with tolerance 0) that has no finite logit-equivalent."""
        if self.penalty_model == 'static':
            if not self.weight:
                raise PybnfError(f"Cannot convert constraint '{self.source_line}' to another "
                                 "qualitative loss: it has zero weight (no finite scale).")
            return 1.0 / self.weight
        if self.penalty_model == 'logit':
            return self.scale
        if self.penalty_model == 'likelihood':
            if not self.tolerance:
                raise PybnfError(f"Cannot convert constraint '{self.source_line}' to another "
                                 "qualitative loss: its probit tolerance is 0 (a step function has "
                                 "no finite scale). Give it a positive tolerance.")
            return self.tolerance / 1.6
        raise PybnfError(f'Unknown penalty model {self.penalty_model}')

    def coerce_penalty_model(self, target):
        """Override this constraint's penalty family to ``target`` in {'hinge', 'probit', 'logit'},
        deriving the target's scale from whatever was authored via the scale-matching table in
        :meth:`_intrinsic_scale` (logit ``s`` <-> hinge ``weight = 1/s`` <-> probit
        ``tolerance = 1.6 s``). ``'auto'`` / ``None`` is a no-op. This is the mechanism behind the
        ``qualitative_loss`` config key -- a benchmarking convenience so one ``.prop`` set runs
        under all three losses without re-authoring. Authored ``pmin/pmax``
        label smoothing is preserved where the target supports it; a probit target with no authored
        smoothing gets the unclipped ``pmin=0, pmax=1``."""
        if target in (None, 'auto'):
            return
        base = self._intrinsic_scale()
        pmin, pmax = self.pmin, self.pmax
        # Clear every family's scale fields; each branch sets only the ones its model reads.
        self.weight = None
        self.scale = None
        self.tolerance = None
        self.pmin = None
        self.pmax = None
        if target == 'hinge':
            self.penalty_model = 'static'
            self.weight = 1.0 / base
        elif target == 'logit':
            self.penalty_model = 'logit'
            self.scale = base
            self.pmin, self.pmax = pmin, pmax   # keep label smoothing if it was authored
        elif target == 'probit':
            self.penalty_model = 'likelihood'
            self.tolerance = 1.6 * base
            self.pmin = 0.0 if pmin is None else pmin
            self.pmax = 1.0 if pmax is None else pmax
        else:
            raise PybnfError(f"Unknown qualitative_loss target '{target}'. "
                             "Expected one of 'auto', 'hinge', 'probit', 'logit'.")

    def bind_scale_param(self, name):
        """Tie this constraint's scale (the logit ``s`` or probit ``sigma``) to an estimated free
        parameter ``name``, so a fit can infer it jointly with the model parameters. Only the likelihood families carry a scale: a static (hinge) constraint has no
        scale to estimate, so binding one is an error. The authored literal scale is kept as the
        fallback/initial value (used when no pset is supplied, e.g. a bare diagnostic penalty())."""
        if self.penalty_model == 'static':
            raise PybnfError(
                f"Constraint '{self.source_line}' uses the static (hinge) penalty, which has no "
                "scale to estimate. qualitative_scale = fit needs logit or probit constraints "
                "(convert the whole fit with qualitative_loss = logit or probit).")
        self.scale_param = name

    def _effective_scale(self, pset_values):
        """This constraint's live scale: the current value of the tied free parameter if its scale
        is estimated (``scale_param`` set) and that value is available in ``pset_values``, otherwise
        the authored literal (logit ``self.scale`` / probit ``self.tolerance``). ``pset_values`` is
        the objective's ``{name: value}`` map for the evaluation; ``None`` (a bare penalty() call
        outside a fit) falls back to the literal."""
        if self.scale_param is not None and pset_values is not None and self.scale_param in pset_values:
            return pset_values[self.scale_param]
        return self.scale if self.penalty_model == 'logit' else self.tolerance

    def _scale_derivative(self, difference, s):
        """Closed-form ``d(penalty)/d(scale)`` at the interval's achieving ``difference`` -- the scalar-gradient contribution to the tied scale parameter's column, needing
        no forward sensitivity (the scale does not move the readout). Mirrors the estimated-noise
        ``d loss/d sigma`` term. Logit: ``-sigma(d/s) * d / s^2`` (unclipped), or the clipped form
        when pmin/pmax are set. Probit: ``-(pmax-pmin) * phi(-d/s) * d / (s^2 * adjusted_prob)``."""
        if self.penalty_model == 'logit':
            x = difference / s
            if self.pmin is None:
                return -_sigmoid(x) * difference / (s * s)
            adjusted = self.pmin + (self.pmax - self.pmin) * _sigmoid(-x)
            return -(self.pmax - self.pmin) * _sigmoid(-x) * _sigmoid(x) * difference / (s * s * adjusted)
        # probit (Gaussian-CDF likelihood)
        u = -difference / s
        phi = np.exp(-0.5 * u * u) / np.sqrt(2. * np.pi)
        prob = (1. + erf(u / np.sqrt(2.))) / 2.
        adjusted = (self.pmax - self.pmin) * prob + self.pmin
        return -(self.pmax - self.pmin) * phi * difference / (s * s * adjusted)

    def get_penalty(self, sim_data_dict, imin, imax, once=False, require_length=None, imin2=None,
                    imax2=None, pset_values=None):
        """
        Helper function for calculating the penalty, that can be called from the subclasses.
        Chooses to call either get_static_penalty or get_log_likelihood depending on the penalty model
        used for this constraint. ``pset_values`` (the objective's live ``{name: value}`` map) lets a
        likelihood constraint resolve an estimated scale parameter.
        """
        if self.penalty_model == 'static':
            return self.get_static_penalty(sim_data_dict, imin, imax, once, require_length, imin2, imax2)
        elif self.penalty_model == 'likelihood':
            return self.get_log_likelihood(sim_data_dict, imin, imax, once, require_length, imin2, imax2,
                                           pset_values=pset_values)
        elif self.penalty_model == 'logit':
            return self.get_log_likelihood_logit(sim_data_dict, imin, imax, once, require_length, imin2, imax2,
                                                 pset_values=pset_values)
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

    def get_log_likelihood(self, sim_data_dict, imin, imax, once=False, require_length=None, imin2=None, imax2=None,
                           pset_values=None):
        """
        Helper function for calculating the negative log likelihood of constraint satisfaction given the parameters,
        i.e. a likelihood-based penalty function.
        The likelihood is calculated with a Gaussian CDF defined in terms of this constraint's confidence
        and tolerance (or an estimated scale parameter resolved from ``pset_values``)

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
        k = self._effective_scale(pset_values)
        if k == 0:
            # Edge case where cdf is a step function
            if difference < 0 or (difference == 0 and self.or_equal):
                log_likelihood = -np.log(self.pmax)
            else:
                log_likelihood = -np.log(self.pmin)
        else:
            # Standard case
            # Note that a negative difference is good, hence the sign convention in the calculation below.
            prob = cdf(-difference/k)
            adjusted_prob = (self.pmax - self.pmin) * prob + self.pmin
            log_likelihood = -np.log(adjusted_prob)
        return log_likelihood

    def get_log_likelihood_logit(self, sim_data_dict, imin, imax, once=False, require_length=None, imin2=None,
                                 imax2=None, pset_values=None):
        """Negative log likelihood of constraint satisfaction under the **logit** (Miller et al.
        2025) noise model: the comparison outcome is Bernoulli with P(holds) = sigma(-difference/s),
        so the single-sided BPSL penalty collapses to the softplus

            F(difference; s) = ln(1 + e^{difference/s}) = softplus(difference/s).

        ``difference`` is the signed constraint margin (negative == satisfied, :meth:`get_difference`).
        As ``s -> 0`` with ``weight = 1/s`` this is the 2018 hinge (the large-margin asymptote);
        larger ``s`` is a softer penalty. ``np.logaddexp(0, x)`` evaluates the
        softplus without overflow. If optional ``pmin/pmax`` were supplied (label smoothing, for
        probit parity), the probability is clipped to ``pmin + (pmax-pmin)*sigma(-difference/s)``
        before the ``-log`` -- default off (faithful to the arXiv derivation).

        :param sim_data_dict: The dictionary of data objects
        :param imin: First index at which to check the constraint
        :param imax: Last index at which to check the constraint (exclusive)
        :param once: If true, enforce that the constraint holds once at some point during the interval
        :param require_length: If set, raise if the selected data column length differs (unsupported case)
        :param imin2: If specified, use this different index for quantity 2
        :param imax2: If specified, use this different index for quantity 2
        """
        difference = self.get_difference(sim_data_dict, imin, imax, once, require_length, imin2, imax2)
        x = difference / self._effective_scale(pset_values)
        if self.pmin is None:
            # Faithful softplus: -log sigma(-x) = ln(1 + e^{x}), stable via logaddexp.
            return float(np.logaddexp(0.0, x))
        # Label-smoothed (probit-parity) variant.
        adjusted_prob = self.pmin + (self.pmax - self.pmin) * _sigmoid(-x)
        return float(-np.log(adjusted_prob))



    def penalty(self, sim_data_dict, pset_values=None):
        """
        penalty function for violating the constraint. Returns 0 if constraint is satisfied, or a positive value
        if the constraint is violated.

        Shared across subclasses: each subclass declares the enforcement interval(s) it checks
        via :meth:`_penalty_intervals` (its "at" / "between" / "always" / "once" logic), and the
        penalty is the sum of :meth:`get_penalty` over them (the static or likelihood model).
        Factored from the per-subclass inline loops so :meth:`penalty_gradient` differentiates
        exactly the same intervals (#456/#385).

        :param sim_data_dict: Dictionary of the form {modelname: {suffix1: Data1}} containing the simulated data objects
        :type sim_data_dict: dict
        :param pset_values: The objective's live ``{name: value}`` map, so a likelihood constraint whose
            scale is tied to an estimated free parameter reads its current value.
            ``None`` (a bare diagnostic call) falls back to the authored literal scale.
        """
        return sum((self.get_penalty(sim_data_dict, pset_values=pset_values, **interval)
                    for interval in self._penalty_intervals(sim_data_dict)), 0.)

    def _penalty_intervals(self, sim_data_dict):
        """The enforcement interval(s) this constraint checks -- a list of keyword-argument dicts
        for :meth:`get_penalty` / :meth:`get_penalty_gradient` (``imin``, ``imax``, and optionally
        ``once`` / ``require_length`` / ``imin2`` / ``imax2``). Each subclass's "at" / "between" /
        "always" / "once" enforcement logic lives here, so ``penalty`` and ``penalty_gradient``
        evaluate identical intervals by construction; the data keys are resolved on first use (the
        lazy ``find_keys``)."""
        raise NotImplementedError('Subclasses of Constraint must override _penalty_intervals()')

    def _difference_argmax(self, sim_data_dict, imin, imax, once=False, require_length=None,
                           imin2=None, imax2=None):
        """The signed constraint difference ``max_i(q1_i - q2_i)`` (or ``min`` if ``once``) over
        the interval -- the same value :meth:`get_difference` returns -- plus the offset ``a`` of
        the achieving point within the sliced interval, which is where the gradient reads the
        readout's forward sensitivity (Danskin's theorem: the gradient of a max over a set is the
        gradient of the achieving term). The achieving rows in the full column are ``imin + a``
        for ``q1`` and ``imin2 + a`` for ``q2``."""
        if imin2 is None:
            imin2 = imin
        if imax2 is None:
            imax2 = imax
        q1 = self.index(sim_data_dict, self.qkeys1)[imin:imax] if isinstance(self.quant1, str) else self.quant1
        q2 = self.index(sim_data_dict, self.qkeys2)[imin2:imax2] if isinstance(self.quant2, str) else self.quant2
        diffs = np.atleast_1d(np.asarray(q1, dtype=float) - np.asarray(q2, dtype=float))
        a = int(np.argmin(diffs)) if once else int(np.argmax(diffs))
        return float(diffs[a]), a

    def _readout_gradient(self, raw_sens, n_param, keys1, quant1, row1, keys2, quant2, row2):
        """``d(q1 - q2)/d theta`` at the given rows: each operand's forward sensitivity row from
        ``raw_sens(model, suffix, observable, row)``, or 0 for a constant operand."""
        s1 = raw_sens(*keys1, row1) if isinstance(quant1, str) else np.zeros(n_param)
        s2 = raw_sens(*keys2, row2) if isinstance(quant2, str) else np.zeros(n_param)
        return s1 - s2

    def get_penalty_gradient(self, sim_data_dict, raw_sens, index, n_param, imin, imax,
                             once=False, require_length=None, imin2=None, imax2=None, pset_values=None):
        """The gradient of this interval's penalty w.r.t. the free parameters (#456/#385) --
        the constraint counterpart of the objective's per-point gradient. Dispatches to the
        static or likelihood model, mirroring :meth:`get_penalty`. Both reduce to ``local penalty
        slope * d(readout)/d theta`` at the interval's achieving point: the penalty is a piecewise
        (static) or Gaussian-CDF (likelihood) function of the at-/between-time readout ``q1 - q2``,
        so its gradient is that readout's forward sensitivity times the model's local slope.

        When this constraint's scale is tied to an estimated free parameter (``scale_param``), a closed-form ``d(penalty)/d(scale)`` term is added to that parameter's column --
        a scalar contribution needing no forward sensitivity, exactly like the estimated-noise
        ``d loss/d sigma`` column."""
        if self.penalty_model == 'static':
            return self._static_penalty_gradient(sim_data_dict, raw_sens, n_param,
                                                 imin, imax, once, imin2)
        elif self.penalty_model == 'likelihood':
            grad = self._likelihood_penalty_gradient(sim_data_dict, raw_sens, n_param,
                                                     imin, imax, once, require_length, imin2, imax2,
                                                     pset_values)
        elif self.penalty_model == 'logit':
            grad = self._logit_penalty_gradient(sim_data_dict, raw_sens, n_param,
                                                imin, imax, once, require_length, imin2, imax2,
                                                pset_values)
        else:
            raise ValueError(f'Invalid penalty model: {self.penalty_model}')

        if self.scale_param is not None:
            # Add d(penalty)/d(scale) to the tied parameter's column (scalar; no forward sensitivity).
            difference = self.get_difference(sim_data_dict, imin, imax, once, require_length, imin2, imax2)
            s = self._effective_scale(pset_values)
            grad = np.array(grad, dtype=float)
            grad[index[self.scale_param]] += self._scale_derivative(difference, s)
        return grad

    def _static_penalty_gradient(self, sim_data_dict, raw_sens, n_param, imin, imax, once, imin2):
        """Gradient of the static penalty ``weight * max(0, difference)`` (#456), mirroring
        :meth:`get_static_penalty` branch for branch. Zero in the satisfied region and in the flat
        ``min_penalty`` floor (the penalty is locally constant there); ``weight * d(readout)/d
        theta`` where the (alt) difference exceeds the floor. The kink at ``difference == 0`` (and
        at the floor) takes the subgradient 0 -- the satisfied-side value, documented like the
        Laplace kink (#454)."""
        difference, a = self._difference_argmax(sim_data_dict, imin, imax, once, imin2=imin2)
        if not (difference > 0 or (difference == 0. and not self.or_equal)):
            return np.zeros(n_param)   # constraint satisfied -> penalty 0, flat
        if self.alt1:
            alt_diff, alt_a = self._alt_difference_argmax(sim_data_dict, imin, imax, once)
            if max(0., alt_diff) <= self.min_penalty:
                return np.zeros(n_param)   # alt satisfied or floored at min_penalty -> flat
            row = imin + alt_a
            return self.weight * self._readout_gradient(
                raw_sens, n_param, self.akeys1, self.alt1, row, self.akeys2, self.alt2, row)
        if difference <= self.min_penalty:
            return np.zeros(n_param)   # floored at min_penalty -> flat
        imin2_eff = imin if imin2 is None else imin2
        return self.weight * self._readout_gradient(
            raw_sens, n_param, self.qkeys1, self.quant1, imin + a,
            self.qkeys2, self.quant2, imin2_eff + a)

    def _alt_difference_argmax(self, sim_data_dict, imin, imax, once):
        """The alt-penalty difference ``max/min(alt1 - alt2)`` over ``[imin:imax]`` and the
        achieving offset (mirroring :meth:`get_static_penalty`'s altpenalty branch)."""
        a1 = self.index(sim_data_dict, self.akeys1)[imin:imax] if isinstance(self.alt1, str) else self.alt1
        a2 = self.index(sim_data_dict, self.akeys2)[imin:imax] if isinstance(self.alt2, str) else self.alt2
        diffs = np.atleast_1d(np.asarray(a1, dtype=float) - np.asarray(a2, dtype=float))
        idx = int(np.argmin(diffs)) if once else int(np.argmax(diffs))
        return float(diffs[idx]), idx

    def _likelihood_penalty_gradient(self, sim_data_dict, raw_sens, n_param, imin, imax, once,
                                     require_length, imin2, imax2, pset_values=None):
        """Gradient of the likelihood penalty ``-log((pmax-pmin) * Phi(-difference/k) + pmin)``
        (#456). The Gaussian-CDF model is smooth, with local slope
        ``d/d difference = (pmax-pmin) * phi(-difference/k) / (k * adjusted_prob)`` (> 0: a larger
        difference is a worse violation, a larger penalty), times ``d(readout)/d theta``. A zero
        tolerance is a step function (flat almost everywhere) -> zero gradient, matching the
        non-differentiable edge case :meth:`get_log_likelihood` special-cases. ``k`` is the effective
        scale -- the estimated free parameter's live value if tied, else the
        authored tolerance."""
        difference, a = self._difference_argmax(sim_data_dict, imin, imax, once, require_length, imin2, imax2)
        if self._effective_scale(pset_values) == 0:
            return np.zeros(n_param)
        slope = self._smooth_slope(difference, pset_values)
        imin2_eff = imin if imin2 is None else imin2
        return slope * self._readout_gradient(
            raw_sens, n_param, self.qkeys1, self.quant1, imin + a,
            self.qkeys2, self.quant2, imin2_eff + a)

    def _logit_penalty_gradient(self, sim_data_dict, raw_sens, n_param, imin, imax, once,
                                require_length, imin2, imax2, pset_values=None):
        """Gradient of the logit softplus penalty ``ln(1 + e^{difference/s})``
        (:meth:`get_log_likelihood_logit`). The loss is smooth everywhere (``s > 0``), with local
        slope ``dF/d difference = sigma(difference/s) / s`` (the logistic; > 0, a larger difference
        is a worse violation), times ``d(readout)/d theta`` at the interval's achieving point --
        the same Danskin-argmax structure the probit gradient uses. With optional ``pmin/pmax``
        label smoothing the slope becomes
        ``(pmax-pmin) * sigma(-x) * sigma(x) / (s * adjusted_prob)`` with ``x = difference/s`` and
        ``adjusted_prob = pmin + (pmax-pmin)*sigma(-x)`` -- collapsing to ``sigma(x)/s`` when
        unclipped (``pmin=0, pmax=1``). ``s`` is the effective scale -- the estimated free
        parameter's live value if tied, else the authored scale."""
        difference, a = self._difference_argmax(sim_data_dict, imin, imax, once, require_length, imin2, imax2)
        slope = self._smooth_slope(difference, pset_values)
        imin2_eff = imin if imin2 is None else imin2
        return slope * self._readout_gradient(
            raw_sens, n_param, self.qkeys1, self.quant1, imin + a,
            self.qkeys2, self.quant2, imin2_eff + a)

    def _smooth_slope(self, difference, pset_values):
        """The smooth penalty's local slope ``P'(difference)`` -- the probit / logit factor the
        gradient multiplies ``d(readout)/d theta`` by (#456), extracted so it is a **pure scalar
        function of the readout ``difference``** that the EFIM curvature (#481) central-differences
        for ``P''`` (consistent-by-construction with the gradient). Reads the effective scale (an
        estimated free parameter's live value if tied, else the authored tolerance / logit scale)
        through :meth:`_effective_scale`. Returns 0 for a zero probit tolerance (the step-function
        edge the gradient special-cases). Only called for the smooth models (static has a constant
        slope handled inline)."""
        if self.penalty_model == 'likelihood':
            k = self._effective_scale(pset_values)
            if k == 0:
                return 0.0
            x = -difference / k
            phi = np.exp(-0.5 * x * x) / np.sqrt(2. * np.pi)
            prob = (1. + erf(x / np.sqrt(2.))) / 2.
            adjusted = (self.pmax - self.pmin) * prob + self.pmin
            return (self.pmax - self.pmin) * phi / (k * adjusted)
        # logit
        s = self._effective_scale(pset_values)
        x = difference / s
        if self.pmin is None:
            return _sigmoid(x) / s
        adjusted = self.pmin + (self.pmax - self.pmin) * _sigmoid(-x)
        return (self.pmax - self.pmin) * _sigmoid(-x) * _sigmoid(x) / (s * adjusted)

    def penalty_gradient(self, sim_data_dict, raw_sens, index, n_param, pset_values=None):
        """The gradient of this constraint's total penalty w.r.t. the free parameters (layer I,
        #456/#385) -- ``raw_sens(model, suffix, observable, row)`` supplies each readout's native-
        space forward sensitivity row (the #447 tensor, factor-folded per #448 routing), ``index``
        maps a free-parameter name to its column. Sums :meth:`get_penalty_gradient` over the same
        enforcement intervals :meth:`penalty` sums :meth:`get_penalty` over, so the gradient
        differentiates exactly the penalty PyBNF scores.

        Lives on the **scalar-gradient path only**: a penalty is a piecewise / CDF function, not a
        sum of squares, so (like an estimated noise normalizer, #451) it cannot enter the residual-
        Jacobian -- a fit with active constraints is not ``least_squares_exact``. The static model
        is non-smooth at the constraint boundary and at the ``min_penalty`` floor; the subgradient
        0 is taken there (documented like the Laplace kink, #454). ``pset_values`` carries the live
        parameter values so a constraint with an estimated scale contributes its
        ``d(penalty)/d(scale)`` column."""
        grad = np.zeros(n_param)
        for interval in self._penalty_intervals(sim_data_dict):
            grad += self.get_penalty_gradient(sim_data_dict, raw_sens, index, n_param,
                                              pset_values=pset_values, **interval)
        return grad

    def get_penalty_curvature(self, sim_data_dict, raw_sens, index, n_param, imin, imax,
                              once=False, require_length=None, imin2=None, imax2=None,
                              pset_values=None):
        """The **Gauss-Newton curvature** of this interval's penalty w.r.t. the free parameters --
        the constraint block of the EFIM trust-region Hessian (``job_type = gntr``, #481), the
        curvature twin of :meth:`get_penalty_gradient`. The penalty is ``P(q)`` for the at-/
        between-time readout ``q = q1 - q2``, so its Gauss-Newton curvature is
        ``max(P''(q), 0) * outer(grad q, grad q)`` -- the second-order readout sensitivity
        ``hess q`` dropped (as the EFIM drops it for the data fit) and ``P''`` clamped to its
        positive part (keeping the block PSD). Taken at the same achieving point (argmax) and with
        the same ``_readout_gradient`` the gradient uses (Danskin).

        * **static** (piecewise-linear hinge, ``.con``): ``P'' == 0`` -- a linear penalty carries no
          curvature, so the constraint contributes none here (its pull rides the gradient, and the
          data-fit Fisher + ridge supply the step's curvature). Returns zeros.
        * **smooth** (probit / logit ``.prop``): ``P''`` by a central finite difference of the
          analytic slope :meth:`_smooth_slope` (accurate to ~1e-9 on the smooth slope, and
          consistent-by-construction with the gradient's slope), clamped to ``>= 0``."""
        if self.penalty_model == 'static':
            return np.zeros((n_param, n_param))
        difference, a = self._difference_argmax(sim_data_dict, imin, imax, once, require_length,
                                                imin2, imax2)
        scale = self._effective_scale(pset_values)
        if scale == 0:
            return np.zeros((n_param, n_param))
        h = 1e-6 * max(abs(difference), scale)
        curv = (self._smooth_slope(difference + h, pset_values)
                - self._smooth_slope(difference - h, pset_values)) / (2. * h)
        curv = max(curv, 0.0)
        if curv == 0.0:
            return np.zeros((n_param, n_param))
        imin2_eff = imin if imin2 is None else imin2
        grad_q = self._readout_gradient(raw_sens, n_param, self.qkeys1, self.quant1, imin + a,
                                        self.qkeys2, self.quant2, imin2_eff + a)
        return curv * np.outer(grad_q, grad_q)

    def penalty_curvature(self, sim_data_dict, raw_sens, index, n_param, pset_values=None):
        """The Gauss-Newton curvature of this constraint's **total** penalty (summed over the same
        enforcement intervals :meth:`penalty_gradient` sums the gradient over) -- the constraint
        block the EFIM trust-region path (``job_type = gntr``) adds to the data-fit Hessian (#481),
        the curvature sibling of :meth:`penalty_gradient`.

        An **estimated constraint scale** (``scale_param``, ADR-0061) couples the readout to the
        scale (an off-diagonal Fisher block this cut does not assemble), so it is refused with a
        pointer to the scalar-gradient (L-BFGS-B) path -- on which the estimated-scale column still
        fits (it rides :meth:`get_penalty_gradient`). Returns zeros for an empty constraint set (a
        satisfied constraint, or an all-static one, adds no curvature)."""
        if self.scale_param is not None:
            from .gradient.errors import GradientNotSupported
            raise GradientNotSupported(
                "A constraint with an estimated scale ('%s') couples the readout and scale on the "
                "EFIM trust-region path (job_type = gntr, #481); use job_type = lbfgs." % self.scale_param)
        curv = np.zeros((n_param, n_param))
        for interval in self._penalty_intervals(sim_data_dict):
            curv += self.get_penalty_curvature(sim_data_dict, raw_sens, index, n_param,
                                               pset_values=pset_values, **interval)
        return curv


class AtConstraint(Constraint):
    def __init__(self, quant1, sign, quant2, base_model, base_suffix, weight, atvar, atval, altpenalty=None, minpenalty=0.,
                 repeat=False, before=False, pmin=None, pmax=None, tolerance=None, scale=None):
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
                         tolerance, scale)
        self.atvar = atvar
        self.atval = atval
        self.repeat = repeat
        self.before = before

        self.atkeys = None
        logger.debug(f"Created 'at' constraint {self.quant1}<{self.quant2}")

    def _find_extra_keys(self, sim_data_dict):
        """Resolve the 'at' condition variable, defaulting to the indvar."""
        self.atkeys = self._atvar_key(self.atvar, sim_data_dict)

    def _penalty_intervals(self, sim_data_dict):
        """Each "at" crossing of the condition variable is a single-row interval (the next/before
        row, per the equal/``before`` adjustments) -- one per crossing if ``repeat``, else the
        first only."""
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

        intervals = []
        for fi in flip_inds:
            # Make sure we pick the correct end of the interval if there's a point equal to the "at"
            if np.isclose(atdata[fi+1], self.atval, atol=0.) and not (np.isclose(atdata[fi], self.atval, atol=0.)):
                fi += 1
            # If constraint was declared with "before", go back 1.
            if self.before and fi > 0:
                fi -= 1
            intervals.append(dict(imin=fi, imax=fi+1, require_length=len(at_col)))
            # Todo - if atvar and quant1 were simulated on different time scales, need to scour independent variable cols

        return intervals

class SplitAtConstraint(Constraint):
    def __init__(self, quant1, atvar1, atval1, sign, quant2, atvar2, atval2, base_model, base_suffix,
                 weight, altpenalty=None, minpenalty=0.,
                 repeat=False, before1=False, before2=False, pmin=None, pmax=None, tolerance=None, scale=None):
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
                         tolerance, scale)

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

    def _penalty_intervals(self, sim_data_dict):
        """A single split interval: ``q1`` at its "at" crossing vs ``q2`` at its own "at"
        crossing (the ``imin2``/``imax2`` second readout). Empty if either condition is unmet."""
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
            return []

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
        # Todo - if atvar and quant1 were simulated on different time scales, need to scour independent variable cols
        return [dict(imin=fi1, imax=fi1+1, require_length=len(at_col1), imin2=fi2, imax2=fi2+1)]

class BetweenConstraint(Constraint):
    def __init__(self, quant1, sign, quant2, base_model, base_suffix, weight, startvar, startval, endvar, endval, altpenalty=None,
                 minpenalty=0., pmin=None, pmax=None, tolerance=None, once=False, scale=None):
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
                         tolerance, scale)

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

    def _penalty_intervals(self, sim_data_dict):
        """A single interval ``[start, end]`` bounded by the start/end condition crossings, over
        which the constraint is enforced at its worst point (or, if ``once``, its best)."""
        if not self.startkeys:
            self.find_keys(sim_data_dict)

        # Find the indices bounding the condition
        startdat = self.index(sim_data_dict, self.startkeys)
        startcol = startdat < self.startval
        start = np.nonzero(startcol[:-1] != startcol[1:])[0]
        if len(start) == 0:
            # Interval never started
            return []
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

        return [dict(imin=start, imax=end+1, require_length=len(endcol), once=self.once)]


class AlwaysConstraint(Constraint):
    def __init__(self, quant1, sign, quant2, base_model, base_suffix, weight, altpenalty=None, minpenalty=0.,
                 pmin=None, pmax=None, tolerance=None, scale=None):
        """
        Creates a new constraint of the form

        X1>X2 always
        """

        super().__init__(quant1, sign, quant2, base_model, base_suffix, weight, altpenalty, minpenalty, pmin, pmax,
                         tolerance, scale)
        logger.debug(f"Created 'always' constraint {self.quant1}<{self.quant2}")

    def _penalty_intervals(self, sim_data_dict):
        """The whole data column (worst miss over all time) is the single enforcement interval."""
        if not self.qkeys1 and not self.qkeys2:
            self.find_keys(sim_data_dict)
        # Todo - if q1 and q2 are on different time scales
        # Note: Indexing by 0:None takes the entire column
        return [dict(imin=0, imax=None)]


class OnceConstraint(Constraint):
    def __init__(self, quant1, sign, quant2, base_model, base_suffix, weight, altpenalty=None, minpenalty=0.,
                 pmin=None, pmax=None, tolerance=None, scale=None):
        """
        Creates a new constraint of the form

        X1>X2 once
        """

        super().__init__(quant1, sign, quant2, base_model, base_suffix, weight, altpenalty, minpenalty, pmin, pmax,
                         tolerance, scale)
        logger.debug(f"Created 'once' constraint {self.quant1}<{self.quant2}")

    def _penalty_intervals(self, sim_data_dict):
        """The whole data column is the interval, enforced ``once`` (the best point must satisfy)."""
        if not self.qkeys1 and not self.qkeys2:
            self.find_keys(sim_data_dict)
        # Note: Indexing by 0:None takes the entire column
        return [dict(imin=0, imax=None, once=True)]
