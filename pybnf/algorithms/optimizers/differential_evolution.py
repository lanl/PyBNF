"""The Differential Evolution optimizer family (``de`` and ``ade`` fit types).

DifferentialEvolutionBase is the shared base; DifferentialEvolution (``de``) and
AsynchronousDifferentialEvolution (``ade``) subclass it. Extracted byte-identical
(M1 Step 4). The family makes no core.* call of its own — the run loop and the
execution seam are inherited from Algorithm.

With ``de_adapt_mutation = 1`` the family learns its two mutation settings during the
run from a success history (Tanabe and Fukunaga 2013; #667, ADR-0142) instead of using
``mutation_rate`` and ``mutation_factor`` as written: :class:`SuccessHistory` holds the
memory, the base draws each candidate's pair from it and records the outcome, and each
subclass says when a generation has ended.

With ``de_force_mutation`` on (the default under ``edition = 2``, and always with the
learned settings) no candidate is an exact copy of the parameter set it was built from
(#698, ADR-0143). A candidate is its base with some parameters moved by the donors'
difference, and nothing else guaranteed that any parameter moved. Under the default seed
policy a copy runs the same simulations as its base and ties it exactly, so it takes any
slot whose member is worse; in ``ade`` those copies became bases for more copies until
the whole population was one parameter set and the convergence test stopped the run.
Binomial crossover's answer is one parameter chosen in advance that is always mutated.
PyBNF crosses the mutant with the base rather than with the slot the candidate competes
for, so an accepted candidate shares its unmutated values with a member that stays in the
population, members come to share values, and the donors' difference is often zero in
the parameter chosen. The guarantee therefore chooses again among the parameters the
difference does move, and draws other donors when it moves none.

With ``de_cross_with_target = 1`` (off by default, under every edition) the mutant is
crossed with the member the candidate will replace, its target, as published differential
evolution does (#700, ADR-0144), instead of with its base: an unmutated parameter keeps the
target's own value. Crossed with the base, the values a winning candidate kept from its
base sat in two members, spread through the population, and froze any parameter whose
value every member came to share, since no donor difference can move it. Under ``all`` the
base is the target and nothing changes. The copy guarantee then moves one parameter to a
value neither the base nor the target holds, since a zero donor difference would leave the
base's value and a candidate made only of those would be a copy of its base; the learned
settings judge a success against the target.
"""


from ..base import Algorithm
from .multistart import MultiStartConfig, MultiStartOptimizer
from ... import edition
from ...config_schema import PyBNFConfigModel
from ...pset import PSet
from ...printing import print1, print2, PybnfError
from ...registry import register_fit_type

import logging
from typing import Optional

import numpy as np
from pydantic import Field
import re
import copy


# Preserve the original module logger name so log records keep the
# 'pybnf.algorithms' channel.
logger = logging.getLogger('pybnf.algorithms')


class SuccessHistory:
    """The success history of the mutation settings (Tanabe and Fukunaga 2013; #667,
    ADR-0142): a short memory of the ``mutation_rate`` / ``mutation_factor`` pairs that
    recently produced a candidate better than the member its mutant was crossed with, and
    the draw that turns that memory into the pair for the next candidate.

    The memory has ``size`` slots, each a (rate, factor) pair, all starting at the
    configured pair so a run that never records a success keeps drawing around the
    settings its author chose. A candidate's pair is drawn around a slot picked at
    random: the rate from a normal distribution with spread 0.1, clipped to [0, 1]; the
    factor from a Cauchy distribution with the same spread, capped at 1 and drawn again
    while it is not positive. The Cauchy tail is what lets a memory sitting at 0.5 still
    try a factor near 1 now and then, so the history can move once the search's needs
    change. Successes accumulate as they are reported and are folded into the memory one
    slot at a time when the method says a generation has ended: the rate as the mean of
    the successful rates weighted by how much each improved on its reference, the factor as
    the weighted Lehmer mean, which leans toward the larger factors because a small step
    succeeds more often but by less, and would otherwise pull the memory toward ever
    smaller steps. A generation with no success leaves the memory as it was.

    Plain lists and floats, so it rides the backup pickle with the rest of the optimizer
    (ADR-0007). ``de`` keeps one per island, ``ade`` keeps one.
    """

    #: The spread of a draw around the remembered pair, on both settings (SHADE's 0.1).
    SPREAD = 0.1

    def __init__(self, size, mutation_rate, mutation_factor):
        self.size = max(1, int(size))
        self.rates = [float(mutation_rate)] * self.size
        self.factors = [float(mutation_factor)] * self.size
        self.next_slot = 0
        self.pending = []   # (rate, factor, gain) of every success since the last flush

    def draw(self, rng):
        """The (rate, factor) pair for one new candidate, from the algorithm's ``rng``."""
        slot = int(rng.integers(self.size))
        rate = min(1.0, max(0.0, rng.normal(self.rates[slot], self.SPREAD)))
        factor = self.factors[slot] + self.SPREAD * rng.standard_cauchy()
        while factor <= 0.0:
            factor = self.factors[slot] + self.SPREAD * rng.standard_cauchy()
        return rate, min(1.0, factor)

    def record(self, rate, factor, gain):
        """A success: the pair built a candidate better than its reference by ``gain``,
        which the caller guarantees is positive and finite."""
        self.pending.append((float(rate), float(factor), float(gain)))

    def flush(self):
        """Fold the successes since the last flush into the next slot. Returns whether
        anything was folded (nothing is, and the memory is untouched, when no candidate
        succeeded)."""
        if not self.pending:
            return False
        rates, factors, gains = (np.asarray(col, dtype=float) for col in zip(*self.pending))
        weights = gains / gains.sum()
        self.rates[self.next_slot] = float(np.dot(weights, rates))
        self.factors[self.next_slot] = float(np.dot(weights, factors ** 2) / np.dot(weights, factors))
        self.next_slot = (self.next_slot + 1) % self.size
        self.pending = []
        return True

    def means(self):
        """The memory's mean (rate, factor), for reporting."""
        return float(np.mean(self.rates)), float(np.mean(self.factors))


class DEFamilyConfig(PyBNFConfigModel):
    """Config fields shared by the whole DE family, co-located with
    ``DifferentialEvolutionBase`` (ADR-0002, ADR-0006) -- exactly the keys that
    base ``__init__`` reads, and nothing more. Neither ``de`` nor ``ade`` registers
    against this base directly: each extends it with its own subclass carrying the
    shared ``n_starts`` multi-start field (``MultiStartConfig``) -- ``de`` also adds
    the island/migration fields (:class:`DifferentialEvolutionConfig`), ``ade`` adds
    nothing else (:class:`AsyncDEConfig`). Keeping the shared base itself key-minimal
    preserves the ADR-0006 "``ade`` adds no keys to the family base" seam. Values are
    byte-identical to the old ``GlobalConfig`` defaults.
    """

    mutation_rate: float = 0.5
    mutation_factor: float = 0.5
    stop_tolerance: float = 0.002
    # The DE-family convergence tolerance as an absolute range in OBJECTIVE units
    # (#561, ADR-0115): the population is converged when the spread of its finite
    # fitnesses, ``max - min``, has collapsed to within this value. ``stop_tolerance``
    # was a *ratio* (``max/min``), which only reads as convergence on a positive
    # objective bounded below by 0; ``de_tolfun`` measures the range directly, so it
    # is well-defined for a likelihood objective too. It is a range in objective units
    # where ``stop_tolerance`` was dimensionless, so it gets its own key -- and falls
    # back to ``stop_tolerance`` when unset (mirroring ``cmaes_tolfun`` -> ``cmaes_stop_tol``,
    # ADR-0106), so an existing config keeps the threshold magnitude it had.
    de_tolfun: Optional[float] = Field(default=None, ge=0.0)
    de_strategy: str = 'rand1'
    # Learn ``mutation_rate`` and ``mutation_factor`` during the run from a success
    # history (#667, ADR-0142) instead of using the pair above as written. Off by
    # default: an existing configuration runs exactly as it did. The configured pair is
    # where the learning starts.
    de_adapt_mutation: int = 0
    # How many generations' worth of successful settings the history remembers (the
    # memory size H of Tanabe and Fukunaga; L-SHADE's 6).
    de_adapt_memory: int = Field(default=6, ge=1)
    # Never propose a candidate that is an exact copy of its base (#698, ADR-0143): one
    # parameter the donors' difference moves is always mutated. Unset resolves to on under
    # edition 2 and off under the legacy edition, whose contract is that an unchanged conf
    # keeps behaving as it always has; the learned settings above turn it on regardless.
    de_force_mutation: Optional[int] = None
    # Cross the mutant with the member the candidate will replace, its target, as published
    # differential evolution does, rather than with its base (#700, ADR-0144): an unmutated
    # parameter keeps the target's own value, so values stop spreading from the base into
    # other members. Off under every edition: it removed the frozen parameters of the base
    # crossover on analytical targets, but did worse on two real tutorial models and no better
    # on the stochastic recovery benchmark.
    de_cross_with_target: int = 0


class DifferentialEvolutionConfig(MultiStartConfig, DEFamilyConfig):
    """``de``-specific config: the island/migration fields only synchronous DE
    reads (``ade`` is async and ignores them), plus the shared ``n_starts``
    multi-start field (``MultiStartConfig``, #498). The ``n_starts`` key rides each
    method's own subclass, not the shared ``DEFamilyConfig`` base, so the ADR-0006
    "``ade`` adds no keys to the family base" seam stays intact -- ``ade`` gets
    ``n_starts`` through its own :class:`AsyncDEConfig` (#501). Demonstrates the
    shared-base pattern the MCMC family reuses (ADR-0006)."""

    islands: int = 1
    migrate_every: int = 20
    num_to_migrate: int = 3


class AsyncDEConfig(MultiStartConfig, DEFamilyConfig):
    """``ade``-specific config: the shared DE family fields plus the ``n_starts``
    multi-start field (``MultiStartConfig``, #498/ADR-0071), and nothing else. ``ade``
    opts into multi-start through its own subclass -- mirroring how ``de`` extends the
    shared ``DEFamilyConfig`` base (:class:`DifferentialEvolutionConfig`) -- so the
    ADR-0006 "``ade`` adds no keys to the family base" seam stays intact: ``n_starts``
    rides this subclass, not the base ``de`` and ``ade`` share. Unlike ``de``, ``ade``
    has no islands or migrations, so this adds no fields of its own -- it is exactly
    ``DEFamilyConfig`` + ``n_starts`` (#501)."""


class DifferentialEvolutionBase(Algorithm):

    def __init__(self, config):
        super().__init__(config)

        self.mutation_rate = config.config['mutation_rate']
        self.mutation_factor = config.config['mutation_factor']
        self.max_iterations = config.config['max_iterations']
        self.stop_tolerance = config.config['stop_tolerance']
        # The convergence tolerance the run actually uses (#561, ADR-0115): an
        # absolute range in objective units. Unset ``de_tolfun`` falls back to
        # ``stop_tolerance`` -- the single knob the test used before -- so an existing
        # config keeps its threshold magnitude.
        #
        # Whether it was set MATTERS, not just what it is (#648, ADR-0127). An explicit
        # ``de_tolfun`` is a range the author chose in their own objective's units and is
        # always honoured as one. An unset one is the legacy ``stop_tolerance``, which
        # was a dimensionless RATIO, and reading a ratio's magnitude as a range is what
        # #648 is: on an objective of 2e-05 the ratio stops at a spread of 4e-08 while
        # the range stops at 0.002, roughly fifty thousand times looser.
        configured_tolfun = config.config['de_tolfun']
        self.de_tolfun_is_explicit = configured_tolfun is not None
        self.de_tolfun = (self.stop_tolerance if configured_tolfun is None
                          else float(configured_tolfun))

        self.strategy = config.config['de_strategy']
        options = ('rand1', 'rand2', 'best1', 'best2', 'all1', 'all2')
        if self.strategy not in options:
            raise PybnfError('Invalid differential evolution strategy "{}". Options are: {}'.format(self.strategy, ','.join(options)))

        # How many distinct members each candidate is built from, which is the floor on
        # the population (per island): the '1' strategies draw a base and one donor pair
        # (3), the '2' strategies a base and two donor pairs (5). new_individual draws
        # exactly this many with replace=False, so a smaller population makes that draw
        # raise (#708); the subclasses clamp population_size up to it and each is the
        # single source of truth for the other.
        self.min_population = 3 if '1' in self.strategy else 5

        # The learned mutation settings (#667, ADR-0142): whether to learn them, how much
        # to remember, one success history per island (``ade``: one), and the settings of
        # every candidate still in flight, keyed by the candidate, since ``ade`` returns
        # results in whatever order the simulations finish.
        self.adapt_mutation = bool(config.config['de_adapt_mutation'])
        self.adapt_memory = int(config.config['de_adapt_memory'])
        self.histories = []
        self._trial_settings = dict()

        # Whether a candidate always mutates a parameter its donors move, so it is never an
        # exact copy of its base (#698, ADR-0143).
        self.force_mutation = self._resolve_force_mutation()
        # Whether the mutant is crossed with the member the candidate will replace rather
        # than with its base (#700, ADR-0144); off unless asked for.
        self.cross_with_target = bool(config.config['de_cross_with_target'])

    def _resolve_force_mutation(self):
        """Whether the copy guarantee is on (#698, ADR-0143). The learned settings always
        need it, since a learned rate can sit near 0 where most candidates would otherwise
        be copies, and a copy can never be a success; otherwise an explicit
        ``de_force_mutation`` wins, and unset it is on under a modern edition and off under
        the legacy one, whose contract is that an unchanged conf keeps behaving as it always
        has (ADR-0031)."""
        configured = self.config.config.get('de_force_mutation')
        if self.adapt_mutation:
            if configured is not None and not int(configured):
                message = ('de_force_mutation = 0 is set, but de_adapt_mutation = 1 learns a '
                           'mutation rate that can sit near 0, where most candidates would be '
                           'copies of the parameter set they were built from; every candidate '
                           'still mutates at least one parameter')
                logger.warning(message)
                print1('Note: ' + message)
            return True
        if configured is not None:
            return bool(int(configured))
        return edition.is_modern(edition.resolve_edition(self.config.config.get('edition')))

    def new_individual(self, individuals, base_index=None, island=0, target_index=None,
                       name=None):
        """
        Create a new individual for the specified island, according to the set strategy

        :param individuals: The island's current population
        :param base_index: The index to use for the new individual, or None for a random index.
        :param island: Which island the individual is for. With ``de_adapt_mutation`` on, that
            island's success history supplies the mutation settings and learns from the outcome
            (``ade`` has a single island, 0).
        :param target_index: The index of the member the new individual will compete with. With
            ``de_cross_with_target`` on, its unmutated parameters are that member's; without an
            index, or with the key off, they are the base's.
        :param name: The candidate's name, assigned here rather than by the caller because it is
            the key its learned-mutation record is filed under, and the record is written below
            (#812). A caller that only wants the mutation arithmetic may leave it out; a caller
            putting the candidate in flight must not, since the name is its identity.
        :return:
        """

        # Choose a starting parameter set (either a random one or the base_index specified)
        # and others to cross over (always random). The number of distinct members a
        # candidate needs -- a base plus one donor pair ('1') or two ('2') -- is the same
        # count the population is floored at (see min_population), so the subclasses
        # guarantee this draw has enough to draw from (#708).
        pickn = self.min_population

        # Choose pickn random unique indices, or if base_index was given, choose base_index followed by pickn-1 unique
        # indices
        picks = self.rng.choice(len(individuals), pickn, replace=False)
        if base_index is not None:
            if base_index in picks:
                # If we accidentally picked base_index, replace it with picks[0], preserving uniqueness in our list
                iswitch = list(picks).index(base_index)
                picks[iswitch] = picks[0]
            # Now overwrite picks[0] with base_index. If we have base_index, picks[0] was an "extra pick" we only needed
            # in case we sampled base_index and had to replace it.
            picks[0] = base_index
        base = individuals[picks[0]]
        others = [individuals[p] for p in picks[1:]]

        # The member the mutant is crossed with, whose values the unmutated parameters keep:
        # the target the candidate will replace (#700, ADR-0144), or, as before, the base.
        crossed = picks[0]
        if self.cross_with_target and target_index is not None:
            crossed = target_index
        target = individuals[crossed]

        # The mutation settings for this candidate: the configured pair, or a draw from the
        # island's success history (#667, ADR-0142). Off, nothing here touches the rng.
        rate, factor = self.mutation_rate, self.mutation_factor
        if self.adapt_mutation:
            rate, factor = self.histories[island].draw(self.rng)

        # The copy guarantee (#698, ADR-0143): one parameter is always moved by the donors to a
        # value that neither the base nor the member it is crossed with holds, so the candidate
        # is an exact copy of neither, which under the default seed policy would tie it
        # exactly. It can change the donors. Off, nothing here touches the rng.
        forced, forced_value = None, None
        if self.force_mutation:
            others, forced, forced_value = self._forced_parameter(
                individuals, picks[0], base, target, others, factor)

        # Iterate through parameters; decide whether to mutate or leave the same.
        new_pset_vars = []
        for i, p in enumerate(base):
            if self.rng.random() < rate or i == forced:
                if i == forced:
                    new_pset_vars.append(forced_value)
                else:
                    new_pset_vars.append(p.add(self._difference(p.name, others, factor)))
            else:
                new_pset_vars.append(target.get_param(p.name))

        new_pset = PSet(new_pset_vars)
        new_pset.name = name
        if self.adapt_mutation:
            # What the outcome will be judged against: the fitness, now, of the member the
            # mutant was crossed with, which is what these settings were applied to (see
            # _note_trial_result).
            reference_fitness = float(self._island_fitnesses(island)[crossed])
            # Filed under the name, not the candidate. Keyed by the candidate, two that
            # landed on identical parameters were one key and the second write dropped the
            # first one's record -- which #775 answered by moving one of them off the
            # other, and #812 answers by giving the register an identity that does not
            # depend on where the candidate landed. Names being unique per in-flight
            # candidate, ``ade`` no longer drops a record either (#730).
            self._trial_settings[new_pset.name] = (rate, factor, reference_fitness, island)
        return new_pset

    def _difference(self, name, donors, factor):
        """How far the donors move parameter ``name``, in its sampling space: ``factor`` times
        the difference between the first two donors, plus the same for the second two under a
        ``2`` strategy."""
        step = factor * donors[0].get_param(name).diff(donors[1].get_param(name))
        if '1' not in self.strategy:
            step = step + factor * donors[2].get_param(name).diff(donors[3].get_param(name))
        return step

    def _moved(self, param, kept, donors, factor):
        """``param`` (the base's) moved by the donors' difference, or ``None`` when that is no
        move: when the difference does not change the base's value (it is zero, the usual
        case, settled before the moved parameter is built, since building it is the costly
        part; or too small to change the value in floating point), or when the moved value is
        ``kept``, the value the candidate keeps where it is not mutated. Crossed with the base,
        ``kept`` is the base's own value and the two tests are one. Crossed with the target
        they are two, and both matter: a zero difference would leave the base's value, so a
        candidate whose every mutated parameter had one would be an exact copy of its base."""
        step = self._difference(param.name, donors, factor)
        if step == 0.0:
            return None
        moved = param.add(step)
        if moved.value == param.value or moved.value == kept.value:
            return None
        return moved

    def _forced_parameter(self, individuals, base_pick, base, crossed, donors, factor):
        """The parameter a candidate always mutates, with the donors it is mutated by (#698,
        ADR-0143). ``base_pick`` is the base's position in ``individuals``, and ``crossed`` is
        the member the mutant is crossed with: the base, or the target (#700). Returns
        ``(donors, position, moved parameter)``.

        One parameter is chosen at random, the draw binomial crossover makes. If the donors'
        difference does not move it (see :meth:`_moved`: the donors share its value, or the
        moved value is the one the candidate keeps from ``crossed`` anyway), the choice is made
        again among the parameters it does move (drawn at random from those with a nonzero
        difference, dropping any the difference still does not move), so each parameter it
        moves is equally likely to be the one the candidate is sure to change. If it moves none
        (the donors are one parameter set, or their two differences cancel), other donors are
        drawn from the members other than the base, up to as many times as the population has
        members. If none of them moves anything either, the position is ``None`` and the
        candidate is left a copy of ``crossed``. That many failures in a row are likely only
        when all but one or two of the other members are one parameter set, so only a
        population that has all but collapsed can still propose a copy. Otherwise the
        candidate differs from its base and from ``crossed`` in the parameter moved.

        When the first parameter chosen changes, as every parameter does until members share
        values, this is exactly the draw the learned settings made before (#667), so such a
        run is unchanged until the first time the choice is made again.
        """
        params = list(base)
        kept = [crossed.get_param(p.name) for p in params]
        forced = int(self.rng.integers(len(params)))
        moved = self._moved(params[forced], kept[forced], donors, factor)
        if moved is not None:
            return donors, forced, moved
        pool = [k for k in range(len(individuals)) if k != base_pick]
        for attempt in range(len(individuals) + 1):
            if attempt:
                donors = [individuals[pool[k]]
                          for k in self.rng.choice(len(pool), len(donors), replace=False)]
            nonzero = [i for i, p in enumerate(params)
                       if self._difference(p.name, donors, factor) != 0.0]
            while nonzero:
                position = nonzero.pop(int(self.rng.integers(len(nonzero))))
                moved = self._moved(params[position], kept[position], donors, factor)
                if moved is not None:
                    return donors, position, moved
        return donors, None, None

    # --- the learned mutation settings (#667, ADR-0142) ------------------------- #
    def _island_fitnesses(self, island):
        """The fitness list of ``island``'s current population, parallel to the
        ``individuals`` list ``new_individual`` is given. Subclasses provide it."""
        raise NotImplementedError

    def _reset_adaptation(self, n_islands):
        """Start the success histories over, one per island at the configured pair, with no
        candidate in flight. Called wherever the search state resets, so each start of a
        multi-start run learns from its own population rather than inheriting the settings
        the previous start ended on, which suit a search that is finishing, not one that is
        beginning."""
        self.histories = [SuccessHistory(self.adapt_memory, self.mutation_rate, self.mutation_factor)
                          for _ in range(n_islands)]
        self._trial_settings = dict()

    def _note_trial_result(self, pset, score):
        """Report a finished candidate's score to the history that built it.

        A success is a candidate that scored strictly better, by a finite amount, than the
        member its mutant was crossed with, the member whose values it keeps where it was
        not mutated; the improvement is the success's weight when the history folds it in.
        That member is the reference because it is what the settings were applied to.
        Crossed with the target (#700, ADR-0144), as published differential evolution
        crosses, it is the member the candidate competes with, and this is SHADE's own rule.
        Crossed with the base, under the ``rand`` and ``best`` strategies it is not: judged
        against the slot, a candidate that copies a better base would win about half its
        contests by a wide margin without its settings having done anything, teaching the
        history that a rate near 0 is best (ADR-0142), so it is judged against the base.
        A failed simulation on either side is not evidence about the settings, since
        anything finite beats infinity, so it records nothing. A candidate this history did
        not build -- the initial population -- records nothing either; it is looked up by
        name, and its name is not one the register holds.

        A duplicate used to fall in that last group too: keyed by the candidate, two on
        identical parameters were one key and one record was lost (#730 for ``ade``, which
        tolerated it; #775 for ``de``, which moved one of them off the other to avoid it).
        The register is keyed by name now, so neither happens (#812).
        """
        record = self._trial_settings.pop(pset.name, None)
        if record is None:
            return
        rate, factor, reference_fitness, island = record
        if np.isfinite(reference_fitness) and np.isfinite(score) and score < reference_fitness:
            self.histories[island].record(rate, factor, reference_fitness - score)

    def _flush_adaptation(self, island):
        """A generation of ``island`` has ended: fold its successes into the history."""
        if self.adapt_mutation and self.histories:
            history = self.histories[island]
            n = len(history.pending)
            if history.flush():
                logger.debug('Island %d folded %d successful candidate(s) into its mutation history; '
                             'memory now averages rate %.3f, factor %.3f'
                             % (island, n, *history.means()))

    def _learned_settings(self):
        """The (rate, factor) the histories average to, over their slots and the islands,
        for the progress report; the configured pair before any history exists."""
        if not self.histories:
            return float(self.mutation_rate), float(self.mutation_factor)
        rates, factors = zip(*(history.means() for history in self.histories))
        return float(np.mean(rates)), float(np.mean(factors))

    def output_results(self, name='', no_move=False):
        """As the base does, and at the end of the run say what the mutation settings were
        learned to be, so a run at normal verbosity hears it too (the per-iteration line
        prints at verbosity 2 only) and the pair can be carried into a fixed-setting run."""
        super().output_results(name, no_move)
        if name == 'final' and self.adapt_mutation:
            rate, factor = self._learned_settings()
            message = ('Mutation settings learned by the end of the run: rate %.2f, factor %.2f '
                       '(the run started from mutation_rate %g, mutation_factor %g)'
                       % (rate, factor, self.mutation_rate, self.mutation_factor))
            logger.info(message)
            print1(message)

    def _population_converged(self):
        """The DE-family convergence test (#561, ADR-0115; #648, ADR-0127), shared by
        ``de`` and ``ade``.

        The population is converged when the spread of its objective values, ``max - min``
        over the **finite** fitnesses only, has collapsed to within the threshold. What
        that threshold *means* depends on where it came from, because the two sources
        carry different units (#648, ADR-0127):

        * An **explicit** ``de_tolfun`` is a range in objective units, chosen by someone
          who can see their own objective's scale. It is used exactly as written, on any
          sign of objective.
        * An **unset** ``de_tolfun`` falls back to ``stop_tolerance``, which has always
          been a **dimensionless ratio**. Where the objective is positive it is still read
          as one, as a spread relative to the best member: ``max - min <= tol * min``,
          which is the ``max / min <= 1 + tol`` this family used before #561, written
          without the division. Where the objective is not positive a ratio means nothing,
          so the same number is read as an absolute range, which is what #561 needs.

        ADR-0115 justified reading the legacy magnitude as a range by arguing that "an
        absolute range at the same magnitude is a stricter, well-defined stop". That holds
        only for an objective above 1. Below it the range is *looser*, without limit: at
        the 2e-05 a well-scaled sum-of-squares fit reaches, a 0.002 range is fifty thousand
        times looser than the 0.002 ratio it replaced, so the population satisfies it long
        before the parameters have separated. The fit stops early and reports an excellent
        objective at the wrong point (#648).

        What #561 actually found is that the ratio is a convergence statement only when
        the objective is positive and bounded below by 0 (a chi-square, an SSE). On a
        likelihood objective -- a negative log-likelihood, unbounded below -- an
        all-negative population lands the ratio in ``(0, 1]`` and it fires at generation 0
        regardless of the spread; and any single ``inf``-scored failed simulation, paired
        with a negative fitness, makes the ratio ``-inf`` and defeats *every* threshold.
        Both failure modes are the negative sign, not a real convergence, so the family
        was unrunnable on any estimated-sigma likelihood fit.

        ADR-0115 answered that by dropping the ratio everywhere. ADR-0127 narrows the
        answer to where the defect is: the ratio is kept where it is well defined, and the
        range is used where it is not. Neither reading is dropped, and no fit that worked
        under either one changes.

        Ignoring the non-finite entries is what makes a failed simulation unable to either
        satisfy or defeat the test, on both branches. Scaling by ``low`` rather than
        dividing by it subsumes the original ``!= 0`` guard against an all-zero population,
        which ``ade`` never had -- an all-zero population takes the range branch anyway,
        since ``low`` is not positive. An empty finite set (every member still ``inf``,
        e.g. before the first result) is never "converged".
        """
        finite = np.asarray(self.fitnesses, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not finite.size:
            return False
        low = finite.min()
        spread = finite.max() - low
        if self.de_tolfun_is_explicit or low <= 0.0:
            return bool(spread <= self.de_tolfun)
        # A positive objective under the legacy ratio: scale the threshold by the best
        # member rather than dividing by it, so an all-zero population cannot divide.
        return bool(spread <= self.stop_tolerance * low)

    def start_run(self):
        return NotImplementedError("start_run() not implemented in DifferentialEvolutionBase class")

    def got_result(self, res):
        return NotImplementedError("got_result() not implemented in DifferentialEvolutionBase class")


@register_fit_type('de', family='optimizer', display_name='Differential Evolution',
                   schema=DifferentialEvolutionConfig)
class DifferentialEvolution(MultiStartOptimizer, DifferentialEvolutionBase):
    """
    Implements the parallelized, island-based differential evolution algorithm
    described in Penas et al 2015.

    In some cases, I had to make my own decisions for specifics I couldn't find in the original paper. Namely:
    At each migration, a user-defined number of individuals are migrated from each island. For each individual, a
    random index is chosen; the same index for all islands. A random permutation is used to redistribute individuals
    with that index to different islands.

    Each island performs its migration individually, on the first callback when all islands are ready for that
    migration. It receives individuals from the migration iteration, regardless of what the current iteration is.
    This can sometimes lead to wasted effort.
    For example, suppose migration is set to occur at iteration 40, but island 1 has reached iteration 42 by the time
    all islands reach 40. Individual j on island 1 after iteration 42 gets replaced with individual j on island X
    after iteration 40. Some other island Y receives individual j on island 1 after iteration 40.

    """

    # This fit runs one generation at a time and waits for the whole generation to finish
    # before proposing the next, so some idle workers toward the end of each generation are
    # expected (#621). The asynchronous variant below does not wait, so it leaves this False.
    waits_for_full_generation = True

    def __init__(self, config):
        """
        Initializes algorithm based on the config object.

        The following config keys specify algorithm parameters. For move information, see config_documentation.txt
        population_size
        num_islands
        max_iterations
        mutation_rate
        mutation_factor
        de_adapt_mutation
        de_adapt_memory
        migrate_every
        num_to_migrate

        """
        super().__init__(config)

        self.num_islands = config.config['islands']
        self.num_per_island = int(config.config['population_size'] / self.num_islands)
        if self.num_per_island < self.min_population:
            self.num_per_island = self.min_population
            if self.num_islands == 1:
                print1('Differential evolution with de_strategy "%s" requires a population size of at least %i. '
                       'Increased the population size to %i.'
                       % (self.strategy, self.min_population, self.min_population))
                logger.warning('Increased population size to minimum allowed value of %i' % self.min_population)
            else:
                print1('Island-based differential evolution with de_strategy "%s" requires a population size of at '
                       'least %i times the number of islands. Increased the population size to %i.'
                       % (self.strategy, self.min_population, self.min_population * self.num_islands))
                logger.warning('Increased population size to minimum allowed value of %i per island' % self.min_population)
        if config.config['population_size'] % config.config['islands'] != 0:
            logger.warning('Reduced population_size to %i to evenly distribute it over %i islands' %
                            (self.num_islands * self.num_per_island, self.num_islands))
        self.migrate_every = config.config['migrate_every']
        if self.num_islands == 1:
            self.migrate_every = np.inf
        self.num_to_migrate = config.config['num_to_migrate']

        # Maps the NAME of each in-flight candidate to its location (island, individual_i).
        #
        # Keyed by name and not by the candidate, because a PSet hashes and compares by
        # value and two candidates on identical parameters are a legitimate state -- the
        # ordinary end state of a converging population, and likelier still under
        # ``de_adapt_mutation``, where a low drawn rate leaves most parameters unmutated.
        # Keyed by value the map could hold only one of them: the second registration
        # overwrote the first, a slot was credited with another slot's score, and since a
        # generation does not re-register a key until all of its results are in, the
        # duplicate's result had nowhere to go and ended the fit with ``KeyError`` (#812).
        # The proposal path answered that by perturbing one candidate off the other, in a
        # loop whose fixed 1e-6 step cannot move a linear parameter above ~1e10 at all, so
        # it never terminated (#721 in ``pso``, the same code here).
        #
        # A name is unique per in-flight candidate by construction and says which slot the
        # result belongs to, so each of two coincident candidates is scored into its own
        # slot. ``ade`` has always read its slot straight out of the name, which is why it
        # never had either failure; ``simplex``, ``scatter_search`` and (since #811)
        # ``pso`` key their in-flight registers the same way.
        self.island_map = dict()
        self.iter_num = [0] * self.num_islands  # Count the number of completed iterations on each island
        self.waiting_count = []  # Count of the number of PSets that are pending evaluation on the current iteration of each island.
        self.individuals = []  # Nested list; individuals[i][j] gives individual j on island i.
        self.proposed_individuals = []  # Nested list of the same shape, gives individuals proposed for replacement in next generation
        self.fitnesses = []  # Nested list of same shape, gives fitness of each individual
        self.migration_ready = [0] * self.num_islands  # What migration number is each island ready for
        self.migration_done = [0] * self.num_islands  # What migration number has each island completed

        # These variables store data related to individual migrations.
        # Each one has migration number as keys. When the first island starts migration, the required entries are
        # created. When the last island completes migration, they are deleted to keep these structures small.
        self.migration_transit = dict()  # Store (PSet, fitness) tuples here that are getting migrated - one list per island
        self.migration_indices = dict()  # Which individual numbers are migrating in migration i - a single tuple for
        # each migration, used for all islands
        self.migration_perms = dict()  # How do we rearrange between islands on migration i?
        # For each migration, a list of num_to_migrate permutations of range(num_islands)

    def reset(self, bootstrap=None):
        super().reset(bootstrap)
        self._reset_search_state()

    def _reset_search_state(self):
        """Clear all search-specific state (populations, per-island iteration and
        migration counters) WITHOUT touching the trajectory -- so multi-start's
        :meth:`_search_start_run` can begin a fresh, independent DE run each start while
        the trajectory keeps accumulating the global best across starts (#498). Shared by
        ``reset`` (which also resets the trajectory via ``super().reset``) and by each
        multi-start."""
        self.island_map = dict()
        self.iter_num = [0] * self.num_islands
        self.waiting_count = []
        self.individuals = []
        self.proposed_individuals = []
        self.fitnesses = []
        self.migration_ready = [0] * self.num_islands
        self.migration_done = [0] * self.num_islands

        self.migration_transit = dict()
        self.migration_indices = dict()
        self.migration_perms = dict()
        self._reset_adaptation(self.num_islands)

    def _island_fitnesses(self, island):
        return self.fitnesses[island]

    def _candidate_name(self, gen, island, j):
        """The name of the candidate in slot ``j`` of ``island`` at generation ``gen``.

        One place, because the name is now the key ``island_map`` and the learned-mutation
        register are filed under (#812), and two formats that drift apart would be two
        candidates the fit cannot tell apart. A single-island run keeps the island out of the
        name, so its simulation folders are named exactly as they always were."""
        if self.num_islands == 1:
            return 'gen%iind%i' % (gen, j)
        return 'gen%iisl%iind%i' % (gen, island, j)

    def _search_start_run(self):
        # Reset every search counter first (the per-island iteration and migration
        # state start_run itself does not touch), so a multi-start restart begins a
        # genuinely fresh DE run rather than resuming at the previous start's iteration.
        self._reset_search_state()
        if self.num_islands == 1:
            print2('Running Differential Evolution with population size %i for up to %i iterations' %
                   (self.num_per_island, self.max_iterations))
        else:
            print2('Running island-based Differential Evolution with %i islands of %i individuals each, '
                   'for up to %i iterations' % (self.num_islands, self.num_per_island, self.max_iterations))

        # Initialize random individuals
        if self.config.config['initialization'] == 'lh':
            psets = self.random_latin_hypercube_psets(self.num_islands*self.num_per_island)
            self.proposed_individuals = [psets[i * self.num_per_island: (i + 1) * self.num_per_island]
                                         for i in range(self.num_islands)]
        else:
            self.proposed_individuals = [[self.random_pset() for i in range(self.num_per_island)]
                                         for j in range(self.num_islands)]

        # ADR-0043 Phase 2: seed exactly one member of the initial population at the
        # initial_value point (a no-op unless a parameter: record declares one). One
        # member only -- the rest stay random so the islands keep their diversity.
        self.proposed_individuals[0][0] = self._seed_start_point_pset(self.proposed_individuals[0][0])

        # Initialize the individual list to empty, will be filled with the proposed_individuals once their fitnesses
        # are computed.
        self.individuals = [[None
                             for i in range(self.num_per_island)]
                            for j in range(self.num_islands)]

        # Set all fitnesses to Inf, guaranteeing a replacement by the first proposed individual
        self.fitnesses = [[np.inf
                           for i in range(self.num_per_island)]
                          for j in range(self.num_islands)]

        for i in range(len(self.proposed_individuals)):
            for j in range(len(self.proposed_individuals[i])):
                # Named before it is registered, the name being the key.
                self.proposed_individuals[i][j].name = self._candidate_name(0, i, j)
                self.island_map[self.proposed_individuals[i][j].name] = (i, j)

        self.waiting_count = [self.num_per_island] * self.num_islands

        return [ind for island in self.proposed_individuals for ind in island]

    def _search_got_result(self, res):
        """
        Called when a simulation run finishes

        This is not thread safe - the Scheduler must ensure only one process at a time enters
        this function.
        (or, I should rewrite this function to make it thread safe)

        :param res: Result object
        :return:
        """

        pset = res.pset
        score = res.score

        # Tell the success history how the candidate did, against the base it was built
        # from (a no-op unless de_adapt_mutation built it).
        self._note_trial_result(pset, score)

        # Calculate the fitness of this individual, and replace if it is better than the previous one.
        island, j = self.island_map.pop(pset.name)
        fitness = score
        if fitness <= self.fitnesses[island][j]:
            self.individuals[island][j] = pset
            self.fitnesses[island][j] = fitness

        self.waiting_count[island] -= 1

        # Determine if the current iteration is over for the current island
        if self.waiting_count[island] == 0:

            self.iter_num[island] += 1
            # The island's generation is over: fold its successes into its history before
            # the next generation draws from it.
            self._flush_adaptation(island)
            if min(self.iter_num) == self.iter_num[island]:
                # Last island to complete this iteration
                if self.iter_num[island] % self.config.config['output_every'] == 0:
                    self.output_results()
                if self.iter_num[island] % 10 == 0:
                    print1('Completed %i of %i iterations' % (self.iter_num[island], self.max_iterations))
                else:
                    print2('Completed %i of %i iterations' % (self.iter_num[island], self.max_iterations))
                print2('Current population fitnesses:')
                for l in self.fitnesses:
                    print2(sorted(l))
                if self.adapt_mutation:
                    print2('Mutation settings learned so far: rate %.2f, factor %.2f' % self._learned_settings())

            if self.iter_num[island] == self.max_iterations:
                # Submit no more jobs for this island
                # Once all islands reach this, simulation is over.
                if min(self.iter_num) == self.max_iterations:
                    return 'STOP'
                else:
                    return []

            if self.iter_num[island] % self.migrate_every == 0:
                # This island prepares for migration
                migration_num = int(self.iter_num[island] / self.migrate_every)
                if max(self.migration_ready) < migration_num:
                    # This is the first island to reach this migration.
                    # Need to set global parameters for this migration.
                    self.migration_transit[migration_num] = [list() for i in range(self.num_islands)]
                    self.migration_indices[migration_num] = self.rng.choice(self.num_per_island,
                                                                            size=self.num_to_migrate, replace=False)
                    self.migration_perms[migration_num] = [self.rng.permutation(self.num_islands)
                                                           for i in range(self.num_to_migrate)]
                    logger.debug('Island %i just set up the migration.' % island)

                # Send the required PSets to migration_transit
                for j in self.migration_indices[migration_num]:
                    self.migration_transit[migration_num][island].append((self.individuals[island][j],
                                                                          self.fitnesses[island][j]))
                # Tell other islands that this one is ready for this migration.
                self.migration_ready[island] = migration_num

            if self.migration_done[island] < min(self.migration_ready):
                # This island performs a migration
                logger.debug('Island %i is migrating!' % island)
                migration_num = self.migration_done[island] + 1

                # Fetch the appropriate new individuals from migration_transit
                for migrater_index in range(self.num_to_migrate):
                    j = self.migration_indices[migration_num][migrater_index]  # Index of the individual
                    newisland = self.migration_perms[migration_num][migrater_index][island]
                    self.individuals[island][j], self.fitnesses[island][j] = \
                        self.migration_transit[migration_num][newisland][migrater_index]

                    logger.debug('Island %i gained new individual with fitness %f' % (island, self.fitnesses[island][j]))

                self.migration_done[island] = migration_num
                if min(self.migration_done) == migration_num:
                    # This is the last island to complete this migration
                    # Delete the migration data to free space.
                    del self.migration_transit[migration_num]
                    del self.migration_perms[migration_num]
                    del self.migration_indices[migration_num]

            # Set up the next generation
            best = np.argmin(self.fitnesses[island])
            for jj in range(self.num_per_island):
                name = self._candidate_name(self.iter_num[island], island, jj)
                if 'best' in self.strategy:
                    new_pset = self.new_individual(self.individuals[island], best, island=island,
                                                   target_index=jj, name=name)
                elif 'all' in self.strategy:
                    new_pset = self.new_individual(self.individuals[island], jj, island=island,
                                                   target_index=jj, name=name)
                else:
                    new_pset = self.new_individual(self.individuals[island], island=island,
                                                   target_index=jj, name=name)
                # new_individual named it, so the register below and the learned-mutation
                # record it wrote are filed under the same key.
                self.proposed_individuals[island][jj] = new_pset
                self.island_map[new_pset.name] = (island, jj)

            self.waiting_count[island] = self.num_per_island

            if self.iter_num[island] % 20 == 0:
                logger.debug('Island %i completed %i iterations' % (island, self.iter_num[island]))
                # print(sorted(self.fitnesses[island]))

            # Convergence check: absolute range of the finite population (#561, ADR-0115).
            # Only assessed once EVERY island has completed at least one iteration
            # (``min(self.iter_num) >= 1``): until then some islands are still at their
            # initial ``inf`` sentinel, and since the test (correctly) ignores non-finite
            # fitnesses, a single finished island's collapsed subpopulation could
            # otherwise stop the whole run before the others have searched at all. The
            # old ratio test got this for free -- an unevaluated island's ``inf`` made the
            # global ``max`` infinite -- but that same coupling is what let a failed sim
            # defeat the test; the gate restores the "wait for the whole population" half
            # without the sign/failed-sim half. (For a single island this holds from the
            # first iteration on, so single-island DE is unaffected.)
            if min(self.iter_num) >= 1 and self._population_converged():
                return 'STOP'

            # Return a copy, so our internal data structure is not tampered with.
            return copy.copy(self.proposed_individuals[island])

        else:
            # Add no new jobs, wait for this generation to complete.
            return []


@register_fit_type('ade', family='optimizer', display_name='Asynchronous Differential Evolution',
                   schema=AsyncDEConfig)
class AsynchronousDifferentialEvolution(MultiStartOptimizer, DifferentialEvolutionBase):
    """
    Implements a simple asynchronous differential evolution algorithm.

    Contains no islands or migrations. Instead, each time a PSet finishes, proposes a new PSet at the same index using
    the standard DE formula and whatever the current population happens to be at the time.

    Opts into ``n_starts`` sequential-restart multi-start (#498/#501): the
    :class:`~pybnf.algorithms.optimizers.multistart.MultiStartOptimizer` mixin (before
    the family base in the MRO) runs ``n_starts`` independent searches and keeps the
    global best. ``ade`` is the async one-in-one-out case the mixin's *draining* path
    was built for -- a full population stays in flight at each inner ``STOP``, so the
    mixin drains those stragglers (already scored into the trajectory) before seeding
    the next start. ``n_starts == 1`` (the default) is byte-identical to the single-run
    behavior.
    """

    def __init__(self, config):
        """
        Initializes algorithm based on the config object.

        """
        super().__init__(config)

        self.population_size = config.config['population_size']
        if self.population_size < self.min_population:
            self.population_size = self.min_population
            self.config.config['population_size'] = self.min_population
            print1('Asynchronous differential evolution with de_strategy "%s" requires a population size of at least '
                   '%i. Increasing the population size to %i.'
                   % (self.strategy, self.min_population, self.min_population))
            logger.warning('Increased population_size to the minimum allowed value of %i' % self.min_population)

        self.sims_completed = 0
        self.individuals = []  # List of individuals
        self.fitnesses = []  # List of same shape, gives fitness of each individual

    def reset(self, bootstrap=None):
        super().reset(bootstrap)
        self._reset_search_state()

    def _reset_search_state(self):
        """Clear the search-specific state (the population, its fitnesses, and the
        completed-sim counter) WITHOUT touching the trajectory -- so multi-start's
        :meth:`_search_start_run` can begin a fresh, independent ADE run each start
        while the trajectory keeps accumulating the global best across starts (#498).
        Shared by ``reset`` (which also resets the trajectory via ``super().reset``) and
        by each multi-start."""
        self.sims_completed = 0
        self.individuals = []
        self.fitnesses = []
        self._reset_adaptation(1)

    def _island_fitnesses(self, island):
        return self.fitnesses

    def _search_start_run(self):
        # Reset the search counter/population first (a no-op on the first start), so a
        # multi-start restart begins a genuinely fresh ADE run rather than resuming at
        # the previous start's population and sims_completed count.
        self._reset_search_state()
        print2('Running Asyncrhonous Differential Evolution with population size %i for up to %i iterations' %
               (self.population_size, self.max_iterations))

        # Initialize random individuals
        if self.config.config['initialization'] == 'lh':
            self.individuals = self.random_latin_hypercube_psets(self.population_size)
        else:
            self.individuals = [self.random_pset() for i in range(self.population_size)]

        # ADR-0043 Phase 2: seed exactly one member at the initial_value point (a no-op
        # unless a parameter: record declares one); the rest stay random for diversity.
        self.individuals[0] = self._seed_start_point_pset(self.individuals[0])

        # Set all fitnesses to Inf, guaranteeing a replacement by the first proposed individual.
        # The first replacement will replace with a copy of the same PSet, with the correct objective calculated.
        self.fitnesses = [np.inf for i in range(self.population_size)]

        for i in range(len(self.individuals)):
            self.individuals[i].name = 'gen0ind%i' % i

        return copy.deepcopy(self.individuals)

    def _search_got_result(self, res):
        """
        Called when a simulation run finishes

        :param res: Result object
        :return:
        """

        pset = res.pset
        fitness = res.score

        # Tell the success history how the candidate did, against the base it was built
        # from (a no-op unless de_adapt_mutation built it). The record travelled with the
        # candidate, so it does not matter that results come back in any order.
        self._note_trial_result(pset, fitness)

        gen = int(re.search(r'(?<=gen)\d+', pset.name).group(0))
        j = int(re.search(r'(?<=ind)\d+', pset.name).group(0))

        if fitness <= self.fitnesses[j]:
            self.individuals[j] = pset
            self.fitnesses[j] = fitness

        self.sims_completed += 1

        # Do various "per iteration" stuff
        if self.sims_completed % self.population_size == 0:
            iters_complete = self.sims_completed / self.population_size
            # A population's worth of results is this method's generation: fold the
            # successes among them into the history before the next candidate draws from it.
            self._flush_adaptation(0)
            if iters_complete % self.config.config['output_every'] == 0:
                self.output_results()
            if iters_complete % 10 == 0:
                print1('Completed %i of %i simulations' % (self.sims_completed, self.max_iterations * self.population_size))
            else:
                print2('Completed %i of %i simulations' % (self.sims_completed, self.max_iterations * self.population_size))
            print2('Current population fitnesses:')
            print2(sorted(self.fitnesses))
            if self.adapt_mutation:
                print2('Mutation settings learned so far: rate %.2f, factor %.2f' % self._learned_settings())
            if iters_complete % 20 == 0:
                logger.debug('Completed %i simulations' % self.sims_completed)
            if iters_complete >= self.max_iterations:
                return 'STOP'
            # Convergence check: absolute range of the finite population (#561, ADR-0115).
            if self._population_converged():
                return 'STOP'

        # Named as it is built, so the learned-mutation record new_individual writes is
        # filed under the name this method reads back off the result (#812).
        name = 'gen%iind%i' % (gen+1, j)
        if 'best' in self.strategy:
            best = np.argmin(self.fitnesses)
            new_pset = self.new_individual(self.individuals, best, target_index=j, name=name)
        elif 'all' in self.strategy:
            new_pset = self.new_individual(self.individuals, j, target_index=j, name=name)
        else:
            new_pset = self.new_individual(self.individuals, target_index=j, name=name)

        return [new_pset]
