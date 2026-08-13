"""The record of which methods a run actually executed (``Results/method_chain.json``,
#564/ADR-0107).

A conf states a *requested* method chain -- ``job_type = cmaes`` with ``refine = 1,
refine_method = gntr`` requests "search globally with CMA-ES, then polish with GNTR".
What the run *executed* can be shorter: a phase is skipped when the wall-clock budget
is spent, when ``refine_method`` names the algorithm the fit itself just ran, or (for
bootstrap) when fewer replicates fit in the budget than were asked for. Until this
file, the only trace of that was a line on stdout -- so a harness that scores a
directory could not tell whether the method it believed it measured had run at all
(#564). ``Results/stop_reason.txt`` (ADR-0093) says the *run* stopped early; it does
not say which phases that cost.

The file is written after every phase, so it is on disk even if a later phase raises,
and it is written for every run, budget or no budget: provenance a consumer can only
rely on when a run went wrong is provenance it cannot assert on.

Deliberately dependency-free, like :mod:`pybnf.budget`: a phase log with a clock. The
policy that fills it in lives in :mod:`pybnf.pybnf`, which is where a run's phases are
sequenced in the first place.
"""

import json
import logging
import math
import time
from pathlib import Path

logger = logging.getLogger(__name__)


#: The phase ran and ended on its own terms (its stop criterion, its iteration cap,
#: an exhausted job pool). NOT a claim that it converged -- only that no outside
#: deadline cut it off.
COMPLETED = 'completed'

#: The phase ran but was ended by the wall-clock budget (``wall_time_fit``), so its
#: result is whatever it had reached when the clock ran out.
WALL_TIME_EXPIRED = 'wall_time_expired'

#: The phase was requested but never ran. ``reason`` says why.
SKIPPED = 'skipped'

#: The schema version of the emitted file. Bump on any incompatible change to the
#: object shape, so a consumer can refuse a file it does not understand rather than
#: silently misreading one.
FORMAT_VERSION = 1


def _jsonable(value):
    """A JSON-safe float, or ``None``. Non-finite objectives (``inf`` from a run whose
    every simulation failed) become ``null``: ``Infinity`` is not valid JSON, and a
    strict parser refusing the whole file would lose the rest of the record."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class MethodChain:
    """The ordered log of the phases a run executed, kept on disk as it grows.

    :param path: Where to write the JSON (``Results/method_chain.json``).
    :param requested: The chain the conf asked for, as method codes -- e.g.
        ``['cmaes', 'gntr']``. Recorded verbatim so a consumer can compare it with
        what ran without re-deriving it from the conf.
    :param context: Free-form scalars describing the run as a whole (the budget, the
        refine reserve, the PyBNF version). Merged into the top level of the file.
    :param clock: The monotonic clock each phase's elapsed time is measured on;
        injectable for tests.
    """

    def __init__(self, path, requested=(), context=None, clock=time.monotonic):
        self.path = str(path)
        self.requested = list(requested)
        self.context = dict(context or {})
        self.phases = []
        self._clock = clock
        self._phase_started_at = clock()

    def record(self, phase, method, status, reason=None, simulations=None,
               best_objective=None, bootstrap_replicate=None, extra=None):
        """Append one executed (or skipped) phase and rewrite the file.

        :param phase: The phase's role in the run -- ``'fit'``, ``'refine'``,
            ``'bootstrap'``. Distinct from ``method``, the algorithm code that filled
            it: ``cmaes`` is a fit here and a refine there.
        :param method: The ``fit_type`` / ``refine_method`` code that ran.
        :param status: :data:`COMPLETED`, :data:`WALL_TIME_EXPIRED`, or :data:`SKIPPED`.
        :param reason: The one-line human explanation -- an algorithm's stop reason, or
            why a phase was skipped. ``None`` when there is nothing to add to *status*.
        :param simulations: Simulations this phase completed, if it counted them.
        :param best_objective: The trajectory's best objective when the phase ended.
        :param bootstrap_replicate: The replicate this phase belongs to, or ``None`` for
            the main run. Replicates re-run the whole fit-and-refine chain, so they are
            recorded here but kept out of :meth:`executed_methods` -- "which methods did
            this run execute" is a question about the run, not about replicate 17.
        :param extra: Additional phase-specific fields (bootstrap's replicate counts).
        """
        now = self._clock()
        entry = {
            'phase': phase,
            'method': method,
            'status': status,
            'reason': reason,
            'elapsed_seconds': round(max(0.0, now - self._phase_started_at), 3),
            'simulations': None if simulations is None else int(simulations),
            'best_objective': _jsonable(best_objective),
            'bootstrap_replicate': bootstrap_replicate,
        }
        entry.update(extra or {})
        self._phase_started_at = now
        self.phases.append(entry)
        self.write()
        return entry

    def executed_methods(self):
        """The method codes the main run actually executed, in order -- the list to
        compare against ``requested_methods``. A shorter list is the silent downgrade
        this file exists to make loud."""
        return [p['method'] for p in self.phases
                if p['status'] != SKIPPED and p['bootstrap_replicate'] is None
                and p['phase'] in ('fit', 'refine')]

    def as_dict(self):
        """The whole record, ready for :func:`json.dump`."""
        doc = {'format_version': FORMAT_VERSION}
        doc.update(self.context)
        doc['requested_methods'] = list(self.requested)
        doc['executed_methods'] = self.executed_methods()
        doc['phases'] = list(self.phases)
        return doc

    def write(self):
        """Write the record. Every failure is logged and swallowed: a provenance file
        must never abort a run that has otherwise completed (the rule
        ``information_criteria.txt`` and ``stop_reason.txt`` already follow)."""
        try:
            with open(self.path, 'w') as f:
                json.dump(self.as_dict(), f, indent=2, sort_keys=False)
                f.write('\n')
        except Exception:
            logger.exception('Failed to write %s', self.path)
            return False
        logger.info('Wrote method chain %s', self.path)
        return True


def requested_methods(conf):
    """The method chain a configuration asks for, as a list of codes.

    ``['cmaes']`` for a plain fit; ``['cmaes', 'gntr']`` when ``refine = 1`` names a
    different ``refine_method``. A ``refine_method`` equal to ``fit_type`` is *not* a
    second entry -- there is nothing further for that algorithm to refine, and
    :func:`pybnf.pybnf._refine_best_fit` has always skipped it -- so the requested
    chain does not promise a phase the request could never produce.
    """
    fit_type = conf.get('fit_type')
    chain = [fit_type]
    if conf.get('refine') == 1:
        method = conf.get('refine_method', 'sim')
        if method != fit_type:
            chain.append(method)
    return chain


def chain_for_run(res_dir, conf, budget=None, version=None, clock=time.monotonic):
    """Build the :class:`MethodChain` for a run, from its config and budget."""
    context = {
        'pybnf_version': version,
        'job_type': conf.get('fit_type'),
        'wall_time_fit': int(conf.get('wall_time_fit') or 0),
        'refine_reserve_seconds': round(budget.reserve, 3) if budget is not None else 0.0,
    }
    return MethodChain(Path(res_dir) / 'method_chain.json',
                       requested=requested_methods(conf), context=context, clock=clock)
