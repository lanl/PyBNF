"""Problem definitions and scoring rules for the stochastic recovery benchmark.

Pure Python: nothing here imports PyBNF or a simulator, so the frozen problem
definitions can be loaded and a result file scored anywhere.

A **problem** is a directory holding ``problem.json`` (the frozen definition, see
:class:`Problem`), the model it names, and the data file it names. Everything a fit
needs to be repeatable two years from now is in the JSON or committed next to it: the
true parameter values, the search bounds, the simulation method and sampling times,
the observables, how many replicates the data averages and the seed they were drawn
with, and the simulation budget a fit is allowed.

A **fit record** is what one fit of one problem by one method from one seed produced
(see :func:`score_fit`): the estimate, its distance from the truth, how many
simulations it spent, and the trace of its best-so-far answer against simulations
spent, from which the simulations-to-success statistic is read.

Scoring rules
-------------

* **Distance.** Every free parameter is searched on a log scale, so the error of an
  estimate is ``|log10(estimate / true)|`` in decades, per parameter. A fit's error is
  the largest of these over the parameters the definition marks identifiable; a
  parameter marked not identifiable is reported but never scored, because no method
  can be expected to recover it from these data.
* **Success.** A fit succeeds at the loose tolerance when every identifiable parameter
  is within a factor of two of its true value (0.301 decades), and at the tight
  tolerance when every one is within 26 percent (0.1 decades). The loose tolerance is
  the headline: the data are replicate means of a stochastic process and the fit's own
  objective is a noisy estimate, so a factor of two is what "found it" means here.
* **Cost.** The currency is simulations, not evaluations, because a method that runs
  more replicates per parameter set is spending more, and every simulation the fit
  runs counts: the search, the end-of-fit confirmation that decides the answer, and
  the replicates PyBNF runs to report information criteria. Simulations-to-success is
  the number spent when the fit's reported best parameter set first came within the
  loose tolerance; it is read off the trace and reported only for fits whose final
  answer is within it.
* **Repetition.** Both the fitting method and the simulator are random, so every
  (problem, method) pair is run from several fit seeds and the success rate over those
  seeds is the primary statistic; the median error and the median simulations to
  success are reported next to it.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

#: Version of the problem-definition format this module reads.
FORMAT_VERSION = 1

#: Success tolerances in decades of log10 parameter error.
TOL_LOOSE = math.log10(2.0)   # within a factor of two
TOL_TIGHT = 0.1               # within 26 percent


@dataclass(frozen=True)
class FreeParameter:
    """One free parameter of a problem: its true value, its search bounds, and whether
    the data are expected to determine it."""
    name: str
    true: float
    low: float
    high: float
    identifiable: bool = True
    note: str = ''


@dataclass(frozen=True)
class Problem:
    """A frozen benchmark problem, read from ``problem.json``."""
    id: str
    version: int
    title: str
    reference: str
    source: str
    model: str                    # model file, relative to the problem directory
    method: str                   # simulate method: ssa or nf
    suffix: str                   # the simulate action's suffix (= data basename)
    t_start: float
    t_end: float
    n_steps: int
    observables: tuple            # the data columns (each also has a <name>_SD column)
    parameters: tuple             # FreeParameter, in a fixed order
    data_file: str                # relative to the problem directory
    data_replicates: int          # trajectories averaged into the data
    data_seed_offset: int         # replicate-index offset the data were drawn at
    sd_floor_fraction: float      # sigma floor as a fraction of an observable's peak mean
    budget_simulations: int       # simulations one fit may spend
    smoothing: int                # replicates per evaluation the baseline methods use
    notes: str = ''
    directory: Path = field(default=None, compare=False, repr=False)

    # -- convenience -----------------------------------------------------------
    @property
    def model_path(self) -> Path:
        return self.directory / self.model

    @property
    def data_path(self) -> Path:
        return self.directory / self.data_file

    @property
    def names(self):
        return [p.name for p in self.parameters]

    @property
    def truth(self):
        return {p.name: p.true for p in self.parameters}

    @property
    def identifiable(self):
        return [p.name for p in self.parameters if p.identifiable]

    @property
    def sample_times(self):
        step = (self.t_end - self.t_start) / self.n_steps
        return [self.t_start + i * step for i in range(self.n_steps + 1)]


def load_problem(directory) -> Problem:
    """Read one problem directory's ``problem.json``."""
    directory = Path(directory)
    raw = json.loads((directory / 'problem.json').read_text())
    if raw.get('version') != FORMAT_VERSION:
        raise ValueError('%s: problem format version %r, expected %d'
                         % (directory, raw.get('version'), FORMAT_VERSION))
    params = tuple(FreeParameter(name=name, true=float(spec['true']), low=float(spec['low']),
                                 high=float(spec['high']),
                                 identifiable=bool(spec.get('identifiable', True)),
                                 note=spec.get('note', ''))
                   for name, spec in raw['parameters'].items())
    sim = raw['simulation']
    data = raw['data']
    fit = raw['fit']
    return Problem(
        id=raw['id'], version=raw['version'], title=raw['title'], reference=raw['reference'],
        source=raw['source'], model=raw['model'], method=sim['method'], suffix=sim['suffix'],
        t_start=float(sim['t_start']), t_end=float(sim['t_end']), n_steps=int(sim['n_steps']),
        observables=tuple(raw['observables']), parameters=params,
        data_file=data['file'], data_replicates=int(data['replicates']),
        data_seed_offset=int(data['seed_offset']), sd_floor_fraction=float(data['sd_floor_fraction']),
        budget_simulations=int(fit['budget_simulations']), smoothing=int(fit['smoothing']),
        notes=raw.get('notes', ''), directory=directory,
    )


def problems_root() -> Path:
    return Path(__file__).resolve().parent / 'problems'


def load_problems(root=None, ids=None):
    """Every problem under ``root`` (default: the bundled ``problems/``), in id order,
    or just the ones named in ``ids``."""
    root = Path(root) if root is not None else problems_root()
    found = {}
    for d in sorted(root.iterdir()):
        if (d / 'problem.json').is_file():
            p = load_problem(d)
            found[p.id] = p
    if ids is None:
        return list(found.values())
    missing = [i for i in ids if i not in found]
    if missing:
        raise KeyError('unknown problem id(s): %s (have %s)' % (missing, sorted(found)))
    return [found[i] for i in ids]


# --------------------------------------------------------------------------- #
# Scoring one fit
# --------------------------------------------------------------------------- #
def log10_errors(estimate, truth):
    """Per-parameter ``|log10(estimate / true)|`` in decades. A missing, non-positive,
    or non-finite estimate scores as infinite error."""
    out = {}
    for name, true in truth.items():
        value = estimate.get(name)
        try:
            ok = value is not None and math.isfinite(value) and value > 0 and true > 0
        except TypeError:
            ok = False
        out[name] = abs(math.log10(value / true)) if ok else math.inf
    return out


def max_error(errors, identifiable):
    """The largest per-parameter error over the identifiable parameters (inf if any is
    missing; 0.0 if nothing is identifiable, which a definition should never say)."""
    if not identifiable:
        return 0.0
    return max(errors.get(name, math.inf) for name in identifiable)


def rms_error(errors, identifiable):
    if not identifiable:
        return 0.0
    vals = [errors.get(name, math.inf) for name in identifiable]
    if any(math.isinf(v) for v in vals):
        return math.inf
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def simulations_to_success(trace, identifiable, tol=TOL_LOOSE):
    """The simulation count at which the fit's reported best parameter set first came
    within ``tol`` decades of the truth on every identifiable parameter, read from
    ``trace`` -- a list of ``(simulations_spent, {name: error})`` pairs in the order
    they were recorded -- or None if it never did."""
    for sims, errors in trace:
        if max_error(errors, identifiable) <= tol:
            return sims
    return None


def score_fit(problem: Problem, estimate, simulations, trace, **extra):
    """Build a fit record: the estimate and everything scored from it.

    :param estimate: ``{name: value}`` the fit reported as its answer
    :param simulations: total simulations the fit spent, confirmation stage included
    :param trace: ``[(simulations_spent, {name: error}), ...]`` as the fit progressed:
        the per-parameter errors of the reported best each time it changed. Kept per
        parameter so a record can be scored again (:func:`rescore`) should a
        definition's identifiability flags be revised.
    :param extra: anything else worth keeping (method, seed, wall time, ...)
    """
    errors = log10_errors(estimate, problem.truth)
    record = {
        'problem': problem.id,
        'estimate': {k: estimate.get(k) for k in problem.names},
        'errors': errors,
        'simulations': int(simulations),
        'trace': [(int(s), {k: (None if math.isinf(v) else float(v)) for k, v in e.items()})
                  for s, e in trace],
    }
    record.update(extra)
    _score(record, problem)
    return record


def _score(record, problem):
    """Fill (or refresh) the fields scored from a record's errors and trace."""
    ident = problem.identifiable
    errors = {k: (math.inf if v is None else v) for k, v in record['errors'].items()}
    trace = [(s, {k: (math.inf if v is None else v) for k, v in e.items()}) for s, e in record['trace']]
    err_max = max_error(errors, ident)
    success = bool(err_max <= TOL_LOOSE)
    first_within = simulations_to_success(trace, ident, TOL_LOOSE)
    record['max_error'] = err_max
    record['rms_error'] = rms_error(errors, ident)
    record['success_loose'] = success
    record['success_tight'] = bool(err_max <= TOL_TIGHT)
    # Reported only for a fit whose final answer is within the loose tolerance: the
    # reported best can enter the tolerance and leave it again (the end-of-fit
    # confirmation may pick another candidate), and a cost-to-success for a fit that did
    # not succeed would mislead. ``first_within_loose`` keeps the raw crossing.
    record['simulations_to_success'] = first_within if success else None
    record['first_within_loose'] = first_within
    return record


def rescore(records, problems):
    """Score ``records`` again under the given problem definitions (a list or
    ``{id: Problem}``), for a results file whose definitions have since changed their
    identifiability flags. Returns the records, modified in place."""
    if not isinstance(problems, dict):
        problems = {p.id: p for p in problems}
    for r in records:
        _score(r, problems[r['problem']])
    return records


# --------------------------------------------------------------------------- #
# Aggregating over seeds
# --------------------------------------------------------------------------- #
def _median(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isinf(v))]
    return statistics.median(values) if values else None


def aggregate(records):
    """Per (problem, method) summary rows over the seeds each pair was run from."""
    groups = {}
    for r in records:
        groups.setdefault((r['problem'], r.get('method', '?')), []).append(r)
    rows = []
    for (problem, method), recs in sorted(groups.items()):
        n = len(recs)
        rows.append({
            'problem': problem, 'method': method, 'n_seeds': n,
            'success_loose': sum(r['success_loose'] for r in recs) / n,
            'success_tight': sum(r['success_tight'] for r in recs) / n,
            'median_max_error': _median([r['max_error'] for r in recs]),
            'median_simulations_to_success': _median([r['simulations_to_success'] for r in recs
                                                      if r['success_loose']]),
            'mean_simulations': sum(r['simulations'] for r in recs) / n,
            'mean_wall_time': (sum(r['wall_time'] for r in recs) / n
                               if all('wall_time' in r for r in recs) else None),
        })
    return rows


def format_table(rows):
    """The summary rows as a Markdown table."""
    head = ('| problem | method | seeds | success (factor 2) | success (26%) | median max error (decades) '
            '| median sims to success | mean sims | mean wall s |')
    sep = '|---|---|---:|---:|---:|---:|---:|---:|---:|'
    lines = [head, sep]

    def fmt(v, spec):
        return '-' if v is None else format(v, spec)
    for r in rows:
        lines.append('| %s | %s | %d | %s | %s | %s | %s | %s | %s |' % (
            r['problem'], r['method'], r['n_seeds'],
            fmt(r['success_loose'], '.0%'), fmt(r['success_tight'], '.0%'),
            fmt(r['median_max_error'], '.3f'),
            fmt(r['median_simulations_to_success'], '.0f'),
            fmt(r['mean_simulations'], '.0f'), fmt(r['mean_wall_time'], '.0f')))
    return '\n'.join(lines)


# --------------------------------------------------------------------------- #
# Comparing two methods over the seeds they share
# --------------------------------------------------------------------------- #
#: The fractions of a problem's budget the progress table reports by default. On a
#: 20,000-simulation problem these are 2,000 / 5,000 / 10,000 / 15,000 / 20,000, the
#: checkpoints the #660 study read its progress curves at.
DEFAULT_CHECKPOINT_FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)


def binomial_p_two_sided(k, n):
    """The two-sided exact binomial p-value for ``k`` successes of ``n`` at even odds.

    Both paired tests here are this test on different counts: the sign test asks whether
    a method beats another on more than half of the seeds where they differ, and
    McNemar's exact test asks the same of the seeds where exactly one of them succeeded.
    At even odds the distribution is symmetric, so doubling the smaller tail is exact
    rather than an approximation. ``None`` when nothing differs.
    """
    if n <= 0:
        return None
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def sign_test(pairs):
    """Counts and a p-value for paired values where **lower is better**.

    ``pairs`` is a list of ``(a, b)``; ``better`` counts the pairs where ``b`` (the
    method under test) is below ``a`` (the baseline). Pairs that are equal -- including
    two failures, both infinite -- are ties and are excluded from the test, as a sign
    test must: they carry no direction.
    """
    better = sum(1 for a, b in pairs if b < a)
    worse = sum(1 for a, b in pairs if b > a)
    ties = len(pairs) - better - worse
    return {'better': better, 'worse': worse, 'ties': ties, 'n': len(pairs),
            'p': binomial_p_two_sided(better, better + worse)}


def mcnemar(pairs):
    """McNemar's exact test on paired successes: ``pairs`` is a list of ``(a, b)``
    booleans, and the test is on the seeds exactly one of them succeeded at."""
    only_b = sum(1 for a, b in pairs if b and not a)
    only_a = sum(1 for a, b in pairs if a and not b)
    return {'only_b': only_b, 'only_a': only_a, 'both': sum(1 for a, b in pairs if a and b),
            'neither': sum(1 for a, b in pairs if not a and not b),
            'p': binomial_p_two_sided(only_b, only_a + only_b)}


def best_so_far(trace, identifiable, checkpoint):
    """The error of the fit's reported best at ``checkpoint`` simulations: the last
    trace entry recorded at or before it, or ``None`` if the fit had reported nothing
    by then."""
    value = None
    for sims, errors in trace:
        if sims > checkpoint:
            break
        value = max_error({k: (math.inf if v is None else v) for k, v in errors.items()},
                          identifiable)
    return value


def index_records(records):
    """``{(problem, method, seed): record}``, the last record of a repeated triple
    winning, so a results file appended to twice reads as its newest run."""
    return {(r['problem'], r.get('method', '?'), r['seed']): r for r in records}


def methods_in(records):
    """Every method name in ``records``, in first-seen order."""
    seen = []
    for r in records:
        m = r.get('method', '?')
        if m not in seen:
            seen.append(m)
    return seen


def paired(records, baseline, method, problems=None):
    """``[(problem, seed, baseline_record, method_record), ...]`` for every
    (problem, seed) both methods were run from, in problem then seed order.

    Pairing is what makes the tests here valid: both methods start from the same
    initial population under the same fit seed, so the difference between the two
    records of a pair is the method's, not the draw's.
    """
    index = index_records(records)
    keys = sorted({(p, s) for (p, m, s) in index if m == baseline}
                  & {(p, s) for (p, m, s) in index if m == method})
    if problems is not None:
        keys = [k for k in keys if k[0] in problems]
    return [(p, s, index[(p, baseline, s)], index[(p, method, s)]) for p, s in keys]


def _pooled(recs):
    """Successes, median and mean final error over a list of records."""
    errors = [r['max_error'] for r in recs]
    finite = [e for e in errors if not math.isinf(e)]
    return {'n': len(recs), 'successes': sum(1 for r in recs if r['success_loose']),
            'median_max_error': statistics.median(finite) if finite else None,
            'mean_max_error': (sum(finite) / len(finite)) if finite else None,
            'failures': len(errors) - len(finite)}


def compare(records, baseline, methods=None, problems=None, checkpoints=None,
            problem_defs=None):
    """Score every method in ``methods`` against ``baseline`` over the seeds they share.

    Returns ``{'baseline', 'problems', 'methods': [...], 'checkpoints': {...}}`` where
    each method entry holds its pooled summary, the sign test on the final error, the
    McNemar test on success, and the same per problem. ``checkpoints`` is a list of
    simulation counts, or ``None`` for :data:`DEFAULT_CHECKPOINT_FRACTIONS` of each
    problem's own budget (which needs ``problem_defs``).
    """
    defs = {p.id: p for p in (problem_defs if problem_defs is not None else load_problems())}
    if methods is None:
        methods = [m for m in methods_in(records) if m != baseline]
    out = {'baseline': baseline, 'methods': [], 'checkpoints': {}}
    ids, baseline_keys = set(), {}
    for method in methods:
        pairs = paired(records, baseline, method, problems)
        ids.update(p for p, _, _, _ in pairs)
        baseline_keys.update({(p, s): b for p, s, b, _ in pairs})
        entry = {'method': method, 'n_pairs': len(pairs),
                 'baseline': _pooled([b for _, _, b, _ in pairs]),
                 'method_summary': _pooled([m for _, _, _, m in pairs]),
                 'sign': sign_test([(b['max_error'], m['max_error']) for _, _, b, m in pairs]),
                 'mcnemar': mcnemar([(b['success_loose'], m['success_loose'])
                                     for _, _, b, m in pairs]),
                 'per_problem': []}
        for pid in sorted({p for p, _, _, _ in pairs}):
            sub = [t for t in pairs if t[0] == pid]
            entry['per_problem'].append({
                'problem': pid, 'n_pairs': len(sub),
                'baseline': _pooled([b for _, _, b, _ in sub]),
                'method_summary': _pooled([m for _, _, _, m in sub]),
                'sign': sign_test([(b['max_error'], m['max_error']) for _, _, b, m in sub]),
                'mcnemar': mcnemar([(b['success_loose'], m['success_loose'])
                                    for _, _, b, m in sub]),
            })
        out['methods'].append(entry)
    # The baseline over every seed any compared method was paired with, which is the same
    # set for every method unless one of them was run from fewer.
    out['baseline_summary'] = _pooled(list(baseline_keys.values()))
    out['problems'] = sorted(ids)
    for pid in out['problems']:
        problem = defs[pid]
        points = (list(checkpoints) if checkpoints is not None
                  else [int(round(f * problem.budget_simulations))
                        for f in DEFAULT_CHECKPOINT_FRACTIONS])
        rows = {}
        for method in [baseline] + list(methods):
            recs = [r for r in records if r['problem'] == pid and r.get('method') == method]
            if not recs:
                continue
            rows[method] = [_median([best_so_far(r['trace'], problem.identifiable, c)
                                     for r in recs]) for c in points]
        out['checkpoints'][pid] = {'points': points, 'rows': rows}
    return out


def _p(value):
    return '-' if value is None else ('%.3f' % value)


def _err(value):
    return '-' if value is None else ('%.3f' % value)


def format_comparison(result):
    """A comparison as Markdown: the pooled table, the paired tests, the same per
    problem, and the progress at simulation checkpoints."""
    base = result['baseline']
    lines = ['Seed-paired against `%s` on %d problem(s): %s.'
             % (base, len(result['problems']), ', '.join(result['problems'])), '']
    lines += ['| method | paired seeds | successes | median final error | mean final error |',
              '|---|---:|---:|---:|---:|']
    rows = [(base, ' (baseline)', result['baseline_summary'], result['baseline_summary']['n'])]
    rows += [(e['method'], '', e['method_summary'], e['n_pairs']) for e in result['methods']]
    for name, note, s, n in rows:
        lines.append('| `%s`%s | %d | %d | %s | %s |'
                     % (name, note, n, s['successes'], _err(s['median_max_error']),
                        _err(s['mean_max_error'])))
    failed = ['%s %d' % (name, s['failures']) for name, _, s, _ in rows if s['failures']]
    if failed:
        lines += ['', 'The median and mean leave out the fits that reported no usable '
                      'estimate: ' + ', '.join(failed) + '.']
    lines += ['', 'Paired tests against `%s` (sign test on the final error, McNemar\'s '
                  'exact test on success):' % base, '',
              '| method | error better / worse / tie | sign p | only it / only `%s` | McNemar p |' % base,
              '|---|---:|---:|---:|---:|']
    for e in result['methods']:
        sign, mc = e['sign'], e['mcnemar']
        lines.append('| `%s` | %d / %d / %d | %s | %d / %d | %s |'
                     % (e['method'], sign['better'], sign['worse'], sign['ties'], _p(sign['p']),
                        mc['only_b'], mc['only_a'], _p(mc['p'])))
    lines += ['', 'Per problem (successes of the paired seeds, median final error, and the '
                  'sign test on the final error):', '',
              '| problem | method | seeds | successes | median final error | better / worse / tie | sign p |',
              '|---|---|---:|---:|---:|---:|---:|']
    for pid in result['problems']:
        shown_baseline = False
        for e in result['methods']:
            for row in e['per_problem']:
                if row['problem'] != pid:
                    continue
                if not shown_baseline:
                    lines.append('| %s | `%s` (baseline) | %d | %d | %s | | |'
                                 % (pid, base, row['n_pairs'], row['baseline']['successes'],
                                    _err(row['baseline']['median_max_error'])))
                    shown_baseline = True
                s, sign = row['method_summary'], row['sign']
                lines.append('| %s | `%s` | %d | %d | %s | %d / %d / %d | %s |'
                             % (pid, e['method'], row['n_pairs'], s['successes'],
                                _err(s['median_max_error']),
                                sign['better'], sign['worse'], sign['ties'], _p(sign['p'])))
    lines += ['', 'Error of the reported best at simulation checkpoints, median over every '
                  'seed the method was run from:', '']
    for pid in result['problems']:
        cp = result['checkpoints'].get(pid)
        if not cp or not cp['rows']:
            continue
        lines += ['| %s | %s |' % (pid, ' | '.join(str(c) for c in cp['points'])),
                  '|---|' + '---:|' * len(cp['points'])]
        for method, values in cp['rows'].items():
            lines.append('| `%s` | %s |' % (method, ' | '.join(_err(v) for v in values)))
        lines.append('')
    return '\n'.join(lines).rstrip() + '\n'
