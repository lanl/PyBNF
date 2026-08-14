"""The bngsim ``.net`` segment backend -- multiple shooting on a reaction network (#577).

The SBML/Antimony backend (:mod:`pybnf.shooting.bngsim_backend`) got the state for free: that
path reports its species *as* the trajectory's columns and labels its forward-sensitivity
selectors ``species:<name>`` on both axes, so one run hands over everything a knot needs. The
``.net`` path does not, and for a while ``job_type = ms`` refused it on that basis -- wrongly
framed as a property of the model. It is not one. A ``.net`` file is a fully expanded reaction
network with exactly the same kind of ODE state, and bngsim returns both its species
trajectory and its ``d(species)/d(species_0)`` when asked. What was missing was that nobody
asked: :meth:`~pybnf.bngsim_model.net_model.BngsimModel._build_data` assembles
``time + observables + expressions``, and the net backend's sensitivity request names
``observable:`` / ``expression:`` selectors.

This module asks. It is the one place in :mod:`pybnf.shooting` that needs *both* selector
families on one run, and that is the whole of what makes it different from its SBML peer:

* **an experiment scores observables, a continuity row is a difference of species.** On the
  SBML path those are the same columns; here they are not. ``OutputSensitivities.selectors``
  is a plain list, so one tensor carries the observable/expression rows the data terms read
  and the species rows the continuity block reads -- requested together, off one integration,
  because asking twice would mean integrating twice.
* **the segment's ``Data`` carries both too**, the ordinary observable/expression columns the
  objective scores plus the species columns
  :func:`~pybnf.shooting.backend.trace_from_data` reads the end-knot state from. Species
  names carry parentheses (``A(b!1).B(a!1)``) and observables do not, so a collision is
  vanishingly unlikely -- and is refused rather than silently overwritten, because a species
  column quietly shadowing an observable would change what the fit scores.

Nothing in the net backend is modified: this composes its existing pieces (``_build_data``,
the engine clone, the mutant copy, the species-initializer sync) from outside, so every
ordinary ``.net`` fit is untouched.

The cost, which is real and is the model's, not the method's
-------------------------------------------------------------
The auxiliary block is ``(m - 1) x n_species`` wide and the initial-condition sensitivity
system is ``n_species`` wide, so the transcription scales with the **expanded** species count
rather than with the number of fitted parameters. On a small network that is nothing; on a
combinatorially expanded one it is the dominant term -- ``egfr_ground.net`` (356 species) at
``m = 4`` adds ~1068 auxiliary variables. That is a property of writing multiple shooting on
the state of a rule-based model, not something this backend can arrange away, so
``job_type = ms`` reports the added width when it starts rather than letting a user discover
it from the run time.
"""

import numpy as np

from ..data import Data, OutputSensitivities
from ..printing import PybnfError
from .backend import SegmentBackend, SegmentSimulationFailed


class NetSegmentBackend(SegmentBackend):
    """One scored ``(model, condition)`` pair's segment simulator, on the ``.net`` path.

    :param model: The :class:`~pybnf.bngsim_model.net_model.BngsimModel`. This backend
        **owns** it, assigning the parameter set in place rather than deep-copying per
        evaluation -- sound because the whole multiple-shooting driver runs single-threaded
        on the master (a segment is not a :class:`~pybnf.pset.PSet` evaluation and never
        reaches a worker).
    :param sim_params: The parsed ``simulate()`` action this experiment is measured by, as
        :func:`~pybnf.bngsim_model.parsing._parse_simulate_action` returns it.
    :param mutant: The ``MutationSet`` (condition) it is measured under.
    :param suffix: The full output suffix, ``action suffix + mutant suffix``.
    :param timeout: Per-segment wall-clock bound, from ``wall_time_sim``.
    """

    def __init__(self, model, sim_params, mutant, suffix, timeout=None):
        self.model = model
        self.sim_params = dict(sim_params or {})
        self.mutant = mutant
        self.suffix = str(suffix)
        self.timeout = timeout
        self.print_functions = _flag(self.sim_params.get('print_functions', 0))
        self._states = tuple(model._engine_model.species_names)
        self._nominal = self._declared_state()
        self.n_simulations = 0
        self._point = None          # identity of the PSet the prepared engine holds
        self._prepared = None       # the per-point model copy (base or mutant)
        self._engine = None
        self._sim = None

    # -- the contract -----------------------------------------------------------

    @property
    def state_names(self):
        return self._states

    @property
    def nominal_state(self):
        return self._nominal

    def simulate(self, pset, sample_times, initial_state=None):
        times = [float(t) for t in np.asarray(sample_times, dtype=float).reshape(-1)]
        if len(times) < 2:
            raise PybnfError('A multiple-shooting segment needs at least two output times; '
                             'got %r.' % (times,))
        prepared, engine, sim = self._prepare(pset)
        if initial_state:
            for name, value in initial_state.items():
                if name not in self._state_set:
                    raise PybnfError(
                        "Multiple shooting tried to restart model '%s' from a state named "
                        "'%s', which is not one of its species."
                        % (getattr(self.model, 'name', '?'), name))
                engine.set_concentration(name, float(value))
            engine.save_concentrations()
        engine.reset()
        self.n_simulations += 1
        try:
            result = sim.run(t_span=(times[0], times[-1]), n_points=len(times),
                             sample_times=times, **self._run_kwargs())
            data = self._data_with_state(prepared, result)
        except Exception as exc:
            # A non-integrable point is a property of the point, not of the run. The
            # prepared engine is dropped: whatever state a failed solve left it in is not
            # one to restart the next segment from.
            self._point = None
            raise SegmentSimulationFailed('%s: %s' % (type(exc).__name__, exc)) from exc
        if not np.all(np.isfinite(np.asarray(data.data, dtype=float))):
            raise SegmentSimulationFailed('the segment produced a non-finite trajectory')
        return data

    # -- one engine + one simulator per parameter point --------------------------

    @property
    def _state_set(self):
        return set(self._states)

    def _prepare(self, pset):
        """The per-point model copy, engine model and ``Simulator``, built once and reused
        across that point's segments.

        Mirrors ``BngsimModel.execute``'s preamble -- apply the parameter set, re-derive the
        species initial concentrations from it (a free parameter that only seeds an IC is a
        silent no-op without that sync), reset -- on a *clone*, via the same
        ``_get_mutant_model_bngsim`` copy the ordinary mutant path uses. The wildtype's empty
        ``MutationSet`` goes through it too, so there is one code path rather than a special
        case that could drift from it.
        """
        if self._point is pset and self._engine is not None:
            return self._prepared, self._engine, self._sim
        self.model.param_set = pset
        # The per-action sensitivity gate (#475) reads this: without it a scored
        # experiment's segments would run sensitivity-free and the assembly would find no
        # tensor to differentiate.
        self.model._current_action_suffix = self.sim_params.get('suffix', 'time_course')
        try:
            prepared = self.model._get_mutant_model_bngsim(self.mutant)
            engine = prepared._engine_model
            for name in (prepared.param_set or {}).keys():
                try:
                    engine.set_param(name, prepared.param_set[name])
                except Exception:
                    # A free parameter of another model in a multi-model fit; the ordinary
                    # execute path warns and carries on, and so does this one.
                    pass
            prepared._sync_species_initial_concentrations(engine)
            engine.reset()
            sim = _runtime().bngsim.Simulator(
                engine, method='ode', **prepared._codegen_kwargs('ode'),
                **prepared._sensitivity_request_kwargs('ode'))
        except PybnfError:
            self._point = None
            raise
        except Exception as exc:
            self._point = None
            raise SegmentSimulationFailed(
                'the model could not be prepared at this point (%s: %s)'
                % (type(exc).__name__, exc)) from exc
        self._point, self._prepared, self._engine, self._sim = pset, prepared, engine, sim
        return prepared, engine, sim

    def _run_kwargs(self):
        """The action's own tolerances and this fit's ``wall_time_sim``, as the ordinary
        net simulate path passes them."""
        kwargs = {}
        for key in ('atol', 'rtol'):
            if key in self.sim_params:
                try:
                    kwargs[key] = float(self.sim_params[key])
                except (TypeError, ValueError):
                    pass
        try:
            timeout = float(self.timeout)
        except (TypeError, ValueError):
            timeout = 0.0
        if timeout > 0.0:
            kwargs['timeout'] = timeout
        return kwargs

    # -- both selector families, off one run -------------------------------------

    def _data_with_state(self, prepared, result):
        """The ordinary net ``Data`` plus the species columns and their sensitivity rows.

        The value columns are the net backend's own (``_build_data``, unmodified), so what
        the objective scores is byte-identical to an ordinary fit's; the species columns are
        appended, and the sensitivity tensor is requested over the union of both selector
        families.
        """
        data = prepared._build_data(result, print_functions=self.print_functions)
        species = list(result.species_names)
        clash = sorted(set(species) & set(data.cols))
        if clash:
            raise PybnfError(
                "Multiple shooting adds model '%s'\\'s species columns to each segment's "
                "trajectory so a knot can carry the state, and species %s already name a "
                "scored column of that trajectory. One name cannot mean both an observable "
                "and a species here." % (getattr(self.model, 'name', '?'), ', '.join(clash)))
        headers = [data.headers[i] for i in sorted(data.headers)] + species
        out = Data.from_columns(
            np.column_stack([np.asarray(data.data, dtype=float),
                             np.asarray(result.species, dtype=float)]), headers)
        sens = self._union_sensitivities(prepared, result, species)
        if sens is not None:
            out.output_sensitivities = sens
        return out

    def _union_sensitivities(self, prepared, result, species):
        """One tensor over the observable/expression rows the data terms read **and** the
        species rows the continuity block reads.

        bngsim answers a mixed selector list from a single run, which is what makes this one
        integration rather than two. ``None`` on the scalar path, or when the run carried no
        sensitivities.
        """
        if getattr(prepared, '_sensitivity_request', None) is None:
            return None
        if not (getattr(result, 'has_sensitivities', False)
                or getattr(result, 'has_sensitivities_ic', False)):
            return None
        selectors = ['observable:%s' % name for name in result.observable_names]
        if self.print_functions and getattr(result, 'has_sensitivities_expressions', False):
            selectors += ['expression:%s' % name
                          for name in prepared._differentiable_expression_names(result)]
        selectors += ['species:%s' % name for name in species]
        param_names = list(result.sensitivity_params)
        ic_species = list(result.sensitivity_ic_species)
        d_param = (np.asarray(result.output_sensitivities(selectors, axis='parameter'),
                              dtype=float) if param_names else None)
        d_ic = (np.asarray(result.output_sensitivities(selectors, axis='ic'), dtype=float)
                if ic_species else None)
        return OutputSensitivities(selectors=selectors, param_names=param_names,
                                   ic_species=ic_species, d_param=d_param, d_ic=d_ic)

    def _declared_state(self):
        """Each species' declared magnitude, read off a freshly reset engine clone.

        The constraint scales and the centre of each auxiliary variable's box need only a
        representative number. A species declared at zero -- on a network that grows its
        products, most of them -- says nothing about its own magnitude, so it inherits the
        model's scalar one (the median of the positive declarations), the same substitution
        ADR-0105 makes for the per-species absolute tolerance and for the same reason: the
        alternative is a scale of zero, which the transcription layer refuses outright.
        """
        engine = self.model._engine_model
        declared = np.array([float(engine.get_concentration(name)) for name in self._states],
                            dtype=float)
        declared[~np.isfinite(declared)] = 0.0
        positive = declared[declared > 0.0]
        fallback = float(np.median(positive)) if positive.size else 1.0
        return np.where(declared > 0.0, declared, fallback)

    def __repr__(self):
        return 'NetSegmentBackend(%r, states=%i)' % (self.suffix, len(self._states))


def _runtime():
    """The net backend's lazily-imported bngsim runtime handle."""
    from ..bngsim_model import _runtime as rt
    return rt


def _flag(value):
    try:
        return bool(int(float(str(value))))
    except (TypeError, ValueError):
        return False
