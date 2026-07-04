"""Regression guard: the run loop must not ship a scattered Future as a Job attribute.

``Algorithm.run`` scatters the model_list (and, by default, the ObjectiveCalculator)
once per fit and hands the resulting Futures to ``client.submit``. dask substitutes a
Future for its concrete value **only** when the Future is a *direct* argument to
``submit``; a Future left buried as an attribute of the submitted Job is pickled
verbatim and, since distributed 2026.6.0, deserializes "not properly initialized" on
the worker, so ``.result()`` raises there. The symptom was that every simulation job
failed with::

    RuntimeError: <class 'distributed.client.Future'> object not properly
    initialized. This can happen if the object is being deserialized outside of
    the context of a Client or Worker.

and the real ``pybnf -c`` path was dead for any dask-dispatched backend (lanl/PyBNF #476).
The fix passes both scattered Futures as direct ``models=`` / ``calc=`` kwargs to
``client.submit`` and rebinds their worker-resolved values in ``core.run_job``
(``core._ResolvedFuture``).

Why a whole separate test file with a *real* client: the rest of the suite drives fits
through a **synchronous fake** client (``integration_harness`` / ``test_run_loop``) that
runs jobs inline and never serializes anything -- so it is structurally blind to this
bug (that is exactly why the regression shipped green). This test stands up a genuine,
in-process **threaded** ``distributed.Client`` and drives a real optimizer end to end
against the backend-free analytical target: the exact serialization path the regression
broke, with no simulation backend (BNG2.pl) required, cheap enough to run in default CI.
"""
import os

import numpy as np
import pytest

from . import integration_harness as H
from .context import algorithms

distributed = pytest.importorskip('distributed')


@pytest.fixture
def real_threaded_client():
    """A real in-process, threaded ``distributed.Client`` set as the default client
    (so ``core.as_completed`` -- called without an explicit client -- resolves it).

    Threaded workers (``processes=False``) still round-trip task args and scattered
    data through distributed's (de)serialization, which is all that is needed to
    reproduce the buried-Future break, while staying fast and subprocess-free.
    ``dashboard_address=None`` avoids binding a bokeh port under parallel CI.
    """
    client = distributed.Client(processes=False, n_workers=1, threads_per_worker=2,
                                dashboard_address=None)
    try:
        yield client
    finally:
        client.close()


def test_real_dask_fit_recovers_mode_end_to_end(tmp_path, real_threaded_client):
    """A real ``DifferentialEvolution`` fit completes through a genuine dask client
    and recovers the analytical Gaussian mode.

    Pre-fix, every worker raised ``RuntimeError: <Future> object not properly
    initialized`` resolving the model_list Future stashed as a Job attribute, so no
    job produced a result and ``run()`` aborted with a PybnfError ("all jobs failing").
    The default ``local_objective_eval=0`` also offloads scoring, so the scattered
    ObjectiveCalculator Future travels the same submit path -- this covers both Futures.
    """
    mean, var = [2.0, -1.0], [1.0, 1.0]
    tgt, exp = H.write_target(tmp_path, H.gaussian_spec(mean, var))
    conf = H.make_config(tmp_path, 'de', tgt, exp, n_params=2,
                         population_size=16, max_iterations=30, stop_tolerance=1e-6)
    alg = algorithms.DifferentialEvolution(conf)
    # The scaffolding main() normally makes before run() (integration_harness.drive).
    os.makedirs(alg.sim_dir, exist_ok=True)
    os.makedirs(alg.res_dir, exist_ok=True)

    # Pre-fix this raises PybnfError (every job dies deserializing a Future);
    # post-fix it completes.
    alg.run(real_threaded_client)

    # Jobs actually ran on the worker and came back scored -- the crux of the fix.
    assert alg.success_count > 0, 'no job completed through the real dask client'
    assert alg.fail_count == 0, 'jobs failed through the real dask client'
    # The whole pipeline works: the optimizer drove the best fit to the known mode.
    recovered = H.best_params(alg, 2)
    assert np.allclose(recovered, mean, atol=0.3), \
        'real-dask fit recovered %s, expected ~%s' % (recovered, mean)
    assert alg.trajectory.best_score() < 0.1
