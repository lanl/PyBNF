"""On-disk action-output writing for the bngsim_model package.

Writes each action's Data to PyBNF-compatible .gdat/.scan files (the BNG2.pl
on-disk layout). numpy only; no simulator dependency.
"""


import numpy as np


def _ext_for_simtype(simtype):
    """Return the file extension PyBNF uses for a given action type."""
    return 'scan' if simtype == 'parameter_scan' else 'gdat'


def _write_saved_action_outputs(folder, filename, suffixes, ds):
    """Write each action's Data to a PyBNF-compatible .gdat/.scan file.

    Mirrors the BNG2.pl on-disk layout that BNGLModel/NetModel rely on:
    one file per (action, mutant) keyed as ``{folder}/{filename}_{suffix}.{ext}``.
    Each file has a leading ``#``-prefixed header line listing column names
    in the same order as ``Data.headers``, followed by whitespace-separated
    numeric rows — re-readable via :class:`pybnf.data.Data`.
    """
    for simtype, suffix in suffixes:
        data = ds.get(suffix)
        if data is None:
            continue
        headers = [data.headers[i] for i in range(data.data.shape[1])]
        path = f'{folder}/{filename}_{suffix}.{_ext_for_simtype(simtype)}'
        np.savetxt(path, data.data, header=' '.join(headers))
