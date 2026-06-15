"""PEtab v2 exporter: a PyBNF/BNGL job -> PEtab v2 artifacts (issue #407; ADR-0025).

The **exporter-first** direction of the PEtab interop (ADR-0025): a working PyBNF
BNGL job and a native ``.conf`` are *read* and serialized to a PEtab v2 problem
(``parameters.tsv`` / ``observables.tsv`` / ``measurements.tsv`` / ``problem.yaml`` +
a PEtab-clean copy of the model), rather than *generating* BNGL from a declarative
PEtab spec (the harder importer direction, deferred). The reverse asset mappings live
beside their importer twins (``parameters.petab_parameter_row``,
``observables.petab_observable_row``, ``measurements.measurement_rows_from_data``);
this module is the *disposable* glue: it reads the job (the stdlib ``ploop`` config
parser, a focused BNGL block reader, and :class:`pybnf.data.Data` for the ``.exp``)
and writes the files.

**Why a function is the measurement model.** A fitted ``.exp`` column matches a BNGL
**observable** *or* a **function** (PyBNF forces ``print_functions=>1``), and the
function is usually the measurement model. So an observable column exports to
``observableFormula = <name>`` and a function column to ``observableFormula =
<name>`` too -- always the *bare model name*, with the function carried verbatim in
the model file (ADR-0025). PEtab ids are prefixed (``obs_``/``func_`` for
observables, the unprefixed model name for parameters) to keep the PEtab-id namespace
disjoint from the model-entity namespace.

**Scope (chunk 1).** A single BNGL model, one base **time-course** ``.exp`` (no
conditions), the ``chi_sq`` objective, ``uniform_var`` free parameters, per-point
``_SD`` noise. Everything else -- a second model, a ``.exp`` bound to a dose-response
``parameter_scan``, ``mutant`` conditions, a non-``chi_sq`` objective, a non-uniform
prior, an SBML model -- raises ``NotImplementedError`` (the boundary is in code, not
silent; ADR-0019/0023 discipline). The oracle is petab's **model-less table-level
validation** (``petablint`` cannot load a BNGL model in petab 0.8.2), wired into the
tests; see ADR-0025.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from ..data import Data
from ..parse import ploop
from ..printing import PybnfError
from ..pset import FreeParameter
from ._tsv import num
from .measurements import measurement_rows_from_data, write_measurement_table
from .observables import petab_observable_row, write_observable_table
from .parameters import petab_parameter_row, write_parameter_table

# The job objective -> the PEtab v2 noiseDistribution it maps to (ADR-0023 reversed).
# Chunk 1 supports the global ``chi_sq`` (Gaussian) objective; other objectives and
# per-observable ``noise_model`` lines are a later export chunk.
_OBJFUNC_TO_NOISE_DISTRIBUTION = {'chi_sq': 'normal'}

# Free-parameter declaration keywords (the ``(keyword, name)`` tuple keys ``ploop``
# emits). Only ``uniform_var`` exports in chunk 1; the rest raise.
_VAR_DECL = re.compile(r'(_var$|^var$|^logvar$)')

_FREE_TOKEN = re.compile(r'\w+__FREE')


@dataclass(frozen=True)
class _BnglModel:
    """The bits of a BNGL model the exporter reads (a focused, dependency-free parse)."""

    text: str
    free_to_param: dict          # 'v1__FREE' -> 'v1' (the model parameter it drives)
    observable_names: frozenset  # {'x'}
    function_names: frozenset    # {'y'}


# ---------------------------------------------------------------------------
# The exporter driver
# ---------------------------------------------------------------------------

def export_job(conf_path, out_dir):
    """Export the PyBNF job at ``conf_path`` to a PEtab v2 problem in ``out_dir``.

    Writes ``parameters.tsv``, ``observables.tsv``, ``measurements.tsv``,
    ``problem.yaml``, and a PEtab-clean copy of the BNGL model. Returns the
    ``out_dir`` path. Raises ``NotImplementedError`` at every chunk-1 boundary and
    ``PybnfError`` for a malformed/unsupported job (see the module docstring).
    """
    conf_path = Path(conf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conf = _read_conf_dict(conf_path)
    model_file, exp_file, noise_distribution = _resolve_scope(conf)

    free_params = _free_parameters_from_conf(conf)
    bngl = _read_bngl(conf_path.parent / model_file)
    data = Data(file_name=str(conf_path.parent / exp_file))

    observable_rows, column_to_observable_id = _observable_rows(
        data, bngl, noise_distribution, model_file)
    parameter_rows, free_to_nominal = _parameter_rows(free_params, bngl, model_file)
    measurement_rows = measurement_rows_from_data(
        data, column_to_observable_id, experiment_id='')

    model_filename = Path(model_file).name
    model_id = re.sub(r'\.bngl$', '', model_filename)
    write_parameter_table(parameter_rows, out_dir / 'parameters.tsv')
    write_observable_table(observable_rows, out_dir / 'observables.tsv')
    write_measurement_table(measurement_rows, out_dir / 'measurements.tsv')
    (out_dir / model_filename).write_text(
        clean_model_for_petab(bngl.text, free_to_nominal))
    write_problem_yaml(out_dir / 'problem.yaml', model_filename, model_id)
    return out_dir


def _resolve_scope(conf):
    """Enforce the chunk-1 scope and return ``(model_file, exp_file, noise_dist)``."""
    if 'mutant' in conf:
        raise NotImplementedError(
            "This job has 'mutant' conditions, which export to PEtab conditions/"
            "experiments -- a later chunk (ADR-0025, #407).")
    if 'param_scan' in conf:
        raise NotImplementedError(
            "This job has a 'param_scan' (dose-response) action; its independent axis "
            "is a swept parameter that lives in the PEtab conditions/experiments tables "
            "-- a later export chunk (ADR-0025, #407).")

    models = sorted(conf.get('models', set()))
    if len(models) != 1:
        raise NotImplementedError(
            f"Chunk 1 exports a single-model job; this one has {len(models)} models "
            f"({models}). Multi-model export (PEtab modelId) is a later chunk.")
    model_file = models[0]
    if not model_file.endswith('.bngl'):
        raise NotImplementedError(
            f"Chunk 1 exports BNGL models; '{model_file}' is not '.bngl'. SBML/Antimony "
            f"export is a later chunk (ADR-0025 settles the BNGL side first).")

    exp_files = [e for e in conf.get(model_file, []) if e.endswith('.exp')]
    others = [e for e in conf.get(model_file, []) if not e.endswith('.exp')]
    if others:
        raise NotImplementedError(
            f"Model '{model_file}' has non-.exp data ({others}); constraint (.con/.prop) "
            f"export is a later chunk.")
    if len(exp_files) != 1:
        raise NotImplementedError(
            f"Chunk 1 exports a single experiment; model '{model_file}' is bound to "
            f"{len(exp_files)} .exp files ({exp_files}). Multiple experiments map to the "
            f"PEtab experiments table -- a later chunk.")

    objfunc = conf.get('objfunc', 'chi_sq')
    if objfunc not in _OBJFUNC_TO_NOISE_DISTRIBUTION:
        raise NotImplementedError(
            f"Chunk 1 exports the '{', '.join(_OBJFUNC_TO_NOISE_DISTRIBUTION)}' "
            f"objective; this job uses objfunc='{objfunc}'. Other objectives / "
            f"per-observable noise_model lines are a later chunk (ADR-0023 reversed).")
    return model_file, exp_files[0], _OBJFUNC_TO_NOISE_DISTRIBUTION[objfunc]


def _observable_rows(data, bngl, noise_distribution, model_file):
    """Classify each fitted ``.exp`` column as a model observable or function and map it."""
    indvar = min(data.cols, key=data.cols.get)
    observable_rows = []
    column_to_observable_id = {}
    for col in sorted(data.cols, key=data.cols.get):
        if col == indvar or col.endswith('_SD'):
            continue
        if col in bngl.observable_names:
            kind = 'observable'
        elif col in bngl.function_names:
            kind = 'function'
        else:
            raise PybnfError(
                f"Exp column '{col}' matches no observable or function in model "
                f"'{model_file}' (its observables: {sorted(bngl.observable_names)}; "
                f"functions: {sorted(bngl.function_names)}).")
        sd_from_data = (col + '_SD') in data.cols
        row = petab_observable_row(col, kind, noise_distribution, sd_from_data)
        observable_rows.append(row)
        column_to_observable_id[col] = row.observable_id
    if not observable_rows:
        raise PybnfError(
            f"Exp file for model '{model_file}' has no fittable observable/function "
            f"columns (only an independent variable and/or _SD columns).")
    return observable_rows, column_to_observable_id


def _parameter_rows(free_params, bngl, model_file):
    """Map each free parameter to a row, resolving its model parameter from the BNGL."""
    parameter_rows = []
    free_to_nominal = {}
    for fp in free_params:
        model_param = bngl.free_to_param.get(fp.name)
        if model_param is None:
            raise PybnfError(
                f"Free parameter '{fp.name}' is not assigned to any parameter in model "
                f"'{model_file}' (expected a line '<param> {fp.name}' in begin "
                f"parameters).")
        parameter_rows.append(petab_parameter_row(fp, parameter_id=model_param))
        # A syntactically valid default for the PEtab-clean model; PEtab overrides it
        # during estimation. The bounds' midpoint is a reasonable in-range nominal.
        free_to_nominal[fp.name] = (float(fp.p1) + float(fp.p2)) / 2.0
    return parameter_rows, free_to_nominal


# ---------------------------------------------------------------------------
# Reading the job (the disposable input half of the seam)
# ---------------------------------------------------------------------------

def _read_conf_dict(conf_path):
    """Parse a ``.conf`` to the raw ``ploop`` dict (no model loading, no BNG)."""
    with open(conf_path) as fh:
        return ploop(fh.readlines())


def _free_parameters_from_conf(conf):
    """Build ``FreeParameter`` objects from the config's ``(keyword, name)`` entries."""
    free_params = []
    for key, value in conf.items():
        if not (isinstance(key, tuple) and len(key) == 2
                and isinstance(key[0], str) and isinstance(key[1], str)):
            continue
        keyword, name = key
        if not _VAR_DECL.search(keyword):
            continue
        if keyword != 'uniform_var':
            raise NotImplementedError(
                f"Free parameter '{name}' is a '{keyword}'; chunk 1 exports only "
                f"'uniform_var' (estimate=true with bounds). Other priors are a later "
                f"export chunk (ADR-0025).")
        free_params.append(FreeParameter(name, 'uniform_var', float(value[0]),
                                         float(value[1])))
    if not free_params:
        raise PybnfError("No free parameters (uniform_var) found in the config.")
    return free_params


def _read_bngl(path):
    """Read a BNGL model's parameters/observables/functions (focused, dependency-free)."""
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    free_to_param = {}
    for line in _block_lines(text, 'parameters'):
        tokens = line.split()
        if len(tokens) >= 2 and _FREE_TOKEN.fullmatch(tokens[1]):
            free_to_param[tokens[1]] = tokens[0]
    observable_names = frozenset(
        n for n in (_observable_name(line) for line in _block_lines(text, 'observables'))
        if n)
    function_names = frozenset(
        n for n in (_function_name(line) for line in _block_lines(text, 'functions'))
        if n)
    return _BnglModel(text, free_to_param, observable_names, function_names)


def _block_lines(text, block_name):
    """Yield the comment-stripped, non-blank lines inside a ``begin/end <block>``."""
    begin = re.compile(rf'^begin\s+{block_name}\b', re.I)
    end = re.compile(rf'^end\s+{block_name}\b', re.I)
    lines = []
    in_block = False
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].strip()
        if begin.match(line):
            in_block = True
        elif end.match(line):
            in_block = False
        elif in_block and line:
            lines.append(line)
    return lines


def _observable_name(line):
    """The observable name in a ``Molecules <name> ...`` / ``Species <name> ...`` line."""
    tokens = line.split()
    return tokens[1] if len(tokens) >= 2 else None


def _function_name(line):
    """The function name in a ``<name>() = ...`` (or ``<name> = ...``) functions line."""
    m = re.match(r'(\w+)\s*\(', line) or re.match(r'(\w+)\s*=', line)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Emitting the PEtab-clean model and problem.yaml
# ---------------------------------------------------------------------------

def clean_model_for_petab(text, free_to_nominal):
    """Return a PEtab-clean copy of a BNGL model.

    PEtab estimates the model parameters directly, so the ``__FREE`` markers are
    replaced by a plain in-range nominal value (``v1 v1__FREE`` -> ``v1 5``), and the
    ``begin actions`` block is stripped (PEtab drives simulation via the measurement
    times / experiments, not the model's own actions). The reaction network and the
    ``begin functions`` block -- which carry the measurement model -- are left intact.
    """
    for free_name, nominal in free_to_nominal.items():
        text = re.sub(rf'\b{re.escape(free_name)}\b', num(nominal), text)
    text = re.sub(r'^[ \t]*begin\s+actions\b.*?^[ \t]*end\s+actions\b[^\n]*\n?',
                  '', text, flags=re.S | re.I | re.M)
    return text


def write_problem_yaml(path, model_filename, model_id):
    """Write a PEtab v2 ``problem.yaml`` referencing the three tables and the model."""
    Path(path).write_text(
        'format_version: 2.0.0\n'
        'parameter_files:\n  - parameters.tsv\n'
        'observable_files:\n  - observables.tsv\n'
        'measurement_files:\n  - measurements.tsv\n'
        'model_files:\n'
        f'  {model_id}:\n'
        f'    location: {model_filename}\n'
        '    language: bngl\n')
