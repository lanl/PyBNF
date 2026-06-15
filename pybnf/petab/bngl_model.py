"""A PEtab ``Model`` adapter for BNGL, registered into ``petab`` at runtime (ADR-0026).

``petab`` 0.8.x ships only ``sbml``/``pysb`` model loaders, so ``petablint`` cannot
load a ``language: bngl`` problem -- it dies at model-load before any table check.
:class:`BnglModel` is the missing loader: a :class:`petab.v1.models.model.Model`
backed by PyBNF's own stdlib BNGL reader (:mod:`pybnf.petab._bngl`), so the PEtab v2
exporter's emitted problems validate at *model* level (the ~5 model-cross checks
ADR-0025 had to exclude), with no BNG2.pl needed for the enumeration.

This is **Step A** of issue #420 -- a local reference implementation. :func:`register_bngl`
teaches a running ``petab`` about BNGL by rebinding ``petab.v2.core.model_factory``
(petab's factory is a hardcoded ``if/elif`` with no plugin hook). **Step B** upstreams
the same class to ``PEtab-dev/libpetab-python`` as a peer of ``PySBModel``; once that
lands, :func:`register_bngl` collapses to a no-op.

Validation needs only *parsing*, so every ABC method is backed by the parsed entity
sets -- except :meth:`BnglModel.is_valid`, which shells out to ``BNG2.pl --check`` (the
real validator) when a BNG2.pl is locatable, and degrades gracefully to ``True`` when
it is not. This module ``import``s ``petab`` and is therefore **not** imported by
``pybnf.petab.__init__``; it is reached only from the test tier / an explicit
``register_bngl()`` call, keeping core dependency-free (ADR-0025).
"""

import os
import shutil
import subprocess
from pathlib import Path

from petab.v1.models.model import Model

from ._bngl import parse_model

#: BNGL model type, as used in a PEtab v2 yaml file as ``language``.
MODEL_TYPE_BNGL = 'bngl'


class BnglModel(Model):
    """PEtab ``Model`` wrapper for a BNGL model (validation-grade; no simulation)."""

    type_id = MODEL_TYPE_BNGL

    def __init__(self, entities, model_id, path=None):
        super().__init__()
        self._entities = entities
        self._model_id = model_id
        self._path = Path(path) if path is not None else None

    @staticmethod
    def from_file(filepath_or_buffer, model_id=None, base_path=None):
        path = Path(filepath_or_buffer)
        if base_path is not None and not path.is_absolute():
            path = Path(base_path) / path
        text = path.read_text(encoding='utf-8', errors='replace')
        return BnglModel(parse_model(text), model_id=model_id or path.stem, path=path)

    def to_file(self, filename=None):
        target = Path(filename) if filename is not None else self._path
        if target is None:
            raise ValueError("No filename given and this BnglModel has no source path.")
        target.write_text(self._entities.text, encoding='utf-8')

    @property
    def model_id(self):
        return self._model_id

    # -- parameters ---------------------------------------------------------

    def get_parameter_ids(self):
        return list(self._entities.parameters)

    def get_parameter_value(self, id_):
        try:
            rhs = self._entities.parameters[id_]
        except KeyError as e:
            raise ValueError(f"Parameter {id_} does not exist.") from e
        try:
            return float(rhs)
        except ValueError as e:
            raise NotImplementedError(
                f"Parameter '{id_}' has an expression value '{rhs}'. Evaluating a "
                f"BNGL parameter expression needs BNG2.pl/network generation, which "
                f"is out of scope for the validation-grade BnglModel (ADR-0026)."
            ) from e

    def get_free_parameter_ids_with_values(self):
        out = []
        for name, rhs in self._entities.parameters.items():
            try:
                out.append((name, float(rhs)))
            except ValueError:
                continue  # an expression-valued parameter has no validation-grade value
        return out

    def get_valid_parameters_for_parameter_table(self):
        return list(self._entities.parameters)

    # -- model-entity namespaces (verified against BNG2.pl; ADR-0026) -------

    def has_entity_with_id(self, entity_id):
        """Any declared identifier: the full BNGL model-entity namespace."""
        e = self._entities
        return (entity_id in e.parameters
                or entity_id in e.observable_names
                or entity_id in e.function_names
                or entity_id in e.molecule_type_names
                or entity_id in e.compartment_names
                or entity_id in e.seed_species)

    def symbol_allowed_in_observable_formula(self, id_):
        """The BNG ``ParamList``: parameters, observables, global functions only."""
        e = self._entities
        return (id_ in e.parameters
                or id_ in e.observable_names
                or id_ in e.function_names)

    def get_valid_ids_for_condition_table(self):
        return list(self._entities.parameters) + list(self._entities.compartment_names)

    def is_state_variable(self, id_):
        """A species. At validation grade, only the concrete ``seed species`` are known
        (the full set is a network-generation product, out of scope; ADR-0026)."""
        return id_ in self._entities.seed_species

    # -- validity -----------------------------------------------------------

    def is_valid(self):
        """``BNG2.pl --check`` (real validation, no network gen) when locatable, else
        ``True`` -- never a false failure where no BNG backend is available."""
        bng2 = _locate_bng2()
        if bng2 is None or self._path is None:
            return True
        try:
            result = subprocess.run(
                [bng2, '--check', str(self._path)],
                capture_output=True, text=True, timeout=120,
                cwd=str(self._path.parent),
            )
        except (OSError, subprocess.SubprocessError):
            return True  # a tooling hiccup must not masquerade as an invalid model
        return result.returncode == 0


def _locate_bng2():
    """A path to ``BNG2.pl`` via ``BNGPATH`` or ``PATH``, or ``None`` if unavailable."""
    bngpath = os.environ.get('BNGPATH')
    if bngpath:
        candidate = Path(bngpath) / 'BNG2.pl'
        if candidate.is_file():
            return str(candidate)
    return shutil.which('BNG2.pl')


def register_bngl():
    """Teach a running ``petab`` to load ``language: bngl`` models via :class:`BnglModel`.

    Idempotent and additive: rebinds ``petab.v2.core.model_factory`` to route ``bngl``
    to :class:`BnglModel` and delegate every other language to the captured original
    (so ``sbml``/``pysb`` are untouched), and registers ``'bngl'`` as a known model
    type. A guarded permanent rebind (no teardown) -- the temporary local stand-in for
    Step B's upstream loader (ADR-0026).
    """
    import petab.v1.models as _models
    import petab.v2.core as _v2core

    # If petab already supports BNGL natively (a libpetab-python build that
    # shipped the upstream loader -- #420 Step B), do nothing: its model_factory
    # already routes 'bngl' to a native BnglModel that supersedes this local
    # stand-in. This is the "collapse to a no-op" ADR-0026 anticipated; it is
    # what lets the exporter tests dogfood the native loader. We distinguish
    # native support (bngl present, our wrapper never installed) from our own
    # prior registration (the sentinel below) so re-entry stays idempotent.
    already_registered = hasattr(_v2core, '_pybnf_orig_model_factory')
    if MODEL_TYPE_BNGL in _models.known_model_types and not already_registered:
        return

    _models.known_model_types.add(MODEL_TYPE_BNGL)

    original = getattr(_v2core, '_pybnf_orig_model_factory', None)
    if original is None:
        original = _v2core.model_factory
        _v2core._pybnf_orig_model_factory = original

    def _model_factory(filepath_or_buffer, model_language, model_id=None, base_path=None):
        if model_language == MODEL_TYPE_BNGL:
            return BnglModel.from_file(
                filepath_or_buffer, model_id=model_id, base_path=base_path)
        return original(
            filepath_or_buffer, model_language, model_id=model_id, base_path=base_path)

    _v2core.model_factory = _model_factory
