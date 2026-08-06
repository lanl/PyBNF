"""Optional Antimony simulation using bngsim."""


import logging

from ._bngsim_caps import (
    BNGSIM_ANTIMONY_ERROR,
    BNGSIM_HAS_ANTIMONY,
    BNGSIM_HAS_ANTIMONY_PY as ANTIMONY_AVAILABLE,
    bngsim,
)
from .bngsim_sbml_model import (
    BngsimSbmlModelNoTimeout,
)
from .pset import ModelError


logger = logging.getLogger(__name__)


try:
    import antimony
except ImportError:
    antimony = None


def _require_bngsim_antimony_support():
    if not BNGSIM_HAS_ANTIMONY:
        raise RuntimeError(BNGSIM_ANTIMONY_ERROR)


def _antimony_last_error():
    get_last_error = getattr(antimony, 'getLastError', None)
    if get_last_error is None:
        return ''
    try:
        return get_last_error().strip()
    except Exception:
        return ''


def _antimony_text_to_sbml_text(text, source_desc):
    if not ANTIMONY_AVAILABLE:
        raise ModelError(BNGSIM_ANTIMONY_ERROR)

    clear_loads = getattr(antimony, 'clearPreviousLoads', None)
    if clear_loads is not None:
        clear_loads()

    try:
        load_result = antimony.loadString(text)
        if isinstance(load_result, (int, float)) and load_result < 0:
            error = _antimony_last_error() or 'unknown Antimony parse error'
            raise ModelError(
                f'Failed to parse Antimony from {source_desc}: {error}'
            )

        module_names = list(antimony.getModuleNames() or [])
        module_name = None
        for candidate in reversed(module_names):
            if candidate != '__main__':
                module_name = candidate
                break
        if module_name is None and module_names:
            module_name = module_names[-1]
        if module_name is None:
            error = _antimony_last_error() or 'Antimony did not report a loadable module'
            raise ModelError(
                f'Failed to parse Antimony from {source_desc}: {error}'
            )

        sbml_text = antimony.getSBMLString(module_name)
        if not sbml_text:
            error = _antimony_last_error() or 'Antimony did not produce SBML output'
            raise ModelError(
                f'Failed to convert Antimony from {source_desc} to SBML: {error}'
            )
        return sbml_text
    finally:
        if clear_loads is not None:
            clear_loads()


class BngsimAntimonyModelNoTimeout(BngsimSbmlModelNoTimeout):
    def __init__(self, file, abs_file, pset=None, actions=(), save_files=False, integrator='cvode',
                 strict_ssa=True, rtol=None, atol=None):
        if integrator != 'cvode':
            raise ModelError(
                'Antimony models currently support only sbml_integrator = cvode'
            )

        _require_bngsim_antimony_support()

        self._init_common_attrs(file, abs_file, pset, actions, save_files, integrator, strict_ssa,
                                file_ext='.ant', rtol=rtol, atol=atol)
        self.stochastic = False

        with open(self.abs_file_path, encoding='utf-8', errors='replace') as fh:
            self._base_antimony_text = fh.read()

        self._base_sbml_text = _antimony_text_to_sbml_text(self._base_antimony_text, self.file_path)
        self._extract_sbml_structure()
        self._load_engine_model_or_raise(
            f'Failed to load model {self.name}.ant - There were errors in parsing this Antimony file. '
            'See log for details.'
        )

        logger.debug('Loaded model %s with bngsim Antimony backend', self.name)

    def _load_bngsim_model_from_path(self, path):
        return bngsim.Model.from_antimony(path)

    def _save_antimony_source(self, file_prefix):
        with open(file_prefix + '.ant', 'w') as out:
            out.write(self._base_antimony_text)

    def save(self, file_prefix):
        self._save_antimony_source(file_prefix)
        super().save(file_prefix)

    def save_all(self, file_prefix):
        for mut in self.mutants:
            mutant_prefix = file_prefix + mut.suffix
            self._save_antimony_source(mutant_prefix)
            with open(f'{mutant_prefix}.xml', 'w') as out:
                out.write(self.model_text(mut=mut))


# Retained as an alias for backwards compatibility. bngsim now enforces the
# wall-clock budget in-process, so the subprocess wrapper is unnecessary.
BngsimAntimonyModel = BngsimAntimonyModelNoTimeout
