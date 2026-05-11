"""Optional Antimony simulation using bngsim."""


import logging

from .bngsim_sbml_model import (
    BNGSIM_AVAILABLE,
    LIBSBML_AVAILABLE,
    BngsimSbmlModelNoTimeout,
    _sbml_doc_from_text,
    bngsim,
)
from .pset import ModelError, MutationSet


logger = logging.getLogger(__name__)


try:
    import antimony
    ANTIMONY_AVAILABLE = True
except ImportError:
    antimony = None
    ANTIMONY_AVAILABLE = False


def _detect_bngsim_antimony_support():
    if not BNGSIM_AVAILABLE:
        return False, 'bngsim is not available'

    model_cls = getattr(bngsim, 'Model', None)
    if model_cls is None or not hasattr(model_cls, 'from_antimony'):
        return False, 'installed bngsim does not expose Antimony loading'

    if not ANTIMONY_AVAILABLE:
        return False, 'antimony is not installed. Install with: pip install antimony python-libsbml'

    if not LIBSBML_AVAILABLE:
        return False, 'python-libsbml is not installed. Install with: pip install antimony python-libsbml'

    return True, ''


BNGSIM_HAS_ANTIMONY, BNGSIM_ANTIMONY_ERROR = _detect_bngsim_antimony_support()


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
                'Failed to parse Antimony from %s: %s' % (source_desc, error)
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
                'Failed to parse Antimony from %s: %s' % (source_desc, error)
            )

        sbml_text = antimony.getSBMLString(module_name)
        if not sbml_text:
            error = _antimony_last_error() or 'Antimony did not produce SBML output'
            raise ModelError(
                'Failed to convert Antimony from %s to SBML: %s' % (source_desc, error)
            )
        return sbml_text
    finally:
        if clear_loads is not None:
            clear_loads()


class BngsimAntimonyModelNoTimeout(BngsimSbmlModelNoTimeout):
    def __init__(self, file, abs_file, pset=None, actions=(), save_files=False, integrator='cvode',
                 strict_ssa=True):
        if integrator != 'cvode':
            raise ModelError(
                'Antimony models currently support only sbml_integrator = cvode'
            )

        _require_bngsim_antimony_support()

        self.file_path = file
        self.abs_file_path = abs_file
        self.param_set = pset
        self.name = file[file.rfind('/') + 1:].rsplit('.ant', 1)[0]
        self.save_files = save_files
        self.actions = list(actions)
        self.integrator = integrator
        self.strict_ssa = bool(strict_ssa)
        self.suffixes = [(a.bng_codeword, a.suffix) for a in actions]
        self.stochastic = False
        self.mutants = [MutationSet()]

        try:
            with open(self.abs_file_path, encoding='utf-8', errors='replace') as fh:
                self._base_antimony_text = fh.read()
        except FileNotFoundError:
            raise

        self._base_sbml_text = _antimony_text_to_sbml_text(self._base_antimony_text, self.file_path)
        doc = _sbml_doc_from_text(self._base_sbml_text, self.file_path)
        self._species_names = tuple(
            doc.getModel().getSpecies(i).getId()
            for i in range(doc.getModel().getNumSpecies())
        )
        self._species_name_set = set(self._species_names)
        self._global_param_names = tuple(
            doc.getModel().getParameter(i).getId()
            for i in range(doc.getModel().getNumParameters())
        )
        self.global_param_names = self._global_param_names
        self.param_names = self._species_name_set.union(set(self._global_param_names))

        try:
            self._load_bngsim_model_from_path(self.abs_file_path)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise ModelError(
                'Failed to load model %s.ant - There were errors in parsing this Antimony file. '
                'See log for details.' % self.name
            ) from exc

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
            with open('%s.xml' % mutant_prefix, 'w') as out:
                out.write(self.model_text(mut=mut))


# Retained as an alias for backwards compatibility. bngsim now enforces the
# wall-clock budget in-process, so the subprocess wrapper is unnecessary.
BngsimAntimonyModel = BngsimAntimonyModelNoTimeout
