"""Classes for storing models, parameter sets, and the fitting trajectory"""


from .printing import print0, print1, PybnfError

import hashlib
import logging
import numpy as np
import re
import copy
import signal
from subprocess import Popen, STDOUT, PIPE, CalledProcessError
from .data import Data
from .priors import TruncatedPrior, build_prior
import heapq
import time
import traceback
import roadrunner as rr
import pickle
import os
import shutil
import tempfile
from sys import executable

rr.Logger.disableLogging()

logger = logging.getLogger(__name__)

INITIALIZATION_PRIOR = 'prior'
INITIALIZATION_BOUNDS = 'bounds'
INITIALIZATION_DISTRIBUTIONS = frozenset({
    INITIALIZATION_PRIOR,
    INITIALIZATION_BOUNDS,
})


_TFUN_FILE_REF_RE = re.compile(
    r'(?P<prefix>\btfun\s*\(\s*)(?P<quote>[\'"])(?P<path>[^\'"]+)(?P=quote)'
)


def _format_bngl_number(x):
    """Format a float for a BNGL ``sample_times`` / ``par_scan_vals`` list (ADR-0028).

    ``repr`` gives the shortest string that round-trips back to the same float, so a
    data point like ``0.0065`` stays ``0.0065`` (not ``0.006500000000001``) when it is
    re-parsed by BNG2.pl, bngsim, or ``find_t_length``. BioNetGen and bngsim both accept
    decimal and scientific notation.
    """
    return repr(float(x))


def _collapse_line_continuations(text):
    """Collapse BNGL-style trailing backslash continuations."""
    return re.sub(r'\\\s*\n\s*', '', text)


def _strip_hash_comments(text):
    """Remove # comments line by line for lightweight tfun path discovery."""
    stripped = []
    for line in text.splitlines():
        comment_index = line.find('#')
        stripped.append(line if comment_index < 0 else line[:comment_index])
    return '\n'.join(stripped)


def _extract_tfun_file_refs(text):
    """
    Return the ordered list of file-based lowercase tfun() path arguments.

    Inline tfun([..], [..], ..) calls are ignored because there is no external
    file dependency to stage.
    """
    refs = []
    collapsed = _collapse_line_continuations(text)
    uncommented = _strip_hash_comments(collapsed)
    for match in _TFUN_FILE_REF_RE.finditer(uncommented):
        path_ref = match.group('path').strip()
        if path_ref not in refs:
            refs.append(path_ref)
    return refs


def _rewrite_tfun_file_refs(text, path_map):
    """Rewrite lowercase file-based tfun() paths according to path_map."""

    def repl(match):
        path_ref = match.group('path')
        if path_ref not in path_map:
            return match.group(0)
        return '{}{}{}{}'.format(
            match.group('prefix'),
            match.group('quote'),
            path_map[path_ref],
            match.group('quote'),
        )

    return _TFUN_FILE_REF_RE.sub(repl, text)


def _stage_and_rewrite_tfun_files(text, source_dir, dest_dir):
    """
    Copy relative .tfun dependencies into dest_dir and rewrite tfun() paths.

    This keeps BNGL and .net files runnable after PyBNF stages them into a
    different working directory.
    """
    if source_dir is None:
        return text

    refs = _extract_tfun_file_refs(text)
    if len(refs) == 0:
        return text

    source_dir = os.path.abspath(source_dir if source_dir else os.curdir)
    dest_dir = os.path.abspath(dest_dir if dest_dir else os.curdir)
    path_map = {}

    for path_ref in refs:
        if os.path.isabs(path_ref):
            continue

        source_path = os.path.abspath(os.path.join(source_dir, path_ref))
        if not os.path.isfile(source_path):
            raise FileNotFoundError(
                f"Required tfun file '{path_ref}' referenced from '{source_dir}' was not found at '{source_path}'"
            )

        staged_name = '{}_{}'.format(
            hashlib.sha1(path_ref.encode('utf-8')).hexdigest()[:12],
            os.path.basename(path_ref),
        )
        staged_rel = os.path.join('__pybnf_tfun__', staged_name)
        staged_path = os.path.join(dest_dir, staged_rel)
        staged_parent = os.path.dirname(staged_path)
        if staged_parent and not os.path.isdir(staged_parent):
            os.makedirs(staged_parent)
        shutil.copy2(source_path, staged_path)
        path_map[path_ref] = staged_rel.replace(os.sep, '/')

    if len(path_map) == 0:
        return text

    return _rewrite_tfun_file_refs(text, path_map)


def _subprocess_env(cmd, env=None):
    """
    Return an environment for cmd, aligning BioNetGen scripts with their Perl2 root.

    BioNetGen's BNG2.pl prefers $BNGPATH/$BioNetGenRoot over its own location when
    resolving Perl modules. If the user's shell exports an older BNGPATH, invoking a
    newer checkout via an absolute bng_command path can silently load mismatched Perl2
    modules. When cmd includes BNG2.pl, force those variables to the script's parent
    directory so the command uses a self-consistent BioNetGen tree.
    """
    bng_root = None
    for token in cmd:
        if not isinstance(token, str):
            continue
        expanded = os.path.expanduser(token)
        if os.path.basename(expanded) == 'BNG2.pl':
            bng_root = os.path.dirname(os.path.abspath(expanded))
            break

    if bng_root is None:
        return env

    resolved_env = dict(os.environ)
    if env is not None:
        resolved_env.update(env)
    resolved_env['BNGPATH'] = bng_root
    resolved_env['BioNetGenRoot'] = bng_root
    return resolved_env


# --------------------------------------------------------------------------- #
# Live-simulation registry (node-local) for aborting a fit cleanly.
#
# Simulation subprocesses are launched with start_new_session=True (see
# run_subprocess), which detaches each into its own session/process group. That
# is needed so a timeout can killpg the sim's whole tree without touching PyBNF
# -- but it also means a terminal Ctrl-C (delivered only to PyBNF's foreground
# process group) never reaches a running sim. And on abort the dask worker that
# launched the sim is SIGKILLed by its nanny, so no in-worker cleanup runs; the
# sim re-parents to init and orphans (the "Ctrl-C does nothing; kill -9 needed"
# problem). To kill in-flight sims reliably, run_subprocess records each sim's
# process-group id (PGID) as an empty file named by the PGID under a node-local
# directory, and pybnf.main() reaps that directory on *any* exit (Ctrl-C, crash,
# or normal completion) by SIGKILL-ing each recorded group -- see
# reap_active_sims(). The directory is node-local (each host's own tempdir), so
# its PGIDs are only ever killed by a process on the same host, because killpg is
# host-local. Remote dask-ssh worker nodes get no registry (the env var below is
# not inherited across ssh); their orphans are cleaned by SLURM's cgroup tracking
# at step/job teardown.
#
# Storage is bounded: the files are empty (the PGID *is* the filename) and are
# removed when their sim finishes, so the live count tracks the number of
# concurrent sims (~= the worker count), never the total number of evaluations.
# The whole directory is rmtree'd at run end (clear_sim_registry); a startup
# sweep (set_sim_registry) clears any directory orphaned by a prior `kill -9`.
# --------------------------------------------------------------------------- #
_REGISTRY_ENV_VAR = 'PYBNF_SIM_REGISTRY'


def _registry_path(run_id):
    """Node-local registry directory path for ``run_id`` (sanitised to a safe
    path component). Uses this host's own tempdir, so the same run_id resolves to
    a distinct, node-local directory on every machine."""
    safe = re.sub(r'[^A-Za-z0-9._-]', '_', str(run_id))
    return os.path.join(tempfile.gettempdir(), f'pybnf_sims_{safe}')


def _registry_dir():
    """This process's node-local registry directory (created on demand), or None
    if the registry is disabled (``PYBNF_SIM_REGISTRY`` unset). The env var is the
    single source of truth -- set by pybnf.main() and inherited by locally-spawned
    dask workers -- so workers need no broadcast to find the same directory."""
    run_id = os.environ.get(_REGISTRY_ENV_VAR)
    if not run_id:
        return None
    path = _registry_path(run_id)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        logger.debug('Could not create sim registry %s', path, exc_info=True)
        return None
    return path


def set_sim_registry(run_id):
    """Enable the live-sim registry for this run (called once by pybnf.main()).

    Sets ``PYBNF_SIM_REGISTRY`` so locally-spawned dask workers inherit it, sweeps
    stale registry directories from previous ``kill -9``'d runs, and creates this
    run's directory up front.
    """
    os.environ[_REGISTRY_ENV_VAR] = str(run_id)
    _sweep_stale_sim_registries()
    _registry_dir()


def _register_sim(pgid):
    """Record a running sim's process group; best-effort, never blocks the sim."""
    d = _registry_dir()
    if d:
        try:
            # 'x': the PGID is the filename, and concurrent workers can't clobber.
            open(os.path.join(d, str(pgid)), 'x').close()
        except OSError:
            pass


def _deregister_sim(pgid):
    """Drop a finished sim's process group from the registry; best-effort."""
    d = _registry_dir()
    if d:
        try:
            os.remove(os.path.join(d, str(pgid)))
        except OSError:
            pass


def reap_active_sims():
    """SIGKILL every process group recorded in this node's registry, returning the
    PGIDs killed. Called from pybnf.main()'s finally on every exit -- a no-op on
    normal completion (finished sims have deregistered). Already-dead groups
    (ProcessLookupError) and non-PGID filenames are skipped."""
    d = _registry_dir()
    if not d:
        return []
    reaped = []
    for name in os.listdir(d):
        try:
            pgid = int(name)
        except ValueError:
            continue  # not a PGID entry
        try:
            os.killpg(pgid, signal.SIGKILL)
            reaped.append(pgid)
        except ProcessLookupError:
            pass  # already gone
        except OSError:
            logger.debug('Could not kill sim process group %d', pgid, exc_info=True)
        try:
            os.remove(os.path.join(d, name))
        except OSError:
            pass
    return reaped


def clear_sim_registry():
    """Remove this run's registry directory and disable the registry (called at
    normal run end so the directory does not linger in tempdir)."""
    d = _registry_dir()
    if d:
        shutil.rmtree(d, ignore_errors=True)
    os.environ.pop(_REGISTRY_ENV_VAR, None)


def _sweep_stale_sim_registries(max_age_seconds=86400):
    """Remove ``pybnf_sims_*`` directories older than ``max_age_seconds`` (default
    1 day) -- registries orphaned by a previous ``kill -9`` that bypassed
    clear_sim_registry. The age guard ensures a concurrent run's registry is never
    touched. Best-effort housekeeping; failures are ignored."""
    tmp = tempfile.gettempdir()
    try:
        entries = os.listdir(tmp)
    except OSError:
        return
    now = time.time()
    for name in entries:
        if not name.startswith('pybnf_sims_'):
            continue
        path = os.path.join(tmp, name)
        try:
            if os.path.isdir(path) and now - os.path.getmtime(path) > max_age_seconds:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


def run_subprocess(cmd, timeout, stdout=None, stderr=None, input=None, env=None):
    """
    Run a subprocess with process-group-based cleanup on timeout.

    Uses start_new_session=True so that on timeout, the entire process group
    (including any grandchild processes) is killed via os.killpg(). This prevents
    zombie processes when e.g. run_network spawns children that outlive the parent.

    On Windows, falls back to proc.kill() (no process group support).

    :param cmd: Command to run (list of strings)
    :param timeout: Timeout in seconds, or None for no timeout
    :param stdout: File object or subprocess constant for stdout
    :param stderr: File object or subprocess constant for stderr
    :param input: Bytes to send to stdin, or None
    :param env: Optional environment overrides
    :raises CalledProcessError: If the subprocess exits with non-zero return code
    :raises TimeoutExpired: If the subprocess exceeds the timeout
    :return: stdout bytes if stdout=PIPE, else None
    """
    use_pgid = (os.name != 'nt')
    proc = Popen(cmd, stdout=stdout, stderr=stderr,
                 stdin=PIPE if input is not None else None,
                 start_new_session=use_pgid,
                 env=_subprocess_env(cmd, env))
    # Record the sim's process group so an aborting fit can reap it (see the
    # registry block above). getpgid can race a sim that exits immediately.
    pgid = None
    if use_pgid:
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            pgid = None
    if pgid is not None:
        _register_sim(pgid)
    try:
        try:
            stdout_data, _ = proc.communicate(input=input, timeout=timeout)
        except BaseException:
            # Any interruption -- TimeoutExpired, KeyboardInterrupt, task
            # cancellation -- must take the whole detached sim process group down
            # with it, so a sim never outlives the call that launched it.
            # (Previously only TimeoutExpired was handled, orphaning the sim on
            # every other interruption.)
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                proc.kill()
            proc.wait()
            raise
    finally:
        if pgid is not None:
            _deregister_sim(pgid)
    if proc.returncode != 0:
        raise CalledProcessError(proc.returncode, cmd)
    return stdout_data


class Model:
    """
    An abstract class representing an executable model
    """

    def copy_with_param_set(self, pset):
        """Returns a copy of the model with a new parameter set

        :param pset: A new parameter set
        :type pset: PSet
        :return: Model
        """
        raise NotImplementedError("copy_with_param_set is not implemented")

    def save(self, file_prefix, **kwargs):
        """
        Saves the model to file

        :return:
        """
        raise NotImplementedError("save is not implemented")

    def save_all(self, file_prefix):
        logger.warning(f'Model of type {type(self)} does not implement save_all(). Falling back to save()')
        self.save(file_prefix)

    def execute(self, folder, filename, timeout):
        """
        Executes the model, working in folder/filename, with a max runtime of timeout.
        Loads the resulting data, and returns a dictionary mapping suffixes to data objects. For model types without a
        notion of suffixes, the dictionary will contain one key mapping to one Data object

        :param folder: The folder to save to, eg 'Simulations/init22'
        :param filename: The name of the model file to create, not including the extension, eg 'init22'
        :param timeout: Maximum runtime in seconds
        :return: dict of Data
        """
        raise NotImplementedError("Subclasses of Model must override execute()")

    def add_action(self, action):
        raise PybnfError(f'Model type {type(self)} does not support adding actions')

    def get_suffixes(self):
        """
        Return a list of valid data suffixes to use in this model, including all combinations of action suffix +
        mutation name
        """
        raise NotImplementedError('Subclasses of Model must implement get_suffixes()')

    def add_mutant(self, mut_set):
        """
        Add a mutant to run along with this model
        :param mut_set: MutationSet that should be applied to this mutant
        :type mut_set: MutationSet
        :return:
        """
        try:
            self.mutants.append(mut_set)
        except AttributeError:
            raise PybnfError(f'Model type {type(self)} does not support adding mutations')


class BNGLModel(Model):
    """
    Class representing a BNGL model


    """

    def __init__(self, bngl_file, pset=None, suppress_free_param_error=False):
        """
        Loads the model from the given .bngl file

        :param bngl_file: str address of the bngl file
        :param pset: PSet to initialize the model with. Defaults to None
        :param suppress_free_param_error: If True, suppress the error that would occur if the model has no free
        parameters declared
        """
        self.file_path = bngl_file
        self.name = re.sub(".bngl", "", self.file_path[self.file_path.rfind("/")+1:])
        self.suffixes = []  # list of 2-tuples (sim_type, prefix)
        self.bng_command = ''

        # Read the file
        with open(self.file_path, encoding='utf-8', errors='replace') as file:
            self.bngl_file_text = file.read()

        # Scan the file's lines
        # Check for various things to fill out all of the following attributes needed for model writing
        self.generates_network = False
        self.generate_network_line = None
        self.seeded = False
        self.actions = []
        self.protocol = []
        self.mutants = []
        self.stochastic = False  # Update during parsing. Used to warn about misuse of 'smoothing'
        self.has_observables = False
        param_names_set = set()
        self.split_line_index = None  # for insertion of free parameters
        all_lines = [x.strip() for x in self.bngl_file_text.splitlines()]
        skip_lines = set()  # Indices of lines that should not go into self.model_lines

        in_action_block = False
        in_protocol_block = False
        in_no_block = True
        in_observables_block = False
        continuation = ''
        continuation_raw = ''
        continuation_indices = set()
        for i, rawline in enumerate(all_lines):
            commenti = rawline.find('#')
            line = rawline if commenti == -1 else rawline[:commenti]

            if re.match(r'^\s*$', line):
                # Blank or comment. Handle before continuation
                if in_action_block:
                    # Keep it in the actions block
                    self.actions.append(rawline)
                    skip_lines.add(i)
                elif in_protocol_block:
                    self.protocol.append(rawline)
                    skip_lines.add(i)
                continue

            # Handle case where '\' is used to continue on the next line
            line = continuation + line
            rawline = continuation_raw + rawline
            indices = continuation_indices.union({i})  # The set of indices that are all part of this line due to continuation
            continuation = ''
            continuation_raw = ''
            continuation_indices = set()
            continue_match = re.search(r'\\\s*$', line)
            if continue_match:
                # This line continues on the next line
                continuation = line[:continue_match.start()]
                continuation_raw = rawline + '\n'
                continuation_indices = indices
                continue

            # Find every item matching [alphanumeric]__FREE
            params = re.findall(r'[A-Za-z_]\w*__FREE', line)
            for p in params:
                param_names_set.add(p)

            # Make sure setOption (if present) doesn't get passed to the actions block
            if re.match(r'\s*(setOption|setModelName|substanceUnits|version)', line):
                continue

            # Check if this is the 'begin parameters' line
            if re.match(r'begin\s+parameters', line.strip()):
                # Generate index into self.model_lines based on i and number of skipped lines (probably 0 at this point)
                self.split_line_index = i + 1 - len(skip_lines)

            # "begin model" doesn't work like a regular block, so escape before we start handling blocks.
            if re.match(r'(begin|end)\s+model', line.strip()):
                continue

            if re.match(r'begin\s+actions', line.strip()):
                in_action_block = True
                in_no_block = False
                skip_lines.update(indices)
                continue
            elif re.match(r'end\s+actions', line.strip()):
                in_action_block = False
                in_no_block = True
                skip_lines.update(indices)
                continue

            if re.match(r'begin\s+protocol', line.strip()):
                in_protocol_block = True
                in_no_block = False
                skip_lines.update(indices)
                continue
            elif re.match(r'end\s+protocol', line.strip()):
                in_protocol_block = False
                in_no_block = True
                skip_lines.update(indices)
                continue

            if in_protocol_block:
                skip_lines.update(indices)
                self.protocol.append(rawline)
                continue

            # To keep track of whether we're in no block, which counts as an action block, check for
            # begin and end keywords
            if re.match(r'begin\s+[a-z][a-z\s]*', line.strip()):
                in_no_block = False
                if re.match(r'begin\s+observables', line.strip()):
                    in_observables_block = True
            elif in_observables_block:
                if re.match(r'end\s+observables', line.strip()):
                    in_observables_block = False
                else:
                    self.has_observables = True

            if in_action_block or in_no_block:
                skip_lines.update(indices)
                action_suffix = self._get_action_suffix(line)
                if action_suffix is not None:
                    self.suffixes.append(action_suffix)

                if re.match('generate_network', line.strip()):
                    self.generates_network = True
                    self.generate_network_line = line
                    continue
                if re.search('simulate_((ode)|(ssa)|(pla))', line) or re.search(
                        '(simulate|parameter_scan|bifurcate).*method=>(\'|")((ode)|(ssa)|(pla)|(protocol))("|\')', line):
                    self.generates_network = True  # in case there is no "generate_network" command present
                if re.search('simulate_((nf)|(ssa)|(pla))', line) or re.search(
                        '(simulate|parameter_scan|bifurcate).*method=>(\'|")'
                        '((nf)|(nf_reject)|(nfsim)|(rm)|(rulemonkey)|(nf_exact)|(ssa)|(pla))("|\')', line):
                    self.stochastic = True
                if re.search(r'seed=>\d+', line):
                    self.seeded = True
                self.actions.append(rawline)

            if re.match(r'end\s+[a-z][a-z\s]*', line.strip()):
                in_no_block = True

        if self.split_line_index is None:
            raise ModelError("'begin parameters' not found in BNGL file")
        self.model_lines = [all_lines[i] for i in range(len(all_lines)) if i not in skip_lines]
        if self.generates_network and self.generate_network_line is None:
            self.generate_network_line = 'generate_network({overwrite=>1})'

        # The full ``begin parameters`` namespace: every parameter id, in source
        # order. Distinct from ``param_names`` (the legacy ``__FREE`` tokens scanned
        # above): a new-era (edition >= 2) config free parameter binds to a model
        # parameter *by id* (ADR-0034), the same contract the SBML backend uses
        # (``param_names`` = species + globals), so bind-by-id and its typo check
        # resolve against this set rather than the marker tokens. Parsed with the
        # same rule the in-process bngsim NF backend applies to ``model_lines``
        # (``_parse_bngl_param_block``) so the exposed namespace is exactly what the
        # engine binds; a test pins the two together.
        self.model_param_names = self._parse_param_block_names(self.model_lines)

        if len(param_names_set) == 0 and not suppress_free_param_error:
            raise ModelError(f"No free parameters found in model {bngl_file}. Your model file needs to include variable names "
                             "that end in '__FREE' to tell BioNetFit which parameters to fit.")

        # Save model_params as a sorted tuple
        param_names_list = list(param_names_set)
        param_names_list.sort()
        self.param_names = tuple(param_names_list)

        if pset:
            # If this model is to be initialized with a PSet, check that it has the correct parameter names
            if pset.keys_to_string() != '\t'.join(self.param_names):
                raise ValueError('Parameter names in the PSet do not match those in the Model')

        self.param_set = pset

    @staticmethod
    def _parse_param_block_names(model_lines):
        """Every parameter id declared in the ``begin parameters`` block, in order.

        The full bind-by-id namespace exposed as ``model_param_names`` (ADR-0034).
        Reads the comment-stripped, continuation-joined ``model_lines`` already
        built by the constructor; the name is the first token of a ``name = expr``
        or ``name expr`` definition. Kept byte-for-byte in step with the in-process
        bngsim NF backend's ``_parse_bngl_param_block`` (which parses the same
        ``model_lines``) so the namespace this advertises is the one the engine
        actually binds.
        """
        names = []
        in_block = False
        for raw_line in model_lines:
            line = raw_line.strip()
            comment_idx = line.find('#')
            if comment_idx >= 0:
                line = line[:comment_idx].strip()
            if not line:
                continue
            if re.match(r'begin\s+parameters', line):
                in_block = True
                continue
            if re.match(r'end\s+parameters', line):
                break
            if not in_block:
                continue
            eq_match = re.match(r'([A-Za-z_]\w*)\s*=\s*(.+)', line)
            if eq_match:
                names.append(eq_match.group(1))
                continue
            space_match = re.match(r'([A-Za-z_]\w*)\s+(.+)', line)
            if space_match:
                names.append(space_match.group(1))
        return tuple(names)

    @staticmethod
    def _get_action_suffix(line):
        sim_match = re.match("(simulate|parameter_scan)", line.strip())
        if sim_match:
            act_type = sim_match.group(1)
            match = re.search(r"""suffix\s*=>\s*['"](.*?)['"]\s*[,}]""", line)
            if match is not None:
                return act_type, match.group(1)
        return None

    def find_t_length(self):
        """
        Builds a dict mapping each simulate action's suffix to the number of output
        time points minus one, which is used to size the trajectory-output arrays
        (an array of length ``time + 1`` holds one entry per output row).

        The number of output rows depends on how the simulation length is specified:

        * ``n_steps=>N`` produces ``N + 1`` rows (including ``t_start``), so ``time = N``.
        * ``sample_times=>[t0,...,tM]`` produces one row per listed time, so
          ``time = len(sample_times) - 1``.

        :return: dict keyed on suffix string with integer values
        """
        # Join line continuations so each action occupies a single line.
        lines = self.bngl_file_text.replace('\\\n', '').split('\n')
        timeDict = {}
        for line in lines:
            # Skip commented-out lines (including any possible commented-out
            # actions with n_steps/sample_times in them).
            if '#' in line or 'simulate' not in line:
                continue
            suffix_match = re.search(r"suffix\s*=>\s*['\"](.*?)['\"]", line)
            if suffix_match is None:
                continue
            suffix = suffix_match.group(1)

            sample_match = re.search(r"sample_times\s*=>\s*\[(.*?)\]", line)
            n_steps_match = re.search(r"n_steps\s*=>\s*(\d+)", line)
            if sample_match is not None:
                # One output row per listed sample time.
                n_times = len([t for t in sample_match.group(1).split(',') if t.strip() != ''])
                timeDict[suffix] = n_times - 1
            elif n_steps_match is not None:
                timeDict[suffix] = int(n_steps_match.group(1))
            else:
                raise PybnfError(
                    "Could not determine the simulation length for the simulate action with "
                    f"suffix '{suffix}' in model {self.file_path}: the action specifies neither n_steps nor "
                    "sample_times.")

        return timeDict

    def copy_with_param_set(self, pset):
        """
        Returns a copy of this model containing the specified parameter set.

        :param pset: A PSet object containing the parameters for the new instance
        :type pset: PSet
        :return: BNGLModel
        """
        # Check that the PSet has definitions for the right parameters for this model
        if not set(pset.keys()) >= set(self.param_names):
            raise PybnfError(f'Parameter names in the PSet do not match those in the Model\n{pset.keys()}\n{self.param_names}')

        # Legacy must-fit completeness check: every declared __FREE token should be in
        # the pset (a shared multi-model pset legitimately carries more). A new-era
        # model (ADR-0034) declares no __FREE markers -- its free parameters bind by id
        # and come from the conf -- so there is no must-fit set to be missing, and this
        # warning would fire on every evaluation; gate it on param_names being present.
        if self.param_names and set(pset.keys()) != set(self.param_names):
            logger.warning(f'Model {self.name} does not contain all defined free parameters')

        newmodel = copy.deepcopy(self)
        newmodel.param_set = pset
        return newmodel

    def model_text(self, gen_only=False):
        """
        Returns the text of a runnable BNGL file, which includes the contents of the original BNGL file, and also values
        assigned to each __FREE parameter, as determined by this model's PSet

        :return: str
        """

        # Check that the model has an associated PSet
        if self.param_set is None:
            raise ModelError('Must assign a PSet to the model before calling model_text()')

        if len(self.actions) == 0:
            raise ModelError("No actions found in model")

        # Generate the text associated with defining __FREE parameter values
        param_text_lines = [f'{k} {str(self.param_set[k])}' for k in self.param_names]

        # New-era bind-by-id (ADR-0034): a free parameter named like a real parameter id
        # (no __FREE marker) overrides that parameter's value *in place* in the
        # parameters block -- the same value the in-process backend would set via
        # set_param -- rather than injecting a duplicate definition. A legacy pset holds
        # only __FREE tokens, which are disjoint from the parameter ids
        # (model_param_names), so this branch never fires for a legacy job and leaves its
        # output byte-identical.
        model_param_names = getattr(self, 'model_param_names', ())
        bind_by_id = {k: self.param_set[k] for k in self.param_set.keys()
                      if k in model_param_names and k not in self.param_names}
        source_lines = (self._override_param_block_values(self.model_lines, bind_by_id)
                        if bind_by_id else self.model_lines)

        # Insert the generated text at the correct point within the text of the model
        if gen_only:
            action_lines = [
                'begin actions\n',
                self.generate_network_line + '\n',
                'end actions'
            ]
        else:
            action_lines = ['begin actions\n']
            if self.generates_network:
                action_lines.append(self.generate_network_line)
            action_lines += self.actions + ['end actions']

        protocol_lines = []
        if self.protocol:
            protocol_lines = ['begin protocol\n'] + self.protocol + ['end protocol\n']

        all_lines = \
            source_lines[:self.split_line_index] + \
            param_text_lines + \
            source_lines[self.split_line_index:] + \
            protocol_lines + \
            action_lines

        return '\n'.join(all_lines) + '\n'

    @staticmethod
    def _override_param_block_values(model_lines, overrides):
        """A copy of ``model_lines`` with each ``begin parameters`` definition whose id
        is in ``overrides`` rewritten to ``<id> <value>`` (new-era bind-by-id, ADR-0034).

        Only the parameters block is touched; every other line passes verbatim. The id
        is extracted exactly as :meth:`_parse_param_block_names` extracts it, so an
        override keyed by a ``model_param_names`` id rewrites the one line that defines
        it (replacing its expression/nominal value, and any trailing comment, with the
        fit value)."""
        out = []
        in_block = False
        for raw_line in model_lines:
            body = raw_line.strip()
            comment_idx = body.find('#')
            if comment_idx >= 0:
                body = body[:comment_idx].strip()
            if re.match(r'begin\s+parameters', body):
                in_block = True
                out.append(raw_line)
                continue
            if re.match(r'end\s+parameters', body):
                in_block = False
                out.append(raw_line)
                continue
            if in_block and body:
                eq_match = re.match(r'([A-Za-z_]\w*)\s*=\s*(.+)', body)
                space_match = eq_match or re.match(r'([A-Za-z_]\w*)\s+(.+)', body)
                if space_match and space_match.group(1) in overrides:
                    name = space_match.group(1)
                    out.append(f'{name} {str(overrides[name])}')
                    continue
            out.append(raw_line)
        return out

    def save(self, file_prefix, gen_only=False, pset=None):
        """
        Saves a runnable BNGL file of the model, including definitions of the __FREE parameter values that are defined
        by this model's pset, to the specified location.

        :param file_prefix: str, path where the file should be saved
        :param gen_only: bool, output model with only generate_network action if True
        """

        # Call model_text(), then write the output to the file.
        if self.param_set is None:
            self.param_set = pset

        text = self.model_text(gen_only)
        text = _stage_and_rewrite_tfun_files(
            text,
            os.path.dirname(self.file_path),
            os.path.dirname(file_prefix),
        )
        with open(file_prefix + '.bngl', 'w') as f:
            f.write(text)

    def save_all(self, file_prefix):
        """
        Saves BNGL files of the original model and all mutants
        :param file_prefix:
        """
        self.save(file_prefix)
        for mut in self.mutants:
            mut_model = self._get_mutant_model(mut)
            mut_model.save(file_prefix+mut.suffix)

    def execute(self, folder, filename, timeout, with_mutants=True):
        """

        :param folder: Folder in which to do all the file creation
        :return: Data object
        """
        # Create the modified BNGL file
        file = f'{folder}/{filename}'
        self.save(file)

        # Run BioNetGen
        cmd = [self.bng_command, f'{file}.bngl', '--outdir', folder]
        log_file = f'{file}.log'
        if os.name == 'nt':  # Windows
            # Explicitly call perl because the #! line in BNG2.pl is not supported.
            cmd = ['perl'] + cmd
        with open(log_file, 'w') as lf:
            run_subprocess(cmd, timeout=timeout, stdout=lf, stderr=STDOUT)

        # Load the data file(s)
        ds = self._load_simdata(folder, filename)

        if with_mutants:
            for mut in self.mutants:
                # Inefficient iteration over PSet to build the mutant one, but hopefully not performance-critical
                logger.debug(f'Working on mutant {mut.suffix}')
                mut_model = self._get_mutant_model(mut)
                mut_data = mut_model.execute(folder, filename+mut.suffix, timeout, with_mutants=False)
                for suff in mut_data:
                    ds[suff + mut.suffix] = mut_data[suff]
                logger.debug(f'Finished mutant {mut.suffix}')
        return ds

    def _get_mutant_model(self, mut):
        """
        Creates a copy of the model, with the parameter set changed as specified by MutationSet mut
        :param mut: The MutationSet to apply
        """
        params = {p.name: p.value for p in self.param_set}
        for mi in mut:
            params[mi.name] = mi.mutate(params[mi.name])
        mut_param_list = [FreeParameter(pname, 'uniform_var', -np.inf, np.inf, value=params[pname], bounded=True)
                          for pname in params]
        mut_pset = PSet(mut_param_list)
        mut_model = self.copy_with_param_set(mut_pset)
        return mut_model

    def _load_simdata(self, folder, filename):
        """
        Function to load simulation data after executing all simulations for an evaluation

        Returns a nested dictionary structure.  Top-level keys are model names and values are
        dictionaries whose keys are action suffixes and values are Data instances

        :return: dict of Data
        """
        ds = {}
        for suff in self.suffixes:
            if suff[0] == 'simulate':
                data_file = f'{folder}/{filename}_{suff[1]}.gdat'
                data = Data(file_name=data_file)
            else:  # suff[0] == 'parameter_scan'
                data_file = f'{folder}/{filename}_{suff[1]}.scan'
                data = Data(file_name=data_file)
            ds[suff[1]] = data
        return ds

    def add_action(self, action):
        """Append a config-file action as a BNGL action string.

        Translates a :class:`TimeCourse` or :class:`ParamScan` object into a
        BNGL action line and appends it to the model's action list.  Only a
        subset of BioNetGen arguments are supported here; for full control,
        write actions in the BNGL file's ``begin actions`` block instead.
        """
        if isinstance(action, TimeCourse):
            if action.explicit_points is not None:
                # New-era explicit output points (ADR-0028): emit BioNetGen sample_times
                # so the gdat lands on exactly the data's time points. Covers BNG2.pl
                # (BNGAction.pm) and bngsim (parses the bracket list out of the action
                # text). BNG requires 3 or more sample times; explicit_points already
                # includes t=0, so a single positive data time still yields 2 -- guard
                # for the (rare) too-short grid with a clear message.
                if len(action.explicit_points) < 3:
                    raise PybnfError(
                        f'The time_course action with suffix "{action.suffix}" has only '
                        f'{len(action.explicit_points)} explicit output time point(s) (including t=0), but '
                        "BioNetGen's sample_times requires 3 or more. Provide a data set with at least two "
                        'positive time points.')
                sample_times = ','.join(_format_bngl_number(p) for p in action.explicit_points)
                line = f'simulate({{method=>"{action.method}",t_start=>0,sample_times=>[{sample_times}],suffix=>"{action.suffix}",print_functions=>1}})'
            else:
                line = f'simulate({{method=>"{action.method}",t_start=>0,t_end=>{action.time},n_steps=>{action.stepnumber},suffix=>"{action.suffix}",print_functions=>1}})'
        elif isinstance(action, ParamScan):
            if action.explicit_points is not None:
                # New-era explicit scan values (ADR-0028): emit par_scan_vals so the scan
                # samples exactly the data's swept-parameter values. par_min/par_max/
                # n_scan_pts are deliberately omitted -- when present BioNetGen ignores
                # par_scan_vals (BNGAction.pm: "defined min/max takes precedence").
                par_scan_vals = ','.join(_format_bngl_number(p) for p in action.explicit_points)
                line = f'parameter_scan({{parameter=>"{action.param}",method=>"{action.method}",t_start=>0,t_end=>{action.time},' \
                       f'par_scan_vals=>[{par_scan_vals}],log_scale=>{action.logspace},suffix=>"{action.suffix}",print_functions=>1}})'
            else:
                line = f'parameter_scan({{parameter=>"{action.param}",method=>"{action.method}",t_start=>0,t_end=>{action.time},par_min=>{action.min},par_max=>{action.max},' \
                       f'n_scan_pts=>{action.stepnumber + 1},log_scale=>{action.logspace},suffix=>"{action.suffix}",print_functions=>1}})'
        else:
            raise RuntimeError(f'Unknown action type {type(action)}')
        # Config actions are assumed to be independent, so need to reset concentrations before each one.
        self.actions.append('resetConcentrations()')
        self.actions.append(line)
        self.generates_network = True
        if self.generate_network_line is None:
            self.generate_network_line = 'generate_network({overwrite=>1})'
        self.suffixes.append((action.bng_codeword, action.suffix))

    def get_suffixes(self):
        """
        Return a list of valid data suffixes to use in this model, including all combinations of action suffix +
        mutation name
        """
        result = []
        for s in self.suffixes:
            result.append(s[1])
            for mut in self.mutants:
                result.append(s[1]+mut.suffix)
        return result



class NetModel(BNGLModel):
    def __init__(self, name, acts, suffs, mutants, ls=None, nf=None, source_dir=None):
        self.name = name
        self.actions = acts
        self.config_actions = []
        self.suffixes = suffs
        self.mutants = mutants
        self.param_set = None
        self.bng_command = ''
        self.net_file_dir = None

        if not (ls or nf):
            raise ModelError("Must specify a file name or a list of strings corresponding to the .net file's lines")
        elif ls:
            self.netfile_lines = ls
            if source_dir is not None:
                self.net_file_dir = os.path.abspath(source_dir)
        else:
            self.file_name = nf
            self.net_file_dir = os.path.dirname(os.path.abspath(nf))
            with open(self.file_name, encoding='utf-8', errors='replace') as f:
                self.netfile_lines = f.readlines()

    def copy_with_param_set(self, pset):
        """
        Returns a copy of the model in .net format, but with a new parameter set

        :param pset: A set of new parameters for the model
        :type pset: PSet
        :return: NetModel
        """
        lines_copy = copy.deepcopy(self.netfile_lines)
        in_params_block = False
        for i, l in enumerate(lines_copy):
            if re.match(r'begin\s+parameters', l.strip()):
                in_params_block = True
            elif re.match(r'end\s+parameters', l.strip()):
                in_params_block = False
            elif in_params_block:
                m = re.match(r'(\s+)(\d+)\s+([A-Za-z_]\w*)(\s+)([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)(?=\s+)', l)
                if m:
                    if m.group(3) in pset.keys():
                        lines_copy[i] = f'{m.group(1)}{m.group(2)} {m.group(3)}{m.group(4)}{str(pset[m.group(3)])}\n'

        newmodel = NetModel(self.name, self.actions, self.suffixes, self.mutants, ls=lines_copy,
                            source_dir=self.net_file_dir)
        newmodel.bng_command = self.bng_command
        newmodel.param_set = pset
        return newmodel

    def save(self, file_prefix):
        net_text = ''.join(self.netfile_lines)
        net_text = _stage_and_rewrite_tfun_files(
            net_text,
            self.net_file_dir,
            os.path.dirname(file_prefix),
        )
        with open(file_prefix + '.net', 'w') as wf:
            wf.write(net_text)
        with open(file_prefix + '.bngl', 'w') as wf:
            wf.write('readFile({file=>"%s"})\n' % (file_prefix + '.net'))
            wf.write('begin actions\n\n{}\n\nend actions\n'.format('\n'.join(self.actions)))


# Process-level cache of compiled RoadRunner instances, keyed by the model's
# absolute file path. A RoadRunner depends only on the model *structure* (the
# compiled SBML), not on FreeParameter values -- those are applied per
# evaluation by SbmlModelNoTimeout._modify_params. Compiling one from XML
# costs ~3.8 ms on raf.xml; reusing one and re-applying parameters via
# setattr/init + reset is ~0.01 ms (a ~400x per-evaluation saving on the
# default SBML backend). The runner is not picklable and the model is
# unpickled fresh in each dask worker per evaluation, so an instance attribute
# would rebuild the runner every evaluation; the cache lives at module scope
# (one per worker process) to amortize across the fit -- mirroring the
# engine-template cache added for issue #415 in bngsim_sbml_model.py. See
# SbmlModelNoTimeout._load_runner.
#
# WARNING: do NOT introduce any in-process RoadRunner ``loadState`` call while
# a runner is cached here. RoadRunner 2.9.2's ``loadState`` has a global side
# effect that silently disables ``setattr``-based global-parameter writes on
# every *previously built* runner (verified: a fresh ``RoadRunner(xml)`` is
# unaffected, but a cached one is poisoned). That is why the cached runner is
# always compiled from XML rather than restored from a saved ``.rr`` state --
# the per-evaluation reload that state caching used to optimize no longer
# exists once the runner is reused, so loadState would buy nothing and risk
# corrupting results.
_RUNNER_CACHE = {}


class SbmlModelNoTimeout(Model):

    def __init__(self, file, abs_file, pset=None, actions=(), save_files=False, integrator='cvode'):
        """
        :param file: The file path to the model as it was defined in the config. Used when indexing into the config dict
        :param abs_file: The absolute file path to the model. Used to actually load the model
        :param pset: The parameter set for the model
        :param actions: Iterable of actions to run on the model
        :param save_files: Whether to save the simulation output to file each time the model is run
        """

        self.file_path = file
        self.abs_file_path = abs_file
        self.param_set = pset
        self.name = re.sub(".xml", "", self.file_path[self.file_path.rfind("/") + 1:])
        self.save_files = save_files
        self.actions = list(actions)
        self.integrator = integrator
        self.suffixes = [a.suffix for a in actions]
        self.stochastic = True if integrator == 'gillespie' else False
        self.mutants = [MutationSet()]  # Start with one MutationSet containing no mutations (ie the model as is)

        try:
            rr.Logger.enableConsoleLogging()
            runner = rr.RoadRunner(self.abs_file_path)
            rr.Logger.disableLogging()
        except RuntimeError:
            # XML was not found, or had a bug in it, or some other problem in RoadRunner
            logger.exception(f'Failed to load model {self.name} in Roadrunner')
            exceptiondata = traceback.format_exc().splitlines()
            if f'could not open {file} as a file' in exceptiondata[-1]:
                message = f'File {file} was not found'
            else:
                message = 'There were errors in parsing this SBML file. See log for details.'
            raise PybnfError(f'Failed to load model {self.name}.xml - {message}')

        self.species_names = set(runner.model.getFloatingSpeciesIds()).union(set(runner.model.getBoundarySpeciesIds()))
        self.global_param_names = tuple(runner.model.getGlobalParameterIds())
        self.param_names = self.species_names.union(set(self.global_param_names))

        logger.debug(f'Loaded model {self.name} with Roadrunner')

    def copy_with_param_set(self, pset):

        newmodel = copy.deepcopy(self)
        newmodel.param_set = pset
        return newmodel

    def _build_runner(self):
        """Compile a fresh RoadRunner from the model's XML.

        Always loads from XML (never ``loadState``): the runner is built once
        per process and reused, so the per-evaluation reload that a saved
        compiled state would speed up no longer happens, and ``loadState``
        would poison other cached runners (see the ``_RUNNER_CACHE`` note).
        ``setattr`` on an XML-loaded runner is PyBNF's long-standing,
        always-correct parameter-mutation path.
        """
        return rr.RoadRunner(self.abs_file_path)

    def _load_runner(self):
        """Return a RoadRunner ready for this evaluation, reusing one per process.

        The runner is parameter-independent (FreeParameter values are applied
        per evaluation by :meth:`_modify_params`), so it is built once per
        worker process and reused across evaluations via ``_RUNNER_CACHE``.
        Before each reuse it is restored to its freshly-loaded baseline with
        ``resetToOrigin()``, which resets global parameters and species initial
        values to the loaded XML state -- making a reused runner bit-identical
        to a fresh load (verified on raf.xml) while skipping the ~3.8 ms
        rebuild. RoadRunner builds that predate ``resetToOrigin()`` are never
        cached, so they fall back to rebuilding a fresh runner each call
        (prior behavior).
        """
        key = self.abs_file_path
        runner = _RUNNER_CACHE.get(key)
        if runner is not None:
            runner.resetToOrigin()
            return runner
        runner = self._build_runner()
        if hasattr(runner, 'resetToOrigin'):
            _RUNNER_CACHE[key] = runner
        return runner

    def model_text(self, mut=None):
        """
        Generates the XML text of the model, optionally applying the MutationSet mut
        Should only be used when saving the model to disk, which is not often done.
        :return:
        """
        logger.info(f'Generating model text for {self.name}')
        runner = self._load_runner()
        self._modify_params(runner)
        if mut:
            self._apply_mutant(mut, runner)
        runner.reset()
        return runner.getCurrentSBML()

    def save(self, file_prefix):
        with open(f'{file_prefix}.xml', 'w') as out:
            out.write(self.model_text())

    def save_all(self, file_prefix):
        for mut in self.mutants:
            with open(f'{file_prefix}{mut.suffix}.xml', 'w') as out:
                out.write(self.model_text(mut=mut))

    def add_action(self, action):
        if action.method not in ('ode', 'ssa'):
            raise PybnfError(f'time_course or param_scan method {action.method} is not possible with an SBML model. Options are '
                             'ode or ssa.')
        self.actions.append(action)
        self.suffixes.append((action.bng_codeword, action.suffix))
        if action.method == 'ssa':
            self.stochastic = True

    def get_suffixes(self):
        """
        Return a list of valid data suffixes to use in this model, including all combinations of action suffix +
        mutation name
        """
        result = []
        for s in self.suffixes:
            for mut in self.mutants:
                result.append(s[1]+mut.suffix)
        return result

    def _modify_params(self, runner):
        """Modify the parameters in this runner instance according to my current PSet"""
        for p in self.param_set.keys():
            if p in self.species_names:
                # Initial condition
                runner.model[f'init([{p}])'] = self.param_set[p]
            elif p in self.param_names:
                setattr(runner, p, self.param_set[p])
            # else The parameter does not appear in this model (might appear in another model, so not an error)

    def _apply_mutant(self, mut, runner):
        """Modify the parameters in this runner instance according to the MutationSet mut"""
        for mi in mut:
            if mi.name in self.species_names:
                runner.model[f'init([{mi.name}])'] = mi.mutate(runner.model[f'init([{mi.name}])'])
            elif mi.name in self.param_names:
                setattr(runner, mi.name, mi.mutate(getattr(runner, mi.name)))

    def _undo_mutant(self, mut, runner):
        """ Undo the application of the MutationSet mut. Should only be called after previously calling
        _apply_mutant()"""
        for mi in mut:
            if mi.name in self.species_names:
                runner.model[f'init([{mi.name}])'] = mi.undo()
            elif mi.name in self.param_names:
                setattr(runner, mi.name, mi.undo())

    def execute(self, folder, filename, timeout):
        runner = self._load_runner()

        # Do parameter modifications
        self._modify_params(runner)

        # Run the model actions
        result_dict = dict()
        selection = ['time'] + [f'[{s}]' for s in self.species_names]
        for mut in self.mutants:
            # Apply all mutations
            self._apply_mutant(mut, runner)

            for act in self.actions:
                runner.reset()
                if act.method == 'ssa' or self.integrator == 'gillespie':
                    runner.setIntegrator('gillespie')
                    runner.getIntegrator().setValue('variable_step_size', False)
                else:
                    runner.setIntegrator(self.integrator)
                    if self.integrator == 'euler':
                        runner.integrator.subdivision_steps = act.subdivisions
                if isinstance(act, TimeCourse):
                    try:
                        if act.explicit_points is not None:
                            # New-era explicit output points (ADR-0028): output at exactly
                            # the data's time points. RoadRunner's times= overrides
                            # start/end/points and integrates from the first listed time --
                            # explicit_points always includes t=0 (TimeCourse) so this
                            # starts at the model baseline.
                            res_array = runner.simulate(times=act.explicit_points, selections=selection)
                        else:
                            res_array = runner.simulate(0., act.time, steps=act.stepnumber, selections=selection)
                    except RuntimeError:
                        # Rethrow simulation errors as something more specific to be caught
                        raise FailedSimulationError
                    res = Data(named_arr=res_array)
                    result_dict[act.suffix + mut.suffix] = res
                    if self.save_files:
                        np.savetxt(f'{folder}/{filename}_{act.suffix}{mut.suffix}.gdat', res_array,
                                   header=' '.join(res_array.colnames))
                elif isinstance(act, ParamScan):
                    # Manually run parameter scan with several simulate commands
                    if act.param not in self.param_names:
                        raise PybnfError(f'Parameter_scan parameter {act.param} was not found in model {self.name}')
                    if act.param in self.species_names:
                        icscan = True
                        init_val = runner.model[f'init([{act.param}])']
                    else:
                        icscan = False
                        init_val = getattr(runner, act.param)
                    # New-era explicit scan values (ADR-0028): sweep exactly the data's
                    # swept-parameter values instead of a uniform np.linspace grid.
                    if act.explicit_points is not None:
                        points = act.explicit_points
                    else:
                        points = np.linspace(act.min, act.max, act.stepnumber + 1)
                    res_array = None
                    labels = None
                    for i, x in enumerate(points):
                        if icscan:
                            runner.model[f'init([{act.param}])'] = x
                        else:
                            setattr(runner, act.param, x)
                        runner.reset()  # Reset concentrations to current ICs
                        try:
                            i_array = runner.simulate(0., act.time, steps=1, selections=selection)
                        except RuntimeError:
                            raise FailedSimulationError
                        if res_array is None:  # First iteration
                            res_array = np.zeros((len(points), 1+i_array.shape[1]))
                            if icscan:
                                # is an initial condition
                                labels = [act.param + '_0'] + i_array.colnames
                            else:
                                labels = [act.param] + i_array.colnames
                        res_array[i, 0] = x
                        res_array[i, 1:] = i_array[1, :]
                    # Restore the original value of the scanned param, for any future actions / models
                    if icscan:
                        runner.model[f'init([{act.param}])'] = init_val
                    else:
                        setattr(runner, act.param, init_val)
                    res = Data(arr=res_array)
                    res.load_rr_header(labels)
                    result_dict[act.suffix + mut.suffix] = res
                    if self.save_files:
                        np.savetxt(f'{folder}/{filename}_{act.suffix}{mut.suffix}.scan', res_array,
                                   header=' '.join([act.param] + i_array.colnames))
                else:
                    raise NotImplementedError('Unknown action type')
            # Undo all mutations
            self._undo_mutant(mut, runner)
        return result_dict


class SbmlModel(SbmlModelNoTimeout):

    def execute(self, folder, filename, timeout):
        self.curr_folder = folder
        self.curr_file = filename
        arg = pickle.dumps(self)
        with open(f'{folder}/{filename}.log', 'w') as errout:
            stdout_data = run_subprocess([executable, '-m', 'pybnf.sbml_runner'],
                                         timeout=timeout, stdout=PIPE, stderr=errout, input=arg)
        result = pickle.loads(stdout_data)
        return result

    def super_execute(self):
        return super().execute(self.curr_folder, self.curr_file, None)


class FailedSimulationError(Exception):
    """
    Raised when a simulation fails that was not a result of a subprocess.run() call (currently only use with
    SbmlModelNoTimeout)
    """
    pass


class Action:
    """
    Represents a simulation action performed within a model
    """
    pass


class TimeCourse(Action):
    """A time-course simulation action parsed from the PyBNF configuration file.

    This supports a subset of BioNetGen's ``simulate`` arguments.  For BNGL
    models, users should prefer writing actions directly in the BNGL file's
    ``begin actions`` block, which supports the full set of BioNetGen arguments
    (e.g., ``steady_state``, ``atol``, ``rtol``, ``continue``, ``stop_if``).
    Config-file actions are primarily intended for SBML models, which have no
    native action syntax.
    """

    def __init__(self, d, explicit_points=None):
        """
        :param d: A dict with string:string key-value pairs made up of user-entered data, specifying the attributes
        of this action.
        Valid dict keys are time:number, step:number, model:str (unused here), suffix: str,
        values: list of numbers (not implemented)
        Raises a PyBNF error if anything is wrong with the dict.
        :param explicit_points: Optional iterable of independent-variable (time) values
        at which the simulation should output, derived from an experiment's data (ADR-0028,
        Chunk 3). When given, the action outputs at exactly these time points instead of a
        uniform ``n_steps`` grid (BNGL ``sample_times`` / RoadRunner ``simulate(times=...)``).
        ``t=0`` is always included so integration starts at the model's baseline.
        """
        # Available keys and default values
        num_keys = {'time', 'step'}
        str_keys = {'model', 'suffix', 'method'}
        int_keys = {'subdivisions'}
        # Default values
        self.time = None  # Required (or derived from explicit_points below)
        self.step = 1.
        self.subdivisions = 1
        self.model = ''
        self.suffix = 'time_course'
        self.method = 'ode'


        # Transfer all the keys in the dict to my attributes of the same name
        for k in d:
            if k in num_keys:
                try:
                    num = float(d[k])
                except ValueError:
                    raise PybnfError(f'For key "time_course", the value of "{k}" must be a number.')
                self.__setattr__(k, num)
            elif k in int_keys:
                try:
                    num = int(d[k])
                except ValueError:
                    raise PybnfError(f'For key "time_course", the value of "{k}" must be an integer.')
                self.__setattr__(k, num)
            elif k in str_keys:
                self.__setattr__(k, d[k])
            else:
                raise PybnfError(f'"{k}" is not a valid attribute for "time_course".',
                                 '"{}" is not a valid attribute for "time_course". Possible attributes are: {}'.format(k, ','.join(num_keys.union(str_keys))))

        # New-era explicit output points (ADR-0028): when an experiment derives its
        # output grid from the data's independent-variable column, the time course
        # outputs at exactly those points rather than a uniform n_steps grid. ``t=0``
        # is forced in so integration always starts at the model's baseline -- BNG2.pl's
        # t_start defaults to 0, but RoadRunner's simulate(times=...) and bngsim's
        # sample_times path both begin integrating at the FIRST listed time (verified:
        # RoadRunner with times=[10,...] returns the initial condition at t=10, not the
        # evolved value), so without a leading 0 a data grid that starts at t>0 would
        # silently produce the wrong trajectory. The extra t=0 output row is harmless to
        # scoring, which matches by independent-variable value and ignores sim rows with
        # no corresponding data point (objective.py). Points are sorted and deduplicated:
        # the simulation outputs one row per distinct time; replicate data with repeated
        # times stacks extra measurement rows that all match that one sim row.
        self.explicit_points = None
        if explicit_points is not None:
            pts = sorted({float(p) for p in explicit_points} | {0.0})
            if len(pts) < 2:
                raise PybnfError('A time_course with explicit output points needs at least one '
                                 'positive time point.')
            self.explicit_points = pts
            if self.time is None:
                self.time = pts[-1]

        if self.time is None:
            raise PybnfError('For key "time_course" a value for "end" must be specified.')

        if self.method not in ('ode', 'ssa', 'pla', 'nf'):
            raise PybnfError(f'Invalid time course method {self.method}. Options are ode, ssa, pla, nf')

        if self.step == 0:
            raise PybnfError('For key "time_course", the value of "step" must be nonzero.')
        self.stepnumber = int(np.round(self.time/self.step))
        self.bng_codeword = 'simulate'

class ParamScan(Action):
    """A parameter-scan action parsed from the PyBNF configuration file.

    This supports a subset of BioNetGen's ``parameter_scan`` arguments.  For
    BNGL models, users should prefer writing actions directly in the BNGL
    file's ``begin actions`` block, which supports the full set of BioNetGen
    arguments (e.g., ``steady_state``, ``atol``, ``rtol``).  Config-file
    actions are primarily intended for SBML models, which have no native action
    syntax.
    """

    def __init__(self, d, explicit_points=None):
        """
        :param d: A dict with string:string key-value pairs made up of user-entered data, specifying the attributes
        of this action.
        Valid dict keys are min:number, max:number, step:number, time:number, model:str (unused here), suffix: str,
        logspace: 0 or 1, param: str, values: list of numbers (not implemented)
        Raises a PyBNF error if anything is wrong with the dict.
        :param explicit_points: Optional iterable of scanned-parameter values at which the
        scan should output, derived from an experiment's data (ADR-0028, Chunk 3). When
        given, the scan outputs at exactly these values (BNGL ``par_scan_vals`` / a
        RoadRunner simulate loop over the values) instead of a uniform ``par_min..par_max``
        grid, and ``min``/``max``/``step`` become optional (derived from the values).
        """
        # Available keys and default values
        num_keys = {'min', 'max', 'step', 'time'}
        str_keys = {'model', 'suffix', 'param', 'method'}
        int_keys = {'subdivisions', 'logspace'}
        required_keys = {'min', 'max', 'step', 'time', 'param'}
        # Default values
        self.min = None
        self.max = None
        self.step = None
        self.time = None
        self.logspace = 0
        self.param = None
        self.model = ''
        self.suffix = 'param_scan'
        self.method = 'ode'
        self.subdivisions = 1000

        # Transfer all the keys in the dict to my attributes of the same name
        for k in d:
            if k in num_keys:
                try:
                    num = float(d[k])
                except ValueError:
                    raise PybnfError(f'For key "param_scan", the value of "{k}" must be a number.')
                self.__setattr__(k, num)
            elif k in int_keys:
                try:
                    num = int(d[k])
                except ValueError:
                    raise PybnfError(f'For key "time_course", the value of "{k}" must be an integer.')
                self.__setattr__(k, num)
            elif k in str_keys:
                self.__setattr__(k, d[k])
            else:
                raise PybnfError(f'"{k}" is not a valid attribute for "param_scan".',
                                 '"{}" is not a valid attribute for "param_scan". Possible attributes are: {}'.format(k, ','.join(num_keys.union(str_keys))))

        # New-era explicit scan values (ADR-0028): a parameter scan derives its swept
        # values from the data's independent-variable column (the scanned parameter),
        # outputting at exactly those values rather than a uniform par_min..par_max grid.
        # Unlike a time course, no 0 is forced in -- each scan point is an independent
        # simulation from the baseline, so there is no integration-start point to add and a
        # spurious 0 dose is not wanted. min/max/step are derived so the legacy required-key
        # check below is satisfied; explicit_points takes precedence at emit/run time.
        self.explicit_points = None
        if explicit_points is not None:
            pts = sorted({float(p) for p in explicit_points})
            if len(pts) == 0:
                raise PybnfError('A param_scan with explicit values needs at least one value.')
            self.explicit_points = pts
            if self.min is None:
                self.min = pts[0]
            if self.max is None:
                self.max = pts[-1]
            if self.step is None:
                self.step = (pts[-1] - pts[0]) if pts[-1] != pts[0] else 1.0

        for k in required_keys:
            if self.__getattribute__(k) is None:
                raise PybnfError(f'For key "param_scan" a value for "{k}" must be specified.')
        self.logspace = int(self.logspace)
        if self.logspace not in (0, 1):
            raise PybnfError('For key "param_scan", the value for "logspace" must be 0 or 1')
        if self.method not in ('ode', 'ssa', 'pla', 'nf'):
            raise PybnfError(f'Invalid time course method {self.method}. Options are ode, ssa, pla, nf')

        if self.step == 0:
            raise PybnfError('For key "param_scan", the value of "step" must be nonzero.')
        self.stepnumber = int(np.round((self.max - self.min) / self.step))
        self.bng_codeword = 'parameter_scan'


class Mutation:

    def __init__(self, name, operation, value):
        """
        Create a mutation
        :param name: Name of the variable to mutate
        :type name: str
        :param operation: Operation to perform on the target variable; one of + - * / =
        :type operation: str
        :param value: The value to add/subtract/etc (depending on the operation)
        :type value: float
        """
        self.name = name
        self.operation = operation
        self.value = value
        if operation not in ('+', '-', '*', '/', '='):
            raise RuntimeError(f'Invalid mutation operation {operation}')
        self.old = None
        logger.debug(f'Created mutation {self.name} {self.operation} {self.value}')

    def mutate(self, num):
        """
        Applies this mutation
        :param num:
        :return: float
        """
        self.old = num
        if self.operation == '=':
            return self.value
        elif self.operation == '+':
            return num + self.value
        elif self.operation == '-':
            return num - self.value
        elif self.operation == '*':
            return num * self.value
        elif self.operation == '/':
            return num / self.value

    def undo(self):
        """
        Undo the mutation we just did
        :return: float
        """
        if self.old is None:
            raise RuntimeError('Called undo() on a Mutation that was not performed')
        old = self.old
        self.old = None
        return old


class MutationSet:
    """
    A set of mutations that represents a mutant model
    """
    def __init__(self, mutations=(), suffix=''):
        """

        :param mutations: The mutations to include in this MutationSet
        :type mutations: iterable of Mutant
        :param suffix: The simulation suffix for this mutant. This will be appended to the action suffix
        :type suffix: str
        """
        self.mutations = mutations
        self.suffix = suffix
        logger.debug('Created MutationSet with %i mutations' % len(self.mutations))

    def __iter__(self):
        return iter(self.mutations)


class ModelError(Exception):
    # These are sometimes but not always user-generated, so need to be able to pass the info back to the
    # user exception handler.
    def __init__(self, message):
        self.message = message


class FreeParameter:
    """
    Class representing a free parameter in a model
    """

    def __init__(self, name, type, p1, p2, value=None, bounded=True, lb=None, ub=None,
                 initialization_distribution=INITIALIZATION_PRIOR,
                 initialization_lb=None, initialization_ub=None):
        """
        Initializes a FreeParameter object based on information parsed from the configuration file

        :param name: The name of the parameter as it appears in the model
        :type name: str
        :param type: The type of the parameter as defined in the configuration file
        :type type: str
        :param p1: The first value governing the variable (lower bound or mean or initial value)
        :type p1: float
        :param p2: The second value governing the parameter (upper bound or standard deviation or step size)
        :type p2: float
        :param value: The parameter's numerical value
        :type value: float
        :param bounded: Determines whether the parameter should be bounded after initial sampling
         (only relevant if parameter's initial distribution is bounded)
        :param lb: Optional lower truncation bound (in theta) for an unbounded-support
         prior (normal/laplace/log-*). With ``ub``, the prior is truncated to a
         finite reflecting box (ADR-0020, issue #411). Distinct from ``p1``, which
         is the family's location/scale, not a bound. Ignored for the Uniform
         families, whose bounds are ``p1``/``p2``.
        :param ub: Optional upper truncation bound (in theta); see ``lb``.
        :param initialization_distribution: Which distribution should generate
         algorithm start points: ``prior`` (backward-compatible) or ``bounds``.
        :param initialization_lb: Optional lower initialization bound in theta,
         used when ``initialization_distribution`` is ``bounds``.
        :param initialization_ub: Optional upper initialization bound in theta.
        """
        self.name = name
        self.type = type
        self.p1 = p1
        self.p2 = p2
        # The truncation box (in theta) requested for an unbounded-support family,
        # preserved verbatim so set_value can reconstruct the parameter (ADR-0020).
        self.trunc_lb = lb
        self.trunc_ub = ub
        self.initialization_distribution = initialization_distribution or INITIALIZATION_PRIOR
        self.initialization_lb = initialization_lb
        self.initialization_ub = initialization_ub
        if self.initialization_distribution not in INITIALIZATION_DISTRIBUTIONS:
            valid = ', '.join(sorted(INITIALIZATION_DISTRIBUTIONS))
            raise PybnfError(
                f"Parameter {self.name}: unknown initialization_distribution "
                f"{self.initialization_distribution!r}; expected one of {valid}.")
        if (self.initialization_lb is None) != (self.initialization_ub is None):
            raise PybnfError(
                f"Parameter {self.name}: initialization bounds need both lb and ub "
                f"(got lb={self.initialization_lb}, ub={self.initialization_ub}).")

        # The prior (distribution family in sampling space u) and the scale
        # (theta<->u transform) are resolved from the legacy *_var keyword via the
        # registry-derived map -- a behavior-preserving split (ADR-0010, M2.3).
        self._prior, self._scale = build_prior(type, p1, p2)

        if self._prior.has_bounded_support:
            # Uniform: the bounds are p1/p2 in theta; the reflecting box is gated by
            # the b/u flag (replaces re.search('uniform', type)). Truncation bounds
            # do not apply -- the support is the family's own box.
            self.bounded = bounded
            self.lower_bound = -np.inf if not self.bounded else self.p1
            self.upper_bound = np.inf if not self.bounded else self.p2
        elif lb is None and ub is None:
            # Unbounded, untruncated -- today's behavior (the default bounded flag
            # is meaningless for an unbounded-support family without a box).
            self.bounded = False
            self.lower_bound = -np.inf
            self.upper_bound = np.inf
        else:
            # Truncated unbounded-support prior (ADR-0020): a finite, two-sided
            # reflecting box that also truncates the prior density and sampling.
            # The family-agnostic reflection fold (_reflect) needs two finite
            # bounds, so one-sided truncation has no box to fold into -- raise
            # rather than silently drop the bound.
            if lb is None or ub is None:
                raise PybnfError(
                    f"Parameter {self.name}: truncating an unbounded-support prior "
                    f"needs both lb and ub (got lb={lb}, ub={ub}); one-sided "
                    f"truncation has no finite reflecting box.")
            self.lower_bound = float(lb)
            self.upper_bound = float(ub)
            if not (np.isfinite(self.lower_bound) and np.isfinite(self.upper_bound)):
                raise PybnfError(
                    f"Parameter {self.name}: truncation bounds must both be finite "
                    f"to form a reflecting box, got [{self.lower_bound}, {self.upper_bound}]")
            self.bounded = True
            # Wrap the family in a TruncatedPrior carrying the box in sampling
            # space u (the scale owns theta<->u; log parameters truncate in log10).
            self._prior = TruncatedPrior(
                self._prior, self._scale.forward(self.lower_bound),
                self._scale.forward(self.upper_bound))

        if self.lower_bound > self.upper_bound:
            raise PybnfError(f"Parameter {self.name} has a lower bound that is greater than its upper bound")
        if self.initialization_distribution == INITIALIZATION_BOUNDS:
            self._initialization_bounds_u()

        # Determine a positive value that can serve as the default for network generation
        self.default_value = None
        if self.lower_bound > 0.0:
            self.default_value = self.lower_bound
        elif np.isfinite(self.upper_bound):
            self.default_value = self.upper_bound
        else:
            self.default_value = 1.0

        if value:
            if not self.lower_bound <= value <= self.upper_bound:  # not quite precise, but works well
                raise OutOfBoundsException(f"Free parameter {self.name} cannot be assigned the value {value}")
        self.value = value

        self.log_space = self._scale.is_log
        self.scale_name = self._scale.name

    @property
    def _distribution(self):
        """The prior's underlying frozen scipy.stats distribution, or None for a
        no-prior var/logvar. A back-compat passthrough: the family/scale split now
        owns the math (ADR-0010), but load_priors and the tests still read this."""
        return self._prior.frozen

    def to_sampling_space(self, theta):
        """Map a stored value ``theta`` into the sampling space ``u`` the prior and
        the proposal arithmetic operate in -- ``log10(theta)`` for a log parameter,
        identity otherwise.

        The **public** peer of the private ``_scale.forward`` (ADR-0010), added so
        the algorithm layer asks the parameter for the transform instead of
        inlining ``np.log10(x) if v.log_space else x`` at a dozen sites (#412).
        Accepts a scalar or a numpy array (the histogram path passes a column)."""
        return self._scale.forward(theta)

    def from_sampling_space(self, u):
        """Map a sampling-space value ``u`` back to a stored value -- ``10**u`` for a
        log parameter, identity otherwise. The inverse of :meth:`to_sampling_space`
        and the public peer of ``_scale.inverse``.

        Unguarded, bit-for-bit ``10.0 ** u`` (matching the inline ``10**`` the
        proposal arithmetic already used); an out-of-range result is handled by the
        box clamp / reflection at the call site. The *guarded* inverse for
        user-supplied ``logvar`` / ``lognormal`` start values is the separate
        ``exp10`` helper, which raises a configuration hint on overflow (#412)."""
        return self._scale.inverse(u)

    def set_value(self, new_value, reflect=True):
        """
        Creates a copy of the parameter with the given value

        :param new_value: A numeric value assigned to the FreeParameter
        :type new_value: float
        :param reflect: Determines whether to reflect the parameter value if it is outside of the defined bounds
        :type reflect: bool
        :return: FreeParameter
        """
        if new_value < self.lower_bound or new_value > self.upper_bound:
            if not reflect:
                raise OutOfBoundsException(f"Parameter {self} is outside of bounds")
            if self.value is None:
                self.value = self.lower_bound
                logger.info(f"Assigning parameter {self.name} to take a value equal to its lower bound: {self.lower_bound}")
            # reflective number line, can never realize self.lower_bound or self.upper_bound this way
            adj = self._reflect(new_value)
            logger.debug(f'Assigned value {new_value:f} is out of defined bounds: [{self.lower_bound}, {self.upper_bound}].  '
                           f'Adjusted to {adj:f}')
            new_value = adj
        return FreeParameter(self.name, self.type, self.p1, self.p2, new_value,
                             bounded=self.bounded, lb=self.trunc_lb, ub=self.trunc_ub,
                             initialization_distribution=self.initialization_distribution,
                             initialization_lb=self.initialization_lb,
                             initialization_ub=self.initialization_ub)

    def _reflect(self, new):
        """Reflect a proposed value back inside the parameter's bounds.

        The reflection is the triangle-wave fold of ``new`` into ``[lb, ub]`` --
        a deterministic, measure-preserving involution. Because it is symmetric
        and depends only on the proposed value (not the current one), a
        random-walk proposal followed by this fold stays symmetric, so the plain
        Metropolis acceptance used by the MCMC samplers still targets the correct
        bound-restricted posterior. Computing the fold in closed form (rather than
        iterating one reflection at a time) means it is exact for an arbitrarily
        large step, with no iteration cap and no balance-breaking random fallback.
        """
        ub = self.upper_bound
        lb = self.lower_bound
        if lb == ub:
            return lb
        # Fold in sampling space u; the scale owns the theta<->u transform, so log
        # parameters reflect in log10 space (Linear's transform is the identity).
        ub = self._scale.forward(ub)
        lb = self._scale.forward(lb)
        new = self._scale.forward(new)
        if self._scale.is_log:
            logger.debug(f"Reflecting in log space: new={new} lb={lb} ub={ub}")

        width = ub - lb
        q = (new - lb) % (2.0 * width)
        folded = lb + q if q <= width else ub - (q - width)

        return self._scale.inverse(folded)

    def sample_value(self, rng):
        """
        Samples a value from this parameter's objective prior distribution.

        :param rng: the caller's np.random.Generator, passed through to the prior
        :return: new FreeParameter instance or None
        """
        if not self._prior.has_prior:
            raise PybnfError(f"Parameter {self.name} does not have a sampling distribution")

        return self.set_value(self._scale.inverse(self._prior.rvs(rng)))

    def _initialization_bounds_u(self):
        """Finite initialization box in sampling space ``u``.

        ``bounds`` initialization spans the parameter's reflecting box when one
        exists (including PEtab-truncated normal/laplace priors) and otherwise
        falls back to a finite prior support (plain uniform/loguniform). The
        algorithms also operate in ``u``, so log parameters initialize uniformly
        over log-bounds.
        """
        if self.initialization_lb is not None:
            lo_u = self._scale.forward(float(self.initialization_lb))
            hi_u = self._scale.forward(float(self.initialization_ub))
        elif np.isfinite(self.lower_bound) and np.isfinite(self.upper_bound):
            lo_u = self._scale.forward(self.lower_bound)
            hi_u = self._scale.forward(self.upper_bound)
        elif self._prior.has_bounded_support:
            lo_u, hi_u = self._prior.support()
        else:
            raise PybnfError(
                f"Parameter {self.name}: initialization_distribution='bounds' "
                f"requires finite parameter bounds or a finite-support prior.")

        if not (np.isfinite(lo_u) and np.isfinite(hi_u)) or lo_u >= hi_u:
            raise PybnfError(
                f"Parameter {self.name}: initialization bounds must be finite and "
                f"increasing in sampling space, got [{lo_u}, {hi_u}].")
        return lo_u, hi_u

    @property
    def has_bounded_initialization(self):
        """Whether this parameter can participate in Latin-hypercube seeding."""
        if self.initialization_distribution == INITIALIZATION_BOUNDS:
            return True
        return self._prior.has_bounded_support

    def sample_initial_value(self, rng):
        """Draw a start-point value from the configured initialization distribution."""
        if self.initialization_distribution == INITIALIZATION_PRIOR:
            return self.sample_value(rng)
        return self.initial_value_from_quantile(rng.random())

    def prior_logpdf(self, value):
        """
        Evaluate the log prior density for a regular-space parameter value.

        For log-space variables, the prior is evaluated in base-10 logarithmic
        space to match the historical parameterization of lognormal_var and
        loguniform_var.
        """
        if not self._prior.has_prior:
            return 0.
        if self.log_space and value <= 0.:
            return -np.inf
        return float(self._prior.logpdf(self._scale.forward(value)))

    @property
    def has_prior(self):
        """Whether this parameter has a proper prior distribution (False for the
        no-prior var/logvar Simplex start points). Used by samplers to decide
        which parameters contribute to the log prior."""
        return self._prior.has_prior

    @property
    def has_bounded_support(self):
        """Whether the prior family has finite support (the Uniform families).
        Drives latin-hypercube participation and the box-escape warning -- the
        property the algorithms ask instead of matching the *_var type string."""
        return self._prior.has_bounded_support

    def value_from_quantile(self, q):
        """Map a [0, 1] quantile to a value via the prior's inverse CDF, in scale.

        For the bounded (Uniform) families this is the latin-hypercube rescale:
        scale.inverse(lo + q*(hi - lo)) -- equal bit-for-bit to the historical
        p1 + q*(p2 - p1) (linear) / exp10(log10(p1) + q*...) (log10)."""
        return self.set_value(self._scale.inverse(self._prior.ppf(q)))

    def initial_value_from_quantile(self, q):
        """Map a [0, 1] quantile through the initialization distribution."""
        if self.initialization_distribution == INITIALIZATION_PRIOR:
            return self.value_from_quantile(q)
        lo_u, hi_u = self._initialization_bounds_u()
        return self.set_value(self._scale.inverse(lo_u + q * (hi_u - lo_u)))

    def add(self, summand, reflect=True):
        """
        Adds a value to the existing value and returns a new FreeParameter instance.  Since free parameters
        can exist in regular or logarithmic space, the value to add is expected to already be transformed
        to the appropriate space

        :param summand: Value to add
        :return:
        """
        if self.value is None:
            logger.error('Cannot add to FreeParameter with "None" value')
        # Add the summand in sampling space u, then map back (Linear is identity;
        # Log10 gives 10**(log10(value) + summand)).
        return self.set_value(self._scale.inverse(self._scale.forward(self.value) + summand), reflect)

    def multiply(self, summand, reflect=True):
        """
        Adds a value to the existing value and returns a new FreeParameter instance. This version of add does
        not consider the space that the value is in and just sums them

        :param summand: Value to a multiply
        :return:
        """
        if self.value is None:
            logger.error('Cannot multiply to FreeParameter with "None" value')
        
        return self.set_value(self.value * summand, reflect)

    def add_rand(self, lb, ub, rng, reflect=True):
        """
        Like FreeParameter.add but instead adds a uniformly distributed random value according to the
        bounds provided

        :param lb:
        :param ub:
        :param rng: the caller's np.random.Generator
        :return:
        """
        try:
            # lb + (ub-lb)*U, matching legacy np.random.uniform(lb, ub). Computed
            # explicitly (rather than rng.uniform) because callers may pass lb > ub
            # (e.g. Scatter Search), which Generator.uniform rejects but the legacy
            # RandomState.uniform silently accepted.
            r = lb + (ub - lb) * rng.random()
        except OverflowError:
            logger.error(f'Random number overflow with lower bound {lb}, upper bound {ub}')
            r = 0.
        return self.add(r, reflect)

    def diff(self, other):
        """
        Calculates the difference between two FreeParameter instances.  Both instances must occupy the same space
        (log or regular) and if they are both in log space, the difference will be calculated based on their
        logarithms.
        :param other: A FreeParameter from which the difference will be calculated
        :return:
        """
        if not isinstance(other, FreeParameter):
            raise ValueError("Cannot compare FreeParameter with another object")
        if not self.log_space == other.log_space:
            raise ValueError("Cannot calculate diff between two FreeParameter instances that are not varying in the same"
                             "space")
        if self.log_space:
            # Difference in the parameter's sampling space; ``_scale.forward`` is the base
            # (log10 or ln), so this is log10(a/b) for a log10 parameter -- bit-identical to
            # the old hardcode -- and ln(a/b) for a natural-log one (ADR-0043).
            return self._scale.forward(self.value / other.value)
        else:
            return self.value - other.value

    def __hash__(self):
        return hash((self.name, self.value))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.name, self.type, self.value, self.p1, self.p2) == \
                   (other.name, other.type, other.value, other.p1, other.p2)
        return False

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return f"FreeParameter: {self.name} = {self.value} -- [{self.lower_bound}, {self.upper_bound}]"

    def __repr__(self):
        return self.__str__()


class PSet:
    """
    Class representing a parameter set

    """

    def __init__(self, fps):
        """
        Creates a Pset based on the given dictionary

        :param fps: A list of FreeParameter instances whose values are not None
        """

        self._param_dict = {}
        self.fps = fps

        for fp in fps:
            if fp.value is None:
                raise PybnfError(f"Parameter {fp.name} has no value")
            elif fp.name in self._param_dict.keys():
                raise PybnfError("Parameters must have unique names")
            self._param_dict[fp.name] = fp

        self.name = None  # Can be set by Algorithms to give it a meaningful label in output file.

    def __iter__(self):
        # Return a fresh iterator over the parameter list rather than making the
        # PSet its own iterator with a cursor stored on self -- the latter is not
        # reentrant (nested or concurrent iteration over one PSet clobbers idx).
        return iter(self.fps)

    def __getitem__(self, item):
        """
        Returns the value of the specified parameter.

        This allows the standard dictionary syntax ps['paramname']
         to be used for accessing (but not changing) parameters.

        :param item: The str name of the parameter to look up
        :return: float
        """
        return self._param_dict[item].value

    def get_param(self, name):
        """
        Gets the full FreeParameter based on its name

        :param name:
        :return:
        """
        return self._param_dict[name]

    def __len__(self):
        return len(self._param_dict)

    def get_id(self):
        return self.__hash__()

    def __hash__(self):
        """
        Returns a unique identifier for this parameter set
        Two PSets will have the same identifier if they have the same keys and corresponding values

        :return: int
        """
        return hash(frozenset(self._param_dict.values()))

    def __str__(self):
        """
        When a PSet is converted to a str, returns "PSet:" followed by the parameter dict.
        :return: str
        """
        return "PSet:" + str(self._param_dict)

    def __repr__(self):
        """

        :return: str
        """
        return self.__str__()

    def __eq__(self, other):
        """
        Checks equality to another PSet by comparing the _param_dicts

        :param other:
        :return:
        """

        return self._param_dict == other._param_dict

    def keys(self):
        """
        Returns a list of the parameter keys
        :return: list
        """
        return self._param_dict.keys()

    def keys_to_string(self):
        """
        Returns the keys (parameter names) in a tab-separated str in alphabetical order

        :return: str
        """
        keys = [str(k) for k in self._param_dict.keys()]
        keys.sort()
        return '\t'.join(keys)

    def values_to_string(self):
        """
        Returns the parameter values in a tab-separated str, in alphabetical order
        according to the parameter name
        :return: str
        """
        keys = [str(k) for k in self._param_dict.keys()]
        keys.sort()
        values = [str(self[k]) for k in keys]  # Values are in alpha order by key name
        return '\t'.join(values)


class Trajectory:
    """
    Tracks the various PSet instances and the corresponding objective function values
    """

    def __init__(self, max_output):
        # self._trajectory is a heap-based priority queue
        # Contains tuples (-score, name, PSet) - allows us to efficiently toss the worst PSet when we get a new one
        # Note we use -score so popping the worst entry is fast
        # As long as you follow the rule of no duplicate names, this is safe and won't compare PSets.
        self._trajectory = []
        self.max_output = max_output

    def _valid_pset(self, pset):
        """
        Checks to confirm that a PSet is compatible with this Trajectory

        :param pset: A PSet instance
        :return: bool
        """
        existing_pset = self._trajectory[0][2]
        return pset.keys() == existing_pset.keys()

    def add(self, pset, obj, name, append_file=None, first=False):
        """
        Adds a PSet to the fitting trajectory

        :param pset: A particular point in parameter space
        :param obj: The objective function value upon executing the model at this point in parameter space
        :raises: Exception
        """
        if len(self._trajectory) > 0:
            if not self._valid_pset(pset):
                raise ValueError(f"PSet {pset} has incompatible parameters")
        if np.isnan(obj):
            # Treat nan values as Inf in order to sort correctly
            obj = np.inf

        if len(self._trajectory) < self.max_output:
            heapq.heappush(self._trajectory, (-obj, name, pset))
        else:
            # Add the current pset, and throw away the worst one
            heapq.heappushpop(self._trajectory, (-obj, name, pset))

        if append_file:
            with open(append_file, 'a') as af:
                if first:
                    af.write(self._traj_write_header())
                af.write(self._traj_entry_format((-obj, name, pset)))

    def _traj_write_header(self):
        header = self._trajectory[0][2].keys_to_string()
        return f'#\tSimulation\tObj\t{header}\n'

    def _traj_entry_format(self, entry):
        """
        Formats a tuple (-obj, name, pset) as stored in self.trajectory into a string for printing
        """
        return f'\t{entry[1]}\t{-entry[0]}\t{entry[2].values_to_string()}\n'

    def _write(self):
        """Writes the Trajectory in a tab-delimited format"""
        s = self._traj_write_header()
        num_output = 0
        for k in sorted(self._trajectory, reverse=True):
            s += self._traj_entry_format(k)
            num_output += 1
            if num_output == self.max_output:
                break
        return s

    @staticmethod
    def load_trajectory(filename, variables, max_output):
        """Loads a Trajectory from file given Algorithm.variables information"""

        logger.info(f'Loading trajectory from {filename}')
        with open(filename, encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        if len(lines) == 0:
            raise IOError(f'Empty parameters file {filename}')
        var_names = re.split(r'\s+', lines[0].strip('#').strip())[2:]

        t = Trajectory(max_output)
        for l in lines[1:]:
            xs = re.split(r'\s+', l.strip())
            name = xs[0]
            obj = float(xs[1])
            var_dict = {var_names[i]: float(x) for i, x in enumerate(xs[2:])}
            pset = PSet([v.set_value(var_dict[v.name]) for v in variables])

            t.add(pset, obj, name)

        return t

    def write_to_file(self, filename):
        """
        Writes the Trajectory to a specified file

        :param filename: File to store Trajectory
        """
        try:
            with open(filename, 'w') as f:
                f.write(self._write())
                f.close()
        except IOError as e:
            logger.exception('Failed to save parameter sets to file')
            print1('Failed to save parameter sets to file.\nSee log for more information')
            if e.strerror == 'Too many open files':
                print0('Too many open files! See "Troubleshooting" in the documentation for how to deal with this '
                       'problem.')

    def best_fit(self):
        """
        Finds the best fit parameter set

        :return: PSet
        """
        return max(self._trajectory)[2]

    def best_fit_name(self):
        """
        Finds the name of the best fit parameter set (which is also the folder
        where that result is stored)

        :return: str
        """
        return max(self._trajectory)[1]

    def best_score(self):
        """
        Returns the best objective value in this trajectory
        :return: float
        """
        return -max(self._trajectory)[0]


class OutOfBoundsException(Exception):
    pass
