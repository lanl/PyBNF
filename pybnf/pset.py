"""pybnf.pset: classes for storing models, parameter sets, and the fitting trajectory"""


from .printing import print1, PybnfError

import logging
import numpy as np
import re
import copy
import xml.etree.ElementTree as ET
from subprocess import run, STDOUT
from .data import Data

logger = logging.getLogger(__name__)


class Model(object):
    """
    An abstract class representing an executable model
    """

    def copy_with_param_set(self, pset):
        """Returns a copy of the model with a new parameter set

        :param pset: A new parameter set
        :type pset: PSet
        :return: Model
        """
        NotImplementedError("copy_with_param_set is not implemented")

    def save(self, file_prefix, **kwargs):
        """
        Saves the model to file

        :return:
        """
        NotImplementedError("save is not implemented")

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


class BNGLModel(Model):
    """
    Class representing a BNGL model


    """
    bng_command = ''

    def __init__(self, bngl_file, pset=None):
        """
        Loads the model from the given .bngl file

        :param bngl_file: str address of the bngl file
        :param pset: PSet to initialize the model with. Defaults to None
        """
        self.file_path = bngl_file
        self.name = re.sub(".bngl", "", self.file_path[self.file_path.rfind("/")+1:])
        self.suffixes = []  # list of 2-tuples (sim_type, prefix)

        self.generates_network = False
        self.generate_network_line = None
        self.generate_network_line_index = -1
        self.action_line_indices = []
        self.actions = []

        self.stochastic = False  # Update during parsing. Used to warn about misuse of 'smoothing'

        # TODO BNGL file reading is rather convoluted and could probably be cleaned up
        # Read the file
        with open(self.file_path) as file:
            self.bngl_file_text = file.read()

        # Scan the file's lines
        # Find param names of type __FREE__, and also the 'begin parameters' declaration
        param_names_set = set()
        self.split_line_index = None # for insertion of free parameters
        self.model_lines = [x.strip() for x in self.bngl_file_text.splitlines()]
        in_action_block = False
        in_no_block = True

        continuation = ''
        for i, line in enumerate(self.model_lines):
            commenti = line.find('#')
            if commenti != -1:
                line = line[:commenti]

            if re.match(r'^\s*$', line):
                continue  # Blank line - must handle before line continuation

            # Handle case where '\' is used to continue on the next line
            line = continuation + line
            continuation = ''
            continue_match = re.search(r'\\\s*$', line)
            if continue_match:
                # This line continues on the next line
                continuation = line[:continue_match.start()]
                continue

            # Find every item matching [alphanumeric]__FREE__
            params = re.findall('[A-Za-z_]\w*__FREE__', line)
            for p in params:
                param_names_set.add(p)

            # Make sure setOption (if present) doesn't get passed to the actions block
            if re.match('\s*setOption', line):
                continue

            # Check if this is the 'begin parameters' line
            if re.match('begin\s+parameters', line.strip()):
                self.split_line_index = i + 1
            elif re.search('generate_network', line):
                self.generates_network = True
                self.generate_network_line = line
                self.generate_network_line_index = i
            elif re.search('simulate_((ode)|(ssa)|(pla))', line) or re.search('simulate.*method=>(\'|")((ode)|(ssa)|(pla))("|\')', line):
                self.generates_network = True  # in case there is no "generate_network" command present

            action_suffix = self._get_action_suffix(line)
            if action_suffix is not None:
                self.suffixes.append(action_suffix)

            # "begin model" doesn't work like a regular block, so escape before we start handling blocks.
            if re.match('(begin|end)\s+model', line.strip()):
                continue

            if re.match('begin\s+actions', line.strip()):
                in_action_block = True
                in_no_block = False
                self.action_line_indices.append(i)
                continue
            elif re.match('end\s+actions', line.strip()):
                in_action_block = False
                in_no_block = True
                self.action_line_indices.append(i)
                continue

            # To keep track of whether we're in no block, which counts as an action block, check for
            # begin and end keywords
            if re.match('begin\s+[a-z][a-z\s]*', line.strip()) and not re.match('begin\s+model', line.strip()):
                in_no_block = False
                continue

            if re.match('end\s+[a-z][a-z\s]*', line.strip()) and not re.match('end\s+model', line.strip()):
                in_no_block = True
                continue

            if in_action_block or in_no_block:
                if re.match('generate_network', line.strip()):
                    continue
                else:
                    if 'method=>"nf"' in line or 'method=>"ssa"' in line or 'method=>"pla"' in line or \
                            'simulate_nf' in line or 'simulate_ssa' in line or 'simulate_pla' in line:
                        self.stochastic = True
                    if re.search('seed=>\d+', line):
                        # There's probably a better way to handle this.
                        print1("Warning: Your model file specifies the 'seed' argument. This means that if you are "
                               "using the 'smoothing' feature, all of your replicates will come out the same.")
                    self.action_line_indices.append(i)
                    self.actions.append(line)

        if self.generates_network and self.generate_network_line is None:
            self.generate_network_line = 'generate_network({overwrite=>1})'
            first_action_index = min(self.action_line_indices)

            # places generate_network command directly prior to actions
            self.generate_network_line_index = first_action_index
            self.model_lines.insert(self.generate_network_line_index, self.generate_network_line)
            for i, v in enumerate(self.action_line_indices):
                self.action_line_indices[i] = 1 + v

        if len(param_names_set) == 0:
            raise ModelError("No free parameters found in model %s. Your model file needs to include variable names "
                             "that end in '__FREE__' to tell BioNetFit which parameters to fit." % bngl_file)

        if self.split_line_index is None:
            raise ModelError("'begin parameters' not found in BNGL file")

        if not self.action_line_indices:
            raise ModelError("No actions found in model")

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
    def _get_action_suffix(line):
        sim_match = re.match("(simulate|parameter_scan)", line.strip())
        if sim_match:
            act_type = sim_match.group(1)
            match = re.search("suffix\s*=>\s*['\"](.*?)['\"]\s*[,}]", line)
            if match is not None:
                return act_type, match.group(1)
        return None

    def copy_with_param_set(self, pset):
        """
        Returns a copy of this model containing the specified parameter set.

        :param pset: A PSet object containing the parameters for the new instance
        :type pset: PSet
        :return: BNGLModel
        """
        # Check that the PSet has definitions for the right parameters for this model
        if not set(pset.keys()) >= set(self.param_names):
            raise PybnfError('Parameter names in the PSet do not match those in the Model\n%s\n%s' %
                             (pset.keys(), self.param_names))

        if set(pset.keys()) != set(self.param_names):
            logger.warning('Model %s does not contain all defined free parameters' % self.name)

        newmodel = copy.deepcopy(self)
        newmodel.param_set = pset
        return newmodel

    def model_text(self, gen_only=False):
        """
        Returns the text of a runnable BNGL file, which includes the contents of the original BNGL file, and also values
        assigned to each __FREE__ parameter, as determined by this model's PSet

        :return: str
        """

        # Check that the model has an associated PSet
        if self.param_set is None:
            raise ModelError('Must assign a PSet to the model before calling model_text()')

        # Generate the text associated with defining __FREE__ parameter values
        param_text_lines = ['%s %s' % (k, str(self.param_set[k])) for k in self.param_names]

        # Insert the generated text at the correct point within the text of the model
        if gen_only:
            action_lines = [
                'begin actions\n',
                self.generate_network_line + '\n',
                'end actions'
            ]
            self.model_lines = \
                self.model_lines[:self.split_line_index] + \
                param_text_lines + \
                [self.model_lines[i] for i in range(self.split_line_index, len(self.model_lines))
                 if i not in self.action_line_indices + [self.generate_network_line_index]] + \
                action_lines
        else:
            self.model_lines = \
                self.model_lines[:self.split_line_index] + \
                param_text_lines + \
                self.model_lines[self.split_line_index:]

        return '\n'.join(self.model_lines) + '\n'

    def save(self, file_prefix, gen_only=False):
        """
        Saves a runnable BNGL file of the model, including definitions of the __FREE__ parameter values that are defined
        by this model's pset, to the specified location.

        :param file_prefix: str, path where the file should be saved
        :param gen_only: bool, output model with only generate_network action if True
        """

        # Call model_text(), then write the output to the file.
        if self.param_set is None:
            self.param_set = PSet({k: 1.0 for k in self.param_names})

        text = self.model_text(gen_only)
        f = open(file_prefix + '.bngl', 'w')
        f.write(text)
        f.close()

    def execute(self, folder, filename, timeout):
        """

        :param folder: Folder in which to do all the file creation
        :return: Data object
        """
        # Create the modified BNGL file
        file = '%s/%s' % (folder, filename)
        self.save(file)

        # Run BioNetGen
        cmd = [self.bng_command, '%s.bngl' % file, '--outdir', folder]
        log_file = '%s.log' % filename
        with open(log_file, 'w') as lf:
            run(cmd, check=True, stderr=STDOUT, stdout=lf, timeout=timeout)

        # Load the data file(s)
        ds = self._load_simdata(folder, filename)
        return ds

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
                data_file = '%s/%s_%s.gdat' % (folder, filename, suff[1])
                data = Data(file_name=data_file)
            else:  # suff[0] == 'parameter_scan'
                data_file = '%s/%s_%s.scan' % (folder, filename, suff[1])
                data = Data(file_name=data_file)
            ds[suff[1]] = data
        return ds


class NetModel(BNGLModel):
    def __init__(self, name, acts, suffs, ls=None, nf=None):
        self.name = name
        self.actions = acts
        self.suffixes = suffs
        self.param_set = None

        if not (ls or nf):
            raise ModelError("Must specify a file name or a list of strings corresponding to the .net file's lines")
        elif ls:
            self.netfile_lines = ls
        else:
            self.file_name = nf
            with open(self.file_name) as f:
                self.netfile_lines = f.readlines()

    def copy_with_param_set(self, pset):
        """
        Returns a copy of the model in .net format, but with a new parameter set

        :param pset: A set of new parameters for the model
        :type pset: PSet
        :return: NetModel
        """
        self.param_set = pset
        lines_copy = copy.deepcopy(self.netfile_lines)
        in_params_block = False
        for i, l in enumerate(lines_copy):
            if re.match('begin\s+parameters', l.strip()):
                in_params_block = True
            elif re.match('end\s+parameters', l.strip()):
                in_params_block = False
            elif in_params_block:
                m = re.match('(\s+)(\d)+\s+([A-Za-z_]\w*)(\s+)([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)(?=\s+)', l)
                if m:
                    if m.group(3) in self.param_set.keys():
                        lines_copy[i] = '%s%s %s%s%s\n' % (m.group(1), m.group(2), m.group(3), m.group(4), str(self.param_set[m.group(3)]))

        return NetModel(self.name, self.actions, self.suffixes, ls=lines_copy)

    def save(self, file_prefix):
        with open(file_prefix + '.net', 'w') as wf:
            wf.write(''.join(self.netfile_lines))
        with open(file_prefix + '.bngl', 'w') as wf:
            wf.write('readFile({file=>"%s"})\n' % (file_prefix + '.net'))
            wf.write('begin actions\n\n%s\n\nend actions\n' % '\n'.join(self.actions))


class SbmlModel(Model):

    copasi_command = ''

    def __init__(self, file, pset=None):
        self.file_path = file
        self.name = re.sub(".xml", "", self.file_path[self.file_path.rfind("/") + 1:])
        self.xml = ET.parse(file)
        self.param_set = None
        if pset:
            self._set_param_set(pset)
        self.stochastic = False

    def copy_with_param_set(self, pset):

        newmodel = copy.deepcopy(self)
        newmodel._set_param_set(pset)
        return newmodel

    def _set_param_set(self, pset):
        self.param_set = pset
        root = self.xml.getroot()

        # The xml file is full of "namespaces" designed to make it difficult to parse, so extra acrobatics are required
        # here
        ns = re.search('(?<={).*(?=})', root.tag).group(0)  # Extract the namespace from the root
        space = {'sbml': ns}

        params = root.findall('sbml:model/sbml:listOfParameters/sbml:parameter', namespaces=space)
        for p in params:
            pname = p.get('name')
            if pname in self.param_set.keys():
                p.set('value', str(self.param_set[pname]))

    def model_text(self):
        return ET.tostring(self.xml.root)

    def save(self, file_prefix):
        self.xml.write('%s.xml' % file_prefix, encoding='unicode')


class ModelError(Exception):
    # These are sometimes but not always user-generated, so need to be able to pass the info back to the
    # user exception handler.
    def __init__(self, message):
        self.message = message


class PSet(object):
    """
    Class representing a parameter set

    """

    def __init__(self, param_dict, allow_negative=False):
        """
        Creates a Pset based on the given dictionary

        :param param_dict: A dictionary containing the parameters to initialize, in the form of str:float pairs,
            {"paramname1:paramvalue1, ...}
        """

        # Check input values are the correct type
        for key in param_dict:
            value = param_dict[key]
            if type(key) != str:
                raise TypeError("Parameter key " + str(key) + " is not of type str")
            if not allow_negative and value < 0:
                raise ValueError("Parameter value " + str(value) + " with key " + str(key) + " is negative")
            if np.isnan(value) or np.isinf(value):
                raise ValueError("Parameter value " + str(value) + " with key " + str(key) + " is invalid")

        self._param_dict = param_dict
        self.name = None  # Can be set by Algorithms to give it a meaningful label in output file.

    def __getitem__(self, item):
        """
        Returns the value of the specified parameter.

        This allows the standard dictionary syntax ps['paramname']
         to be used for accessing (but not changing) parameters.

        :param item: The str name of the parameter to look up
        :return: float
        """
        return self._param_dict[item]

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
        unique_str = ''.join([self.keys_to_string(), self.values_to_string()])
        return hash(unique_str)

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


class Trajectory(object):
    """
    Tracks the various PSet instances and the corresponding objective function values
    """

    def __init__(self, max_output):
        self.trajectory = dict()
        self.names = dict()
        self.max_output = max_output

    def _valid_pset(self, pset):
        """
        Checks to confirm that a PSet is compatible with this Trajectory

        :param pset: A PSet instance
        :return: bool
        """
        existing_pset = next(iter(self.trajectory.keys()))
        return pset.keys() == existing_pset.keys()

    def add(self, pset, obj, name):
        """
        Adds a PSet to the fitting trajectory

        :param pset: A particular point in parameter space
        :param obj: The objective function value upon executing the model at this point in parameter space
        :raises: Exception
        """
        if len(self.trajectory) > 0:
            if not self._valid_pset(pset):
                raise ValueError("PSet %s has incompatible parameters" % pset)
        if np.isnan(obj):
            # Treat nan values as Inf in order to sort correctly
            self.trajectory[pset] = np.inf
        else:
            self.trajectory[pset] = obj
        self.names[pset] = name

    def _write(self):
        """Writes the Trajectory in a tab-delimited format"""
        s = ''
        header = next(iter(self.trajectory.keys())).keys_to_string()
        s += '#\tSimulation\tObj\t%s\n' % header
        num_output = 0
        for k in sorted(self.trajectory, key=self.trajectory.get):
            s += '\t%s\t%s\t%s\n' % (self.names[k], self.trajectory[k], k.values_to_string())
            num_output += 1
            if num_output == self.max_output:
                break
        return s

    def write_to_file(self, filename):
        """
        Writes the Trajectory to a specified file

        :param filename: File to store Trajectory
        """
        with open(filename, 'w') as f:
            f.write(self._write())
            f.close()

    def best_fit(self):
        """
        Finds the best fit parameter set

        :return: PSet
        """
        return min(self.trajectory, key=self.trajectory.get)

    def best_fit_name(self):
        """
        Finds the name of the best fit parameter set (which is also the folder
        where that result is stored)

        :return: str
        """
        return self.names[self.best_fit()]
