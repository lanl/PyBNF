"""pybnf.pset: classes for storing models, parameter sets, and the fitting trajectory"""


from .printing import print0, print1, PybnfError

import logging
import numpy as np
import re
import copy
import xml.etree.ElementTree as ET
from subprocess import run, STDOUT, DEVNULL
from .data import Data
import os
from lxml import etree

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

    def add_action(self, action):
        pass


class BNGLModel(Model):
    """
    Class representing a BNGL model


    """

    def __init__(self, bngl_file, pset=None):
        """
        Loads the model from the given .bngl file

        :param bngl_file: str address of the bngl file
        :param pset: PSet to initialize the model with. Defaults to None
        """
        self.file_path = bngl_file
        self.name = re.sub(".bngl", "", self.file_path[self.file_path.rfind("/")+1:])
        self.suffixes = []  # list of 2-tuples (sim_type, prefix)
        self.bng_command = ''

        self.generates_network = False
        self.generate_network_line = None
        self.generate_network_line_index = -1
        self.action_line_indices = []
        self.actions = []
        self.config_actions = []

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
        log_file = '%s.log' % file
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

    def add_action(self, action):
        self.config_actions.append(action)
        print0('Warning: Adding actions to BNGL models with config options is not yet supported')


class NetModel(BNGLModel):
    def __init__(self, name, acts, suffs, ls=None, nf=None):
        self.name = name
        self.actions = acts
        self.config_actions = []
        self.suffixes = suffs
        self.param_set = None
        self.bng_command = ''

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
                    if m.group(3) in pset.keys():
                        lines_copy[i] = '%s%s %s%s%s\n' % (m.group(1), m.group(2), m.group(3), m.group(4), str(pset[m.group(3)]))

        newmodel = NetModel(self.name, self.actions, self.suffixes, ls=lines_copy)
        newmodel.bng_command = self.bng_command
        newmodel.param_set = pset
        return newmodel

    def save(self, file_prefix):
        with open(file_prefix + '.net', 'w') as wf:
            wf.write(''.join(self.netfile_lines))
        with open(file_prefix + '.bngl', 'w') as wf:
            wf.write('readFile({file=>"%s"})\n' % (file_prefix + '.net'))
            wf.write('begin actions\n\n%s\n\nend actions\n' % '\n'.join(self.actions))


class SbmlModel(Model):

    def __init__(self, file, pset=None, actions=()):
        self.file_path = file
        self.name = re.sub(".xml", "", self.file_path[self.file_path.rfind("/") + 1:])
        self.xml = ET.parse(file)
        self.actions = list(actions)
        self.suffixes = []
        self.species = []
        self.param_set = None
        if pset:
            self._set_param_set(pset)
        self.stochastic = False
        self.copasi_command = ''

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

        # Todo: Potentially usable to set initial conditions.(would need to traverse even if copied from previous model)
        # Currently this check is unnecessary because this method is only called on model instances with
        # param_set = None
        if len(self.species) == 0:
            species = root.findall('sbml:model/sbml:listOfSpecies/sbml:species', namespaces=space)
            for s in species:
                self.species.append(s.get('name'))

    def model_text(self):
        return ET.tostring(self.xml.root)

    def save(self, file_prefix):
        self.xml.write('%s.xml' % file_prefix, encoding='unicode')

    def add_action(self, action):
        self.actions.append(action)
        self.suffixes.append((action.bng_codeword, action.suffix))

    def execute(self, folder, filename, timeout):
        # Create the modified XML file
        file = '%s/%s' % (folder, filename)
        self.save(file)

        # Convert to Copasi file
        run([self.copasi_command, '-i', '%s.xml' % file], check=True, stderr=STDOUT, stdout=DEVNULL, timeout=timeout)

        # Edit the Copasi file to include the required actions
        # Python thinks Copasi XML is invalid because of more "namespace" nonsense
        # To get around this, need to do some more complicated stuff...
        # cps = ET.parse('%s.cps' % file)  # Fails because namespaces
        # f = open('%s.cps' % file)
        # text = f.read()
        # f.close()
        parser = etree.XMLParser(recover=True)
        # Write the updated Copasi back to file
        cps = etree.ElementTree(file='%s.cps' % file, parser=parser)
        root = cps.getroot()
        # root = etree.fromstring(bytes(text, 'utf-8'), parser)

        # root = cps.getroot()
        ns = re.search('(?<={).*(?=})', root.tag).group(0)  # Extract the namespace from the root
        space = {'cps': ns}
        if len(self.actions) > 1:
            raise PybnfError('Currently only 1 action per SBML file is supported')
        if len(self.actions) == 0:
            raise ValueError('Cannot run model with no actions')

        model_elem = root.findall('cps:Model', namespaces=space)[0]
        model_name = model_elem.get('name')  # We'll need the value of this thing later for setting various attributes.
        for action in self.actions:
            # Todo: Support more than 1 action
            # Set the save file to be the same as for a BNGL file
            if action.bng_codeword == 'simulate':
                ext = 'gdat'
            else:
                ext = 'scan'
            target_file = '%s_%s_%s.%s' % (self.name, self.param_set.name, action.suffix, ext)
            if isinstance(action, TimeCourse):
                # Find the time course task in the xml file
                tasks = root.findall('cps:ListOfTasks/cps:Task', namespaces=space)
                time_task = None
                for t in tasks:
                    if t.get('name') == 'Time-Course':
                        time_task = t
                        break
                if time_task is None:
                    raise RuntimeError('Time-Course task unexpectedly missing from cps file')
                # Update attributes of time course task
                time_task.set('scheduled', 'true')
                for param in time_task.findall('cps:Problem/cps:Parameter', namespaces=space):
                    if param.get('name') == 'StepNumber':
                        param.set('value', str(action.stepnumber))
                    elif param.get('name') == 'StepSize':
                        param.set('value', str(action.step))
                    elif param.get('name') == 'Duration':
                        param.set('value', str(action.time))
                report_elem = etree.Element('Report', reference='BNF_Report_1', target=target_file, append='1',
                                            confirmOverwrite='1')
                time_task.insert(0, report_elem)
                # action-type-specific Report settings
                task_type = 'timeCourse'
                first_col_string = 'CN=Root,Model=%s,Reference=Time' % model_name
            elif isinstance(action, ParamScan):
                tasks = root.findall('cps:ListOfTasks/cps:Task', namespaces=space)
                # Find the 2 tasks that we need to edit
                time_task = None
                scan_task = None
                for t in tasks:
                    if t.get('name') == 'Time-Course':
                        time_task = t
                    if t.get('name') == 'Scan':
                        scan_task = t
                if time_task is None or scan_task is None:
                    raise RuntimeError('Time-Course and/or Scan tasks unexpectedly missing from cps file')

                # Edit the time course task so it prints one time point at the t where we're param scanning
                for param in time_task.findall('cps:Problem/cps:Parameter', namespaces=space):
                    if param.get('name') == 'StepNumber':
                        param.set('value', '1')
                    elif param.get('name') == 'StepSize':
                        param.set('value', str(action.time))
                    elif param.get('name') == 'Duration':
                        param.set('value', str(action.time))
                    elif param.get('name') == 'OutputStartTime':
                        param.set('value', '1.0e-8')  # Suppress the output at time 0
                # Edit the scan task so it runs with the chosen specs
                scan_task.set('scheduled', 'true')
                report_elem = etree.Element('Report', reference='BNF_Report_1', target=target_file, append='0',
                                            confirmOverwrite='0')
                scan_task.insert(0, report_elem)
                param_parent = scan_task.findall('cps:Problem/cps:ParameterGroup', namespaces=space)[0]
                param_subparent = etree.Element('ParameterGroup', name='ScanItem')
                param_parent.append(param_subparent)
                param_subparent.append(etree.Element('Parameter', name='Number of steps', type='unsignedInteger',
                                                     value=str(action.stepnumber)))
                param_subparent.append(etree.Element('Parameter', name='Type', type='unsignedInteger', value='1'))
                param_subparent.append(etree.Element('Parameter', name='Object', type='cn',
                                                     value='CN=Root,Model=%s,Vector=Values[%s],Reference=InitialValue' % (
                                                     model_name, action.variable)))
                param_subparent.append(etree.Element('Parameter', name='Minimum', type='float',
                                                     value=str(action.var_min)))
                param_subparent.append(etree.Element('Parameter', name='Maximum', type='float',
                                                     value=str(action.var_max)))
                param_subparent.append(etree.Element('Parameter', name='log', type='bool',
                                                     value=str(action.logspace)))
                # action-type-specific Report settings
                task_type = 'scan'
                first_col_string = 'CN=Root,Model=%s,Vector=Values[%s],Reference=InitialValue' % (model_name,
                                                                                                  action.variable)
            else:
                raise NotImplementedError('Only implemented action types are Time Course and Param Scan')

            # Create the report object
            report_list = root.findall('cps:ListOfReports', namespaces=space)[0]
            report = etree.Element('Report', key='BNF_Report_1', name='BNF_Report', taskType=task_type, separator='\t',
                                precision='6')
            report_list.append(report)
            comment = etree.Element('Comment')
            comment.text = 'Automatically generated by BioNetFit'
            report.append(comment)
            report_table = etree.Element('Table', printTitle='1')
            report.append(report_table)
            tablestring = "CN=Root,Model=%s,Vector=Compartments[compartment],Vector=Metabolites[%s],Reference=Concentration"
            # For the Model=?? attribute of tablestring, we need to get a name from somewhere else in the file
            model_elem = root.findall('cps:Model', namespaces=space)[0]
            model_name = model_elem.get('name')
            obj = etree.Element('Object', cn=first_col_string)
            report_table.append(obj)
            for s in self.species:
                obj = etree.Element('Object', cn=tablestring % (model_name, s))
                report_table.append(obj)


        cps.write('%s.cps' % file)

        # Run Copasi
        # Todo: The save file was specified in the cps file (target_file), but I don't know what directory it goes to
        # It might be your current directory, but previously that gave us problems in parallel so gah.

        cmd = [self.copasi_command, '%s.cps' % file]
        log_file = '%s.log' % file
        with open(log_file, 'w') as lf:
            run(cmd, check=True, stderr=STDOUT, stdout=lf, timeout=timeout)

        if task_type == 'timeCourse':
            # Tab-delimited, header row with no '#', extra [] around each variable
            res = Data()
            res.load_data('%s/%s' % (folder, target_file), flags=('time',))
        elif task_type == 'scan':
            res = Data()
            res.load_data('%s/%s' % (folder, target_file), flags=('copasi-scan',))
        else:
            raise RuntimeError('Unknown task type')
        return {action.suffix: res}


class Action:
    """
    Represents a simulation action performed within a model
    """
    pass


class TimeCourse(Action):

    def __init__(self, time, step):
        self.time = time
        self.step = step
        self.stepnumber = int(np.round(time/step))
        self.suffix = 'time_course'
        self.bng_codeword = 'simulate'


class ParamScan(Action):

    def __init__(self, variable, var_min, var_max, step, time, logspace=False):
        self.variable = variable
        self.var_min = var_min
        self.var_max = var_max
        self.step = step
        self.stepnumber = int(np.round((var_max-var_min) / step))
        self.time = time
        self.logspace = int(logspace)
        self.suffix = 'param_scan'
        self.bng_codeword = 'parameter_scan'


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
