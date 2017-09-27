import numpy as np
import re
import copy
import warnings


class Model(object):
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

        # Read the file
        with open(self.file_path) as file:
            self.bngl_file_text = file.read()

        # Scan the file's lines
        # Find param names of type __FREE__, and also the 'begin parameters' declaration
        param_names_set = set()
        split_line = None
        linelist = self.bngl_file_text.splitlines()
        for linei in range(len(linelist)):
            line = linelist[linei]
            # Remove comment if present
            commenti = line.find('#')
            if commenti != -1:
                line = line[:commenti]

            # Find every item matching [alphanumeric]__FREE__
            params = re.findall('[A-Za-z_]\w*__FREE__', line)
            for p in params:
                param_names_set.add(p)

            # Check if this is the 'begin parameters' line
            if re.match('begin\s+parameters', line.strip()):
                if split_line is not None:
                    raise ModelError("Found a second instance of 'begin parameters' at line " + str(linei))
                split_line = linei + 1

            action_suffix = self._get_action_suffix(line)
            if action_suffix is not None:
                self.suffixes.append(action_suffix)

        if len(param_names_set) == 0:
            raise ModelError("No free parameters found")

        if split_line is None:
            raise ModelError("'begin parameters' not found in BNGL file")

        # Two pieces of the model text. The full model with params should be written as _model_text_start + (free param definitions) + _model_text_end
        self._model_text_start = '\n'.join(linelist[:split_line] + [''])
        self._model_text_end = '\n'.join(linelist[split_line:])

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

    def set_param_set(self, pset):
        """
        Assign a set of parameters to the model

        :param pset: A PSet object containing the parameters to be set.
        """

        warnings.warn('set_param_set() is deprecated. For work with the parallel scheduler, instead use '
                      'copy_with_param_set() to create a new Model instance.', DeprecationWarning)

        # Check that the PSet has definitions for the right parameters for this model
        if pset.keys_to_string() != '\t'.join(self.param_names):
            raise ValueError('Parameter names in the PSet do not match those in the Model')

        self.param_set = pset

    def copy_with_param_set(self, pset):
        """
        Returns a copy of this model containing the specified parameter set.

        :param pset: A PSet object containing the parameters for the new instance
        :return:
        """
        # Check that the PSet has definitions for the right parameters for this model
        if set(pset.keys()) != set(self.param_names):
            raise ValueError('Parameter names in the PSet do not match those in the Model\n%s\n%s' % (pset.keys(), self.param_names))

        newmodel = copy.deepcopy(self)
        newmodel.param_set = pset
        return newmodel

    def model_text(self):
        """
        Returns the text of a runnable BNGL file, which includes the contents of the original BNGL file, and also values
        assigned to each __FREE__ parameter, as determined by this model's PSet

        :return: str
        """

        # Check that the model has an associated PSet
        if self.param_set is None:
            raise ModelError('Must assign a PSet to the model before calling model_text()')

        # Generate the text associated with defining __FREE__ parameter values
        param_text_lines = [k + ' ' + str(self.param_set[k]) for k in self.param_names]
        param_text = '\n'.join(param_text_lines)

        # Insert the generated text at the correct point within the text of the model
        return ''.join([self._model_text_start, param_text, self._model_text_end])

    def save(self, filename):
        """
        Saves a runnable BNGL file of the model, including definitions of the __FREE__ parameter values that are defined
        by this model's pset, to the specified location.

        :param filename: str, path where the file should be saved
        """

        # Call model_text(), then write the output to the file.
        text = self.model_text()
        f = open(filename, 'w')
        f.write(text)
        f.close()


class ModelError(Exception):
    pass


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

    def __getitem__(self, item):
        """
        Returns the value of the specified parameter.

        This allows the standard dictionary syntax ps['paramname']
         to be used for accessing (but not changing) parameters.

        :param item: The str name of the parameter to look up
        :return: float
        """
        return self._param_dict[item]

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

    def __init__(self):
        self.trajectory = dict()

    def _valid_pset(self, pset):
        """
        Checks to confirm that a PSet is compatible with this Trajectory

        :param pset: A PSet instance
        :return: bool
        """
        existing_pset = next(iter(self.trajectory.keys()))
        return pset.keys() == existing_pset.keys()

    def add(self, pset, obj):
        """
        Adds a PSet to the fitting trajectory

        :param pset: A particular point in parameter space
        :param obj: The objective function value upon executing the model at this point in parameter space
        :raises: Exception
        """
        if len(self.trajectory) > 0:
            if not self._valid_pset(pset):
                raise ValueError("PSet %s has incompatible parameters" % pset)
        self.trajectory[pset] = obj

    def _write(self):
        """Writes the Trajectory in a tab-delimited format"""
        s = ''
        header = next(iter(self.trajectory.keys())).keys_to_string()
        s += '#\t%s\tObj\n' % header
        for k in sorted(self.trajectory, key=self.trajectory.get):
            s += '\t%s\t%s\n' % (k.values_to_string(), self.trajectory[k])
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
