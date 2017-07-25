import numpy as np
import re


class Model(object):
    """
    Class representing a BNGL model


    """

    def __init__(self, bngl_file):
        """
        Loads the model from the given .bngl file

        :param bngl_file: str address of the bngl file
        """

        # Read the file
        with open(bngl_file) as file:
            self.bngl_file_text = file.read()

        # Scan the file for param names of type __FREE__
        param_names_set = set()
        for line in self.bngl_file_text.splitlines():
            # Remove comment if present
            commenti = line.find('#')
            if commenti != -1:
                line = line[:commenti]

            # Find every item matching [alphanumeric]__FREE__
            params = re.findall('[A-Za-z_]\w*__FREE__', line)
            for p in params:
                param_names_set.add(p)

        if len(param_names_set) == 0:
            raise ModelError("No free parameters found")

        # Save model_params as a sorted tuple
        param_names_list = list(param_names_set)
        param_names_list.sort()
        self.param_names = tuple(param_names_list)

        # Find the point in the file where we would eventually write in the values of the free paramters

        split_index = self.bngl_file_text.find(
            '\nbegin parameters\n') + 18  # Position after the "begin parameters" line
        if split_index == 17:
            raise ModelError("'begin parameters' not found in BNGL file")

        # Two pieces of the model text. The full model with params should be written as _model_text_start + (free param definitions) + _model_text_end
        self._model_text_start = self.bngl_file_text[:split_index]
        self._model_text_end = self.bngl_file_text[split_index:]

        self.param_set = None

    def set_param_set(self, pset):
        """
        Assign a set of parameters to the model

        :param pset: A PSet object containing the parameters to be set.
        """

        # Check that the PSet has definitions for the right parameters for this model
        if pset.keys_to_string() != '\t'.join(self.param_names):
            raise ValueError('Parameter names in the PSet do not match those in the Model')

        self.param_set = pset

    def model_text(self):
        """
        Returns the text of a runnable BNGL file, which includes the contents of the original BNGL file, and also values
        assigned to each __FREE__ parameter, as determined by this model's PSet

        :return: str
        """

        # Check that the model has an associated PSet
        if self.param_set == None:
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

    def __init__(self, param_dict):
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
            if type(value) != float:
                raise TypeError("Parameter value " + str(value) + " with key " + str(key) + " is not of type float")
            if value < 0:
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
