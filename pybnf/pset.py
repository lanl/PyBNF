import numpy as np
import re


class Model(object):

    """
    Class representing a BNGL model

    """

    def __init__(self,bngl_file):
        """
        Loads the model from the given .bngl file

        :param bngl_file: str address of the bngl file
        """

        # Read the file
        with open(bngl_file) as file:
            self.model_text = file.read()

        # Scan the file for param names of type __FREE__
        param_names_set = set()
        for line in self.model_text.splitlines():
            # Remove comment if present
            commenti = line.find('#')
            if commenti!=-1:
                line = line[:commenti]

            #Find every item matching [alphanumeric]__FREE__
            params = re.findall('[A-Za-z_]\w*__FREE__',line)
            for p in params:
                param_names_set.add(p)

        if len(param_names_set) == 0:
            raise ParseError("No free parameters found")

        # Save model_params as a sorted tuple
        param_names_list = list(param_names_set)
        param_names_list.sort()
        self.param_names = tuple(param_names_list)

        # Find the point in the file where we would eventually write in the values of the free paramters

        split_index = self.model_text.find('\nbegin parameters\n') + 18 # Position after the "begin parameters" line
        if split_index == 17:
            raise ParseError("'begin parameters' not found in BNGL file")

        # Two pieces of the model text. The full model with params should be written as _model_text_start + (free param definitions) + _model_text_end
        self._model_text_start = self.model_text[:split_index]
        self._model_text_end = self.model_text[split_index:]



class ParseError(Exception):
    pass

class ParseWarning(Warning):
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
