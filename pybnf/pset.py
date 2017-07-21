class Pset(dict):
    """
    Class representing a parameter set

    Pset is a subclass of the built-in dict class, intended to contain str keys and float values, of the form
    {'paramname1':paramvalue1}

    Pset can be constructed the same way as a dictionary, either by passing a dict to the constructor, or by passing no
    arguments to the constructor, and subsequently setting individual parameters

    The standard dict syntax may be used to access individual parameter values, e.g. ps['paramname1']

    """

    # No separate constructor for Pset; use the dict constructor.

    # def __init__(self,iterable=None,**kwargs):
    #
    #     if iterable==None:
    #         iterable = dict()
    #
    #     dict.__init__(self, iterable, **kwargs)


    def get_id(self):

        return self.__hash__()


    def __hash__(self):
        """
        Returns a unique identifier for this parameter set
        Two Psets will have the same identifier if they have the same keys and corresponding values

        :return: int
        """
        unique_str = ''.join([self.keys_to_string(), self.values_to_string()])
        return hash(unique_str)


    def keys_to_string(self):
        """
        Returns the keys (parameter names) in a tab-separated str in alphabetical order

        :return: str
        """
        keys = [str(k) for k in self.keys()]
        keys.sort()
        return '\t'.join(keys)

    def values_to_string(self):
        """
        Returns the parameter values in a tab-separated str, in alphabetical order
        according to the parameter name
        :return: str
        """
        keys = [str(k) for k in self.keys()]
        keys.sort()
        values = [str(self[k]) for k in self.keys()]  # Values are in alpha order by key name
        return '\t'.join(values)
