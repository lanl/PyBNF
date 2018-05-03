Algorithm Development
=====================

PyBNF was designed with extensibility in mind.  As a result, all of the algorithms implemented here subclass the
Algorithm class found in :ref:`alg_module`.

Implementation
--------------
A new algorithm can be written by creating a class that subclasses the Algorithm class::

    class NewAlgorithm(Algorithm):
        def __init__(self, config, **kwargs):
            super(NewAlgorithm, self).__init__(config)
            # Other setup that may involve additional arguments
            self.current_iter_results = []
            self.ready_for_next_iter = False

        def generate_random_psets(self):
            # User defined support function that
            # generates a list of PSet instances
            ...

The new algorithm requires defining three methods, with the first being the ``__init__`` constructor method.  This
method will likely take a Configuration object as its first argument.  The other two required methods that must be
implemented are the ``start_run`` and ``got_result`` methods.

The ``start_run`` method will return a list of PSet instances that correspond to the first batch of parameter set
evaluations.::

    def start_run(self):
        return self.generate_random_psets()

The ``got_result`` method takes a Result instance as an argument and returns either a list of new PSet instances for
another round of parameter set evaluations, or the string "STOP" to terminate the fitting run.  Note that an empty list
is valid if the algorithm requires synchronization (and thus must wait for all jobs in the current iteration to
finish).::

    def got_result(self, res):
        if self.satisfies_stop_condition(res):
            return "STOP"  # Terminates algorithm

        self.current_iter_results.append(res)
        if self.ready_for_next_iter:  # Synchronization check
            new_psets = []
            for r in self.current_iter_results:
                new_psets.append(self.generate_new_pset(r))
            return new_psets

        return []  # Waiting for synchronization

Adding configuration options
----------------------------

If the new algorithm requires user configuration via the configuration file, new options may be added to the
``pybnf.parse`` module.  The configuration parser uses the ``pyparsing`` module and new grammars for parsing individual
lines may be added to the ``pybnf.parse.parse`` function based on the key text.  Default values for parameters may be
added to the ``Configuration`` object via its ``default_config`` method in the ``pybnf.config`` module if desired.
Other supporting configuration methods should also be added to the ``Configuration`` object if necessary.

Pull requests
-------------

To have new algorithms added into the PyBNF software suite, submit a pull request to the master branch at
`<https://github.com/NAU-BioNetFit/PyBNF>`_

