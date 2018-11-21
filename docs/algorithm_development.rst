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

        def my_custom_function(self):
            # User defined support function
            ...

The new algorithm requires defining three methods, with the first being the ``__init__`` constructor method.  This
method will likely take a Configuration object as its first argument.  The other two required methods that must be
implemented are the ``start_run`` and ``got_result`` methods.

The ``start_run`` method must return a list of PSet instances that correspond to the first batch of parameter set
evaluations. The Algorithm superclass functions ``random_pset`` and ``random_latin_hypercube_psets`` may be useful::

    def start_run(self):
        return self.random_latin_hypercube_psets(self.population_size)

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


Four additional support methods in the Algorithm superclass may optionally be overridden, depending on the details of the new algorithm, 
such that the new algorithm is compatible with all features of PyBNF. 

    * ``add_iterations(self,n)`` is required to support adding extra iterations with the ``-r`` flag. This method should add ``n`` iterations to the algorithm's maximum iteration count. The superclass implementation simply adds ``n`` to the attribute ``self.max_iterations``. You should override the method if your algorithm tracks iteration count in a different way. 
    * ``reset(self, bootstrap)`` is required to support bootstrapping. This method should call the superclass method, and then reset the state of the algorithm so that another fitting replicate can be run. 
    * ``get_backup_every(self)`` helps choose when to save a backup of the algorithm. This method should return an integer telling after how many individual simulations we should back up the algorithm. The superclass implementation uses a formula that should work in most cases, but you can override this depending on details of your algorithm. 
    * ``cleanup(self)`` is used to clean up after an error. This method is called just before PyBNF exits due to an error or keyboard interrupt, and may be used to save any useful files before exiting. 



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
`<https://github.com/lanl/PyBNF>`_

