Algorithm Development
=====================

PyBNF was designed with extensibility in mind.  As a result, all of the algorithms implemented here subclass the
Algorithm class found in :ref:`alg_module`.

Implementation
--------------
A new algorithm can be written by creating a class that subclasses the Algorithm class and
registering it with the ``@register_fit_type`` decorator (from ``pybnf.registry``), which
names the ``job_type`` code(s) that select it, its family (``optimizer``, ``sampler``, or
``checker``), a display name, and its config schema (see `Adding configuration options`_)::

    from pybnf.registry import register_fit_type

    @register_fit_type('newalg', family='optimizer', display_name='My Algorithm',
                       schema=NewAlgorithmConfig)
    class NewAlgorithm(Algorithm):
        def __init__(self, config, **kwargs):
            super(NewAlgorithm, self).__init__(config)
            # Other setup that may involve additional arguments

        def my_custom_function(self):
            # User defined support function
            ...

Your algorithm is then selected with ``job_type = newalg``. The decorator, the registry it
fills (``FIT_TYPE_REGISTRY``), and the config slot they are read from keep the older
``fit_type`` spelling: the rename to ``job_type`` was deliberately confined to the
configuration surface, since the key selects samplers and the model checker as well as
fits, and renaming the internal symbols was out of its scope (ADR-0028). Read ``fit_type``
in code as the internal name for the same thing a user writes as ``job_type`` — or, in a
legacy (edition-1) config, as ``fit_type``.

The new algorithm requires defining three methods, with the first being the ``__init__`` constructor method.  This
method will likely take a Configuration object as its first argument.  The other two required methods that must be
implemented are the ``start_run`` and ``got_result`` methods.

The ``start_run`` method is called at the start of the run. It must return a list of PSet instances, as the first batch of parameter sets
to be evaluated. The Algorithm superclass functions ``random_pset`` and ``random_latin_hypercube_psets`` may be useful::

    def start_run(self):
        return self.random_latin_hypercube_psets(self.population_size)

The ``got_result`` method is called each time an evaluation of a PSet is completed on a worker. It takes a Result instance 
as an argument and returns either a list of new PSet instances for
another round of parameter set evaluations, or the string "STOP" to terminate the run.  Note that an empty list
is valid if the algorithm requires synchronization (and thus must wait for all jobs in the current iteration to
finish). For example::

    def got_result(self, res):
        if self.satisfies_stop_condition(res):
            return "STOP"  # Terminates algorithm

        self.current_iter_results.append(res)
        if self.ready_for_next_iter:  # Synchronization check
            new_psets = []
            for r in self.current_iter_results:
                new_psets.append(self.generate_new_pset(r))
            return new_psets
        else:
            return []  # Waiting for synchronization


Four additional support methods in the Algorithm superclass may optionally be overridden, depending on the details of the new algorithm, 
such that the new algorithm is compatible with all features of PyBNF. 

    * ``add_iterations(self,n)`` is required to support adding extra iterations with the ``-r`` flag. This method should add ``n`` iterations to the algorithm's maximum iteration count. The superclass implementation simply adds ``n`` to the attribute ``self.max_iterations``. You should override the method if your algorithm tracks iteration count in a different way. 
    * ``reset(self, bootstrap)`` is required to support bootstrapping. This method should call the superclass method, and then reset the state of the algorithm so that another fitting replicate can be run. 
    * ``get_backup_every(self)`` helps choose when to save a backup of the algorithm. This method should return an integer telling after how many individual simulations we should back up the algorithm. The superclass implementation uses a formula that should work in most cases, but you can override this depending on details of your algorithm. 
    * ``cleanup(self)`` is used to clean up after an error. This method is called just before PyBNF exits due to an error or keyboard interrupt, and may be used to save any useful files before exiting. 



Adding configuration options
----------------------------

An algorithm's user-settable options live in a **Pydantic config schema** — a class
subclassing ``PyBNFConfigModel`` (or a family base such as ``MCMCFamilyConfig``), co-located
with the algorithm class and wired to the job type through the ``schema=`` argument of
``@register_fit_type``. The schema is the single source of truth for the method's option
names, types, and defaults; each job's effective config is narrowed to the global keys plus
its own schema's keys, so a method only ever sees the options it reads::

    from pybnf.config_schema import PyBNFConfigModel

    class NewAlgorithmConfig(PyBNFConfigModel):
        my_option: float = 1.0          # a new key, with its type and default
        my_count: int = 10

A new numeric key must also be listed in the appropriate token-type list in ``pybnf.parse``
(for example ``numkeys_int`` or ``numkeys_float``) so the configuration parser reads and
coerces its value. The parser uses the ``pyparsing`` module; a key with a more elaborate
grammar than a single scalar adds a parse rule there as well.

Pull requests
-------------

To have new algorithms added into the PyBNF software suite, submit a pull request to the main branch at
`<https://github.com/lanl/PyBNF>`_

