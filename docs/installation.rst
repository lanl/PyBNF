Installation
============

Python
------

PyBNF requires an installation of Python version 3.5 or higher. This should come built-in with most new Linux and Mac
operating systems.  However, we recommend installing the `Anaconda`_ Python distribution for Python v3.5 or higher.
Installing `Anaconda`_ facilitates managing and installing Python packages as well as maintaining multiple Python
environments. Instructions for installing on various platforms can be found on the Anaconda website.

PyBNF
-----
The ``pip`` package manager comes with `Anaconda`_ and should be used to install PyBNF from the command line.

Installing from PyPI
^^^^^^^^^^^^^^^^^^^^
.. note::
    This option is not yet available

Simply type the following in a terminal:

    :command:`pip install pybnf`

The above command will install the most recent version of PyBNF released on the Python Package Index

Installing from source
^^^^^^^^^^^^^^^^^^^^^^
To use bleeding edge PyBNF, the source code may be found on GitHub at https://github.com/NAU-BioNetFit/PyBNF .  To use,
simply download or clone the repository and run the following command in the repository's root directory:

    :command:`pip install -e .`

This also allows developers to modify the source code while still having access to the commmand line functionality
anywhere in the filesystem.

(Optional) Logging configuration for remote machines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default, PyBNF logs to the file ``bnf_timestamp.log`` to maintain a record of important events in the application.
When running PyBNF on a cluster, some of the logs may be written while on a node distinct from the main thread. If
these logs are desired, the user must configure the scheduler to retrieve these logs.

Upon installation of PyBNF, the dependencies ``dask`` and ``distributed`` should be installed. Installing them will
create a ``.dask/`` folder in the home directory with a single file: ``config.yaml``. Open this file to find a
``logging:`` block containing information for how distributed outputs logs. Add the following line to the file,
appropriately indented:

    :command:`pybnf.algorithms.job: info`

where ``info`` can be any string corresponding to a Python logging level (e.g. ``info``, ``debug``, ``warning``)

Installation of External Simulators
-----------------------------------

BioNetGen
^^^^^^^^^
PyBNF is designed to work with simulators present in the `BioNetGen`_ software suite, version 2.3. The current
`BioNetGen`_ distribution includes support for both network-based simulations and network-free simulations.

PyBNF will need to know the location of `BioNetGen`_ – specifically the location of the script ``BNG2.pl`` within the
`BioNetGen`_ installation. This path can be included in the PyBNF configuration file (see below). A convenient alternative
is to set the environment variable ``BNGPATH`` to the BioNetGen directory using the following command:

    :command:`export BNGPATH=/path/to/bng2`

which can also be made permanent as of your next login, by copying above command into the file ``.bash_profile``
in your home directory.


.. _Anaconda: https://www.anaconda.com/download
.. _BioNetGen: http://www.bionetgen.org