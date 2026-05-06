.. _installation:

Installation
============

Operating System
----------------
PyBNF can be installed on any recent Linux or macOS operating system.

PyBNF can also be installed on Windows, but functionality on Windows has been less extensively tested (in particular, Windows clusters and multicore workstations have not been tested). 

Python
------

PyBNF requires an installation of Python version 3.10 or higher.
This requirement is declared in the package metadata, so installers such as ``pip`` and ``uv`` will refuse to install PyBNF into unsupported Python environments with a clear Python-version error.

Linux and Mac
^^^^^^^^^^^^^

Python 3 comes built-in on many new Linux and Mac operating systems. 
To check if you have a supported Python 3, run the command ``python3 --version``. This will print
the version number, or will give an error if you don't have Python 3. Confirm that the version is 3.10 or higher.

Also confirm that your Python 3 has the ``pip`` package manager, which is used to install PyBNF. Run the command ``python3 -m pip``. This will give a help message if you have pip, or an error if not. 

If you are missing python3 or pip, an easy way to get them is by installing the `Anaconda`_ Python distribution for Python v3.10 or higher.
Instructions for installing on various platforms can be found on the `Anaconda`_ website.

.. _windows_install:

Windows
^^^^^^^

Windows does not come with built-in Python, so it must be installed separately. Additionally, if :ref:`BioNetGen <bng_install>` will be used, Perl installation is required in the same environment as the python installation (i.e., the commands ``python`` and ``perl`` must both work on the same command line).

Our recommended configuration consists of installing `Strawberry Perl`_ and `Anaconda`_ Python 3.10 or higher. The Windows distribution of Anaconda includes the application "Anaconda Prompt", which provides a command line. This is the command line that you should use whenever this documentation refers to the command line or terminal. After installing both Anaconda and Strawberry Perl, a system restart may be required for Anaconda Prompt to find the Perl installation.

For troubleshooting, or more advanced configuration, note that the requirement is to have both Python 3 and Perl on the current path. The current path can be checked with the command ``echo %PATH%`` and set (temporarily) with the command ``set PATH=[newpath]``, where ``[newpath]`` is a semicolon-delimited list of directories to search. 

.. Permanently setting the path is a nightmare: https://stackoverflow.com/questions/19287379/how-do-i-add-to-the-windows-path-variable-using-setx-having-weird-problems


PyBNF
-----

Installing from PyPI
^^^^^^^^^^^^^^^^^^^^

Simply type the following in a terminal:

    :command:`python3 -m pip install pybnf`

Windows users running Anaconda Python 3 from "Anaconda Prompt" should instead type only ``pip install pybnf``.

The above command will use your current version of Python 3 to install the most recent version of PyBNF released on the Python Package Index, along with all required dependencies. 
If your Python version is older than 3.10, ``pip`` will stop before installation and report that PyBNF requires Python 3.10 or higher.

Depending on your Python configuration, the above command may require root access and install PyBNF for all users on the computer. If you don't want to do this, you may add the flag ``--user`` to the end of the command, to install without root access for only the current user. 

Advanced Python users may consider installing PyBNF in a `virtualenv`_ (which also does not require root access) to avoid conflicts between PyBNF's dependencies and other uses of Python on the computer. 

Installing with uv
^^^^^^^^^^^^^^^^^^

If `uv`_ is installed, PyBNF can be installed as an isolated command-line tool:

    :command:`uv tool install pybnf`

To run PyBNF once without keeping a tool installation, use:

    :command:`uvx pybnf --help`

As with ``pip``, ``uv`` will select a Python version compatible with PyBNF or report that Python 3.10 or higher is required.

Installing from source
^^^^^^^^^^^^^^^^^^^^^^
To use bleeding edge PyBNF, the source code may be found on GitHub at https://github.com/lanl/PyBNF .  To use,
simply download or clone the repository and run the following command in the repository's root directory:

    :command:`python3 -m pip install -e .`

This also allows developers to modify the source code while still having access to the command line functionality
anywhere in the filesystem.

With ``uv``, the equivalent editable source setup is:

    :command:`uv venv --python 3.12`

    :command:`uv pip install -e .`


Installation of External Simulators
-----------------------------------

BNGsim
^^^^^^

The BNGsim Python package is installed automatically as a PyBNF dependency. PyBNF uses BNGsim for supported in-process BNGL simulation paths when it is available, and for SBML models when ``sbml_backend = bngsim`` is configured.

BNGsim does not replace all uses of BioNetGen in PyBNF. BNGL workflows can still require :ref:`BioNetGen <bng_install>` so PyBNF can run ``BNG2.pl`` to prepare ``.net`` or XML simulator inputs.

.. _bng_install:

BioNetGen
^^^^^^^^^
PyBNF is designed to work with simulators present in the BioNetGen software suite, version 2.3 or higher, available for download from
the `BioNetGen`_ website. 
Note that for Linux distributions other than Ubuntu, the pre-built binary is unreliable, and it is necessary to rebuild BioNetGen from source. 
For Windows, Perl must be installed separately, as described :ref:`above <windows_install>`.
The current BioNetGen distribution includes support for both network-based simulations and network-free simulations. 

.. _set_bng_path:
\


PyBNF will need to know the location of BioNetGen – specifically the location of the script ``BNG2.pl`` within the
BioNetGen installation. This path can be included in the PyBNF configuration file with the :ref:`bng_command <bng_command>` key. 
A convenient alternative is to set the environment variable ``BNGPATH`` to the BioNetGen directory using the following command:

    :command:`export BNGPATH=/path/to/bng2`

where ``/path/to/bng2`` is the path to the folder containing ``BNG2.pl``, not including the "BNG2.pl" file name. This 
setting may be made permanent as of your next login, by copying above command into the file ``.bash_profile``
in your home directory.

On Windows systems, the equivalent commands are ``set BNGPATH=C:\path\to\bng2`` to set on the current command line, 
and ``setx BNGPATH C:\path\to\bng2`` to permanently set for all future command lines (but not the current one). 

SBML
^^^^
PyBNF runs simulations of `SBML`_ models using `libroadrunner`_, which is installed automatically through ``pip`` as part of 
PyBNF installation. 

To work with SBML files, it is useful to install software such as `COPASI`_ that is capable of reading and writing models in
SBML format. 


.. _Anaconda: https://www.anaconda.com/download
.. _BioNetGen: http://www.bionetgen.org
.. _SBML: http://sbml.org/
.. _libroadrunner: http://libroadrunner.org/
.. _COPASI: http://copasi.org/
.. _virtualenv: https://packaging.python.org/guides/installing-using-pip-and-virtualenv/
.. _Strawberry Perl: http://strawberryperl.com/
.. _uv: https://docs.astral.sh/uv/
