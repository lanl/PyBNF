.. _installation:

Installation
============

Operating System
----------------
PyBNF can be installed on any recent Linux or macOS (Apple Silicon and Intel) operating system.

PyBNF can also be installed on Windows, but functionality on Windows has been less extensively tested (in particular, Windows clusters and multicore workstations have not been tested). 

Python
------

PyBNF requires an installation of Python version 3.10 or higher. 

Installation of PyBNF in a virtual environment (``python -m venv .venv``) is recommended. You can also create an environment using Conda.

.. code-block:: bash
    
    conda create -n pybnf_env python=3.12
    conda activate pybnf_env
    python -m pip install -U pip
    python -m pip install "pybnf[sbml]"

If you use Conda, install packages with ``python -m pip`` **inside the activated Conda environment** (as shown above). Do not install into the base environment.

Linux and macOS
^^^^^^^^^^^^^^^

Linux distributions typically include Python 3; current macOS releases often do not, so you may need to install it (e.g., via Anaconda/Conda or Homebrew). To check if you have Python 3, run ``python3 --version``. To install Python 3 using Homebrew:

.. code-block:: bash

    brew install python

Confirm the CLI is on PATH:

.. code-block:: bash

    command -v pybnf # POSIX

Confirm that your installation of Python 3 has the ``pip`` package manager, which is used to install PyBNF. Run the command ``python3 -m pip --version``.

If you are missing Python 3 or pip, an easy way to get them is by installing the `Anaconda`_ Python distribution for Python v3.10 or higher.
Instructions for installing on various platforms can be found on the `Anaconda`_ website.

.. _windows_install:

Windows
^^^^^^^

Windows does not come with built-in Python, so it must be installed separately. Additionally, if :ref:`BioNetGen <bng_install>` will be used, Perl installation is required in the same environment as the Python installation (i.e., the commands ``python`` and ``perl`` must both work on the same command line).

Our recommended configuration consists of installing `Strawberry Perl`_ and `Anaconda`_ Python 3. The Windows distribution of Anaconda includes the application "Anaconda Prompt", which provides a command line. This is the command line that you should use whenever this documentation refers to the command line or terminal. After installing both Anaconda and Strawberry Perl, a system restart may be required for Anaconda Prompt to find the Perl installation. Do not mix base cmd.exe and Anaconda Prompt. Use one shell per environment. Quick checks:

.. code-block:: bat

    where python
    perl -v

For troubleshooting, or more advanced configuration, note that the requirement is to have both Python 3 and Perl on the current path. The current path can be checked with the command ``echo %PATH%`` and set (temporarily) with the command ``set PATH=[newpath]``, where ``[newpath]`` is a semicolon-delimited list of directories to search. 

.. Permanently setting the path is a nightmare: https://stackoverflow.com/questions/19287379/how-do-i-add-to-the-windows-path-variable-using-setx-having-weird-problems


PyBNF
-----

Installing from PyPI
^^^^^^^^^^^^^^^^^^^^

Use a clean virtual environment if possible.

.. code-block:: bash

   python3 -m pip install -U pip
   python3 -m pip install "pybnf[sbml]"

Windows users running Anaconda Python from "Anaconda Prompt" can use:

.. code-block:: bat

   python -m pip install -U pip
   python -m pip install "pybnf[sbml]"

The command above installs PyBNF plus SBML support (via ``libroadrunner``). For BNGL-only workflows:

.. code-block:: bash

   python3 -m pip install pybnf

If you lack write permissions, append ``--user`` or install inside a virtual environment.

Installing from source
^^^^^^^^^^^^^^^^^^^^^^
Bleeding-edge source is at https://github.com/lanl/PyBNF

Choose one of the following editable install options, depending on whether you need SBML support.

.. code-block:: bash

   # upgrade pip (optional but recommended):
   python3 -m pip install -U pip

   # Editable install for development (without SBML):
   python3 -m pip install -e ".[dev]"

   # Editable install with SBML support:
   python3 -m pip install -e ".[dev,sbml]"

This lets you edit the source while keeping the ``pybnf`` CLI on your PATH.

Verify installation
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Check CLI is installed
   pybnf --help

   # If using BNGL, verify BioNetGen is visible
   echo "$BNGPATH"
   test -f "$BNGPATH/BNG2.pl" && echo "Found BNG2.pl" || echo "BNG2.pl not found"
   "$BNGPATH/BNG2.pl" --help | head -n 3

If the last command fails, set ``BNGPATH`` as described above or specify ``bng_command`` in your ``.conf`` file.

.. code-block:: bash

   # If using SBML/libRoadRunner, verify RoadRunner import (POSIX shells)
   python3 - <<'PY'
   import roadrunner as rr
   print("libRoadRunner:", rr.__version__)
   PY

.. code-block:: bat

    rem On Windows (Anaconda Prompt):
    python -c "import roadrunner as rr; print('libRoadRunner:', rr.__version__)"

Installation of External Simulators
-----------------------------------

.. _bng_install:

BioNetGen
^^^^^^^^^

PyBNF works with simulators from the BioNetGen suite. We recommend BioNetGen **2.9.x** or newer.
Prebuilt distributions are available for Linux, macOS, and Windows from the `BioNetGen`_ website.

PyBNF needs the location of ``BNG2.pl``. You can either set it per-run in your ``.conf`` file
via ``bng_command`` or 
set the environment variable ``BNGPATH`` to the directory **containing** ``BNG2.pl`` (not to the script itself).

.. code-block:: bash

   # Example (macOS or Linux):
   export BNGPATH=/path/to/BioNetGen-2.9.3

   # Alternative: put BNG in PATH so BNG2.pl is directly runnable
   export PATH="$BNGPATH:$PATH"

On Windows (Anaconda Prompt):

.. code-block:: bat

   set BNGPATH=C:\BioNetGen-2.9.3
   rem (Optional) add to PATH for this session:
   set PATH=%BNGPATH%;%PATH%

If the path contains spaces, quote it: ``set BNGPATH="C:\Program Files\BioNetGen-2.9.3"``

Perl is required to run ``BNG2.pl`` (macOS/Linux users typically have it; Windows users can install Strawberry Perl).

SBML
^^^^

PyBNF runs SBML models via ``libroadrunner``. Install SBML support with:

.. code-block:: bash

   python3 -m pip install "pybnf[sbml]"

SBML support requires available wheels for your OS/Python (PyBNF supports Python 3.10–3.13). If import fails, ensure that you installed with ``[sbml]`` and that your Python version is supported.

To work with SBML files, tools such as `COPASI`_ can be useful for viewing and editing models.


.. _Anaconda: https://www.anaconda.com/download
.. _BioNetGen: https://www.bionetgen.org
.. _SBML: https://sbml.org/
.. _libRoadRunner: https://libroadrunner.org/
.. _COPASI: https://copasi.org/
.. _virtualenv: https://packaging.python.org/guides/installing-using-pip-and-virtualenv/
.. _Strawberry Perl: https://strawberryperl.com/
