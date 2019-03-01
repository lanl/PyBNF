.. PyBNF documentation master file, created by
   sphinx-quickstart on Thu Apr 19 09:26:34 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyBioNetFit
=================================

PyBNF is a general-purpose program for parameterizing biological models specified using the BioNetGen rule-based
modeling language (`BNGL`_) or the Systems Biology Markup Language (`SBML`_). In addition to model parameterization, 
PyBNF supports uncertainty quantification by bootstrapping or Bayesian approaches, and model checking. PyBNF includes
the Biological Property Specification Language (BPSL) for defining qualitative data for use in parameterization or
checking. It runs on most Linux and macOS
workstations as well on computing clusters. 

To get started using PyBNF, follow the :ref:`installation <installation>` instructions, then look at the examples in the :ref:`Quick Start <quickstart>`. 


.. _BNGL: http://www.bionetgen.org
.. _SBML: http://sbml.org/Main_Page

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   config
   config_keys
   algorithms
   advanced
   algorithm_development
   cluster
   examples
   troubleshooting
   
PyBNF Module Reference
----------------------

Detailed documentation of the PyBNF code base for advanced users

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   modules/index.rst

