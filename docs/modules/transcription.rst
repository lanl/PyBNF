.. _transcription_module:

===============================================================
Constrained transcription (:py:mod:`pybnf.transcription`)
===============================================================

A *transcription* restates the fit PyBNF was asked to run as a larger problem with
internal auxiliary variables and equality constraints that tie them back together. At
the solution the two problems coincide; on the way there the enlarged one can be far
better conditioned. Multiple shooting is the first consumer (issue #563): split an
experiment at knots, introduce segment-start states ``z_j``, simulate each segment
independently, and impose continuity ``c_j = Phi_j(z_j, theta) - z_{j+1} = 0``. Direct
collocation, latent-state estimation, and path constraints are the same shape.

The :py:mod:`pybnf.transcription` package is that shape and nothing else. It holds no
dynamics, makes no simulator call, defines no configuration key and no ``job_type``; its
only dependency inside PyBNF is :py:mod:`pybnf.printing`, for the exception base class.
A consumer supplies two methods — score-and-differentiate the objective at an augmented
point, and linearise the equality constraints there — plus a certification hook, and gets
the augmented-Lagrangian outer loop, the penalty schedule, the transcription homotopy,
best-iterate certification, and the reporting.

The design and the measurements behind its defaults are recorded in ADR-0109.

Errors
======

.. automodule:: pybnf.transcription.errors
   :members:

Augmented variable layout
=========================

.. automodule:: pybnf.transcription.layout
   :members:

Equality residuals and Jacobians
================================

.. automodule:: pybnf.transcription.equality
   :members:

The augmented Lagrangian at a point
===================================

.. automodule:: pybnf.transcription.augmented
   :members:

The outer loop
==============

.. automodule:: pybnf.transcription.outer
   :members:

The transcription homotopy
==========================

.. automodule:: pybnf.transcription.homotopy
   :members:
