.. _tutorial:

Tutorial
========

PyBNF ships a hands-on tutorial that tours its modern (**edition-2**) features
on small ODE models with known closed-form solutions. Every lesson fits
synthetic data generated from the model at known-true parameters, so a correct
fit recovers the truth — which makes each lesson both a teaching example and an
automated regression test.

Each lesson is a self-contained folder: a commented model, its data, one or more
heavily-commented fits, and a short walkthrough. Browse the full lesson index and
the per-lesson walkthroughs on GitHub:

  `PyBNF tutorial on GitHub <https://github.com/lanl/PyBNF/tree/main/examples/tutorial>`__

Run any lesson from its own folder, for example::

  cd examples/tutorial/01_logistic_growth
  pybnf -c logistic_growth_trf.conf

Results land in an ``output/`` directory inside the lesson folder.

.. note::

   A curated, grouped index of every lesson will be added to this page. For now,
   the tutorial README linked above is the authoritative lesson map.
