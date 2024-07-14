.. Barnacle documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Barnacle's documentation!
====================================

Barnacle is a Python package for fitting 
:doc:`sparse tensor decomposition models<model>`.
These models are useful tools for analyzing multi-way datasets, 
in which a response variable has been measured
in regards to three or more modes of variation. Barnacle was 
originally developed for the purpose of analysing metatranscriptomic datasets, 
but could feasibly be applied to any multi-way dataset. To jump right
into using Barnacle, see the :doc:`installation<installation>` instructions and 
peruse the :doc:`example gallery<examples>`. Go to the 
:doc:`model description page<model>` to learn more about Barnacle's 
implementation of a sparse tensor decomposition model, as well as more 
information information about tensor decomposition models in general. To learn 
more about applying Barnacle to metatranscriptomes and other omics datatest, 
navigate to the :doc:`omics<omics>` page.

This project is dependent on, and we are grateful for the work of many open 
source projects. In particular, we rely on the Python package
`Tensorly <https://tensorly.org>`_ :cite:p:`kossaifi2019tensorly` for most  
foundational tensor manipulations. The `Tensorly <https://tensorly.org>`_ website
is an excellent resource for tools and information on working with tensors, 
including tensor decompositions. 

.. toctree::
   :maxdepth: 2

   installation
   model
   omics
   examples
   autodoc/barnacle
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
