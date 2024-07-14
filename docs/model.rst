Sparse tensor decomposition
===========================

Barnacle was developed to implement a sparse tensor decomposition 
model. Let's unpack that.

What is a tensor?
-----------------

Tensors are multidimensional arrays. Put another way, a tensor is a
generalization of an array, in that a 0th order tensor is a scalar, a 1st
order tensor a vector, a 2nd order tensor a matrix, and 3rd order tensors 
and up are higher dimensional arrays. Practically, tensors are a good way of 
organizing datasets that may be characterized by multilinear relationships, 
such as the expression of related genes in different organisms found in 
different environmental conditions. 

What is tensor decomposition?
-----------------------------

.. figure:: figures/decomposition-diagram.svg
   :alt: Diagram of a tensor decomposition model
   :width: 90 %

   Diagram of a tensor decomposition model

What about the "sparse" part?
-----------------------------

Got it. What's it good for?
---------------------------

Barnacle is primarily intended as a tool for unsupervised signal discovery. 
In other words, it can help you figure out the major patterns driving the 
variation in a large multi-way dataset. The sparsity constraint also enables 
clusters to be derived from components by identifying in each component 
the indices associated with non-zero weights.

Barnacle was developed with metatranscriptomic datasets in mind. However, it 
could feasibly be applied to any multi-way dataset, and would be especially
well suited for other omics datasets. For more on this please see the 
:doc:`omics<omics>` section.

Tensor decomposition models have additionally been used to analyze everything
from MRI data to fish populations. How could tensor decomposition help you 
better understand your data?

Can you get a little more technical?
------------------------------------

Yes. Our sparse tensor decomposition model aims to solve the following
objective function:


