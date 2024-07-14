Installing Barnacle
===================

You can install Barnacle and its dependencies by running 

.. code-block:: bash
    
    pip install barnacle

Using Barnacle usually requires interacting with additional libraries. 
We recommend using virtual environments to manage this library ecosystem. 
In particular, we used `Poetry <https://python-poetry.org/>`_ environments 
in developing the Barnacle project. If you have poetry installed, you can 
replicate the Poetry environment used for developing Barnacle by downloading 
the ``pyproject.toml`` file from the 
`Barnacle github page <https://github.com/blasks/barnacle/blob/main/pyproject.toml>`_
and then running

.. code-block:: bash
    
    poetry install

This will build a new poetry environment, and install in that environment 
Barnacle, Barnacle's dependencies, and additional libraries useful for 
utilizing Barnacle.