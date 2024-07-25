# Barnacle

Barnacle is a Python library that implements a sparse tensor decomposition model. It was initially developed for with metatranscriptomic data in mind, but it could feasibly be applied to any multi-way dataset. To learn more about sparse tensor decomposition and its applications, please see the [documentation website](https://barnacle-py.readthedocs.io). 

## Installation 

You can install Barnacle and its dependencies by running 

```
pip install barnacle
```

Using Barnacle usually requires interacting with additional libraries. We recommend using virtual environments to manage this library ecosystem. In particular, we used [Poetry](https://python-poetry.org/) environments while developing Barnacle. You can replicate the Barnacle development using the [`pyproject.toml`](https://github.com/blasks/barnacle/blob/main/pyproject.toml) file published in this repository. If you have Poetry downloaded, running

```
poetry install
```

in the same directory as the `pyproject.toml` file should set up your environment and install dependencies. For more detailed information, refer to the [Poetry documentation for installing dependencies](https://python-poetry.org/docs/basic-usage/#installing-dependencies). 

## Documentation

Details on Barnacle usage can be found on the associated [documentation website](https://barnacle-py.readthedocs.io). The documentation includes:

- [An overview of tensors and tensor decomposition.](https://barnacle-py.readthedocs.io/en/latest/model.html)
- [A gallery of examples that demonstrate tensor analysis with Barnacle.](https://barnacle-py.readthedocs.io/en/latest/examples.html)
    - The example jupyer notebooks can be found [here](https://github.com/blasks/barnacle/tree/main/docs/notebooks)
- [An API reference of Barnacle modules.](https://barnacle-py.readthedocs.io/en/latest/autodoc/barnacle.html)

For a more technical discussion of the sparse tensor decomposition model implemented in Barnacle, please see the Methods section of the research article in which we introduce Barnacle:

_Simultaneous acclimation to nitrogen and iron scarcity in open ocean cyanobacteria revealed by sparse tensor decomposition of metatranscriptomes_
- Blaskowski, S., Roald, M., Berube, P. M., Braakman, R., & Armbrust, E. V.
- bioRxiv (pre-print), 2024
- https://doi.org/10.1101/2024.07.15.603627

## Usage

In addition to the [example gallery](https://barnacle-py.readthedocs.io/en/latest/examples.html), our research article details using Barnacle to analyze metatranscriptomes of cyanobacterial gene expression in the open ocean. All of the scripts used to conduct those analyses can be found in a separate [manuscript repository](https://github.com/blasks/barnacle-manuscript) published alongside the article.

