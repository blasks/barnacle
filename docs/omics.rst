Omics analysis
==============

Barnacle was initially developed for unsupervised analysis of 
environmental metatranscriptomics. What does that mean?

Omics data
----------

Metatranscriptomics is a category of "omics" analysis
that catalogs and quantifies all of the RNA molecules (transcripts) 
in a sample. The "meta" part of the term indicates that the sample contains 
multiple taxa. As such, metatranscriptomes measures the gene expression of a 
community of organisms, such as a microbiome. 

Other types of omics data focus on different biological molecules either for 
a microbiome (meta-omics) or a single organism:

- Genomics --> DNA
- Proteomics --> proteins
- Lipidomis --> lipids 
- Metabolomics --> metabolites

Omics datasets are often characterized by certain properties that complicate 
their analysis, including high dimensionality, and technical noise. Metaomics 
are further complicated by properties such as overdispersion, variable 
community composition, and pervasive zero values :cite:p:`zhang2021statistical`. 
Dealing with this complexity in order to understand insights contained in 
omics datasets requires specially equipped analytical tools.

Tensor decomposition for omics analysis
---------------------------------------

Tensor decomposition tools have previously been applied to omics datasets in 
order to deal with some of the challenges mentioned above, and reveal patterns 
structuring the data. For example, a non-negative tensor decomposition method 
was applied to human gene expression dataset to reveal patterns associated 
with particular diseases across tissue types and demographics :cite:p:`wang2019`. 

One of the main advantages of using tensor decomposition models to analyze omics 
data is that it is an unsupervised technique. This allows researchers to 
discover patterns structuring the data, independent of their own pre-conceived 
notions. The unsupervised nature of the analysis also allows un-characterized 
genes to be analyzed alongside annotated genes, whereas other analyses tend to 
throw out these data. In metaomics this functionality is especially important 
because in many datasets over half of the genes have never been previously 
observed, much less functionally characterized 
:cite:p:`pavlopoulos2023unraveling`. Tensor decomposition can help generate 
inferences about this "functional dark matter" based on the association of 
uncharacterized genes with better characterized co-variates. 

The sparse tensor decomposition model presented in Barnacle demonstrates that 
sparsity constraints may be a useful addition to tensor decomposition models 
applied to omics datasets. In the case of transcriptomics, the sparse 
components output by the model can be interpreted as clusters of co-expressed 
genes, which may be functionally related. Similarly, in other omics datasets 
Barnacle clusters could help identify groups of other biological molecules with 
common abundance profiles across samples and conditions. 

Example usage
-------------

For an in-depth example of using Barnacle for analyzing metatranscriptomics 
data, please see our research paper *Simultaneous acclimation to nitrogen 
and iron scarcity in open ocean cyanobacteria revealed by sparse tensor 
decomposition of metatranscriptomes*. All of the scripts used to generate the 
analyses in the paper are available in the 
`manuscript repository <https://github.com/blasks/barnacle-manuscript>`_. In 
particular, check out the following scripts and notebooks:

- `Normalization and tensorization <https://github.com/blasks/barnacle-manuscript/blob/main/analyses/3-normalization/0-normalization-sctransform.ipynb>`_
  - Uses `sctransform <https://satijalab.org/seurat/articles/sctransform_vignette>`_ :cite:p:`hafemeister2019normalization` for normalization
  - Uses `xarray <https://docs.xarray.dev/en/stable/index.html>`_ :cite:p:`hoyer2017xarray` for organizing data into tensors
- `Parameter grid search <https://github.com/blasks/barnacle-manuscript/blob/main/analyses/4-fitting/grid-search.py>`_
  - Performs cross-validated grid search with bootstrapping
- `Compile bootstraps <https://github.com/blasks/barnacle-manuscript/blob/main/analyses/5-models/0-compile-bootstraps.ipynb>`_
  - Compiles component bootstraps for models with best fit parameters
- `Component profiles <https://github.com/blasks/barnacle-manuscript/blob/main/analyses/6-clusters/0-component-profiles.ipynb>`_
  - Generates summary statistics and profile visualizations for each model component
