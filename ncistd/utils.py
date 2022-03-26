"""The utils module contains tools for visualizing and manipulating tensors"""
import numpy as np
import pandas as pd
import seaborn as sns
import tensorly as tl
from matplotlib import pyplot as plt
from plotly.express import scatter_3d
from sklearn.metrics import jaccard_score
from tensorly import check_random_state

##########
# Function to plot interactive visualization of 3d tensor
##########
def visualize_3d_tensor(tensor, 
                        shell=True, 
                        midpoint=None, 
                        range_color=None, 
                        opacity=0.5, 
                        aspectmode='data'):
    """Plot an interactive visualization of a 3d tensor using plotly
    
        This method uses the plotly.express.scatter_3d() function to plot a 
        visualization of the input data tensor. It is intended primarily for 
        use with three dimensional tensors, but can handle lower dimensional 
        data as well.
    
    Parameters
    ----------
    tensor : 3 dimensional numpy.ndarray 
        Three dimensional numpy array containing tensor data
    shell : bool
        Plot only the points on the outer sides of the tensor. Default is True.
    midpoint : float, optional
        Midpoint value of the color scale, coded white.
    range_color : 2-tuple, optional
        Range of the color scale, formatted (low, high).
    opacity : float
        Opacity of the points on the plot, ranges from 0 to 1. Default is 0.5.
    aspectmode : {'data', 'cube', 'auto'}
        Option passed to plotly to control the proportions of the axes:
            'data' : axes are in proportion to the data ranges
            'cube' : axes are drawn as a cube, regardless of data ranges
            'auto' : 'data' if no axis is > 4x any other axis, otherwise 'cube'
        Default is 'data'.
            
    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        Plotly figure object. A number of options are available to this object, 
        including show() and save().
    """
    data = dict()
    axis_columns = []
    for i, v in enumerate(np.indices(tensor.shape)):
        axis = 'axis{}'.format(i)
        axis_columns.append(axis)
        data.update({axis: v.flatten()})
    data.update({'expression': tensor.flatten()})
    data.update({'abs_exp': np.abs(tensor.flatten())})
    # make dataframe
    df = pd.DataFrame(data)
    # down-select only outer indices
    if shell:
        # mask all but the indices with a zero in them
        mask = df[axis_columns].eq(0).any(axis=1)
        for col in axis_columns:
            # add in the max index for each dimension
            mask = np.any([mask, df[col].eq(df[col].max())], axis=0)
        df = df[mask]
    # make the figure 
    fig = scatter_3d(df, x='axis0', y='axis1', z='axis2', color='expression', 
                     size='abs_exp', opacity=opacity, 
                     color_continuous_scale='RdBu_r', 
                     color_continuous_midpoint=midpoint, 
                     range_color=range_color)
    fig.update_layout(scene=dict(xaxis=dict(showbackground=False), 
                                 yaxis=dict(showbackground=False), 
                                 zaxis=dict(showbackground=False), 
                                 aspectmode=aspectmode), 
                      width=700)
    fig.update_traces(marker=dict(line=dict(color=None, width=0)))
    return fig

##########
# Function to plot heatmap of factors
##########
def plot_factors_heatmap(factors, 
                         ratios=False, 
                         mask_thold=None, 
                         reference_factors=None, 
                         figsize=None, 
                         heatmap_kwargs=None):
    """Plot a heatmap visualization of cp_tensor factors
    
    Parameters
    ----------
    factors : list of numpy.ndarray
        The factors to be plotted
    ratios : {bool, list of ints},
        True = heights of plots are proportional to dimensions of factors
        False = heights of plots are identical
        list of ints = manual assignment of height ratios
    mask_thold : tuple of floats
        Interval (inclusive) between which all values will be masked out of heatmaps. 
        Example: 
            (0, 0) = only values that are exactly zero will be masked.
    reference_factors : list of numpy.ndarray, optional
        A second set of baseline factors to be plotted. Sizes and shapes are 
        assumed to be the same as in `factors`. If not None, `reference_factors`
        will be plotted in the first column, and `factors` in the second.
    figsize : 2-tuple, optional
        Size of the figure
    heatmap_kwargs : dict, optional
        Keword arguments to be passed to each heatmap in the figure
            
    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure object
    ax : matplotlib.axes._subplots.AxesSubplot
        Matplotlib axes object
    """
    columns = 1 + (reference_factors is not None)
    rows = len(factors)
    if figsize is None:
        figsize = (5 * columns, 5 * rows)
    if type(ratios) is list:
        assert len(ratios) == rows
    elif ratios:
        ratios = [factor.shape[0] for factor in factors]
        ratios = [r / min(ratios) for r in ratios]
    else:
        ratios = np.ones(rows)
    if heatmap_kwargs is None:
        heatmap_kwargs = {}
    fig, ax = plt.subplots(
        rows, 
        columns, 
        figsize=figsize, 
        gridspec_kw={'height_ratios': ratios}
    )
    for i in range(rows):
        # get factor mask
        if mask_thold is not None:
            fact_mask = np.logical_and((factors[i] >= mask_thold[0]), 
                                        (factors[i] <= mask_thold[1]))
        else:
            fact_mask = None
        # plot reference
        if reference_factors is not None:
            # get reference mask
            if mask_thold is not None:
                ref_mask = np.logical_and((reference_factors[i] >= mask_thold[0]), 
                                          (reference_factors[i] <= mask_thold[1]))
            else:
                ref_mask = None
            sns.heatmap(
                reference_factors[i], 
                mask=ref_mask, 
                **heatmap_kwargs, 
                ax=ax[i][0]
            )
            sns.heatmap(
                factors[i], 
                mask=fact_mask, 
                **heatmap_kwargs, 
                ax=ax[i][1]
            )
        else:
            sns.heatmap(
                factors[i], 
                mask=fact_mask, 
                **heatmap_kwargs, 
                ax=ax[i]
            )
    return fig, ax

##########
# function to generate tensorly CPTensor of all zeros
##########
def zeros_cp(shape, rank):
    """Return tensorly.CPTensor of all zeros of the specified
    size and rank.
    """
    weights = tl.zeros(rank)
    factors = []
    for dim in shape:
        factors.append(tl.zeros([dim, rank]))
    return(CPTensor((weights, factors)))

##########
# function to generate random folds
##########
def random_folds(shape, folds, random_state=None):
    """Generates ``folds`` random masks in same shape as tensor. Each mask represents a nearly even
    fraction of the input tensor (remainder distributed), without overlap. 
    
    Tests: 
    # Full coverage even split
    test = np.zeros(shape)
    for i in range(folds):
        test = np.logical_or(test, ~masks[i])
    assert np.all(test)
    
    # No overlaps
    test = np.zeros(shape)
    for i in range(folds):
        test = np.logical_and(test, ~masks[i])
    assert not np.any(test)
    """
    # initialize random generator
    rns = check_random_state(random_state)
    size = np.product(shape)
    assert folds <= size, "The value of ``folds`` cannot exceed the size of the array."
    n = size // folds
    remainder = size % folds
    # delineate indices of even groups
    indices = np.cumsum([0] + [n+1 if i < remainder else n for i in range(folds)])
    mask = np.zeros(size, dtype=int)
    # mark off groups in parent mask
    for i in range(folds):
        mask[indices[i]:indices[i+1]] = i
    rns.shuffle(mask)
    # reshape parent mask
    mask = mask.reshape(shape)
    # generate mask children from parent mask
    masks = []
    for i in range(folds):
        masks.append(mask != i)
    return masks

##########
# function to generate random mask
##########
def random_mask(tensor, fraction_masked=0.0, random_state=None, dtype=bool):
    """Generate random mask in same shape as tensor. True indicates unmasked
    data point and False indicates masked data point.
    """
    # initialize random generator
    rns = check_random_state(random_state)
    mask = np.ones(tensor.size, dtype=int)
    n_masked = int(mask.size * fraction_masked)
    mask[:n_masked] = 0
    rns.shuffle(mask)
    # reshape and cast as requested data type
    mask = mask.reshape(tensor.shape).astype(dtype)
    return mask

##########
# function to permute tensor
##########
def permute_tensor(tensor, mode, random_state=None):
    rns = check_random_state(random_state)
    permuted_tensor = list()
    for fiber in tl.unfold(tensor, mode).T:
        permuted_tensor.append(rns.permutation(fiber))
    return tl.fold(np.array(permuted_tensor).T, mode, tensor.shape) 

##########
# function to generate and store empirical distribution of eigenvalues across a particular mode
##########
def get_null_loadings(tensor, permutation_mode, n_permutations, decomposition_method, decomp_rank, 
                      decomp_params=None, verbose=0, random_state=None):
    # check random state
    rns = check_random_state(random_state)
    # instantiate null data tensor
    null_weights = list()
    null_factors = list()
    # iterate through permutations
    for i in range(n_permutations):
        if verbose > 0:
            print('Beginning permutation {} of {}'.format(i+1, n_permutations))
        permuted_tensor = permute_tensor(tensor, permutation_mode, random_state=rns)
        (weights, factors) = decomposition_method(permuted_tensor, rank=decomp_rank, 
                                                  random_state=rns, verbose=verbose-1, 
                                                  **decomp_params)
        null_weights.append(weights[permutation_mode])
        null_factors.append(factors[permutation_mode])
    return np.array(null_factors).transpose(), np.array(null_weights)

##########
# function to derive clusters from null loadings
##########
def extract_significant_loadings(factor, null_loadings, quantile=0.05):
    cluster = np.less(factor, np.quantile(null_loadings, quantile, axis=1))
    cluster = cluster + np.greater(factor, np.quantile(null_loadings, 1-quantile, axis=1))
    return cluster

##########
# functions evaluate cluster recovery (Saelens et al.)
##########
def jaccard_matrix(true_clusters, inferred_clusters):
    matrix = np.ndarray((len(true_clusters), len(inferred_clusters)))
    for i, true in enumerate(true_clusters):
        for j, inferred in enumerate(inferred_clusters):
            matrix[i][j] = jaccard_score(true, inferred)
    return matrix

def recovery_relevance(true_clusters, inferred_clusters):
    j_matrix = jaccard_matrix(true_clusters, inferred_clusters)
    recovery = j_matrix.max(axis=0).mean()
    relevance = j_matrix.max(axis=1).mean()
    return recovery, relevance

def cluster_buddies_matrix(cluster_set):
    matrix = np.zeros((len(cluster_set[0]), len(cluster_set[0])))
    for cluster in cluster_set:
        matrix += np.outer(cluster, cluster)
    # fill zero values with np.nan
    matrix[np.equal(matrix, 0)] = np.nan
    return matrix

def pairs_precision_recall(true_clusters, inferred_clusters):
    true_matrix = cluster_buddies_matrix(true_clusters)
    inferred_matrix = cluster_buddies_matrix(inferred_clusters)
    min_matrix = np.min(np.stack([true_matrix, inferred_matrix]), axis=0)
    # fill nan values with zeros in min matrix
    min_matrix[np.isnan(min_matrix)] = 0.0
    # calculate precicion
    precision_values = (min_matrix / inferred_matrix)[np.tril_indices(60, k=-1)]
    precision = precision_values[~np.isnan(precision_values)].mean()
    # calculate recall
    recall_values = (min_matrix / true_matrix)[np.tril_indices(60, k=-1)]
    recall = recall_values[~np.isnan(recall_values)].mean()
    return precision, recall
