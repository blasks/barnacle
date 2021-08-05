"""The utils module contains tools for visualizing and manipulating tensors"""
import numpy as np
import pandas as pd
import tensorly as tl
from plotly.express import scatter_3d
from sklearn.metrics import jaccard_score
from tensorly.random import check_random_state

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
