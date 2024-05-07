import numpy as np
import pandas as pd
import seaborn as sns
import tensorly as tl
from matplotlib import pyplot as plt
from plotly.express import scatter_3d
from sklearn.metrics import jaccard_score
from tensorly import check_random_state
from tensorly.cp_tensor import CPTensor


def visualize_3d_tensor(
    tensor, 
    shell=True, 
    midpoint=None, 
    range_color=None, 
    opacity=0.5, 
    bg_color='#fff', 
    aspectmode='data',
    show_colorbar=True, 
    label_axes=True, 
    axes_names=None,
    figure_kwargs=None
):
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
    bg_color : str
        Color of the plot background. Set to 'rgba(0,0,0,0)' for transparent 
        background. Default is '#fff'.
    aspectmode : {'data', 'cube', 'auto'}
        Option passed to plotly to control the proportions of the axes:
            'data' : axes are in proportion to the data ranges
            'cube' : axes are drawn as a cube, regardless of data ranges
            'auto' : 'data' if no axis is > 4x any other axis, otherwise 'cube'
            
        Default is 'data'.
    label_axes : bool
        Plot axes label names and scales. Defalut is True.
    show_colorbar : bool
        Plot legend. Defalut is True
    axes_names : list, optional 
        Names of axes. Length must equal the number of modes in the tensor.
        When set to None, default names in the form of 'axisX' are used.
        Default is None.
    figure_kwargs : dict, optional
        Keyword arguments to be passed to the plotly.express.scatter_3d() 
        function. Default is None.
            
    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        Plotly figure object. A number of options are available to this object, 
        including show() and save().
    """
    data = dict()
    n_modes = len(tensor.shape)
    if axes_names is not None:
        if len(axes_names) != n_modes:
            raise AssertionError(('Length of `axes_names` does not match modes in tensor'))
        axis_columns = axes_names
    else:
        axis_columns = []
    for i, v in enumerate(np.indices(tensor.shape)):
        if axes_names is None:
            axis = 'axis{}'.format(i)
            axis_columns.append(axis)
        else:
            axis = axis_columns[i]
        data.update({axis: v.flatten()})
    data.update({'abundance': tensor.flatten()})
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
    # make keyword arguments
    kwargs = dict(
        x=axis_columns[0], 
        y=axis_columns[1], 
        z=axis_columns[2], 
        color='abundance', 
        size='abs_exp', 
        opacity=opacity, 
        color_continuous_scale='RdBu_r', 
        color_continuous_midpoint=midpoint, 
        range_color=range_color
    )
    # update with figure_kwargs
    if figure_kwargs is not None:  
        for k, v in figure_kwargs.items():
            kwargs[k] = v
    # make the figure 
    fig = scatter_3d(df, **kwargs)
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                showbackground=False, 
                visible=label_axes
            ), 
            yaxis=dict(
                showbackground=False, 
                visible=label_axes
            ), 
            zaxis=dict(
                showbackground=False, 
                visible=label_axes
            ), 
            aspectmode=aspectmode
        ), 
        width=700, 
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color
    )
    fig.update_coloraxes(
        showscale=show_colorbar
    )
    fig.update_traces(
        marker=dict(
            line=dict(
                color=None, 
                width=0
            )
        )
    )
    return fig


def plot_factors_heatmap(
    factors, 
    ratios=False, 
    mask_thold=None, 
    reference_factors=None, 
    figsize=None, 
    heatmap_kwargs=None
):
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
        Example: (0, 0) = only values that are exactly zero will be masked.
    reference_factors : list of numpy.ndarray, optional
        A second set of baseline factors to be plotted. Sizes and shapes are 
        assumed to be the same as in `factors`. If not None, `reference_factors`
        will be plotted in the first column, and `factors` in the second.
    figsize : 2-tuple, optional
        Size of the figure
    heatmap_kwargs : dict, optional
        Keyword arguments to be passed to each heatmap in the figure
            
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


def consolidate_cp(cp_tensor):
    """Removes zeroed out factors from cp tensor.
    
    Parameters
    ----------
    cp_tensor : tensorly.cp_tensor.CPTensor
        CP tensor decomposition.
    
    Returns
    -------
    cleaned_cp : tensorly.cp_tensor.CPTensor
        CP tensor decomposition with zero factors removed. Rank of cleaned_cp
        is less than or equal to rank of input cp_tensor.
    """
    indexer = np.array(cp_tensor.weights != 0)
    for factor in cp_tensor.factors:
        indexer *= (np.linalg.norm(factor, axis=0) != 0)
    cleaned_factors = []
    for factor in cp_tensor.factors:
        cleaned_factors.append(factor.T[indexer].T)
    cleaned_weights = cp_tensor.weights[indexer]
    cleaned_cp = CPTensor((cleaned_weights, cleaned_factors))
    return cleaned_cp


def zeros_cp(shape, rank):
    """Return a tensorly.cp_tensor.CPTensor of the specified size and rank, 
    consisting of all zeros.
    
    Parameters
    ----------
    shape : tuple of ints
        Tensor shape where len(shape) = n modes in tensor.
    rank : int
        The number of components in the tensor. 
    
    Returns
    -------
    zero_cp : tensorly.cp_tensor.CPTensor
        CPTensor of all zeros.
    """
    weights = tl.zeros(rank)
    factors = []
    for dim in shape:
        factors.append(tl.zeros([dim, rank]))
    zero_cp = CPTensor((weights, factors))
    return zero_cp


def subset_cp_tensor(cp_tensor, subset_indices):
    """Selects subset of cp_tensor based on provided indices
    
    Parameters
    ----------
    cp_tensor : tensorly.CPTensor
        CPTensor object with (weights, factors).
    subset_indices : dict(int: index-like)
        Dictionary with mode as key and value an integer index of 
        the positions to be downselected from `cp_tensor`.
        Example: {1: [0, 1, 3, 4, 5, 8]}
        
    Returns
    -------
    subset_cp : tensorly.CPTensor
        Subset CPTensor.
    """
    weights, factors = cp_tensor
    new_factors = factors.copy()
    for mode, index in subset_indices.items():
        new_factors[mode] = factors[mode][index]
    return(CPTensor((weights, new_factors)))


def permute_tensor(tensor, mode, random_state=None):
    """Function to independently permute each of the fibers of an input tensor
    along a specified mode.
    
    Parameters
    ----------
    tensor : numpy.ndarray
        Input tensor.
    mode : int
        Mode along which fibers will be permuted.
    random_state : {None, int, numpy.random.RandomState}, default is None
        Random state used to randomly permute tensor fibers.
    
    Returns
    -------
    tensor_out : numpy.ndarray
        Randomly permuted tensor.
    """
    rns = check_random_state(random_state)
    permuted_tensor = list()
    for fiber in tl.unfold(tensor, mode).T:
        permuted_tensor.append(rns.permutation(fiber))
    tensor_out = tl.fold(np.array(permuted_tensor).T, mode, tensor.shape)
    return tensor_out


def jaccard_matrix(true_clusters, inferred_clusters):
    """Generates an n x m matrix where n is the number of true_clusters and m 
    is the number of inferred_clusters, and where each entry matrix[i][j] is
    the jaccard score comparing true_clusters[i] with inferred_clusters[j]. 
    
    Note that the length of the boolean array indicating cluster membership 
    should be the same for each cluster.
    
    Parameters
    ----------
    true_clusters : list of boolean numpy.ndarrays
        List of boolean arrays indicating cluster membership for each of the
        ground truth clusters. 
    inferred_clusters : list of boolean numpy.ndarrays
        List of boolean arrays indicating cluster membership for each of the
        clusters to be compared against ground truth.
    
    Returns
    -------
    matrix : numpy.ndarray
        An n x m matrix where n is the number of true_clusters and m is the
        number of inferred_clusters, and where each entry matrix[i][j] is
        the jaccard score comparing true_clusters[i] with inferred_clusters[j].
    """
    matrix = np.ndarray((len(true_clusters), len(inferred_clusters)))
    for i, true in enumerate(true_clusters):
        for j, inferred in enumerate(inferred_clusters):
            matrix[i][j] = jaccard_score(true, inferred)
    return matrix


def recovery_relevance(true_clusters, inferred_clusters):
    """Evaluates two metrics, recovery and relevance, designed to compare a 
    set of `inferred_clusters` against a ground truth set of `true_clusters`. 
    Recovery measures how well the true clusters are recovered by the inferred 
    clusters, and is the mean of the Jaccard indices of the best match from the
    `inferred_clusters` for each of the `true_clusters`. Relevance measures 
    how well the inferred clusters are representative of the true clusters, and
    is the mean of the Jaccard indices of the best match from the 
    `true_clusters` for each of the `inferred_clusters`. Both metrics range 
    between 0 and 1. 
    
    For a complete description, see Methods section and Supplementary Figure 1
    of Saelens et al. (2018).
    
    Parameters
    ----------
    true_clusters : list of boolean numpy.ndarrays
        List of boolean arrays indicating cluster membership for each of the
        ground truth clusters. Each array should be the same length.
    inferred_clusters : list of boolean numpy.ndarrays
        List of boolean arrays indicating cluster membership for each of the
        clusters to be compared against ground truth. Each array should be the 
        same length.
    
    Returns
    -------
    recovery : float
        Recovery score (ranges between 0 and 1).
    relevance : float
        Relevance score (ranges between 0 and 1).
    """
    j_matrix = jaccard_matrix(true_clusters, inferred_clusters)
    recovery = j_matrix.max(axis=0).mean()
    relevance = j_matrix.max(axis=1).mean()
    return recovery, relevance


def cluster_buddies_matrix(cluster_set):
    """Generates a square matrix of size n x n where n is the length of the 
    boolean array indicating membership in each cluster. Note that this length
    should be the same for each cluster included in cluster_set. Each entry
    matrix[i][j] is the number of times that element i and element j co-occur
    in the same cluster (i.e. the number of times index i and index j are both
    True in the same cluster).
    
    Parameters
    ----------
    cluster_set : list of boolean numpy.ndarrays
        List of boolean arrays indicating cluster membership for each cluster
        included in the set. 
    
    Returns
    -------
    matrix : numpy.ndarray
        An n x n array of integers in which each entry matrix[i][j] is the
        number of times element i and element j co-occur in a cluster included
        in the `cluster_set`.
    """
    matrix = np.zeros((len(cluster_set[0]), len(cluster_set[0])))
    for cluster in cluster_set:
        matrix += np.outer(cluster, cluster)
    # fill zero values with np.nan
    matrix[np.equal(matrix, 0)] = np.nan
    return matrix


def pairs_precision_recall(true_clusters, inferred_clusters):
    """Evaluates two metrics, precision and recall, designed to compare pairwise
    membership of the elements included in a set of `inferred_clusters` against 
    the pairwise membership of elements in a ground truth set of `true_clusters`. 
    To evaluate, the number of times each pair of elements (e.g. cluster[0] and 
    cluster[1]) co-occurs is counted in both the set of `true_clusters` as well
    as the set of `inferred_clusters`. Precision measures the proportion of 
    pairwise co-occurrences in the `inferred_clusters` that are also found in 
    the `true_clusters`. Recall measures the proportion of pairwise 
    co-occurences in the `true_clusters` that are reproduced in the 
    `inferred_clusters`. Both metrics range between 0 and 1.
    
    For a complete description, see Methods section and Supplementary Figure 1
    of Saelens et al. (2018).
    
    Parameters
    ----------
    true_clusters : list of boolean numpy.ndarrays
        List of boolean arrays indicating cluster membership for each of the
        ground truth clusters. Each array should be the same length.
    inferred_clusters : list of boolean numpy.ndarrays
        List of boolean arrays indicating cluster membership for each of the
        clusters to be compared against ground truth. Each array should be the 
        same length.
    
    Returns
    -------
    precision : float
        Precision score (ranges between 0 and 1).
    recall : float
        Recall score (ranges between 0 and 1).
    """
    true_matrix = cluster_buddies_matrix(true_clusters)
    inferred_matrix = cluster_buddies_matrix(inferred_clusters)
    min_matrix = np.min(np.stack([true_matrix, inferred_matrix]), axis=0)
    # fill nan values with zeros in min matrix
    min_matrix[np.isnan(min_matrix)] = 0.0
    # get index of lower triangle of matrices
    index = np.tril_indices(len(min_matrix), k=-1)
    # calculate precicion
    precision_values = (min_matrix / inferred_matrix)[index]
    if np.all(np.isnan(precision_values)):
        precision = 0.0
    else:
        precision = precision_values[~np.isnan(precision_values)].mean()
    # calculate recall
    recall_values = (min_matrix / true_matrix)[index]
    if np.all(np.isnan(recall_values)):
        recall = 0.0
    else:
        recall = recall_values[~np.isnan(recall_values)].mean()
    return precision, recall
