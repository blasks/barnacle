"""The simulation module contains tools for generating tensors of simulated 
cluster data based on several different models"""
import numpy as np
import itertools
import scipy.sparse as sparse
import scipy.stats as stats
from tensorly import check_random_state
from tensorly.tenalg import outer
from tensorly.cp_tensor import CPTensor


##########
# Cluster class
##########
class Cluster:
    """Object to contain indices and value of model cluster
    """
    def __init__(self, indices, value):
        """Initialize a Cluster object
        
        Parameters
        ----------
        indices : list of one-dimensional np.ndarray objects
            Indices of cluster membership in each dimension
        value : float
            Cluster value
        """
        self.indices = indices
        self.value = value
        self.dim = len(self.indices)
        
    def __repr__(self):
        message = '{} dimensional Cluster'.format(self.dim)
        return message
        
    def shape(self):
        return [len(index) for index in self.indices]

    def size(self):
        return np.product([len(index) for index in self.indices])
    
    def to_tensor(self, supertensor_shape=None):
        """Generate a tensor from the Cluster indices and value.
        
        Parameters
        ----------
        supertensor_shape : list
            The shape of the supertensor in which to embed the Cluster tensor.
            Must be greater than or equal to the dimensionality of the Cluster.
            Each supertensor dimension must be greater than or equal to the 
            maximum index value of the corresponding Cluster dimension.
        
        Returns
        -------
        data : numpy.ndarray
            Tensorized Cluster formatted in an n-dimensional numpy array.
        """
        cluster_bounds = [(np.min(idx), np.max(idx)) for idx in self.indices]
        # check supertensor_shape
        if supertensor_shape is None:
            supertensor_shape = [h - l + 1 for (l, h) in cluster_bounds]
            # set offset to minimum index of each dimension
            offset = [l for (l, _) in cluster_bounds]
        # check that supertensor_shape >= Cluster.dim
        elif len(supertensor_shape) < self.dim:
            raise ValueError('The supertensor_shape {} cannot contain '
                             '{}'.format(supertensor_shape, self))
        # check that each range fits the corresponding cluster dimension
        else:
            for i, size in enumerate(supertensor_shape):
                if size < cluster_bounds[i][1]:    # max index
                    raise ValueError('Dimension {} of size {} cannot contain '
                                     'the index of the corresponding Cluster '
                                     'dimension: {}'.format(i, 
                                                            size, 
                                                            self.indices[i]))
            # set offset to zero
            offset = [0 for i in range(self.dim)]
        # make vectors from supertensor_shape
        vectors = [np.zeros(shape) for shape in supertensor_shape]
        for i, idx in enumerate(self.indices):
            vectors[i][idx - offset[i]] = 1
        # tensorize 
        tensor = self.value * outer(vectors)
        return tensor
        
                
        
        
##########
# SimulationTensor class
##########
class SimulationTensor(CPTensor):
    """Class container for simulated data models and methods"""
    def __init__(self, 
                 cp_tensor):
        super().__init__(cp_tensor)
        
    def __repr__(self):
        message = 'Rank-{} SimulationTensor of shape {}'.format(self.rank, 
                                                                self.shape)
        return message
        
    def to_tensor(self, 
                  noise_level=0, 
                  sparse_noise=False, 
                  noise_distribution=None, 
                  random_state=None):
        """Generate optionally noisey data tensor from factorized CP tensor.
        
        Parameters
        ----------
        noise_level : float, optional
            Scale factor for the noise tensor, relative to the l2 norms.
        sparse_noise : bool
            If True, will set all positions in the noise matrix that correspond
            to sparse positions in the signal matrix to zero. Default is False.
        noise_distribution : scipy.stats.rv_continuous, optional
            Parameterized continuous distribution to generate the noise tensor.
            This parameter cannot be None if noise_level > 0.
        random_state : {None, int, numpy.random.RandomState}, optional
            Random state to seed the noise_distribution generator.
                
        Returns
        -------
        data : numpy.ndarray
            Tensorized data formatted in an n-dimensional numpy array.
        """
        # get tensorized data
        data = super().to_tensor()
        if noise_level == 0:
            return data
        else:
            if noise_distribution is None:
                noise_distribution = stats.norm()
            # initialize random generator
            rns = check_random_state(random_state)
            # add noise to data tensor
            noise = noise_distribution.rvs(size=self.shape, random_state=rns)
            if sparse_noise:
                noise = noise * (data != 0)
            noise /= np.linalg.norm(noise)
            noise *= noise_level * self.norm()
            data += noise
            return data
        
    def get_clusters(self):
        """Generate list of Cluster objects from model factors."""
        clusters = []
        for i in range(self.rank):
            indices = [np.nonzero(factor.T[i])[0] for factor in self.factors]
            value = self.weights[i]
            clusters.append(Cluster(indices, value))
        return clusters


##########
# Function to generate Overlapping Block Model
##########
def overlapping_block_model(shape, 
                            rank, 
                            value_distribution, 
                            cluster_size_bounds=None, 
                            contiguous_clusters=False, 
                            random_state=None):
    """Generate an Overlapping Block Model.
    
        Returns a SimulationTensor object parameterized with input Overlapping 
        Block Model parameters.
    
    Parameters
    ----------
    shape : tuple of ints
        Tensor shape where len(shape) = n dimensions in tensor.
    rank : int
        Tensor rank, equal to the number of clusters generated. 
    value_distribution : scipy.stats.rv_continuous
        Parameterized continuous distribution, from which the value of each 
        cluster will be drawn. This is equivalent to the weights in a 
        factorized CP tensor.  
    cluster_size_bounds : {None, list of 2-tuples of ints}
        The number of tuples must equal the number of dimensions in the `shape` 
        parameter, each having the form (`low`, `high`) such that: 
            1 <= `low` <= `high` <= `shape`[i]
        Each cluster size will then be drawn from a uniform random distribution 
        on the interval [`low`, `high`].
        Example:
            For a model of shape (3, 4, 5):
                [(1, 2), (1, 1), (4, 5)] are valid size bounds.
            This would result in four possible cluster shapes:
                (1, 1, 4), (1, 1, 5), (2, 1, 4), or (2, 1, 5)
        The default is None in which case cluster_size_bounds is set to:
            [(1, i) for i in `shape`].
    contiguous_clusters : bool
        If contiguous_clusters=True, then each cluster will be generated as a 
        contiguous subtensor. The default is False, in which case cluster 
        indices are chosen from a uniform random distribution.
    random_state : {None, int, np.random.RandomState}
        Random state to seed the value_distribution and 
        cluster_size_distribution generators.
            
    Returns
    -------
    model : SimulationTensor
        Parameterized overlapping block model.
    """
    # check random state
    rns = check_random_state(random_state)
    # set default cluster_size_bounds if needed
    if cluster_size_bounds is None:
        cluster_size_bounds = [(1, i) for i in shape]
    # generate factors
    factors = []
    # iterate through dimensions
    for i, (low, high) in enumerate(cluster_size_bounds):
        # check cluster size bounds
        if not 1 <= low <= high <= shape[i]:
            raise ValueError('The cluster_size_bounds (low, high) for each '
                             'dimension i must conform to 1 <= low <= high <= '
                             'shape[i]. However dimension {} of size {} was '
                             'assigned cluster_size_bounds '
                             '{}'.format(i, shape[i], (low, high))) 
        factor_matrix_i = np.zeros((rank, shape[i]), dtype=int)
        # iterate through each cluster and select indices
        for j in range(rank):
            if contiguous_clusters:
                # randomly choose cluster size 
                index = np.arange(rns.choice(range(low, high+1)), dtype=int)
                # choose and add starting point
                index += rns.choice(range(0, shape[i]))
                # make index wrap around if need be
                index = index % shape[i]
            else:
                # randomly choose cluster size and indices
                index = rns.choice(range(0, shape[i]), 
                                   size=rns.choice(range(low, high+1)), 
                                   replace=False)
            # set cluster indices to 1
            factor_matrix_i[j][index] = 1
        factors.append(factor_matrix_i.T)
    # generate weights
    weights = value_distribution.rvs(rank, random_state=rns)
    # build model
    model = SimulationTensor((weights, factors))
    return model

##########
# Function to generate simulated sparse tensor
##########

def simulated_sparse_tensor(shape, 
                            rank, 
                            densities=None, 
                            factor_dist_list=None,  
                            weights=None, 
                            random_state=None):
    """Generates simulated data in the form of a sparse cp_tensor
    
    Parameters
    ----------
    shape : tuple of ints
        Tensor shape where len(shape) = n dimensions in tensor.
    rank : int
        The number of components in the tensor. 
    densities : list of floats [0.0, 1.0], optional
        The proportion of elements that are non-zero in the factor matrices. 
        Must be the same length as the `shape` parameter.
        If not set, the densities are set to 1 for fully dense factor matrices.
    factor_dist_list : list of scipy.stats._distn_infrastructure.rv_frozen, optional
        Distributions from which the factor matrices will be drawn. Must be the
        same length as the `shape` parameter and must have a .rvs() method 
        for drawing random values, and a `random_state` attribute specifying state.
        Example:
            scipy.stats.uniform()
    weights : list of floats, optional
        Weights to assign to each factor. If not set, then defaults to ones.
    random_state : {None, int, np.random.RandomState}
        Random state to seed the value_distribution and 
        cluster_size_distribution generators.
            
    Returns
    -------
    model : CPTensor
        Parameterized simulated data.
    """
    rns = check_random_state(random_state)
    if densities is None:
        densities = np.ones(rank)
    if factor_dist_list is None:
        factor_dist_list = [stats.uniform() for i in range(rank)]
    if weights is None:
        weights = np.ones(rank)
    factors = []
    for i, dim in enumerate(shape):
        dist = factor_dist_list[i]
        dist.random_state = rns
        factor = sparse.random(
            dim,        
            rank,       
            density=densities[i],        
            random_state=rns, 
            data_rvs=dist.rvs
        )
        factors.append(factor.A)
    return SimulationTensor((weights, factors))

# ##########
# # Function to generate original Block Model
# ##########
# def generate_block_model():
#     model = SimulationTensor((weights, factors))
#     return model

