"""The simulation module contains tools for generating tensors of simulated 
cluster data based on several different models"""
import itertools
import numpy as np
import scipy
from tensorly.random import check_random_state
from tensorly.cp_tensor import CPTensor


##########
# Cluster class
##########
class Cluster:
    """Object to contain indices and value of model cluster
    """
    def __init__(self, indices, value):
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
                  noise_distribution=scipy.stats.norm(0, 1), 
                  random_state=None):
        """Generate optionally noisey data tensor from factorized CP tensor."""
        # get tensorized data
        data = super().to_tensor()
        if noise_level == 0:
            return data
        else:
            # initialize random generator
            rns = check_random_state(random_state)
            # add noise to data tensor
            noise = noise_distribution.rvs(size=self.shape, random_state=rns)
            noise /= np.linalg.norm(noise)
            noise *= noise_level * self.norm()
            data += noise
            return data
        
    def clusters(self):
        """Generate list of Cluster objects from model factors."""
        clusters = []
        for i in range(self.rank):
            indices = [np.nonzero(factor.T[i])[0] for factor in self.factors]
            value = self.weights[i]
            clusters.append(Cluster(indices, value))
        return clusters


# def generate_block_model():
#     model = SimulationTensor((weights, factors))
#     return model

def generate_overlapping_block_model(shape, 
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
    random_state : {None, int, np.random.RandomState()}
        Random state to seed the value_distribution and 
        cluster_size_distribution generators.
            
    Returns
    -------
    model : SimulationTensor
        Parameterized overlapping block model tensor.
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












# ##########
# # BlockModel class 
# ##########
# class BlockModel:
#     """Wang et al. 2019 block model
#     """
#     def __init__(self, shape, clusters_shape, method, noise_level):
#         self.shape = shape
#         self.clusters_shape = clusters_shape
#         self.method = method
#         self.noise_level = noise_level
#         self.marginals = None
#         self.clusters = None
#         self.data = None

#     def n_blocks(self):
#         return np.product(self.clusters_shape)

#     def validate_parameters(self):
#         # TODO
#         # Check data shape & cluster shape
#         # Check method is valid
#         return 
        
#     def generate_clusters(self, marginals_distribution: scipy.stats.rv_continuous, random_state=None) -> [Cluster]:
#         """Generate Cluster objects from parameters
        
#         Arguments
#         ---------
#         distribution: a continuous random variable distribution function from scipy.stats
#         distribution_params: a dictionary of distribution parameters to be passed onto the distribution
#         """
#         # initialize attributes
#         self.marginals = dict()
#         self.clusters = list()
#         cluster_limits = dict()
#         # initialize random generator
#         rns = check_random_state(random_state)
#         # generate marginal means and associated indices
#         for i, n in enumerate(self.clusters_shape):
#             # randomly draw cluster values
#             self.marginals[i] = marginals_distribution.rvs(n, random_state=rns)
#             # randomly select cluster limits
#             limits = rns.choice(np.arange(1, self.shape[i]), n-1, replace=False)
#             limits = np.append(limits, [0, self.shape[i]])
#             limits = np.sort(limits)
#             # store cluster limits
#             cluster_limits[i] = dict()
#             for j in range(n):
#                 cluster_limits[i][j] = (limits[j], limits[j+1])
#         # store cluster limits and values
#         for index in itertools.product(*[range(i) for i in self.clusters_shape]):
#             limits = [cluster_limits[i][j] for i, j in enumerate(index)]
#             if self.method == 'add':
#                 value = np.sum([self.marginals[i][j] for i, j in enumerate(index)])
#             elif self.method == 'mult':
#                 value = np.product([self.marginals[i][j] for i, j in enumerate(index)])
#             else:
#                 raise Exception('Block model mode not recognized')
#             self.clusters.append(Cluster(limits, value))
#         # return clusters
#         return self.clusters
    
#     def generate_data_tensor(self, marginals_distribution, noise_distribution, random_state=None): 
#         # get random state
#         rns = check_random_state(random_state)
#         # generate clusters if it hasn't been done yet
#         if self.clusters is None:
#             self.generate_clusters(marginals_distribution, random_state=rns)
#         # iteratively add the value of each cluster to the data tensor
#         self.data = np.zeros(self.shape)
#         for cluster in self.clusters:
#             indices = [range(limit[0], limit[1]) for limit in cluster.limits]
#             self.data[tuple(zip(*itertools.product(*indices)))] = cluster.value
#         # add noise to data tensor
#         noise = noise_distribution.rvs(size=self.shape, random_state=rns)
#         noise /= np.linalg.norm(noise)
#         noise *= self.noise_level * np.linalg.norm(self.data)
#         self.data += noise
#         # return data tensor
#         return self.data
    
#     def cluster_masks(self, mode):
#         masks = list()
#         for cluster in self.clusters:
#             index = np.arange(*cluster.limits[mode]) % self.shape[mode]            
#             mask = np.zeros(self.shape[mode], dtype=bool)
#             mask[index] = True
#             masks.append(mask)
#         return masks
    
#     def cluster_indices(self, mode):
#         masks = self.cluster_masks(mode)
#         indices = [np.arange(self.shape[mode])[mask] for mask in masks]
#         return indices


# ##########
# # OverlappingBlockModel class 
# ##########
# class OverlappingBlockModel:
#     """Each cluster is a unique subtensor, and they all get added together
#     """
#     def __init__(self, shape, n_clusters, cluster_size_bounds, method, noise_level):
#         self.shape = shape
#         self.n_clusters = n_clusters
#         self.cluster_size_bounds = cluster_size_bounds
#         self.method = method
#         self.noise_level = noise_level
#         self.clusters = None
#         self.data = None

#     def validate_parameters(self):
#         # TODO
#         # Check data shape & cluster shape
#         # Check mode is valid
#         return 
        
#     def generate_clusters(self, value_distribution: scipy.stats.rv_continuous, random_state=None) -> [Cluster]:
#         # initialize attributes
#         self.clusters = list()
#         # check random state
#         rns = check_random_state(random_state)
#         # generate cluster objects
#         for _ in range(self.n_clusters):
#             # generate limits
#             limits = list()
#             for j, bounds in enumerate(self.cluster_size_bounds):
#                 # randomly select cluster limits
#                 size = rns.choice(range(bounds[0], bounds[1]))
#                 start = rns.choice(range(self.shape[j]))
#                 limits.append((start, start + size))
#             # generate block value
#             value = value_distribution.rvs(random_state=rns)
#             # store cluster limits and values
#             self.clusters.append(Cluster(limits, value))
    
#     def generate_data_tensor(self, value_distribution, noise_distribution, random_state=None): 
#         # check random state
#         rns = check_random_state(random_state)
#         # generate clusters if it hasn't been done yet
#         if self.clusters is None:
#             self.generate_clusters(value_distribution, random_state=rns)
#         # initialize data tensor
#         if self.method == 'add':
#             self.data = np.zeros(self.shape)
#         elif self.method == 'mult':
#             self.data = np.ones(self.shape)
#         else: 
#             raise Exception('Block model mode not recognized')
#         for cluster in self.clusters:
#             # if limits extend beyond size of dimension, wrap index around back to 0
#             indices = [np.arange(limit[0], limit[1]) % self.shape[i] for i, limit in enumerate(cluster.limits)]
#             if self.method == 'add':
#                 self.data[tuple(zip(*itertools.product(*indices)))] += cluster.value
#             elif self.method == 'mult':
#                 self.data[tuple(zip(*itertools.product(*indices)))] *= cluster.value
#         # add noise to data tensor
#         noise = noise_distribution.rvs(size=self.shape, random_state=rns)
#         noise /= np.linalg.norm(noise)
#         noise *= self.noise_level * np.linalg.norm(self.data)
#         self.data += noise
#         # return data tensor
#         return self.data
    
#     def cluster_masks(self, mode):
#         masks = list()
#         for cluster in self.clusters:
#             index = np.arange(*cluster.limits[mode]) % self.shape[mode]            
#             mask = np.zeros(self.shape[mode], dtype=bool)
#             mask[index] = True
#             masks.append(mask)
#         return masks
    
#     def cluster_indices(self, mode):
#         masks = self.cluster_masks(mode)
#         indices = [np.arange(self.shape[mode])[mask] for mask in masks]
#         return indices
