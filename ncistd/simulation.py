"""The simulation module contains tools for generating tensors of simulated cluster data based on several different models"""
import itertools
import numpy as np
import scipy
from tensorly.random import check_random_state

##########
# Cluster class
##########
class Cluster:
    """Object to contain indices and value of model cluster
    """
    def __init__(self, limits, value):
        self.limits = limits
        self.value = value
        
    def shape(self):
        return [limit[1] - limit[0] for limit in self.limits]

    def size(self):
        return np.product([limit[1] - limit[0] for limit in self.limits])


##########
# BlockModel class 
##########
class BlockModel:
    """Wang et al. 2019 block model
    """
    def __init__(self, shape, clusters_shape, method, noise_level):
        self.shape = shape
        self.clusters_shape = clusters_shape
        self.method = method
        self.noise_level = noise_level
        self.marginals = None
        self.clusters = None
        self.data = None

    def n_blocks(self):
        return np.product(self.clusters_shape)

    def validate_parameters(self):
        # TODO
        # Check data shape & cluster shape
        # Check method is valid
        return 
        
    def generate_clusters(self, marginals_distribution: scipy.stats.rv_continuous, random_state=None) -> [Cluster]:
        """Generate Cluster objects from parameters
        
        Arguments
        ---------
        distribution: a continuous random variable distribution function from scipy.stats
        distribution_params: a dictionary of distribution parameters to be passed onto the distribution
        """
        # initialize attributes
        self.marginals = dict()
        self.clusters = list()
        cluster_limits = dict()
        # initialize random generator
        rns = check_random_state(random_state)
        # generate marginal means and associated indices
        for i, n in enumerate(self.clusters_shape):
            # randomly draw cluster values
            self.marginals[i] = marginals_distribution.rvs(n, random_state=rns)
            # randomly select cluster limits
            limits = rns.choice(np.arange(1, self.shape[i]), n-1, replace=False)
            limits = np.append(limits, [0, self.shape[i]])
            limits = np.sort(limits)
            # store cluster limits
            cluster_limits[i] = dict()
            for j in range(n):
                cluster_limits[i][j] = (limits[j], limits[j+1])
        # store cluster limits and values
        for index in itertools.product(*[range(i) for i in self.clusters_shape]):
            limits = [cluster_limits[i][j] for i, j in enumerate(index)]
            if self.method == 'add':
                value = np.sum([self.marginals[i][j] for i, j in enumerate(index)])
            elif self.method == 'mult':
                value = np.product([self.marginals[i][j] for i, j in enumerate(index)])
            else:
                raise Exception('Block model mode not recognized')
            self.clusters.append(Cluster(limits, value))
        # return clusters
        return self.clusters
    
    def generate_data_tensor(self, marginals_distribution, noise_distribution, random_state=None): 
        # get random state
        rns = check_random_state(random_state)
        # generate clusters if it hasn't been done yet
        if self.clusters is None:
            self.generate_clusters(marginals_distribution, random_state=rns)
        # iteratively add the value of each cluster to the data tensor
        self.data = np.zeros(self.shape)
        for cluster in self.clusters:
            indices = [range(limit[0], limit[1]) for limit in cluster.limits]
            self.data[tuple(zip(*itertools.product(*indices)))] = cluster.value
        # add noise to data tensor
        noise = noise_distribution.rvs(size=self.shape, random_state=rns)
        noise /= np.linalg.norm(noise)
        noise *= self.noise_level * np.linalg.norm(self.data)
        self.data += noise
        # return data tensor
        return self.data
    
    def cluster_masks(self, mode):
        masks = list()
        for cluster in self.clusters:
            index = np.arange(*cluster.limits[mode]) % self.shape[mode]            
            mask = np.zeros(self.shape[mode], dtype=bool)
            mask[index] = True
            masks.append(mask)
        return masks
    
    def cluster_indices(self, mode):
        masks = self.cluster_masks(mode)
        indices = [np.arange(self.shape[mode])[mask] for mask in masks]
        return indices


##########
# OverlappingBlockModel class 
##########
class OverlappingBlockModel:
    """Each cluster is a unique subtensor, and they all get added together
    """
    def __init__(self, shape, n_clusters, cluster_size_bounds, method, noise_level):
        self.shape = shape
        self.n_clusters = n_clusters
        self.cluster_size_bounds = cluster_size_bounds
        self.method = method
        self.noise_level = noise_level
        self.clusters = None
        self.data = None

    def validate_parameters(self):
        # TODO
        # Check data shape & cluster shape
        # Check mode is valid
        return 
        
    def generate_clusters(self, value_distribution: scipy.stats.rv_continuous, random_state=None) -> [Cluster]:
        # initialize attributes
        self.clusters = list()
        # check random state
        rns = check_random_state(random_state)
        # generate cluster objects
        for _ in range(self.n_clusters):
            # generate limits
            limits = list()
            for j, bounds in enumerate(self.cluster_size_bounds):
                # randomly select cluster limits
                size = rns.choice(range(bounds[0], bounds[1]))
                start = rns.choice(range(self.shape[j]))
                limits.append((start, start + size))
            # generate block value
            value = value_distribution.rvs(random_state=rns)
            # store cluster limits and values
            self.clusters.append(Cluster(limits, value))
    
    def generate_data_tensor(self, value_distribution, noise_distribution, random_state=None): 
        # check random state
        rns = check_random_state(random_state)
        # generate clusters if it hasn't been done yet
        if self.clusters is None:
            self.generate_clusters(value_distribution, random_state=rns)
        # initialize data tensor
        if self.method == 'add':
            self.data = np.zeros(self.shape)
        elif self.method == 'mult':
            self.data = np.ones(self.shape)
        else: 
            raise Exception('Block model mode not recognized')
        for cluster in self.clusters:
            # if limits extend beyond size of dimension, wrap index around back to 0
            indices = [np.arange(limit[0], limit[1]) % self.shape[i] for i, limit in enumerate(cluster.limits)]
            if self.method == 'add':
                self.data[tuple(zip(*itertools.product(*indices)))] += cluster.value
            elif self.method == 'mult':
                self.data[tuple(zip(*itertools.product(*indices)))] *= cluster.value
        # add noise to data tensor
        noise = noise_distribution.rvs(size=self.shape, random_state=rns)
        noise /= np.linalg.norm(noise)
        noise *= self.noise_level * np.linalg.norm(self.data)
        self.data += noise
        # return data tensor
        return self.data
    
    def cluster_masks(self, mode):
        masks = list()
        for cluster in self.clusters:
            index = np.arange(*cluster.limits[mode]) % self.shape[mode]            
            mask = np.zeros(self.shape[mode], dtype=bool)
            mask[index] = True
            masks.append(mask)
        return masks
    
    def cluster_indices(self, mode):
        masks = self.cluster_masks(mode)
        indices = [np.arange(self.shape[mode])[mask] for mask in masks]
        return indices
