"""The simulation module contains tools for generating tensors of simulated 
cluster data based on several different models"""
import numpy as np
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
                  noise_distribution=None, 
                  random_state=None):
        """Generate optionally noisey data tensor from factorized CP tensor.
        
        Parameters
        ----------
        noise_level : float, optional
            Scale factor for the noise tensor, relative to the l2 norms.
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
        elif noise_distribution is None:
            raise ValueError('For noise_level > 0, you must pass a ' 
                             'parameterized scipy.stats.rv_continuous '
                             'distribution from which the noise tensor '
                             'will be drawn.')
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


##########
# Function to generate Overlapping Block Model
##########
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


# ##########
# # Function to generate original Block Model
# ##########
# def generate_block_model():
#     model = SimulationTensor((weights, factors))
#     return model

