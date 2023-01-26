import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats
from tensorly import check_random_state
from tensorly.cp_tensor import CPTensor

        
class Component(CPTensor):
    """Rank-1 CPTensor that constitutes one component of the CPTensor from
    which it was derived.
    """
    def __init__(self, component):
        if component.rank != 1:
            raise ValueError('Component object must be a rank-1 CPTensor')
        super().__init__(component)
        
    def __repr__(self):
        message = 'CPTensor component of shape {}'.format(self.shape)
        return message
    
    def indices(self, modes=None, tholds=(0, 0)):
        """Method that returns the indices of all elements less than
        `tholds[0]` and greater than `tholds[1]`
                
        Parameters
        ----------
        modes : list of ints, default is None
            Modes of Component to extract indices from. If `modes=None`, all
            Component modes will be included.
        tholds : tuple of ints
            Thresholds of values to be included in indices. Indices of values 
            greater than or equal to `tholds[0]` and less than or equal to 
            `tholds[1]` will not be included in output.
        
        Returns
        -------
        indices : list of numpy.ndarrays
            Arrays of Component indices. One array for each mode.
        """
        if modes is None:
            modes = [i for i in range(len(self.shape))]
        indices = []
        for i, f in self.factors:
            if i in modes:
                index = np.where(np.any([f < tholds[0], f > tholds[1]], axis=0))[0]
                indices.append(index)
        return indices


class SparseCPTensor(CPTensor):
    """Class container for methods related to sparse CP tensors.
    """
    def __init__(self, cp_tensor):
        super().__init__(cp_tensor)
        
    def __repr__(self):
        message = 'Rank-{} SparseCPTensor of shape {}'.format(self.rank, self.shape)
        return message
    
    def get_components(self):
        """Generate list of Component objects from SparseCPTensor factors.
        
        Returns
        -------
        components : list of Components
            List of Component objects, where components[i] is the i-th factor
            of the parent SparseCPTensor.
        """
        components = []
        for i in range(self.rank):
            factor_weights = [factor.T[i].T for factor in self.factors]
            component_weight = self.weights[i]
            components.append(Component(component_weight, factor_weights))
        return components
        

class SimSparseCPTensor(SparseCPTensor):
    """Class container for methods related to simulated sparse CP tensors.
    """
    def __init__(self, 
                 cp_tensor):
        super().__init__(cp_tensor)
        
    def __repr__(self):
        message = 'Rank-{} SimSparseCPTensor of shape {}'.format(self.rank, self.shape)
        return message
        
    def to_tensor(
        self, 
        noise_level=0, 
        sparse_noise=False, 
        noise_distribution=None, 
        random_state=None
    ):
        """Generate optionally noisey data tensor from factorized CP tensor.
        This method overwrites the tensorly.cp_tensor.CPTensor.to_tensor()
        parent method.
        
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


def simulated_sparse_tensor(
    shape, 
    rank, 
    densities=None, 
    factor_dist_list=None,  
    weights=None, 
    random_state=None
):
    """Generates simulated data in the form of a sparse cp_tensor
    
    Parameters
    ----------
    shape : tuple of ints
        Tensor shape where len(shape) = n modes in tensor.
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
    sim_cp : SimSparseCPTensor
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
    sim_cp = SimSparseCPTensor((weights, factors))
    return sim_cp
