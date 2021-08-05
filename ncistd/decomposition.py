import numpy as np
from spams import lasso, lassoMask
import tensorly as tl 
from tensorly import unfold
from tensorly.cp_tensor import CPTensor
from tensorly.decomposition._base_decomposition import DecompositionMixin
from tensorly.decomposition._cp import initialize_cp
from tensorly.tenalg import khatri_rao

# class SparseCP(DecompositionMixin):
#     """Sparse Candecomp-Parafac decomposition 
    
#     """
#     def __init__(self, rank, init='random'):
#         self.rank = rank
#         self.init = init
        
#     def fit_transform(self, X):
#         self.decomposition_ = modified_als(X, rank=self.rank, init=self.init)
#         return self.decomposition_
    
#     def __repr__(self):
#         return '{} decomposition of rank {}'.format(self.__class__.__name__, self.rank)
    

def als_lasso(tensor, 
              rank, 
              lambdas, 
              mask=None,  
              n_iter_max=100, 
              init='random', 
              normalize_factors=False, 
              tol=1e-8, 
              random_state=None, 
              verbose=0, 
              return_errors=False, 
              cvg_criterion='abs_rec_error'):
    """Sparse CP decomposition by L1-penalized Alternating Least Squares (ALS)
    
    Computes a rank-`rank` decomposition of `tensor` such that::
        tensor = [|weights; factors[0], ..., factors[-1] |].
    
    Parameters
    ----------
    tensor : numpy.ndarray
        Input data tensor.
    rank  : int
        Number of components.
    lambda : numpy.array
        Vector of length tensor.ndim in which lambda[i] is the l1 sparsity 
        parameter for factor[i].
    mask : numpy.ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else. Note:  if tensor is
        sparse, then mask should also be sparse with a fill value of 1 (or
        True). Allows for missing values.
    n_iter_max : int
        Maximum number of iteration
    init : {'svd', 'random', CPTensor}, optional
        Type of factor matrix initialization.
        If a CPTensor is passed, this is directly used for initalization.
        See `initialize_factors`.
    normalize_factors : if True, aggregate the weights of each factor in a 1D-tensor
        of shape (rank, ), which will contain the norms of the factors
    tol : float, optional
        (Default: 1e-6) Relative reconstruction error tolerance. The
        algorithm is considered to have found the global minimum when the
        reconstruction error is less than `tol`.
    random_state : {None, int, numpy.random.RandomState}
    verbose : int, optional
        Level of verbosity
    return_errors : bool, optional
        Activate return of iteration errors
    cvg_criterion : {'abs_rec_error', 'rec_error'}, optional
       Stopping criterion for ALS, works if `tol` is not None. 
       If 'rec_error',  ALS stops at current iteration if ``(previous rec_error - current rec_error) < tol``.
       If 'abs_rec_error', ALS terminates when `|previous rec_error - current rec_error| < tol`.

    Returns
    -------
    CPTensor : (weight, factors)
        * weights : 1D array of shape (rank, )
          * all ones if normalize_factors is False (default)
          * weights of the (normalized) factors otherwise
        * factors : List of factors of the CP decomposition element `i` is of shape ``(tensor.shape[i], rank)``
        * sparse_component : nD array of shape tensor.shape. Returns only if `sparsity` is not None.
    errors : list
        A list of reconstruction errors at each iteration of the algorithms.
    """
    # get all the parts ready
    if mask is None:
        mask = np.ones_like(tensor)
    rec_errors = []
    modes_list = [mode for mode in range(tl.ndim(tensor))]
    norm_tensor = tl.norm(tensor, 2)

    # initialize factors and weights
    weights, factors = initialize_cp(tensor, rank, init=init, 
                                     random_state=random_state, 
                                     normalize_factors=normalize_factors)
    
    # begin iterations
    for iteration in range(n_iter_max):
        if verbose > 1:
            print('Starting iteration {}'.format(iteration + 1))
        for mode in modes_list:
            if verbose > 1:
                print('Mode {} of {}'.format(mode, tl.ndim(tensor)))
            # take the khatri_rao product of all factors except factors[mode]
            kr_product = khatri_rao(factors, weights, skip_matrix=mode)
            X_unfolded = unfold(tensor, mode)
            mask_unfolded = unfold(mask, mode)
            factor_update = lassoMask(X=np.asfortranarray(X_unfolded.T), 
                                      D=np.asfortranarray(kr_product), 
                                      B=np.asfortranarray(mask_unfolded.T), 
                                      lambda1=lambdas[mode])
            # normalize new factor
            # update factor
            factors[mode] = factor_update.toarray().T
            # update weights
            
        # check convergence
        
        
        
    # return result
    cptensor = CPTensor(weights, factors)
    if return_errors:
        return cp_tensor, rec_errors
    else:
        return cp_tensor

