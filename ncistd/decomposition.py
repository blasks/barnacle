import numpy as np
from spams import lasso, lassoMask
import tensorly as tl 
from tensorly import unfold
from tensorly.cp_tensor import CPTensor, cp_to_tensor
from tensorly.decomposition._base_decomposition import DecompositionMixin
from tensorly.decomposition._cp import initialize_cp
from tensorly.tenalg import khatri_rao
# from .utils import zeros_cp

##########
# function to generate tensorly CPTensor of all zeros
# NOTE: This is redundant - need to figure this out later
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

class SparseCP(DecompositionMixin):
    """Sparse Candecomp-Parafac decomposition 
    
    """
    def __init__(self, max_rank, lambdas, nonneg_modes=None, 
                 n_iter_max=100, init='random', normalize_factors='l2', 
                 tolerance=1e-6, random_state=None, verbose=0, 
                 cvg_criterion='rec_error'):
        self.max_rank = max_rank
        self.lambdas = lambdas
        self.nonneg_modes = nonneg_modes
        self.n_iter_max = n_iter_max
        self.init = init
        self.normalize_factors = normalize_factors
        self.tolerance = tolerance
        self.random_state = random_state
        self.verbose = verbose
        self.cvg_criterion = cvg_criterion
        
    def fit_transform(self, X, mask=None, return_errors=True):
        """Fit model to data using deflation algorithm with optional mask
        Parameters
        ----------
        self : SparseCP
            Initialized model
        X : numpy.ndarray
            Input data tensor to fit
        mask : numpy.ndarray
            Array of booleans with the same shape as X. Should be 0 where
            the values are missing and 1 everywhere else. 
            Allows for missing values.
        return_errors : bool, optional
            Activate return of iteration errors
            
        Returns
        -------
        decomposition : tensorly.CPTensor
            A CPTensor of the decomposition, where each factor is derived
            from each successive deflation up to ``self.max_rank`` iterations.
        errors : list
            A list of reconstruction errors at each iteration of the algorithm.
        """
        if return_errors:
            error_list = []
        decomposition = zeros_cp(X.shape, self.max_rank)
        residual_tensor = X.copy()
        ndim = len(X.shape)
        for r in range(self.max_rank):
            if self.verbose > 0:
                print('Fitting factor {}'.format(r))
            layer, errors = als_lasso(tensor=residual_tensor, 
                                    rank=1, 
                                    lambdas=self.lambdas, 
                                    nonneg_modes=self.nonneg_modes, 
                                    mask=mask, 
                                    n_iter_max=self.n_iter_max, 
                                    init=self.init, 
                                    normalize_factors=self.normalize_factors, 
                                    tolerance=self.tolerance, 
                                    random_state=self.random_state, 
                                    verbose=self.verbose - 1, 
                                    return_errors=True, 
                                    cvg_criterion=self.cvg_criterion
                                    )
            # update weights with layer weight
            decomposition.weights[r] = layer.weights[0]
            # update factors with layer factors
            for d in range(ndim):
                decomposition.factors[d][:, r] = layer.factors[d][:, 0]
            # record errors
            if return_errors:
                error_list.append(errors)
            # deflate residual tensor
            residual_tensor = residual_tensor - layer.to_tensor()
        self.decomposition_ = decomposition
        if return_errors:
            return decomposition, errors
        else:
            return decomposition
    
    def __repr__(self):
        return '{} decomposition with max rank {}'.format(self.__class__.__name__, self.max_rank)
    

def als_lasso(tensor, 
              rank, 
              lambdas, 
              nonneg_modes=None, 
              mask=None,  
              n_iter_max=100, 
              init='random', 
              normalize_factors='l2', 
              tolerance=1e-6, 
              random_state=None, 
              verbose=0, 
              return_errors=False, 
              cvg_criterion='rec_error'):
    """Sparse CP decomposition by L1-penalized Alternating Least Squares (ALS)
    
    Computes a rank-`rank` decomposition of `tensor` such that::
        tensor = [|weights; factors[0], ..., factors[-1] |].
    
    Parameters
    ----------
    tensor : numpy.ndarray
        Input data tensor.
    rank  : int
        Number of components.
    lambdas : [float]
        Vector of length tensor.ndim in which lambda[i] is the l1 sparsity 
        parameter for factor[i].
    nonneg_modes : [int]
        List of modes to force to be non-negative. Default is None.
    mask : numpy.ndarray, optional
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
    normalize_factors : {'l2', 'max', None}
        Method by which factors will be normalized, with normalization values
        being stored in `weights`. If `normalize_factors`=None, no normalization
        will be computed and `weights` will be returned as a vector of ones.
        Default is `normalize_factors`='l2'. 
    tolerance : float, optional
        (Default: 1e-6) Convergence tolerance. The
        algorithm is considered to have found the global minimum when the
        convergence criterion is less than `tolerance`.
    random_state : {None, int, numpy.random.RandomState}
    verbose : int, optional
        Level of verbosity
    return_errors : bool, optional
        Activate return of iteration errors
    cvg_criterion : {'rec_error', 'norm'}, optional
        Stopping criterion for ALS, works if `tol` is not None. 
        'rec_error' : `|previous rec_error - current rec_error| < tol`
        'norm' : `|previous norm(tensor) - current norm(tensor)| < tol`
            Only works if there are masked values to be imputed.

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
    # get mask ready
    if mask is None:
        mask = np.ones_like(tensor, dtype=bool)
    else:
        # make sure mask is boolean type
        mask = np.array(mask, dtype=bool)
        
    # set modes to be non-negative
    if nonneg_modes is None:
        nonneg = [False for i in range(tensor.ndim)]
    else:
        nonneg = [True if i in nonneg_modes else False for i in range(tensor.ndim)]
        
    # double check lambdas are floats
    lambdas = np.array(lambdas, dtype=float)
        
    # initialize list to store reconstruction errors
    rec_errors = []

    # initialize factors and weights
    weights, factors = initialize_cp(tensor, rank, init=init, 
                                     random_state=random_state, 
                                     normalize_factors=normalize_factors)
    
    # begin iterations
    for iteration in range(n_iter_max):
        if verbose > 1:
            print('Starting iteration {}'.format(iteration + 1))
        
        # save previous tensor values if need be
        if cvg_criterion == 'norm':
            old_tensor = tensor
            
        # loop through modes
        for mode in range(tl.ndim(tensor)):
            if verbose > 1:
                print('Mode {} of {}'.format(mode, tl.ndim(tensor)))
            
            # take the khatri_rao product of all factors except factors[mode]
            kr_product = khatri_rao(factors, None, skip_matrix=mode)
            # unfold data tensor and mask along mode
            X_unfolded = unfold(tensor, mode)
            if iteration == 0:
                # unfold the mask as well
                mask_unfolded = unfold(mask, mode)
                # generate new factor with masked lasso decomposition
                factor_update = lassoMask(X=np.asfortranarray(X_unfolded.T), 
                                        D=np.asfortranarray(kr_product),  
                                        B=np.asfortranarray(mask_unfolded.T), 
                                        lambda1=lambdas[mode], 
                                        pos=nonneg[mode])
            else: 
                # generate new factor with lasso decomposition (no mask)
                factor_update = lasso(X=np.asfortranarray(X_unfolded.T), 
                                      D=np.asfortranarray(kr_product), 
                                      lambda1=lambdas[mode], 
                                      pos=nonneg[mode])
            
            # convert factor back to numpy array and transpose
            factor_update = factor_update.toarray().T
            
            # normalize new factor
            if normalize_factors == 'l2' or (iteration == 0 and normalize_factors == 'max'):
                scales = tl.norm(factor_update, 2, axis=0)
                # replace zeros with ones
                weights = tl.where(scales==0, 
                                   tl.ones(tl.shape(scales), 
                                           **tl.context(factor_update)), 
                                   scales)
                factor_update = factor_update / tl.reshape(weights, (1, -1))
            elif normalize_factors == 'max':
                weights = np.max(factor_update, 0)
                # WARNING: This will replace zero weights with 1 AS WELL AS 
                # replacing weights between 0 and 1 with 1. Do I want this?
                weights = np.max([weights, np.ones_like(weights)], 0)
                factor_update = factor_update / tl.reshape(weights, (1, -1))
            elif normalize_factors is None:
                pass
            else:
                raise ValueError('Invalid option passed to `normalize_factors`')
            
            # update factor
            factors[mode] = factor_update
        
        # build reconstruction from CP decomposition
        reconstruction = cp_to_tensor((weights, factors))
        # update completed tensor with most recent imputations
        tensor = tensor * mask + reconstruction * (1 - mask)
        # compute reconstruction error if needed
        if cvg_criterion == 'rec_error' or return_errors:
            # compute normalized reconstruction error
            rec_error = tl.norm(tensor - reconstruction, 2) / tl.norm(tensor, 2)
            rec_errors.append(rec_error)
            if verbose:
                print('reconstruction error: {}'.format(rec_errors[-1]))
        
        # check convergence
        if tolerance != 0 and iteration != 0:
            if cvg_criterion == 'rec_error':
                fit_change = abs(rec_errors[-2] - rec_errors[-1])
            elif cvg_criterion == 'norm':
                fit_change = tl.norm(old_tensor - tensor, 2)
            else:
                raise ValueError('Invalid convergence criterion: {}'.format(cvg_criterion))
            # compare fit change to tolerance
            if fit_change < tolerance:
                if verbose:
                    print('Algorithm converged after {} iterations'.format(iteration))
                break
        
    # return result
    cp_tensor = CPTensor((weights, factors))
    if return_errors:
        return cp_tensor, rec_errors
    else:
        return cp_tensor

