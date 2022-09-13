import numpy as np
import opt_einsum as oe
from spams import lasso
import tensorly as tl 
from tensorly import unfold, check_random_state, cp_normalize
from tensorly.cp_tensor import cp_to_tensor, unfolding_dot_khatri_rao
from tensorly.decomposition._base_decomposition import DecompositionMixin
from tensorly.decomposition._cp import initialize_cp
from tensorly.tenalg import khatri_rao
from threadpoolctl import threadpool_limits
import warnings


def _create_mttkrp_function(shape, rank):
    """Helper function to generate the mttkrp computation function to be 
    used with the `mttkrp_optimization='opt_einsum'` option in `als_lasso`. 
    """
    # define einsum paths for each skip mode
    mttkrp_paths = [
        oe.contract_expression('jr,kr,ijk -> ir', (shape[1], rank), (shape[2], rank), shape),
        oe.contract_expression('ir,kr,ijk -> jr', (shape[0], rank), (shape[2], rank), shape),
        oe.contract_expression('ir,jr,ijk -> kr', (shape[0], rank), (shape[1], rank), shape)
    ]
    # define mttkrp function
    def mttkrp(X, factors, mode):
        mttkrp_path = mttkrp_paths[mode]
        summed_factors = [fm for m, fm in enumerate(factors) if m != mode]
        return mttkrp_path(*summed_factors, X)
    # return the function
    return mttkrp


def als_lasso(
    tensor, 
    rank, 
    lambdas, 
    nonneg_modes=None, 
    init='random', 
    tol=1e-6, 
    n_iter_max=1000, 
    mttkrp_optimization='opt_einsum', 
    random_state=None, 
    threads=None, 
    verbose=0, 
    return_losses=False
):
    """Sparse CP decomposition by L1-penalized Alternating Least Squares (ALS)
    
    Computes a rank-`rank` decomposition of `tensor` such that::
    
        tensor = [|weights; factors[0], ..., factors[-1]|].
        
    The algorithm aims to minimize the loss as defined by::
    
        loss = `tl.norm(tensor - reconstruction, 2) ** 2 + penalties`
            where `penalties` are calculated as the dot product of the `lambdas` 
            sparsity coefficients and the L1 norms of the factor matrices.
            
    Furthermore, the factor matrices indicated in `nonneg_modes` are forced
    to be non-negative, and the L2 norm of any factor matrices without an L1
    sparsity penalty (lambda=0.0) is constrained to be unit length.
    
    Parameters
    ----------
    tensor : numpy.ndarray
        Input data tensor.
    rank : int
        Number of components.
    lambdas : [float]
        Vector of length tensor.ndim in which lambdas[i] is the l1 sparsity 
        coefficient for factor[i]. If `lambdas` is set to all zeros, this is
        the equivalent of fitting a standard CP decomposition without any
        sparsity constraints.
    nonneg_modes : [int], default is None
        List of modes forced to be non-negative.
    init : {'random', CPTensor}, default is 'random'.
        Values used to initialized the factor matrices. If `init == 'random'` 
        then factor matrices are initialized with uniform distribution using 
        `random_state`. If init is a previously initialized `cp tensor`, any 
        weights are incorporated into the last factor, and then the initial 
        weight values for the output decomposition are set to '1'.
    tol : float, default is 1e-6
        Convergence tolerance. The algorithm is considered to have found the 
        global minimum when the change in loss from one iteration to the next 
        falls below the `tol` threshold.
    n_iter_max : int, default is 1000
        Maximum number of iterations. If the algorithm fails to converge 
        according to the `tol` threshold set, an error will be raised.
    mttkrp_optimization : {'opt_einsum', 'tensorly', None}, default is 'opt_einsum'
        Inner loop optimization using the Matricicized Tensor Times Khatri-Rao
        Product (MTTKRP).
            'opt_einsum' : The opt_einsum package is used to produce the MTTKRP 
                (only works for mode-3 tensors).
            'tensorly' : The tensorly.cp_tensor.unfolding_dot_khatri_rao 
                function is used to produce the MTTKRP.
            None : No MTTKRP optimization is used.
    random_state : {None, int, numpy.random.RandomState}, default is None
        Used to initialized factor matrices and weights.
    threads : int, default is None
        Maximum number of threads allocated to the algorithm. If `threads`=None, 
        then all available threads will be used.
    verbose : int, default is 0
        Level of verbosity.
    return_losses : bool, default is False
        Activate return of iteration loss values at each iteration.
        
    Returns
    -------
    cp_tensor : (weight, factors)
        * weights : 1D array of shape (rank,) that contains the weights of the
            factors, in which the L2 norm has been normalized to unit lenght.
        * factors : List of factors of the CP decomposition where factor matrix 
            `i` is of shape ``(tensor.shape[i], rank)``
    losses : list
        A list of loss values calculated at each iteration of the algorithm. 
        Only returned when `return_losses` is set to True.
    """
    # set threads for lasso function
    lasso_threads = -1 if threads is None else threads
    
    # wrap operations in threadpool limit
    with threadpool_limits(limits=threads, user_api='blas'):
    
        # calculate number of modes in tensor
        n_modes = tl.ndim(tensor)
            
        # set modes to be non-negative
        if nonneg_modes is None:
            nonneg = [False for i in range(n_modes)]
        else:
            nonneg = [True if i in nonneg_modes else False for i in range(n_modes)]
            
        # check lambdas
        lambdas = np.array(lambdas, dtype=float)
        if len(lambdas) != n_modes:
            raise ValueError(
                'The number of sparsity coefficients in `lambdas`' +
                'is not equal to the number of modes in `tensor`.'
            )
            
        # set modes without L2 sparsity penalization to be L2 normalized
        normalize_modes = [i for i, l in enumerate(lambdas) if l == 0.0]
            
        # initialize list to store losses
        losses = []

        # initialize factors and weights
        weights, factors = initialize_cp(
            tensor, 
            rank, 
            init=init, 
            random_state=random_state
        )
        
        # build MTTKRP function if opt_einsum optimization has been selected
        if mttkrp_optimization == 'opt_einsum':
            compute_mttkrp = _create_mttkrp_function(tensor.shape, rank)
        
        # begin iterations
        for iteration in range(n_iter_max):
            if verbose > 2:
                print('\nStarting iteration {}'.format(iteration), flush=True)
                
            # loop through modes
            for mode in range(n_modes):
                if verbose > 3:
                    print('\tMode {} of {}'.format(mode, n_modes), flush=True)
                
                # unfold data tensor along mode
                X_unfolded = unfold(tensor, mode)
                
                # ALS lasso without MTTKRP optimization
                if mttkrp_optimization is None:
                    # take the khatri_rao product of all factors except factors[mode]
                    kr_product = khatri_rao(factors, None, skip_matrix=mode)
                    # generate new factor by solving lasso problem
                    factor_update = lasso(
                        X=np.asfortranarray(X_unfolded.T), 
                        D=np.asfortranarray(kr_product), 
                        lambda1=lambdas[mode], 
                        pos=nonneg[mode], 
                        numThreads=lasso_threads
                    )
                    
                # ALS lasso with MTTKRP optimization
                else:
                    # form DtD, containing kr_product.T @ kr_product
                    DtD = np.ones((rank, rank))
                    for krp_mode in range(n_modes):
                        if krp_mode == mode:
                            continue
                        DtD *= np.matmul(factors[krp_mode].T, factors[krp_mode])
                        
                    # form MTTKRP with opt_einsum optimization
                    if mttkrp_optimization == 'opt_einsum':
                        # form DtX, containing kr_product.T @ X_unfolded
                        DtX = compute_mttkrp(tensor, factors, mode)
                        
                    # form MTTKRP with tensorly function
                    elif mttkrp_optimization == 'tensorly':
                        # form DtX, containing kr_product.T @ X_unfolded
                        DtX = unfolding_dot_khatri_rao(tensor, (None, factors), mode)
                    
                    # unrecognized optimization option 
                    else:
                        raise ValueError('The argument passed to ' + 
                                        '`mttkrp_optimization` is not reconized.')
                
                    # generate new factor by solving lasso problem with MTTKRP
                    factor_update = lasso(
                        X=np.asfortranarray(X_unfolded.T), 
                        Q=np.asfortranarray(DtD), 
                        q=np.asfortranarray(DtX.T), 
                        lambda1=lambdas[mode], 
                        pos=nonneg[mode], 
                        numThreads=lasso_threads
                    )
                    
                # convert factor back to numpy array and transpose
                factor_update = factor_update.toarray().T
                
                # normalize new factor if no L1 sparsity penalty was applied
                if mode in normalize_modes:
                    scales = tl.norm(factor_update, 2, axis=0)
                    # replace zeros with ones
                    weights = tl.where(
                        scales==0, 
                        tl.ones(tl.shape(scales), **tl.context(factor_update)), 
                        scales
                    )
                    # normalize factor update
                    factor_update = factor_update / tl.reshape(weights, (1, -1))
                
                # update factor
                factors[mode] = factor_update
                    
            # compute loss using tensor reconstructed from latest factor updates
            reconstruction = cp_to_tensor((weights, factors))
            # calculate L1 sparsity penalties
            factor_l1_norms = np.array([tl.norm(f, 1) for f in factors])
            penalties = tl.dot(lambdas, factor_l1_norms)
            # compute loss
            loss = tl.norm(tensor - reconstruction, 2) ** 2 + penalties
            # append loss to history
            losses.append(loss)
            if verbose > 1:
                print('loss: {}'.format(losses[-1]), flush=True)
            
            # check convergence
            if tol != 0 and iteration != 0:
                # calculate change in loss
                loss_change = abs(losses[-2] - losses[-1])
                # compare change in loss to tolerance
                if loss_change < tol:
                    if verbose > 0:
                        message = 'Algorithm converged after {} iterations'.format(iteration+1)
                        print(message, flush=True)
                    break
                # close out with warnings if the iteration maximum has been reached
                elif iteration == n_iter_max - 1:
                    message = 'Algorithm failed to converge after {} iterations'.format(iteration+1)
                    if verbose > 0:
                        print(message, flush=True)
                    warnings.warn(message)
        
        # normalize converged cp tensor
        cp_tensor = cp_normalize((weights, factors))
        
        # return result
        if return_losses:
            return cp_tensor, losses
        else:
            return cp_tensor


class SparseCP(DecompositionMixin):
    """Sparse Candecomp-Parafac decomposition.
    
    """
    def __init__(
        self, 
        rank, 
        lambdas, 
        nonneg_modes=None, 
        init='random', 
        tol=1e-6, 
        n_iter_max=1000, 
        random_state=None, 
        n_initializations=1
    ):
        # initialize passed parameters
        self.rank = rank
        self.lambdas = lambdas
        self.nonneg_modes = nonneg_modes
        self.init = init
        self.tol = tol
        self.n_iter_max = n_iter_max
        self.random_state = random_state
        self.n_initializations = n_initializations
        
        # initialize internal parameters
        self._best_cp_index = None
        
    @property  
    def decomposition(self):
        if self._best_cp_index is None:
            raise AttributeError('The model has not been fit with data.')
        else:
            return self.candidates_[self._best_cp_index]
        
    @property  
    def loss(self):
        if self._best_cp_index is None:
            raise AttributeError('The model has not been fit with data.')
        else:
            return self.candidate_losses_[self._best_cp_index]
        
    def fit_transform(
        self, 
        tensor, 
        mttkrp_optimization='opt_einsum', 
        threads=None, 
        verbose=0, 
        return_losses=False
    ):
        """Fit model to data
        
        """
        # initialize lists of candidate cp_tensors and their losses
        candidates = list()
        candidate_losses = list()
        
        # initialize lowest error
        lowest_err = float('inf')
        
        # initialize random state
        rns = check_random_state(self.random_state)
        
        # run multiple initializations
        for i in range(self.n_initializations):
            if verbose > 0:
                print('\nBeginning initialization {} of {}'.format(
                    i+1, self.n_initializations))
            # fit model
            cp, loss = als_lasso(
                tensor, 
                self.rank, 
                self.lambdas, 
                nonneg_modes=self.nonneg_modes, 
                init=self.init, 
                tol=self.tol, 
                n_iter_max=self.n_iter_max, 
                mttkrp_optimization=mttkrp_optimization, 
                random_state=rns, 
                threads=threads, 
                verbose=verbose - 1, 
                return_losses=True
            )
            # store candidates
            candidates.append(cp)
            candidate_losses.append(loss)
            # keep best fit
            if loss[-1] < lowest_err:
                lowest_err = loss[-1]
                best_cp_index = i
        
        # store results
        self.candidates_ = candidates
        self.candidate_losses_ = candidate_losses
        self._best_cp_index = best_cp_index
        
        # return best decomposition and, optionally, losses
        if return_losses:
            return candidates[best_cp_index], candidate_losses[best_cp_index]
        else:
            return candidates[best_cp_index]
    