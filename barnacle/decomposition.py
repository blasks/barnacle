import warnings

import numpy as np
import opt_einsum as oe
import tensorly as tl
from tensorly import check_random_state
from tensorly.decomposition._base_decomposition import DecompositionMixin
from tensorly.decomposition._cp import initialize_cp
from threadpoolctl import threadpool_limits

from .fista import fista_solve
from .tensors import SparseCPTensor


def _create_mttkrp_function(shape, rank):
    """Helper function to generate the function for calculating the Matricized
    Tensor Times Khatri-Rao Product (MTTKRP) using the opt_einsum package.
    
    Parameters
    ----------
    shape : [int, int, int]
        List of 3 integers delineating the shape of the input tensor.
    rank : int
        Number of components in the CP tensor decomposition model.
        
    Returns
    -------
    mttkrp : function
        MTTKRP function parameterized to the input tensor size and rank.
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
    norm_constraint=True, 
    init='random', 
    tol=1e-6, 
    n_iter_max=1000,  
    random_state=None, 
    threads=None, 
    verbose=0, 
    return_losses=False
):
    """Computes a rank-`rank` decomposition of `tensor` such that::
    
        tensor = [|weights; factors[0], ..., factors[-1]|].
        
    The algorithm aims to minimize the loss as defined by::
    
        loss = `tl.norm(tensor - reconstruction, 2) ** 2 + penalties`
            where `penalties` are calculated as the dot product of the `lambdas` 
            sparsity coefficients and the L1 norms of the factor matrices.
            
    Furthermore, the factor matrices indicated in `nonneg_modes` are forced
    to be non-negative, and if `norm_constraint`=True, the L2 norm of any 
    factor matrix without an L1 sparsity penalty (lambda=0.0) is constrained to 
    be unit length.
    
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
    norm_constraint : bool, default is True
        If `norm_constraint`=True, the L2 norm of any factor matrix without an 
        L1 sparsity penalty (lambda=0.0) is constrained to unit length. If the
        sparsity penalty of every mode is 0.0, the L2 norm constraint is 
        automatically turned off in every mode.
    init : {'random', CPTensor}, default is 'random'.
        Values used to initialized the factor matrices. If `init == 'random'` 
        then factor matrices are initialized with uniform distribution using 
        `random_state`. If init is a previously initialized `cp tensor`, any 
        weights are incorporated into the last factor, and then the initial 
        weight values for the output decomposition are set to '1'.
    tol : float, default is 1e-6
        Convergence tolerance. The algorithm is considered to have found the 
        global minimum when the change in loss from one iteration 
        to the next falls below the `tol` threshold. The calculated change in
        loss is relative when loss > 1 and absolute when loss <= 1. In other 
        words, convergence is defined as :
        
        .. math::
        
            \frac{\|l^{(n-1)} - l^{(n)}\|}{\max(l^{(n)}, 1)} < t
        
        where :math: `l^{(n)}` is the loss at iteration :math: `n`, and
        :math: `t` is the tolerance threshold set by `tol`. 
        
    n_iter_max : int, default is 1000
        Maximum number of iterations. If the algorithm fails to converge 
        according to the `tol` threshold set, a warning will be raised.
    random_state : {None, int, numpy.random.RandomState}, default is None
        Random state used to initialized factor matrices and weights.
    threads : int, default is None
        Maximum number of threads allocated to the algorithm. 
        If `threads` = None, then all available threads will be used.
    verbose : int, default is 0
        Level of verbosity.
    return_losses : bool, default is False
        Activate return of iteration loss values at each iteration.
        
 
    cp_tensor : (weight, factors)
        * weights : 1D array of shape (rank,) that contains the weights denoting
            the relative contributio of each factor.
        * factors : List of factors of the CP decomposition where factor matrix 
            `i` is of shape `(tensor.shape[i], rank)`
    losses : list
            A list of loss values calculated at each iteration of the algorithm. 
            Only returned when `return_losses` is set to True.
    """    
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
        if any(lambdas < 0):
            raise ValueError('L1 sparsity penalty must be nonnegative.')
        
        # set normalization modes
        if not norm_constraint or not np.any(lambdas):
            normalize = [False for i in range(n_modes)]
        else:
            normalize = [True if lambdas[i] == 0 else False for i in range(n_modes)]
            
        # initialize list to store losses
        losses = []

        # initialize factors and weights
        weights, factors = initialize_cp(
            tensor, 
            rank, 
            init=init, 
            random_state=random_state
        )
        weights = None  # Make sure the weights are 1

        tensor_norm_sq = tl.norm(tensor)**2
        
        # build MTTKRP function if opt_einsum optimization has been selected
        compute_mttkrp = _create_mttkrp_function(tensor.shape, rank)
        
        # begin iterations
        for iteration in range(n_iter_max):
            if verbose > 2:
                print('\nStarting iteration {}'.format(iteration), flush=True)
                
            # loop through modes
            for mode in range(n_modes):
                if verbose > 3:
                    print('\tMode {} of {}'.format(mode, n_modes), flush=True)

                # form DtD, containing kr_product.T @ kr_product
                DtD = np.ones((rank, rank))
                for krp_mode in range(n_modes):
                    if krp_mode == mode:
                        continue
                    DtD *= np.matmul(factors[krp_mode].T, factors[krp_mode])
                DtX = compute_mttkrp(tensor, factors, mode)

                # generate new factor by solving lasso problem with MTTKRP
                # fista_solve solves
                #     min 0.5 ||DY - X||^2 + lambda ||Y||_1.
                # We want to solve
                #     min ||DY - X||^2 + lambda ||Y||_1,
                # so we multiply the whole problem by 2 (doesn't change anything) and see that
                # fista_solve solves
                #     min ||DY - X|| + 2 lambda ||Y||_1.
                # If we multiply the lambda by 0.5, then we (informally) get
                #     min ||DY - X|| + 2 (0.5) lambda ||Y||_1 = min ||DY - X|| + lambda ||Y||_1
                # which is what we want to solve.
                factors[mode] = fista_solve(
                    lhs=DtD, 
                    rhs=DtX.T,
                    l1_reg=0.5*lambdas[mode],
                    nonnegative=nonneg[mode], 
                    normalize=normalize[mode],
                    init=factors[mode].T
                ).T
                
            # Compute loss using tensor reconstructed from latest factor updates
            # Faster version to compute the loss, uses the fact that
            # ||DY - X||^2 = Tr(Y^T D^TD X) - 2 Tr(X^T D^T B) + Tr(B^T B)
            #              = ||X_estimated||^2 - 2 Tr(X^T D^T B) + ||X||^2
            #              = cp_norm^2 + ||X||^2 - 2 sum(X^T * D^T B)
            # where * is hadamard product and cp_norm can be estimated efficiently
            iprod = tl.sum(tl.sum(DtX * factors[mode], axis=0))
            factors_norm_sq = tl.cp_tensor.cp_norm((weights, factors))**2
            sse = tensor_norm_sq + factors_norm_sq - 2*iprod

            # calculate L1 sparsity penalties
            factor_l1_norms = np.array([tl.norm(f, 1) for f in factors])
            penalties = tl.dot(lambdas, factor_l1_norms)
            # compute loss
            loss = sse + penalties
            # append loss to history
            losses.append(loss)
            if verbose > 1:
                print('loss: {}'.format(losses[-1]), flush=True)
            
            # stop iterations if loss has acheived zero 
            if loss == 0.0:
                if verbose > 0:
                    message = 'Algorithm converged after {} iterations'.format(iteration+1)
                    print(message, flush=True)
                break
            
            # check convergence
            if tol != 0 and iteration != 0:
                # calculate change in loss (relative if loss is > 0, abso)
                loss_change = abs(losses[-2] - losses[-1]) / max(losses[-1], 1)
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
        
        # return result
        if return_losses:
            return SparseCPTensor((None, factors)), losses
        else:
            return SparseCPTensor((None, factors))


class SparseCP(DecompositionMixin):
    """Sparse CP decomposition by L1-penalized Alternating Least Squares (ALS)
    
    Parameterizes a rank-`rank` decomposition of `tensor` such that::
    
        tensor = [|weights; factors[0], ..., factors[-1]|].
             
    The `lambdas` values indicate the sparsity-inducing l1 regularization 
    coefficient to be applied to each mode when the model is fit to data. The 
    factor matrices indicated in `nonneg_modes` are forced to be non-negative, 
    and if `norm_constraint` = True, the L2 norm of any factor matrix without an 
    L1 sparsity penalty (lambda=0.0) is constrained to be unit length.
    
    
    rank : int
        Number of components.
    lambdas : [float]
        Vector of length tensor.ndim in which lambdas[i] is the l1 sparsity 
        coefficient for factor[i]. If `lambdas` is set to all zeros, this is
        the equivalent of fitting a standard CP decomposition without any
        sparsity constraints.
    nonneg_modes : [int], default is None
        List of modes forced to be non-negative.
    norm_constraint : bool, default is True
        If `norm_constraint` = True, the L2 norm of any factor matrix without an 
        L1 sparsity penalty (lambda=0.0) is constrained to unit length. If the
        sparsity penalty of every mode is 0.0, the L2 norm constraint is 
        automatically turned off in every mode.
    init : {'random', CPTensor}, default is 'random'.
        Values used to initialized the factor matrices. If `init == 'random'` 
        then factor matrices are initialized with uniform distribution using 
        `random_state`. If init is a previously initialized `cp tensor`, any 
        weights are incorporated into the last factor, and then the initial 
        weight values for the output decomposition are set to '1'.
    tol : float, default is 1e-6
        Convergence tolerance. The algorithm is considered to have found the 
        global minimum when the change in loss from one iteration 
        to the next falls below the `tol` threshold. The calculated change in
        loss is relative when loss > 1 and absolute when loss <= 1. In other 
        words, convergence is defined as :
        
        .. math::
        
            \frac{\|l^{(n-1)} - l^{(n)}\|}{\max(l^{(n)}, 1)} < t
        
        where :math: `l^{(n)}` is the loss at iteration :math: `n`, and
        :math: `t` is the tolerance threshold set by ``tol``. 
        
    n_iter_max : int, default is 1000
        Maximum number of iterations. If the algorithm fails to converge 
        according to the `tol` threshold set, a warning will be raised.
    random_state : {None, int, numpy.random.RandomState}, default is None
        Random state used to initialized factor matrices and weights.
    n_initializations : int, default is 1
        Number of random initializations to compute.
    """
    def __init__(
        self, 
        rank, 
        lambdas, 
        nonneg_modes=None, 
        norm_constraint=True, 
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
        self.norm_constraint = norm_constraint
        self.init = init
        self.tol = tol
        self.n_iter_max = n_iter_max
        self.random_state = random_state
        self.n_initializations = n_initializations
        
        # initialize internal parameters
        self._best_cp_index = None
        
    @property  
    def decomposition_(self):
        if self._best_cp_index is None:
            raise AttributeError('The model has not yet been fit with data.')
        else:
            return self.candidates_[self._best_cp_index]
        
    @property  
    def loss_(self):
        if self._best_cp_index is None:
            raise AttributeError('The model has not yet been fit with data.')
        else:
            return self.candidate_losses_[self._best_cp_index]
        
    def fit_transform(
        self, 
        tensor, 
        threads=None, 
        verbose=0, 
        return_losses=False
    ):
        """Fits `n_initializations` sparse tensor decomposition models to the
        provided data tensor, using the als_lasso() method. Fit models are
        stored with the SparseCP object, and are accessible via the 
        SparseCP.decomposition_(), SparseCP.loss_(), SparseCP.candidates_(), 
        and SparseCP.candidate_losses_() properties.
        
        Parameters
        ----------
        tensor : numpy.ndarray
            Input data tensor.
        threads : int, default is None
            Maximum number of threads allocated to the algorithm. If 
            `threads` = None, then all available threads will be used.
        verbose : int, default is 0
            Level of verbosity.
        return_losses : bool, default is False
            Activate return of iteration loss values at each iteration.
            
        Returns
        -------
        cp_tensor : (weight, factors)
            * weights : 1D array of shape (rank,) that contains the weights 
                denoting the relative contributio of each factor.
            * factors : List of factors of the CP decomposition where factor 
                matrix `i` is of shape `(tensor.shape[i], rank)`
        losses : list
            A list of loss values calculated at each iteration of the algorithm. 
            Only returned when `return_losses` is set to True.
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
                norm_constraint=self.norm_constraint, 
                init=self.init, 
                tol=self.tol, 
                n_iter_max=self.n_iter_max, 
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
    