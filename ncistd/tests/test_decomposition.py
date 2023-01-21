from ncistd import (
    simulated_sparse_tensor, 
    als_lasso,
    consolidate_cp
)
import numpy as np
from numpy.testing import (
    assert_, 
    assert_allclose, 
    assert_array_equal, 
    assert_array_almost_equal
)
import pytest
import scipy
import tensorly as tl
from tlviz.factor_tools import factor_match_score


@pytest.fixture
def seed():
    return 1991

@pytest.fixture
def rns(seed):
    return np.random.RandomState(seed)

@pytest.fixture
def tolerance():
    return 1e-6

@pytest.fixture
def n_iter_max():
    return 100

@pytest.fixture(
    scope='module', 
    params=[
        {'id': 0, 'shape': [10, 10, 10], 'rank': 5, 'density': 0.3, 'nonneg': [1, 2]}, 
        {'id': 1, 'shape': [100, 10, 10], 'rank': 5, 'density': 0.3, 'nonneg': [1, 2]}, 
        {'id': 2, 'shape': [10, 100, 10], 'rank': 1, 'density': 1.0, 'nonneg': [1, 2]}, 
        {'id': 3, 'shape': [10, 10, 100], 'rank': 10, 'density': 0.4, 'nonneg': []}, 
    ]
)
def simulated_data(request):
    params = request.param
    # assign distributions to draw from in each mode
    distribution_list = []
    for i in range(3):
        if i in params['nonneg']:
            # draw from nonnegative uniform distribution [0, 1]
            distribution_list.append(scipy.stats.uniform())
        else:
            # from from uniform distribution spanning [-1, 1]
            distribution_list.append(scipy.stats.uniform(loc=-1, scale=2))
    sim_tensor = simulated_sparse_tensor(
        shape=params['shape'],                
        rank=params['rank'],                         
        densities=[params['density'] for i in range(3)], 
        factor_dist_list=distribution_list, 
        random_state=1991
    )
    return sim_tensor, params


@pytest.mark.parametrize('noise_level', [0.0, 1.0])
@pytest.mark.parametrize('rank', [1, 5, 10])
@pytest.mark.parametrize('lambdas', [[0, 0, 0], [.05, 0, 0], [.05, .05, .05], [1, 0, 0]])
@pytest.mark.parametrize('nonneg_modes', [[], [1, 2], [0, 1, 2]])
def test_als_lasso(
    simulated_data, 
    noise_level, 
    rank, 
    lambdas, 
    nonneg_modes, 
    tolerance, 
    n_iter_max, 
    rns
):
    # get simulated data and parameters from fixture
    sim_tensor, sim_params = simulated_data
    # tensorize simulated data
    X = sim_tensor.to_tensor(
        noise_level=noise_level, 
        sparse_noise=True, 
        random_state=rns
    )
    # decompose simulated tensor with given parameters
    cp, loss = als_lasso(
        tensor=X, 
        rank=rank, 
        lambdas=lambdas, 
        nonneg_modes=nonneg_modes, 
        init='random', 
        tol=tolerance, 
        n_iter_max=n_iter_max,  
        random_state=rns, 
        threads=None, 
        verbose=0, 
        return_losses=True
    )
    
    # check that no loss values are NaNs
    assert_(not np.any(np.isnan(loss)), 'Loss contains NaN values.')
    
    # check that no factor values are NaNs
    nans = [np.any(np.isnan(f)) for f in cp.factors]
    assert_(not np.any(nans), 'CP factors contain NaN values.') 
    
    # check that no weight weights are NaNs
    assert_(not np.any(np.isnan(cp.weights)), 'CP weights contain NaN values.') 
    
    # check that loss matches objective function
    sse = tl.norm(X - cp.to_tensor())**2
    # calculate L1 sparsity penalties
    factor_l1_norms = np.array([tl.norm(f, 1) for f in cp.factors])
    penalties = tl.dot(np.array(lambdas), factor_l1_norms)
    # compute loss
    correct_loss = sse + penalties
    assert_allclose(loss[-1], correct_loss, atol=1e-12)
    
    # check that loss is monotonically decreasing (down to double float precision)
    delta_loss = np.diff(loss)
    assert_(np.all(delta_loss <= 1e-12), 'Loss is not monotonically decreasing.')
    
    # check convergence criteria
    if len(loss) < n_iter_max:
        # decomposition converged: check that final delta loss is less than tolerance
        assert_(delta_loss[-1] < tolerance, 'Algorithm stopped iterating prematurely.')
    
    # check factor properties of consolidated cp (fully zero components removed)
    consolidated_cp = consolidate_cp(cp)
    for i, factor in enumerate(consolidated_cp.factors):
        # check that nonnegative mode factors are nonnegative
        if i in nonneg_modes:
            assert_(
                np.all(factor >= 0.0), 
                'Factor matrix {} did not meet nonnegativity constraint'.format(i)
            )
        # check that l2 norm of unit norm constrained factors is = 1
        # only valid for models in which at least one mode had an l1 penalty applied
        if lambdas[i] == 0 and any(lambdas):
            # set the tolerance based on whether or not the algorithm converged 
            if len(loss) < n_iter_max:
                rtol = tolerance
            else:
                rtol = 0.1
            # check the l2 norm
            assert_allclose(
                np.linalg.norm(factor, axis=0), 
                np.ones(consolidated_cp.rank),  
                rtol=rtol, 
                err_msg='Factor matrix {} did not meet unit 2-norm constraint'.format(i)
            )
    
    # TODO: Find optimal sparsity penalties and limits for these test tensors
    # for optimal decomposition parameters
    # if rank == sim_params['rank'] and nonneg_modes == sim_params['nonneg'] and 0.1 in lambdas:
        # check difference between simulated data and reconstruction is within tolerance
        # check difference between true factors and learned factors is within tolerance
        # check factor match score is within tolerance
        
        # lambdas = [0.05, 0, 0] for all
        
        # 0::: l: 0.01, fit: 0.9999, fms: 0.70 (no noise)
        # 1::: l: 0.01, fit: 0.9999, fms: 0.985 (no noise)
        # 1::: l: 0.1, fit: 0.572, fms: 0.702 (1:1 noise)
        # 2::: l: , fit: , fms: ()
        # 3::: l: , fit: , fms: ()


@pytest.mark.parametrize('noise_level', [0.1])
@pytest.mark.parametrize('lambdas', [[.1, 0, 0]])
def test_als_lasso_random_seed(
    simulated_data, 
    noise_level, 
    lambdas, 
    tolerance, 
    n_iter_max, 
    seed
):
    # get simulated data and parameters from fixture
    sim_tensor, sim_params = simulated_data
    # decompose simulated tensor with given parameters
    cp, loss = als_lasso(
        tensor=sim_tensor.to_tensor(
            noise_level=noise_level, 
            sparse_noise=True, 
            random_state=seed
        ), 
        rank=sim_params['rank'], 
        lambdas=lambdas, 
        nonneg_modes=sim_params['nonneg'], 
        init='random', 
        tol=tolerance, 
        n_iter_max=n_iter_max,  
        random_state=seed, 
        threads=None, 
        verbose=0, 
        return_losses=True
    )
    # check that same random seeding yeilds same results
    second_cp, second_loss = als_lasso(
        tensor=sim_tensor.to_tensor(
            noise_level=noise_level, 
            sparse_noise=True, 
            random_state=seed
        ), 
        rank=sim_params['rank'], 
        lambdas=lambdas, 
        nonneg_modes=sim_params['nonneg'], 
        init='random', 
        tol=tolerance, 
        n_iter_max=n_iter_max,  
        random_state=seed, 
        threads=None, 
        verbose=0, 
        return_losses=True
    )
    assert_array_equal(loss, second_loss)
    assert_array_equal(cp.factors[0], second_cp.factors[0])
    assert_array_equal(cp.factors[1], second_cp.factors[1])
    assert_array_equal(cp.factors[2], second_cp.factors[2])


# Testing outline

# tests for every decomposition:
#   - loss actually matches loss equation
#   - correct factors (within tolerance) acheived with optimal settings

# tests for SparseCP interface:
#   - max_iter=None, tol=1e-8 -> acheives convergence
#   - n_inits=10
#       - returned model actually has lowest loss
#       - all losses are different 
#   - same random seed results in same answers
#       - same factors between each initialization
#       - same losses between each initialization
#   - different random seed results in different answers


# # resources

# https://realpython.com/pytest-python-testing/
# https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#how-to-fixtures
# https://docs.pytest.org/en/7.1.x/reference/reference.html?highlight=tolerance
# http://tensorly.org/viz/stable/index.html
