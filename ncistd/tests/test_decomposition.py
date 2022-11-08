from ncistd import (
    simulated_sparse_tensor, 
    als_lasso
)
import numpy as np
from numpy.testing import (
    assert_, 
    assert_array_almost_equal
)
import pytest
import scipy
from tlviz.factor_tools import factor_match_score


# arrange simulated data
@pytest.fixture(
    scope='module', 
    params=[
        {'shape': [10, 10, 10], 'rank': 5, 'density': 0.3, 'nonneg': [1, 2]}, 
        # {'shape': [100, 10, 10], 'rank': 5, 'density': 0.3, 'nonneg': [1, 2]}, 
        # {'shape': [10, 100, 10], 'rank': 1, 'density': 0.3, 'nonneg': [1, 2, 3]}, 
        # {'shape': [10, 10, 100], 'rank': 10, 'density': 1.0, 'nonneg': []}, 
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
        random_state=1987
    )
    return sim_tensor, params

# @pytest.mark.parametrize('noise_level', [0.0])
# @pytest.mark.parametrize('rank', [5])
# @pytest.mark.parametrize('lambdas', [[.1, 0, 0]])
# @pytest.mark.parametrize('nonneg_modes', [[1, 2]])
@pytest.mark.parametrize('noise_level', [0.0, 1.0])
@pytest.mark.parametrize('rank', [1, 5, 10])
@pytest.mark.parametrize('lambdas', [[0, 0, 0], [.1, 0, 0], [.1, .1, 0], [.1, .1, .1], [1, 0, 0]])
@pytest.mark.parametrize('nonneg_modes', [[], [1, 2], [0, 1, 2]])
@pytest.mark.parametrize('tol', [1e-6])
@pytest.mark.parametrize('n_iter_max', [100])
@pytest.mark.parametrize('random_seed', [1987])
def test_als_lasso(
    simulated_data, 
    noise_level, 
    rank, 
    lambdas, 
    nonneg_modes, 
    tol, 
    n_iter_max, 
    random_seed
):
    # get simulated data and parameters from fixture
    sim_tensor, sim_params = simulated_data
    # decompose simulated tensor with given parameters
    cp, loss = als_lasso(
        tensor=sim_tensor.to_tensor(
            noise_level=noise_level, 
            sparse_noise=True, 
            random_state=random_seed
        ), 
        rank=rank, 
        lambdas=lambdas, 
        nonneg_modes=nonneg_modes, 
        init='random', 
        tol=tol, 
        n_iter_max=n_iter_max,  
        random_state=random_seed, 
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
    # TODO: Finish this
    
    # check that loss is monotonically decreasing
    delta_loss = np.diff(loss)
    assert_(np.all(delta_loss <= 0.0), 'Loss is not monotonically decreasing.')
    
    # check convergence criteria
    if len(loss) < n_iter_max + 1:
        # decomposition converged: check that final delta loss is less than tolerance
        assert_(delta_loss[-1] < tol, 'Algorithm stopped iterating prematurely.')
    else:
        # TODO: catch warning -- it's possible that it converges exactly at n_iter_max
        # decomposition didn't converge: 
        # check that final delta loss is greater than tolerance
        assert_(delta_loss[-1] >= tol, 'Algorithm failed to stop iterating after convergence.')
    
    # check factor properties
    for i, factor in enumerate(cp.factors):
        # check that nonnegative mode factors are nonnegative
        if i in nonneg_modes:
            assert_(
                np.all(factor >= 0.0), 
                'Factor matrix {} did not meet nonnegativity constraint'.format(i)
            )
        # # check that l2 norm of unit norm constrained factors is = 1
        # if lambdas[i] == 0:
        #     # TODO: change this to incorporate tolerance
        #     assert_(
        #         np.all(np.linalg.norm(factor, 2, axis=0) == 1), 
        #         'Factor matrix {} did not meet unit 2-norm constraint'.format(i)
        #     )
    
    # TODO: Find optimal sparsity penalties and limits for these test tensors
    # for optimal decomposition parameters
    # if rank == sim_params['rank'] and nonneg_modes == sim_params['nonneg'] and 0.1 in lambdas:
        # check difference between simulated data and reconstruction is within tolerance
        # check difference between true factors and learned factors is within tolerance
        # check factor match score is within tolerance
    
    # check that same random seeding yeilds same results
    second_cp, second_loss = als_lasso(
        tensor=sim_tensor.to_tensor(
            noise_level=noise_level, 
            sparse_noise=True, 
            random_state=random_seed
        ), 
        rank=rank, 
        lambdas=lambdas, 
        nonneg_modes=nonneg_modes, 
        init='random', 
        tol=tol, 
        n_iter_max=n_iter_max,  
        random_state=random_seed, 
        threads=None, 
        verbose=0, 
        return_losses=True
    )
    assert_array_almost_equal(loss, second_loss)
    assert_array_almost_equal(cp.factors[0], second_cp.factors[0])
    assert_array_almost_equal(cp.factors[1], second_cp.factors[1])
    assert_array_almost_equal(cp.factors[2], second_cp.factors[2])


# Testing outline

# small suite of tensors (4):
#    - different shapes
#       - [10, 10, 10], [100, 10, 10], [10, 100, 10], [10, 10, 100]
#    - different ranks
#       - 5, 5, 1, 10
#    - different sparsities
#       - 0.3, 0.3, 0.3, 1
#    - different nn patterns
#       - [1, 2], [1, 2], [1, 2, 3], []

# a suite of different decompositions (testing als_lasso):
#    - noisy and non-noisy
#    - correct rank, under rank, over rank
#       - 1, 5, 10
#    - no sparsity, optimal sparsity, too much sparsity
#       - [0, 0, 0], [.1, 0, 0], [.1, .1, 0], [.1, .1, .1], [10, 0, 0]
#    - no nn, correct nn, too much nn
#       - [], [1, 2], [1, 2, 3]
#    - max_iter=100, tol=1e-6, n_inits=1

# tests for every decomposition:
#   - all factors are floats
#   - loss is floats
#   - loss is monotonically decreasing
#   - last delta loss is less than tolerance for converged tensors
#   - last delta loss is greater than tolerance for non-converged tensors
#   - loss actually matches loss equation
#   - l2 norm of norm-constrained factors is 1
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
