from ncistd import (
    simulated_sparse_tensor, 
    als_lasso
)
import numpy as np
import pytest
from pytest import approx
import scipy
import tlviz


# Arrange: simulated data
@pytest.fixture
def simulated_data():
    sim_tensor = simulated_sparse_tensor(
        shape=[10, 10, 10],                
        rank=5,                         
        densities=[.3, .3, .3], 
        factor_dist_list=[scipy.stats.uniform(loc=-1, scale=2), 
                          scipy.stats.uniform(), 
                          scipy.stats.uniform()], 
        random_state=1987
    )
    return sim_tensor


# Act: decompose tensor with als_lasso()
@pytest.fixture
def decomposition(simulated_data):
    # decompose simulated tensor
    cp, loss = als_lasso(
        tensor=simulated_data.to_tensor(
            noise_level=0, 
            sparse_noise=True, 
            random_state=None
        ), 
        rank=5, 
        lambdas=[0.1, 0.0, 0.0], 
        nonneg_modes=[1, 2], 
        init='random', 
        tol=1e-6, 
        n_iter_max=1000,  
        random_state=1987, 
        threads=None, 
        verbose=0, 
        return_losses=True
    )
    return cp, loss


def test_no_nans_in_cptensor(decomposition):
    cp, _ = decomposition
    nans = [np.any(np.isnan(f)) for f in cp.factors]
    assert not np.any(nans)
    
def test_no_nans_in_loss(decomposition):
    _, loss = decomposition
    assert not np.any(np.isnan(loss))
    
def test_loss_monotonically_decreasing(decomposition):
    _, loss = decomposition
    assert np.all(np.diff(loss) <= 0.0)
    
# def test_loss_matches_

# tests for every decomposition:
#   - all factors are floats
#   - loss is floats
#   - loss is monotonically decreasing
#   - last delta loss is less than tolerance for converged tensors
#   - last delta loss is greater than tolerance for non-converged tensors
#   - loss actually matches loss equation
#   - l2 norm of norm-constrained factors is 1
#   - correct factors (within tolerance) acheived with optimal settings



# small suite of tensors (4):
#    - different sizes
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

# special decomposition tests (testing SparseCP):
#   - max_iter=None, tol=1e-8 -> acheives convergence
#   - n_inits=10
#       - returned model actually has lowest loss
#       - all losses are different 
#   - same random seed results in same answers
#       - same factors between each initialization
#       - same losses between each initialization
#   - different random seed results in different answers

# decomposition with no sparsity penalty, nn constraint
# should be equivalent to standard CP decomposition by ALS 
# (with norm constraint?)

# # resources

# https://realpython.com/pytest-python-testing/
# https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#how-to-fixtures
# https://docs.pytest.org/en/7.1.x/reference/reference.html?highlight=tolerance
# http://tensorly.org/viz/stable/index.html
 