from ncistd import (
    simulated_sparse_tensor, 
    als_lasso
)
import pytest
from pytest import approx
import scipy
import tlviz

# simulated data fixture
@pytest.fixture
def simulated_data():
    sim_tensor = simulated_sparse_tensor(
        shape=[50, 30, 20],                
        rank=10,                         
        densities=[.2, .2, .5], 
        factor_dist_list=[scipy.stats.uniform(loc=-1, scale=2), 
                          scipy.stats.uniform(), 
                          scipy.stats.uniform()], 
        random_state=1987
    )
    return sim_tensor


# best of 10 decompositions fixture
@pytest.fixture
def best_decomposition(simulated_data):
    # assign simulated cp data
    cp_sim = simulated_data
    # parameters
    sparsity = [0.0001, 0.0, 0.0]
    n_components = cp_sim.rank
    nonnegativity = [1, 2]
    # decompose
    lowest_err = float('inf')
    # multiple initializations
    for i in range(10):
        # fit model
        cp_i, err = als_lasso(
            cp_sim.to_tensor(), 
            rank=n_components, 
            lambdas=sparsity, 
            nonneg_modes=nonnegativity, 
            random_state=i, 
            mttkrp_optimization='opt_einsum', 
            return_losses=True
        )
        # keep best fit
        if err[-1] < lowest_err:
            lowest_err = err[-1]
            best_cp = cp_i
            best_err = err
    return cp_sim, best_cp


def test_correct_factors(best_decomposition):
    # get simulated data and decomposition
    cp_sim, cp_fit = best_decomposition
    # calculated fms
    fms = tlviz.factor_tools.factor_match_score(cp_sim, cp_fit)
    # check that the fms is almost identical
    assert fms == approx(0.9999997)

# TODO: Make decomposition test class
#   Within this the data can be generated once, and the decomposition 
#   can be called once. Then the output can be tested in multiple ways.


# decomposition with no sparsity penalty should be equivalent to standard
# CP decomposition by ALS

# decomposition 