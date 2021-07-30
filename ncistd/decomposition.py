import tensorly as tl 
from tensorly.cp_tensor import CPTensor

# class SparseCP:
#     def __init__(self, rank, init='random'):
#         self.rank = rank
#         self.init = init
        
#     def fit_transform(self, X):
#         self.decomposition_ = modified_als(X, rank=self.rank, init=self.init)
#         return self.decomposition_
    
#     def __repr__(self):
#         return '{} decomposition of rank {}'.format(self.__class__.__name__, self.rank)
    

# def als_lasso(tensor, mask, rank, n_iter_max=100, init='random', ):
#     """
#     Algorithm for solving l1-penalized tensor decomposition problem
#     """
#     decomposition = X * rank
#     return decomposition


# strategy: 
#     - recapitulate ALS using TAM implementation
#     - add l1 penalty 
#     - figure out SGD
#     - redo overlapping block model using CP class
