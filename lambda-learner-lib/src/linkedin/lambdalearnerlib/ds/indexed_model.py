import numpy as np
from scipy import sparse


class IndexedModel:
    def __init__(self, theta: np.ndarray, hessian: sparse.spmatrix = None):
        self.theta = theta
        self.hessian = hessian

        if self.hessian is not None:
            # Theta and hessian must have compatible shapes.
            assert len(theta) == hessian.shape[0]
            assert len(theta) == hessian.shape[1]
