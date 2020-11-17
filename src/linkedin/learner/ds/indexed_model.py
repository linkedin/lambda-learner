import numpy as np
from scipy import sparse


class IndexedModel:
    """A Model represented in the index domain.

    This is the internal representation of a model used during optimization.
    """

    def __init__(self, theta: np.ndarray, hessian: sparse.spmatrix = None):
        self.theta = theta
        self.hessian = hessian

        if self.hessian is not None:
            # Theta and Hessian must have compatible shapes.
            assert len(theta) == hessian.shape[0]  # type: ignore
            assert len(theta) == hessian.shape[1]  # type: ignore
