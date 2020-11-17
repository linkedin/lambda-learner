import numpy as np
from scipy import sparse


class IndexedDataset:
    """A dataset represented in the index domain.

    This is the internal representation of a dataset used during optimization.
    """

    def __init__(self, data: sparse.spmatrix, labels: np.ndarray, labels_weights: np.ndarray, offsets: np.ndarray):
        self.X = data
        self.y = labels
        self.w = labels_weights
        self.offsets = offsets
        self.num_rows = self.X.shape[0]
        self.num_cols = self.X.shape[1]
        self.num_examples = self.num_rows
        self.num_features = self.num_cols

        assert self.num_rows == len(labels)
        assert self.num_rows == len(labels_weights)
        assert self.num_rows == len(offsets)
