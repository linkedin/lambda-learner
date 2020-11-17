import numpy as np
from data_generation import (read_ntv_records_from_json,
                             training_record_from_json)
from scipy import sparse

from linkedin.learner.ds.index_map import IndexMap
from linkedin.learner.ds.indexed_dataset import IndexedDataset
from linkedin.learner.ds.indexed_model import IndexedModel
from linkedin.learner.ds.representation_utils import \
    nt_domain_data_to_index_domain_data
from linkedin.learner.utils.functions import sparse_diag_matrix


def generate_mock_training_data(num_examples: int = -1):
    """Load large mock data and index it.

    :param num_examples: How many examples to load.
    :returns: A tuple of IndexedDataset, IndexMap
    """
    raw_data = [training_record_from_json(_) for _ in read_ntv_records_from_json(num_examples)]
    imap, _ = IndexMap.from_records_means_and_variances(raw_data, [], [])
    indexed_data = nt_domain_data_to_index_domain_data(raw_data, imap)
    return indexed_data, imap


def simple_mock_data():
    """Load a very small mock dataset.

    :returns: A tuple of IndexedDataset, IndexedModel
    """
    row = np.array([0, 0, 0, 1, 1, 1])
    col = np.array([0, 1, 2, 0, 1, 2])
    data = np.array([1, 0.5, 3, 0.5, 0, 1.0])
    X = sparse.coo_matrix((data, (row, col)), shape=(2, 3), dtype=np.float64)
    y = np.array([-1.0, 1.0])
    offsets = np.array([-0.25, 1.0])
    w = np.array([1.0, 1.0])
    indexed_data = IndexedDataset(X, y, w, offsets)

    theta = np.array([0.25, 0.75, -1.5])
    model = IndexedModel(theta=theta, hessian=sparse_diag_matrix([1, 1, 1]))

    return indexed_data, model
