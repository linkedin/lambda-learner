import numpy as np
from scipy import sparse

from data_generation import from_offline_training_example_avro, read_ntv_records_from_json
from linkedin.lambdalearnerlib.ds.index_map import IndexMap
from linkedin.lambdalearnerlib.ds.indexed_dataset import IndexedDataset
from linkedin.lambdalearnerlib.ds.indexed_model import IndexedModel
from linkedin.lambdalearnerlib.ds.representation_utils import nt_domain_data_to_index_domain_data
from linkedin.lambdalearnerlib.utils.functions import sparse_diag_matrix


def generate_mock_training_data(num_examples: int = -1):
    raw_data = [from_offline_training_example_avro(_) for _ in read_ntv_records_from_json(num_examples)]
    imap, _ = IndexMap.from_records_means_and_variances(raw_data, [], [])
    indexed_data = nt_domain_data_to_index_domain_data(raw_data, imap)
    return indexed_data, imap


def simple_mock_data():
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

    return indexed_data, model, theta
