import unittest

import numpy as np
from scipy import sparse

from linkedin.learner.ds.indexed_dataset import IndexedDataset
from linkedin.learner.ds.indexed_model import IndexedModel
from linkedin.learner.prediction.linear_scorer import score_linear_model
from linkedin.learner.utils.functions import sparse_diag_matrix


def mock_indexed_data():
    """Get a very small mock indexed dataset."""
    labels = np.array([1.0, -1.0, 1.0])
    weights = np.array([1, 1, 1])
    offsets = np.array([0.2, 0.3, 0.4])
    design_matrix = sparse.csc_matrix(np.array([[1.0, 1.0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0.5, 1.0, 1.3, 0.8]]))
    return IndexedDataset(design_matrix, labels, weights, offsets)


class LinearScorerTest(unittest.TestCase):
    def test_score_linear_model(self):
        """Test linear scoring in the general case."""
        theta = np.array([1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        hessian = sparse_diag_matrix([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6])
        indexed_model = IndexedModel(theta, hessian)

        indexed_data = mock_indexed_data()

        scores = score_linear_model(indexed_model, indexed_data)

        expected_scores = np.array(
            [0.2 + 1.0 * 1.0 + 1.0 * 0.1, 0.3 + 1.0 * 1.0, 0.4 + 1.0 * 1.0 + 0.5 * 0.3 + 1.0 * 0.4 + 1.3 * 0.5 + 0.8 * 0.6]
        )

        self.assertTrue(all(scores == expected_scores), f"Scores match expected scores. Actual {scores} == expected {expected_scores}.")

        scores_with_exploration = score_linear_model(indexed_model, indexed_data, exploration_threshold=1)

        self.assertTrue(
            all(scores_with_exploration != expected_scores),
            f"Scores with exploration don't exactly match expected scores. Actual {scores} == expected {expected_scores}.",
        )

    def test_score_linear_model_offsets_only(self):
        """Test linear scoring when all coefficients are zero."""
        num_features = 7
        theta = np.zeros(num_features)
        indexed_model = IndexedModel(theta)
        indexed_data = mock_indexed_data()

        scores = score_linear_model(indexed_model, indexed_data)

        expected_scores = np.array([0.2, 0.3, 0.4])

        self.assertTrue(
            all(scores == expected_scores),
            f"Offsets are used for scores when model coeffs are all zero/missing. Actual {scores} == expected {expected_scores}.",
        )
