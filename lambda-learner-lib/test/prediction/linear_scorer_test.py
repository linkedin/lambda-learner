import unittest

import numpy as np
from scipy import sparse

from linkedin.lambdalearnerlib.ds.indexed_dataset import IndexedDataset
from linkedin.lambdalearnerlib.ds.indexed_model import IndexedModel
from linkedin.lambdalearnerlib.prediction.linear_scorer import score_linear_model
from linkedin.lambdalearnerlib.utils.functions import sparse_diag_matrix


def mock_indexed_data():
    labels = np.array([1.0, -1.0, 1.0])
    weights = np.array([1, 1, 1])
    offsets = np.array([0.2, 0.3, 0.4])
    design_matrix = sparse.csc_matrix(np.array([[1.0, 1.0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0.5, 1.0, 1.3, 0.8]]))
    return IndexedDataset(design_matrix, labels, weights, offsets)


class LinearScorerTest(unittest.TestCase):
    def test_score_linear_model(self):
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

        # TODO ktalanin - Ideally we'd test that the exploration follows the expected distribution, but I don't know
        #                 of a way to do this in finite time without a non-zero chance of false positive test failure.

    def test_score_linear_model_offsets_only(self):
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
