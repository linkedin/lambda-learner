import unittest
from itertools import permutations

import numpy as np
from scipy import sparse

from linkedin.learner.ds.feature import Feature
from linkedin.learner.ds.index_map import IndexMap
from linkedin.learner.ds.indexed_model import IndexedModel
from linkedin.learner.ds.record import TrainingRecord
from linkedin.learner.ds.representation_utils import (
    index_domain_coeffs_to_nt_domain_coeffs,
    nt_domain_coeffs_to_index_domain_coeffs,
    nt_domain_data_to_index_domain_data)
from linkedin.learner.utils.functions import sparse_diag_matrix


class RepresentationUtilsTest(unittest.TestCase):
    def test_nt_domain_data_to_index_domain_data(self):
        records = [
            TrainingRecord(1.0, 1.0, 0.2, [Feature("n1", "t1", 1.0)]),
            TrainingRecord(0.0, 1.0, 0.3, [Feature("n2", "t3", 0.0)]),
            TrainingRecord(
                1.0, 1.0, 0.4, [Feature("n3", "t1", 0.5), Feature("n3", "t2", 1.0), Feature("n3", "t3", 1.3), Feature("n3", "t4", 0.8)]
            ),
        ]

        imap, _ = IndexMap.from_records_means_and_variances(records, [], [])

        indexed_data = nt_domain_data_to_index_domain_data(records, imap)

        expected_y = np.array([1.0, -1.0, 1.0])
        expected_w = np.array([1, 1, 1])
        expected_offsets = np.array([0.2, 0.3, 0.4])

        self.assertTrue(all(indexed_data.y == expected_y), "Index-domain labels are correct.")
        self.assertTrue(all(indexed_data.w == expected_w), "Index-domain weights are correct.")
        self.assertTrue(all(indexed_data.offsets == expected_offsets), "Index-domain offsets are correct.")

        expected_sparse_matrix = sparse.csc_matrix(
            np.array([[1.0, 1.0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0.5, 1.0, 1.3, 0.8]])
        )

        num_mismatches = (indexed_data.X != expected_sparse_matrix).nnz

        self.assertEqual(num_mismatches, 0, "Index-domain design matrix is correct.")

    def test_nt_domain_coeffs_to_index_domain_coeffs(self):
        means = [Feature("n1", "t1", 0.1), Feature("n1", "t2", 0.2), Feature("n2", "t3", 0.3)]
        variances = [Feature("n1", "t1", 1.1), Feature("n1", "t2", 1.2), Feature("n2", "t3", 1.3)]
        imap, _ = IndexMap.from_records_means_and_variances([], means, variances)
        indexed_model = nt_domain_coeffs_to_index_domain_coeffs(means, variances, imap, regularization_param=0)

        expected_theta = np.array([0.0, 0.1, 0.2, 0.3])
        expected_hessian = sparse_diag_matrix([0.0, 1 / 1.1, 1 / 1.2, 1 / 1.3])

        self.assertTrue(
            all(indexed_model.theta == expected_theta),
            f"(Index-domain) theta is correct. Actual {indexed_model.theta} ==  expected {expected_theta}.",
        )
        self.assertTrue(
            all((indexed_model.hessian == expected_hessian).toarray().flatten()),
            f"(Index-domain) hessian is correct. Actual {indexed_model.hessian} ==  expected {expected_hessian}.",
        )

    def test_index_domain_coeffs_to_nt_domain_coeffs(self):
        means = [Feature("intercept", "intercept", 1.0), Feature("n1", "t1", 0.1), Feature("n1", "t2", 0.2), Feature("n1", "t3", 0.2)]
        variances = [Feature("intercept", "intercept", 2.0), Feature("n1", "t1", 1.1), Feature("n1", "t2", 1.2), Feature("n1", "t3", 1.3)]
        imap, _ = IndexMap.from_records_means_and_variances([], means, variances)
        theta = np.array([1.0, 0.1, 0.2, 0.3])
        hessian = sparse_diag_matrix([1 / 2.0, 1 / 1.1, 1 / 1.2, 1 / 1.3])
        indexed_model = IndexedModel(theta, hessian)
        mean_dict, variance_dict = index_domain_coeffs_to_nt_domain_coeffs(indexed_model, imap)

        expected_mean_dict = {("intercept", "intercept"): 1.0, ("n1", "t1"): 0.1, ("n1", "t2"): 0.2, ("n1", "t3"): 0.3}
        expected_variance_dict = {("intercept", "intercept"): 2.0, ("n1", "t1"): 1.1, ("n1", "t2"): 1.2, ("n1", "t3"): 1.3}

        self.assertEqual(mean_dict, expected_mean_dict, "(Index-domain) theta is correct.")
        self.assertEqual(variance_dict, expected_variance_dict, "(Index-domain) hessian is correct.")

    def test_nt_index_domain_full_cycle(self):
        features = [Feature("intercept", "intercept", 1.0), Feature("n1", "t1", 0.1), Feature("n1", "t2", 0.2), Feature("n1", "t3", 0.3)]

        expected_coeff_dict = {("intercept", "intercept"): 1.0, ("n1", "t1"): 0.1, ("n1", "t2"): 0.2, ("n1", "t3"): 0.3}

        feature_permutations = list(permutations(features))

        # Test all permutations, to ensure correctness is independent of the particular mapping the imap produces.
        for ordered_features in feature_permutations:
            means, variances = ordered_features, ordered_features
            imap, _ = IndexMap.from_records_means_and_variances([], means, variances)
            indexed_model = nt_domain_coeffs_to_index_domain_coeffs(means, variances, imap, regularization_param=0)
            mean_dict, variance_dict = index_domain_coeffs_to_nt_domain_coeffs(indexed_model, imap)

            self.assertEqual(
                mean_dict, expected_coeff_dict, f"(Index-domain) theta is correct, given feature ordering: {ordered_features}."
            )
            self.assertEqual(
                variance_dict, expected_coeff_dict, f"(Index-domain) hessian is correct, given feature ordering: {ordered_features}."
            )
