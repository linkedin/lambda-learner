import unittest

import numpy as np
from prediction.fixtures import generate_mock_training_data, simple_mock_data
from test_utils import (PLACES_PRECISION, matrices_almost_equal,
                        sequences_almost_equal, ensure_str)

from linkedin.learner.ds.indexed_model import IndexedModel
from linkedin.learner.prediction.evaluator import evaluate
from linkedin.learner.prediction.hessian_type import HessianType
from linkedin.learner.prediction.linear_scorer import score_linear_model
from linkedin.learner.prediction.trainer_sequential_bayesian_logistic_loss_with_l2 import \
    TrainerSequentialBayesianLogisticLossWithL2


class TrainerSequentialBayesianLogisticLossWithL2Test(unittest.TestCase):
    def test_incremental_lr_loss_and_gradient(self):
        """Test the loss and gradient functions.

        The expected values in this test are set using this implementation, so this
        is a test against regression, rather than strictly a test of correctness.
        """
        indexed_data, _ = generate_mock_training_data()

        theta_0 = np.array([4, 100, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)
        initial_model = IndexedModel(theta=theta_0)
        #  Discrepancies with offline: (1) Labels are 0/1; in reality the data processing will set it to -1/+1.
        #  (2) Intercept gets regularized nearline but not offline, (3) This test should ideally only set initial_theta
        #  without changing hist_theta, but current implementation does not allow that.

        lr = TrainerSequentialBayesianLogisticLossWithL2(training_data=indexed_data, initial_model=initial_model, penalty=10, delta=0.8)
        initial_loss = lr.loss(theta_0)

        self.assertAlmostEqual(initial_loss, 112946.61089443817, PLACES_PRECISION, "Expect loss is calculated correctly.")

        initial_gradient = lr.gradient(theta_0)

        self.assertTrue(
            sequences_almost_equal(initial_gradient, [1006.0, 1198.0, 871.0, 871.0, 1000.0, 1000.0, 126.0, 126.0, 7.0, 7.0]),
            "Expect gradient is calculated correctly.",
        )

        final_model, final_loss, metadata = lr.train()

        self.assertTrue(
            sequences_almost_equal(
                final_model.theta,
                [
                    -13.37418015,
                    63.42582255,
                    -7.97896804,
                    -7.97896804,
                    -15.77418024,
                    -15.77418024,
                    -6.49519736,
                    -6.49519736,
                    0.29998522,
                    0.29998522,
                ],
            ),
            "Expect theta after optimization is correct.",
        )
        self.assertAlmostEqual(final_loss, 15104.873256903807, PLACES_PRECISION, "Expect lost after optimization is correct.")
        self.assertEqual(metadata["warnflag"], 0, "Expect convergence status is 0, meaning successful convergence.")
        self.assertEqual(metadata["nit"], 20, "Expect number of iterations is correct.")
        self.assertEqual(metadata["funcalls"], 28, "Expect number of function calls is correct.")
        self.assertEqual(ensure_str(metadata["task"]), "CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH", f"Expect convergence reason is as expected")

        # Score (on the training data)
        initial_scores = score_linear_model(model=initial_model, test_data=indexed_data)
        initial_auc = evaluate(metric_list=["auc"], y_scores=initial_scores, y_targets=indexed_data.y)["auc"]
        final_scores = score_linear_model(model=final_model, test_data=indexed_data)
        final_auc = evaluate(metric_list=["auc"], y_scores=final_scores, y_targets=indexed_data.y)["auc"]

        expected_initial_auc = 0.5290581162324649
        expected_final_auc = 0.7059118236472945

        self.assertAlmostEqual(
            initial_auc, expected_initial_auc, PLACES_PRECISION, f"Initial AUC: expected {expected_initial_auc} == actual {initial_auc}"
        )
        self.assertAlmostEqual(
            final_auc, expected_final_auc, PLACES_PRECISION, f"Final AUC: expected {expected_final_auc} == actual {final_auc}"
        )

    def test_incremental_lr_update_hessian(self):
        """Test the Hessian update when using a full Hessian."""
        indexed_data, model = simple_mock_data()
        lr = TrainerSequentialBayesianLogisticLossWithL2(
            training_data=indexed_data, initial_model=model, hessian_type=HessianType.FULL, penalty=10, delta=1.0
        )
        hessian = lr._update_hessian(model.theta)
        expected_hessian = np.array(
            [[1.076006603, 0.007826689, 0.167666585], [0.007826689, 1.003913344, 0.023480067], [0.167666585, 0.023480067, 1.382293306]]
        )
        self.assertTrue(
            matrices_almost_equal(hessian, expected_hessian),
            f"(Full) Hessian computation is correct. Actual {hessian} == Expected {expected_hessian}.",
        )

    def test_incremental_lr_update_hessian_diagonal(self):
        """Test the Hessian update when using a diagonal Hessian."""
        indexed_data, model = simple_mock_data()
        lr = TrainerSequentialBayesianLogisticLossWithL2(
            training_data=indexed_data, initial_model=model, hessian_type=HessianType.DIAGONAL, penalty=10, delta=1.0
        )
        hessian = lr._update_hessian(model.theta).toarray()
        expected_hessian = np.diag([1.076006603, 1.003913344, 1.382293306])
        self.assertTrue(
            matrices_almost_equal(hessian, expected_hessian),
            f"(Diagonal) Hessian computation is correct. Actual {hessian} == Expected {expected_hessian}.",
        )

    def test_incremental_lr_update_hessian_identity(self):
        """Test the Hessian update when using an identity Hessian."""
        indexed_data, model = simple_mock_data()
        lr = TrainerSequentialBayesianLogisticLossWithL2(
            training_data=indexed_data, initial_model=model, hessian_type=HessianType.IDENTITY, penalty=10, delta=1.0
        )
        hessian = lr._update_hessian(model.theta).toarray()
        expected_hessian = np.diag([1.0, 1.0, 1.0])
        self.assertTrue(
            matrices_almost_equal(hessian, expected_hessian),
            f"Hessian computation is correct. Actual {hessian} == Expected {expected_hessian}.",
        )

    def test_hessian_computation_is_optional(self):
        """Test that the Hessian update is optional."""
        indexed_data, model = simple_mock_data()

        trainer = TrainerSequentialBayesianLogisticLossWithL2(
            training_data=indexed_data, initial_model=model, penalty=10, hessian_type=HessianType.NONE
        )
        final_model, _, _ = trainer.train(precision=1e4, num_corrections=7, max_iterations=50)

        self.assertTrue(final_model.hessian is None, "Hessian computation optional.")
