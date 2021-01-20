import unittest

import numpy as np
from prediction.fixtures import generate_mock_training_data, simple_mock_data
from test_utils import (PLACES_PRECISION, matrices_almost_equal,
                        sequences_almost_equal, ensure_str)

from linkedin.learner.ds.indexed_model import IndexedModel
from linkedin.learner.prediction.evaluator import evaluate
from linkedin.learner.prediction.hessian_type import HessianType
from linkedin.learner.prediction.linear_scorer import score_linear_model
from linkedin.learner.prediction.trainer_logistic_loss_with_l2 import \
    TrainerLogisticLossWithL2


class TrainerLogisticLossWithL2Test(unittest.TestCase):
    def test_lr_loss_and_gradient(self):
        """Test the loss and gradient functions.

        The expected values in this test are set using this implementation, so this
        is a test against regression, rather than strictly a test of correctness.
        """
        indexed_data, _ = generate_mock_training_data()

        theta_0 = np.array([4, 100, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)
        initial_model = IndexedModel(theta=theta_0)

        lr = TrainerLogisticLossWithL2(training_data=indexed_data, initial_model=initial_model, penalty=10)
        initial_loss = lr.loss(theta_0)

        self.assertAlmostEqual(initial_loss, 153042.61089443817, PLACES_PRECISION, "Expect loss is calculated correctly.")

        initial_gradient = lr.gradient(theta_0)

        self.assertTrue(
            sequences_almost_equal(initial_gradient, [1038.0, 1998.0, 879.0, 879.0, 1008.0, 1008.0, 134.0, 134.0, 15.0, 15.0]),
            "Expect gradient is calculated correctly.",
        )

        final_model, final_loss, metadata = lr.train()

        self.assertTrue(
            sequences_almost_equal(
                final_model.theta,
                [
                    -0.14982811627991494,
                    -0.14982824552049578,
                    -0.19636393879356379,
                    -0.19636393879356379,
                    -0.1498281122411468,
                    -0.1498281122411468,
                    0.04935191200563128,
                    0.04935191200563128,
                    -0.0028160881457262527,
                    -0.0028160881457262527,
                ],
            ),
            "Expect theta after optimization is correct.",
        )
        self.assertAlmostEqual(final_loss, 15.666764037573559, PLACES_PRECISION, "Expect lost after optimization is correct.")
        self.assertEqual(metadata["warnflag"], 0, "Expect convergence status is 0, meaning successful convergence.")
        self.assertEqual(metadata["nit"], 13, "Expect number of iterations is correct.")
        self.assertEqual(metadata["funcalls"], 18, "Expect number of function calls is correct.")
        self.assertEqual(ensure_str(metadata["task"]), "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL", f"Expect convergence reason is as expected")

        # Score (on the training data)
        initial_scores = score_linear_model(model=initial_model, test_data=indexed_data)
        initial_auc = evaluate(metric_list=["auc"], y_scores=initial_scores, y_targets=indexed_data.y)["auc"]
        final_scores = score_linear_model(model=final_model, test_data=indexed_data)
        final_auc = evaluate(metric_list=["auc"], y_scores=final_scores, y_targets=indexed_data.y)["auc"]

        expected_initial_auc = 0.5290581162324649
        expected_final_auc = 0.6137274549098197

        self.assertAlmostEqual(
            initial_auc, expected_initial_auc, PLACES_PRECISION, f"Initial AUC: expected {expected_initial_auc} == actual {initial_auc}"
        )
        self.assertAlmostEqual(
            final_auc, expected_final_auc, PLACES_PRECISION, f"Final AUC: expected {expected_final_auc} == actual {final_auc}"
        )

    def test_lr_update_hessian(self):
        """Test the Hessian update."""
        indexed_data, model = simple_mock_data()

        lr = TrainerLogisticLossWithL2(training_data=indexed_data, initial_model=model, penalty=10, hessian_type=HessianType.FULL)
        hessian = lr._update_full_hessian(model.theta)

        expected_hessian = np.array(
            [
                [10.076006603, 0.007826689207, 0.16766658587],
                [0.007826689207, 10.003913344, 0.023480067621],
                [0.16766658587, 0.023480067621, 10.382293306],
            ]
        )

        self.assertTrue(
            matrices_almost_equal(hessian, expected_hessian),
            f"Hessian computation is correct. Actual {hessian} == Expected {expected_hessian}.",
        )
