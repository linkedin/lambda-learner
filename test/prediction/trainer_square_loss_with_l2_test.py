import unittest

import numpy as np
from data_generation import (read_expected_movie_lens_training_result,
                             read_movie_lens_data)
from prediction.fixtures import simple_mock_data
from test_utils import matrices_almost_equal, sequences_almost_equal

from linkedin.learner.ds.indexed_model import IndexedModel
from linkedin.learner.prediction.evaluator import evaluate
from linkedin.learner.prediction.hessian_type import HessianType
from linkedin.learner.prediction.linear_scorer import score_linear_model
from linkedin.learner.prediction.trainer_square_loss_with_l2 import \
    TrainerSquareLossWithL2


class TrainerSquareLossWithL2Test(unittest.TestCase):
    def test_square_loss_with_l2_loss_and_gradient(self):
        """Test the loss and gradient functions.

        The expected values in this test are set using this implementation, so this
        is a test against regression, rather than strictly a test of correctness.
        """
        training_data, test_data = read_movie_lens_data()

        theta_0 = np.zeros(training_data.num_features)
        initial_model = IndexedModel(theta=theta_0)

        trainer = TrainerSquareLossWithL2(training_data=training_data, initial_model=initial_model, penalty=10)
        final_model, final_loss, metadata = trainer.train(precision=1e4, num_corrections=7, max_iterations=50)

        actual_scores = score_linear_model(model=final_model, test_data=test_data)
        final_rmse = evaluate(metric_list=["rmse"], y_scores=actual_scores, y_targets=test_data.y)["rmse"]

        expected_theta = read_expected_movie_lens_training_result()
        expected_model = IndexedModel(theta=expected_theta)
        expected_scores = score_linear_model(expected_model, test_data)
        expected_rmse = evaluate(metric_list=["rmse"], y_scores=expected_scores, y_targets=test_data.y)["rmse"]

        self.assertAlmostEqual(expected_rmse, final_rmse)

        self.assertTrue(
            sequences_almost_equal(expected_theta, final_model.theta),
            f"Theta as expected: Expected = {expected_theta}, Actual = {final_model.theta}",
        )

    def test_square_loss_with_l2_update_hessian(self):
        """Test the Hessian update."""
        indexed_data, model = simple_mock_data()

        lr = TrainerSquareLossWithL2(training_data=indexed_data, initial_model=model, penalty=10, hessian_type=HessianType.FULL)
        hessian = lr._update_full_hessian(model.theta)

        expected_hessian = np.array(
            [
                [0.0760066037, 0.00782668920, 0.1676665858],
                [0.00782668920, 0.00391334460, 0.02348006762],
                [0.1676665858, 0.02348006762, 0.382293306],
            ]
        )

        self.assertTrue(
            matrices_almost_equal(hessian, expected_hessian),
            f"Hessian computation is correct. Actual {hessian} == Expected {expected_hessian}.",
        )
