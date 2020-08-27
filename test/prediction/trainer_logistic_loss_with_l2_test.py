import unittest

import numpy as np

from linkedin.lambdalearnerlib.prediction.hessian_type import HessianType
from linkedin.lambdalearnerlib.prediction.trainer_logistic_loss_with_l2 import TrainerLogisticLossWithL2
from prediction.fixtures import simple_mock_data
from test_utils import matrices_almost_equal


class TrainerLogisticLossWithL2Test(unittest.TestCase):
    def test_logistic_regression(self):
        # TODO ktalanin - Write these tests.
        pass

    def test_lr_update_hessian(self):
        indexed_data, model, theta = simple_mock_data()

        lr = TrainerLogisticLossWithL2(training_data=indexed_data, initial_model=model, penalty=10, hessian_type=HessianType.FULL)
        hessian = lr._update_full_hessian(theta)

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
