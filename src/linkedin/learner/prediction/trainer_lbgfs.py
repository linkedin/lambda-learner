from abc import abstractmethod
from copy import copy
from typing import Any, Dict, Tuple

import numpy as np
from scipy import sparse
from scipy.optimize import fmin_l_bfgs_b

from linkedin.learner.ds.indexed_dataset import IndexedDataset
from linkedin.learner.ds.indexed_model import IndexedModel
from linkedin.learner.utils.functions import sparse_diag_matrix

from .hessian_type import HessianType
from .trainer import Trainer, with_previous_theta_transform_memoized


class TrainerLBGFS(Trainer):
    """An trainer using LBGFS optimization and supporting coefficient Hessian update.

    This is an abstract trainer and must be sub-classed.
    """

    def __init__(
        self,
        *,  # Force all args to require keyword
        training_data: IndexedDataset,
        initial_model: IndexedModel,
        hessian_type: HessianType = HessianType.FULL,
        penalty: float = 0.0,
    ):
        """Instantiate a Trainer, for a given dataset and model.

        :param training_data: An indexed dataset to train the model on.
        :param initial_model: The model to be trained.
        :param hessian_type: How precise should the Hessian update be?
        :param penalty: Regularization penalty hyper-parameter.
        """
        super().__init__()

        # Set default values for theta and hessian if they are not provided.
        if initial_model.theta is None or initial_model.hessian is None:
            initial_model = copy(initial_model)
            if initial_model.theta is None:
                initial_model.theta = np.zeros(training_data.num_features)
            if initial_model.hessian is None:
                initial_model.hessian = sparse_diag_matrix(np.ones(training_data.num_features) * penalty)

        self.initial_model = initial_model
        self.data = training_data
        self.param_reg = penalty
        self.hessian_type = hessian_type

        lambda_vector = self.param_reg * np.ones(training_data.num_cols)
        self.param_lambda = sparse_diag_matrix(lambda_vector)

    @with_previous_theta_transform_memoized
    def _exp_affine(self, theta):
        """Compute a term used in the loss, gradient, and Hessian update.

        The most recent result is cached for efficiency, so that when LBGFS calls
        loss and gradient computations for a single point consecutively, this
        term can be reused.
        """
        return np.exp(-self.data.y * (self.data.X * theta + self.data.offsets))

    def _estimate_hessian(self, theta: np.ndarray) -> sparse.spmatrix:
        """Estimate posterior variances.

        We assume X is binary, and we care only about diagonal entries.

        :param theta: coefficient vector that is output from l-bfgs.
        :return: The Hessian diagonal vector.
        """
        score = 1.0 / (1 + self._exp_affine(theta))
        D = sparse_diag_matrix(score * (1 - score))
        X = self.data.X
        full_hessian = X.T * D * X
        return full_hessian

    @abstractmethod
    def _update_full_hessian(self, new_theta: np.array) -> sparse.spmatrix:
        """Update the posterior Hessian value.

        Used by `train` method to update the Hessian after training.

        :param new_theta: new value of theta after training.
        :return: The updated Hessian.
        """
        raise NotImplementedError

    def _update_hessian(self, new_theta: np.array) -> sparse.spmatrix:
        """Update the Hessian given the value of hessian_type.

        :param new_theta: The post-training model coefficient vector.
        :return: The updated Hessian.
        """
        if self.hessian_type == HessianType.NONE:
            hessian = None
        elif self.hessian_type == HessianType.IDENTITY:
            hessian = sparse_diag_matrix([1] * len(new_theta))
        elif self.hessian_type == HessianType.DIAGONAL:
            hessian = sparse_diag_matrix(self._update_full_hessian(new_theta).diagonal())
        elif self.hessian_type == HessianType.FULL:
            hessian = self._update_full_hessian(new_theta)
        else:
            raise ValueError(f"{self.hessian_type} is not a supported HessianType.")

        return hessian

    def train(
        self,
        max_iterations: int = 15000,
        max_funcalls: int = 15000,
        max_linesearch: int = 15000,
        num_corrections: int = 10,
        precision: float = 1e7,
        gradient_tolerance: float = 1e-5,
    ) -> Tuple[IndexedModel, float, Dict[str, Any]]:
        """Optimize the model coefficients and update the coefficient variances.

        Minimizes the loss function using LBFGS, and updates the hessian estimate.
        See fmin_l_bfgs_b for documentation of the params. Default values are the
        fmin_l_bfgs_b defaults.

        :return: Coefficient vector and Hessian matrix, posterior loss, and training metadata.
        """

        # Minimize a function func using the L-BFGS-B algorithm.
        result = fmin_l_bfgs_b(
            func=self.loss,
            x0=self.initial_model.theta,
            approx_grad=False,
            fprime=self.gradient,
            m=num_corrections,  # number of variable metrics corrections. default is 10.
            factr=precision,  # control precision, smaller the better. 1e7 is default.
            maxiter=max_iterations,
            disp=0,
        )

        theta = result[0]
        trained_loss = result[1]
        training_outcome_metadata = result[2]

        # The coefficients maybe corrupted (e.g. if line search failed).
        # In such cases we skip training and just return the initial values.
        # Refer: https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
        if training_outcome_metadata["warnflag"] == 2:
            return self.initial_model, trained_loss, training_outcome_metadata

        # Model coefficients
        hessian = self._update_hessian(theta)
        updated_model = IndexedModel(theta, hessian)

        return updated_model, trained_loss, training_outcome_metadata
