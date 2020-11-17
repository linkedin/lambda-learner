import numpy as np
from scipy import sparse

from linkedin.learner.ds.indexed_dataset import IndexedDataset
from linkedin.learner.ds.indexed_model import IndexedModel

from .hessian_type import HessianType
from .trainer_lbgfs import TrainerLBGFS


class TrainerSequentialBayesianLogisticLossWithL2(TrainerLBGFS):
    """Implementation of sparse incremental Bayesian logistic regression.

    The current implementation only supports diagonal Hessian update.

    This algorithm is described in the paper: "Lambda Learner: Fast Incremental
    Learning on Data Streams". See https://arxiv.org/abs/2010.05154 for details.

    Example Usage:

    lr_trainer = TrainerSequentialBayesianLogisticLossWithL2(
        training_data=training_data,
        initial_model=initial_model,
        hessian_type=HessianType.FULL,
        penalty=10.0,
        delta=0.8)
    updated_model, trained_loss, training_metadata = lr_trainer.train()
    """

    def __init__(
        self,
        *,  # Force all args to require keyword
        training_data: IndexedDataset,
        initial_model: IndexedModel,
        hessian_type: HessianType = HessianType.FULL,
        penalty: float = 0.0,
        delta: float = 0.8,  # default value based on offline simulation framework experiments
    ):
        """Instantiate a TrainerSequentialBayesianLogisticLossWithL2, for a given dataset and model.

        :param training_data: An indexed dataset to train the model on.
        :param initial_model: The model to be trained.
        :param hessian_type: How precise should the Hessian update be?
        :param penalty: Regularization penalty hyper-parameter.
        :param delta: The "forgetting factor". How much to weigh the past loss function
                      approximation, where 0 means completely disregard the past.
        """
        super().__init__(training_data=training_data, initial_model=initial_model, hessian_type=hessian_type, penalty=penalty)
        self.param_delta = delta

    def loss(self, theta: np.ndarray) -> float:
        """Compute the incremental Bayesian loss for logistic regression.

        :param theta: The coefficient vector.
        :returns: Value of the loss at this point.
        """
        penalty = theta.dot(theta) * self.param_reg * (1 - self.param_delta) / 2.0
        log_likelihood = -(self.data.w * np.log(self._exp_affine(theta) + 1)).sum() - penalty
        deviation = theta - self.initial_model.theta
        # Beware: M * v and M.dot(v) are identical in implementation here...
        product = self.initial_model.hessian * deviation
        # ... but not identical here (or in general). Be extra careful to check any changes to these.
        past_loss = deviation.dot(product) * self.param_delta * 0.5
        return -log_likelihood + past_loss

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """Compute the incremental Bayesian loss gradient for logistic regression.

        :param theta: The coefficient vector.
        :returns: Value of the loss gradient at this point.
        """
        penalty_prime = theta * self.param_reg * (1 - self.param_delta)
        denominator = 1 / self._exp_affine(theta) + 1
        log_likelihood_prime = ((self.data.y * self.data.w / denominator) * self.data.X) - penalty_prime
        deviation = theta - self.initial_model.theta
        past_gradient = self.initial_model.hessian * deviation * self.param_delta
        return -log_likelihood_prime + past_gradient

    def _update_full_hessian(self, new_theta: np.ndarray) -> sparse.spmatrix:
        """Compute the updated coefficient Hessian value, post optimization.

        :param new_theta: The post-optimization coefficient vector.
        :returns: New value of the coefficient Hessian.
        """
        return (
            self.initial_model.hessian * self.param_delta  # type: ignore
            + self._estimate_hessian(new_theta)
            + (1 - self.param_delta) * self.param_lambda
        )
