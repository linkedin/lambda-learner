import numpy as np
from scipy import sparse

from linkedin.learner.ds.indexed_dataset import IndexedDataset
from linkedin.learner.ds.indexed_model import IndexedModel

from .hessian_type import HessianType
from .trainer_lbgfs import TrainerLBGFS


class TrainerLogisticLossWithL2(TrainerLBGFS):
    """Implementation of sparse logistic regression.

    Example Usage:

    lr_trainer = TrainerLogisticLossWithL2(
        training_data=training_data,
        initial_model=initial_model,
        penalty=10.0)
    updated_model, trained_loss, training_metadata = lr_trainer.train()
    """

    def __init__(
        self,
        *,  # Force all args to require keyword
        training_data: IndexedDataset,
        initial_model: IndexedModel,
        hessian_type: HessianType = HessianType.NONE,
        penalty: float = 0.0,
    ):
        """Instantiate a TrainerLogisticLossWithL2 trainer, for a given dataset and model.

        :param training_data: An indexed dataset to train the model on.
        :param initial_model: The model to be trained.
        :param hessian_type: How precise should the Hessian update be?
        :param penalty: Regularization penalty hyper-parameter.
        """
        super().__init__(training_data=training_data, initial_model=initial_model, hessian_type=hessian_type, penalty=penalty)

    def loss(self, theta: np.ndarray) -> float:
        """Compute the cross entropy loss for logistic regression.

        :param theta: The coefficient vector.
        :returns: Value of the loss at this point.
        """
        penalty = theta.dot(theta) * self.param_reg / 2.0
        log_likelihood = -(self.data.w * np.log(self._exp_affine(theta) + 1)).sum() - penalty
        return -log_likelihood

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """Compute the cross entropy loss gradient for logistic regression.

        :param theta: The coefficient vector.
        :returns: Value of the loss gradient at this point.
        """
        penalty_prime = theta * self.param_reg
        numerator = sparse.diags(self.data.y * self.data.w, 0) * self.data.X
        denominator = 1 / self._exp_affine(theta) + 1
        log_likelihood_prime = np.asarray((sparse.diags(1.0 / denominator, 0) * numerator).sum(axis=0)).squeeze() - penalty_prime
        return -log_likelihood_prime

    def _update_full_hessian(self, new_theta: np.ndarray) -> sparse.spmatrix:
        """Compute the updated coefficient Hessian value, post optimization.

        :param new_theta: The post-optimization coefficient vector.
        :returns: New value of the coefficient Hessian.
        """
        return self._estimate_hessian(new_theta) + self.param_lambda
