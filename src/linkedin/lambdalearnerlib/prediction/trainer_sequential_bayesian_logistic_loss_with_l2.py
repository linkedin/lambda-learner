import numpy as np
from scipy import sparse

from linkedin.lambdalearnerlib.ds.indexed_dataset import IndexedDataset
from linkedin.lambdalearnerlib.ds.indexed_model import IndexedModel

from .hessian_type import HessianType
from .trainer_lbgfs import TrainerLBGFS


class TrainerSequentialBayesianLogisticLossWithL2(TrainerLBGFS):
    """
    Implementation of sparse incremental logistic regression.
    See parent class for usage example.
    TODO: The current implementation only supports diagonal hessian update
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
        """
        Instantiate a Trainer, for a given dataset and model. Additional settings and hyperparams beyond
        those for TrainerLBGFS are:
        :param delta - The "forgetting factor". How much to weigh the past loss function approximation,
                       where 0 means completely disregard the past.
        """
        super().__init__(training_data=training_data, initial_model=initial_model, hessian_type=hessian_type, penalty=penalty)
        self.param_delta = delta

    def loss(self, theta: np.ndarray) -> float:
        penalty = theta.dot(theta) * self.param_reg * (1 - self.param_delta) / 2.0
        log_likelihood = -(self.data.w * np.log(np.exp(-self._affine_transform(theta)) + 1)).sum() - penalty
        deviation = theta - self.initial_model.theta
        # Beware: M * v and M.dot(v) are identical in implementation here...
        product = self.initial_model.hessian * deviation
        # ... but not identical here (or in general). Be extra careful to check any changes to these.
        past_loss = deviation.dot(product) * self.param_delta * 0.5
        return -log_likelihood + past_loss

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        penalty_prime = theta * self.param_reg * (1 - self.param_delta)
        denominator = np.exp(self._affine_transform(theta)) + 1
        log_likelihood_prime = ((self.data.y * self.data.w / denominator) * self.data.X) - penalty_prime
        deviation = theta - self.initial_model.theta
        past_gradient = self.initial_model.hessian * deviation * self.param_delta
        return -log_likelihood_prime + past_gradient

    def _update_full_hessian(self, new_theta: np.ndarray) -> sparse.spmatrix:
        return (
            self.initial_model.hessian * self.param_delta + self._estimate_hessian(new_theta) + (1 - self.param_delta) * self.param_lambda
        )
