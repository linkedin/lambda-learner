import numpy as np
from scipy import sparse

from linkedin.lambdalearnerlib.ds.indexed_dataset import IndexedDataset
from linkedin.lambdalearnerlib.ds.indexed_model import IndexedModel

from .hessian_type import HessianType
from .trainer_lbgfs import TrainerLBGFS


class TrainerLogisticLossWithL2(TrainerLBGFS):
    """
    Implementation of sparse logistic regression.
    See parent class for usage example.
    """

    def __init__(
        self,
        *,  # Force all args to require keyword
        training_data: IndexedDataset,
        initial_model: IndexedModel,
        hessian_type: HessianType = HessianType.NONE,
        penalty: float = 0.0,
    ):
        """
        Instantiate a TrainerLogisticLossWithL2 trainer, for a given dataset and model.
        See TrainerLBGFS for documentation of hyperparams.
        """
        super().__init__(training_data=training_data, initial_model=initial_model, hessian_type=hessian_type, penalty=penalty)

    def loss(self, theta: np.ndarray) -> float:
        penalty = theta.dot(theta) * self.param_reg / 2.0
        log_likelihood = -(self.data.w * np.log(np.exp(-self._affine_transform(theta)) + 1)).sum() - penalty
        return -log_likelihood

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        penalty_prime = theta * self.param_reg
        numerator = sparse.diags(self.data.y * self.data.w, 0) * self.data.X
        denominator = np.exp(self._affine_transform(theta)) + 1
        log_likelihood_prime = np.asarray((sparse.diags(1.0 / denominator, 0) * numerator).sum(axis=0)).squeeze() - penalty_prime
        return -log_likelihood_prime

    def _update_full_hessian(self, new_theta: np.ndarray) -> sparse.spmatrix:
        return self._estimate_hessian(new_theta) + self.param_lambda
