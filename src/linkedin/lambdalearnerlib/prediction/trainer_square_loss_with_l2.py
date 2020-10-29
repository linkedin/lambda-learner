import numpy as np
from scipy import sparse

from linkedin.lambdalearnerlib.ds.indexed_dataset import IndexedDataset
from linkedin.lambdalearnerlib.ds.indexed_model import IndexedModel

from .hessian_type import HessianType
from .trainer import with_previous_theta_transform_memoized
from .trainer_lbgfs import TrainerLBGFS


class TrainerSquareLossWithL2(TrainerLBGFS):
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
        Instantiate a TrainerSquareLossWithL2 trainer, for a given dataset and model.
        See TrainerLBGFS for documentation of hyperparams.
        """
        super().__init__(training_data=training_data, initial_model=initial_model, hessian_type=hessian_type, penalty=penalty)

    @with_previous_theta_transform_memoized
    def _residuals(self, theta: np.array) -> np.ndarray:
        """
        Computes residuals, caching the previous result. These are used in both the loss and gradient computation.
        """
        return self.data.X * theta + self.data.offsets - self.data.y

    def loss(self, theta: np.ndarray) -> float:
        """
        loss = 1/2 sum(Xθ + offset - y)^2 + 1/2 λθᵀθ
        """
        residuals = self._residuals(theta)
        prediction_error = 0.5 * residuals.dot(residuals)
        penalty = 0.5 * self.param_reg * theta.dot(theta)
        return prediction_error + penalty

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """
        Derivation from loss:

        Using identities:
            ∂/∂x(xᵀMx) = (M + Mᵀ)x. And if M symmetric, ∂/∂x(xᵀMx) = 2Mx.
            ∂/∂y(xᵀy) = x
            ∂/∂x(xᵀy) = y

        We have:
        ∂/∂θ[(Xθ + b)^2] = ∂/∂θ(θᵀXᵀXθ + bᵀXθ + θᵀXᵀb + bᵀb) = 2XᵀXθ + 2Xᵀb = 2Xᵀ(Xθ + b)
        where b = offset - y.

        Therefore:
        gradient = Xᵀ(Xθ + offset - y) + λθ
        """
        residuals = self._residuals(theta)
        grad_prediction_error = self.data.X.T * residuals
        grad_penalty = self.param_reg * theta
        return grad_prediction_error + grad_penalty

    def _update_full_hessian(self, new_theta: np.ndarray) -> sparse.spmatrix:
        return self._estimate_hessian(new_theta)
