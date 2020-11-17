import enum


class HessianType(enum.Enum):
    """Enum of supported Hessian update types.

    FULL = Compute a full Hessian.
    DIAGONAL = Compute a diagonal Hessian approximation.
    IDENTITY = Always use an identity Hessian.
    NONE = Don't update the Hessian.
    """

    FULL = 0
    DIAGONAL = 1
    IDENTITY = 2
    NONE = 4
