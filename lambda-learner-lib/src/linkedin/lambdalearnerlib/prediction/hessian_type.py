import enum


class HessianType(enum.Enum):
    FULL = 0
    DIAGONAL = 1
    IDENTITY = 2
    NONE = 4
