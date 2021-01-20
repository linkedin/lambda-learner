import numpy as np

# How precise to make "almost equal" comparisons
PLACES_PRECISION = 8
RELATIVE_PRECISION = 10e-6


def sequences_almost_equal(a, b, rel_precision: float = RELATIVE_PRECISION):
    """Test whether two sequences are uniformly pointwise different by at most a given factor.

    This is a test helper intended to be used with [[assertTrue]] in a [[unittest.TestCase]]
    """
    a_ndarray = np.array(a)
    b_ndarray = np.array(b)
    zero_adjustment = ((b_ndarray == 0) + 0) * (rel_precision / 1000)
    return all((abs(1 - (a_ndarray + zero_adjustment) / (b_ndarray + zero_adjustment)) < rel_precision).flatten())


def matrices_almost_equal(a, b, rel_precision: float = RELATIVE_PRECISION):
    """Test whether two matrices are uniformly pointwise different by at most a given factor.

    This is a test helper intended to be used with [[assertTrue]] in a [[unittest.TestCase]]
    """
    zero_adjustment = ((b == 0) + 0) * (rel_precision / 1000)
    return all((np.array(abs(1 - (a + zero_adjustment) / (b + zero_adjustment)) < rel_precision)).flatten())


def ensure_str(bytes_or_str, encoding: str = "utf-8"):
    """
    Ensures that an object which is either a string or bytes is treated as a string.
    """
    return str(bytes_or_str, encoding) if isinstance(bytes_or_str, bytes) else bytes_or_str
