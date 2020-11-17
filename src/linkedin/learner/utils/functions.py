import time
from itertools import chain
from math import isnan
from typing import Any, Iterable, List, Set

from scipy import sparse

CONTAINER_CLASS_REPR_SEPARATOR = ", "
PRIVATE_MEMBER_PREFIX = "_"


def flatten(list_of_lists: Iterable[Iterable[Any]]) -> Iterable[Any]:
    """Flatten a list of lists into a shallow list."""
    return list(chain.from_iterable(list_of_lists))


def count_nans(num_list: Iterable[float]) -> int:
    """Count how many Nans appear in a list."""
    return len(list(x for x in num_list if isnan(x)))


def current_time_in_ms() -> int:
    """Get the current unix epoch time in milliseconds"""
    return int(round(time.time() * 1000))


def get_public_attrs(obj: object):
    """Get the public attributes of an object.

    :param obj: Any python object
    :return: A list of all the public attributes.
    """
    return [a for a in dir(obj) if not a.startswith(PRIVATE_MEMBER_PREFIX) and not callable(getattr(obj, a))]


def represent_container_class(obj: object):
    """Represent an object as a string in a generic way.

    Can be used as a function, or as an object's __repr__ method.

    This intended for use with classes whose public members are precisely the
    constructor args, as this is what the resulting string implies.

    :param obj: Any python object.
    :return: A string "MyClass(arg1=val_1, arg2=val_2, ...)".
    """
    attribute_names = get_public_attrs(obj)
    attrs_string = CONTAINER_CLASS_REPR_SEPARATOR.join(f"{name}={getattr(obj, name)}" for name in attribute_names)
    return f"{obj.__class__.__name__}({attrs_string})"


def dedupe_preserve_order(seq: Iterable) -> List:
    """Deduplicate the elements in an ordered collection, preserving order.

    :param seq: An ordered collection.
    :return: List with unique elements, with original order preserved.
    """
    seen: Set[Any] = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def sparse_diag_matrix(diagonal: Iterable) -> sparse.spmatrix:
    """Create a sparse diagonal matrix with the given diagonal values.

    :param diagonal: the diagonal values for the matrix.
    :return: A sparse diagonal matrix.
    """
    return sparse.diags([diagonal], [0]).tocsc()
