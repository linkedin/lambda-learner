import time
from itertools import chain
from math import isnan
from typing import Any, Iterable, List

from scipy import sparse


def flatten(list_of_lists: Iterable[Iterable[Any]]) -> Iterable[Any]:
    """Flatten a list of lists into a shallow list."""
    return list(chain.from_iterable(list_of_lists))


def count_nans(num_list: Iterable[float]) -> int:
    """Count how many Nans appear in a list"""
    return len([x for x in num_list if isnan(x)])


def current_time_in_ms() -> int:
    return int(round(time.time() * 1000))


def get_public_attrs(obj: object):
    """
    :param obj - Any python object
    :return - A list of all the public attrs.
    """
    return [a for a in dir(obj) if not a.startswith("_") and not callable(getattr(obj, a))]


def represent_container_class(obj: object):
    """
    Represent an object as a string. Can be used as a function, or as an object's __repr__ method.
    This intended for use with classes whose public members are precisely the constructor args, as
    this is what the resulting string implies.It also works with subclasses of cfg2's AppDefPlugin.
    :param obj - Any python object
    :return - A string "MyClass(arg1=val_1, arg2=val_2, ...)
    """
    attribute_names = get_public_attrs(obj)
    attrs_string = ", ".join(f"{name}={getattr(obj, name)}" for name in attribute_names)
    return f"{obj.__class__.__name__}({attrs_string})"


def dedupe_preserve_order(seq: Iterable) -> List:
    """
    Order-stable deduplication for an ordered collection.
    :param seq: An ordered collection.
    :return: List with unique elements, in the same order as they appeared int he input collection.
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def sparse_diag_matrix(diagonal: Iterable) -> sparse.spmatrix:
    """
    Creates a sparse diagonal matrix with the given diagonal values.
    Note: see go/ll-matrix-performance

    :param diagonal: the diagonal values for the matrix.
    :return: A sparse diagonal matrix.
    """
    return sparse.diags([diagonal], [0]).tocsc()
