from math import isclose

from linkedin.learner.ds.types import Name, NameTerm, Term, Value


class Feature:
    """Data structure to hold a single feature.

    The current implementation supports the flattened name-term-value format with additional
    metadata to hold an index value which is used to convert to and from a sparse representation.

    :param name: Feature section name, e.g. ads.creative.id
    :param term: Feature term name, e.g. 13126317
    :param value: Value (or score) associated with the given feature, e.g. categorical features will have 1.0
    """

    def __init__(self, name: Name, term: Term, value: Value):
        self.name: Name = name
        self.term: Term = term
        self.value: Value = value

    @property
    def name_term(self) -> NameTerm:
        return self.name, self.term

    def __str__(self):
        return "[name: {}, term: {}, value: {}]".format(self.name, self.term, self.value)

    def __eq__(self, other):
        return self.name == other.name and self.term == other.term and isclose(self.value, other.value)
