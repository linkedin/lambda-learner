import unittest
from unittest import mock
import uuid

import numpy as np

from linkedin.lambdalearnerlib.prediction.trainer import with_previous_theta_transform_memoized


class SubTermMemoizationFixture:
    """
    Fixture class for testing subterm caching. The key idea is that [[target_op]] will return the same value
    only on the same instance, and only if the previously computed result was cached and the uuid computation
    is bypassed.
    """
    def target_op(self, x: np.ndarray):
        return hash(self), uuid.uuid1()

    @with_previous_theta_transform_memoized
    def memoized_target_op(self, x: np.ndarray):
        return self.target_op(x)

    @with_previous_theta_transform_memoized
    def memoized_target_op_2(self, x: np.ndarray):
        return self.target_op(x)


class SubTermCachingTest(unittest.TestCase):

    def test_sub_term_caching_reuse_vs_compute_behavior(self):
        """
        Tests that subterm caching caches and reuses the result for precisely the previous ndarray arg,
        i.e. keeps only one result and does not accumulate past results.
        """
        obj = SubTermMemoizationFixture()
        x = np.array([1, 2, 3])

        with mock.patch.object(SubTermMemoizationFixture,
                               'target_op',
                               wraps=obj.target_op) as mock_target_op:
            self.assertEqual(mock_target_op.call_count, 0, "Zero calls initially.")
            obj.memoized_target_op(x)
            self.assertEqual(mock_target_op.call_count, 1)
            obj.memoized_target_op(x)
            self.assertEqual(mock_target_op.call_count, 1,
                             "Consecutive subsequent calls with same theta return cached result.")
            obj.memoized_target_op(x + 1)
            self.assertEqual(mock_target_op.call_count, 2,
                             "Subsequent calls with different theta result in recomputation.")
            obj.memoized_target_op(x + 1)
            self.assertEqual(mock_target_op.call_count, 2)
            obj.memoized_target_op(x)
            self.assertEqual(mock_target_op.call_count, 3,
                             "Subsequent calls with a previous theta result in recomputation"
                             "if it was not the most recent theta arg used.")

    def test_sub_term_caching_does_not_leak_between_instances(self):
        """
        This test guards against a bug where cache state leaks between instances through subterm caching. Calls
        to the cached method should not be cached across instances.
        """
        i1 = SubTermMemoizationFixture()
        i2 = SubTermMemoizationFixture()

        x = np.array([1, 2, 3])

        self.assertEqual(i1.memoized_target_op(x), i1.memoized_target_op(x),
                         "Consecutive same-instance same-x same-method calls: use memoized.")
        self.assertNotEqual(i1.memoized_target_op(x), i1.memoized_target_op_2(x),
                            "Consecutive same-instance same-x diff-method calls: (re)compute.")
        self.assertNotEqual(i1.memoized_target_op(x), i1.memoized_target_op(x + 1),
                            "Consecutive same-instance different-x calls: (re)compute.")
        self.assertNotEqual(i1.memoized_target_op(x), i2.memoized_target_op(x),
                            "Consecutive diff-instance same-x calls: (re)compute.")
