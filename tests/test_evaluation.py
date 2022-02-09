import numpy as np
import contagion.evaluation as me


def test_top_percent():
    np.testing.assert_array_equal(me._top_percent(np.array([3, 1, 2, 5, 0]), 60), np.array([1, 0, 1, 1, 0]))


def test_top_percent_same():
    # Picks the first element if two elements have the same value
    np.testing.assert_array_equal(me._top_percent(np.array([3, 1, 2, 5, 2]), 60), np.array([1, 0, 0, 1, 1]))


def test_confusion_matrix_():
    actual = me.cm_sum_percent(np.array([[1, 0, 1, 1, 0], [1, 0, 1, 1, 0]]),
                               np.array([[3, 1, 2, 5, 0], [3, 1, 2, 5, 0]]), percentages=[60, 100])
    np.testing.assert_array_equal(np.array([[4., 0., 0., 6.], [0., 4., 0., 6.]]), actual)

