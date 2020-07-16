import numpy as np
import src.timestamps as ts

start_time = 2
t1 = [np.array([1, 2]), np.array([]), np.array([1])]


def test_split_timestamps():
    actual_left, actual_right = ts.split_timestamps(t1, 2)
    expected_left = [np.array([1]), np.array([]), np.array([1])]
    expected_right = [np.array([2]), np.array([]), np.array([])]
    for a, b in zip(actual_left, expected_left):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(actual_right, expected_right):
        np.testing.assert_array_equal(a, b)


def test_get_infected_nodes_single():
    np.testing.assert_array_equal(ts.get_infected_nodes(t1, [1]), np.array([[1, 0, 1]]))


def test_get_infected_nodes_multi():
    np.testing.assert_array_equal(ts.get_infected_nodes(t1, [1, 3]), np.array([[1, 0, 1], [0, 0, 0]]))


def test_get_infected_nodes_range():
    np.testing.assert_array_equal(ts.get_infected_nodes(t1, range(2)), np.array([[0, 0, 0], [1, 0, 1]]))


def test_group_sort_timestamps():
    timestamps = np.array([1, 2, 3, 1])
    timestamps_nodes = np.array([0, 2, 0, 0])
    node_list = [2, 1, 0]
    expected = [np.array([2]), np.array([]), np.array([1, 1, 3])]
    actual = ts.group_sort_timestamps(timestamps, timestamps_nodes, node_list)
    for i, j in zip(expected, actual):
        np.testing.assert_array_equal(i, j)


def test_group_sort_timestamps_node_list_range():
    timestamps = np.array([1, 2, 3, 1])
    timestamps_nodes = np.array([0, 3, 0, 0])
    expected = [np.array([1, 1, 3]), np.array([]), np.array([]), np.array([2])]
    node_list = list(range(3))
    actual = ts.group_sort_timestamps(timestamps, timestamps_nodes, node_list)
    for i, j in zip(expected, actual):
        np.testing.assert_array_equal(i, j)
