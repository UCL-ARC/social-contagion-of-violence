import numpy as np
import src.shuffle as st
import pytest

data_diff_all = [([1], [2], [1]),
                 (np.array([1]), np.array([2]), np.array([1])),
                 ([1], [1, 2], [0, 1]),
                 ([1, 4], [1, 2], [0, 1, 3, 2]),
                 ([1], [], []),
                 ([], [], [])]


@pytest.mark.parametrize("v1,v2,expected", data_diff_all, ids=str)
def test_diff_vectors(v1, v2, expected):
    np.testing.assert_array_almost_equal(st.diff_vectors(v1, v2), expected)


@pytest.mark.parametrize("v,expected", [([1, 4, 6], [3, 5, 2]), ([], [])], ids=str)
def test_diff_vector(v, expected):
    np.testing.assert_array_almost_equal(st.diff_vector(v), expected)


adj = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]  # represents a graph with two nodes where node 1 is connected to node 0 and 2
t0 = [[], [2], []]  # represents one event occurring at t=1 at node 1
t1 = [[1], [2], []]  # represents two events occurring at t=1 at node 0 and t=2 at node 1
t2 = [[1], [], [2]]  # represents two events occurring at t=1 at node 0 and t=2 at node 2
t4 = [np.array([]), np.array([2, 3, 4, 5]), np.array([1, ])]


@pytest.mark.parametrize("timestamps,expected", [(t0, []), (t1, [1]), (t2, [])], ids=str)
def test_diff_neighbours(timestamps, expected):
    np.testing.assert_array_almost_equal(st.diff_neighbours(adj, timestamps), np.array(expected))


@pytest.mark.parametrize("timestamps,expected", [(t0, []), (t1, [1]), (t2, [])], ids=str)
def test_diff_neighbours_self_excitation(timestamps, expected):
    np.testing.assert_array_almost_equal(st.diff_neighbours(adj + np.eye(3), timestamps), np.array(expected))


def test_group_timestamps_by_node():
    timestamps = np.array([1, 2, 1])
    timestamps_nodes = np.array(['a', 'c', 'a'])
    node_list = ['c', 'b', 'a']
    expected = [np.array([2]), np.array([]), np.array([1, 1])]
    actual = st.group_timestamps_by_node(timestamps, timestamps_nodes, node_list)
    for i, j in zip(expected, actual):
        np.testing.assert_array_almost_equal(i, j)


def test_shuffle_timestamps():
    expected = [np.array([3., 4.]), np.array([]), np.array([1., 2., 5.])]
    actual = st.shuffle_timestamps(t4, seed=1)
    for i, j in zip(expected, actual):
        np.testing.assert_array_almost_equal(i, j)


def test_repeat_shuttle_dif():
    expected = [np.array([2., 4., 3., 1.]), np.array([1., 1., 1., 3., 2., 4.]), np.array([1., 2., 3., 1.])]
    actual = st.repeat_shuffle_diff(adj, t4, 3, seed=10)
    for i, j in zip(expected, actual):
        np.testing.assert_array_almost_equal(i, j)
